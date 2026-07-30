import cv2 as cv
import numpy as np


def estimate_pose_pnp(
    tracked_map_points,
    kp,
    K,
    min_points=15,
    min_inliers=12,
    reprojection_error_threshold=12.0,
    confidence=0.99,
    iterations_count=100
):
    """
    Оценивает pose текущей камеры через PnP.

    Вход:
        tracked_map_points:
            список пар [(MapPoint, keypoint_idx), ...]

        kp:
            keypoints текущего кадра

        K:
            camera intrinsic matrix

    PnP использует соответствия:
        3D MapPoint.position в мире
        <->
        2D kp[keypoint_idx].pt на текущем изображении

    OpenCV solvePnP возвращает Rcw/tcw:
        Pc = Rcw @ Pw + tcw

    В текущем SLAM convention нужно Rwc/twc:
        Pw = Rwc @ Pc + twc
    """

    if len(tracked_map_points) < min_points:
        return False, None, None, [], {
            "points": len(tracked_map_points),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "reason": "too_few_tracked_pairs"
        }

    object_points = []
    image_points = []
    valid_tracked_pairs = []

    for mp, kp_idx in tracked_map_points:

        if mp is None:
            continue

        if mp.position is None:
            continue

        if kp_idx < 0 or kp_idx >= len(kp):
            continue

        object_points.append(
            np.asarray(mp.position, dtype=np.float32).reshape(3)
        )

        image_points.append(
            np.asarray(kp[kp_idx].pt, dtype=np.float32).reshape(2)
        )

        valid_tracked_pairs.append(
            (mp, kp_idx)
        )

    if len(object_points) < min_points:
        return False, None, None, [], {
            "points": len(object_points),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "reason": "too_few_valid_points"
        }

    object_points = np.asarray(
        object_points,
        dtype=np.float32
    ).reshape(-1, 1, 3)

    image_points = np.asarray(
        image_points,
        dtype=np.float32
    ).reshape(-1, 1, 2)

    success, rvec, tvec, inliers = cv.solvePnPRansac(
        object_points,
        image_points,
        K,
        None,
        iterationsCount=iterations_count,
        reprojectionError=reprojection_error_threshold,
        confidence=confidence,
        flags=cv.SOLVEPNP_ITERATIVE
    )

    if not success or inliers is None:
        return False, None, None, [], {
            "points": len(object_points),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "reason": "solvepnp_failed"
        }

    inlier_indices = inliers.ravel().astype(int)

    if len(inlier_indices) < min_inliers:
        return False, None, None, [], {
            "points": len(object_points),
            "inliers": len(inlier_indices),
            "inlier_ratio": len(inlier_indices) / len(object_points),
            "reason": "too_few_inliers"
        }

    # OpenCV pose: world -> camera
    Rcw, _ = cv.Rodrigues(rvec)
    tcw = tvec.reshape(3, 1)

    # Конвертируем в нашу SLAM convention: camera -> world
    Rwc = Rcw.T
    twc = -Rcw.T @ tcw

    inlier_tracked_pairs = [
        valid_tracked_pairs[i]
        for i in inlier_indices
    ]

    pnp_stats = {
        "points": len(object_points),
        "inliers": len(inlier_indices),
        "inlier_ratio": len(inlier_indices) / len(object_points),
        "reason": "success"
    }

    return True, Rwc, twc, inlier_tracked_pairs, pnp_stats

# --------------------------------------------------------------------------------------------------------------------------------------------
def update_pose_from_recover_pose(prev_Rwc, prev_twc, R, t):
    curr_Rwc = prev_Rwc @ R.T
    curr_twc = prev_twc - prev_Rwc @ R.T @ t
    return curr_Rwc, curr_twc

# --------------------------------------------------------------------------------------------------------------------------------------------
def compute_pose_delta(
    current_R,
    current_t,
    reference_R,
    reference_t
):
    """
    Считает разницу между двумя pose камеры.

    current_R / current_t:
        candidate pose, которую мы хотим проверить.

    reference_R / reference_t:
        предыдущая надёжная pose, относительно которой считаем jump.

    Возвращает:
        translation_jump: float
        rotation_jump: float
    """

    if reference_R is None or reference_t is None:
        return 0.0, 0.0

    translation_jump = np.linalg.norm(
        current_t - reference_t
    )

    R_delta = reference_R.T @ current_R

    rvec, _ = cv.Rodrigues(
        R_delta
    )

    rotation_jump = np.linalg.norm(
        rvec
    )

    return float(translation_jump), float(rotation_jump)

# -------------------------------------------------------------------------------------------------------------------------------------------
def check_pnp_pose_safety(
    candidate_Rwc,
    candidate_twc,
    reference_Rwc,
    reference_twc,
    pnp_stats,
    min_inliers,
    min_inlier_ratio,
    max_translation_jump,
    max_rotation_jump
):
    """
    Проверяет, можно ли принимать PnP pose.

    candidate_Rwc / candidate_twc:
        pose, которую предложил PnP.

    reference_Rwc / reference_twc:
        предыдущая принятая pose, относительно которой проверяем jump.

    Возвращает:
        pose_is_safe: bool
        reject_reason: str | None
        translation_jump: float
        rotation_jump: float
    """

    translation_jump, rotation_jump = compute_pose_delta(
        candidate_Rwc,
        candidate_twc,
        reference_Rwc,
        reference_twc
    )

    if pnp_stats["inliers"] < min_inliers:
        return (
            False,
            "reject_too_few_inliers",
            translation_jump,
            rotation_jump
        )

    if pnp_stats["inlier_ratio"] < min_inlier_ratio:
        return (
            False,
            "reject_low_inlier_ratio",
            translation_jump,
            rotation_jump
        )

    if translation_jump > max_translation_jump:
        return (
            False,
            "reject_translation_jump",
            translation_jump,
            rotation_jump
        )

    if rotation_jump > max_rotation_jump:
        return (
            False,
            "reject_rotation_jump",
            translation_jump,
            rotation_jump
        )

    return (
        True,
        None,
        translation_jump,
        rotation_jump
    )