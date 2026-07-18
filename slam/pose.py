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
            "inlier_ratio": 0.0
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
            "inlier_ratio": 0.0
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
            "inlier_ratio": 0.0
        }

    inlier_indices = inliers.ravel().astype(int)

    if len(inlier_indices) < min_inliers:
        return False, None, None, [], {
            "points": len(object_points),
            "inliers": len(inlier_indices),
            "inlier_ratio": len(inlier_indices) / len(object_points)
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
        "inlier_ratio": len(inlier_indices) / len(object_points)
    }

    return True, Rwc, twc, inlier_tracked_pairs, pnp_stats

# -------------------------------------------------------------------------------
def update_pose_from_recover_pose(prev_Rwc, prev_twc, R, t):
    curr_Rwc = prev_Rwc @ R.T
    curr_twc = prev_twc - prev_Rwc @ R.T @ t
    return curr_Rwc, curr_twc