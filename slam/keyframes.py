import cv2 as cv
import numpy as np

from slam.map import KeyFrame, Observation


def create_KeyFrame(
        global_R,
        global_t,
        kp,
        des,
        current_frame_map_points,
        keyframes,
        keyframe_id,
):
    """
    создаёт новый KeyFrame.
    ничего не возвращает.
    """

    kf = KeyFrame(
        global_R.copy(),
        global_t.copy(),
        kp,
        des
    )
    kf.id = keyframe_id

    seen_mappoint_ids = set()

    # сохраняем в текущий keyframe все точки, которые он увидел в кадре
    for mp, valid_kp_idx in current_frame_map_points:

        if mp.id in seen_mappoint_ids:
            continue

        seen_mappoint_ids.add(mp.id)

        obs = Observation(
            kf,
            valid_kp_idx
        )

        mp.observations.append(obs)

        kf.map_points.append(mp)

    keyframes.append(kf)

# ---------------------------------------------------------------------------------------------------------------------------------------------
def compute_keyframe_motion(
    current_R,
    current_t,
    last_keyframe_R,
    last_keyframe_t
):
    """
    Считает, насколько текущая pose камеры отличается от последнего KeyFrame.

    Возвращает:
        translation: расстояние между текущей позицией камеры и позицией последнего KeyFrame
        rotation: угол поворота между текущей ориентацией камеры и ориентацией последнего KeyFrame
    """

    if last_keyframe_R is None or last_keyframe_t is None:
        return 0.0, 0.0

    translation = np.linalg.norm(
        current_t - last_keyframe_t
    )

    R_delta = last_keyframe_R.T @ current_R

    rvec, _ = cv.Rodrigues(
        R_delta
    )

    rotation = np.linalg.norm(
        rvec
    )

    return float(translation), float(rotation)

# ----------------------------------------------------------------------------------------------------------------------------------------------
def should_create_keyframe(
    translation,
    rotation,
    tracking_stats,
    pose_source,
    map_points_count,
    frames_since_last_keyframe,
    translation_threshold=10.0,
    rotation_threshold=0.6,
    min_frames_between=10,
    min_tracked_points=80,
    min_pnp_inlier_ratio=0.55,
    min_map_points_for_tracking_check=300,
    weak_tracking_min_translation=2.0,
    weak_tracking_min_rotation=0.15
):
    """
    Решает, нужно ли создавать новый KeyFrame.

    Возвращает:
        create_keyframe: bool
        reason: str
    """

    if frames_since_last_keyframe < min_frames_between:
        return False, "too_soon"

    motion_by_translation = (
        translation > translation_threshold
    )

    motion_by_rotation = (
        rotation > rotation_threshold
    )

    if motion_by_translation and motion_by_rotation:
        return True, "motion_translation_rotation"

    if motion_by_translation:
        return True, "motion_translation"

    if motion_by_rotation:
        return True, "motion_rotation"

    if tracking_stats is None:
        tracking_stats = {}

    tracking_check_enabled = (
        map_points_count >= min_map_points_for_tracking_check
    )

    if not tracking_check_enabled:
        return False, "not_needed"

    # Первый вариант делаем консервативным:
    # weak tracking создаёт KeyFrame только если pose была получена через PnP.
    # Если pose_source == recoverPose, не форсируем KeyFrame по tracking quality,
    # потому что recoverPose-pose менее надёжна.
    if pose_source != "PnP":
        return False, "not_needed"

    weak_tracking_has_enough_motion = (
        translation > weak_tracking_min_translation
        or rotation > weak_tracking_min_rotation
    )
    
    if not weak_tracking_has_enough_motion:
        return False, "weak_tracking_too_little_motion"

    tracked_used = tracking_stats.get(
        "tracked_used",
        tracking_stats.get("tracked", 0)
    )

    pnp_inlier_ratio = tracking_stats.get(
        "pnp_inlier_ratio",
        0.0
    )

    if tracked_used < min_tracked_points:
        return True, "weak_tracking_points"

    if pnp_inlier_ratio < min_pnp_inlier_ratio:
        return True, "weak_tracking_pnp_ratio"

    return False, "not_needed"