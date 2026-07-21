import numpy as np

from slam.tracking import project_point


def compute_mappoint_reprojection_errors(
    mp,
    K
):
    """
    Считает reprojection errors для одной MapPoint по всем её observations.

    Возвращает список ошибок в пикселях.
    """

    errors = []

    for obs in mp.observations:

        kf = obs.keyframe
        kp_idx = obs.keypoint_idx

        if kf is None:
            return None

        if kp_idx < 0 or kp_idx >= len(kf.kp):
            return None

        predicted_uv = project_point(
            mp.position,
            kf.R,
            kf.t,
            K
        )

        if predicted_uv is None:
            return None

        observed_uv = np.array(
            kf.kp[kp_idx].pt,
            dtype=np.float64
        )

        error = np.linalg.norm(
            predicted_uv - observed_uv
        )

        errors.append(error)

    return errors

# ---------------------------------------------------------------------------------------------------------------------------------------
def is_bad_mappoint(
    mp,
    K,
    current_keyframe_id,
    min_observations=2,
    min_unique_keyframes=2,
    max_mean_reprojection_error=20.0,
    min_age_keyframes=3
):
    """
    Проверяет, нужно ли удалить MapPoint из карты.

    Логика:
        - явно битые MapPoints удаляются сразу;
        - молодые MapPoints с малым числом observations временно сохраняются;
        - молодые MapPoints с достаточным числом observations уже проходят reprojection check;
        - зрелые MapPoints проверяются строго.
    """

    if mp is None:
        return True, "none"

    if mp.position is None:
        return True, "no_position"

    if not np.isfinite(mp.position).all():
        return True, "bad_position"

    if mp.descriptor is None:
        return True, "no_descriptor"

    is_young = False

    if mp.created_keyframe_id is not None and current_keyframe_id is not None:

        age_keyframes = current_keyframe_id - mp.created_keyframe_id

        if age_keyframes < min_age_keyframes:
            is_young = True

    observations_count = len(mp.observations)

    if observations_count < min_observations:

        if is_young:
            return False, "young"

        return True, "too_few_observations"

    unique_keyframe_ids = {
        obs.keyframe.id
        for obs in mp.observations
        if obs.keyframe is not None
    }

    if len(unique_keyframe_ids) < min_unique_keyframes:

        if is_young:
            return False, "young_too_few_keyframes"

        return True, "too_few_keyframes"

    errors = compute_mappoint_reprojection_errors(
        mp,
        K
    )

    if errors is None:
        return True, "bad_reprojection"

    if len(errors) == 0:

        if is_young:
            return False, "young_no_reprojection_errors"

        return True, "no_reprojection_errors"

    mean_error = np.mean(errors)

    if mean_error > max_mean_reprojection_error:
        return True, "high_reprojection_error"

    if is_young:
        return False, "young_good"

    return False, "good"

# --------------------------------------------------------------------------------------------------------------------------------------
def remove_mappoint_references_from_keyframes(
    keyframes,
    removed_mappoint_ids
):
    """
    Удаляет ссылки на плохие MapPoints из KeyFrame.map_points.
    """

    for kf in keyframes:

        kf.map_points = [
            mp
            for mp in kf.map_points
            if mp.id not in removed_mappoint_ids
        ]

# ---------------------------------------------------------------------------------------------------------------------------------------
def cull_mappoints(
    map_points,
    keyframes,
    K,
    current_keyframe_id,
    min_observations=2,
    min_unique_keyframes=2,
    max_mean_reprojection_error=20.0,
    min_age_keyframes=3
):
    """
    Удаляет плохие MapPoints из карты.

    Возвращает:
        new_map_points
        stats
    """

    good_points = []
    removed_points = []

    removal_reasons = {}
    kept_reasons = {}

    for mp in map_points:

        is_bad, reason = is_bad_mappoint(
            mp,
            K,
            current_keyframe_id,
            min_observations=min_observations,
            min_unique_keyframes=min_unique_keyframes,
            max_mean_reprojection_error=max_mean_reprojection_error,
            min_age_keyframes=min_age_keyframes
        )

        if is_bad:
            removed_points.append(mp)
            removal_reasons[reason] = removal_reasons.get(reason, 0) + 1
        else:
            good_points.append(mp)
            kept_reasons[reason] = kept_reasons.get(reason, 0) + 1

    removed_mappoint_ids = {
        mp.id
        for mp in removed_points
        if mp.id is not None
    }

    remove_mappoint_references_from_keyframes(
        keyframes,
        removed_mappoint_ids
    )

    stats = {
        "before": len(map_points),
        "after": len(good_points),
        "removed": len(removed_points),
        "removal_reasons": removal_reasons,
        "kept_reasons": kept_reasons
    }

    return good_points, stats