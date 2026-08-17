import numpy as np

from scipy.optimize import least_squares

from slam.ba import (
    build_ba_indices,
    build_residual_vector,
    pack_mappoint_parameters,
    pack_keyframe_parameters,
    pack_all_parameters,
    unpack_all_parameters,
    ba_residuals,
    build_jac_sparsity,
    collect_clean_observations_for_graph,
    filter_mappoints_by_observations,
    filter_keyframes_by_observations,
)


def make_residual_summary(residuals):
    """
    Считает короткую статистику по residual vector.
    """

    if residuals is None or len(residuals) == 0:
        return {
            "mean": float("inf"),
            "median": float("inf"),
            "max": float("inf"),
        }

    abs_residuals = np.abs(
        np.asarray(
            residuals,
            dtype=np.float64
        )
    )

    return {
        "mean": float(np.mean(abs_residuals)),
        "median": float(np.median(abs_residuals)),
        "max": float(np.max(abs_residuals)),
    }


# ---------------------------------------------------------------------------------------------------------------------------------
def select_local_keyframes(
    keyframes,
    current_keyframe,
    window_size
):
    """
    Выбирает KeyFrames для Local BA.

    Первый KeyFrame в возвращаемом списке будет fixed/anchor,
    остальные будут оптимизироваться.
    """

    if current_keyframe is None:
        return []

    if len(keyframes) == 0:
        return []

    local_keyframes = list(
        keyframes[-window_size:]
    )

    if current_keyframe not in local_keyframes:
        local_keyframes.append(
            current_keyframe
        )

    local_keyframes = sorted(
        local_keyframes,
        key=lambda kf: kf.id
    )

    return local_keyframes


# ---------------------------------------------------------------------------------------------------------------------------------
def select_local_mappoints(
    map_points,
    local_keyframes,
    current_keyframe,
    max_map_points
):
    """
    Выбирает MapPoints, которые видны в local KeyFrames.

    Приоритет:
        1. точки, видимые в current_keyframe;
        2. точки с большим числом observations;
        3. более новые точки.
    """

    if len(local_keyframes) == 0:
        return []

    local_keyframe_ids = {
        kf.id
        for kf in local_keyframes
    }

    current_keyframe_id = None

    if current_keyframe is not None:
        current_keyframe_id = current_keyframe.id

    selected_by_id = {}

    for mp in map_points:

        if mp is None:
            continue

        if mp.id is None:
            continue

        if len(mp.observations) == 0:
            continue

        for obs in mp.observations:

            if obs.keyframe is None:
                continue

            if obs.keyframe.id in local_keyframe_ids:
                selected_by_id[mp.id] = mp
                break

    selected = list(
        selected_by_id.values()
    )

    def mappoint_priority(mp):

        seen_in_current_keyframe = 0

        if current_keyframe_id is not None:

            for obs in mp.observations:

                if obs.keyframe is None:
                    continue

                if obs.keyframe.id == current_keyframe_id:
                    seen_in_current_keyframe = 1
                    break

        return (
            seen_in_current_keyframe,
            getattr(mp, "num_observations", 0),
            mp.id if mp.id is not None else -1
        )

    selected = sorted(
        selected,
        key=mappoint_priority,
        reverse=True
    )

    return selected[:max_map_points]


# ---------------------------------------------------------------------------------------------------------------------------------
def cleanup_local_ba_graph(
    keyframes,
    map_points,
    K,
    min_mp_observations=2,
    min_kf_observations=10,
    max_initial_residual=50.0,
    max_iterations=10
):
    """
    Очищает локальный BA-граф без изменения глобальных списков.

    Возвращает:
        optimized_keyframes,
        optimized_map_points,
        observations,
        idx_to_mp,
        idx_to_kf
    """

    keyframes = list(keyframes)
    map_points = list(map_points)

    if len(keyframes) == 0:
        return [], [], [], {}, {}

    fixed_keyframe_id = keyframes[0].id
    observations = []

    for _ in range(max_iterations):

        old_counts = (
            len(keyframes),
            len(map_points),
            len(observations)
        )

        observations = collect_clean_observations_for_graph(
            map_points,
            keyframes,
            K,
            max_initial_residual=max_initial_residual
        )

        map_points = filter_mappoints_by_observations(
            map_points,
            observations,
            min_observations=min_mp_observations
        )

        observations = collect_clean_observations_for_graph(
            map_points,
            keyframes,
            K,
            max_initial_residual=max_initial_residual
        )

        keyframes = filter_keyframes_by_observations(
            keyframes,
            observations,
            fixed_keyframe_id=fixed_keyframe_id,
            min_observations=min_kf_observations
        )

        observations = collect_clean_observations_for_graph(
            map_points,
            keyframes,
            K,
            max_initial_residual=max_initial_residual
        )

        map_points = filter_mappoints_by_observations(
            map_points,
            observations,
            min_observations=min_mp_observations
        )

        observations = collect_clean_observations_for_graph(
            map_points,
            keyframes,
            K,
            max_initial_residual=max_initial_residual
        )

        new_counts = (
            len(keyframes),
            len(map_points),
            len(observations)
        )

        if old_counts == new_counts:
            break

    idx_to_mp, idx_to_kf = build_ba_indices(
        map_points,
        keyframes
    )

    return keyframes, map_points, observations, idx_to_mp, idx_to_kf


# ---------------------------------------------------------------------------------------------------------------------------------
def backup_local_ba_state(
    keyframes,
    map_points
):
    """
    Сохраняет pose KeyFrames и positions MapPoints перед Local BA.
    """

    keyframe_backup = {}

    for kf in keyframes:
        keyframe_backup[kf.id] = {
            "R": kf.R.copy(),
            "t": kf.t.copy()
        }

    mappoint_backup = {}

    for mp in map_points:
        mappoint_backup[mp.id] = mp.position.copy()

    return keyframe_backup, mappoint_backup


# ---------------------------------------------------------------------------------------------------------------------------------
def restore_local_ba_state(
    keyframes,
    map_points,
    keyframe_backup,
    mappoint_backup
):
    """
    Откатывает изменения Local BA.
    """

    for kf in keyframes:

        if kf.id not in keyframe_backup:
            continue

        kf.R = keyframe_backup[kf.id]["R"].copy()
        kf.t = keyframe_backup[kf.id]["t"].copy()

    for mp in map_points:

        if mp.id not in mappoint_backup:
            continue

        mp.position = mappoint_backup[mp.id].copy()


# ---------------------------------------------------------------------------------------------------------------------------------
def compute_max_keyframe_shift(
    keyframes,
    keyframe_backup
):
    """
    Считает максимальное смещение камеры после Local BA.
    """

    max_shift = 0.0
    max_shift_keyframe_id = None

    for kf in keyframes:

        if kf.id not in keyframe_backup:
            continue

        old_t = keyframe_backup[kf.id]["t"]
        new_t = kf.t

        shift = np.linalg.norm(
            new_t - old_t
        )

        if shift > max_shift:
            max_shift = float(shift)
            max_shift_keyframe_id = kf.id

    return max_shift, max_shift_keyframe_id

# ---------------------------------------------------------------------------------------------------------------------------------
def project_mappoint_to_keyframe(
    map_point,
    keyframe,
    K
):
    """
    Проецирует MapPoint в изображение конкретного KeyFrame.

    Возвращает:
        np.array([u, v]) или None, если точка находится за камерой.
    """

    if map_point is None:
        return None

    if keyframe is None:
        return None

    if map_point.position is None:
        return None

    Pw = map_point.position.reshape(3, 1)

    Pc = keyframe.R.T @ (
        Pw - keyframe.t
    )

    if Pc[2, 0] <= 0:
        return None

    pixel = K @ Pc

    u = pixel[0, 0] / pixel[2, 0]
    v = pixel[1, 0] / pixel[2, 0]

    if not np.isfinite(u) or not np.isfinite(v):
        return None

    return np.array(
        [u, v],
        dtype=np.float64
    )


# ---------------------------------------------------------------------------------------------------------------------------------
def compute_observation_reprojection_error(
    map_point,
    observation,
    K
):
    """
    Считает reprojection error для одной observation.
    """

    if observation is None:
        return None

    keyframe = observation.keyframe

    if keyframe is None:
        return None

    if observation.keypoint_idx is None:
        return None

    keypoint_idx = int(
        observation.keypoint_idx
    )

    if keypoint_idx < 0:
        return None

    if keypoint_idx >= len(keyframe.kp):
        return None

    projected_uv = project_mappoint_to_keyframe(
        map_point,
        keyframe,
        K
    )

    if projected_uv is None:
        return None

    observed_uv = np.array(
        keyframe.kp[keypoint_idx].pt,
        dtype=np.float64
    )

    reprojection_error = np.linalg.norm(
        projected_uv - observed_uv
    )

    if not np.isfinite(reprojection_error):
        return None

    return float(
        reprojection_error
    )


# ---------------------------------------------------------------------------------------------------------------------------------
def keyframe_still_observes_mappoint(
    keyframe,
    map_point
):
    """
    Проверяет, осталась ли у MapPoint observation в данном KeyFrame.
    """

    for observation in map_point.observations:

        if observation.keyframe is keyframe:
            return True

    return False


# ---------------------------------------------------------------------------------------------------------------------------------
def remove_observation_link(
    map_point,
    observation
):
    """
    Удаляет одну observation-связь из MapPoint и соответствующего KeyFrame.
    """

    keyframe = observation.keyframe

    before_observations = len(
        map_point.observations
    )

    map_point.observations = [
        current_observation
        for current_observation in map_point.observations
        if current_observation is not observation
    ]

    after_observations = len(
        map_point.observations
    )

    removed_from_mappoint = (
        after_observations < before_observations
    )

    map_point.num_observations = len(
        map_point.observations
    )

    removed_from_keyframe = False

    if keyframe is not None:

        if not keyframe_still_observes_mappoint(
            keyframe,
            map_point
        ):

            before_keyframe_points = len(
                keyframe.map_points
            )

            keyframe.map_points = [
                current_map_point
                for current_map_point in keyframe.map_points
                if current_map_point is not map_point
            ]

            after_keyframe_points = len(
                keyframe.map_points
            )

            removed_from_keyframe = (
                after_keyframe_points < before_keyframe_points
            )

    return removed_from_mappoint, removed_from_keyframe


# ---------------------------------------------------------------------------------------------------------------------------------
def reject_local_ba_outliers(
    keyframes,
    map_points,
    K,
    max_reprojection_error=12.0,
    min_observations_to_keep=2
):
    """
    Удаляет плохие observation-связи после accepted Local BA.

    Важно:
        - MapPoints напрямую не удаляются.
        - Удаляются только плохие observations.
        - Слабые MapPoints потом должен удалить обычный culling.
    """

    stats = {
        "outlier_rejection_ran": True,
        "outlier_checked_observations": 0,
        "outlier_removed_observations": 0,
        "outlier_removed_from_keyframes": 0,
        "outlier_affected_mappoints": 0,
        "outlier_affected_keyframes": 0,
        "outlier_skipped_min_observations": 0,
        "outlier_invalid_projection": 0,
        "outlier_high_error": 0,
        "outlier_error_mean": None,
        "outlier_error_median": None,
        "outlier_error_max": None,
    }

    local_keyframe_ids = {
        keyframe.id
        for keyframe in keyframes
    }

    valid_errors = []

    affected_mappoint_ids = set()
    affected_keyframe_ids = set()

    for map_point in map_points:

        if map_point is None:
            continue

        if map_point.id is None:
            continue

        observations_copy = list(
            map_point.observations
        )

        for observation in observations_copy:

            if observation is None:
                continue

            keyframe = observation.keyframe

            if keyframe is None:
                continue

            if keyframe.id not in local_keyframe_ids:
                continue

            stats["outlier_checked_observations"] += 1

            reprojection_error = compute_observation_reprojection_error(
                map_point,
                observation,
                K
            )

            is_invalid_projection = (
                reprojection_error is None
            )

            is_high_error = (
                reprojection_error is not None
                and reprojection_error > max_reprojection_error
            )

            if reprojection_error is not None:
                valid_errors.append(
                    reprojection_error
                )

            if not is_invalid_projection and not is_high_error:
                continue

            if len(map_point.observations) <= min_observations_to_keep:

                stats["outlier_skipped_min_observations"] += 1
                continue

            removed_from_mappoint, removed_from_keyframe = remove_observation_link(
                map_point,
                observation
            )

            if not removed_from_mappoint:
                continue

            stats["outlier_removed_observations"] += 1

            affected_mappoint_ids.add(
                map_point.id
            )

            if keyframe.id is not None:
                affected_keyframe_ids.add(
                    keyframe.id
                )

            if removed_from_keyframe:
                stats["outlier_removed_from_keyframes"] += 1

            if is_invalid_projection:
                stats["outlier_invalid_projection"] += 1

            if is_high_error:
                stats["outlier_high_error"] += 1

    if len(valid_errors) > 0:

        valid_errors = np.asarray(
            valid_errors,
            dtype=np.float64
        )

        stats["outlier_error_mean"] = float(
            np.mean(valid_errors)
        )

        stats["outlier_error_median"] = float(
            np.median(valid_errors)
        )

        stats["outlier_error_max"] = float(
            np.max(valid_errors)
        )

    stats["outlier_affected_mappoints"] = len(
        affected_mappoint_ids
    )

    stats["outlier_affected_keyframes"] = len(
        affected_keyframe_ids
    )

    return stats


# ---------------------------------------------------------------------------------------------------------------------------------
def run_local_ba(
    keyframes,
    map_points,
    K,
    current_keyframe,
    window_size=7,
    max_map_points=800,
    min_keyframes=4,
    min_mp_observations=2,
    min_kf_observations=10,
    max_initial_residual=50.0,
    max_cleanup_iterations=10,
    ba_max_nfev=20,
    huber_f_scale=5.0,
    max_mean_residual_increase=1.05,
    max_camera_shift=5.0,
    outlier_rejection_enabled=False,
    outlier_reprojection_error=12.0,
    outlier_min_observations_to_keep=2,
    verbose=False
):
    """
    Запускает синхронный Local Bundle Adjustment.

    Local BA оптимизирует:
        - последние local KeyFrames, кроме первого fixed/anchor;
        - MapPoints, видимые в этих KeyFrames.

    Если результат выглядит опасно, изменения откатываются.
    """

    stats = {
        "ran": False,
        "accepted": False,
        "reason": "not_started",
        "keyframes": 0,
        "map_points": 0,
        "observations": 0,
        "mean_before": None,
        "mean_after": None,
        "median_before": None,
        "median_after": None,
        "max_before": None,
        "max_after": None,
        "max_camera_shift": None,
        "max_camera_shift_keyframe_id": None,
        "cost": None,
        "optimality": None,
        "nfev": None,
        "success": None,
        "message": None,
        "outlier_rejection_ran": False,
        "outlier_checked_observations": 0,
        "outlier_removed_observations": 0,
        "outlier_removed_from_keyframes": 0,
        "outlier_affected_mappoints": 0,
        "outlier_affected_keyframes": 0,
        "outlier_skipped_min_observations": 0,
        "outlier_invalid_projection": 0,
        "outlier_high_error": 0,
        "outlier_error_mean": None,
        "outlier_error_median": None,
        "outlier_error_max": None,
    }

    local_keyframes = select_local_keyframes(
        keyframes,
        current_keyframe,
        window_size
    )

    if len(local_keyframes) < min_keyframes:
        stats["reason"] = "too_few_local_keyframes"
        stats["keyframes"] = len(local_keyframes)
        return stats

    local_map_points = select_local_mappoints(
        map_points,
        local_keyframes,
        current_keyframe,
        max_map_points
    )

    if len(local_map_points) == 0:
        stats["reason"] = "no_local_mappoints"
        stats["keyframes"] = len(local_keyframes)
        return stats

    optimized_keyframes, optimized_map_points, ba_observations, idx_to_mp, idx_to_kf = cleanup_local_ba_graph(
        local_keyframes,
        local_map_points,
        K,
        min_mp_observations=min_mp_observations,
        min_kf_observations=min_kf_observations,
        max_initial_residual=max_initial_residual,
        max_iterations=max_cleanup_iterations
    )

    stats["keyframes"] = len(optimized_keyframes)
    stats["map_points"] = len(optimized_map_points)
    stats["observations"] = len(ba_observations)

    if len(optimized_keyframes) < 2:
        stats["reason"] = "too_few_optimized_keyframes"
        return stats

    if len(optimized_map_points) == 0:
        stats["reason"] = "no_optimized_mappoints"
        return stats

    if len(ba_observations) == 0:
        stats["reason"] = "no_ba_observations"
        return stats

    residuals_before = build_residual_vector(
        ba_observations,
        idx_to_mp,
        idx_to_kf,
        K
    )

    residual_summary_before = make_residual_summary(
        residuals_before
    )

    stats["mean_before"] = residual_summary_before["mean"]
    stats["median_before"] = residual_summary_before["median"]
    stats["max_before"] = residual_summary_before["max"]

    keyframe_backup, mappoint_backup = backup_local_ba_state(
        optimized_keyframes,
        optimized_map_points
    )

    packed_keyframe_parameters = pack_keyframe_parameters(
        optimized_keyframes
    )

    packed_mappoint_parameters = pack_mappoint_parameters(
        optimized_map_points
    )

    packed_parameters = pack_all_parameters(
        packed_keyframe_parameters,
        packed_mappoint_parameters
    )

    jac_sparsity = build_jac_sparsity(
        ba_observations,
        optimized_keyframes,
        optimized_map_points
    )

    stats["ran"] = True

    try:
        result = least_squares(
            ba_residuals,
            packed_parameters,
            args=(
                ba_observations,
                optimized_keyframes,
                optimized_map_points,
                idx_to_mp,
                idx_to_kf,
                K
            ),
            loss="huber",
            f_scale=huber_f_scale,
            method="trf",
            jac_sparsity=jac_sparsity,
            max_nfev=ba_max_nfev,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
            verbose=2 if verbose else 0
        )

    except Exception as exc:
        restore_local_ba_state(
            optimized_keyframes,
            optimized_map_points,
            keyframe_backup,
            mappoint_backup
        )

        stats["reason"] = f"optimizer_exception: {type(exc).__name__}"
        return stats

    unpack_all_parameters(
        result.x,
        optimized_keyframes,
        optimized_map_points
    )

    residuals_after = build_residual_vector(
        ba_observations,
        idx_to_mp,
        idx_to_kf,
        K
    )

    residual_summary_after = make_residual_summary(
        residuals_after
    )

    stats["mean_after"] = residual_summary_after["mean"]
    stats["median_after"] = residual_summary_after["median"]
    stats["max_after"] = residual_summary_after["max"]

    max_shift, max_shift_keyframe_id = compute_max_keyframe_shift(
        optimized_keyframes,
        keyframe_backup
    )

    stats["max_camera_shift"] = max_shift
    stats["max_camera_shift_keyframe_id"] = max_shift_keyframe_id

    stats["cost"] = float(result.cost)
    stats["optimality"] = float(result.optimality)
    stats["nfev"] = int(result.nfev)
    stats["success"] = bool(result.success)
    stats["message"] = str(result.message)

    mean_before = stats["mean_before"]
    mean_after = stats["mean_after"]

    if not np.isfinite(mean_after):
        stats["reason"] = "non_finite_residual_after"

    elif mean_after > mean_before * max_mean_residual_increase:
        stats["reason"] = "mean_residual_increased_too_much"

    elif max_shift > max_camera_shift:
        stats["reason"] = "camera_shift_too_large"

    else:

        if outlier_rejection_enabled:

            outlier_stats = reject_local_ba_outliers(
                optimized_keyframes,
                optimized_map_points,
                K,
                max_reprojection_error=outlier_reprojection_error,
                min_observations_to_keep=outlier_min_observations_to_keep
            )

            stats.update(
                outlier_stats
            )

        stats["accepted"] = True
        stats["reason"] = "accepted"
        return stats

    restore_local_ba_state(
        optimized_keyframes,
        optimized_map_points,
        keyframe_backup,
        mappoint_backup
    )

    return stats