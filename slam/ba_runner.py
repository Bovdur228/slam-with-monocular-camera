import numpy as np

from scipy.optimize import least_squares

from slam.ba import (
    build_residual_vector,
    pack_mappoint_parameters,
    pack_keyframe_parameters,
    pack_all_parameters,
    unpack_all_parameters,
    ba_residuals,
    build_jac_sparsity,
    cleanup_ba_graph,
)

from slam.ba_diagnostics import (
    print_ba_graph_diagnostics,
)

# ---------------------------------------------------------------------------------------------------------------------------
def run_ba_test(
    keyframes,
    map_points,
    K,
    pnp_success_count=0,
    recoverpose_count=0,
    max_test_keyframes=30,
    min_mp_observations=3,
    min_kf_observations=10,
    max_initial_residual=50,
    ba_max_nfev=500
):
    """
    Запускает offline Bundle Adjustment test/evaluation
    после завершения основной SLAM-сессии.
    """

    print("=" * 100)
    print(f"PnP frames: {pnp_success_count}")
    print(f"recoverPose frames: {recoverpose_count}")

    print("-" * 100)

    test_keyframes = keyframes[:max_test_keyframes]

    if len(test_keyframes) < 2:
        print("Слишком мало KeyFrames для BA.")
        print(f"KeyFrames: {len(test_keyframes)}")
        return None

    fixed_keyframe = test_keyframes[0]
    candidate_keyframes = test_keyframes[1:]

    test_kf_ids = {
        kf.id
        for kf in test_keyframes
    }

    test_map_points = [
        mp
        for mp in map_points
        if any(obs.keyframe.id in test_kf_ids for obs in mp.observations)
    ]

    print_ba_graph_diagnostics(
        test_map_points,
        test_keyframes,
        K,
        max_initial_residual=max_initial_residual,
        title="BEFORE cleanup_ba_graph"
    )

    print("-" * 100)

    optimized_keyframes, optimized_map_points, ba_observations, idx_to_mp, idx_to_kf = cleanup_ba_graph(
        [fixed_keyframe] + candidate_keyframes,
        test_map_points,
        K,
        min_mp_observations=min_mp_observations,
        min_kf_observations=min_kf_observations,
        max_initial_residual=max_initial_residual,
    )

    if not ba_observations or not optimized_keyframes or not optimized_map_points:
        print("BA граф пуст после cleanup — недостаточно наблюдений для оптимизации.")
        print(
            f"KFs: {len(optimized_keyframes)}, "
            f"MPs: {len(optimized_map_points)}, "
            f"Obs: {len(ba_observations)}"
        )
        return None

    print_ba_graph_diagnostics(
        optimized_map_points,
        optimized_keyframes,
        K,
        max_initial_residual=max_initial_residual,
        title="AFTER cleanup_ba_graph"
    )

    print("-" * 100)

    kf_obs_count = {}

    for _, kf_id, _ in ba_observations:
        kf_obs_count[kf_id] = kf_obs_count.get(kf_id, 0) + 1

    print("Observations per KeyFrame:")

    for kf in optimized_keyframes:
        count = kf_obs_count.get(kf.id, 0)
        print(f"KF {kf.id}: {count}")

    print("-" * 100)

    print(
        f"Observations / KeyFrame = "
        f"{len(ba_observations) / len(optimized_keyframes):.2f}"
    )

    print(
        f"Observations / MapPoint = "
        f"{len(ba_observations) / len(optimized_map_points):.2f}"
    )

    print("-" * 100)

    residuals_before = build_residual_vector(
        ba_observations,
        idx_to_mp,
        idx_to_kf,
        K
    )

    print("\n=== BEFORE BA ===")
    print("Mean abs residual:", np.mean(np.abs(residuals_before)))
    print("Median abs residual:", np.median(np.abs(residuals_before)))
    print("Max abs residual:", np.max(np.abs(residuals_before)))
    print("-" * 100)

    kf_before = {}

    for kf in optimized_keyframes:
        kf_before[kf.id] = {
            "R": kf.R.copy(),
            "t": kf.t.copy()
        }

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

    print("Bundle Adjustment statistics")
    print(f"Observations:       {len(ba_observations)}")
    print(f"Optimized KeyFrames:{len(optimized_keyframes)}")
    print(f"Optimized MapPoints:{len(optimized_map_points)}")

    print(
        f"Optimization parameters: "
        f"{(len(optimized_keyframes) - 1) * 6 + len(optimized_map_points) * 3}"
    )

    print("-" * 100)

    thresholds = [5, 10, 20, 50, 100, 500, 1000, 5000]

    for t in thresholds:
        count = np.sum(np.abs(residuals_before) > t)
        print(f"Residual > {t:5}: {count}")

    print("-" * 100)

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
        f_scale=5.0,
        method="trf",
        jac_sparsity=jac_sparsity,
        max_nfev=ba_max_nfev,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        verbose=2
    )

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

    print("-" * 100)
    print("Residual thresholds AFTER BA")

    for t in thresholds:
        count = np.sum(np.abs(residuals_after) > t)
        print(f"Residual > {t:5}: {count}")

    print("-" * 100)

    print("\n=== AFTER BA ===")
    print("Mean abs residual:", np.mean(np.abs(residuals_after)))
    print("Median abs residual:", np.median(np.abs(residuals_after)))
    print("Max abs residual:", np.max(np.abs(residuals_after)))

    print("-" * 100)

    max_shift = 0
    max_kf_id = None

    for kf in optimized_keyframes:

        old_t = kf_before[kf.id]["t"]
        new_t = kf.t

        translation_shift = np.linalg.norm(
            new_t - old_t
        )

        print(
            f"KF {kf.id}: "
            f"translation shift = {translation_shift:.6f}"
        )

        if translation_shift > max_shift:
            max_shift = translation_shift
            max_kf_id = kf.id

    print("-" * 100)

    print(
        f"Max camera shift: KF {max_kf_id}, "
        f"shift = {max_shift:.6f}"
    )

    print("-" * 100)

    print("Bundle Adjustment result")
    print("Success:", result.success)
    print("Message:", result.message)
    print("Final cost:", result.cost)
    print("Optimality:", result.optimality)
    print("Function evaluations:", result.nfev)

    print("-" * 100)

    return result