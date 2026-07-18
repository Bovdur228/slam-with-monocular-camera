import cv2 as cv
import numpy as np

from scipy.sparse import lil_matrix

from slam.tracking import project_point


def reprojection_error(
    point_world,
    observed_uv,
    kf_R,
    kf_t,
    K
):
    """
    Считает reprojection error для одной точки.
    """
    
    predicted_uv = project_point(
        point_world,
        kf_R,
        kf_t,
        K
    )

    if predicted_uv is None:
        return float("inf")

    error = np.linalg.norm(
        predicted_uv - observed_uv
    )

    return error

# -------------------------------------------------------------------------------------------------------------------------------------------
def build_ba_indices(
    map_points,
    keyframes
):
    """
    Назначает каждой MapPoint уникальный индекс в векторе параметров
    Bundle Adjustment.

    Создаёт словарь соответствия:

        MapPoint -> индекс в параметрах оптимизатора

    Это позволяет быстро находить участок общего вектора параметров,
    отвечающий за координаты конкретной MapPoint.
    """
    idx_to_mp = {}
    idx_to_kf = {}

    for mp in map_points:
        idx_to_mp[mp.id] = mp

    for kf in keyframes:
        idx_to_kf[kf.id] = kf

    return idx_to_mp, idx_to_kf

# --------------------------------------------------------------------------------------------------------------------------------------------
def collect_ba_observations(
    map_points,
    K
):
    """
    Собирает все наблюдения (Observations), которые будут использоваться
    в Bundle Adjustment.

    Для каждой MapPoint проходит по списку её наблюдений и формирует
    набор данных вида:

        MapPoint
        KeyFrame
        keypoint_idx

    Возвращает список всех связей между точками карты и KeyFrames,
    необходимых для вычисления reprojection error.
    """
    ba_observations = []

    for mp in map_points:
        for obs in mp.observations:

            keyframe = obs.keyframe
            observed_uv = keyframe.kp[obs.keypoint_idx].pt

            # проверяем что точка вообще проецируется перед камерой
            predicted_uv = project_point(
                mp.position,
                keyframe.R,
                keyframe.t,
                K
            )

            if predicted_uv is None:
                continue  # точка за камерой — пропускаем это наблюдение

            ba_observations.append((mp.id, keyframe.id, observed_uv))

    return ba_observations

# -------------------------------------------------------------------------------------------
def compute_residual(
        point_world,
        observed_uv,
        kf_R,
        kf_t,
        K
):
    """
    Вычисляет reprojection residual для одного наблюдения.

    Проецирует 3D-точку мира на изображение соответствующего KeyFrame,
    после чего вычисляет разницу между:

        - наблюдаемыми пиксельными координатами
        - предсказанными пиксельными координатами

    Возвращает двумерный residual:

        [du, dv]

    который будет минимизироваться во время Bundle Adjustment.
    """
    predicted_uv = project_point(
        point_world,
        kf_R,
        kf_t,
        K,
    )

    if predicted_uv is None:
        return 1e4, 1e4
    
    dx = predicted_uv[0] - observed_uv[0]
    dy = predicted_uv[1] - observed_uv[1]

    return dx, dy

#---------------------------------------------------------------------------------------
def build_residual_vector(
    ba_observations,
    id_to_mp,
    id_to_kf,
    K
):
    """
    Формирует общий вектор residuals для всех наблюдений.

    Последовательно вычисляет residual каждого Observation и
    объединяет их в один большой вектор вида:

        [du1, dv1,
         du2, dv2,
         du3, dv3,
         ...]

    Именно этот вектор ошибок передаётся оптимизатору
    scipy.optimize.least_squares.
    """
    residual_vector = []

    for mp_id, kf_id, observed_uv in ba_observations:
        mp = id_to_mp[mp_id]
        point_world = mp.position

        kf = id_to_kf[kf_id]
        kf_R = kf.R
        kf_t = kf.t

        dx, dy = compute_residual(point_world, observed_uv, kf_R, kf_t, K)

        residual_vector.append(dx)
        residual_vector.append(dy)

    return np.array(residual_vector)

# ----------------------------------------------------------------------------------------------------------------------------------
def pack_mappoint_parameters(
        map_points,
):
    """
    Упаковывает координаты всех MapPoints в единый вектор параметров.

    Преобразует набор точек:

        [(X1,Y1,Z1),
         (X2,Y2,Z2),
         ...]

    в плоский массив:

        [X1,Y1,Z1,
         X2,Y2,Z2,
         ...]

    который может быть передан оптимизатору Bundle Adjustment.
    """
    packed_mappoint_parameters = []

    for mp in map_points:
        packed_mappoint_parameters.append(mp.position[0])
        packed_mappoint_parameters.append(mp.position[1])
        packed_mappoint_parameters.append(mp.position[2])

    return np.array(packed_mappoint_parameters)

# -------------------------------------------------------------------------------------------------------------------------------------
def unpack_mappoint_parameters(
    packed_map_point_parameters,
    optimized_map_points
):
    """
    Распаковывает вектор параметров Bundle Adjustment обратно
    в координаты MapPoints.

    Выполняет обратную операцию по отношению к
    pack_mappoint_parameters().

    Обновляет позиции MapPoints значениями, найденными
    оптимизатором.
    """
    packed_points = np.array(
        packed_map_point_parameters
    ).reshape(-1, 3)

    for mp, point in zip(optimized_map_points, packed_points):
        mp.position = point

# ------------------------------------------------------------------------------------------------------------------------------------------
def pack_keyframe_parameters(keyframes):
    """
    Пакует ВСЕ keyframes, КРОМЕ первого (он — fixed/anchor).
    """
    packed = []

    for kf in keyframes[1:]:
        rvec, _ = cv.Rodrigues(kf.R)

        packed.extend(rvec.flatten())
        packed.extend(kf.t.flatten())

    return np.array(packed)

# ------------------------------------------------------------------------------------------------------------------------------------------
def unpack_keyframe_parameters(packed, keyframes):
    """
    Распаковывает ВСЕ keyframes, КРОМЕ первого (он — fixed/anchor).
    """
    for i, kf in enumerate(keyframes[1:]):
        rvec = packed[i*6 : i*6 + 3]
        t    = packed[i*6 + 3 : i*6 + 6]

        kf.R, _ = cv.Rodrigues(rvec)   # rvec → R
        kf.t = t.reshape(3, 1)

# -------------------------------------------------------------------------------------------------------------------------------------------
def pack_all_parameters(
    packed_keyframe_parameters,
    packed_mappoint_parameters
):

    packed_all_parameters = np.concatenate(
    [
        packed_keyframe_parameters,
        packed_mappoint_parameters
    ]
)

    return packed_all_parameters

# -------------------------------------------------------------------------------------------------------------------------------------------
def unpack_all_parameters(
    packed_all_parameters,
    keyframes,
    map_points
):
    all_keyframes = (len(keyframes) - 1) * 6

    packed_keyframes = packed_all_parameters[:all_keyframes]
    packed_map_points = packed_all_parameters[all_keyframes:]

    unpack_keyframe_parameters(packed_keyframes, keyframes)
    unpack_mappoint_parameters(packed_map_points, map_points)


# -------------------------------------------------------------------------------------------------------------------------------------------
def ba_residuals(
    packed_all_parameters,
    ba_observations,
    keyframes,
    map_points,
    id_to_mp,
    id_to_kf,
    K
):
    """
    Целевая функция Bundle Adjustment для scipy.optimize.least_squares.

    Получает текущий вектор параметров KeyFrames и MapPoints, обновляет
     позу Keyframes и координаты точек карты, и вычисляет reprojection residuals
    для всех наблюдений.

    Возвращает единый вектор ошибок:

        [du1, dv1,
         du2, dv2,
         ...]

    который минимизируется оптимизатором во время Bundle Adjustment.

    Чем меньше значения residuals, тем лучше согласованы
    положения MapPoints с наблюдениями KeyFrames.
    """

    unpack_all_parameters(packed_all_parameters, keyframes, map_points)

    residual_vector = build_residual_vector(
        ba_observations,
        id_to_mp,
        id_to_kf,
        K
    )

    return residual_vector

# ------------------------------------------------------------------------------------------------------------------------------------
def build_jac_sparsity(ba_observations, optimized_keyframes, optimized_map_points):

    free_keyframes = optimized_keyframes[1:]

    n_residuals = len(ba_observations) * 2
    n_kf_params = len(free_keyframes) * 6
    n_mp_params = len(optimized_map_points) * 3
    n_params = n_kf_params + n_mp_params

    kf_id_to_idx = {kf.id: i for i, kf in enumerate(free_keyframes)}
    mp_id_to_idx = {mp.id: i for i, mp in enumerate(optimized_map_points)}

    sparsity = lil_matrix((n_residuals, n_params), dtype=int)

    for obs_idx, (mp_id, kf_id, _) in enumerate(ba_observations):

        row_dx = obs_idx * 2
        row_dy = obs_idx * 2 + 1

        if kf_id in kf_id_to_idx:
            kf_idx = kf_id_to_idx[kf_id]
            for col_offset in range(6):
                col = kf_idx * 6 + col_offset
                sparsity[row_dx, col] = 1
                sparsity[row_dy, col] = 1

        if mp_id in mp_id_to_idx:
            mp_idx = mp_id_to_idx[mp_id]
            for col_offset in range(3):
                col = n_kf_params + mp_idx * 3 + col_offset
                sparsity[row_dx, col] = 1
                sparsity[row_dy, col] = 1

    return sparsity

# ----------------------------------------------------------------------------
def collect_and_filter_observations(
    map_points,
    K,
    max_initial_residual=50.0
):
    ba_observations = collect_ba_observations(map_points, K)

    ba_observations_clean = []

    mp_lookup = {
    mp.id: mp
    for mp in map_points
    }

    kf_lookup = {}
    for mp in map_points:
        for obs in mp.observations:
            kf_lookup[obs.keyframe.id] = obs.keyframe

    for mp_id, kf_id, observed_uv in ba_observations:

        mp = mp_lookup[mp_id]
        kf = kf_lookup[kf_id]

        err = reprojection_error(
            mp.position,
            np.array(observed_uv),
            kf.R,
            kf.t,
            K
        )

        if err < max_initial_residual:
            ba_observations_clean.append(
                (mp_id, kf_id, observed_uv)
            )

    #print(
        #f"Observations before filter: {len(ba_observations)}"
    #)

    #print(
        #f"Observations after filter: {len(ba_observations_clean)}"
    #)

    return ba_observations_clean

# --------------------------------------------------------------------------------------------------------------------------------
def filter_mappoints_by_observations(
    map_points,
    ba_observations,
    min_observations=3
):
    """
    Удаляет MapPoints, которые наблюдались
    меньше чем в min_observations уникальных KeyFrames.
    """

    mp_to_keyframes = {}

    for mp_id, kf_id, _ in ba_observations:
        if mp_id not in mp_to_keyframes:
            mp_to_keyframes[mp_id] = set()

        mp_to_keyframes[mp_id].add(kf_id)

    filtered_points = [
        mp
        for mp in map_points
        if len(mp_to_keyframes.get(mp.id, set())) >= min_observations
    ]

    return filtered_points

# ---------------------------------------------------------------------------------------------------------------------------------
def filter_keyframes_by_observations(
    keyframes,
    ba_observations,
    fixed_keyframe_id,
    min_observations=10
):
    """
    Удаляет KeyFrames, которые видят слишком мало
    уникальных MapPoints.
    """

    kf_to_mappoints = {}

    for mp_id, kf_id, _ in ba_observations:
        if kf_id not in kf_to_mappoints:
            kf_to_mappoints[kf_id] = set()

        kf_to_mappoints[kf_id].add(mp_id)

    filtered_keyframes = [
        kf
        for kf in keyframes
        if kf.id == fixed_keyframe_id
        or len(kf_to_mappoints.get(kf.id, set())) >= min_observations
    ]

    return filtered_keyframes

# --------------------------------------------------------------------------------------------------------------------------------
def filter_observations(
    ba_observations,
    map_points,
    keyframes
):
    """
    Оставляет только observations,
    относящиеся к существующим MapPoints
    и существующим KeyFrames.
    """

    valid_mp_ids = {mp.id for mp in map_points}

    valid_kf_ids = {kf.id for kf in keyframes}

    filtered_observations = [
        (mp_id, kf_id, uv)
        for mp_id, kf_id, uv in ba_observations
        if mp_id in valid_mp_ids
        and kf_id in valid_kf_ids
    ]

    return filtered_observations

# ---------------------------------------------------------------------------------------------------------------------------------
def deduplicate_observations(
    ba_observations
):
    """
    Оставляет максимум одно observation для пары:
        (MapPoint, KeyFrame)

    Если одна MapPoint несколько раз наблюдалась в одном KeyFrame,
    оставляется первое observation.
    """

    seen_pairs = set()
    unique_observations = []

    for mp_id, kf_id, observed_uv in ba_observations:

        pair = (mp_id, kf_id)

        if pair in seen_pairs:
            continue

        seen_pairs.add(pair)

        unique_observations.append(
            (mp_id, kf_id, observed_uv)
        )

    return unique_observations

# ---------------------------------------------------------------------------------------------------------------------------------
def collect_clean_observations_for_graph(
    map_points,
    keyframes,
    K,
    max_initial_residual=50.0
):
    observations = collect_and_filter_observations(
        map_points,
        K,
        max_initial_residual
    )

    observations = filter_observations(
        observations,
        map_points,
        keyframes
    )

    observations = deduplicate_observations(
        observations
    )

    return observations

# ------------------------------------------------------------------------------------------------------
def cleanup_ba_graph(
    fixed_keyframe,
    map_points,
    K,
    min_mp_observations=2, # временные значения, потом возможно исправить
    min_kf_observations=5, # временные значения, потом возможно исправить
    max_initial_residual=50,
    max_iterations=20
):
    keyframes = list(fixed_keyframe)
    map_points = list(map_points)

    fixed_keyframe_id = keyframes[0].id
    observations = []

    for iteration in range(1, max_iterations + 1):

        old_counts = (
            len(map_points),
            len(keyframes),
            len(observations)
        )

        print(
            f"Before Iteration {iteration}: "
            f"{len(keyframes)} KFs, "
            f"{len(map_points)} MPs, "
            f"{len(observations)} observations"
        )
        print("-" * 100)

        # 1. Собираем observations только внутри текущего графа
        observations = collect_clean_observations_for_graph(
            map_points,
            keyframes,
            K,
            max_initial_residual
        )

        # 2. Удаляем слабые MapPoints
        map_points = filter_mappoints_by_observations(
            map_points,
            observations,
            min_mp_observations
        )

        observations = collect_clean_observations_for_graph(
            map_points,
            keyframes,
            K,
            max_initial_residual
        )

        # 3. Удаляем слабые KeyFrames, но сохраняем fixed
        keyframes = filter_keyframes_by_observations(
            keyframes,
            observations,
            fixed_keyframe_id,
            min_kf_observations
        )

        observations = collect_clean_observations_for_graph(
            map_points,
            keyframes,
            K,
            max_initial_residual
        )

        # 4. После удаления KeyFrames снова удаляем MapPoints, которые потеряли observations
        map_points = filter_mappoints_by_observations(
            map_points,
            observations,
            min_mp_observations
        )

        observations = collect_clean_observations_for_graph(
            map_points,
            keyframes,
            K,
            max_initial_residual
        )

        new_counts = (
            len(map_points),
            len(keyframes),
            len(observations)
        )

        print(
            f"After Iteration {iteration}: "
            f"{len(keyframes)} KFs, "
            f"{len(map_points)} MPs, "
            f"{len(observations)} observations"
        )
        print("-" * 100)

        if old_counts == new_counts:
            break

    else:
        print("cleanup_ba_graph: reached max_iterations without full convergence.")

    idx_to_mp, idx_to_kf = build_ba_indices(
        map_points,
        keyframes
    )

    return keyframes, map_points, observations, idx_to_mp, idx_to_kf