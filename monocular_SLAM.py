import cv2 as cv
import numpy as np
import os

from collections import Counter

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


# Класс MapPoint и его вспомогательные классы ----------------------------------------------------------------------------------------------------

# вспомогательный класс Observation, где хранятся KeyFrames, которые наблюдали конкретную Map Point и их keypoint, под которым она наблюдалась
class Observation:

    def __init__(
        self,
        keyframe,
        keypoint_idx
    ):

        self.keyframe = keyframe
        self.keypoint_idx = keypoint_idx

# Класс, описывающий 3D точку в мире
class MapPoint:

    def __init__(
        self,
        position,
        descriptor
    ):

        self.position = position
        self.descriptor = descriptor

        # список из объектов класса Observation, где хранятся KeyFrames, которые наблюдали эту Map Point и их keypoint, под которым она наблюдалась
        self.observations = []
        self.num_observations = 1

        self.id = None


# Класс, описывающий строение KeyFrame. Хранит состояние камеры в некоторый важный момент времени ----------------------------------------------
class KeyFrame:

    def __init__(
        self,
        R,
        t,
        kp,
        des
    ):

        self.R = R
        self.t = t

        self.kp = kp
        self.des = des

        self.id = None

        self.map_points = []


# Временное хранилище наблюдений текущего кадра ------------------------------------------------------------------------------------------------
class FrameObservation:

    def __init__(
        self,
        point_world,
        descriptor,
        kp_idx
    ):
        self.point_world = point_world
        self.descriptor = descriptor
        self.kp_idx = kp_idx


# функции =======================================================================================================================================
def find_existing_mappoint(
    descriptor,
    candidate_map_points,
    threshold=30
):
    """
    Ищет похожую MapPoint среди заранее выбранных кандидатов.

    Важно:
    candidate_map_points должен быть не всей глобальной картой,
    а local map / рабочим списком кандидатов.


    ищет существующую Map Point, проходясь по дескрипторам всех Map Points и сравнивая их с дескриптором "кандидата".
    если нашли, значит такая Map Point уже существует и мы возвращаем её.
    если не нашли, то значит такой Map Point на карте нет и мы возвращаем None.
    """

    best_mp = None
    best_distance = float("inf")

    for mp in candidate_map_points:

        if mp.descriptor is None:
            continue

        distance = cv.norm(
            descriptor,
            mp.descriptor,
            cv.NORM_HAMMING
        )

        if distance < best_distance:
            best_distance = distance
            best_mp = mp

    if best_distance < threshold:
        return best_mp

    return None

# ----------------------------------------------------------------------------------------------------------------------------------------------------
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

# ФУНКЦИИ ДЛЯ BUNDLE ADJUSTMENT ===============================================================================================================

#------------------------------------------------------------------------------------------------------------------------------------------------
def project_point(
    point_world,
    kf_R,
    kf_t,
    K
):
    """
    Переводит мировую точку Pw в Pc и затем в пиксельные координаты изображения.
    """

    Pc = kf_R.T @ (point_world.reshape(3,1) - kf_t)

    if Pc[2, 0] <= 0:
        return None

    pixel = K @ Pc

    u = pixel[0,0] / pixel[2,0]
    v = pixel[1,0] / pixel[2,0]

    return np.array([u,v])

# --------------------------------------------------------------------------------------------------------------------------------------------
def is_inside_image(
    uv,
    width,
    height,
    margin=20
):
    """
    Проверяет, находится ли projected point внутри изображения.

    margin нужен, чтобы не использовать точки слишком близко к краям,
    где matching обычно менее стабилен.
    """

    u, v = uv

    return (
        margin <= u < width - margin
        and margin <= v < height - margin
    )

# ------------------------------------------------------------------------------------------------------------------------------------------
def get_local_mappoints(
    keyframes,
    map_points,
    num_recent_keyframes=10,
    fallback_max_points=3000
):
    """
    Собирает local map — MapPoints, которые наблюдались
    в последних num_recent_keyframes KeyFrames.

    Если KeyFrames ещё нет, временно возвращает последние fallback_max_points
    из общей карты.
    """

    # Если KeyFrames ещё нет, fallback на последние точки
    if len(keyframes) == 0:
        return list(map_points[-fallback_max_points:])

    local_mappoints_by_id = {}

    recent_keyframes = keyframes[-num_recent_keyframes:]

    for kf in recent_keyframes:
        for mp in kf.map_points:

            if mp is None:
                continue

            if mp.id is None:
                continue

            local_mappoints_by_id[mp.id] = mp

    # Если почему-то последние KeyFrames не дали точек,
    # тоже fallback на последние точки карты
    if len(local_mappoints_by_id) == 0:
        return list(map_points[-fallback_max_points:])

    return list(local_mappoints_by_id.values())

# ------------------------------------------------------------------------------------------------------------------------------------------
def track_existing_mappoints(
    kp,
    des,
    map_points,
    global_R,
    global_t,
    K,
    image_shape,
    search_radius=40,
    descriptor_threshold=50,
    max_points=3000
):
    """
    Пытается найти уже существующие MapPoints в текущем кадре.

    Логика:
        1. Берём существующую MapPoint.
        2. Проецируем её в текущий кадр через project_point().
        3. Если проекция попала в изображение, ищем рядом ORB keypoints.
        4. Среди близких keypoints выбираем тот, у которого descriptor
           наиболее похож на descriptor MapPoint.
        5. return возвращает список два результата:
               1) список найденных MapPoints: [(MapPoint, keypoint_idx), ...]
               2) результат действия функции
    """

    if des is None or len(kp) == 0 or len(map_points) == 0:
        return [], {
            "candidates": 0,
            "projected": 0,
            "inside": 0,
            "tracked": 0
        }

    height, width = image_shape[:2]

    tracked_map_points = []

    used_kp_idxes = set()
    used_mp_ids = set()

    projected_count = 0
    inside_count = 0

    kp_points = np.array(
        [keypoint.pt for keypoint in kp],
        dtype=np.float32
    )

    if max_points is not None and len(map_points) > max_points:
        candidate_map_points = map_points[-max_points:]
    else:
        candidate_map_points = map_points

    for mp in candidate_map_points:

        if mp.descriptor is None:
            continue

        projected_uv = project_point(
            mp.position,
            global_R,
            global_t,
            K
        )

        if projected_uv is None:
            continue

        projected_count += 1

        if not is_inside_image(
            projected_uv,
            width,
            height
        ):
            continue

        inside_count += 1

        distances_2d = np.linalg.norm(
            kp_points - projected_uv.reshape(1, 2),
            axis=1
        )

        nearby_kp_idxes = np.where(
            distances_2d < search_radius
        )[0]

        best_kp_idx = None
        best_descriptor_distance = float("inf")

        for kp_idx in nearby_kp_idxes:

            kp_idx = int(kp_idx)

            if kp_idx in used_kp_idxes:
                continue

            descriptor_distance = cv.norm(
                mp.descriptor,
                des[kp_idx],
                cv.NORM_HAMMING
            )

            if descriptor_distance < best_descriptor_distance:
                best_descriptor_distance = descriptor_distance
                best_kp_idx = kp_idx

        if (
            best_kp_idx is not None
            and best_descriptor_distance < descriptor_threshold
            and mp.id not in used_mp_ids
        ):
            tracked_map_points.append(
                (mp, best_kp_idx)
            )

            used_kp_idxes.add(best_kp_idx)
            used_mp_ids.add(mp.id)

    stats = {
        "candidates": len(candidate_map_points),
        "projected": projected_count,
        "inside": inside_count,
        "tracked": len(tracked_map_points)
    }

    return tracked_map_points, stats

# -----------------------------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------------------------
def projection_matrix_from_pose(
    Rwc,
    twc,
    K
):
    """
    Создаёт projection matrix P = K @ [Rcw | tcw]
    из нашей pose convention Rwc/twc.

    У нас:
        Pw = Rwc @ Pc + twc

    Для projection нужно:
        Pc = Rcw @ Pw + tcw
    """

    Rcw = Rwc.T
    tcw = -Rcw @ twc

    P = K @ np.hstack(
        (
            Rcw,
            tcw
        )
    )

    return P

# -----------------------------------------------------------------------------------------------------------------------------------------
def triangulate_points_world_from_poses(
    pts1,
    pts2,
    prev_Rwc,
    prev_twc,
    curr_Rwc,
    curr_twc,
    K,
    max_depth=100
):
    """
    Триангулирует точки сразу в world coordinates,
    используя global pose предыдущего и текущего кадра.

    pts1 — points на предыдущем кадре
    pts2 — points на текущем кадре
    """

    P1 = projection_matrix_from_pose(
        prev_Rwc,
        prev_twc,
        K
    )

    P2 = projection_matrix_from_pose(
        curr_Rwc,
        curr_twc,
        K
    )

    points_4d = cv.triangulatePoints(
        P1,
        P2,
        pts1.T,
        pts2.T
    )

    w = points_4d[3]

    valid_w_mask = np.abs(w) > 1e-8

    points_world = np.zeros(
        (points_4d.shape[1], 3),
        dtype=np.float64
    )

    points_world[valid_w_mask] = (
        points_4d[:3, valid_w_mask] / w[valid_w_mask]
    ).T

    points_cam_prev = (
        prev_Rwc.T @ (points_world.T - prev_twc)
    ).T

    points_cam_curr = (
        curr_Rwc.T @ (points_world.T - curr_twc)
    ).T

    valid_mask = (
        valid_w_mask
        &
        np.isfinite(points_world).all(axis=1)
        &
        np.isfinite(points_cam_prev).all(axis=1)
        &
        np.isfinite(points_cam_curr).all(axis=1)
        &
        (points_cam_prev[:, 2] > 0)
        &
        (points_cam_curr[:, 2] > 0)
        &
        (points_cam_prev[:, 2] < max_depth)
        &
        (points_cam_curr[:, 2] < max_depth)
    )

    return points_world[valid_mask], valid_mask

# ------------------------------------------------------------------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------------------------------------------------------
def cull_mappoints_by_observations(
    map_points,
    min_observations=2
):
    good_points = []

    for mp in map_points:

        if len(mp.observations) >= min_observations:
            good_points.append(mp)

    return good_points

# -----------------------------------------------------------------------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------------------------------------------------------------
def print_count_distribution(
    counts,
    title
):
    """
    Печатает распределение количества наблюдений.

    counts — список чисел, например:
        [len(mp.observations) for mp in map_points]

    Используется для диагностики:
        - сколько MapPoints имеют 0 observations
        - сколько имеют 1 observation
        - сколько имеют 2
        - сколько имеют 3+
    """

    print("-" * 100)
    print(title)

    if len(counts) == 0:
        print("No items.")
        return

    counter = Counter(counts)

    total = len(counts)

    zero = counter.get(0, 0)
    one = counter.get(1, 0)
    two = counter.get(2, 0)
    three = counter.get(3, 0)

    four_to_five = sum(
        value
        for count, value in counter.items()
        if 4 <= count <= 5
    )

    six_to_ten = sum(
        value
        for count, value in counter.items()
        if 6 <= count <= 10
    )

    more_than_ten = sum(
        value
        for count, value in counter.items()
        if count > 10
    )

    print(f"Total items:        {total}")
    print(f"Mean count:         {np.mean(counts):.2f}")
    print(f"Median count:       {np.median(counts):.2f}")
    print(f"Max count:          {np.max(counts)}")

    print("-" * 100)

    print(f"0 observations:     {zero}")
    print(f"1 observation:      {one}")
    print(f"2 observations:     {two}")
    print(f"3 observations:     {three}")
    print(f"4-5 observations:   {four_to_five}")
    print(f"6-10 observations:  {six_to_ten}")
    print(f">10 observations:   {more_than_ten}")

# ---------------------------------------------------------------------------------------------------------------------------------
def print_ba_graph_diagnostics(
    map_points,
    keyframes,
    K,
    max_initial_residual=50.0,
    title="BA Graph Diagnostics"
):
    """
    Диагностирует качество BA-графа.

    Показывает:
        1. Сколько raw observations есть у MapPoints.
        2. Сколько observations приходится на KeyFrames.
        3. Сколько observations проходят projection check.
        4. Сколько observations проходят reprojection error filter.
        5. Распределение clean observations по MapPoints и KeyFrames.
    """

    print("=" * 100)
    print(title)
    print("=" * 100)

    print(f"MapPoints: {len(map_points)}")
    print(f"KeyFrames: {len(keyframes)}")

    # ------------------------------------------------------------------
    # 1. Raw observations per MapPoint
    # ------------------------------------------------------------------

    raw_mp_counts = [
        len(mp.observations)
        for mp in map_points
    ]

    print_count_distribution(
        raw_mp_counts,
        "Raw observations per MapPoint"
    )

    # ------------------------------------------------------------------
    # 2. Raw observations per selected KeyFrame
    # ------------------------------------------------------------------

    keyframe_ids = {
        kf.id
        for kf in keyframes
    }

    raw_kf_counts = {
        kf.id: 0
        for kf in keyframes
    }

    for mp in map_points:
        for obs in mp.observations:

            kf_id = obs.keyframe.id

            if kf_id in keyframe_ids:
                raw_kf_counts[kf_id] += 1

    print_count_distribution(
        list(raw_kf_counts.values()),
        "Raw observations per selected KeyFrame"
    )

    # ------------------------------------------------------------------
    # 3. Observations after projection check
    # ------------------------------------------------------------------

    ba_observations = collect_ba_observations(
        map_points,
        K
    )

    print("-" * 100)
    print(f"collect_ba_observations: {len(ba_observations)}")

    # ------------------------------------------------------------------
    # 4. Observations after reprojection error filter
    # ------------------------------------------------------------------

    clean_observations = collect_and_filter_observations(
        map_points,
        K,
        max_initial_residual
    )

    print(
        f"collect_and_filter_observations "
        f"(max residual {max_initial_residual}): "
        f"{len(clean_observations)}"
    )

    # ------------------------------------------------------------------
    # 5. Clean observations per MapPoint
    # ------------------------------------------------------------------

    clean_mp_counts = {
        mp.id: 0
        for mp in map_points
    }

    for mp_id, _, _ in clean_observations:
        if mp_id in clean_mp_counts:
            clean_mp_counts[mp_id] += 1

    print_count_distribution(
        list(clean_mp_counts.values()),
        "Clean BA observations per MapPoint"
    )

    # ------------------------------------------------------------------
    # 6. Clean observations per selected KeyFrame
    # ------------------------------------------------------------------

    clean_kf_counts = {
        kf.id: 0
        for kf in keyframes
    }

    for _, kf_id, _ in clean_observations:
        if kf_id in clean_kf_counts:
            clean_kf_counts[kf_id] += 1

    print_count_distribution(
        list(clean_kf_counts.values()),
        "Clean BA observations per selected KeyFrame"
    )

    print("=" * 100)

# ---------------------------------------------------------------------------------------------------------------------------------
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

# =============================================================================================================================================

# захват видео --------------------------------------------------------------------------------------------------------------------------------
cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# создание ORB и BFMatcher для поиска feature matching ----------------------------------------------------------------------------------------
orb = cv.ORB_create(nfeatures=1000)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# сюда будет записываться предыдущий кадр, а также его keypoints и descriptors ----------------------------------------------------------------
prev_gray = None
prev_kp = None
prev_des = None

# матрица K и distortion, взятые после калибровки камеры из файла mono_calibration.npz --------------------------------------------------------
calib_data = np.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "calibration",
        "mono_calibration.npz"
    )
)
K = calib_data["K"]
dist = calib_data["dist"]
# RMS Error: 0.13271354901396754

# тут будет обновляться глобальная pose камеры ------------------------------------------------------------------------------------------------
global_R = np.eye(3)
global_t = np.zeros((3, 1))


# 2D визуализация перемещения камеры в глобальном мире + место начала перемещения (cv.circle)--------------------------------------------------
traj_img = np.zeros(
    (800, 800, 3),
    dtype=np.uint8
)

cv.circle(
    traj_img,
    (400, 400),
    5,
    (0, 0, 255),
    -1
)

# 2D карта, где будут показываться все Map Points мира --------------------------------------------------------------------------------------------
map_img = np.zeros(
    (800, 800, 3),
    dtype=np.uint8
)

# тут хранятится Pose последнего KeyFrame ---------------------------------------------------------------------------------------------------------
last_keyFrame_t = None
last_keyFrame_R = None

# список всех созданных KeyFrames -----------------------------------------------------------------------------------------------------------------
keyframes = []
keyframe_id = 0

# список всех созданных Map Points ----------------------------------------------------------------------------------------------------------------
map_points = []
map_point_id = 0

# вспомогательная переменная для отрисовки Map Points на 2D карте, чтобы не рисовать каждый раз дубликаты -----------------------------------------
last_drawn_map_point_idx = 0

# счётчик PnP кадров и recoverPose кадров --------------------------------------------------------------------------------------------------
pnp_success_count = 0
recoverpose_count = 0


# НАЧАЛО ЗАПИСИ ===================================================================================================================================
while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # исправляем искажение камеры
    gray = cv.undistort(gray, K, dist)

# feature matching между предыдущим кадром и нынешним, а затем запись valid features в pts1 и pts2 -------------------------------------------------
    kp, des = orb.detectAndCompute(gray, None)

    if prev_des is not None and des is not None:

        matches = bf.match(prev_des, des)

        matches = sorted(matches, key=lambda x: x.distance)

        pts1 = []
        pts2 = []

        current_frame_kp_idxes = []

        matched_descriptors = []

        for match in matches:

            pts1.append(
                prev_kp[match.queryIdx].pt
            )

            pts2.append(
                kp[match.trainIdx].pt
            )

            matched_descriptors.append(
                des[match.trainIdx]
            )

            current_frame_kp_idxes.append(
                match.trainIdx
            )

        pts1 = np.array(
            pts1,
            dtype=np.float32
        )

        pts2 = np.array(
            pts2,
            dtype=np.float32
        )

        matched_descriptors = np.array(
            matched_descriptors
        )

        current_frame_kp_idxes = np.array(
            current_frame_kp_idxes
        )

        # результат по дефолту (временное название) 
        pose_source = "none"
        local_map_points = []

        tracking_stats = {
            "candidates": 0,
            "projected": 0,
            "inside": 0,
            "tracked": 0,
            "tracked_before_pnp": 0,
            "tracked_used": 0,
            "pnp_points": 0,
            "pnp_inliers": 0,
            "pnp_inlier_ratio": 0.0
        }

# если между двумя кадрами нашли 8 или более matches, то вычисляем по ним Essential matrix (E) --------------------------------------------------
        if len(pts1) >= 8:

            E, mask = cv.findEssentialMat(
                pts1,
                pts2,
                K,
                method=cv.RANSAC,
                prob=0.999,
                threshold=1.0
            )

            if E is None:
                continue

            
            # после RANSAC откидываем с помощью mask плохие matches и оставляем в pts1 и pts2 только хорошие
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]

            # также после RANSAC сохраняем descriptors у оставшихся после mask хороших matches
            valid_descriptors = matched_descriptors[mask.ravel() == 1]

            valid_kp_idxes = current_frame_kp_idxes[mask.ravel() == 1]

            # если после RANSAC осталось меньше 8-ми хороших features, то пропускаем данный кадр
            if len(pts1) < 8:
                continue

# далее с помощью матрицы E, pts1, pts2 и матрицы K вычисляем локальное смещение (R,t) камеры между двумя кадрами --------------------------------
            _, R, t, pose_mask = cv.recoverPose(
                E,
                pts1,
                pts2,
                K
            )

# фильтруем точки pts1 и pts2 по pose_mask -----------------------------------------------------------------------------------------------------
            pose_mask = pose_mask.ravel() > 0

            pts1 = pts1[pose_mask]
            pts2 = pts2[pose_mask]

            valid_descriptors = valid_descriptors[pose_mask]
            valid_kp_idxes = valid_kp_idxes[pose_mask]

            if len(pts1) < 8:
                continue

# обновляем глобальное перемещение камеры в мире -------------------------------------------------------------------------------------------------
            # сохраняем pose предыдущего кадра
            prev_Rwc_for_triangulation = global_R.copy()
            prev_twc_for_triangulation = global_t.copy()

            # recoverPose даёт initial estimate текущей pose
            initial_global_R = prev_Rwc_for_triangulation @ R.T
            initial_global_t = (
                prev_twc_for_triangulation
                - prev_Rwc_for_triangulation @ R.T @ t
            )

            global_R = initial_global_R.copy()
            global_t = initial_global_t.copy()

            pose_source = "recoverPose"

# track already existing MapPoints in the current frame----------------------------------------------------------------------------------------
            local_map_points = get_local_mappoints(
                keyframes,
                map_points,
                num_recent_keyframes=10,
                fallback_max_points=3000
            )
            
            tracked_map_points, tracking_stats = track_existing_mappoints(
                kp,
                des,
                local_map_points,
                global_R,
                global_t,
                K,
                gray.shape,
                search_radius=40,
                descriptor_threshold=50,
                max_points=None
            )

            pnp_success, pnp_Rwc, pnp_twc, pnp_inlier_tracked_points, pnp_stats = estimate_pose_pnp(
                tracked_map_points,
                kp,
                K,
                min_points=15,
                min_inliers=12,
                reprojection_error_threshold=12.0,
                confidence=0.99,
                iterations_count=100
            )

            if pnp_success:
                global_R = pnp_Rwc.copy()
                global_t = pnp_twc.copy()

                # оставляем только PnP-inlier tracked points, чтобы outlier matches не попали в KeyFrame observations
                tracked_map_points = pnp_inlier_tracked_points

                pose_source = "PnP"

            tracking_stats["tracked_before_pnp"] = tracking_stats["tracked"]
            tracking_stats["tracked_used"] = len(tracked_map_points)

            tracking_stats["pnp_points"] = pnp_stats["points"]
            tracking_stats["pnp_inliers"] = pnp_stats["inliers"]
            tracking_stats["pnp_inlier_ratio"] = pnp_stats["inlier_ratio"]

            if pose_source == "PnP":
                pnp_success_count += 1
            else:
                recoverpose_count += 1

# список tracked MapPoints, которые попали в этот кадр ----------------------------------------------------------------------------------
            current_frame_map_points = list(tracked_map_points)

            used_mappoint_ids = {
                mp.id
                for mp, _ in current_frame_map_points
            }

            used_kp_idxes = {
                kp_idx
                for _, kp_idx in current_frame_map_points
            }

# триангулируем новые MapPoints сразу в мировых координатах, используя global pose предыдущего и текущего кадра --------------------------
            points_world, triangulation_valid_mask = triangulate_points_world_from_poses(
                pts1,
                pts2,
                prev_Rwc_for_triangulation,
                prev_twc_for_triangulation,
                global_R,
                global_t,
                K,
                max_depth=100
            )

            valid_descriptors = valid_descriptors[triangulation_valid_mask]
            valid_kp_idxes = valid_kp_idxes[triangulation_valid_mask]

            if len(points_world) == 0:
                continue

# Формирует список наблюдений текущего кадра. Для каждой точки сохраняет её мировые координаты, дескриптор и индекс соответствующего keypoint --
            frame_observations = []

            for point, descriptor, kp_idx in zip(
                points_world,
                valid_descriptors,
                valid_kp_idxes
            ):

                frame_observations.append(
                    FrameObservation(
                        point,
                        descriptor,
                        kp_idx
                    )
    )

# критерии оценки новых и уже существующих Map Points, чтобы не добавлять дубликаты в список map_points -------------------------------------------

            # счётчик найденных новых Map Points, увеличиваем его каждый раз, если находим новую Map Point
            new_points_count = 0

            association_candidates = list(local_map_points)

            # проходимся в каждой найденной в кадре Map Point и её дескриптору
            for obs in frame_observations:

                if obs.kp_idx in used_kp_idxes:
                    continue

                existing_mp = find_existing_mappoint(
                    obs.descriptor,
                    association_candidates,
                    threshold=20
                )

                # если такая Map Point в мире уже существует, то не создаём новую Map Point
                if existing_mp is not None:

                    if existing_mp.id in used_mappoint_ids:
                        continue
                    
                    # ВРЕМЕННЫЙ КОСТЫЛЬ, ВМЕСТО BUNDLE ADJUSTMENT! После каждого нового наблюдения точки, проводим статистическое усреднение её позиции
                    existing_mp.position = (existing_mp.position * existing_mp.num_observations + obs.point_world) / (
                                                    existing_mp.num_observations + 1
                                                )
                    
                    existing_mp.num_observations += 1

                    # ТОЖЕ ВРЕМЕННЫЙ КОСТЫЛЬ. ПОТОМ ИСПРАВИТЬ НА ОДНО ИЗ ЭТОГО: descriptor voting, медианный descriptor, лучший descriptor
                    existing_mp.descriptor = obs.descriptor

                    current_frame_map_points.append((existing_mp, obs.kp_idx))

                    used_mappoint_ids.add(existing_mp.id)
                    used_kp_idxes.add(obs.kp_idx)

                # а если не существует, то создаём новый объект класса MapPoint и добавляем её в map_points[]
                else:

                    new_points_count += 1

                    mp = MapPoint(
                        obs.point_world,
                        obs.descriptor
                    )

                    mp.id = map_point_id
                    map_point_id += 1

                    map_points.append(mp)

                    current_frame_map_points.append((mp, obs.kp_idx))

                    used_mappoint_ids.add(mp.id)
                    used_kp_idxes.add(obs.kp_idx)

                    association_candidates.append(mp)
            
            # считаем долю новых Map Points среди всех замеченных в кадре Map Points
            if len(current_frame_map_points) != 0:
                new_points_ratio = new_points_count / len(current_frame_map_points)
            else:
                new_points_ratio = 0

# критерии создания и само создание KeyFrames, сохраняем его данные (глобальные R и t камеры, keypoints и descriptors) -------------------------------

            # если у нас нет ни одного KeyFrame, то создаём его в любом случае
            if not keyframes:
                
                create_KeyFrame(
                    global_R,
                    global_t,
                    kp,
                    des,
                    current_frame_map_points,
                    keyframes,
                    keyframe_id
                )
                keyframe_id += 1

                last_keyFrame_t = global_t.copy()
                last_keyFrame_R = global_R.copy()

                #print(f"Keyframe saved: {len(keyframes)}")
                #print(len(map_points))

            # если хотя бы один KeyFrame есть, то смотрим, нужно ли создать новый KeyFrame
            elif last_keyFrame_t is not None and last_keyFrame_R is not None:

                # на каждом кадре считаем насколько далеко камера переместилась, относительно последнего KeyFrame
                translation = np.linalg.norm(global_t - last_keyFrame_t)

                # на каждом кадре считаем насколько сильно камера повернулась, относительно последнего KeyFrame
                R_delta = last_keyFrame_R.T @ global_R
                rvec, _ = cv.Rodrigues(R_delta)
                rotation = np.linalg.norm(rvec)

                # если камера достаточно далеко переместилась или повернулась, или в кадре заметили много новых Map Points, создаём новый KeyFrame
                if translation > 10 or rotation > 0.6 or (new_points_ratio >= 0.5 and len(map_points) > 200):
                
                    create_KeyFrame(
                        global_R,
                        global_t,
                        kp,
                        des,
                        current_frame_map_points,
                        keyframes,
                        keyframe_id
                    )
                    keyframe_id += 1

                    last_keyFrame_t = global_t.copy()
                    last_keyFrame_R = global_R.copy()

                    print(f"Keyframe saved: {len(keyframes)}")
                    #print(len(map_points))

# рисуем 2D карту мира (вид сверху) --------------------------------------------------------------------------------------------------------------

            # тут визуализация перемещения самой камеры в мире
            x_cam = int(global_t[0, 0]) + 400
            z_cam = int(global_t[2, 0]) + 400

            cv.circle(
                traj_img,
                (x_cam, z_cam),
                2,
                (0, 255, 0),
                -1
            )

            # тут визуализация map_points (Pw) в мире
            for p in map_points[last_drawn_map_point_idx:]:
                map_point_x = int(p.position[0]) + 400
                map_point_z = int(p.position[2]) + 400

                cv.circle(
                    map_img, 
                    (map_point_x, map_point_z), 
                    1, (255, 255, 255), 
                    -1)

            last_drawn_map_point_idx = len(map_points)
        

# склеиваем предыдущий кадр (слева) и текущий кадр (справа) и проводим линии между их matches --------------------------------------------------
        draw_img = cv.drawMatches(
            prev_gray,
            prev_kp,
            gray,
            kp,
            matches[:50],
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # в верхнем левом углу выводим текстом количество matches
        cv.putText(
            draw_img,
            f"Matches: {len(matches)}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv.putText(
            draw_img,
            f"Pose: {pose_source}",
            (10, 65),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv.putText(
            draw_img,
            f"Local MPs: {len(local_map_points)}",
            (10, 100),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv.putText(
            draw_img,
            f"Tracked: {tracking_stats['tracked_used']}/{tracking_stats['tracked_before_pnp']}",
            (10, 135),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv.putText(
            draw_img,
            f"Projected/Inside: {tracking_stats['projected']}/{tracking_stats['inside']}",
            (10, 170),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv.putText(
            draw_img,
            f"PnP inliers: {tracking_stats['pnp_inliers']}/{tracking_stats['pnp_points']}",
            (10, 205),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv.imshow("Feature Matching", draw_img)

        cv.imshow(
            "Trajectory",
            traj_img
        )
        cv.imshow(
            "Point Cloud", 
            map_img)

# в конце текущий кадр, а также его keypoints и descriptors, становится предыдущим кадром и начинаем новую итерацию цикла ----------------------
    prev_gray = gray
    prev_kp = kp
    prev_des = des

    if cv.waitKey(1) & 0xFF == 27:
        if len(map_points) > 0:
            print(f"Map Points count: {len(map_points)}")
            print(f"KeyFrames count: {len(keyframes)}")

        break

cap.release()
cv.destroyAllWindows()

# ТЕСТЫ =========================================================================================================================================

print("=" * 100)
print(f"PnP frames: {pnp_success_count}")
print(f"recoverPose frames: {recoverpose_count}")
# =====================================================================
# 1. Выбираем KeyFrames
# =====================================================================

print("-" * 100)

test_keyframes = keyframes[:30]

if len(test_keyframes) < 2:
    print("Слишком мало KeyFrames для BA.")
    print(f"KeyFrames: {len(test_keyframes)}")
    exit()

fixed_keyframe = test_keyframes[0]
candidate_keyframes = test_keyframes[1:]

# =====================================================================
# 2. Собираем все MapPoints, которые видели эти KeyFrames
# =====================================================================

test_kf_ids = {kf.id for kf in test_keyframes}

test_map_points = [
    mp for mp in map_points
    if any(obs.keyframe.id in test_kf_ids for obs in mp.observations)
]

print_ba_graph_diagnostics(
    test_map_points,
    test_keyframes,
    K,
    max_initial_residual=50,
    title="BEFORE cleanup_ba_graph"
)
print("-" * 100)

# =====================================================================
# 3. Build clean BA graph
# =====================================================================

optimized_keyframes, optimized_map_points, ba_observations, idx_to_mp, idx_to_kf = cleanup_ba_graph(
    [fixed_keyframe] + candidate_keyframes,
    test_map_points,
    K,
    min_mp_observations=3,
    min_kf_observations=10,
    max_initial_residual=50,
)

if not ba_observations or not optimized_keyframes or not optimized_map_points:
    print("BA граф пуст после cleanup — недостаточно наблюдений для оптимизации.")
    print(f"KFs: {len(optimized_keyframes)}, MPs: {len(optimized_map_points)}, Obs: {len(ba_observations)}")
    exit()

print_ba_graph_diagnostics(
    optimized_map_points,
    optimized_keyframes,
    K,
    max_initial_residual=50,
    title="AFTER cleanup_ba_graph"
)
print("-" * 100)

# =====================================================================
# 4. Печатаем статистику
# =====================================================================

print("-" * 100)

kf_obs_count = {}

for _, kf_id, _ in ba_observations:
    kf_obs_count[kf_id] = kf_obs_count.get(kf_id, 0) + 1

print("Observations per KeyFrame:")

for kf in test_keyframes:

    count = kf_obs_count.get(kf.id, 0)

    print(f"KF {kf.id}: {count}")

print("-" * 100)

print(f"Observations / KeyFrame = {len(ba_observations) / len(optimized_keyframes):.2f}")

print(f"Observations / MapPoint = {len(ba_observations) / len(optimized_map_points):.2f}")

print("-" * 100)

# =====================================================================
# 5. Анализ residual ДО Bundle Adjustment
# =====================================================================

residuals_before = build_residual_vector(
    ba_observations,
    idx_to_mp,
    idx_to_kf,
    K
)

print("\n=== BEFORE BA ===")

print("Mean abs residual:",
      np.mean(np.abs(residuals_before)))

print("Median abs residual:",
      np.median(np.abs(residuals_before)))

print("Max abs residual:",
      np.max(np.abs(residuals_before)))

print("-" * 100)

# =====================================================================
# 6. Сохраняем исходные позы камер
# =====================================================================

kf_before = {}

for kf in test_keyframes:
    kf_before[kf.id] = {
        "R": kf.R.copy(),
        "t": kf.t.copy()
    }

print("-" * 100)

# =====================================================================
# 7. Упаковка параметров Bundle Adjustment
# =====================================================================

packed_keyframe_parameters = pack_keyframe_parameters(
    optimized_keyframes
)

packed_mappoint_parameters = pack_mappoint_parameters(
    optimized_map_points
)

# объединяем всё в один параметр-вектор
packed_parameters = pack_all_parameters(
    packed_keyframe_parameters,
    packed_mappoint_parameters
)

# =====================================================================
# 8. Строим маску разреженного Jacobian
# =====================================================================

jac_sparsity = build_jac_sparsity(
    ba_observations,
    optimized_keyframes,
    optimized_map_points
)

# =====================================================================
# 9. Диагностическая информация
# =====================================================================

print("Bundle Adjustment statistics")

print(f"Observations:       {len(ba_observations)}")
print(f"Optimized KeyFrames:{len(optimized_keyframes)}")
print(f"Optimized MapPoints:{len(optimized_map_points)}")

print(
    f"Optimization parameters: "
    f"{(len(optimized_keyframes)-1) * 6 + len(optimized_map_points) * 3}"
)

print("-" * 100)

thresholds = [5, 10, 20, 50, 100, 500, 1000, 5000]

for t in thresholds:
    count = np.sum(np.abs(residuals_before) > t)
    print(f"Residual > {t:5}: {count}")

print("-" * 100)

# =====================================================================
# 10. Запуск Bundle Adjustment
# =====================================================================

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
    loss='huber',
    f_scale=5.0,
    method='trf',
    jac_sparsity=jac_sparsity,
    max_nfev=500,
    ftol=1e-6,
    xtol=1e-6,
    gtol=1e-6,
    verbose=2
)

# =====================================================================
# 11. Распаковка оптимизированных параметров
# =====================================================================

unpack_all_parameters(
    result.x,
    optimized_keyframes,
    optimized_map_points
)

# =====================================================================
# 12. Анализ residual ПОСЛЕ Bundle Adjustment
# =====================================================================

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

# Анализ reprojection error после Bundle Adjustment
print("-" * 100)

print("\n=== AFTER BA ===")

print("Mean abs residual:",
      np.mean(np.abs(residuals_after)))

print("Median abs residual:",
      np.median(np.abs(residuals_after)))

print("Max abs residual:",
      np.max(np.abs(residuals_after)))

print("-" * 100)

# =====================================================================
# 13. Анализ смещения камер
# =====================================================================

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

# =====================================================================
# 14. Итоговая информация об оптимизации
# =====================================================================

print("-" * 100)

print("Bundle Adjustment result")

print("Success:", result.success)
print("Message:", result.message)
print("Final cost:", result.cost)
print("Optimality:", result.optimality)
print("Function evaluations:", result.nfev)

print("-" * 100)