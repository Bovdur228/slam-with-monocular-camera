import cv2 as cv
import numpy as np


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
    min_points=500,
    max_points=3000,
    fallback_recent_points=3000
):
    """
    Выбирает локальные MapPoints для tracking.

    Логика:
        1. Сначала берём MapPoints из последних KeyFrames.
        2. Если их слишком мало, добираем последние MapPoints из global map.
        3. Удаляем дубликаты.
        4. Ограничиваем итоговый список max_points.

    Это защищает от ситуации, когда map_points становится очень большим,
    но в tracking всё равно уходит ограниченное число кандидатов.
    """

    local_mappoints_by_id = {}

    # -------------------------------------------------------------------------
    # 1. MapPoints из последних KeyFrames
    # -------------------------------------------------------------------------
    if len(keyframes) > 0:

        recent_keyframes = keyframes[-num_recent_keyframes:]

        # идём от самых новых KeyFrames к более старым
        for kf in reversed(recent_keyframes):

            for mp in kf.map_points:

                if mp is None:
                    continue

                if mp.id is None:
                    continue

                if mp.descriptor is None:
                    continue

                local_mappoints_by_id[mp.id] = mp

                if len(local_mappoints_by_id) >= max_points:
                    break

            if len(local_mappoints_by_id) >= max_points:
                break

    # -------------------------------------------------------------------------
    # 2. Fallback-добор из последних global map_points
    # -------------------------------------------------------------------------
    if len(local_mappoints_by_id) < min_points:

        fallback_candidates = map_points[-fallback_recent_points:]

        for mp in reversed(fallback_candidates):

            if mp is None:
                continue

            if mp.id is None:
                continue

            if mp.descriptor is None:
                continue

            if mp.id in local_mappoints_by_id:
                continue

            local_mappoints_by_id[mp.id] = mp

            if len(local_mappoints_by_id) >= max_points:
                break

    local_mappoints = list(
        local_mappoints_by_id.values()
    )

    # -------------------------------------------------------------------------
    # 3. Сортировка: более устойчивые и более новые MapPoints выше
    # -------------------------------------------------------------------------
    local_mappoints = sorted(
        local_mappoints,
        key=lambda mp: (
            getattr(mp, "num_observations", 0),
            mp.id if mp.id is not None else -1
        ),
        reverse=True
    )

    # -------------------------------------------------------------------------
    # 4. Жёсткий лимит
    # -------------------------------------------------------------------------
    return local_mappoints[:max_points]

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
