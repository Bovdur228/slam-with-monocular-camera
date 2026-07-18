import cv2 as cv
import numpy as np
import os

from scipy.optimize import least_squares

import slam.config as cfg

from slam.map import MapPoint, FrameObservation

from slam.tracking import (
    find_existing_mappoint,
    get_local_mappoints,
    track_existing_mappoints,
)

from slam.pose import estimate_pose_pnp

from slam.triangulation import (
    triangulate_points_world_from_poses,
)

from slam.keyframes import create_KeyFrame

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

from slam.visualisation import (
    create_trajectory_image,
    create_map_image,
    draw_camera_trajectory,
    draw_new_mappoints,
    draw_feature_matching_view,
    show_slam_windows,
)

from slam.ba_runner import run_ba_test


# =====================================================================================================================================
def cull_mappoints_by_observations(
    map_points,
    min_observations=2
):
    good_points = []

    for mp in map_points:

        if len(mp.observations) >= min_observations:
            good_points.append(mp)

    return good_points


# =============================================================================================================================================

# захват видео --------------------------------------------------------------------------------------------------------------------------------
cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# создание ORB и BFMatcher для поиска feature matching ----------------------------------------------------------------------------------------
orb = cv.ORB_create(nfeatures=cfg.ORB_FEATURES)
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
traj_img = create_trajectory_image(
    size=800,
    origin=(400, 400)
)

# 2D карта, где будут показываться все Map Points мира --------------------------------------------------------------------------------------------
map_img = create_map_image(
    size=800
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
                threshold=cfg.ESSENTIAL_RANSAC_THRESHOLD
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
                num_recent_keyframes=cfg.LOCAL_MAP_RECENT_KEYFRAMES,
                fallback_max_points=cfg.LOCAL_MAP_FALLBACK_POINTS
            )
            
            tracked_map_points, tracking_stats = track_existing_mappoints(
                kp,
                des,
                local_map_points,
                global_R,
                global_t,
                K,
                gray.shape,
                search_radius=cfg.TRACKING_SEARCH_RADIUS,
                descriptor_threshold=cfg.TRACKING_DESCRIPTOR_THRESHOLD,
                max_points=None
            )

            pnp_success, pnp_Rwc, pnp_twc, pnp_inlier_tracked_points, pnp_stats = estimate_pose_pnp(
                tracked_map_points,
                kp,
                K,
                min_points=cfg.PNP_MIN_POINTS,
                min_inliers=cfg.PNP_MIN_INLIERS,
                reprojection_error_threshold=cfg.PNP_REPROJECTION_ERROR,
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
                max_depth=cfg.MAX_TRIANGULATION_DEPTH
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
                if (
                    translation > cfg.KEYFRAME_TRANSLATION_THRESHOLD 
                    or rotation > cfg.KEYFRAME_ROTATION_THRESHOLD 
                    or (
                        new_points_ratio >= cfg.KEYFRAME_NEW_POINTS_RATIO 
                        and len(map_points) > cfg.KEYFRAME_MIN_MAP_POINTS_FOR_NEW_POINTS_RATIO
                        )
                    ):
                
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
            draw_camera_trajectory(
                traj_img,
                global_t,
                origin=(400, 400)
            )

            # тут визуализация map_points (Pw) в мире
            last_drawn_map_point_idx = draw_new_mappoints(
                                            map_img,
                                            map_points,
                                            last_drawn_map_point_idx,
                                            origin=(400, 400)
                                        )
        

# склеиваем предыдущий кадр (слева) и текущий кадр (справа) и проводим линии между их matches --------------------------------------------------

        # в верхнем левом углу выводим текстом количество matches
        draw_img = draw_feature_matching_view(
                        prev_gray,
                        prev_kp,
                        gray,
                        kp,
                        matches,
                        pose_source,
                        local_map_points,
                        tracking_stats,
                        max_matches_to_draw=50
                    )

        show_slam_windows(
            draw_img,
            traj_img,
            map_img
        )

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

run_ba_test(
    keyframes,
    map_points,
    K,
    pnp_success_count=pnp_success_count,
    recoverpose_count=recoverpose_count,
    max_test_keyframes=30
)