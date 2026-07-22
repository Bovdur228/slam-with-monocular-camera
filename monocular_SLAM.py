import cv2 as cv
import numpy as np
import os

import slam.config as cfg

from slam.tracking import (
    get_local_mappoints,
    track_existing_mappoints,
)

from slam.pose import estimate_pose_pnp

from slam.keyframes import create_KeyFrame

from slam.visualisation import (
    create_trajectory_image,
    create_map_image,
    draw_camera_trajectory,
    draw_new_mappoints,
    draw_feature_matching_view,
    show_slam_windows,
)

from slam.ba_runner import run_ba_test

from slam.culling import cull_mappoints

from slam.keyframe_triangulation import triangulate_new_mappoints_between_keyframes


# ========================================================================================================================================

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

        for match in matches:

            pts1.append(
                prev_kp[match.queryIdx].pt
            )

            pts2.append(
                kp[match.trainIdx].pt
            )

        pts1 = np.array(
            pts1,
            dtype=np.float32
        )

        pts2 = np.array(
            pts2,
            dtype=np.float32
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

            # ляляляляляляляляляляляляля
            created_keyframe = False

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

                created_keyframe = True

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
                    ):

                    reference_kf = keyframes[-1]
                
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

                    current_kf = keyframes[-1]

                    map_point_id, triangulation_stats = triangulate_new_mappoints_between_keyframes(
                                                            reference_kf,
                                                            current_kf,
                                                            map_points,
                                                            map_point_id,
                                                            K,
                                                            bf,
                                                            ransac_threshold=cfg.KEYFRAME_TRIANGULATION_RANSAC_THRESHOLD,
                                                            min_ransac_inliers=cfg.KEYFRAME_TRIANGULATION_MIN_RANSAC_INLIERS,
                                                            max_depth=cfg.MAX_TRIANGULATION_DEPTH,
                                                            max_descriptor_distance=cfg.KEYFRAME_TRIANGULATION_MATCH_DISTANCE,
                                                            max_reprojection_error=cfg.KEYFRAME_TRIANGULATION_MAX_REPROJECTION_ERROR,
                                                            max_new_points=cfg.KEYFRAME_TRIANGULATION_MAX_NEW_POINTS
                                                        )

                    last_keyFrame_t = global_t.copy()
                    last_keyFrame_R = global_R.copy()

                    created_keyframe = True

                    print(f"Keyframe saved: {len(keyframes)}")
                    print(f"KeyFrame triangulation: {triangulation_stats}")
                    #print(len(map_points))
            
# Culling MapPoints ---------------------------------------------------------------------------------------------------------------------
            if (
                cfg.CULLING_ENABLED
                and created_keyframe
                and len(keyframes) % cfg.CULLING_EVERY_KEYFRAMES == 0
                and len(map_points) >= cfg.CULLING_MIN_MAP_POINTS
            ):
                current_keyframe_id = keyframes[-1].id

                map_points, culling_stats = cull_mappoints(
                    map_points,
                    keyframes,
                    K,
                    current_keyframe_id=current_keyframe_id,
                    min_observations=cfg.CULLING_MIN_OBSERVATIONS,
                    min_unique_keyframes=cfg.CULLING_MIN_UNIQUE_KEYFRAMES,
                    max_mean_reprojection_error=cfg.CULLING_MAX_REPROJECTION_ERROR,
                    min_age_keyframes=cfg.CULLING_MIN_AGE_KEYFRAMES
                )

                print(
                    f"MapPoint culling: "
                    f"{culling_stats['before']} -> {culling_stats['after']} "
                    f"removed={culling_stats['removed']}"
                )

                print(
                    f"Culling removal reasons: {culling_stats['removal_reasons']}"
                )
                
                print(
                    f"Culling kept reasons: {culling_stats['kept_reasons']}"
                )

                # После удаления MapPoints нужно перерисовать карту, потому что старое map_img всё ещё содержит уже удалённые точки
                map_img = create_map_image(
                    size=800
                )

                last_drawn_map_point_idx = draw_new_mappoints(
                    map_img,
                    map_points,
                    0,
                    origin=(400, 400)
                )

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