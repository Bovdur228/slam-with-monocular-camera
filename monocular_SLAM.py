import cv2 as cv
import numpy as np
import os

import slam.config as cfg

from slam.tracking import (
    get_local_mappoints,
    track_existing_mappoints,
)

from slam.pose import (
    estimate_pose_pnp,
    check_pnp_pose_safety,
)

from slam.keyframes import (
    create_KeyFrame,
    compute_keyframe_motion,
    should_create_keyframe,
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

from slam.culling import cull_mappoints

from slam.keyframe_triangulation import triangulate_new_mappoints_between_keyframes

# ==============================================================================================================================================

def increment_counter(counter, reason):
    if reason is None:
        reason = "unknown"

    counter[reason] = counter.get(reason, 0) + 1

# ---------------------------------------------------------------------------------------------------------------------------------------------
def print_counter(title, counter):
    print(title)

    if not counter:
        print("  none")
        return

    for reason, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count}")

# ------------------------------------------------------------------------------------------------------------------------------------------
def create_tracking_diag():
    return {
        "candidates": [],
        "projected": [],
        "inside": [],
        "tracked": [],
        "pnp_points": [],
        "pnp_inliers": [],
        "pnp_inlier_ratio": [],
        "translation_jump": [],
        "rotation_jump": []
    }

# -------------------------------------------------------------------------------------------------------------------------------------------
def add_tracking_diag(diag, tracking_stats, pnp_stats):
    diag["candidates"].append(
        tracking_stats.get("candidates", 0)
    )

    diag["projected"].append(
        tracking_stats.get("projected", 0)
    )

    diag["inside"].append(
        tracking_stats.get("inside", 0)
    )

    diag["tracked"].append(
        tracking_stats.get("tracked", 0)
    )

    diag["pnp_points"].append(
        pnp_stats.get("points", 0)
    )

    diag["pnp_inliers"].append(
        pnp_stats.get("inliers", 0)
    )

    diag["pnp_inlier_ratio"].append(
        pnp_stats.get("inlier_ratio", 0.0)
    )

    if "translation_jump" in tracking_stats:
        diag["translation_jump"].append(
            tracking_stats["translation_jump"]
        )

    if "rotation_jump" in tracking_stats:
        diag["rotation_jump"].append(
            tracking_stats["rotation_jump"]
        )

# --------------------------------------------------------------------------------------------------------------------------------------------
def print_numeric_summary(name, values):
    if len(values) == 0:
        print(f"  {name}: none")
        return

    arr = np.asarray(values, dtype=np.float64)

    print(
        f"  {name}: "
        f"min={arr.min():.2f}, "
        f"mean={arr.mean():.2f}, "
        f"median={np.median(arr):.2f}, "
        f"max={arr.max():.2f}"
    )

# --------------------------------------------------------------------------------------------------------------------------------------------
def print_tracking_diag(title, diag):
    print(title)

    if len(diag["tracked"]) == 0:
        print("  none")
        return

    print_numeric_summary("candidates", diag["candidates"])
    print_numeric_summary("projected", diag["projected"])
    print_numeric_summary("inside", diag["inside"])
    print_numeric_summary("tracked", diag["tracked"])
    print_numeric_summary("pnp_points", diag["pnp_points"])
    print_numeric_summary("pnp_inliers", diag["pnp_inliers"])
    print_numeric_summary("pnp_inlier_ratio", diag["pnp_inlier_ratio"])

    if len(diag["translation_jump"]) > 0:
        print_numeric_summary(
            "translation_jump",
            diag["translation_jump"]
        )

    if len(diag["rotation_jump"]) > 0:
        print_numeric_summary(
            "rotation_jump",
            diag["rotation_jump"]
        )

# ========================================================================================================================================

# захват видео ----------------------------------------------------------------------------------------------------------------------
cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# создание ORB и BFMatcher для поиска feature matching ------------------------------------------------------------------------------
orb = cv.ORB_create(nfeatures=cfg.ORB_FEATURES)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# сюда будет записываться предыдущий кадр, а также его keypoints и descriptors -----------------------------------------------------
prev_gray = None
prev_kp = None
prev_des = None

# матрица K и distortion, взятые после калибровки камеры из файла mono_calibration.npz ----------------------------------------------
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

# тут будет обновляться глобальная pose камеры ---------------------------------------------------------------------------------------
global_R = np.eye(3)
global_t = np.zeros((3, 1))


# 2D визуализация перемещения камеры в глобальном мире + место начала перемещения (cv.circle)-----------------------------------------
traj_img = create_trajectory_image(
    size=800,
    origin=(400, 400)
)

# 2D карта, где будут показываться все Map Points мира -------------------------------------------------------------------------------
map_img = create_map_image(
    size=800
)

# тут хранятится Pose последнего KeyFrame ---------------------------------------------------------------------------------------------
last_keyFrame_t = None
last_keyFrame_R = None

# список всех созданных KeyFrames ------------------------------------------------------------------------------------------------------
keyframes = []
keyframe_id = 0

# список всех созданных Map Points -----------------------------------------------------------------------------------------------------
map_points = []
map_point_id = 0

# вспомогательная переменная для отрисовки Map Points на 2D карте, чтобы не рисовать каждый раз дубликаты -----------------------------
last_drawn_map_point_idx = 0

# счётчик PnP кадров и recoverPose кадров ----------------------------------------------------------------------------------------------
pnp_success_count = 0              # общий счётчик PnP кадров
pnp_strict_success_count = 0       # счёткий strict PnP кадров
pnp_strict_rejected_count = 0      # счётчик отклонённых strict PnP кадров
recoverpose_count = 0              # счётчик recoverpose кадров
pnp_retry_success_count = 0        # счётчик "relaxed" PnP кадров
pnp_retry_rejected_count = 0       # счётчик отклонённых "relaxed" PnP 
pnp_prev_pose_retry_success_count = 0
pnp_prev_pose_retry_rejected_count = 0

pnp_retry_attempt_count = 0
pnp_retry_raw_success_count = 0

pnp_prev_pose_retry_attempt_count = 0
pnp_prev_pose_retry_raw_success_count = 0

pnp_strict_failure_reasons = {}
pnp_strict_reject_reasons = {}

pnp_retry_failure_reasons = {}
pnp_retry_reject_reasons = {}

pnp_prev_pose_retry_failure_reasons = {}
pnp_prev_pose_retry_reject_reasons = {}

# tracking diagnostics ----------------------------------------------------------------------------------------------------------------
strict_pnp_success_diag = create_tracking_diag()
strict_pnp_failure_diag = create_tracking_diag()
strict_pnp_rejected_diag = create_tracking_diag()

retry_pnp_success_diag = create_tracking_diag()
retry_pnp_failure_diag = create_tracking_diag()
retry_pnp_rejected_diag = create_tracking_diag()

prev_pose_retry_success_diag = create_tracking_diag()
prev_pose_retry_failure_diag = create_tracking_diag()
prev_pose_retry_rejected_diag = create_tracking_diag()


# счётчик кадров после последнего созданного Keyframe ---------------------------------------------------------------------------------
frames_since_last_keyframe = 0


# НАЧАЛО ЗАПИСИ =========================================================================================================================
while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # исправляем искажение камеры
    gray = cv.undistort(gray, K, dist)

# feature matching между предыдущим кадром и нынешним, а затем запись valid features в pts1 и pts2 ---------------------------------------
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

# если между двумя кадрами нашли 8 или более matches, то вычисляем по ним Essential matrix (E) -------------------------------------------------
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

# далее с помощью матрицы E, pts1, pts2 и матрицы K вычисляем локальное смещение (R,t) камеры между двумя кадрами -------------------------------
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

# обновляем глобальное перемещение камеры в мире ------------------------------------------------------------------------------------------------
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
                min_points=cfg.LOCAL_MAP_MIN_POINTS,
                max_points=cfg.LOCAL_MAP_MAX_POINTS,
                fallback_recent_points=cfg.LOCAL_MAP_FALLBACK_RECENT_POINTS
            )
            
            pnp_mode = None

            # -------------------------------------------------------------------------
            # Strict tracking + strict PnP
            # -------------------------------------------------------------------------
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

            strict_pnp_accepted = False

            # -------------------------------------------------------------------------
            # Strict PnP safety check
            # -------------------------------------------------------------------------
            if pnp_success:

                strict_pose_is_safe, strict_reject_reason, strict_translation_jump, strict_rotation_jump = check_pnp_pose_safety(
                    pnp_Rwc,
                    pnp_twc,
                    prev_Rwc_for_triangulation,
                    prev_twc_for_triangulation,
                    pnp_stats,
                    min_inliers=cfg.PNP_ACCEPT_MIN_INLIERS,
                    min_inlier_ratio=cfg.PNP_ACCEPT_MIN_INLIER_RATIO,
                    max_translation_jump=cfg.PNP_MAX_TRANSLATION_JUMP,
                    max_rotation_jump=cfg.PNP_MAX_ROTATION_JUMP
                )

                tracking_stats["strict_translation_jump"] = strict_translation_jump
                tracking_stats["strict_rotation_jump"] = strict_rotation_jump
                tracking_stats["strict_pose_is_safe"] = strict_pose_is_safe

                tracking_stats["translation_jump"] = strict_translation_jump
                tracking_stats["rotation_jump"] = strict_rotation_jump

                if strict_pose_is_safe:

                    add_tracking_diag(
                        strict_pnp_success_diag,
                        tracking_stats,
                        pnp_stats
                    )

                    global_R = pnp_Rwc.copy()
                    global_t = pnp_twc.copy()

                    tracked_map_points = pnp_inlier_tracked_points

                    pose_source = "PnP"
                    pnp_mode = "strict"
                    strict_pnp_accepted = True

                else:

                    pnp_strict_rejected_count += 1

                    increment_counter(
                        pnp_strict_reject_reasons,
                        strict_reject_reason
                    )

                    add_tracking_diag(
                        strict_pnp_rejected_diag,
                        tracking_stats,
                        pnp_stats
                    )

                    pnp_stats = pnp_stats.copy()
                    pnp_stats["reason"] = strict_reject_reason

                    tracked_map_points = []
                    pnp_mode = None

            else:

                increment_counter(
                    pnp_strict_failure_reasons,
                    pnp_stats.get("reason")
                )

                add_tracking_diag(
                    strict_pnp_failure_diag,
                    tracking_stats,
                    pnp_stats
                )

            pnp_accepted = strict_pnp_accepted

            # -------------------------------------------------------------------------
            # Previous-pose tracking + previous-pose retry PnP
            # -------------------------------------------------------------------------
            if not pnp_accepted and cfg.PNP_PREV_POSE_RETRY_ENABLED:

                pnp_prev_pose_retry_attempt_count += 1

                prev_pose_retry_tracked_map_points, prev_pose_retry_tracking_stats = track_existing_mappoints(
                    kp,
                    des,
                    local_map_points,
                    prev_Rwc_for_triangulation,
                    prev_twc_for_triangulation,
                    K,
                    gray.shape,
                    search_radius=cfg.TRACKING_PREV_POSE_RETRY_SEARCH_RADIUS,
                    descriptor_threshold=cfg.TRACKING_PREV_POSE_RETRY_DESCRIPTOR_THRESHOLD,
                    max_points=None
                )

                prev_pose_retry_pnp_success, prev_pose_retry_pnp_Rwc, prev_pose_retry_pnp_twc, prev_pose_retry_pnp_inlier_tracked_points, prev_pose_retry_pnp_stats = estimate_pose_pnp(
                    prev_pose_retry_tracked_map_points,
                    kp,
                    K,
                    min_points=cfg.PNP_PREV_POSE_RETRY_MIN_POINTS,
                    min_inliers=cfg.PNP_PREV_POSE_RETRY_MIN_INLIERS,
                    reprojection_error_threshold=cfg.PNP_PREV_POSE_RETRY_REPROJECTION_ERROR,
                    confidence=cfg.PNP_PREV_POSE_RETRY_CONFIDENCE,
                    iterations_count=cfg.PNP_PREV_POSE_RETRY_ITERATIONS_COUNT
                )

                if prev_pose_retry_pnp_success:

                    pnp_prev_pose_retry_raw_success_count += 1

                    prev_pose_retry_is_safe, prev_pose_retry_reject_reason, prev_pose_retry_translation_jump, prev_pose_retry_rotation_jump = check_pnp_pose_safety(
                        prev_pose_retry_pnp_Rwc,
                        prev_pose_retry_pnp_twc,
                        prev_Rwc_for_triangulation,
                        prev_twc_for_triangulation,
                        prev_pose_retry_pnp_stats,
                        min_inliers=cfg.PNP_PREV_POSE_RETRY_ACCEPT_MIN_INLIERS,
                        min_inlier_ratio=cfg.PNP_PREV_POSE_RETRY_ACCEPT_MIN_INLIER_RATIO,
                        max_translation_jump=cfg.PNP_PREV_POSE_RETRY_MAX_TRANSLATION_JUMP,
                        max_rotation_jump=cfg.PNP_PREV_POSE_RETRY_MAX_ROTATION_JUMP
                    )

                    prev_pose_retry_tracking_stats["prev_pose_retry_translation_jump"] = prev_pose_retry_translation_jump
                    prev_pose_retry_tracking_stats["prev_pose_retry_rotation_jump"] = prev_pose_retry_rotation_jump
                    prev_pose_retry_tracking_stats["prev_pose_retry_pose_is_safe"] = prev_pose_retry_is_safe

                    prev_pose_retry_tracking_stats["translation_jump"] = prev_pose_retry_translation_jump
                    prev_pose_retry_tracking_stats["rotation_jump"] = prev_pose_retry_rotation_jump

                    add_tracking_diag(
                        prev_pose_retry_success_diag,
                        prev_pose_retry_tracking_stats,
                        prev_pose_retry_pnp_stats
                    )

                    if prev_pose_retry_is_safe:

                        global_R = prev_pose_retry_pnp_Rwc.copy()
                        global_t = prev_pose_retry_pnp_twc.copy()

                        tracked_map_points = prev_pose_retry_pnp_inlier_tracked_points

                        tracking_stats = prev_pose_retry_tracking_stats
                        pnp_stats = prev_pose_retry_pnp_stats

                        pose_source = "PnP"
                        pnp_mode = "prev_pose_retry"
                        pnp_accepted = True

                    else:

                        pnp_prev_pose_retry_rejected_count += 1

                        increment_counter(
                            pnp_prev_pose_retry_reject_reasons,
                            prev_pose_retry_reject_reason
                        )

                        prev_pose_retry_tracking_stats["prev_pose_retry_reject_reason"] = prev_pose_retry_reject_reason

                        add_tracking_diag(
                            prev_pose_retry_rejected_diag,
                            prev_pose_retry_tracking_stats,
                            prev_pose_retry_pnp_stats
                        )

                        tracking_stats = prev_pose_retry_tracking_stats
                        pnp_stats = prev_pose_retry_pnp_stats.copy()
                        pnp_stats["reason"] = prev_pose_retry_reject_reason

                        tracked_map_points = []
                        pnp_mode = None

                else:

                    increment_counter(
                        pnp_prev_pose_retry_failure_reasons,
                        prev_pose_retry_pnp_stats.get("reason")
                    )

                    add_tracking_diag(
                        prev_pose_retry_failure_diag,
                        prev_pose_retry_tracking_stats,
                        prev_pose_retry_pnp_stats
                    )

                    tracking_stats = prev_pose_retry_tracking_stats
                    pnp_stats = prev_pose_retry_pnp_stats

                    tracked_map_points = []
                    pnp_mode = None

            # -------------------------------------------------------------------------
            # Relaxed tracking + retry PnP
            # -------------------------------------------------------------------------
            if not pnp_accepted:

                if cfg.PNP_RETRY_ENABLED:

                    pnp_retry_attempt_count += 1

                    retry_tracked_map_points, retry_tracking_stats = track_existing_mappoints(
                        kp,
                        des,
                        local_map_points,
                        global_R,
                        global_t,
                        K,
                        gray.shape,
                        search_radius=cfg.TRACKING_RETRY_SEARCH_RADIUS,
                        descriptor_threshold=cfg.TRACKING_RETRY_DESCRIPTOR_THRESHOLD,
                        max_points=None
                    )

                    retry_pnp_success, retry_pnp_Rwc, retry_pnp_twc, retry_pnp_inlier_tracked_points, retry_pnp_stats = estimate_pose_pnp(
                        retry_tracked_map_points,
                        kp,
                        K,
                        min_points=cfg.PNP_RETRY_MIN_POINTS,
                        min_inliers=cfg.PNP_RETRY_MIN_INLIERS,
                        reprojection_error_threshold=cfg.PNP_RETRY_REPROJECTION_ERROR,
                        confidence=cfg.PNP_RETRY_CONFIDENCE,
                        iterations_count=cfg.PNP_RETRY_ITERATIONS_COUNT
                    )

                    if retry_pnp_success:

                        pnp_retry_raw_success_count += 1
                    
                        retry_pose_is_safe, retry_reject_reason, retry_translation_jump, retry_rotation_jump = check_pnp_pose_safety(
                            retry_pnp_Rwc,
                            retry_pnp_twc,
                            prev_Rwc_for_triangulation,
                            prev_twc_for_triangulation,
                            retry_pnp_stats,
                            min_inliers=cfg.PNP_RETRY_ACCEPT_MIN_INLIERS,
                            min_inlier_ratio=cfg.PNP_RETRY_ACCEPT_MIN_INLIER_RATIO,
                            max_translation_jump=cfg.PNP_RETRY_MAX_TRANSLATION_JUMP,
                            max_rotation_jump=cfg.PNP_RETRY_MAX_ROTATION_JUMP
                        )
                    
                        retry_tracking_stats["retry_translation_jump"] = retry_translation_jump
                        retry_tracking_stats["retry_rotation_jump"] = retry_rotation_jump
                        retry_tracking_stats["retry_pose_is_safe"] = retry_pose_is_safe
                    
                        retry_tracking_stats["translation_jump"] = retry_translation_jump
                        retry_tracking_stats["rotation_jump"] = retry_rotation_jump
                    
                        add_tracking_diag(
                            retry_pnp_success_diag,
                            retry_tracking_stats,
                            retry_pnp_stats
                        )

                        if retry_pose_is_safe:

                            global_R = retry_pnp_Rwc.copy()
                            global_t = retry_pnp_twc.copy()

                            tracked_map_points = retry_pnp_inlier_tracked_points

                            tracking_stats = retry_tracking_stats
                            pnp_stats = retry_pnp_stats

                            pose_source = "PnP"
                            pnp_mode = "retry"
                            pnp_accepted = True

                        else:

                            pnp_retry_rejected_count += 1

                            increment_counter(
                                pnp_retry_reject_reasons,
                                retry_reject_reason
                            )

                            retry_tracking_stats["retry_reject_reason"] = retry_reject_reason

                            add_tracking_diag(
                                retry_pnp_rejected_diag,
                                retry_tracking_stats,
                                retry_pnp_stats
                            )

                            tracking_stats = retry_tracking_stats
                            pnp_stats = retry_pnp_stats.copy()
                            pnp_stats["reason"] = retry_reject_reason

                            tracked_map_points = []
                            pnp_mode = None

                    else:

                        increment_counter(
                            pnp_retry_failure_reasons,
                            retry_pnp_stats.get("reason")
                        )

                        add_tracking_diag(
                            retry_pnp_failure_diag,
                            retry_tracking_stats,
                            retry_pnp_stats
                        )

                        tracking_stats = retry_tracking_stats
                        pnp_stats = retry_pnp_stats

                        tracked_map_points = []
                        pnp_mode = None

                else:

                    tracked_map_points = []
                    pnp_mode = None


            tracking_stats["tracked_before_pnp"] = tracking_stats["tracked"]
            tracking_stats["tracked_used"] = len(tracked_map_points)

            tracking_stats["pnp_points"] = pnp_stats["points"]
            tracking_stats["pnp_inliers"] = pnp_stats["inliers"]
            tracking_stats["pnp_inlier_ratio"] = pnp_stats["inlier_ratio"]
            tracking_stats["pnp_mode"] = pnp_mode
            tracking_stats["pnp_reason"] = pnp_stats.get("reason")

            if pose_source == "PnP":
            
                pnp_success_count += 1

                if pnp_mode == "strict":
                    pnp_strict_success_count += 1

                elif pnp_mode == "prev_pose_retry":
                    pnp_prev_pose_retry_success_count += 1

                elif pnp_mode == "retry":
                    pnp_retry_success_count += 1

            else:
            
                recoverpose_count += 1

            # список tracked MapPoints, которые попали в этот кадр
            current_frame_map_points = list(tracked_map_points)

            # на каждом новом кадре увеличиваем счётчик, чтобы считать сколько кадров прошло с момента создания последнего Keyframe
            frames_since_last_keyframe += 1

            # переменная, которая показывает создался ли на этом кадре Keyframe или нет.
            # нужно в дальнейшем для culling MapPoints
            created_keyframe = False

# критерии создания и само создание KeyFrames, сохраняем его данные (глобальные R и t камеры, keypoints и descriptors) ------------------------

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
                frames_since_last_keyframe = 0

                #print(f"Keyframe saved: {len(keyframes)}")
                #print(len(map_points))

            # если хотя бы один KeyFrame есть, то смотрим, нужно ли создать новый KeyFrame
            elif last_keyFrame_t is not None and last_keyFrame_R is not None:

                # на каждом кадре считаем насколько далеко камера переместилась и повернулась, относительно последнего KeyFrame
                translation, rotation = compute_keyframe_motion(
                    global_R,
                    global_t,
                    last_keyFrame_R,
                    last_keyFrame_t
                )

                # проверяем критерии создания нового Keyframe
                create_keyframe, keyframe_reason = should_create_keyframe(
                    translation=translation,
                    rotation=rotation,
                    tracking_stats=tracking_stats,
                    pose_source=pose_source,
                    map_points_count=len(map_points),
                    frames_since_last_keyframe=frames_since_last_keyframe,
                    translation_threshold=cfg.KEYFRAME_TRANSLATION_THRESHOLD,
                    rotation_threshold=cfg.KEYFRAME_ROTATION_THRESHOLD,
                    min_frames_between=cfg.KEYFRAME_MIN_FRAMES_BETWEEN,
                    min_tracked_points=cfg.KEYFRAME_MIN_TRACKED_POINTS,
                    min_pnp_inlier_ratio=cfg.KEYFRAME_MIN_PNP_INLIER_RATIO,
                    min_map_points_for_tracking_check=cfg.KEYFRAME_MIN_MAP_POINTS_FOR_TRACKING_CHECK,
                    weak_tracking_min_translation=cfg.KEYFRAME_WEAK_TRACKING_MIN_TRANSLATION,
                    weak_tracking_min_rotation=cfg.KEYFRAME_WEAK_TRACKING_MIN_ROTATION
                )

                # если критерии подходят, создаём новый Keyframe
                if create_keyframe:

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
                    frames_since_last_keyframe = 0

                    print(
                        f"Keyframe saved: {len(keyframes)} "
                        f"reason={keyframe_reason} "
                        f"pose_source={pose_source} "
                        f"pnp_mode={tracking_stats.get('pnp_mode')} "
                        f"pnp_reason={tracking_stats.get('pnp_reason')} "
                        f"pnp_inliers={tracking_stats.get('pnp_inliers')} "
                        f"pnp_ratio={tracking_stats.get('pnp_inlier_ratio'):.2f} "
                        f"translation={translation:.3f} "
                        f"rotation={rotation:.3f}"
                    )
                    print("-" * 100)

                    print(f"KeyFrame triangulation: {triangulation_stats}")
                    print("=" * 100)
            
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
print("=" * 100)
print(f"PnP frames: {pnp_success_count}")
print(f"PnP strict frames: {pnp_strict_success_count}")
print(f"PnP strict rejected frames: {pnp_strict_rejected_count}")
print(f"PnP prev-pose retry frames: {pnp_prev_pose_retry_success_count}")
print(f"PnP prev-pose retry attempts: {pnp_prev_pose_retry_attempt_count}")
print(f"PnP prev-pose retry raw success frames: {pnp_prev_pose_retry_raw_success_count}")
print(f"PnP prev-pose retry rejected frames: {pnp_prev_pose_retry_rejected_count}")
print(f"PnP retry frames: {pnp_retry_success_count}")
print(f"PnP retry attempts: {pnp_retry_attempt_count}")
print(f"PnP retry raw success frames: {pnp_retry_raw_success_count}")
print(f"PnP retry rejected frames: {pnp_retry_rejected_count}")
print(f"recoverPose frames: {recoverpose_count}")
print("-" * 100)

print_counter("Strict PnP failure reasons:", pnp_strict_failure_reasons)
print("-" * 100)

print_counter("Strict PnP reject reasons:", pnp_strict_reject_reasons)
print("-" * 100)

print_counter("Prev-pose retry PnP failure reasons:", pnp_prev_pose_retry_failure_reasons)
print("-" * 100)

print_counter("Prev-pose retry PnP reject reasons:", pnp_prev_pose_retry_reject_reasons)
print("-" * 100)

print_counter("Retry PnP failure reasons:", pnp_retry_failure_reasons)
print("-" * 100)

print_counter("Retry PnP reject reasons:", pnp_retry_reject_reasons)
print("-" * 100)

# -------------------------------------------------------------------------------
print_tracking_diag(
    "Strict PnP success tracking diagnostics:",
    strict_pnp_success_diag
)
print("-" * 100)

print_tracking_diag(
    "Strict PnP failure tracking diagnostics:",
    strict_pnp_failure_diag
)
print("-" * 100)

print_tracking_diag(
    "Strict PnP rejected tracking diagnostics:",
    strict_pnp_rejected_diag
)
print("-" * 100)

print_tracking_diag(
    "Prev-pose retry raw success tracking diagnostics:",
    prev_pose_retry_success_diag
)
print("-" * 100)

print_tracking_diag(
    "Prev-pose retry failure tracking diagnostics:",
    prev_pose_retry_failure_diag
)
print("-" * 100)

print_tracking_diag(
    "Prev-pose retry rejected tracking diagnostics:",
    prev_pose_retry_rejected_diag
)
print("-" * 100)

print_tracking_diag(
    "Retry PnP raw success tracking diagnostics:",
    retry_pnp_success_diag
)
print("-" * 100)

print_tracking_diag(
    "Retry PnP failure tracking diagnostics:",
    retry_pnp_failure_diag
)
print("-" * 100)

print_tracking_diag(
    "Retry PnP rejected tracking diagnostics:",
    retry_pnp_rejected_diag
)
print("-" * 100)

# ------------------------------------------------------------------
run_ba_test(
    keyframes,
    map_points,
    K,
    pnp_success_count=pnp_success_count,
    recoverpose_count=recoverpose_count,
    max_test_keyframes=30
)
