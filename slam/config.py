ORB_FEATURES = 1000

ESSENTIAL_RANSAC_THRESHOLD = 1.0 # похож на 38 строку (возможно в будущем убрать)

# =========================================================================================================================================
# PnP
# =========================================================================================================================================

PNP_MIN_POINTS = 15
PNP_MIN_INLIERS = 12
PNP_REPROJECTION_ERROR = 12.0

LOCAL_MAP_RECENT_KEYFRAMES = 10
LOCAL_MAP_FALLBACK_POINTS = 3000

# Strict PnP acceptance ---------------------------------------------------------------------------------------------------------------

PNP_ACCEPT_MIN_INLIERS = 20
PNP_ACCEPT_MIN_INLIER_RATIO = 0.60

PNP_MAX_TRANSLATION_JUMP = 50.0
PNP_MAX_ROTATION_JUMP = 1.5

# PnP retry ------------------------------------------------------------------------------------------------------------------------------

PNP_RETRY_ENABLED = True

TRACKING_RETRY_SEARCH_RADIUS = 60
TRACKING_RETRY_DESCRIPTOR_THRESHOLD = 60

PNP_RETRY_MIN_POINTS = 20
PNP_RETRY_MIN_INLIERS = 15
PNP_RETRY_REPROJECTION_ERROR = 14.0
PNP_RETRY_CONFIDENCE = 0.99
PNP_RETRY_ITERATIONS_COUNT = 150

# PnP retry acceptance -------------------------------------------------------------------------------------------------------------------

PNP_RETRY_ACCEPT_MIN_INLIERS = 16
PNP_RETRY_ACCEPT_MIN_INLIER_RATIO = 0.6

PNP_RETRY_MAX_TRANSLATION_JUMP = 30.0
PNP_RETRY_MAX_ROTATION_JUMP = 1.2

# Previous-pose PnP retry -------------------------------------------------------------------------------------------------------------

PNP_PREV_POSE_RETRY_ENABLED = True

TRACKING_PREV_POSE_RETRY_SEARCH_RADIUS = TRACKING_RETRY_SEARCH_RADIUS
TRACKING_PREV_POSE_RETRY_DESCRIPTOR_THRESHOLD = TRACKING_RETRY_DESCRIPTOR_THRESHOLD

PNP_PREV_POSE_RETRY_MIN_POINTS = PNP_RETRY_MIN_POINTS
PNP_PREV_POSE_RETRY_MIN_INLIERS = PNP_RETRY_MIN_INLIERS
PNP_PREV_POSE_RETRY_REPROJECTION_ERROR = PNP_RETRY_REPROJECTION_ERROR
PNP_PREV_POSE_RETRY_CONFIDENCE = PNP_RETRY_CONFIDENCE
PNP_PREV_POSE_RETRY_ITERATIONS_COUNT = PNP_RETRY_ITERATIONS_COUNT

PNP_PREV_POSE_RETRY_ACCEPT_MIN_INLIERS = PNP_RETRY_ACCEPT_MIN_INLIERS
PNP_PREV_POSE_RETRY_ACCEPT_MIN_INLIER_RATIO = PNP_RETRY_ACCEPT_MIN_INLIER_RATIO

PNP_PREV_POSE_RETRY_MAX_TRANSLATION_JUMP = PNP_RETRY_MAX_TRANSLATION_JUMP
PNP_PREV_POSE_RETRY_MAX_ROTATION_JUMP = PNP_RETRY_MAX_ROTATION_JUMP

# ========================================================================================================================================
#
# =========================================================================================================================================
TRACKING_SEARCH_RADIUS = 40
TRACKING_DESCRIPTOR_THRESHOLD = 50

MAX_TRIANGULATION_DEPTH = 100

KEYFRAME_TRANSLATION_THRESHOLD = 10
KEYFRAME_ROTATION_THRESHOLD = 0.6
KEYFRAME_NEW_POINTS_RATIO = 0.5
KEYFRAME_MIN_MAP_POINTS_FOR_NEW_POINTS_RATIO = 200

# =========================================================================================================================================
# MAPPOINT CULLING
# =========================================================================================================================================

CULLING_ENABLED = True
CULLING_EVERY_KEYFRAMES = 5

CULLING_MIN_OBSERVATIONS = 2
CULLING_MIN_UNIQUE_KEYFRAMES = 2
CULLING_MAX_REPROJECTION_ERROR = 20.0

CULLING_MIN_MAP_POINTS = 1000
CULLING_MIN_AGE_KEYFRAMES = 3

# =========================================================================================================================================
# KEYFRAME
# =========================================================================================================================================

# KeyFrame-based triangulation ------------------------------------------------------------------------------------------------------------

KEYFRAME_TRIANGULATION_MATCH_DISTANCE = 50
KEYFRAME_TRIANGULATION_MAX_REPROJECTION_ERROR = 10.0
KEYFRAME_TRIANGULATION_MAX_NEW_POINTS = 500

KEYFRAME_TRIANGULATION_RANSAC_THRESHOLD = 1.0
KEYFRAME_TRIANGULATION_MIN_RANSAC_INLIERS = 20

# KeyFrame selection -----------------------------------------------------------------------------------------------------------------------

KEYFRAME_MIN_FRAMES_BETWEEN = 15

KEYFRAME_MIN_TRACKED_POINTS = 60
KEYFRAME_MIN_PNP_INLIER_RATIO = 0.55

KEYFRAME_WEAK_TRACKING_MIN_TRANSLATION = 2.0
KEYFRAME_WEAK_TRACKING_MIN_ROTATION = 0.15

KEYFRAME_MIN_MAP_POINTS_FOR_TRACKING_CHECK = 300

# Local map selection ------------------------------------------------------------------------------------------------------------------

LOCAL_MAP_MIN_POINTS = 500
LOCAL_MAP_MAX_POINTS = 3000
LOCAL_MAP_FALLBACK_RECENT_POINTS = 3000

# =========================================================================================================================================
# LOCAL BUNDLE ADJUSTMENT
# =========================================================================================================================================

LOCAL_BA_ENABLED = True

# Запускать Local BA только когда KeyFrames уже достаточно.
LOCAL_BA_MIN_KEYFRAMES = 4

# Размер локального окна.
# Первый KeyFrame внутри окна будет fixed/anchor.
LOCAL_BA_WINDOW_SIZE = 7

# Запускать не на каждом KeyFrame, а через N KeyFrames.
# Для первого теста 2 — более безопасно по скорости.
LOCAL_BA_EVERY_KEYFRAMES = 2

# Сколько MapPoints максимум отдавать в Local BA.
LOCAL_BA_MAX_MAP_POINTS = 800

# Фильтрация BA-графа.
LOCAL_BA_MIN_MP_OBSERVATIONS = 2
LOCAL_BA_MIN_KF_OBSERVATIONS = 10
LOCAL_BA_MAX_INITIAL_RESIDUAL = 50.0
LOCAL_BA_MAX_CLEANUP_ITERATIONS = 10

# Ограничение оптимизации, чтобы runtime не зависал надолго.
LOCAL_BA_MAX_NFEV = 20

# Robust loss.
LOCAL_BA_HUBER_F_SCALE = 5.0

# Safety-check результата.
LOCAL_BA_MAX_MEAN_RESIDUAL_INCREASE = 1.05
LOCAL_BA_MAX_CAMERA_SHIFT = 5.0

# Если True, scipy least_squares будет печатать подробный optimization log.
LOCAL_BA_VERBOSE = False