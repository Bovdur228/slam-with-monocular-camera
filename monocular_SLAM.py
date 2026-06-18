import cv2 as cv
import numpy as np
import os


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

# Класс, описывающий строение MapPoint
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


# Класс, описывающий строение KeyFrame ------------------------------------------------------------------------------------------------------------
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

        self.map_points = []


# mwmwofwpfmp ------------------------------------------------------
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


# функции ==========================================================================================================================================

def find_existing_mappoint(
    descriptor,
    map_points,
    threshold=30
):
    """
    ищет существующую Map Point, проходясь по дескрипторам всех Map Points и сравнивая их с дескриптором "кандидата".
    если нашли, значит такая Map Point уже существует и мы возвращаем её.
    если не нашли, то значит такой Map Point на карте нет и мы возвращаем None.
    """

    best_mp = None
    best_distance = float("inf")

    for mp in map_points[-500:]: # !!!!! временное ограничение поиска для ускорения, позже заменить на пространственный индекс !!!!!

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

    # сохраняем в текущий keyframe все точки, которые он увидел в кадре
    for mp, valid_kp_idx, in current_frame_map_points:
                    
        obs = Observation(
            kf,
            valid_kp_idx
        )

        mp.observations.append(obs)

        kf.map_points.append(mp)

    keyframes.append(kf)

#------------------------------------------------------------------------------------------------------------------------------------------------
def project_point(
    point_world,
    global_R,
    global_t,
    K
):
    """
    Переводит мировую точку Pw в пиксельные координаты изображения.
    """

    Pc = global_R.T @ (point_world.reshape(3,1) - global_t)

    if Pc[2, 0] <= 0:
        return None

    pixel = K @ Pc

    u = pixel[0,0] / pixel[2,0]
    v = pixel[1,0] / pixel[2,0]

    return np.array([u,v])

# ------------------------------------------------------------------------------------------------------------------------------------------------
def reprojection_error(
    point_world,
    observed_uv,
    global_R,
    global_t,
    K
):
    """
    Считает reprojection error для одной точки.
    """
    
    predicted_uv = project_point(
        point_world,
        global_R,
        global_t,
        K
    )

    if predicted_uv is None:
        return float("inf")

    error = np.linalg.norm(
        predicted_uv - observed_uv
    )

    return error

# захват видео -----------------------------------------------------------------------------------------------------------------------------------
cap = cv.VideoCapture(0)

# создание ORB и BFMatcher для поиска feature matching -------------------------------------------------------------------------------------------
orb = cv.ORB_create(nfeatures=1000)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# сюда будет записываться предыдущий кадр, а также его keypoints и descriptors -------------------------------------------------------------------
prev_gray = None
prev_kp = None
prev_des = None

# матрица K и distortion, взятые после калибровки камеры из файла mono_calibration.npz ------------------------------------------------------------
calib_data = np.load(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "mono_calibration.npz")
)
K = calib_data["K"]
dist = calib_data["dist"]
# RMS Error: 0.13271354901396754

# тут будет обновляться глобальная pose камеры ----------------------------------------------------------------------------------------------------
global_R = np.eye(3)
global_t = np.zeros((3, 1))


# 2D визуализация перемещения камеры в глобальном мире + место начала перемещения (cv.circle)------------------------------------------------------
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

# сюда записываются мировые координаты map_points (список из объектов класса MapPoint) ------------------------------------------------------------
map_points = []

# вспомогательная переменная для отрисовки Map Points на 2D карте, чтобы не рисовать каждый раз дубликаты -----------------------------------------
last_drawn_map_point_idx = 0


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

# обновляем глобальное перемещение камеры в мире -------------------------------------------------------------------------------------------------
            global_t = global_t + global_R @ t
            global_R = global_R @ R.T

# # строим две Projection Matrix для текущей пары кадров (prev_frame и current_frame) ------------------------------------------------------------
            P1 = K @ np.hstack(
                (
                    np.eye(3),
                    np.zeros((3, 1))
                )
            )

            P2 = K @ np.hstack(
                (
                    R,
                    t
                )
            )

# с помощью triangulation вычисляем Map Points (Pw) ---------------------------------------------------------------------------------------------

            # поначалу вычисляем !!!!!!!!!!!!!!!!!
            points_4d = cv.triangulatePoints(
                P1,
                P2,
                pts1.T,
                pts2.T
            )

            # далее получаем точки, ОТНОСИТЕЛЬНО prev_frame, НО ПОКА ЧТО ОТНОСИТЕЛЬНО КАМЕРЫ (Pc)
            points_3d_prev_frame = (
                points_4d[:3] / points_4d[3]
            ).T

            # не забываем фильтровать плохие Map Points (все точки что np.isnan(), np.isinf(), а также где z < 0 и z > 1000)
            valid_mask = (
                np.isfinite(points_3d_prev_frame).all(axis=1)
                &
                (points_3d_prev_frame[:, 2] > 0)
                &
                (points_3d_prev_frame[:, 2] < 100)
            )

            points_3d_prev_frame = points_3d_prev_frame[valid_mask]

            # переводим точки Pc в систему current_frame
            points_3d_current = (R @ points_3d_prev_frame.T + t).T

            # не забываем удалять и дескрипторы отфильтрованных плохих Map Points
            valid_descriptors = valid_descriptors[valid_mask]

            # не забываем удалять и индексы отфильтрованных плохих Map Points
            valid_kp_idxes = valid_kp_idxes[valid_mask]

            # переводим Pc в Pw, используя формулу: Pw = global_R @ Pc + global_t (Но для массива точек она немного другая, как видно ниже) !УПРОЩЕНИЕ!
            points_world = (global_R @ points_3d_current.T + global_t).T

# nwinwerlfwnrpf ----------------------------------------------------------------------------------------------------------------------------------
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

            # список, в который сохраняем все Map Points, которые мы увидели в кадре
            current_frame_map_points = []

            # счётчик найденных новых Map Points, увеличиваем его каждый раз, если находим новую Map Point
            new_points_count = 0

            # проходимся в каждой найденной в кадре Map Point и её дескриптору
            for obs in frame_observations:

                existing_mp = find_existing_mappoint(
                    obs.descriptor,
                    map_points,
                    threshold=20
                )

                # если такая Map Point в мире уже существует, то не создаём новую Map Point
                if existing_mp is not None:
                    
                    # ВРЕМЕННЫЙ КОСТЫЛЬ, ВМЕСТО BUNDLE ADJUSTMENT! После каждого нового наблюдения точки, проводим статистическое усреднение её позиции
                    existing_mp.position = (existing_mp.position * existing_mp.num_observations + obs.point_world) / (
                                                    existing_mp.num_observations + 1
                                                )
                    
                    existing_mp.num_observations += 1

                    # ТОЖЕ ВРЕМЕННЫЙ КОСТЫЛЬ. ПОТОМ ИСПРАВИТЬ НА ОДНО ИЗ ЭТОГО: descriptor voting, медианный descriptor, лучший descriptor
                    existing_mp.descriptor = obs.descriptor

                    current_frame_map_points.append((existing_mp, obs.kp_idx))

                # а если не существует, то создаём новый объект класса MapPoint и добавляем её в map_points[]
                else:

                    new_points_count += 1

                    mp = MapPoint(
                        obs.point_world,
                        obs.descriptor
                    )

                    map_points.append(mp)

                    current_frame_map_points.append((mp, obs.kp_idx))
            
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
                    keyframes
                )

                last_keyFrame_t = global_t.copy()
                last_keyFrame_R = global_R.copy()

                print(f"Keyframe saved: {len(keyframes)}")
                print(len(map_points))

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
                        keyframes
                    )

                    last_keyFrame_t = global_t.copy()
                    last_keyFrame_R = global_R.copy()

                    print(f"Keyframe saved: {len(keyframes)}")
                    print(len(map_points))

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
            1,
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
            print(len(map_points))

        break

if len(map_points) > 0:

    mp = map_points[0]

    print("Observations count:", len(mp.observations))

    for obs in mp.observations:

        print(
            "KF:",
            id(obs.keyframe),
            "kp_idx:",
            obs.keypoint_idx
        )

errors = []

for mp in map_points:

    for obs in mp.observations:

        kf = obs.keyframe

        kp = kf.kp[obs.keypoint_idx]

        error = reprojection_error(
            mp.position,
            np.array(kp.pt),
            kf.R,
            kf.t,
            K
        )

        errors.append(error)

if len(errors) > 0:

    print(
        "Mean reprojection error:",
        np.mean(errors)
    )

    print(
        "Median reprojection error:",
        np.median(errors)
    )

    print(
        "Max reprojection error:",
        np.max(errors)
    )

cap.release()
cv.destroyAllWindows()