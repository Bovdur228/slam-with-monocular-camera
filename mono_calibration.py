import cv2 as cv
import numpy as np
import glob
import os

# ==========================================
# Параметры шахматной доски
# ==========================================

CHESSBOARD_SIZE = (9, 6)

# Размер клетки в условных единицах
SQUARE_SIZE = 25.0

# ==========================================
# Пути
# ==========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    SCRIPT_DIR,
    "calibration",
    "mono_dataset"
)

# ==========================================
# Подготовка объектных точек
# ==========================================

objp = np.zeros(
    (CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHESSBOARD_SIZE[0],
    0:CHESSBOARD_SIZE[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

# ==========================================
# Контейнеры
# ==========================================

objpoints = []
imgpoints = []

# ==========================================
# Загрузка изображений
# ==========================================

images = glob.glob(
    os.path.join(DATASET_DIR, "*.jpg")
)

print(f"Found images: {len(images)}")

# ==========================================
# Поиск шахматной доски
# ==========================================

image_size = None

for image_path in images:

    img = cv.imread(image_path)

    gray = cv.cvtColor(
        img,
        cv.COLOR_BGR2GRAY
    )

    image_size = gray.shape[::-1]

    found, corners = cv.findChessboardCornersSB(
        gray,
        CHESSBOARD_SIZE
    )

    print(
        os.path.basename(image_path),
        "FOUND" if found else "NOT FOUND"
    )

    if found:

        criteria = (
            cv.TERM_CRITERIA_EPS +
            cv.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        corners = cv.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        objpoints.append(objp)
        imgpoints.append(corners)

        cv.drawChessboardCorners(
            img,
            CHESSBOARD_SIZE,
            corners,
            found
        )

        cv.imshow(
            "Detected Corners",
            img
        )

        cv.waitKey(1500)

cv.destroyAllWindows()

print(f"Valid images: {len(objpoints)}")

# ==========================================
# Калибровка
# ==========================================

ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)

print("\n==============================")
print("Calibration finished")
print("==============================")

print(f"\nRMS Error: {ret}")

print("\nCamera Matrix K:")
print(K)

print("\nDistortion Coefficients:")
print(dist)

# ==========================================
# Сохранение
# ==========================================

save_path = os.path.join(
    SCRIPT_DIR,
    "mono_calibration.npz"
)

np.savez(
    save_path,
    K=K,
    dist=dist,
    rvecs=rvecs,
    tvecs=tvecs
)

print(f"\nSaved to: {save_path}")