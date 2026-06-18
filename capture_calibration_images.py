import cv2 as cv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SAVE_DIR = os.path.join(
    SCRIPT_DIR,
    "mono_dataset"
)

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv.VideoCapture(0)

img_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv.imshow("Calibration Capture", frame)

    key = cv.waitKey(1) & 0xFF

    # SPACE
    if key == 32:

        filename = os.path.join(
            SAVE_DIR,
            f"img_{img_count:03d}.jpg"
        )

        cv.imwrite(filename, frame)

        img_count += 1

        print(f"Saved {img_count} photo: {filename}")

    # ESC
    elif key == 27:
        break

cap.release()
cv.destroyAllWindows()