import cv2 as cv
import numpy as np
import os
import glob

# ==============================================
# USER SETTINGS
# ==============================================
CALIB_FOLDER = r"C:\Users\YJL3090\PycharmProjects\PythonProject2\input\calibration"
SAVE_PATH = r"C:\Users\YJL3090\PycharmProjects\PythonProject2\output\calibration_result.npz"

PATTERN_SIZE = (9, 6)  # inner corners
SQUARE_SIZE = 1.0  # can be 1.0 for SfM

# ==============================================
# Prepare 3D object points
# ==============================================
cols, rows = PATTERN_SIZE
objp = np.zeros((cols * rows, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points in world
imgpoints = []  # 2D points in image

# ==============================================
# Load calibration images
# ==============================================
image_paths = sorted(
    glob.glob(os.path.join(CALIB_FOLDER, "chessboard*.jpg")) +
    glob.glob(os.path.join(CALIB_FOLDER, "*.png"))
)

print(f"[INFO] Found {len(image_paths)} images:")
for p in image_paths:
    print("  -", p)

if len(image_paths) == 0:
    raise Exception("No calibration images found!")

# ==============================================
# Detect chessboard corners
# ==============================================
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for img_path in image_paths:
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    print(f"[INFO] Processing: {img_path}")

    ret, corners = cv.findChessboardCorners(gray, PATTERN_SIZE, None)

    if ret:
        print("   ✔ Corners detected!")

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)

        # Draw detection and save for preview
        cv.drawChessboardCorners(img, PATTERN_SIZE, corners2, ret)
        preview_path = os.path.join(CALIB_FOLDER, "detected_" + os.path.basename(img_path))
        cv.imwrite(preview_path, img)
    else:
        print("   ✘ Corners NOT detected.")

# ==============================================
# Run calibration
# ==============================================
print("\n[INFO] Running cv.calibrateCamera...")

ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n===== CALIBRATION RESULTS =====")
print("RMS error:", ret)
print("\nCamera Matrix (K):\n", K)
print("\nDistortion Coefficients:\n", dist.ravel())
print("=================================\n")

# ==============================================
# Save results
# ==============================================
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
np.savez(SAVE_PATH, intrinsic_mtx=K, dist=dist)

print(f"[INFO] Calibration saved to: {SAVE_PATH}")
