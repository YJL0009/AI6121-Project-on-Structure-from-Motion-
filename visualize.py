import os
import numpy as np
import open3d as o3d   # pip install open3d


# ======================================================
# USER PATHS – EDIT THESE
# ======================================================
PLY_PATH  = r"C:\Users\YJL3090\PycharmProjects\res\yunnangarden_obelisk1.ply"
POSE_CSV  = r"C:\Users\YJL3090\PycharmProjects\res\yunnangarden_obelisk1_pose_array.csv"
# (change to the actual pose csv you saved)
# ======================================================


def load_point_cloud(ply_path: str) -> o3d.geometry.PointCloud:
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    print("[INFO] Loaded point cloud:", pcd)
    return pcd


def load_poses_with_K(csv_path: str):
    """
    Your SfM code did:
        pose_array = [K.ravel(), P0.ravel(), P1.ravel(), ...]
        np.savetxt(..., pose_array, delimiter='\\n')
    So CSV is a 1D list of numbers: 9 for K + 12 per camera.

    Returns:
        K: 3x3 intrinsic matrix
        P_list: list of 3x4 projection matrices (as np.array)
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Pose CSV not found: {csv_path}")

    vals = np.loadtxt(csv_path)          # 1D
    vals = np.asarray(vals, dtype=float).flatten()
    n = vals.size
    print(f"[INFO] Loaded pose array with {n} values")

    if (n - 9) % 12 != 0:
        # fallback: maybe no K is stored (rare)
        if n % 12 != 0:
            raise RuntimeError(
                f"Pose array length {n} is not 9 + 12*N or 12*N. "
                f"Check the saving code."
            )
        K = None
        P_list = vals.reshape(-1, 12)
        return K, P_list

    # normal case: first 9 are K
    K = vals[:9].reshape(3, 3)
    rest = vals[9:]
    num_cams = rest.size // 12
    P_list = rest.reshape(num_cams, 12)

    print("[INFO] Parsed K from CSV:")
    print(K)
    print(f"[INFO] Number of cameras: {num_cams}")
    return K, P_list


def extrinsics_from_P(K, P):
    """
    Given K (3x3) and P (3x4 = K[R|t]), compute R, t and camera center C.
    """
    if K is not None:
        K_inv = np.linalg.inv(K)
        Rt = K_inv @ P
    else:
        Rt = P

    R = Rt[:, :3]
    t = Rt[:, 3:4]

    # Orthonormalize R (just in case)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    C = -R.T @ t     # camera center in world coords (3x1)
    return R, t, C[:, 0]


def align_cameras_to_cloud(pcd, C_list):
    """
    Cameras are in world coords, but the PLY may be scaled/centered.
    We compute a global scale + translation so cameras roughly match the PLY.
    """
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise RuntimeError("Empty point cloud.")

    center_ply = pts.mean(axis=0)
    extent_ply = pts.max(axis=0) - pts.min(axis=0)
    max_extent_ply = float(extent_ply.max())

    C_arr = np.vstack(C_list)
    center_cam = C_arr.mean(axis=0)
    extent_cam = C_arr.max(axis=0) - C_arr.min(axis=0)
    max_extent_cam = float(extent_cam.max())

    if max_extent_cam < 1e-6:
        s = 1.0
    else:
        s = max_extent_ply / max_extent_cam

    t = center_ply - s * center_cam

    print("\n[INFO] Camera alignment:")
    print(f"  scale s      = {s:.4f}")
    print(f"  translation  = {t}")

    C_disp_list = [s * C + t for C in C_arr]
    return C_disp_list, s, t


def create_flat_camera(R, C_disp, scale=0.2):
    """
    Make a flat rectangular 'camera' at center C_disp, oriented by R.
    """
    w = scale
    h = scale * 0.75

    # rectangle in camera coordinates (z=+1 forward)
    rect_cam = np.array([
        [-w, -h,  1.0],
        [ w, -h,  1.0],
        [ w,  h,  1.0],
        [-w,  h,  1.0]
    ])

    # transform to world: X_world = R^T * X_cam + C
    rect_world = (R.T @ rect_cam.T).T + C_disp

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0]
    ]
    colors = [[1, 0, 0] for _ in lines]   # red outline

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(rect_world)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def main():
    # 1) load cloud & poses
    pcd = load_point_cloud(PLY_PATH)
    K, P_list = load_poses_with_K(POSE_CSV)

    # 2) compute extrinsics + camera centers
    R_list, C_list = [], []
    for P_flat in P_list:
        P = P_flat.reshape(3, 4)
        R, t, C = extrinsics_from_P(K, P)
        R_list.append(R)
        C_list.append(C)

    # 3) align cameras to cloud space
    C_disp_list, s, t = align_cameras_to_cloud(pcd, C_list)

    # 4) create flat camera rectangles
    pts = np.asarray(pcd.points)
    max_extent = float((pts.max(axis=0) - pts.min(axis=0)).max())
    cam_scale = 0.05 * max_extent  # 5% of scene size

    cam_geoms = []
    for R, C_disp in zip(R_list, C_disp_list):
        cam_geoms.append(create_flat_camera(R, C_disp, scale=cam_scale))

    # 5) show
    print("[INFO] Visualizing point cloud + flat cameras ...")
    o3d.visualization.draw_geometries(
        [pcd] + cam_geoms,
        window_name="SfM Reconstruction (flat cameras)",
        width=1600,
        height=900
    )


if __name__ == "__main__":
    main()