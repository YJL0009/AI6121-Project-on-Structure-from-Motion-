import cv2
import numpy as np
import os
from scipy.optimize import least_squares
from tomlkit import boolean
from tqdm import tqdm
import matplotlib.pyplot as plt
import re


class Image_loader():
    def __init__(self, img_dir: str, downscale_factor: float):
        # loading the Camera intrinsic parameters K
        with open(img_dir + '\\K.txt') as f:
            self.K = np.array(
                list((
                    map(
                        lambda x: list(map(lambda x: float(x), x.strip().split(' '))),
                        f.read().split('\n')
                    )
                ))
            )
            self.image_list = []

        # --- collect only image files ---
        all_files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ]

        # --- natural / numeric sort: sfm0, sfm1, sfm2, ..., sfm10, sfm11, ... ---
        def numeric_key(name: str):
            nums = re.findall(r'\d+', name)
            return int(nums[-1]) if nums else -1   # if no number, push to front

        all_files = sorted(all_files, key=numeric_key)

        # build full paths
        for image in all_files:
            self.image_list.append(os.path.join(img_dir, image))

        # debug: see final order
        print("[INFO] Image order:")
        for idx, p in enumerate(self.image_list):
            print(f"  {idx}: {os.path.basename(p)}")

        self.path = os.getcwd()
        self.factor = downscale_factor

        # Only modify K if we actually downscale images
        if self.factor != 1.0:
            self.downscale()

    def downscale(self) -> None:
        """
        Downscales the Image intrinsic parameter according to the downscale factor
        """
        self.K[0, 0] /= self.factor
        self.K[1, 1] /= self.factor
        self.K[0, 2] /= self.factor
        self.K[1, 2] /= self.factor

    def downscale_image(self, image):
        """
        Downscale image consistently with factor, or leave it as is if factor == 1.
        """
        if self.factor == 1.0:
            return image

        for _ in range(1, int(self.factor / 2) + 1):
            image = cv2.pyrDown(image)
        return image


class Sfm():
    def __init__(self, img_dir: str, downscale_factor: float = 2.0) -> None:
        """
        Initialise an Sfm object.
        """
        self.img_obj = Image_loader(img_dir, downscale_factor)

    def triangulation(self, point_2d_1, point_2d_2,
                      projection_matrix_1, projection_matrix_2) -> tuple:
        """
        Triangulates 3D points from 2D vectors and projection matrices.
        Returns:
            projection matrix of first camera (transposed),
            projection matrix of second camera (transposed),
            point cloud in homogeneous coordinates.
        """
        # Note: cv2.triangulatePoints(P1, P2, pts1, pts2)
        pt_cloud = cv2.triangulatePoints(point_2d_1, point_2d_2,
                                         projection_matrix_1.T, projection_matrix_2.T)
        return projection_matrix_1.T, projection_matrix_2.T, (pt_cloud / pt_cloud[3])

    def PnP(self, obj_point, image_point, K, dist_coeff, rot_vector, initial) -> tuple:
        """
        Finds an object pose from 3D-2D point correspondences using RANSAC.
        Returns: rotation matrix, translation vector,
                 filtered image points, filtered object points, rot_vector (extra data).
        """

        # Convert to numpy
        obj_point = np.asarray(obj_point)
        image_point = np.asarray(image_point)

        # ---------- Shape handling ----------
        if initial == 1:
            # First call:
            #   obj_point: (N, 1, 3)  from convertPointsFromHomogeneous
            #   image_point: (2, N)
            #   rot_vector: (2, N)  (actually extra data)
            if obj_point.ndim == 3 and obj_point.shape[1] == 1 and obj_point.shape[2] == 3:
                obj_point = obj_point[:, 0, :]  # -> (N, 3)

            if image_point.ndim == 2 and image_point.shape[0] == 2:
                image_point = image_point.T  # -> (N, 2)

            if isinstance(rot_vector, np.ndarray) and rot_vector.ndim == 2 and rot_vector.shape[0] == 2:
                rot_vector = rot_vector.T  # -> (N, 2)

        else:
            # Later calls:
            #   obj_point: often (N, 1, 3) from previous triangulation
            #   image_point: usually (N, 2), but be tolerant
            if obj_point.ndim == 3 and obj_point.shape[1] == 1 and obj_point.shape[2] == 3:
                obj_point = obj_point[:, 0, :]  # -> (N, 3)

            if image_point.ndim == 3 and image_point.shape[1] == 1 and image_point.shape[2] == 2:
                image_point = image_point[:, 0, :]  # -> (N, 2)

        # Cast to float32
        obj_point = obj_point.astype(np.float32)
        image_point = image_point.astype(np.float32)

        # ---------- Safety check: enough points? ----------
        if obj_point.shape[0] < 4 or image_point.shape[0] < 4:
            # Not enough correspondences; skip PnP but keep pipeline alive
            print(f"[WARN] PnP skipped: only {obj_point.shape[0]} points")
            rot_matrix = np.eye(3, dtype=np.float32)
            tran_vector = np.zeros((3, 1), dtype=np.float32)
            return rot_matrix, tran_vector, image_point, obj_point, rot_vector

        # Reshape for solvePnPRansac: (N,3) -> (N,1,3), (N,2) -> (N,1,2)
        obj_for_pnp = obj_point.reshape(-1, 1, 3)
        img_for_pnp = image_point.reshape(-1, 1, 2)

        # ---------- PnP with RANSAC ----------
        _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(
            obj_for_pnp,
            img_for_pnp,
            K,
            dist_coeff,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        # Convert rotation vector to matrix
        rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)

        # ---------- Filter correspondences by inliers ----------
        if inlier is not None:
            inlier = inlier[:, 0]  # (M,1) -> (M,)
            image_point = image_point[inlier]
            obj_point = obj_point[inlier]

            if isinstance(rot_vector, np.ndarray):
                if rot_vector.shape[0] == len(inlier):
                    rot_vector = rot_vector[inlier]

        return rot_matrix, tran_vector, image_point, obj_point, rot_vector

    def reprojection_error(self, obj_points, image_points,
                           transform_matrix, K, homogenity) -> tuple:
        """
        Calculates the reprojection error i.e. the distance between projected points
        and actual points.
        Returns (mean_error, obj_points).
        """
        # Ensure numpy arrays
        obj_points = np.asarray(obj_points)
        image_points = np.asarray(image_points)

        # ----- Guard: empty / invalid inputs -----
        if obj_points.size == 0 or image_points.size == 0:
            print("[WARN] reprojection_error: empty points, skipping.")
            return 0.0, obj_points

        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)

        # Convert from homogeneous if requested
        if homogenity == 1:
            # Expect shape (4, N) so obj_points.T is (N, 4)
            obj_points = cv2.convertPointsFromHomogeneous(obj_points.T)
            # -> (N, 1, 3)

        # Ensure shape for projectPoints: (N, 1, 3)
        obj_points = np.asarray(obj_points)
        if obj_points.ndim == 3 and obj_points.shape[1] == 1 and obj_points.shape[2] == 3:
            obj_for_proj = obj_points
        elif obj_points.ndim == 2 and obj_points.shape[1] == 3:
            obj_for_proj = obj_points.reshape(-1, 1, 3)
        else:
            obj_for_proj = obj_points.reshape(-1, 3).reshape(-1, 1, 3)

        # Project 3D points
        image_points_calc, _ = cv2.projectPoints(obj_for_proj, rot_vector, tran_vector, K, None)

        if image_points_calc is None:
            print("[WARN] reprojection_error: projectPoints returned None, skipping.")
            return 0.0, obj_points

        image_points_calc = np.float32(image_points_calc[:, 0, :])  # (N, 2)

        # Prepare reference image points
        if homogenity == 1:
            img_ref = np.float32(image_points.T)   # original code used image_points.T
            if img_ref.ndim == 2 and img_ref.shape[0] == 2:
                img_ref = img_ref.T                # ensure (N, 2)
        else:
            img_ref = np.float32(image_points)
            if img_ref.ndim == 3 and img_ref.shape[1] == 1:
                img_ref = img_ref[:, 0, :]         # (N, 2)

        # Align lengths (just in case)
        N = min(len(image_points_calc), len(img_ref))
        if N == 0:
            print("[WARN] reprojection_error: no overlapping points, skipping.")
            return 0.0, obj_points

        image_points_calc = image_points_calc[:N]
        img_ref = img_ref[:N]

        total_error = cv2.norm(image_points_calc, img_ref, cv2.NORM_L2)
        return total_error / N, obj_points

    def optimal_reprojection_error(self, obj_points) -> np.array:
        """
        Calculates the reprojection error during bundle adjustment.
        Returns error vector.
        """
        transform_matrix = obj_points[0:12].reshape((3, 4))
        K = obj_points[12:21].reshape((3, 3))
        rest = int(len(obj_points[21:]) * 0.4)
        p = obj_points[21:21 + rest].reshape((2, int(rest / 2))).T
        obj_points = obj_points[21 + rest:].reshape(
            (int(len(obj_points[21 + rest:]) / 3), 3)
        )
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        image_points, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points = image_points[:, 0, :]
        error = [(p[idx] - image_points[idx]) ** 2 for idx in range(len(p))]
        return np.array(error).ravel() / len(p)

    def bundle_adjustment(self, _3d_point, opt, transform_matrix_new, K, r_error) -> tuple:
        """
        Bundle adjustment for the image and object points.
        Returns object points, image points, transformation matrix.
        """
        opt_variables = np.hstack((transform_matrix_new.ravel(), K.ravel()))
        opt_variables = np.hstack((opt_variables, opt.ravel()))
        opt_variables = np.hstack((opt_variables, _3d_point.ravel()))

        values_corrected = least_squares(
            self.optimal_reprojection_error, opt_variables, gtol=r_error
        ).x
        K = values_corrected[12:21].reshape((3, 3))
        rest = int(len(values_corrected[21:]) * 0.4)
        return (values_corrected[21 + rest:].reshape(
                    (int(len(values_corrected[21 + rest:]) / 3), 3)),
                values_corrected[21:21 + rest].reshape((2, int(rest / 2))).T,
                values_corrected[0:12].reshape((3, 4)))

    def to_ply(self, path, point_cloud, colors) -> None:
        """
        Generates the .ply which can be used to open the point cloud.
        """
        out_points = point_cloud.reshape(-1, 3) * 200
        out_colors = colors.reshape(-1, 3)
        print(out_colors.shape, out_points.shape)
        verts = np.hstack([out_points, out_colors])

        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean
        dist = np.sqrt(scaled_verts[:, 0] ** 2 +
                       scaled_verts[:, 1] ** 2 +
                       scaled_verts[:, 2] ** 2)
        indx = np.where(dist < np.mean(dist) + 300)
        verts = verts[indx]
        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar blue
            property uchar green
            property uchar red
            end_header
            '''
        with open(path + '\\res\\' + self.img_obj.image_list[0].split('\\')[-2] + '.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')

    def common_points(self, image_points_1, image_points_2, image_points_3) -> tuple:
        """
        Finds the common points between image 1 and 2 , image 2 and 3.
        Returns:
            indices of common points in image 1,
            indices of common points in image 2,
            mask of points in image 2,
            mask of points in image 3.
        """
        cm_points_1 = []
        cm_points_2 = []
        for i in range(image_points_1.shape[0]):
            a = np.where(image_points_2 == image_points_1[i, :])
            if a[0].size != 0:
                cm_points_1.append(i)
                cm_points_2.append(a[0][0])

        mask_array_1 = np.ma.array(image_points_2, mask=False)
        mask_array_1.mask[cm_points_2] = True
        mask_array_1 = mask_array_1.compressed()
        mask_array_1 = mask_array_1.reshape(int(mask_array_1.shape[0] / 2), 2)

        mask_array_2 = np.ma.array(image_points_3, mask=False)
        mask_array_2.mask[cm_points_2] = True
        mask_array_2 = mask_array_2.compressed()
        mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)

        print(" Shape New Array", mask_array_1.shape, mask_array_2.shape)

        return (
            np.array(cm_points_1, dtype=np.int32),
            np.array(cm_points_2, dtype=np.int32),
            mask_array_1,
            mask_array_2,
        )

    def find_features(self, image_0, image_1) -> tuple:
        """
        Feature detection using the SIFT algorithm and KNN.
        Returns keypoints (features) in image_0 and image_1.
        """

        # Limit number of features to avoid exploding memory on large images
        sift = cv2.SIFT_create()

        gray0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

        key_points_0, desc_0 = sift.detectAndCompute(gray0, None)
        key_points_1, desc_1 = sift.detectAndCompute(gray1, None)

        # ---- Safety: check if descriptors are valid ----
        if desc_0 is None or desc_1 is None or len(key_points_0) < 2 or len(key_points_1) < 2:
            print(
                f"[WARN] Not enough SIFT keypoints: "
                f"img0={len(key_points_0)}, img1={len(key_points_1)}"
            )
            # Return empty arrays so caller can decide what to do
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        # Use L2 for SIFT (float descriptors)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(desc_0, desc_1, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.70 * n.distance:
                good.append(m)

        if len(good) == 0:
            print("[WARN] No good matches after ratio test.")
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        pts0 = np.float32([key_points_0[m.queryIdx].pt for m in good])
        pts1 = np.float32([key_points_1[m.trainIdx].pt for m in good])

        print(f"[INFO] SIFT matches: kp0={len(key_points_0)}, kp1={len(key_points_1)}, good={len(good)}")
        return pts0, pts1

    def __call__(self, enable_bundle_adjustment: boolean = False):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        pose_array = self.img_obj.K.ravel()
        transform_matrix_0 = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0]])
        transform_matrix_1 = np.empty((3, 4))

        # ---------- Initial two images ----------
        img_path_0 = self.img_obj.image_list[0]
        img_path_1 = self.img_obj.image_list[1]
        print(f"[INIT] Pair: {os.path.basename(img_path_0)}  ->  {os.path.basename(img_path_1)}")

        pose_0 = np.matmul(self.img_obj.K, transform_matrix_0)
        pose_1 = np.empty((3, 4))
        total_points = np.zeros((1, 3))
        total_colors = np.zeros((1, 3))

        image_0 = self.img_obj.downscale_image(cv2.imread(img_path_0))
        image_1 = self.img_obj.downscale_image(cv2.imread(img_path_1))

        feature_0, feature_1 = self.find_features(image_0, image_1)

        # ---- Guard: ensure we have enough initial matches ----
        if feature_0.shape[0] < 8:
            raise RuntimeError(
                f"Not enough matches between {os.path.basename(img_path_0)} "
                f"and {os.path.basename(img_path_1)} (got {feature_0.shape[0]}). "
                f"Try using downscale_factor=2.0 or check image texture/overlap."
            )

        # Essential matrix
        essential_matrix, em_mask = cv2.findEssentialMat(
            feature_0, feature_1, self.img_obj.K,
            method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
        feature_0 = feature_0[em_mask.ravel() == 1]
        feature_1 = feature_1[em_mask.ravel() == 1]

        _, rot_matrix, tran_matrix, em_mask = cv2.recoverPose(
            essential_matrix, feature_0, feature_1, self.img_obj.K)
        feature_0 = feature_0[em_mask.ravel() > 0]
        feature_1 = feature_1[em_mask.ravel() > 0]

        transform_matrix_1[:3, :3] = np.matmul(rot_matrix, transform_matrix_0[:3, :3])
        transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + \
            np.matmul(transform_matrix_0[:3, :3], tran_matrix.ravel())

        pose_1 = np.matmul(self.img_obj.K, transform_matrix_1)

        feature_0, feature_1, points_3d = self.triangulation(
            pose_0, pose_1, feature_0, feature_1)
        error, points_3d = self.reprojection_error(
            points_3d, feature_1, transform_matrix_1, self.img_obj.K, homogenity=1)
        print("REPROJECTION ERROR: ", error)

        _, _, feature_1, points_3d, _ = self.PnP(
            points_3d, feature_1, self.img_obj.K,
            np.zeros((5, 1), dtype=np.float32), feature_0, initial=1)

        total_images = len(self.img_obj.image_list) - 2
        pose_array = np.hstack((np.hstack((pose_array, pose_0.ravel())), pose_1.ravel()))

        threshold = 0.5
        for i in tqdm(range(total_images)):
            prev_idx = i + 1
            new_idx = i + 2
            img_prev = self.img_obj.image_list[prev_idx]
            img_new = self.img_obj.image_list[new_idx]

            print(f"[PAIR {i}] {os.path.basename(img_prev)}  ->  {os.path.basename(img_new)}")

            image_2 = self.img_obj.downscale_image(cv2.imread(img_new))
            features_cur, features_2 = self.find_features(image_1, image_2)

            if i != 0:
                feature_0, feature_1, points_3d = self.triangulation(
                    pose_0, pose_1, feature_0, feature_1)
                feature_1 = feature_1.T
                points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
                points_3d = points_3d[:, 0, :]

            cm_points_0, cm_points_1, cm_mask_0, cm_mask_1 = self.common_points(
                feature_1, features_cur, features_2)
            cm_points_2 = features_2[cm_points_1]
            cm_points_cur = features_cur[cm_points_1]

            rot_matrix, tran_matrix, cm_points_2, points_3d, cm_points_cur = self.PnP(
                points_3d[cm_points_0],
                cm_points_2,
                self.img_obj.K,
                np.zeros((5, 1), dtype=np.float32),
                cm_points_cur,
                initial=0
            )
            transform_matrix_1 = np.hstack((rot_matrix, tran_matrix))
            pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)

            error, points_3d = self.reprojection_error(
                points_3d, cm_points_2, transform_matrix_1, self.img_obj.K,
                homogenity=0)

            cm_mask_0, cm_mask_1, points_3d = self.triangulation(
                pose_1, pose_2, cm_mask_0, cm_mask_1)
            error, points_3d = self.reprojection_error(
                points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K,
                homogenity=1)
            print("Reprojection Error: ", error)
            pose_array = np.hstack((pose_array, pose_2.ravel()))

            if enable_bundle_adjustment:
                points_3d, cm_mask_1, transform_matrix_1 = self.bundle_adjustment(
                    points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K,
                    threshold)
                pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)
                error, points_3d = self.reprojection_error(
                    points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K,
                    homogenity=0)
                print("Bundle Adjusted error: ", error)
                total_points = np.vstack((total_points, points_3d))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left])
                total_colors = np.vstack((total_colors, color_vector))
            else:
                total_points = np.vstack((total_points, points_3d[:, 0, :]))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left.T])
                total_colors = np.vstack((total_colors, color_vector))

            transform_matrix_0 = np.copy(transform_matrix_1)
            pose_0 = np.copy(pose_1)
            pose_1 = np.copy(pose_2)

            plt.scatter(i, error)
            plt.pause(0.05)

            image_0 = np.copy(image_1)
            image_1 = np.copy(image_2)
            feature_0 = np.copy(features_cur)
            feature_1 = np.copy(features_2)

            cv2.imshow(self.img_obj.image_list[0].split('\\')[-2], image_2)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        cv2.destroyAllWindows()

        print("Printing to .ply file")
        print(total_points.shape, total_colors.shape)
        self.to_ply(self.img_obj.path, total_points, total_colors)
        print("Completed Exiting ...")
        np.savetxt(self.img_obj.path + '\\res\\' +
                   self.img_obj.image_list[0].split('\\')[-2] + '_pose_array.csv',
                   pose_array, delimiter='\n')


if __name__ == '__main__':
    # Use factor=2.0 for safer behavior, or 1.0 for no downscale (full-res)
    sfm = Sfm(r"C:\Users\YJL3090\PycharmProjects\yunnangarden_obelisk1", downscale_factor=2)
    sfm()
