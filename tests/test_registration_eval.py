"""Registration RMSE eval 핵심 수학 함수 단위 테스트."""
import unittest
import numpy as np


CLIP_START = 0.1
CLIP_END = 1000.0
GRID_DX = 0.05
GRID_DY = 0.05
GRID_DZ = 0.02


def pixel_to_grid3d(keypoints_2d, depth_map_raw, clip_start=CLIP_START,
                    clip_end=CLIP_END, grid_dx=GRID_DX, grid_dy=GRID_DY,
                    grid_dz=GRID_DZ):
    N = keypoints_2d.shape[0]
    points_3d = np.zeros((N, 3), dtype=np.float64)
    valid_mask = np.zeros(N, dtype=bool)
    H, W = depth_map_raw.shape
    for i in range(N):
        u = int(np.clip(keypoints_2d[i, 0], 0, W - 1))
        v = int(np.clip(keypoints_2d[i, 1], 0, H - 1))
        d_raw = float(depth_map_raw[v, u])
        if d_raw <= 0:
            continue
        depth_real = clip_start + (d_raw / 65535.0) * (clip_end - clip_start)
        points_3d[i] = [u * grid_dx, v * grid_dy, depth_real]
        valid_mask[i] = True
    return points_3d, valid_mask


def rigid_transform_svd(P_src, P_dst):
    c_src = P_src.mean(axis=0)
    c_dst = P_dst.mean(axis=0)
    H = (P_src - c_src).T @ (P_dst - c_dst)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c_dst - R @ c_src
    return R, t


def compute_transform_rmse(P_src, R_est, t_est, R_gt, t_gt):
    P_est = (R_est @ P_src.T).T + t_est
    P_gt = (R_gt @ P_src.T).T + t_gt
    return float(np.sqrt(np.mean(np.sum((P_est - P_gt) ** 2, axis=1))))


class TestPixelToGrid3D(unittest.TestCase):
    def test_basic_conversion(self):
        depth_raw = np.zeros((100, 100), dtype=np.uint16)
        depth_raw[50, 30] = 32768
        kp = np.array([[30.0, 50.0]])
        pts, mask = pixel_to_grid3d(kp, depth_raw)
        self.assertTrue(mask[0])
        self.assertAlmostEqual(pts[0, 0], 30 * 0.05, places=5)
        self.assertAlmostEqual(pts[0, 1], 50 * 0.05, places=5)
        depth_real = 0.1 + (32768 / 65535.0) * 999.9
        self.assertAlmostEqual(pts[0, 2], depth_real, places=5)

    def test_zero_depth_invalid(self):
        depth_raw = np.zeros((100, 100), dtype=np.uint16)
        kp = np.array([[10.0, 10.0]])
        pts, mask = pixel_to_grid3d(kp, depth_raw)
        self.assertFalse(mask[0])
        np.testing.assert_array_equal(pts[0], [0, 0, 0])

    def test_multiple_points(self):
        depth_raw = np.full((100, 100), 65535, dtype=np.uint16)
        kp = np.array([[0.0, 0.0], [99.0, 99.0]])
        pts, mask = pixel_to_grid3d(kp, depth_raw)
        self.assertTrue(mask.all())
        depth_real_max = 0.1 + 1.0 * 999.9
        self.assertAlmostEqual(pts[1, 0], 99 * 0.05, places=5)
        self.assertAlmostEqual(pts[1, 2], depth_real_max, places=5)


class TestRigidTransformSVD(unittest.TestCase):
    def test_identity(self):
        pts = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3], [1, 1, 1]],
                       dtype=np.float64)
        R, t = rigid_transform_svd(pts, pts)
        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=10)
        np.testing.assert_array_almost_equal(t, np.zeros(3), decimal=10)

    def test_known_rotation_translation(self):
        np.random.seed(42)
        pts_src = np.random.randn(50, 3) * 10
        angle = np.pi / 6
        R_true = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])
        t_true = np.array([5.0, -3.0, 2.0])
        pts_dst = (R_true @ pts_src.T).T + t_true
        R_est, t_est = rigid_transform_svd(pts_src, pts_dst)
        np.testing.assert_array_almost_equal(R_est, R_true, decimal=10)
        np.testing.assert_array_almost_equal(t_est, t_true, decimal=10)

    def test_reflection_handling(self):
        pts_src = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                           dtype=np.float64)
        R, t = rigid_transform_svd(pts_src, pts_src)
        self.assertGreater(np.linalg.det(R), 0)


class TestComputeTransformRMSE(unittest.TestCase):
    def test_identical_transforms(self):
        pts = np.random.randn(100, 3)
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        rmse = compute_transform_rmse(pts, R, t, R, t)
        self.assertAlmostEqual(rmse, 0.0, places=10)

    def test_known_error(self):
        pts = np.array([[0, 0, 0]], dtype=np.float64)
        R = np.eye(3)
        t_est = np.array([1.0, 0.0, 0.0])
        t_gt = np.array([0.0, 0.0, 0.0])
        rmse = compute_transform_rmse(pts, R, t_est, R, t_gt)
        self.assertAlmostEqual(rmse, 1.0, places=10)

    def test_multiple_points_uniform_error(self):
        pts = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float64)
        R = np.eye(3)
        t_est = np.array([1.0, 0.0, 0.0])
        t_gt = np.array([0.0, 0.0, 0.0])
        rmse = compute_transform_rmse(pts, R, t_est, R, t_gt)
        self.assertAlmostEqual(rmse, 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
