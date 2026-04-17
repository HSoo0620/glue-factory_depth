import numpy as np
from depth_registration.registration import ransac_rigid


def _random_rigid(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    axis = rng.random(3); axis /= np.linalg.norm(axis)
    angle = rng.uniform(-np.pi/12, np.pi/12)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K @ K)
    t = rng.uniform(-20, 20, size=3)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def test_ransac_rigid_recovers_transform():
    rng = np.random.default_rng(42)
    src = rng.uniform(0, 100, size=(200, 3)).astype(np.float32)
    T_gt = _random_rigid(1)
    dst = (T_gt[:3, :3] @ src.T).T + T_gt[:3, 3]
    outlier_idx = rng.choice(200, 60, replace=False)
    dst[outlier_idx] = rng.uniform(-500, 500, size=(60, 3))

    T_est, inlier = ransac_rigid(src, dst, n_iter=1000, inlier_th=1.0, seed=7)
    assert inlier.sum() >= 100
    assert np.allclose(T_est[:3, :3], T_gt[:3, :3], atol=1e-2)
    assert np.allclose(T_est[:3, 3],  T_gt[:3, 3],  atol=0.5)


def test_ransac_rigid_handles_too_few_points():
    src = np.zeros((2, 3), dtype=np.float32)
    dst = np.zeros((2, 3), dtype=np.float32)
    T, inlier = ransac_rigid(src, dst)
    assert np.allclose(T, np.eye(4))
    assert inlier.sum() == 0
