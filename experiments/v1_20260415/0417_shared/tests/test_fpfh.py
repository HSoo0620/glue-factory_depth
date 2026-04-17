import numpy as np
from depth_registration.descriptors.fpfh import compute_fpfh


def test_compute_fpfh_shape_l2_normalized():
    """v1 `_norm_0417` 훈련 분포: radius-only 이웃 + L2 정규화."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 100, size=(5000, 3)).astype(np.float32)
    kp  = pts[::50]
    desc = compute_fpfh(pts, kp)
    assert desc.shape == (len(kp), 33)
    assert desc.dtype == np.float32
    # L2-normalized: 유효한 descriptor 의 norm 은 1.0, 빈 descriptor 는 0.
    norms = np.linalg.norm(desc, axis=1)
    assert np.all((np.isclose(norms, 1.0, atol=1e-3)) | (norms < 1e-3))
    assert (norms > 0.5).sum() > 0, "no valid descriptors produced"
