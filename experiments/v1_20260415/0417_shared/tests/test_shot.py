"""SHOT352 descriptor: pybind shot_module 가 있을 때만 실행."""
import numpy as np
import pytest

from depth_registration.descriptors.shot import compute_shot, _lazy_import_shot


def _has_shot_module() -> bool:
    try:
        _lazy_import_shot()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_shot_module(),
                    reason="shot_module binding unavailable")
def test_compute_shot_shape_and_unit_norm():
    """SHOT352: PCL 내부에서 unit-sphere 정규화 → norm≈1."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 100, size=(20000, 3)).astype(np.float32)
    kp  = pts[::40].copy()
    desc = compute_shot(pts, kp)
    assert desc.shape == (len(kp), 352)
    assert desc.dtype == np.float32
    norms = np.linalg.norm(desc, axis=1)
    assert np.all((np.isclose(norms, 1.0, atol=1e-3)) | (norms < 1e-3))
    assert (norms > 0.5).sum() > 0
