import numpy as np

from depth_registration.iss import detect_iss_mm
from depth_registration import params as P


def _make_box_pcd(n_per_face: int = 30) -> np.ndarray:
    rng = np.random.default_rng(0)
    face = rng.uniform(0, 100, size=(n_per_face * n_per_face, 2)).astype(np.float32)
    z0 = np.zeros((len(face), 1), dtype=np.float32)
    z1 = np.full((len(face), 1), 40.0, dtype=np.float32)
    bottom = np.concatenate([face, z0], axis=1)
    top    = np.concatenate([face, z1], axis=1)
    return np.concatenate([bottom, top], axis=0).astype(np.float32)


def test_detect_iss_mm_basic():
    pts = _make_box_pcd()
    kps = detect_iss_mm(pts)
    assert kps.ndim == 2 and kps.shape[1] == 3
    assert kps.dtype == np.float32
    assert len(kps) <= P.MAX_KEYPOINTS


def test_detect_iss_mm_empty_input():
    kps = detect_iss_mm(np.zeros((0, 3), dtype=np.float32))
    assert kps.shape == (0, 3)
