import numpy as np
import open3d as o3d
import pytest

from gluefactory.datasets.new_dataset import constants as C
from gluefactory.datasets.new_dataset.iss_detection import (
    build_iss_pcd_uvd_scaled,
    detect_iss_keypoints,
    select_keypoints,
)

DATA_ROOT = C.DEFAULT_DATA_ROOT
HAS_DATA = DATA_ROOT.exists()


def test_build_iss_pcd_excludes_zero_depth():
    zmap = np.zeros((20, 20), dtype=np.uint16)
    zmap[5:15, 5:15] = 1000
    pcd, mask, depth_scale, depth_min = build_iss_pcd_uvd_scaled(
        zmap, erode_boundary=0
    )
    assert len(pcd.points) == 100  # 10x10 interior
    assert mask.sum() == 100
    assert depth_min >= 0


def test_select_keypoints_exceeds_max():
    iss = np.random.RandomState(0).rand(1000, 3) * 100.0
    zmap = np.ones((128, 128), dtype=np.uint16) * 500
    kp_resized, scores, n_valid, kp_uv_orig = select_keypoints(
        iss, zmap, max_num_keypoints=512, erode_mask=None,
        resize_factor=0.5, rng=np.random.default_rng(0)
    )
    assert kp_resized.shape == (512, 2)
    assert scores.shape == (512,)
    assert kp_uv_orig.shape == (512, 2)
    assert n_valid == 512
    assert (scores == 1.0).sum() == 512
    # resize scaling
    np.testing.assert_allclose(kp_resized, kp_uv_orig * 0.5, atol=1e-6)


def test_select_keypoints_below_max_random_fill():
    iss = np.random.RandomState(0).rand(100, 3) * 50.0
    # iss u,v must be inside zmap bounds and zmap > 0 for fill
    iss[:, 0] = np.clip(iss[:, 0], 0, 127)
    iss[:, 1] = np.clip(iss[:, 1], 0, 127)
    zmap = np.ones((128, 128), dtype=np.uint16) * 500
    kp_resized, scores, n_valid, kp_uv_orig = select_keypoints(
        iss, zmap, max_num_keypoints=512, erode_mask=None,
        resize_factor=0.5, rng=np.random.default_rng(0)
    )
    assert kp_resized.shape == (512, 2)
    assert n_valid == 512
    assert (scores == 1.0).sum() == 100
    assert (scores == 0.5).sum() == 412
    assert (scores == 0.0).sum() == 0


def test_select_keypoints_pad_when_mask_too_small():
    iss = np.random.RandomState(0).rand(10, 3) * 10.0
    zmap = np.zeros((128, 128), dtype=np.uint16)
    zmap[:5, :5] = 500  # only 25 fill candidates
    kp_resized, scores, n_valid, kp_uv_orig = select_keypoints(
        iss, zmap, max_num_keypoints=512, erode_mask=None,
        resize_factor=0.5, rng=np.random.default_rng(0)
    )
    # 10 ISS + 25 random = 35, rest padded with zero + scores=0
    assert n_valid == 35
    assert (scores == 1.0).sum() == 10
    assert (scores == 0.5).sum() == 25
    assert (scores == 0.0).sum() == 512 - 35
    # pad entries are zero
    np.testing.assert_allclose(kp_resized[35:], 0.0)


@pytest.mark.skipif(not HAS_DATA, reason="dataset mount not available")
def test_detect_iss_keypoints_smoke():
    import cv2
    zmap = cv2.imread(str(DATA_ROOT / "zmap_0000.png"), cv2.IMREAD_UNCHANGED)
    pcd, mask, _, _ = build_iss_pcd_uvd_scaled(zmap, erode_boundary=5)
    iss = detect_iss_keypoints(pcd)
    assert iss.ndim == 2 and iss.shape[1] == 3
    assert len(iss) > 10    # should be hundreds for a real scene
