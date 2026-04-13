from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from gluefactory.datasets.new_dataset import constants as C
from gluefactory.datasets.new_dataset.coords import (
    SceneConfig,
    build_camera_frame_pcd,
    cam_to_world_xyz,
    load_scene_config,
    pixel_to_cam_xyz,
)

DATA_ROOT = C.DEFAULT_DATA_ROOT
HAS_DATA = DATA_ROOT.exists()


def test_pixel_to_cam_xyz_formula():
    u = np.array([0.0, 100.0, 200.0])
    v = np.array([0.0, 50.0, 100.0])
    raw = np.array([0, 1000, 65535], dtype=np.float64)
    xyz = pixel_to_cam_xyz(u, v, raw)
    assert xyz.shape == (3, 3)
    np.testing.assert_allclose(xyz[:, 0], u * C.LAT_MM)
    np.testing.assert_allclose(xyz[:, 1], v * C.LAT_MM)
    np.testing.assert_allclose(xyz[:, 2], raw * C.VERT_MM)


def test_cam_to_world_roundtrip():
    R = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    cfg = SceneConfig(zmap_shape=(10, 10), R_cam=R, t_cam=t)
    cam = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    world = cam_to_world_xyz(cam, cfg)
    np.testing.assert_allclose(world, cam + t)


def test_cam_to_world_rotation():
    theta = np.pi / 2
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    t = np.zeros(3)
    cfg = SceneConfig(zmap_shape=(10, 10), R_cam=R, t_cam=t)
    cam = np.array([[1.0, 0.0, 0.0]])
    world = cam_to_world_xyz(cam, cfg)
    np.testing.assert_allclose(world, [[0.0, 1.0, 0.0]], atol=1e-10)


@pytest.mark.skipif(not HAS_DATA, reason="dataset mount not available")
def test_load_scene_config_shape_and_dtype():
    cfg = load_scene_config(0, DATA_ROOT)
    assert cfg.R_cam.shape == (3, 3)
    assert cfg.t_cam.shape == (3,)
    assert cfg.zmap_shape[1] == 2413  # W fixed across 641 scenes
    # Rotation matrix should be orthonormal
    np.testing.assert_allclose(cfg.R_cam @ cfg.R_cam.T, np.eye(3), atol=1e-5)


@pytest.mark.skipif(not HAS_DATA, reason="dataset mount not available")
def test_build_camera_frame_pcd_smoke():
    import cv2
    zmap = cv2.imread(str(DATA_ROOT / "zmap_0000.png"), cv2.IMREAD_UNCHANGED)
    pcd, erode_mask = build_camera_frame_pcd(zmap, erode_boundary=5)
    assert isinstance(pcd, o3d.geometry.PointCloud)
    assert erode_mask.shape == zmap.shape
    n = len(pcd.points)
    assert n > 10_000
    pts = np.asarray(pcd.points)
    # Camera-frame mm sanity: X in [0, W·LAT_MM], Z >= 0
    assert pts[:, 0].min() >= 0 and pts[:, 0].max() <= zmap.shape[1] * C.LAT_MM + 1e-6
    assert pts[:, 2].min() >= 0
