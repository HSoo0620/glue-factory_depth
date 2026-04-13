"""Coordinate frames for the new dataset.

Frames:
  - pixel  : (u, v, raw_uint16)
  - camera : (X, Y, Z) mm = (u·LAT_MM, v·LAT_MM, raw·VERT_MM)
  - world  : R_cam @ cam + t_cam  (pcd.ply frame)

pcd.ply is NOT consumed during precompute. It is only used by the
registration evaluation to compare two views in the world frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

from . import constants as C


@dataclass
class SceneConfig:
    zmap_shape: tuple[int, int]  # (H, W)
    R_cam: np.ndarray            # (3, 3) float64
    t_cam: np.ndarray            # (3,)   float64 (mm)


def load_scene_config(scene_id: int, data_root: Path) -> SceneConfig:
    yaml_path = Path(data_root) / f"config_{scene_id:04d}.yaml"
    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)
    rt = meta["camera_rt"]
    R = np.asarray(rt["rotation_matrix"], dtype=np.float64)
    t = np.asarray(rt["translation_mm"], dtype=np.float64)
    png_path = Path(data_root) / f"zmap_{scene_id:04d}.png"
    zmap = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if zmap is None:
        raise FileNotFoundError(png_path)
    return SceneConfig(zmap_shape=zmap.shape, R_cam=R, t_cam=t)


def pixel_to_cam_xyz(u, v, raw) -> np.ndarray:
    """(u, v, raw_uint16) → camera-frame (X, Y, Z) mm. Vectorized."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    return np.stack([u * C.LAT_MM, v * C.LAT_MM, raw * C.VERT_MM], axis=-1)


def cam_to_world_xyz(xyz_cam: np.ndarray, cfg: SceneConfig) -> np.ndarray:
    return xyz_cam @ cfg.R_cam.T + cfg.t_cam


def build_camera_frame_pcd(zmap: np.ndarray, erode_boundary: int = 0):
    """All valid pixels → camera-frame PCD + erosion mask.

    erode_boundary applied only to `mask`; if you need a PCD that matches
    a specific kp mask, pass erode_boundary > 0 and use the returned mask.
    """
    mask = (zmap > 0).astype(np.uint8)
    if erode_boundary > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=erode_boundary)
    vs, us = np.where(mask > 0)
    raw = zmap[vs, us]
    xyz = pixel_to_cam_xyz(us, vs, raw)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return pcd, mask
