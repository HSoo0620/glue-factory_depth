"""ISS keypoint 검출 (mm 공간)."""
from __future__ import annotations
import numpy as np
import open3d as o3d

from . import params as P


def detect_iss_mm(pts_mm: np.ndarray,
                  max_keypoints: int = P.MAX_KEYPOINTS,
                  seed: int = 0) -> np.ndarray:
    """Open3D ISS (mm 공간). K > max_keypoints 이면 랜덤 subsample."""
    if pts_mm.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_mm.astype(np.float64))

    dists = pcd.compute_nearest_neighbor_distance()
    if len(dists) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    avg_nn = float(np.mean(dists))
    salient_r = P.ISS_SALIENT_MULT * avg_nn
    non_max_r = P.ISS_NONMAX_MULT * salient_r

    kp_pcd = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd,
        salient_radius=salient_r,
        non_max_radius=non_max_r,
        gamma_21=P.ISS_GAMMA_21,
        gamma_32=P.ISS_GAMMA_32,
        min_neighbors=P.ISS_MIN_NEIGHBORS,
    )
    kps = np.asarray(kp_pcd.points, dtype=np.float32)

    if len(kps) > max_keypoints:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(kps), max_keypoints, replace=False)
        kps = kps[idx]
    return kps
