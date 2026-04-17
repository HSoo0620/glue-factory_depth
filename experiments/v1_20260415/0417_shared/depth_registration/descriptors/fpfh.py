"""Open3D FPFH (33D, L2-normalized).

v1_20260415 `_norm_0417` 학습은 radius-only neighborhood + L2 정규화된 FPFH 로
수행되었으므로 (`experiments/v1_20260415/precompute/iss_fpfh_norm.py`) 추론에서도
동일 분포를 따른다.
"""
from __future__ import annotations
import numpy as np
import open3d as o3d

from .. import params as P


def compute_fpfh(pts_mm: np.ndarray, kp_mm: np.ndarray,
                 voxel: float = P.VOXEL_MM,
                 normal_r: float = P.NORMAL_R_MM,
                 fpfh_r: float = P.FPFH_R_MM) -> np.ndarray:
    """Dense PCD → voxel → normals(radius) → FPFH(radius) → kp 1-NN → L2 norm."""
    if len(kp_mm) == 0:
        return np.zeros((0, 33), dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_mm.astype(np.float64))
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamRadius(radius=normal_r))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamRadius(radius=fpfh_r))
    hist = np.asarray(fpfh.data).T.astype(np.float32)   # (M, 33) raw

    tree = o3d.geometry.KDTreeFlann(pcd)
    desc = np.zeros((len(kp_mm), 33), dtype=np.float32)
    for i, q in enumerate(kp_mm.astype(np.float64)):
        _, idx, _ = tree.search_knn_vector_3d(q, 1)
        desc[i] = hist[idx[0]]

    # L2 normalize — v1 _norm_0417 학습 분포에 맞춤
    norms = np.linalg.norm(desc, axis=1, keepdims=True)
    desc = desc / (norms + 1e-8)
    return desc.astype(np.float32)
