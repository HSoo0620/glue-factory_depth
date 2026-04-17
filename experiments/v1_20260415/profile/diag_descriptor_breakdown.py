"""One-shot diagnostic: decompose iss_desc_scan stage into sub-steps.

Runs SHOT and FPFH sub-step timings on the same scanned input
to identify the dominant contributor inside the Descriptor stage.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pybind_shot_linux"))
sys.path.insert(0, str(ROOT / "experiments/v1_20260415"))
sys.path.insert(0, str(ROOT / "experiments/v1_20260415/precompute"))

import infer_scanned_vs_master_shot352 as infer  # noqa: E402
import helpers as h  # noqa: E402
import shot_module  # noqa: E402

LATERAL_MM = 0.056
TRANSPORT_MM = 0.056
VERTICAL_MM = 0.0085

SCAN = ROOT / "gluefactory/datasets/scanned/scanned_data1.png"


def tic():
    return time.perf_counter()


def toc(t0, label, n=None):
    dt = time.perf_counter() - t0
    extra = f" | n={n}" if n is not None else ""
    print(f"  {label:<32s} {dt:7.3f}s{extra}")
    return dt


def zmap_to_pts_mm(zmap):
    vs, us = np.where(zmap > 0)
    d = zmap[vs, us].astype(np.float64)
    return np.stack(
        [us * LATERAL_MM, vs * TRANSPORT_MM, d * VERTICAL_MM], axis=1)


def main():
    print(f"[input] {SCAN.name}")
    zmap0 = np.array(Image.open(SCAN))
    masked, _ = infer.mask_scanned_table(zmap0)
    zmap_pre = infer.apply_bilateral(masked)
    print(f"[input] zmap shape: {zmap_pre.shape}")

    print()
    print("================ SHOT sub-steps ================")
    t0 = tic()
    pts = infer.zmap_to_pcd_mm(zmap_pre)
    toc(t0, "zmap_to_pcd_mm", n=len(pts))

    t0 = tic()
    pcd_vox = h.voxel_downsample(pts, voxel_size=h.VOXEL_SIZE)
    vox_pts = np.asarray(pcd_vox.points, dtype=np.float32)
    toc(t0, "voxel_downsample", n=len(vox_pts))

    t0 = tic()
    kp_xyz_iss = h.extract_iss_on_dense(pts)
    toc(t0, "extract_iss_on_dense", n=len(kp_xyz_iss))

    t0 = tic()
    kp_xyz, _, kp_scores, n_iss, n_valid = h.subsample_or_pad_keypoints(
        kp_xyz_iss, pcd_vox, max_n=h.MAX_KEYPOINTS, seed=0)
    toc(t0, "subsample_or_pad_keypoints", n=n_valid)

    t0 = tic()
    result = shot_module.extract_shot_at_keypoints(
        vox_pts, kp_xyz.astype(np.float32),
        voxel_size=h.VOXEL_SIZE,
        normal_radius=h.NORMAL_RADIUS,
        shot_radius=h.SHOT_RADIUS,
    )
    toc(t0, "shot_module.extract_shot (C++)",
        n=int(result["valid_mask"].sum()))

    print()
    print("================ FPFH sub-steps ================")
    t0 = tic()
    pts2 = zmap_to_pts_mm(zmap_pre)
    toc(t0, "zmap_array_to_pts_mm", n=len(pts2))

    t0 = tic()
    pcd_vox2 = h.voxel_downsample(pts2)
    toc(t0, "voxel_downsample", n=len(pcd_vox2.points))

    t0 = tic()
    kp_xyz_iss2 = h.extract_iss_on_dense(pts2)
    toc(t0, "extract_iss_on_dense", n=len(kp_xyz_iss2))

    t0 = tic()
    kp_xyz2, kp_idx, kp_score, _, _ = h.subsample_or_pad_keypoints(
        kp_xyz_iss2, pcd_vox2, max_n=h.MAX_KEYPOINTS, seed=0)
    toc(t0, "subsample_or_pad_keypoints", n=len(kp_xyz2))

    t0 = tic()
    pcd_vox2.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamRadius(h.NORMAL_RADIUS))
    toc(t0, "estimate_normals(r=20mm)")

    t0 = tic()
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_vox2, o3d.geometry.KDTreeSearchParamRadius(h.FPFH_RADIUS))
    toc(t0, "compute_fpfh_feature(r=20mm)")

    t0 = tic()
    desc = np.asarray(fpfh.data, dtype=np.float32)[:, kp_idx].T
    desc = desc / (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12)
    toc(t0, "slice + L2 normalize")


if __name__ == "__main__":
    main()
