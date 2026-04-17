"""Precompute ISS keypoints + dense FPFH descriptors for the new dataset.

ISS uses (u, v, depth_scaled) detection (package: iss_detection);
FPFH uses camera-frame XYZ PCD (package: coords.build_camera_frame_pcd).
Keypoint XYZ is looked up into the dense FPFH via Open3D KDTree.

Cache layout:
    gluefactory/datasets/new_dataset_cache/
      cache_new_iss_fpfh_r{radius}/
        zmap_0000.npz, zmap_0001.npz, ...
        precompute_config.json

npz fields:
    keypoints          (512, 2) float32   # resized (u·0.5, v·0.5)
    keypoint_scores    (512,)   float32   # 1.0 / 0.5 / 0.0
    descriptors        (512, 33) float32  # L2-normalized
    n_valid            int
    n_iss              int

Usage:
    conda activate LightGlue
    cd /home/jhs/work/Registration/glue-factory_depth
    python precompute_new_iss_fpfh.py                          # full (641)
    python precompute_new_iss_fpfh.py --max_images 3 --force   # smoke test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

from gluefactory.datasets.new_dataset import constants as C
from gluefactory.datasets.new_dataset.coords import build_camera_frame_pcd, pixel_to_cam_xyz
from gluefactory.datasets.new_dataset.iss_detection import (
    build_iss_pcd_uvd_scaled, detect_iss_keypoints, select_keypoints,
)


def compute_fpfh_for_keypoints(pcd_xyz: o3d.geometry.PointCloud,
                               kp_xyz: np.ndarray, n_valid: int,
                               fpfh_radius: float, fpfh_normal_radius: float,
                               voxel_size: float = 0.0,
                               fpfh_max_nn: int = 100,
                               normal_max_nn: int = 30,
                               anomaly_dist_mm: float = 10.0):
    max_n = kp_xyz.shape[0]
    out = np.zeros((max_n, 33), dtype=np.float32)
    n_anomaly = 0
    if n_valid < 3 or len(pcd_xyz.points) == 0:
        return out, n_anomaly

    if voxel_size > 0:
        pcd_xyz = pcd_xyz.voxel_down_sample(voxel_size)

    pcd_xyz.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=fpfh_normal_radius, max_nn=normal_max_nn
        )
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_xyz,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=fpfh_max_nn),
    )
    fpfh_dense = np.array(fpfh.data).T.astype(np.float32)  # (M, 33)
    kdtree = o3d.geometry.KDTreeFlann(pcd_xyz)

    pcd_pts = np.asarray(pcd_xyz.points)
    for i in range(n_valid):
        q = kp_xyz[i]
        _, idx, _ = kdtree.search_knn_vector_3d(q, 1)
        j = idx[0]
        dist = float(np.linalg.norm(pcd_pts[j] - q))
        if dist > anomaly_dist_mm:
            n_anomaly += 1
            continue  # leave as zero descriptor
        out[i] = fpfh_dense[j]

    n = np.linalg.norm(out[:n_valid], axis=1, keepdims=True)
    out[:n_valid] = out[:n_valid] / (n + 1e-8)
    return out, n_anomaly


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=str(C.DEFAULT_DATA_ROOT))
    p.add_argument("--cache_root", type=str,
                   default="gluefactory/datasets/new_dataset_cache")
    p.add_argument("--max_num_keypoints", type=int, default=512)
    p.add_argument("--gamma_21", type=float, default=0.5)
    p.add_argument("--gamma_32", type=float, default=0.5)
    p.add_argument("--min_neighbors", type=int, default=5)
    p.add_argument("--erode_boundary", type=int, default=5)
    p.add_argument("--voxel_size", type=float, default=5.0,
                   help="mm; 0 disables voxel downsampling. Diagnostic showed voxel=5 "
                        "gives best discriminability (pos<neg=0.628 vs dense 0.578)")
    p.add_argument("--fpfh_radius", type=float, default=50.0,
                   help="mm; matches SHOT v5 shot_r=50mm (1:5:10 with voxel=5)")
    p.add_argument("--fpfh_normal_radius", type=float, default=25.0,
                   help="mm; matches SHOT v5 normal_r=25mm")
    p.add_argument("--fpfh_max_nn", type=int, default=100)
    p.add_argument("--anomaly_dist_mm", type=float, default=10.0,
                   help="mm; NN threshold from keypoint to voxel-PCD point. "
                        "For voxel=5, expected NN ~2-5mm; 10mm allows margin.")
    p.add_argument("--resize_factor", type=float, default=C.RESIZE_FACTOR)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data_root = Path(args.data_root)
    cache_dir = (Path(args.cache_root)
                 / f"cache_new_iss_fpfh_v{args.voxel_size}_r{args.fpfh_radius}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(data_root.glob("zmap_*.png"))
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    print(f"Found {len(image_paths)} zmaps in {data_root}")
    print(f"Cache → {cache_dir}")
    print(f"FPFH: voxel={args.voxel_size}mm, "
          f"normal_r={args.fpfh_normal_radius}mm, r={args.fpfh_radius}mm")

    rng = np.random.default_rng(args.seed)
    n_iss_list, n_valid_list, n_anom_list = [], [], []

    for img_path in tqdm(image_paths, desc="Precomputing ISS+FPFH"):
        out_path = cache_dir / f"{img_path.stem}.npz"
        if out_path.exists() and not args.force:
            continue

        zmap = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if zmap is None:
            print(f"  [SKIP] cannot read {img_path.name}")
            continue

        pcd_iss, erode_mask, _, _ = build_iss_pcd_uvd_scaled(
            zmap, erode_boundary=args.erode_boundary
        )
        iss_kp_3d = detect_iss_keypoints(
            pcd_iss, gamma_21=args.gamma_21, gamma_32=args.gamma_32,
            min_neighbors=args.min_neighbors,
        )

        kp_resized, scores, n_valid, kp_uv_orig = select_keypoints(
            iss_kp_3d, zmap, max_num_keypoints=args.max_num_keypoints,
            erode_mask=erode_mask, resize_factor=args.resize_factor, rng=rng,
        )

        pcd_xyz, _ = build_camera_frame_pcd(zmap, erode_boundary=args.erode_boundary)
        # keypoint XYZ: look up the raw zmap value at rounded (u,v) (original-res)
        h, w = zmap.shape
        kp_xyz = np.zeros((args.max_num_keypoints, 3), dtype=np.float64)
        for i in range(n_valid):
            u = int(round(float(kp_uv_orig[i, 0])))
            v = int(round(float(kp_uv_orig[i, 1])))
            u = max(0, min(u, w - 1))
            v = max(0, min(v, h - 1))
            raw = zmap[v, u]
            kp_xyz[i] = pixel_to_cam_xyz(u, v, raw)

        desc, n_anom = compute_fpfh_for_keypoints(
            pcd_xyz, kp_xyz, n_valid,
            fpfh_radius=args.fpfh_radius,
            fpfh_normal_radius=args.fpfh_normal_radius,
            voxel_size=args.voxel_size,
            fpfh_max_nn=args.fpfh_max_nn,
            anomaly_dist_mm=args.anomaly_dist_mm,
        )

        np.savez_compressed(
            out_path,
            keypoints=kp_resized,
            keypoint_scores=scores,
            descriptors=desc,
            n_valid=np.array(n_valid, dtype=np.int64),
            n_iss=np.array(len(iss_kp_3d), dtype=np.int64),
        )
        n_iss_list.append(len(iss_kp_3d))
        n_valid_list.append(n_valid)
        n_anom_list.append(n_anom)

    if n_iss_list:
        ii = np.asarray(n_iss_list); vv = np.asarray(n_valid_list); aa = np.asarray(n_anom_list)
        print(f"\n--- Stats ---")
        print(f"ISS    : mean={ii.mean():.1f}, min={ii.min()}, max={ii.max()}")
        print(f"n_valid: mean={vv.mean():.1f}, min={vv.min()}, max={vv.max()}")
        print(f"anomaly kp (>{args.anomaly_dist_mm}mm from nearest PCD pt): mean={aa.mean():.2f} / 512")

    with open(cache_dir / "precompute_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Done! {cache_dir}/")


if __name__ == "__main__":
    main()
