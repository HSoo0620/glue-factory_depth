"""Precompute ISS keypoints + SHOT352 descriptors (from C++ bins) for the new dataset.

Pipeline:
  1. ISS detection in (u, v, depth_scaled) — same as FPFH
  2. Keypoint XYZ in camera frame (mm) via pixel_to_cam_xyz
  3. KDTree lookup into C++ cloud (camera frame mm): X=u·0.056, Y=v·0.056, Z=raw·0.0085

C++ bin name: `<zmap_stem>_shot352.bin` (e.g., zmap_0042_shot352.bin).
Bin format: uint32 N, uint32 D=352, per-point [f32 x, y, z, f32[352]].
NaN descriptors are filtered during load.

Frame note: C++ bin stores points in camera-frame mm (same convention as
coords.pixel_to_cam_xyz). Python KDTree query uses mm directly — no
pixel-raw conversion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from gluefactory.datasets.new_dataset import constants as C
from gluefactory.datasets.new_dataset.coords import pixel_to_cam_xyz
from gluefactory.datasets.new_dataset.iss_detection import (
    build_iss_pcd_uvd_scaled, detect_iss_keypoints, select_keypoints,
)

SHOT_DIM = 352


def load_shot352_bin(bin_path: Path):
    with open(bin_path, "rb") as f:
        N, D = np.frombuffer(f.read(8), dtype=np.uint32)
        N, D = int(N), int(D)
        assert D == SHOT_DIM, f"Expected D={SHOT_DIM}, got {D} in {bin_path}"
        data = np.frombuffer(f.read(N * (3 + D) * 4), dtype=np.float32)
    data = data.reshape(N, 3 + D)
    pts = data[:, :3]
    desc = data[:, 3:]
    valid = ~np.isnan(desc).any(axis=1)
    return pts[valid], desc[valid]


def lookup_shot352(pts_cloud: np.ndarray, desc_cloud: np.ndarray,
                   kp_xyz: np.ndarray, n_valid: int):
    out = np.zeros((kp_xyz.shape[0], SHOT_DIM), dtype=np.float32)
    if n_valid < 1 or len(pts_cloud) < 1:
        return out
    tree = cKDTree(pts_cloud)
    _, idxs = tree.query(kp_xyz[:n_valid])
    out[:n_valid] = desc_cloud[idxs]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=str(C.DEFAULT_DATA_ROOT))
    p.add_argument("--bin_dir", type=str, required=True,
                   help="Directory with <zmap_stem>_shot352.bin files")
    p.add_argument("--cache_root", type=str,
                   default="gluefactory/datasets/new_dataset_cache")
    p.add_argument("--max_num_keypoints", type=int, default=512)
    p.add_argument("--gamma_21", type=float, default=0.5)
    p.add_argument("--gamma_32", type=float, default=0.5)
    p.add_argument("--min_neighbors", type=int, default=5)
    p.add_argument("--erode_boundary", type=int, default=5)
    p.add_argument("--resize_factor", type=float, default=C.RESIZE_FACTOR)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data_root = Path(args.data_root)
    bin_dir = Path(args.bin_dir)
    cache_dir = Path(args.cache_root) / "cache_new_iss_shot352"
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(data_root.glob("zmap_*.png"))
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    print(f"Found {len(image_paths)} zmaps, bins in {bin_dir}, cache → {cache_dir}")

    rng = np.random.default_rng(args.seed)
    n_iss_list, n_valid_list, n_skip = [], [], 0

    for img_path in tqdm(image_paths, desc="Precomputing ISS+SHOT352"):
        out_path = cache_dir / f"{img_path.stem}.npz"
        if out_path.exists() and not args.force:
            continue

        bin_path = bin_dir / f"{img_path.stem}_shot352.bin"
        if not bin_path.exists():
            n_skip += 1
            continue
        pts_cloud, desc_cloud = load_shot352_bin(bin_path)

        zmap = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
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

        h, w = zmap.shape
        kp_xyz = np.zeros((args.max_num_keypoints, 3), dtype=np.float64)
        for i in range(n_valid):
            u = int(round(float(kp_uv_orig[i, 0])))
            v = int(round(float(kp_uv_orig[i, 1])))
            u = max(0, min(u, w - 1)); v = max(0, min(v, h - 1))
            kp_xyz[i] = pixel_to_cam_xyz(u, v, zmap[v, u])

        desc = lookup_shot352(pts_cloud, desc_cloud, kp_xyz, n_valid)

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

    if n_iss_list:
        ii = np.asarray(n_iss_list); vv = np.asarray(n_valid_list)
        print(f"\n--- Stats ---")
        print(f"ISS    : mean={ii.mean():.1f}, min={ii.min()}, max={ii.max()}")
        print(f"n_valid: mean={vv.mean():.1f}, min={vv.min()}, max={vv.max()}")
        print(f"Skipped (bin missing): {n_skip}")

    with open(cache_dir / "precompute_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Done! {cache_dir}/")


if __name__ == "__main__":
    main()
