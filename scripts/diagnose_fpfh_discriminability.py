"""FPFH33 descriptor discriminability sweep.

Same methodology as diagnose_shot_discriminability.py:
  - For each (voxel, normal_r, fpfh_r) combo:
    - Build camera-frame PCD (dense or voxel-downsampled)
    - estimate_normals + compute_fpfh (Open3D)
    - For GT uv pairs, cam-frame mm → KDTree nearest → fetch FPFH(33)
    - Measure L2 positive vs L2 negative (perm within-pair / random cross-cloud)

Report per-pair and mean ratio. Target: pos/neg << 1.0, pos<neg rate > 0.85.

Run: python scripts/diagnose_fpfh_discriminability.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gluefactory.datasets.new_dataset import constants as C
from gluefactory.datasets.new_dataset.coords import (
    build_camera_frame_pcd, pixel_to_cam_xyz,
)

DATA_ROOT = C.DEFAULT_DATA_ROOT
PAIRS = [(0, 1), (0, 2), (0, 5), (1, 2)]
MAX_GT = 200
N_NEG = 1000
RNG = np.random.default_rng(0)

# (voxel_mm, normal_r_mm, fpfh_r_mm)
PARAM_GRID = [
    (0.0, 10.0, 20.0),    # SHOT-current matched (no voxel)
    (0.0, 25.0, 50.0),    # mid
    (0.0, 50.0, 100.0),   # existing FPFH cache (100/50)
    (0.0, 100.0, 200.0),  # large
    (2.0, 10.0, 20.0),    # voxel + SHOT-current
    (2.0, 25.0, 50.0),
    (5.0, 25.0, 50.0),
    (5.0, 50.0, 100.0),
]


def compute_fpfh(zmap, voxel, normal_r, fpfh_r):
    pcd, _ = build_camera_frame_pcd(zmap)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_r, max_nn=30)
    )
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_r, max_nn=100)
    )
    pts = np.asarray(pcd.points, dtype=np.float32)
    desc = np.asarray(feat.data, dtype=np.float32).T  # (N, 33)
    return pts, desc


def fetch_descs(cloud_pts, cloud_desc, query_uv, zmap):
    u = np.clip(np.round(query_uv[:, 0]).astype(int), 0, zmap.shape[1] - 1)
    v = np.clip(np.round(query_uv[:, 1]).astype(int), 0, zmap.shape[0] - 1)
    raw = zmap[v, u]
    valid = raw > 0
    xyz = pixel_to_cam_xyz(u, v, raw).astype(np.float32)
    tree = cKDTree(cloud_pts)
    d_nn, idx = tree.query(xyz)
    return cloud_desc[idx], valid, d_nn


def summarize_pair(pts_m, desc_m, zmap_m, pts_i, desc_i, zmap_i, csv):
    mkp = csv[["master_x", "master_y"]].values.astype(np.float32)
    ikp = csv[["input_x", "input_y"]].values.astype(np.float32)

    dm, vm, _ = fetch_descs(pts_m, desc_m, mkp, zmap_m)
    di, vi, _ = fetch_descs(pts_i, desc_i, ikp, zmap_i)
    ok = vm & vi
    dm, di = dm[ok], di[ok]

    if len(dm) < 10:
        return None

    pos_l2 = np.linalg.norm(dm - di, axis=1)
    perm = RNG.permutation(len(di))
    while len(di) > 1 and np.any(perm == np.arange(len(di))):
        perm = RNG.permutation(len(di))
    neg_l2 = np.linalg.norm(dm - di[perm], axis=1)

    n_rand = min(N_NEG, len(desc_m), len(desc_i))
    rand_m = RNG.choice(len(desc_m), n_rand)
    rand_i = RNG.choice(len(desc_i), n_rand)
    rand_l2 = np.linalg.norm(desc_m[rand_m] - desc_i[rand_i], axis=1)

    zero_ratio = (np.linalg.norm(desc_m, axis=1) == 0).mean()
    return {
        "n_pts_m": len(pts_m),
        "n_pts_i": len(pts_i),
        "n_gt": int(ok.sum()),
        "pos_mean": float(pos_l2.mean()),
        "neg_perm_mean": float(neg_l2.mean()),
        "neg_rand_mean": float(rand_l2.mean()),
        "ratio_perm": float(pos_l2.mean() / max(neg_l2.mean(), 1e-8)),
        "ratio_rand": float(pos_l2.mean() / max(rand_l2.mean(), 1e-8)),
        "pos_lt_neg": float((pos_l2 < neg_l2).mean()),
        "zero_ratio": float(zero_ratio),
    }


def main():
    print(f"Sweeping {len(PARAM_GRID)} param combos over {len(PARAM_GRID) and len(PAIRS)} pairs\n")

    # Load GT CSVs + zmaps once
    zmaps = {}
    csvs = {}
    scene_ids = set()
    for m_id, i_id in PAIRS:
        scene_ids.add(m_id); scene_ids.add(i_id)
        csv_path = DATA_ROOT / f"pair_{m_id:04d}_{i_id:04d}.csv"
        if not csv_path.exists():
            print(f"!! missing {csv_path}, skip"); continue
        csv = pd.read_csv(csv_path)
        csv = csv[csv["occluded"] == 0]
        if len(csv) > MAX_GT:
            csv = csv.sample(MAX_GT, random_state=0)
        csvs[(m_id, i_id)] = csv

    for sid in scene_ids:
        zmaps[sid] = cv2.imread(str(DATA_ROOT / f"zmap_{sid:04d}.png"),
                                cv2.IMREAD_UNCHANGED)

    results = []
    for voxel, nr, fr in PARAM_GRID:
        print(f"--- voxel={voxel}, normal_r={nr}, fpfh_r={fr} ---")
        per_pair_stats = []
        fpfh_cache = {}
        t0 = time.time()
        for sid in scene_ids:
            fpfh_cache[sid] = compute_fpfh(zmaps[sid], voxel, nr, fr)
        t_fpfh = time.time() - t0
        print(f"  FPFH compute ({len(scene_ids)} scenes): {t_fpfh:.1f}s")

        for (m_id, i_id), csv in csvs.items():
            pts_m, desc_m = fpfh_cache[m_id]
            pts_i, desc_i = fpfh_cache[i_id]
            s = summarize_pair(pts_m, desc_m, zmaps[m_id],
                               pts_i, desc_i, zmaps[i_id], csv)
            if s is None:
                continue
            per_pair_stats.append(s)
            print(f"  pair_{m_id:04d}_{i_id:04d}: "
                  f"n_pts=({s['n_pts_m']},{s['n_pts_i']}), "
                  f"pos={s['pos_mean']:.3f}, neg_p={s['neg_perm_mean']:.3f}, "
                  f"ratio(perm)={s['ratio_perm']:.3f}, pos<neg={s['pos_lt_neg']:.3f}")

        if per_pair_stats:
            mean_ratio = np.mean([s["ratio_perm"] for s in per_pair_stats])
            mean_pos_lt = np.mean([s["pos_lt_neg"] for s in per_pair_stats])
            mean_zero = np.mean([s["zero_ratio"] for s in per_pair_stats])
            print(f"  >> mean ratio(perm)={mean_ratio:.3f}, "
                  f"mean pos<neg={mean_pos_lt:.3f}, zero_desc={mean_zero:.4f}\n")
            results.append({
                "voxel": voxel, "normal_r": nr, "fpfh_r": fr,
                "mean_ratio_perm": mean_ratio,
                "mean_pos_lt_neg": mean_pos_lt,
                "zero_desc_ratio": mean_zero,
            })

    print("\n=== SUMMARY (sorted by mean pos<neg rate, higher=better) ===")
    print(f"{'voxel':>6} {'normal_r':>9} {'fpfh_r':>7} {'pos/neg':>9} {'pos<neg':>9} {'zero':>7}")
    for r in sorted(results, key=lambda x: -x["mean_pos_lt_neg"]):
        print(f"{r['voxel']:>6.1f} {r['normal_r']:>9.1f} {r['fpfh_r']:>7.1f} "
              f"{r['mean_ratio_perm']:>9.3f} {r['mean_pos_lt_neg']:>9.3f} "
              f"{r['zero_desc_ratio']:>7.4f}")


if __name__ == "__main__":
    main()
