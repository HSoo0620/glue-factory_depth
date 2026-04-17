"""SHOT352 descriptor discriminability diagnostic.

For a few GT pairs:
  - Load master/input SHOT bins (cam-frame mm)
  - Build KDTree over cloud points
  - For each GT correspondence (master_uv → input_uv), convert to cam-frame mm,
    lookup nearest cloud point, fetch SHOT descriptor
  - L2 distance on positive pairs (matched) vs random negatives
  - Report: pos mean/median, neg mean/median, pos<neg ratio, non-zero bins,
    all-zero descriptor count

Run: python scripts/diagnose_shot_discriminability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gluefactory.datasets.new_dataset import constants as C
from gluefactory.datasets.new_dataset.coords import pixel_to_cam_xyz

BIN_DIR = Path("gluefactory/datasets/Descriptor/output_shot_zmap_v5")
DATA_ROOT = C.DEFAULT_DATA_ROOT
PAIRS = [(0, 1), (0, 2), (0, 5)]
MAX_GT = 200
N_NEG = 1000
RNG = np.random.default_rng(0)


def load_shot352_bin(path: Path):
    with open(path, "rb") as f:
        N, D = np.frombuffer(f.read(8), dtype=np.uint32)
        data = np.frombuffer(f.read(int(N) * (3 + int(D)) * 4), dtype=np.float32)
    data = data.reshape(int(N), 3 + int(D))
    return data[:, :3], data[:, 3:]


def fetch_descs(cloud_pts, cloud_desc, query_uv, zmap):
    """query_uv: (K,2) float — round to int, lookup raw, cam-frame mm, kd-nearest."""
    u = np.clip(np.round(query_uv[:, 0]).astype(int), 0, zmap.shape[1] - 1)
    v = np.clip(np.round(query_uv[:, 1]).astype(int), 0, zmap.shape[0] - 1)
    raw = zmap[v, u]
    valid = raw > 0
    xyz = pixel_to_cam_xyz(u, v, raw)
    tree = cKDTree(cloud_pts)
    d_nn, idx = tree.query(xyz)
    return cloud_desc[idx], valid, d_nn


def main():
    for m_id, i_id in PAIRS:
        print(f"\n=== pair_{m_id:04d}_{i_id:04d} ===")
        bin_m = BIN_DIR / f"zmap_{m_id:04d}_shot352.bin"
        bin_i = BIN_DIR / f"zmap_{i_id:04d}_shot352.bin"
        pts_m, desc_m = load_shot352_bin(bin_m)
        pts_i, desc_i = load_shot352_bin(bin_i)

        zero_m = (np.linalg.norm(desc_m, axis=1) == 0).mean()
        zero_i = (np.linalg.norm(desc_i, axis=1) == 0).mean()
        nnz_m = (desc_m != 0).sum(axis=1).mean()
        nnz_i = (desc_i != 0).sum(axis=1).mean()
        print(f"  N_pts: master={len(pts_m)}, input={len(pts_i)}")
        print(f"  all-zero desc ratio: master={zero_m:.4f}, input={zero_i:.4f}")
        print(f"  non-zero bins / 352: master={nnz_m:.1f}, input={nnz_i:.1f}")

        zmap_m = cv2.imread(str(DATA_ROOT / f"zmap_{m_id:04d}.png"), cv2.IMREAD_UNCHANGED)
        zmap_i = cv2.imread(str(DATA_ROOT / f"zmap_{i_id:04d}.png"), cv2.IMREAD_UNCHANGED)

        csv = pd.read_csv(DATA_ROOT / f"pair_{m_id:04d}_{i_id:04d}.csv")
        csv = csv[csv["occluded"] == 0]
        if len(csv) > MAX_GT:
            csv = csv.sample(MAX_GT, random_state=0)
        mkp = csv[["master_x", "master_y"]].values.astype(np.float32)
        ikp = csv[["input_x", "input_y"]].values.astype(np.float32)

        dm, vm, nn_m = fetch_descs(pts_m, desc_m, mkp, zmap_m)
        di, vi, nn_i = fetch_descs(pts_i, desc_i, ikp, zmap_i)
        ok = vm & vi
        dm, di = dm[ok], di[ok]
        print(f"  GT pairs used: {ok.sum()}/{len(csv)}")
        print(f"  KDTree NN dist (mm): master mean={nn_m.mean():.2f}, input mean={nn_i.mean():.2f}")

        pos_l2 = np.linalg.norm(dm - di, axis=1)

        # negatives: random permutation of input
        perm = RNG.permutation(len(di))
        # ensure not self
        while np.any(perm == np.arange(len(di))):
            perm = RNG.permutation(len(di))
        neg_l2 = np.linalg.norm(dm - di[perm], axis=1)

        # random pairs across the whole cloud (cross-frame)
        rand_m = RNG.choice(len(desc_m), N_NEG)
        rand_i = RNG.choice(len(desc_i), N_NEG)
        rand_l2 = np.linalg.norm(desc_m[rand_m] - desc_i[rand_i], axis=1)

        print(f"  L2 positive : mean={pos_l2.mean():.3f}, median={np.median(pos_l2):.3f}")
        print(f"  L2 neg(perm): mean={neg_l2.mean():.3f}, median={np.median(neg_l2):.3f}")
        print(f"  L2 neg(rand): mean={rand_l2.mean():.3f}, median={np.median(rand_l2):.3f}")
        ratio_perm = pos_l2.mean() / neg_l2.mean()
        ratio_rand = pos_l2.mean() / rand_l2.mean()
        print(f"  pos/neg ratio (perm): {ratio_perm:.3f}  (rand): {ratio_rand:.3f}")
        pos_lt_neg = (pos_l2 < neg_l2).mean()
        print(f"  pos < neg(perm) rate: {pos_lt_neg:.3f}")


if __name__ == "__main__":
    main()
