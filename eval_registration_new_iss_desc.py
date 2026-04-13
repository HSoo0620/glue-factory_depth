"""Registration RMSE evaluation for new_dataset ISS+FPFH/SHOT+LG models.

Skeleton: wires up data loading, inference, coordinate conversion, and a
placeholder RANSAC+SVD call. Refinement (sampling strategy, threshold
tuning, forward/reverse RMSE computation) happens in a follow-up plan.

Usage (skeleton run, limited samples):
    python eval_registration_new_iss_desc.py --experiment 0413_new_iss_fpfh_lg \
        --descriptor_type fpfh --num_samples 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.models import get_model
from gluefactory.datasets.new_dataset import (
    NewDatasetISSDescDataset, collate_fn_dynamic_pad,
)
from gluefactory.datasets.new_dataset import constants as NEW_C
from gluefactory.datasets.new_dataset.coords import (
    cam_to_world_xyz, load_scene_config, pixel_to_cam_xyz,
)
from gluefactory.datasets.new_dataset.split import pair_filename_to_scene_ids
from gluefactory.utils.tensor import batch_to_device


def scene_world_transform(cfg) -> np.ndarray:
    """4x4 world-from-camera transform."""
    T = np.eye(4)
    T[:3, :3] = cfg.R_cam
    T[:3, 3] = cfg.t_cam
    return T


def rigid_transform_svd(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """src → dst. Returns 4x4 transform. Minimum 3 points."""
    assert src.shape == dst.shape and src.shape[0] >= 3
    sc = src.mean(0); dc = dst.mean(0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = dc - R @ sc
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def ransac_rigid(src: np.ndarray, dst: np.ndarray, n_iter: int = 1000,
                 inlier_th: float = 5.0, min_samples: int = 3,
                 rng=None) -> tuple[np.ndarray, np.ndarray]:
    """Vanilla SVD RANSAC. Returns (T 4x4, inlier_mask)."""
    if rng is None:
        rng = np.random.default_rng(0)
    N = src.shape[0]
    if N < min_samples:
        return np.eye(4), np.zeros(N, dtype=bool)
    best_T = np.eye(4); best_inl = np.zeros(N, dtype=bool); best_cnt = -1
    for _ in range(n_iter):
        idx = rng.choice(N, min_samples, replace=False)
        T = rigid_transform_svd(src[idx], dst[idx])
        proj = (src @ T[:3, :3].T) + T[:3, 3]
        d = np.linalg.norm(proj - dst, axis=1)
        inl = d < inlier_th
        if inl.sum() > best_cnt:
            best_cnt = int(inl.sum()); best_T = T; best_inl = inl
    if best_cnt >= min_samples:
        best_T = rigid_transform_svd(src[best_inl], dst[best_inl])
    return best_T, best_inl


def transform_rmse(T_a: np.ndarray, T_b: np.ndarray, n_samples: int = 30000,
                   extent_mm: float = 100.0, rng=None) -> float:
    """Symmetric RMSE: sample random points, apply T_a vs T_b, measure mm L2."""
    if rng is None:
        rng = np.random.default_rng(0)
    pts = (rng.random((n_samples, 3)) - 0.5) * (2 * extent_mm)
    a = pts @ T_a[:3, :3].T + T_a[:3, 3]
    b = pts @ T_b[:3, :3].T + T_b[:3, 3]
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


@torch.no_grad()
def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.checkpoint:
        cp_path = args.checkpoint
    else:
        from gluefactory.utils.experiments import get_best_checkpoint
        cp_path = str(get_best_checkpoint(args.experiment))
    cp = torch.load(cp_path, map_location="cpu")
    conf = OmegaConf.create(cp["conf"])
    model = get_model(conf.model.name)(conf.model).to(device)
    model.load_state_dict(cp["model"], strict=False)
    model.eval()

    if args.descriptor_type == "fpfh":
        cache_dir = Path("gluefactory/datasets/new_dataset_cache") / f"cache_new_iss_fpfh_r{args.fpfh_radius}"
    else:
        cache_dir = Path("gluefactory/datasets/new_dataset_cache") / "cache_new_iss_shot352"

    ds = NewDatasetISSDescDataset(
        split=args.split, cache_dir=cache_dir,
        resize_factor=args.resize_factor, val_ratio=0.05, seed=0,
    )
    rng = np.random.default_rng(0)
    idxs = list(range(min(args.num_samples, len(ds))))
    rmses = []

    for idx in tqdm(idxs, desc="Evaluating"):
        sample = ds[idx]
        batch = collate_fn_dynamic_pad([sample])
        batch = batch_to_device(batch, device)
        pred = model(batch)

        kp0 = pred["keypoints0"][0].cpu().numpy()
        kp1 = pred["keypoints1"][0].cpu().numpy()
        m0 = pred["matches0"][0].cpu().numpy()
        valid = m0 > -1
        if valid.sum() < 3:
            continue
        mkp0 = kp0[valid] / args.resize_factor  # back to original pixels
        mkp1 = kp1[m0[valid]] / args.resize_factor

        master_fname = Path(sample["master_path"]).name
        input_fname = Path(sample["input_path"]).name
        csv_fname = Path(sample["csv_path"]).name
        m_id, i_id = pair_filename_to_scene_ids(csv_fname)

        data_root = Path(sample["master_path"]).parent
        cfg_m = load_scene_config(m_id, data_root)
        cfg_i = load_scene_config(i_id, data_root)
        import cv2
        zmap_m = cv2.imread(sample["master_path"], cv2.IMREAD_UNCHANGED)
        zmap_i = cv2.imread(sample["input_path"], cv2.IMREAD_UNCHANGED)

        def kp_to_world(kp_uv, zmap, cfg):
            u = np.round(kp_uv[:, 0]).astype(int)
            v = np.round(kp_uv[:, 1]).astype(int)
            u = np.clip(u, 0, zmap.shape[1] - 1)
            v = np.clip(v, 0, zmap.shape[0] - 1)
            raw = zmap[v, u]
            cam = pixel_to_cam_xyz(u, v, raw)
            keep = raw > 0
            return cam_to_world_xyz(cam, cfg), keep

        w0, ok0 = kp_to_world(mkp0, zmap_m, cfg_m)
        w1, ok1 = kp_to_world(mkp1, zmap_i, cfg_i)
        ok = ok0 & ok1
        if ok.sum() < 3:
            continue
        T_est, inl = ransac_rigid(w0[ok], w1[ok], n_iter=args.ransac_iter,
                                  inlier_th=args.ransac_th, rng=rng)

        Tm = scene_world_transform(cfg_m)
        Ti = scene_world_transform(cfg_i)
        # T_gt maps master-world → input-world (same object frame reference)
        T_gt = Ti @ np.linalg.inv(Tm)

        rmse = transform_rmse(T_est, T_gt, rng=rng)
        rmses.append(rmse)
        print(f"  pair_{m_id:04d}_{i_id:04d}: inliers={int(inl.sum())}/{int(ok.sum())} "
              f"RMSE={rmse:.3f} mm")

    if rmses:
        arr = np.asarray(rmses)
        print(f"\n=== Summary over {len(rmses)} pairs ===")
        print(f"mean RMSE: {arr.mean():.3f} mm")
        print(f"median   : {np.median(arr):.3f} mm")
        print(f"max      : {arr.max():.3f} mm")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--experiment", type=str, default="0413_new_iss_fpfh_lg")
    p.add_argument("--descriptor_type", type=str, default="fpfh", choices=["fpfh", "shot"])
    p.add_argument("--fpfh_radius", type=float, default=100.0)
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--resize_factor", type=float, default=NEW_C.RESIZE_FACTOR)
    p.add_argument("--ransac_iter", type=int, default=1000)
    p.add_argument("--ransac_th", type=float, default=5.0)
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
