"""Registration RMSE (forward / reverse / bidirectional) for new_dataset ISS+FPFH/SHOT+LG.

이전 `eval_registration_iss_shot.py` 패턴을 new_dataset 규약에 이식:
  - Forward  : view0=master, view1=input → RANSAC(cam_i → cam_m)
  - Reverse  : view0/1 swap → RANSAC(cam_m → cam_i)
  - Bi-RMSE  : (rmse_fwd + rmse_rev) / 2

좌표계: **camera frame XYZ (mm)** — `pixel_to_cam_xyz` 사용.
GT transform: scene config (camera_rt) 로 직접 유도
    R_gt_fwd = R_m.T @ R_i,   t_gt_fwd = R_m.T @ (t_i - t_m)   (cam_i → cam_m)
    R_gt_rev = R_gt_fwd.T,    t_gt_rev = -R_gt_fwd.T @ t_gt_fwd

사용법:
    python eval_registration_new_iss_desc.py --checkpoint outputs/training/0413_new_iss_fpfh_v5_lg/checkpoint_best.tar \
        --indices 0 10 50 90 --descriptor_type fpfh --experiment 0413_new_iss_fpfh_v5_lg
    
    python eval_registration_new_iss_desc.py --checkpoint outputs/training/0413_new_iss_shot_v5_lg/checkpoint_best.tar \
        --indices 0 10 50 90 --descriptor_type shot --experiment 0413_new_iss_shot_v5_lg
-------------------
    python eval_registration_new_iss_desc.py --checkpoint outputs/training/0413_new_iss_fpfh_v5_lg/checkpoint_best.tar \
        --descriptor_type fpfh --experiment 0413_new_iss_fpfh_v5_lg --num_samples 20
    
    python eval_registration_new_iss_desc.py --checkpoint outputs/training/0413_new_iss_shot_v5_lg/checkpoint_best.tar \
        --descriptor_type shot --experiment 0413_new_iss_shot_v5_lg --num_samples 20


        """
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.models import get_model
from gluefactory.datasets.new_dataset import (
    NewDatasetISSDescDataset, collate_fn_dynamic_pad,
)
from gluefactory.datasets.new_dataset import constants as NEW_C
from gluefactory.datasets.new_dataset.coords import (
    load_scene_config, pixel_to_cam_xyz,
)
from gluefactory.datasets.new_dataset.split import pair_filename_to_scene_ids
from gluefactory.utils.tensor import batch_to_device


# ─── Core math ────────────────────────────────

def pixel_to_cam_vec(keypoints_2d: np.ndarray, zmap: np.ndarray):
    """(N,2) keypoints (원본 픽셀) → (N,3) cam-frame mm + valid mask."""
    H, W = zmap.shape
    u = np.clip(np.round(keypoints_2d[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(keypoints_2d[:, 1]).astype(int), 0, H - 1)
    raw = zmap[v, u].astype(np.float64)
    xyz = pixel_to_cam_xyz(u, v, raw)
    return xyz, raw > 0


def sample_camera_pcd(zmap: np.ndarray, n_pts: int = 30000, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vs, us = np.where(zmap > 0)
    n_valid = len(vs)
    if n_valid == 0:
        return np.zeros((0, 3), dtype=np.float64)
    idx = rng.choice(n_valid, min(n_pts, n_valid), replace=False)
    raw = zmap[vs[idx], us[idx]].astype(np.float64)
    return pixel_to_cam_xyz(us[idx].astype(np.float64), vs[idx].astype(np.float64), raw)


def rigid_transform_svd(P_src: np.ndarray, P_dst: np.ndarray):
    c_src = P_src.mean(axis=0)
    c_dst = P_dst.mean(axis=0)
    H = (P_src - c_src).T @ (P_dst - c_dst)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c_dst - R @ c_src
    return R, t


def ransac_rigid(src: np.ndarray, dst: np.ndarray, n_iter: int = 1000,
                 inlier_th: float = 5.0):
    N = src.shape[0]
    if N < 3:
        return np.eye(3), np.zeros(3), np.zeros(N, dtype=bool)
    best_inl = np.zeros(N, dtype=bool)
    best_R, best_t = np.eye(3), np.zeros(3)
    for _ in range(n_iter):
        idx = np.random.choice(N, 3, replace=False)
        try:
            R, t = rigid_transform_svd(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        err = np.linalg.norm((R @ src.T).T + t - dst, axis=1)
        inl = err < inlier_th
        if inl.sum() > best_inl.sum():
            best_inl = inl; best_R, best_t = R, t
    if best_inl.sum() >= 3:
        best_R, best_t = rigid_transform_svd(src[best_inl], dst[best_inl])
    return best_R, best_t, best_inl


def compute_transform_rmse(P_src: np.ndarray, R_est, t_est, R_gt, t_gt) -> float:
    """T_est vs T_gt 를 동일 소스에 적용한 후 RMSE."""
    P_est = (R_est @ P_src.T).T + t_est
    P_gt = (R_gt @ P_src.T).T + t_gt
    return float(np.sqrt(np.mean(np.sum((P_est - P_gt) ** 2, axis=1))))


def gt_transform_cam_i_to_cam_m(cfg_m, cfg_i):
    """cam_m = R @ cam_i + t  (world = R_cam @ cam + t_cam 규약)."""
    R = cfg_m.R_cam.T @ cfg_i.R_cam
    t = cfg_m.R_cam.T @ (cfg_i.t_cam - cfg_m.t_cam)
    return R, t


# ─── Model / inference ────────────────────────

def load_model(cp_path: str, device: str):
    cp = torch.load(cp_path, map_location="cpu")
    conf = OmegaConf.create(cp["conf"])
    model = get_model(conf.model.name)(conf.model).to(device)
    model.load_state_dict(cp["model"], strict=False)
    model.eval()
    print(f"Loaded: {cp_path} (epoch {cp['epoch']})")
    return model


def extract_matches(pred, batch_idx: int, conf_th: float = 0.0):
    kp0 = pred["keypoints0"][batch_idx].cpu().numpy()
    kp1 = pred["keypoints1"][batch_idx].cpu().numpy()
    m0 = pred["matches0"][batch_idx].cpu().numpy()
    if "matching_scores0" in pred:
        scores = pred["matching_scores0"][batch_idx].cpu().numpy()
        valid = (m0 > -1) & (scores > conf_th)
    else:
        valid = m0 > -1
    return kp0[valid], kp1[m0[valid]], int(valid.sum())


def swap_batch(batch):
    """view0 ↔ view1 swap (reverse 방향 추론용). master_path/input_path 도 교체."""
    swapped = {
        "view0": batch["view1"],
        "view1": batch["view0"],
    }
    if "gt_matches" in batch:
        swapped["gt_matches"] = batch["gt_matches"]
    return swapped


def resolve_cache_dir(args) -> Path:
    if args.descriptor_type == "fpfh":
        return (Path("gluefactory/datasets/new_dataset_cache")
                / f"cache_new_iss_fpfh_v{args.voxel_size}_r{args.fpfh_radius}")
    return (Path("gluefactory/datasets/new_dataset_cache")
            / f"cache_new_iss_shot352_{args.shot_version}")


# ─── Per-pair evaluation ──────────────────────

@torch.no_grad()
def process_pair(model, dataset, idx: int, args, device) -> dict | None:
    sample = dataset[idx]
    master_path = Path(sample["master_path"])
    input_path = Path(sample["input_path"])
    csv_fname = Path(sample["csv_path"]).name
    m_id, i_id = pair_filename_to_scene_ids(csv_fname)

    data_root = master_path.parent
    cfg_m = load_scene_config(m_id, data_root)
    cfg_i = load_scene_config(i_id, data_root)
    zmap_m = cv2.imread(str(master_path), cv2.IMREAD_UNCHANGED)
    zmap_i = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)

    R_gt_fwd, t_gt_fwd = gt_transform_cam_i_to_cam_m(cfg_m, cfg_i)
    R_gt_rev = R_gt_fwd.T
    t_gt_rev = -R_gt_fwd.T @ t_gt_fwd

    # ─── Forward: view0=master, view1=input → RANSAC(cam_i → cam_m) ───
    batch_fwd = collate_fn_dynamic_pad([sample])
    batch_fwd_dev = batch_to_device(batch_fwd, device)
    pred_fwd = model(batch_fwd_dev)
    mkp0_f, mkp1_f, n_match_fwd = extract_matches(pred_fwd, 0, args.conf_th)
    if args.resize_factor != 1.0:
        mkp0_f = mkp0_f / args.resize_factor
        mkp1_f = mkp1_f / args.resize_factor
    pts_m_f, vm_m_f = pixel_to_cam_vec(mkp0_f, zmap_m)
    pts_i_f, vm_i_f = pixel_to_cam_vec(mkp1_f, zmap_i)
    ok_f = vm_m_f & vm_i_f
    n_3d_fwd = int(ok_f.sum())
    if n_3d_fwd < 3:
        print(f"  [{idx}] Forward: not enough 3D points ({n_3d_fwd}). Skip.")
        return None
    R_est_fwd, t_est_fwd, inl_fwd = ransac_rigid(
        pts_i_f[ok_f], pts_m_f[ok_f],
        n_iter=args.ransac_iter, inlier_th=args.ransac_th)
    n_inl_fwd = int(inl_fwd.sum())
    P_i = sample_camera_pcd(zmap_i, args.n_sample_pts)
    rmse_fwd = compute_transform_rmse(P_i, R_est_fwd, t_est_fwd, R_gt_fwd, t_gt_fwd)

    # ─── Reverse: view0=input, view1=master → RANSAC(cam_m → cam_i) ───
    batch_rev = swap_batch(batch_fwd)
    batch_rev_dev = batch_to_device(batch_rev, device)
    pred_rev = model(batch_rev_dev)
    mkp0_r, mkp1_r, n_match_rev = extract_matches(pred_rev, 0, args.conf_th)
    if args.resize_factor != 1.0:
        mkp0_r = mkp0_r / args.resize_factor
        mkp1_r = mkp1_r / args.resize_factor
    pts_i_r, vm_i_r = pixel_to_cam_vec(mkp0_r, zmap_i)   # view0(reversed) = input
    pts_m_r, vm_m_r = pixel_to_cam_vec(mkp1_r, zmap_m)   # view1(reversed) = master
    ok_r = vm_i_r & vm_m_r
    n_3d_rev = int(ok_r.sum())
    if n_3d_rev < 3:
        print(f"  [{idx}] Reverse: not enough 3D points ({n_3d_rev}). Skip.")
        return None
    R_est_rev, t_est_rev, inl_rev = ransac_rigid(
        pts_m_r[ok_r], pts_i_r[ok_r],
        n_iter=args.ransac_iter, inlier_th=args.ransac_th)
    n_inl_rev = int(inl_rev.sum())
    P_m = sample_camera_pcd(zmap_m, args.n_sample_pts)
    rmse_rev = compute_transform_rmse(P_m, R_est_rev, t_est_rev, R_gt_rev, t_gt_rev)

    rmse_bi = (rmse_fwd + rmse_rev) / 2.0

    print(f"  pair_{m_id:04d}_{i_id:04d} [{idx}]: "
          f"fwd={rmse_fwd:.3f} rev={rmse_rev:.3f} bi={rmse_bi:.3f} mm  "
          f"(matches fwd/rev = {n_match_fwd}/{n_match_rev}, "
          f"inliers fwd/rev = {n_inl_fwd}/{n_inl_rev})")

    return {
        "idx": idx,
        "master_id": m_id,
        "input_id": i_id,
        "n_matches_fwd": n_match_fwd,
        "n_matches_rev": n_match_rev,
        "n_3d_fwd": n_3d_fwd,
        "n_3d_rev": n_3d_rev,
        "n_inliers_fwd": n_inl_fwd,
        "n_inliers_rev": n_inl_rev,
        "rmse_fwd": float(rmse_fwd),
        "rmse_rev": float(rmse_rev),
        "rmse_bi":  float(rmse_bi),
    }


# ─── Main ─────────────────────────────────────

def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.checkpoint:
        cp_path = args.checkpoint
    else:
        from gluefactory.utils.experiments import get_best_checkpoint
        cp_path = str(get_best_checkpoint(args.experiment))
    model = load_model(cp_path, device)

    cache_dir = resolve_cache_dir(args)
    ds = NewDatasetISSDescDataset(
        split=args.split, cache_dir=cache_dir,
        resize_factor=args.resize_factor, val_ratio=0.05, seed=0,
    )
    print(f"{args.split}: {len(ds)} pairs ({args.descriptor_type}, cache={cache_dir})")

    if args.indices is not None:
        idxs = list(args.indices)
    else:
        idxs = list(range(min(args.num_samples, len(ds))))
    print(f"Evaluating {len(idxs)} pairs: {idxs}")

    records = []
    for idx in tqdm(idxs, desc="Eval"):
        rec = process_pair(model, ds, idx, args, device)
        if rec is not None:
            records.append(rec)

    if not records:
        print("No valid pairs."); return

    df = pd.DataFrame(records)
    print(f"\n=== Summary over {len(df)} pairs ===")
    for col in ("rmse_fwd", "rmse_rev", "rmse_bi"):
        a = df[col].to_numpy()
        print(f"{col:8s}  mean={a.mean():.3f}  median={np.median(a):.3f}  max={a.max():.3f} mm")

    out_dir = Path(args.output_dir or f"results/{args.experiment}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cp_stem = Path(cp_path).stem
    csv_path = out_dir / f"rmse_{args.split}_{cp_stem}.csv"

    summary_rows = []
    for stat_name, fn in (("mean", np.mean), ("median", np.median), ("max", np.max)):
        row = {c: "" for c in df.columns}
        row["idx"] = stat_name
        for col in ("rmse_fwd", "rmse_rev", "rmse_bi"):
            row[col] = float(fn(df[col].to_numpy()))
        summary_rows.append(row)
    df_out = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved per-pair + summary → {csv_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--experiment", type=str, default="0413_new_iss_fpfh_lg")
    p.add_argument("--descriptor_type", type=str, default="fpfh", choices=["fpfh", "shot"])
    p.add_argument("--fpfh_radius", type=float, default=50.0)
    p.add_argument("--voxel_size", type=float, default=5.0)
    p.add_argument("--shot_version", type=str, default="v5")
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--indices", type=int, nargs="*", default=None,
                   help="특정 페어 인덱스만 평가 (미지정 시 앞에서 num_samples개)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="CSV 저장 위치 (미지정 시 results/{experiment}/)")
    p.add_argument("--resize_factor", type=float, default=NEW_C.RESIZE_FACTOR)
    p.add_argument("--ransac_iter", type=int, default=1000)
    p.add_argument("--ransac_th", type=float, default=5.0,
                   help="RANSAC inlier threshold (mm)")
    p.add_argument("--conf_th", type=float, default=0.0)
    p.add_argument("--n_sample_pts", type=int, default=30000,
                   help="RMSE 계산용 소스 포인트 샘플 수")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
