"""v1 (2026-04-15) ISS+FPFH/SHOT+LG Registration 양방향 RMSE 평가.

Forward(input→master) + Reverse(master→input) → Bidirectional RMSE.
좌표계: mm. X=u*0.056, Y=v*0.056, Z=raw*0.0085.
GT: CSV 비-occluded 대응점 → mm → SVD.

사용법:
    python experiments/v1_20260415/eval/rmse.py \
        --experiment iss_fpfh_v1_20260415_norm --descriptor_type fpfh --indices 0 10 20 30
    python experiments/v1_20260415/eval/rmse.py \
        --experiment iss_shot_v1_20260415_dim352 --descriptor_type shot --indices 0 10 20 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from gluefactory.models import get_model
from gluefactory.utils.tensor import batch_to_device


LATERAL_MM = 0.056
TRANSPORT_MM = 0.056
VERTICAL_MM = 0.0085


def _get_dataset_and_collate(descriptor_type: str, split: str):
    if descriptor_type == "fpfh":
        from gluefactory.datasets.mitsubishi_v1_20260415_iss_fpfh_dataset import (
            MitsubishiV1ISSFPFHDataset as Cls, v1_iss_fpfh_collate_fn as fn,
            PAD_H, PAD_W)
    else:
        from gluefactory.datasets.mitsubishi_v1_20260415_iss_shot_dataset import (
            MitsubishiV1ISSSHOTDataset as Cls, v1_iss_shot_collate_fn as fn,
            PAD_H, PAD_W)
    return Cls(split=split), fn, PAD_H, PAD_W


def load_depth_raw(path, pad_h, pad_w):
    img = np.array(Image.open(str(path)))
    if img.dtype == np.uint8:
        img = img.astype(np.uint16) * 257
    out = np.zeros((pad_h, pad_w), dtype=img.dtype)
    out[:img.shape[0], :img.shape[1]] = img
    return out


def pixel_to_3d_mm(keypoints_2d, depth_map_raw):
    N = keypoints_2d.shape[0]
    points_3d = np.zeros((N, 3), dtype=np.float64)
    valid_mask = np.zeros(N, dtype=bool)
    H, W = depth_map_raw.shape
    for i in range(N):
        u = int(np.clip(keypoints_2d[i, 0], 0, W - 1))
        v = int(np.clip(keypoints_2d[i, 1], 0, H - 1))
        d = float(depth_map_raw[v, u])
        if d <= 0:
            continue
        points_3d[i] = [u * LATERAL_MM, v * TRANSPORT_MM, d * VERTICAL_MM]
        valid_mask[i] = True
    return points_3d, valid_mask


def sample_point_cloud(depth_raw, n_pts=30000, seed=42):
    rng = np.random.RandomState(seed)
    vs, us = np.where(depth_raw > 0)
    n_valid = len(vs)
    if n_valid == 0:
        return np.zeros((0, 3), dtype=np.float64)
    idx = rng.choice(n_valid, min(n_pts, n_valid), replace=False)
    us_s = us[idx].astype(np.float64)
    vs_s = vs[idx].astype(np.float64)
    d_raw = depth_raw[vs[idx], us[idx]].astype(np.float64)
    return np.stack([us_s * LATERAL_MM, vs_s * TRANSPORT_MM,
                     d_raw * VERTICAL_MM], axis=1)


def rigid_transform_svd(P_src, P_dst):
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


def ransac_rigid(src, dst, n_iter=1000, inlier_th=5.0):
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
            best_inl = inl
            best_R, best_t = R, t
    if best_inl.sum() >= 3:
        best_R, best_t = rigid_transform_svd(src[best_inl], dst[best_inl])
    return best_R, best_t, best_inl


def bilinear_depth(depth_raw, u, v):
    """Bilinear interpolated depth at subpixel (u, v).
    인접 4픽셀 중 하나라도 0(배경)이면 silhouette boundary 로 간주하고 reject.
    Returns (depth, valid)."""
    H, W = depth_raw.shape
    if not (0 <= u <= W - 1 and 0 <= v <= H - 1):
        return 0.0, False
    u0 = int(np.floor(u))
    v0 = int(np.floor(v))
    u1 = min(u0 + 1, W - 1)
    v1 = min(v0 + 1, H - 1)
    a = u - u0
    b = v - v0
    d00 = float(depth_raw[v0, u0])
    d01 = float(depth_raw[v0, u1])
    d10 = float(depth_raw[v1, u0])
    d11 = float(depth_raw[v1, u1])
    if d00 <= 0 or d01 <= 0 or d10 <= 0 or d11 <= 0:
        return 0.0, False
    d = ((1 - a) * (1 - b) * d00 + a * (1 - b) * d01 +
         (1 - a) * b * d10 + a * b * d11)
    return d, True


def compute_gt_transform(csv_path, depth0_raw, depth1_raw,
                         ransac_iter=1000, inlier_th=5.0):
    """GT CSV 비-occluded 대응점 → subpixel mm 3D → RANSAC SVD 로 T_gt(input→master).
    좌표 절단 제거 + bilinear depth + outlier rejection (모델 추정과 동일 algorithm)."""
    df = pd.read_csv(csv_path)
    valid = df["occluded"] == False  # noqa: E712
    master_xy = df.loc[valid, ["master_x", "master_y"]].values.astype(np.float64)
    input_xy = df.loc[valid, ["input_x", "input_y"]].values.astype(np.float64)

    pts_master, pts_input = [], []
    for i in range(len(master_xy)):
        mx, my = master_xy[i, 0], master_xy[i, 1]
        ix, iy = input_xy[i, 0], input_xy[i, 1]
        d0, ok0 = bilinear_depth(depth0_raw, mx, my)
        d1, ok1 = bilinear_depth(depth1_raw, ix, iy)
        if not (ok0 and ok1):
            continue
        pts_master.append([mx * LATERAL_MM, my * TRANSPORT_MM, d0 * VERTICAL_MM])
        pts_input.append([ix * LATERAL_MM, iy * TRANSPORT_MM, d1 * VERTICAL_MM])

    pts_master = np.array(pts_master)
    pts_input = np.array(pts_input)
    n_pts = len(pts_master)
    if n_pts < 3:
        return np.eye(3), np.zeros(3), n_pts
    R_gt, t_gt, _ = ransac_rigid(pts_input, pts_master,
                                 n_iter=ransac_iter, inlier_th=inlier_th)
    return R_gt, t_gt, n_pts


def compute_transform_rmse(P_src, R_est, t_est, R_gt, t_gt):
    P_est = (R_est @ P_src.T).T + t_est
    P_gt = (R_gt @ P_src.T).T + t_gt
    return float(np.sqrt(np.mean(np.sum((P_est - P_gt) ** 2, axis=1))))


# ─── Model / inference ────────────────────────

def load_model(checkpoint_path, device, config_path=None):
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "conf" in cp:
        conf = OmegaConf.create(cp["conf"])
    elif config_path:
        conf = OmegaConf.load(config_path)
    else:
        raise RuntimeError("checkpoint has no 'conf' and --config not given")
    model = get_model(conf.model.name)(conf.model).to(device)
    miss, unexp = model.load_state_dict(cp["model"], strict=False)
    if miss or unexp:
        print(f"[warn] load_state_dict: missing={len(miss)} unexpected={len(unexp)}")
    model.eval()
    print(f"Loaded: {checkpoint_path} (epoch {cp.get('epoch', '?')})")
    return model


def extract_matches(pred, batch_idx, conf_th=0.0):
    kp0 = pred["keypoints0"][batch_idx].cpu().numpy()
    kp1 = pred["keypoints1"][batch_idx].cpu().numpy()
    m0 = pred["matches0"][batch_idx].cpu().numpy()
    valid = (m0 > -1) & (m0 < kp1.shape[0])
    if "matching_scores0" in pred:
        valid &= (pred["matching_scores0"][batch_idx].cpu().numpy() > conf_th)
    return kp0[valid], kp1[m0[valid]], int(valid.sum())


def swap_batch(batch):
    swapped = {
        "view0": batch["view1"],
        "view1": batch["view0"],
    }
    if "gt_matches" in batch:
        swapped["gt_matches"] = batch["gt_matches"]
    return swapped


# ─── Per-pair evaluation ──────────────────────

@torch.no_grad()
def process_pair(model, dataset, idx, collate_fn, pad_h, pad_w,
                 args, device):
    sample = dataset[idx]
    master_path = sample["master_path"]
    input_path = sample["input_path"]
    csv_path = sample.get("csv_path", None)

    if csv_path is None or not Path(csv_path).exists():
        print(f"  [{idx}] No CSV. Skip.")
        return None

    depth0_raw = load_depth_raw(master_path, pad_h, pad_w)
    depth1_raw = load_depth_raw(input_path, pad_h, pad_w)

    R_gt_fwd, t_gt_fwd, n_gt = compute_gt_transform(
        csv_path, depth0_raw, depth1_raw)
    if n_gt < 3:
        print(f"  [{idx}] Not enough GT correspondences ({n_gt}). Skip.")
        return None

    # Forward: view0=master, view1=input → RANSAC(input→master)
    batch_fwd = collate_fn([sample])
    batch_fwd = batch_to_device(batch_fwd, device)
    pred_fwd = model(batch_fwd)
    mkp0_f, mkp1_f, n_match_fwd = extract_matches(pred_fwd, 0, args.conf_th)
    pts0_f, vm0_f = pixel_to_3d_mm(mkp0_f, depth0_raw)
    pts1_f, vm1_f = pixel_to_3d_mm(mkp1_f, depth1_raw)
    ok_f = vm0_f & vm1_f
    n_3d_fwd = int(ok_f.sum())
    if n_3d_fwd < 3:
        print(f"  [{idx}] Forward: not enough 3D points ({n_3d_fwd}). Skip.")
        return None
    R_est_fwd, t_est_fwd, inl_fwd = ransac_rigid(
        pts1_f[ok_f], pts0_f[ok_f],
        n_iter=args.ransac_iter, inlier_th=args.inlier_th)
    n_inl_fwd = int(inl_fwd.sum())
    P_input = sample_point_cloud(depth1_raw, args.n_sample_pts)
    rmse_fwd = compute_transform_rmse(
        P_input, R_est_fwd, t_est_fwd, R_gt_fwd, t_gt_fwd)

    # Reverse: view0=input, view1=master → RANSAC(master→input)
    batch_rev = swap_batch(batch_fwd)
    pred_rev = model(batch_rev)
    mkp0_r, mkp1_r, n_match_rev = extract_matches(pred_rev, 0, args.conf_th)
    pts0_r, vm0_r = pixel_to_3d_mm(mkp0_r, depth1_raw)
    pts1_r, vm1_r = pixel_to_3d_mm(mkp1_r, depth0_raw)
    ok_r = vm0_r & vm1_r
    n_3d_rev = int(ok_r.sum())
    if n_3d_rev < 3:
        print(f"  [{idx}] Reverse: not enough 3D points ({n_3d_rev}). Skip.")
        return None
    R_est_rev, t_est_rev, inl_rev = ransac_rigid(
        pts1_r[ok_r], pts0_r[ok_r],
        n_iter=args.ransac_iter, inlier_th=args.inlier_th)
    n_inl_rev = int(inl_rev.sum())

    R_gt_rev = R_gt_fwd.T
    t_gt_rev = -R_gt_fwd.T @ t_gt_fwd
    P_master = sample_point_cloud(depth0_raw, args.n_sample_pts)
    rmse_rev = compute_transform_rmse(
        P_master, R_est_rev, t_est_rev, R_gt_rev, t_gt_rev)

    rmse_bi = (rmse_fwd + rmse_rev) / 2.0
    print(f"  [{idx}] fwd={rmse_fwd:.3f} rev={rmse_rev:.3f} bi={rmse_bi:.3f} mm  "
          f"(matches fwd/rev={n_match_fwd}/{n_match_rev}, "
          f"inliers fwd/rev={n_inl_fwd}/{n_inl_rev})")

    return {
        "idx": idx,
        "master": Path(master_path).name,
        "input": Path(input_path).name,
        "n_gt_pts": n_gt,
        "n_matches_fwd": n_match_fwd,
        "n_matches_rev": n_match_rev,
        "n_3d_fwd": n_3d_fwd,
        "n_3d_rev": n_3d_rev,
        "n_inliers_fwd": n_inl_fwd,
        "n_inliers_rev": n_inl_rev,
        "rmse_fwd": float(rmse_fwd),
        "rmse_rev": float(rmse_rev),
        "rmse_bi": float(rmse_bi),
    }


# ─── Main ─────────────────────────────────────

def evaluate(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    cp_path = (args.checkpoint
               or f"outputs/training/{args.experiment}/checkpoint_best.tar")
    model = load_model(cp_path, device, config_path=args.config)

    dataset, collate_fn, PAD_H, PAD_W = _get_dataset_and_collate(
        args.descriptor_type, args.split)
    print(f"{args.split}: {len(dataset)} pairs ({args.descriptor_type})")
    print(f"RANSAC: inlier_th={args.inlier_th}mm, iter={args.ransac_iter}")

    if args.indices is not None:
        indices = list(args.indices)
    else:
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(
            len(dataset), min(args.num_samples, len(dataset)),
            replace=False).tolist())
    print(f"Evaluating {len(indices)} pairs: {indices}")

    records = []
    for idx in tqdm(indices, desc="Eval"):
        rec = process_pair(model, dataset, idx, collate_fn,
                           PAD_H, PAD_W, args, device)
        if rec is not None:
            records.append(rec)

    if not records:
        print("No valid pairs.")
        return

    df = pd.DataFrame(records)
    print(f"\n=== Summary over {len(df)} pairs ===")
    for col in ("rmse_fwd", "rmse_rev", "rmse_bi"):
        a = df[col].to_numpy()
        print(f"  {col:8s}  mean={a.mean():.3f}  median={np.median(a):.3f}"
              f"  max={a.max():.3f} mm")

    exp_name = Path(cp_path).parent.name
    out_dir = Path(args.output_dir
                   or f"experiments/v1_20260415/results/{exp_name}/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    cp_stem = Path(cp_path).stem
    csv_path = out_dir / f"rmse_{args.split}_{cp_stem}.csv"

    summary_rows = []
    for stat_name, fn in (("mean", np.mean), ("median", np.median),
                           ("max", np.max)):
        row = {c: "" for c in df.columns}
        row["idx"] = stat_name
        for c in ("rmse_fwd", "rmse_rev", "rmse_bi"):
            row[c] = float(fn(df[c].to_numpy()))
        summary_rows.append(row)
    df_out = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved per-pair + summary → {csv_path}")


def main():
    p = argparse.ArgumentParser(
        description="v1_20260415 ISS+FPFH/SHOT+LG Registration RMSE"
                    " (bidirectional, mm)")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--experiment", type=str, default="iss_fpfh_v1_20260415")
    p.add_argument("--descriptor_type", type=str, default="fpfh",
                    choices=["fpfh", "shot"])
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="test",
                    choices=["train", "val", "test"])
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--indices", type=int, nargs="*", default=None)
    p.add_argument("--conf_th", type=float, default=0.0)
    p.add_argument("--inlier_th", type=float, default=5.0,
                    help="RANSAC inlier threshold (mm)")
    p.add_argument("--ransac_iter", type=int, default=1000)
    p.add_argument("--n_sample_pts", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
