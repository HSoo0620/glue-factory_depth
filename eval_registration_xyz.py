"""
Registration 평가 — (X, Y, Z) mm 좌표계 버전.

grid 좌표 (u*0.05, v*0.05, depth) 대신 카메라 intrinsics 기반
실제 3D 좌표 (X, Y, Z) mm 사용. RMSE가 순수 mm 단위.

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth_mm

사용법:
    # SHOT
    python eval_registration_xyz.py --experiment 0407_resample2_iss_shot352_lg --descriptor shot --indices 0 10 50 90
    # RoPS
    python eval_registration_xyz.py --experiment 0408_resample2_iss_rops135_lg --descriptor rops --indices 0 10 50 90
    # FPFH
    python eval_registration_xyz.py --experiment 0406_resample2_iss_fpfh_xyz_lg --descriptor fpfh --indices 0 10 50 90
"""

import argparse
import torch
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.utils.tensor import batch_to_device

# ─── 상수 ──────────────────────────────────────
CROP_X0    = 1129
CROP_Y0    = 1081
CROP_SIZE  = 3502
CLIP_START = 0.1
CLIP_END   = 1000.0
FX         = 8001.39
FY         = 8001.39
CX_ORIG    = 2880.5
CY_ORIG    = 2880.5


# ─── Core math (XYZ mm) ──────────────────────

def pixel_to_xyz(keypoints_2d, depth_map_raw):
    """2D keypoint + raw depth -> (X, Y, Z) mm via camera intrinsics.
    keypoints_2d: (N, 2) 원본 해상도(5761) 기준
    depth_map_raw: (H, W) uint16
    """
    N = keypoints_2d.shape[0]
    points_3d = np.zeros((N, 3), dtype=np.float64)
    valid_mask = np.zeros(N, dtype=bool)
    H, W = depth_map_raw.shape
    for i in range(N):
        u = int(np.clip(keypoints_2d[i, 0], 0, W - 1))
        v = int(np.clip(keypoints_2d[i, 1], 0, H - 1))
        d_raw = float(depth_map_raw[v, u])
        if d_raw <= 0:
            continue
        Z = CLIP_START + (d_raw / 65535.0) * (CLIP_END - CLIP_START)
        X = (u - CX_ORIG) * Z / FX
        Y = (v - CY_ORIG) * Z / FY
        points_3d[i] = [X, Y, Z]
        valid_mask[i] = True
    return points_3d, valid_mask


def rigid_transform_svd(P_src, P_dst):
    """SVD로 rigid transform 추정: P_dst ≈ R @ P_src + t"""
    c_src = P_src.mean(axis=0)
    c_dst = P_dst.mean(axis=0)
    H = (P_src - c_src).T @ (P_dst - c_dst)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c_dst - R @ c_src
    return R, t


def compute_gt_transform(gt_csv_path, depth0_raw, depth1_raw):
    """GT CSV 비-occluded 대응점으로 T_gt(input->master) SVD 추정 (XYZ mm).
    Returns: R_gt (3,3), t_gt (3,), n_gt_used (int)
    """
    df = pd.read_csv(gt_csv_path)
    valid = df["occluded"] == False
    master_xy = df.loc[valid, ["master_x", "master_y"]].values.astype(np.float64)
    input_xy  = df.loc[valid, ["input_x",  "input_y" ]].values.astype(np.float64)

    H0, W0 = depth0_raw.shape
    H1, W1 = depth1_raw.shape

    pts_master, pts_input = [], []
    for i in range(len(master_xy)):
        mu = int(np.clip(master_xy[i, 0], 0, W0 - 1))
        mv = int(np.clip(master_xy[i, 1], 0, H0 - 1))
        iu = int(np.clip(input_xy[i, 0],  0, W1 - 1))
        iv = int(np.clip(input_xy[i, 1],  0, H1 - 1))

        d0, d1 = float(depth0_raw[mv, mu]), float(depth1_raw[iv, iu])
        if d0 <= 0 or d1 <= 0:
            continue

        Z0 = CLIP_START + (d0 / 65535.0) * (CLIP_END - CLIP_START)
        Z1 = CLIP_START + (d1 / 65535.0) * (CLIP_END - CLIP_START)
        pts_master.append([(mu - CX_ORIG) * Z0 / FX, (mv - CY_ORIG) * Z0 / FY, Z0])
        pts_input.append( [(iu - CX_ORIG) * Z1 / FX, (iv - CY_ORIG) * Z1 / FY, Z1])

    pts_master = np.array(pts_master)
    pts_input  = np.array(pts_input)
    n_pts = len(pts_master)

    if n_pts < 3:
        return np.eye(3), np.zeros(3), n_pts

    R_gt, t_gt = rigid_transform_svd(pts_input, pts_master)
    return R_gt, t_gt, n_pts


def ransac_rigid(src, dst, n_iter=1000, inlier_th=5.0):
    """Custom RANSAC + SVD rigid transform. src -> dst."""
    N = src.shape[0]
    if N < 3:
        return np.eye(3), np.zeros(3), np.zeros(N, dtype=bool)

    best_inliers = np.zeros(N, dtype=bool)
    best_R, best_t = np.eye(3), np.zeros(3)

    for _ in range(n_iter):
        idx = np.random.choice(N, 3, replace=False)
        try:
            R, t = rigid_transform_svd(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        errors = np.linalg.norm((R @ src.T).T + t - dst, axis=1)
        inliers = errors < inlier_th
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_R, best_t = R, t

    if best_inliers.sum() >= 3:
        best_R, best_t = rigid_transform_svd(src[best_inliers], dst[best_inliers])

    return best_R, best_t, best_inliers


def sample_point_cloud_xyz(depth_raw, n_pts=30000, seed=42):
    """Depth map에서 유효 픽셀 n_pts개 균일 샘플링 -> (X, Y, Z) mm."""
    rng = np.random.RandomState(seed)
    vs, us = np.where(depth_raw > 0)
    n_valid = len(vs)
    if n_valid == 0:
        return np.zeros((0, 3), dtype=np.float64)
    idx = rng.choice(n_valid, min(n_pts, n_valid), replace=False)
    us_s = us[idx].astype(np.float64)
    vs_s = vs[idx].astype(np.float64)
    d_raw = depth_raw[vs[idx], us[idx]].astype(np.float64)
    Z = CLIP_START + (d_raw / 65535.0) * (CLIP_END - CLIP_START)
    X = (us_s - CX_ORIG) * Z / FX
    Y = (vs_s - CY_ORIG) * Z / FY
    return np.stack([X, Y, Z], axis=1)


def compute_transform_rmse(P_src, R_est, t_est, R_gt, t_gt):
    """T_est vs T_gt RMSE (mm)."""
    P_est = (R_est @ P_src.T).T + t_est
    P_gt  = (R_gt  @ P_src.T).T + t_gt
    return float(np.sqrt(np.mean(np.sum((P_est - P_gt) ** 2, axis=1))))


# ─── Model & inference ─────────────────────────

def load_model(checkpoint_path, device):
    cp = torch.load(checkpoint_path, map_location="cpu")
    conf = OmegaConf.create(cp["conf"])
    model = get_model(conf.model.name)(conf.model).to(device)
    model.load_state_dict(cp["model"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {cp['epoch']})")
    return model, conf


@torch.no_grad()
def run_inference(model, batch, device):
    batch = batch_to_device(batch, device)
    return model(batch), batch


def kpts_to_orig(kpts, image_size):
    """crop+resize 좌표(1751px) -> 원본(5761px)."""
    scale = CROP_SIZE / image_size
    kpts_orig = kpts * scale
    kpts_orig[:, 0] += CROP_X0
    kpts_orig[:, 1] += CROP_Y0
    return kpts_orig


def extract_matches(pred, batch_idx, conf_th=0.0):
    kp0 = pred["keypoints0"][batch_idx].cpu().numpy()
    kp1 = pred["keypoints1"][batch_idx].cpu().numpy()
    m0  = pred["matches0"][batch_idx].cpu().numpy()
    scores = pred["matching_scores0"][batch_idx].cpu().numpy()
    valid = (m0 > -1) & (scores > conf_th)
    return kp0[valid], kp1[m0[valid]], int(valid.sum())


def swap_batch(batch):
    return {
        "view0": batch["view1"],
        "view1": batch["view0"],
        "gt_matches": batch.get("gt_matches"),
        "csv_path": batch.get("csv_path"),
        "master_path": batch.get("master_path"),
        "input_path": batch.get("input_path"),
    }


# ─── Dataset loader ──────────────────────────

def get_dataset_and_collate(descriptor):
    if descriptor == "rops":
        from gluefactory.datasets.mitsubishi_resample2_iss_rops_dataset import (
            MitsubishiResample2ISSRoPSDataset as DS,
            resample2_iss_rops_collate_fn as collate_fn,
        )
    elif descriptor == "fpfh":
        from gluefactory.datasets.mitsubishi_resample2_iss_fpfh_dataset import (
            MitsubishiResample2ISSFPFHDataset as DS,
            resample2_iss_fpfh_collate_fn as collate_fn,
        )
    else:
        from gluefactory.datasets.mitsubishi_resample2_iss_shot_dataset import (
            MitsubishiResample2ISSSHOTDataset as DS,
            resample2_iss_shot_collate_fn as collate_fn,
        )
    return DS, collate_fn


# ─── Per-pair evaluation ──────────────────────

def evaluate_pair(model, dataset, collate_fn, idx, combo_row,
                  base_dir, device, image_size, args):
    """단일 pair 양방향 평가 (XYZ mm)."""
    master_fname = Path(combo_row["master_path"]).name
    input_fname  = Path(combo_row["input_path"]).name
    depth0_raw = cv2.imread(
        str(base_dir / "dataset_resample_2" / master_fname), cv2.IMREAD_UNCHANGED)
    depth1_raw = cv2.imread(
        str(base_dir / "dataset_resample_2" / input_fname), cv2.IMREAD_UNCHANGED)
    csv_path = base_dir / combo_row["csv_path"]

    R_gt, t_gt, n_gt = compute_gt_transform(csv_path, depth0_raw, depth1_raw)
    print(f"  GT SVD: {n_gt} non-occluded pairs used")

    # ─── Forward: (master=view0, input=view1) ───
    sample = dataset[idx]
    batch_fwd = collate_fn([sample])
    pred_fwd, batch_fwd = run_inference(model, batch_fwd, device)

    mkp0_fwd, mkp1_fwd, n_match_fwd = extract_matches(pred_fwd, 0, args.conf_th)
    mkp0_orig = kpts_to_orig(mkp0_fwd, image_size)
    mkp1_orig = kpts_to_orig(mkp1_fwd, image_size)

    pts0, vm0 = pixel_to_xyz(mkp0_orig, depth0_raw)
    pts1, vm1 = pixel_to_xyz(mkp1_orig, depth1_raw)
    both = vm0 & vm1
    pts0_v, pts1_v = pts0[both], pts1[both]
    n_3d_fwd = len(pts0_v)

    if n_3d_fwd < 3:
        print(f"  Forward: not enough 3D points ({n_3d_fwd}). Skip.")
        return None

    R_est_fwd, t_est_fwd, inliers_fwd = ransac_rigid(
        pts1_v, pts0_v, n_iter=args.ransac_iter, inlier_th=args.inlier_th)
    n_inl_fwd = int(inliers_fwd.sum())

    pts1_aligned = (R_est_fwd @ pts1_v.T).T + t_est_fwd
    err_before = np.linalg.norm(pts1_v - pts0_v, axis=1).mean()
    err_after  = np.linalg.norm(pts1_aligned - pts0_v, axis=1).mean()
    print(f"  Forward: matches={n_match_fwd}, 3D={n_3d_fwd}, "
          f"inliers={n_inl_fwd}")
    print(f"  err: {err_before:.3f} -> {err_after:.3f} mm")

    P_input = sample_point_cloud_xyz(depth1_raw, args.n_sample_pts)
    rmse_fwd = compute_transform_rmse(P_input, R_est_fwd, t_est_fwd, R_gt, t_gt)
    print(f"  Transform RMSE(fwd): {rmse_fwd:.4f} mm")

    # ─── Reverse: (input=view0, master=view1) ───
    batch_rev = swap_batch(batch_fwd)
    pred_rev, batch_rev = run_inference(model, batch_rev, device)

    mkp0_rev, mkp1_rev, n_match_rev = extract_matches(pred_rev, 0, args.conf_th)
    mkp0_rev_orig = kpts_to_orig(mkp0_rev, image_size)
    mkp1_rev_orig = kpts_to_orig(mkp1_rev, image_size)

    pts0_rev, vm0r = pixel_to_xyz(mkp0_rev_orig, depth1_raw)
    pts1_rev, vm1r = pixel_to_xyz(mkp1_rev_orig, depth0_raw)
    both_r = vm0r & vm1r
    pts0_rv, pts1_rv = pts0_rev[both_r], pts1_rev[both_r]
    n_3d_rev = len(pts0_rv)

    if n_3d_rev < 3:
        print(f"  Reverse: not enough 3D points ({n_3d_rev}). Skip.")
        return None

    R_est_rev, t_est_rev, inliers_rev = ransac_rigid(
        pts1_rv, pts0_rv, n_iter=args.ransac_iter, inlier_th=args.inlier_th)
    n_inl_rev = int(inliers_rev.sum())

    R_gt_rev = R_gt.T
    t_gt_rev = -R_gt.T @ t_gt

    P_master = sample_point_cloud_xyz(depth0_raw, args.n_sample_pts)
    rmse_rev = compute_transform_rmse(P_master, R_est_rev, t_est_rev, R_gt_rev, t_gt_rev)
    print(f"  Reverse: matches={n_match_rev}, 3D={n_3d_rev}, "
          f"inliers={n_inl_rev}, RMSE={rmse_rev:.4f} mm")

    rmse_bi = (rmse_fwd + rmse_rev) / 2.0
    print(f"  Bidirectional RMSE: {rmse_bi:.4f} mm")

    return {
        "pair_idx": idx,
        "master": master_fname,
        "input": input_fname,
        "rmse_fwd": rmse_fwd,
        "rmse_rev": rmse_rev,
        "rmse_bi": rmse_bi,
        "n_matches_fwd": n_match_fwd,
        "n_matches_rev": n_match_rev,
        "n_inliers_fwd": n_inl_fwd,
        "n_inliers_rev": n_inl_rev,
        "n_gt_pts": n_gt,
    }


# ─── Main ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Registration Eval — XYZ mm coordinates (Transform RMSE)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--experiment", type=str,
                        default="0408_resample2_iss_rops135_lg")
    parser.add_argument("--descriptor", type=str, default="rops",
                        choices=["rops", "shot", "fpfh"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--image_size", type=int, default=1751)
    parser.add_argument("--conf_th", type=float, default=0.0)
    parser.add_argument("--inlier_th", type=float, default=5.0,
                        help="RANSAC inlier threshold (mm)")
    parser.add_argument("--ransac_iter", type=int, default=1000)
    parser.add_argument("--n_sample_pts", type=int, default=30000)
    args = parser.parse_args()

    output_dir = Path(
        args.output_dir if args.output_dir
        else f"results/registration/{args.experiment}_eval_xyz")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.checkpoint is None:
        from gluefactory.utils.experiments import get_best_checkpoint
        cp_path = get_best_checkpoint(args.experiment)
        print(f"Auto-loaded best checkpoint: {cp_path}")
    else:
        cp_path = args.checkpoint

    model, conf = load_model(cp_path, device)
    image_size = args.image_size

    base_dir = Path("gluefactory/datasets/mitsubishi")
    combo = pd.read_csv(base_dir / "outputs_txt" / "combination.csv")
    total = len(combo)
    val_size, test_size = 100, 100
    train_end = total - val_size - test_size
    val_end = total - test_size

    if args.split == "train":
        combo_split = combo.iloc[:train_end]
    elif args.split == "val":
        combo_split = combo.iloc[train_end:val_end]
    else:
        combo_split = combo.iloc[val_end:]
    combo_split = combo_split.reset_index(drop=True)

    DS, collate_fn = get_dataset_and_collate(args.descriptor)
    dataset = DS(split=args.split, image_size=image_size)
    print(f"[XYZ mm] {args.split} dataset: {len(dataset)} pairs ({args.descriptor})")
    print(f"RANSAC: inlier_th={args.inlier_th} mm, iter={args.ransac_iter}")

    if args.indices is not None:
        indices = args.indices
    else:
        indices = sorted(np.random.choice(
            len(dataset), min(args.num_samples, len(dataset)), replace=False))

    print(f"Testing {len(indices)} pairs: {indices}")

    results = []
    for idx in indices:
        print(f"\n--- Pair {idx} ---")
        row = combo_split.iloc[idx]
        res = evaluate_pair(
            model, dataset, collate_fn, idx, row,
            base_dir, device, image_size, args)
        if res is None:
            continue
        results.append(res)

    if results:
        df = pd.DataFrame(results)
        csv_out = output_dir / f"rmse_{args.split}_xyz.csv"
        df.to_csv(csv_out, index=False)
        print(f"\n{'='*60}")
        print(f"[XYZ mm] Results saved: {csv_out}")
        print(f"Mean Bidirectional RMSE: {df['rmse_bi'].mean():.4f} mm")
        print(df[["pair_idx", "rmse_fwd", "rmse_rev", "rmse_bi"]].to_string(index=False))

    print(f"\nDone! Output: {output_dir}/")


if __name__ == "__main__":
    main()
