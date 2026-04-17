"""v1 (2026-04-15) ISS+FPFH/SHOT+LG Registration 3×3 시각화.

3×3 scatter: (XY / XZ / YZ) × (Before / Estimated / GT)
+ overlay detail: 1×3 scatter (master + aligned input)

물리 좌표계 (mm): X=u*0.056, Y=v*0.056, Z=raw*0.0085.
GT: CSV 비-occluded 대응점 → mm → SVD.

사용법:
    python experiments/v1_20260415/registration/visualize.py \
        --experiment iss_fpfh_v1_20260415_norm --descriptor_type fpfh --indices 0 10 20 30
    python experiments/v1_20260415/registration/visualize.py \
        --experiment iss_shot_v1_20260415_dim352 --descriptor_type shot --indices 0 10 20 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image

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


def sample_point_cloud(depth_raw, n_pts=15000, seed=42):
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


# ─── Model helpers ────────────────────────────

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


@torch.no_grad()
def run_inference(model, batch, device):
    batch = batch_to_device(batch, device)
    return model(batch), batch


def extract_matches(pred, batch_idx, conf_th=0.0):
    kp0 = pred["keypoints0"][batch_idx].cpu().numpy()
    kp1 = pred["keypoints1"][batch_idx].cpu().numpy()
    m0 = pred["matches0"][batch_idx].cpu().numpy()
    valid = (m0 > -1) & (m0 < kp1.shape[0])
    if "matching_scores0" in pred:
        valid &= (pred["matching_scores0"][batch_idx].cpu().numpy() > conf_th)
    return kp0[valid], kp1[m0[valid]], int(valid.sum())


# ─── Visualization ────────────────────────────

def plot_alignment(pc_master, pc_input, pc_est, pc_gt, title, output_path,
                   n_inliers=0, n_matches=0, rmse_est=None):
    """3×3 scatter. 행: Before / Estimated / GT.  열: XY / XZ / YZ."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    s = 0.5
    alpha = 0.6

    labels_row = ["Before", "Estimated R,t", "GT R,t"]
    labels_col = ["Top-down (X, Y)", "Front (X, Z)", "Side (Y, Z)"]
    col_axes = [(0, 1), (0, 2), (1, 2)]

    C_MASTER = "lightskyblue"
    C_INPUT = "crimson"
    pairs = [
        (pc_master, pc_input),
        (pc_master, pc_est),
        (pc_master, pc_gt),
    ]

    for row, (pc_a, pc_b) in enumerate(pairs):
        for col, (xi, yi) in enumerate(col_axes):
            ax = axes[row, col]
            ax.scatter(pc_a[:, xi], pc_a[:, yi], s=s, c=C_MASTER,
                       alpha=alpha, label="master")
            ax.scatter(pc_b[:, xi], pc_b[:, yi], s=s, c=C_INPUT,
                       alpha=alpha, label="input")
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(labels_col[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(labels_row[row], fontsize=11, fontweight="bold")

    # Y(=v*mm) 는 이미지 좌표계(아래로 증가) → Top-down(XY)만 invert
    for row in range(3):
        axes[row, 0].invert_yaxis()

    info = title
    if n_matches > 0:
        info += f"  |  matches={n_matches}, inliers={n_inliers}"
    if rmse_est is not None:
        info += f", RMSE={rmse_est:.4f} mm"
    fig.suptitle(info, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_overlay_detail(pc_master, pc_est, title, output_path):
    """1×3 overlay scatter: master + aligned input."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    s = 0.5
    alpha = 0.6
    col_axes = [(0, 1), (0, 2), (1, 2)]
    col_labels = ["Top-down (X, Y)", "Front (X, Z)", "Side (Y, Z)"]

    for col, (xi, yi) in enumerate(col_axes):
        ax = axes[col]
        ax.scatter(pc_master[:, xi], pc_master[:, yi], s=s, c="lightskyblue",
                   alpha=alpha, label="master")
        ax.scatter(pc_est[:, xi], pc_est[:, yi], s=s, c="crimson",
                   alpha=alpha, label="aligned")
        ax.set_aspect("equal")
        ax.set_title(col_labels[col], fontsize=11)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.legend(markerscale=5, fontsize=9)

    axes[0].invert_yaxis()
    fig.suptitle(f"Overlay: {title}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─── Main ─────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="v1_20260415 ISS+FPFH/SHOT+LG Registration 3×3 시각화")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--experiment", type=str, default="iss_fpfh_v1_20260415")
    p.add_argument("--descriptor_type", type=str, default="fpfh",
                    choices=["fpfh", "shot"])
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="test",
                    choices=["train", "val", "test"])
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--indices", type=int, nargs="*", default=None)
    p.add_argument("--conf_th", type=float, default=0.0)
    p.add_argument("--inlier_th", type=float, default=5.0,
                    help="RANSAC inlier threshold (mm)")
    p.add_argument("--ransac_iter", type=int, default=1000)
    p.add_argument("--n_sample_pts", type=int, default=15000,
                    help="시각화용 포인트 샘플 수")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    cp_path = (args.checkpoint
               or f"outputs/training/{args.experiment}/checkpoint_best.tar")
    exp_name = Path(cp_path).parent.name
    output_dir = Path(args.output_dir
                       or f"experiments/v1_20260415/results/{exp_name}/registration")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(cp_path, device, config_path=args.config)
    dataset, collate_fn, PAD_H, PAD_W = _get_dataset_and_collate(
        args.descriptor_type, args.split)
    print(f"{args.split} dataset: {len(dataset)} pairs ({args.descriptor_type})")

    if args.indices is not None:
        indices = list(args.indices)
    else:
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(
            len(dataset), min(args.num_samples, len(dataset)),
            replace=False).tolist())
    print(f"Visualizing {len(indices)} pairs: {indices}")

    for idx in indices:
        print(f"\n--- Pair {idx} ---")
        sample = dataset[idx]
        master_path = sample["master_path"]
        input_path = sample["input_path"]
        csv_path = sample.get("csv_path", None)

        depth0_raw = load_depth_raw(master_path, PAD_H, PAD_W)
        depth1_raw = load_depth_raw(input_path, PAD_H, PAD_W)

        batch = collate_fn([sample])
        pred, batch = run_inference(model, batch, device)

        mkp0, mkp1, n_match = extract_matches(pred, 0, args.conf_th)
        pts0, vm0 = pixel_to_3d_mm(mkp0, depth0_raw)
        pts1, vm1 = pixel_to_3d_mm(mkp1, depth1_raw)
        both = vm0 & vm1
        pts0_v, pts1_v = pts0[both], pts1[both]
        n_3d = len(pts0_v)
        if n_3d < 3:
            print(f"  Not enough 3D matches ({n_3d}). Skip.")
            continue

        R_est, t_est, inliers = ransac_rigid(
            pts1_v, pts0_v, n_iter=args.ransac_iter, inlier_th=args.inlier_th)
        n_inl = int(inliers.sum())
        print(f"  Matches={n_match}, valid-depth={n_3d}, inliers={n_inl}")

        if csv_path and Path(csv_path).exists():
            R_gt, t_gt, n_gt = compute_gt_transform(
                csv_path, depth0_raw, depth1_raw)
            print(f"  GT SVD: {n_gt} non-occluded pairs used")
        else:
            R_gt, t_gt = np.eye(3), np.zeros(3)
            print("  No CSV → GT = identity")

        pc_master = sample_point_cloud(depth0_raw, args.n_sample_pts)
        pc_input = sample_point_cloud(depth1_raw, args.n_sample_pts)
        pc_est = (R_est @ pc_input.T).T + t_est
        pc_gt = (R_gt @ pc_input.T).T + t_gt

        P_30k = sample_point_cloud(depth1_raw, 30000, seed=123)
        rmse = compute_transform_rmse(P_30k, R_est, t_est, R_gt, t_gt)
        print(f"  RMSE(est vs GT) = {rmse:.4f} mm")

        pair_title = f"pair {idx} ({args.descriptor_type})"
        plot_alignment(
            pc_master, pc_input, pc_est, pc_gt,
            pair_title,
            output_dir / f"align_{args.split}_{idx:05d}.png",
            n_inliers=n_inl, n_matches=n_match, rmse_est=rmse)
        plot_overlay_detail(
            pc_master, pc_est, pair_title,
            output_dir / f"overlay_{args.split}_{idx:05d}.png")

    print(f"\nDone! Output: {output_dir}/")


if __name__ == "__main__":
    main()
