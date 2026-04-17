"""v1 (2026-04-15) ISS+SHOT+LG Registration 평가 — Transform 기반 양방향 RMSE.

좌표계: mm. X=u*0.056, Y=v*0.056, Z=raw_uint16*0.0085.
T_gt: GT CSV 비-occluded 대응점 → mm 변환 → SVD.
RMSE: 포인트 클라우드 30K 샘플, T_est vs T_gt, 양방향(fwd+rev) 평균.
SHOT descriptor는 PCL 기본 unit-sphere 정규화 (별도 처리 없음).

사용법:
    python experiments/v1_20260415/eval/rmse_shot.py --experiment iss_shot_v1_20260415_dim352 --indices 0 10 20 30

    """

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

from gluefactory.datasets.mitsubishi_v1_20260415_iss_shot_dataset import (
    MitsubishiV1ISSSHOTDataset,
    v1_iss_shot_collate_fn,
    PAD_H,
    PAD_W,
)
from gluefactory.models import get_model
from gluefactory.utils.tensor import batch_to_device


# ─── 상수 (v1 mm 좌표계) ─────────────────────────────────────
LATERAL_MM = 0.056
TRANSPORT_MM = 0.056
VERTICAL_MM = 0.0085

DEFAULT_WEIGHTS = "outputs/training/iss_shot_v1_20260415/checkpoint_best.tar"
DEFAULT_CONFIG = "gluefactory/configs/iss_shot_v1_20260415_lg.yaml"
DEFAULT_OUTPUT_DIR = "results/iss_shot_v1_20260415_eval"


# ─── Depth 로드 ──────────────────────────────────────────────

def load_depth_raw(path):
    """16-bit zmap PNG → zero-pad (PAD_H, PAD_W) uint16."""
    img = np.array(Image.open(str(path)))
    if img.dtype == np.uint8:
        img = img.astype(np.uint16) * 257
    out = np.zeros((PAD_H, PAD_W), dtype=img.dtype)
    out[:img.shape[0], :img.shape[1]] = img
    return out


# ─── Core math ────────────────────────────────────────────────

def pixel_to_3d_mm(keypoints_2d, depth_map_raw):
    """(u, v) + depth_raw → (X, Y, Z) mm."""
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


def rigid_transform_svd(P_src, P_dst):
    """SVD rigid transform: P_dst ≈ R @ P_src + t."""
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
    valid = df["occluded"] == False
    master_xy = df.loc[valid, ["master_x", "master_y"]].values.astype(np.float64)
    input_xy = df.loc[valid, ["input_x", "input_y"]].values.astype(np.float64)

    pts_master = []
    pts_input = []
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


def sample_point_cloud(depth_raw, n_pts=30000):
    """Depth map에서 유효 픽셀 n_pts개 균일 샘플링 → mm 3D."""
    vs, us = np.where(depth_raw > 0)
    n_valid = len(vs)
    if n_valid == 0:
        return np.zeros((0, 3), dtype=np.float64)
    idx = np.random.choice(n_valid, min(n_pts, n_valid), replace=False)
    us_s = us[idx].astype(np.float64)
    vs_s = vs[idx].astype(np.float64)
    d_raw = depth_raw[vs[idx], us[idx]].astype(np.float64)
    return np.stack([us_s * LATERAL_MM, vs_s * TRANSPORT_MM, d_raw * VERTICAL_MM], axis=1)


def compute_transform_rmse(P_src, R_est, t_est, R_gt, t_gt):
    """T_est vs T_gt를 동일 소스 포인트에 적용하여 RMSE 산출 (mm)."""
    P_est = (R_est @ P_src.T).T + t_est
    P_gt = (R_gt @ P_src.T).T + t_gt
    return float(np.sqrt(np.mean(np.sum((P_est - P_gt) ** 2, axis=1))))


# ─── Model ────────────────────────────────────────────────────

def load_model(checkpoint_path, device, config_path=DEFAULT_CONFIG):
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "conf" in cp:
        conf = OmegaConf.create(cp["conf"])
    else:
        conf = OmegaConf.load(config_path)
    model = get_model(conf.model.name)(conf.model).to(device)
    miss, unexp = model.load_state_dict(cp["model"], strict=False)
    if miss or unexp:
        print(f"[warn] load_state_dict: missing={len(miss)} unexpected={len(unexp)}")
    model.eval()
    epoch = cp.get("epoch", "?")
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {epoch})")
    return model, conf


@torch.no_grad()
def run_inference(model, batch, device):
    batch = batch_to_device(batch, device)
    return model(batch), batch


def extract_matches(pred, batch_idx, conf_th=0.0):
    """매칭 pair 추출. Returns: mkp0, mkp1, n_matches."""
    kp0 = pred["keypoints0"][batch_idx].cpu().numpy()
    kp1 = pred["keypoints1"][batch_idx].cpu().numpy()
    m0 = pred["matches0"][batch_idx].cpu().numpy()
    valid = (m0 > -1) & (m0 < kp1.shape[0])
    if "matching_scores0" in pred:
        valid &= (pred["matching_scores0"][batch_idx].cpu().numpy() > conf_th)
    return kp0[valid], kp1[m0[valid]], int(valid.sum())


def swap_batch(batch):
    """view0 ↔ view1 swap for reverse direction."""
    return {
        "view0": batch["view1"],
        "view1": batch["view0"],
        "gt_matches": batch.get("gt_matches"),
    }


# ─── Overlay visualization ────────────────────────────────────

def save_overlay(depth0_raw, depth1_raw, R_est, t_est, output_path):
    """Master(cyan) + warped input(red) overlay."""
    H, W = depth0_raw.shape
    vs, us = np.where(depth1_raw > 0)
    d_raw = depth1_raw[vs, us].astype(np.float64)
    pts = np.stack([
        us.astype(np.float64) * LATERAL_MM,
        vs.astype(np.float64) * TRANSPORT_MM,
        d_raw * VERTICAL_MM,
    ], axis=1)
    pts_aligned = (R_est @ pts.T).T + t_est

    u0 = np.round(pts_aligned[:, 0] / LATERAL_MM).astype(np.int32)
    v0 = np.round(pts_aligned[:, 1] / TRANSPORT_MM).astype(np.int32)
    z0 = pts_aligned[:, 2]

    in_bounds = (u0 >= 0) & (u0 < W) & (v0 >= 0) & (v0 < H) & (z0 > 0)
    u0, v0, z0 = u0[in_bounds], v0[in_bounds], z0[in_bounds]

    warped = np.zeros((H, W), dtype=np.float64)
    zbuf = np.full((H, W), np.inf)
    for i in range(len(u0)):
        if z0[i] < zbuf[v0[i], u0[i]]:
            zbuf[v0[i], u0[i]] = z0[i]
            warped[v0[i], u0[i]] = z0[i]
    if warped.max() > 0:
        warped = (warped / warped.max()).astype(np.float32)

    d0_vis = depth0_raw.astype(np.float32) / 65535.0
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    overlay[:, :, 1] = d0_vis
    overlay[:, :, 2] = d0_vis
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], warped)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(np.clip(overlay, 0, 1))
    ax.set_axis_off()
    ax.set_title("Master(cyan) + Warped Input(red)")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Per-pair evaluation ─────────────────────────────────────

def evaluate_pair(model, dataset, idx, device, args):
    """단일 pair 양방향 RMSE 평가."""
    sample = dataset[idx]
    csv_path = sample["csv_path"]
    master_path = sample["master_path"]
    input_path = sample["input_path"]

    depth0_raw = load_depth_raw(master_path)
    depth1_raw = load_depth_raw(input_path)

    R_gt, t_gt, n_gt = compute_gt_transform(csv_path, depth0_raw, depth1_raw)
    print(f"  GT SVD: {n_gt} non-occluded pairs used")

    if n_gt < 3:
        print("  Not enough GT correspondences. Skip.")
        return None

    # ─── Forward: (master=view0, input=view1) ───
    batch_fwd = v1_iss_shot_collate_fn([sample])
    pred_fwd, batch_fwd = run_inference(model, batch_fwd, device)

    mkp0_fwd, mkp1_fwd, n_match_fwd = extract_matches(pred_fwd, 0, args.conf_th)

    pts0, vm0 = pixel_to_3d_mm(mkp0_fwd, depth0_raw)
    pts1, vm1 = pixel_to_3d_mm(mkp1_fwd, depth1_raw)
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
    err_after = np.linalg.norm(pts1_aligned - pts0_v, axis=1).mean()

    P_input = sample_point_cloud(depth1_raw, args.n_sample_pts)
    rmse_fwd = compute_transform_rmse(P_input, R_est_fwd, t_est_fwd, R_gt, t_gt)
    print(f"  Forward: matches={n_match_fwd}, 3D={n_3d_fwd}, "
          f"inliers={n_inl_fwd}, RMSE(mm)={rmse_fwd:.4f}")
    print(f"    err(mm): {err_before:.3f} -> {err_after:.3f}")

    # ─── Reverse: (input=view0, master=view1) ───
    batch_rev = swap_batch(batch_fwd)
    pred_rev, batch_rev = run_inference(model, batch_rev, device)

    mkp0_rev, mkp1_rev, n_match_rev = extract_matches(pred_rev, 0, args.conf_th)

    pts0_rev, vm0r = pixel_to_3d_mm(mkp0_rev, depth1_raw)
    pts1_rev, vm1r = pixel_to_3d_mm(mkp1_rev, depth0_raw)
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

    P_master = sample_point_cloud(depth0_raw, args.n_sample_pts)
    rmse_rev = compute_transform_rmse(P_master, R_est_rev, t_est_rev, R_gt_rev, t_gt_rev)
    print(f"  Reverse: matches={n_match_rev}, 3D={n_3d_rev}, "
          f"inliers={n_inl_rev}, RMSE(mm)={rmse_rev:.4f}")

    rmse_bi = (rmse_fwd + rmse_rev) / 2.0
    print(f"  Bidirectional RMSE(mm): {rmse_bi:.4f}")

    return {
        "pair_idx": idx,
        "master": Path(master_path).name,
        "input": Path(input_path).name,
        "rmse_fwd": rmse_fwd,
        "rmse_rev": rmse_rev,
        "rmse_bi": rmse_bi,
        "n_matches_fwd": n_match_fwd,
        "n_matches_rev": n_match_rev,
        "n_inliers_fwd": n_inl_fwd,
        "n_inliers_rev": n_inl_rev,
        "n_gt_pts": n_gt,
        "R_est_fwd": R_est_fwd,
        "t_est_fwd": t_est_fwd,
        "depth0_raw": depth0_raw,
        "depth1_raw": depth1_raw,
    }


# ─── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="v1_20260415 ISS+SHOT+LG Registration Eval (Transform RMSE, mm)"
    )
    parser.add_argument("--experiment", type=str, default="iss_shot_v1_20260415",
                        help="Experiment name (auto: outputs/training/{name}/checkpoint_best.tar)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Checkpoint path (.tar), overrides --experiment")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--n_pairs", type=int, default=20)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--conf_th", type=float, default=0.0)
    parser.add_argument("--inlier_th", type=float, default=5.0,
                        help="RANSAC inlier threshold (mm)")
    parser.add_argument("--ransac_iter", type=int, default=1000)
    parser.add_argument("--n_sample_pts", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] requested device={args.device} but CUDA unavailable, falling back to cpu")
        device = "cpu"
    else:
        device = args.device

    weights = args.weights or f"outputs/training/{args.experiment}/checkpoint_best.tar"
    model, conf = load_model(weights, device, config_path=args.config)

    dataset = MitsubishiV1ISSSHOTDataset(split=args.split)
    print(f"{args.split} dataset: {len(dataset)} pairs (pad_w={PAD_W}, pad_h={PAD_H})")
    print(f"RANSAC: inlier_th={args.inlier_th}mm, iter={args.ransac_iter}")
    print(f"Sample pts: {args.n_sample_pts}")

    if args.indices is not None and len(args.indices) > 0:
        indices = args.indices
    else:
        rng = np.random.default_rng(args.seed)
        n = min(args.n_pairs, len(dataset))
        indices = sorted(rng.choice(len(dataset), n, replace=False).tolist())

    print(f"Evaluating {len(indices)} pairs: {indices}")

    results = []
    for idx in indices:
        print(f"\n--- Pair {idx} ---")
        res = evaluate_pair(model, dataset, idx, device, args)
        if res is None:
            continue

        save_overlay(
            res["depth0_raw"], res["depth1_raw"],
            res["R_est_fwd"], res["t_est_fwd"],
            output_dir / f"overlay_{args.split}_{idx:05d}.png")
        print(f"  Saved: overlay_{args.split}_{idx:05d}.png")

        results.append({k: v for k, v in res.items()
                        if k not in ("R_est_fwd", "t_est_fwd",
                                     "depth0_raw", "depth1_raw")})

    if results:
        df = pd.DataFrame(results)
        csv_out = output_dir / f"rmse_{args.split}.csv"
        df.to_csv(csv_out, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved: {csv_out}")
        print(f"Mean Bidirectional RMSE(mm): {df['rmse_bi'].mean():.4f}")
        print(df[["pair_idx", "rmse_fwd", "rmse_rev", "rmse_bi"]].to_string(index=False))

    print(f"\nDone! Output: {output_dir}/")


if __name__ == "__main__":
    main()
