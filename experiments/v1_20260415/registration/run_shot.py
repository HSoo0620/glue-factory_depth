"""v1 (2026-04-15) ISS+SHOT+LG 기반 3D Registration 테스트.

v1 데이터는 zero-pad된 full 이미지 (W=2432, H=3008) 좌표계를 사용.
Crop/Resize 없음. SHOT descriptor는 PCL 기본 unit-sphere 정규화 (별도 처리 없음).

물리 좌표계 (mm 단위):
    X = u * LATERAL_MM   (0.056 mm/pixel)
    Y = v * TRANSPORT_MM (0.056 mm/pixel)
    Z = raw_uint16 * VERTICAL_MM (0.0085 mm/count)

파이프라인:
    1. 학습된 ISS+SHOT+LG로 zero-pad depth image pair 매칭
    2. 매칭된 keypoint를 mm 좌표 (X, Y, Z)로 변환
    3. RANSAC + SVD로 rigid transformation (R, t) 추정
    4. 변환 적용 후 3-panel 정합 시각화 (Master / Input / Overlay)

사용법:
    python experiments/v1_20260415/registration/run_shot.py --experiment iss_shot_v1_20260415_dim352 --indices 0 10 20 30
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


# ─── 상수 (v1 mm 단위) ──────────────────────────────────────
LATERAL_MM = 0.056    # X = u * LATERAL_MM
TRANSPORT_MM = 0.056  # Y = v * TRANSPORT_MM
VERTICAL_MM = 0.0085  # Z = raw_uint16 * VERTICAL_MM

DEFAULT_WEIGHTS = "outputs/training/iss_shot_v1_20260415/checkpoint_best.tar"
DEFAULT_CONFIG = "gluefactory/configs/iss_shot_v1_20260415_lg.yaml"
DEFAULT_OUTPUT_DIR = "results/iss_shot_v1_20260415_registration"


# ─── Depth 로드 (PIL + zero-pad) ────────────────────────────

def load_depth_raw(path):
    """depth 이미지를 uint16로 읽어 (PAD_H, PAD_W)에 zero-pad."""
    img = np.array(Image.open(str(path)))
    if img.dtype == np.uint8:
        img = img.astype(np.uint16) * 257
    depth_raw = np.zeros((PAD_H, PAD_W), dtype=img.dtype)
    depth_raw[:img.shape[0], :img.shape[1]] = img
    return depth_raw


# ─── 2D+Depth → 3D (mm 단위) ────────────────────────────────

def pixel_to_3d_mm(keypoints_2d, depth_map_raw):
    """(u, v) zero-pad 픽셀 + depth_raw uint16 → (X, Y, Z) mm.

    X = u * LATERAL_MM
    Y = v * TRANSPORT_MM
    Z = raw_uint16 * VERTICAL_MM

    depth=0 또는 범위 밖 → valid_mask False.
    """
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


# ─── RANSAC + SVD ──────────────────────────────────────────

def estimate_rigid_svd(src, dst):
    centroid_src = src.mean(axis=0)
    centroid_dst = dst.mean(axis=0)
    H = (src - centroid_src).T @ (dst - centroid_dst)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_dst - R @ centroid_src
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
            R, t = estimate_rigid_svd(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue

        errors = np.linalg.norm((R @ src.T).T + t - dst, axis=1)
        inliers = errors < inlier_th

        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_R, best_t = R, t

    if best_inliers.sum() >= 3:
        best_R, best_t = estimate_rigid_svd(src[best_inliers], dst[best_inliers])

    return best_R, best_t, best_inliers


# ─── Model ─────────────────────────────────────────────────

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
    pred = model(batch)
    return pred, batch


# ─── Warp depth to master (mm 단위) ────────────────────────

def warp_depth_to_master(depth1_raw, R_est, t_est):
    """Input depth를 추정된 R,t로 변환하여 master 좌표계에 투영 (mm 단위)."""
    H, W = depth1_raw.shape
    vs, us = np.where(depth1_raw > 0)
    d_raw = depth1_raw[vs, us].astype(np.float64)
    d_mm = d_raw * VERTICAL_MM

    pts = np.stack([
        us.astype(np.float64) * LATERAL_MM,
        vs.astype(np.float64) * TRANSPORT_MM,
        d_mm,
    ], axis=1)

    pts_aligned = (R_est @ pts.T).T + t_est

    u0 = np.round(pts_aligned[:, 0] / LATERAL_MM).astype(np.int32)
    v0 = np.round(pts_aligned[:, 1] / TRANSPORT_MM).astype(np.int32)
    z0 = pts_aligned[:, 2]  # mm

    in_bounds = (u0 >= 0) & (u0 < W) & (v0 >= 0) & (v0 < H) & (z0 > 0)
    u0, v0, z0 = u0[in_bounds], v0[in_bounds], z0[in_bounds]

    warped = np.zeros((H, W), dtype=np.float64)
    zbuf = np.full((H, W), np.inf)
    for i in range(len(u0)):
        if z0[i] < zbuf[v0[i], u0[i]]:
            zbuf[v0[i], u0[i]] = z0[i]
            warped[v0[i], u0[i]] = z0[i]

    if warped.max() > 0:
        warped = warped / warped.max()
    return warped.astype(np.float32)


# ─── Visualization ─────────────────────────────────────────

def visualize_registration(
    depth0_raw, depth1_raw,
    pred, batch_idx, output_path,
    inlier_th=5.0, conf_th=0.0, ransac_iter=1000,
):
    """3-panel 시각화: Master / Input / Overlay after alignment."""
    kp0 = pred["keypoints0"][batch_idx].cpu().numpy()
    kp1 = pred["keypoints1"][batch_idx].cpu().numpy()
    m0 = pred["matches0"][batch_idx].cpu().numpy()

    valid = (m0 > -1) & (m0 < kp1.shape[0])
    if "matching_scores0" in pred:
        valid &= (pred["matching_scores0"][batch_idx].cpu().numpy() > conf_th)
    mkp0 = kp0[valid]
    mkp1 = kp1[m0[valid]]
    n_matches = int(valid.sum())
    print(f"  conf_th={conf_th:.2f} -> {n_matches} matches (total {int((m0 > -1).sum())})")

    # (u, v, depth_raw) → 3D mm 좌표
    pts0, vmask0 = pixel_to_3d_mm(mkp0, depth0_raw)
    pts1, vmask1 = pixel_to_3d_mm(mkp1, depth1_raw)
    both_valid = vmask0 & vmask1
    pts0_valid = pts0[both_valid]
    pts1_valid = pts1[both_valid]
    n_3d = len(pts0_valid)
    print(f"  3D correspondences: {n_3d}")

    if n_3d < 3:
        print("  Not enough 3D correspondences. Skip.")
        return

    R_est, t_est, inliers = ransac_rigid(
        pts1_valid, pts0_valid, n_iter=ransac_iter, inlier_th=inlier_th
    )
    n_inliers = int(inliers.sum())

    pts1_aligned = (R_est @ pts1_valid.T).T + t_est
    err_before = np.linalg.norm(pts1_valid - pts0_valid, axis=1).mean()
    err_after = np.linalg.norm(pts1_aligned - pts0_valid, axis=1).mean()

    # warp & overlay
    warped_input = warp_depth_to_master(depth1_raw, R_est, t_est)

    d0_vis = depth0_raw.astype(np.float32) / 65535.0
    d1_vis = depth1_raw.astype(np.float32) / 65535.0

    H, W = d0_vis.shape
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    overlay[:, :, 1] = d0_vis   # Master -> cyan (G+B)
    overlay[:, :, 2] = d0_vis
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], warped_input)  # Warped -> red (R)

    master_color = np.zeros((H, W, 3), dtype=np.float32)
    master_color[:, :, 1] = d0_vis
    master_color[:, :, 2] = d0_vis

    input_color = np.zeros((H, W, 3), dtype=np.float32)
    input_color[:, :, 0] = d1_vis

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(np.clip(master_color, 0, 1))
    axes[0].set_title("Master (cyan)", fontsize=13)
    axes[0].set_axis_off()

    axes[1].imshow(np.clip(input_color, 0, 1))
    axes[1].set_title("Input (red)", fontsize=13)
    axes[1].set_axis_off()

    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title("Overlayed", fontsize=13)
    axes[2].set_axis_off()

    fig.suptitle(
        f"[v1 ISS+SHOT+LG]  Matches: {n_matches}  |  3D: {n_3d}  |  "
        f"Inliers: {n_inliers}/{n_3d}  |  "
        f"Error(mm): {err_before:.2f} -> {err_after:.2f}",
        fontsize=11, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"  Saved: {output_path}  |  inliers={n_inliers}/{n_3d}  "
        f"err(mm)={err_before:.2f}->{err_after:.2f}"
    )


# ─── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="v1_20260415 ISS+SHOT+LG 기반 3D Registration"
    )
    parser.add_argument("--experiment", type=str, default="iss_shot_v1_20260415",
                        help="Experiment name (auto: outputs/training/{name}/checkpoint_best.tar)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Checkpoint path (.tar), overrides --experiment")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="Fallback config path if checkpoint has no 'conf'")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--n_pairs", type=int, default=20,
                        help="랜덤 페어 수 (--indices 없을 때 사용)")
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="명시적 페어 인덱스 (--n_pairs 무시)")
    parser.add_argument("--conf_th", type=float, default=0.0,
                        help="매칭 신뢰도 임계값")
    parser.add_argument("--inlier_th", type=float, default=5.0,
                        help="RANSAC inlier 임계값 (mm)")
    parser.add_argument("--ransac_iter", type=int, default=1000)
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
    print(
        f"{args.split} dataset: {len(dataset)} pairs "
        f"(pad_w={PAD_W}, pad_h={PAD_H})"
    )

    if args.indices is not None and len(args.indices) > 0:
        indices = args.indices
    else:
        rng = np.random.default_rng(args.seed)
        n = min(args.n_pairs, len(dataset))
        indices = sorted(rng.choice(len(dataset), n, replace=False).tolist())

    print(f"Testing {len(indices)} pairs: {indices}")

    for idx in indices:
        print(f"\n--- Pair {idx} ---")
        sample = dataset[idx]
        master_path = sample["master_path"]
        input_path = sample["input_path"]

        # depth 원본 로드 (PIL + zero-pad)
        depth0_raw = load_depth_raw(master_path)
        depth1_raw = load_depth_raw(input_path)

        # 모델 추론 (collate는 master_path/input_path 제외)
        batch = v1_iss_shot_collate_fn([sample])
        pred, batch = run_inference(model, batch, device)

        visualize_registration(
            depth0_raw, depth1_raw,
            pred, 0,
            output_dir / f"reg_{args.split}_{idx:05d}.png",
            inlier_th=args.inlier_th,
            conf_th=args.conf_th,
            ransac_iter=args.ransac_iter,
        )

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
