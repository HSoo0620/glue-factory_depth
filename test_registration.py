"""
LightGlue 매칭 결과를 이용한 3D 정합(Registration) 시각화.

파이프라인:
    1. SuperPoint + LightGlue로 매칭
    2. 매칭된 2D keypoint + depth → 3D 대응점 생성
    3. RANSAC + SVD로 rigid transformation (R, t) 추정
    4. 변환 적용 후 정합 결과 시각화

사용법:

    # 특정 인덱스
    python test_registration.py \
        --checkpoint outputs/training/Depth_only_sp_pt/checkpoint_best.tar \
        --indices 0 10 50 90 \
        --output_dir results/registration/Depth_only_sp_pt \
        --conf_th 0.1 \
        --inlier_th 3.0 


    # 커스텀 이미지
    python test_registration.py \
        --checkpoint outputs/training/Depth_only_sp/checkpoint_best.tar \
        --img0 /path/to/master.png --img1 /path/to/input.png \
        --calib0 /path/to/calib_master.ini --calib1 /path/to/calib_input.ini
"""

import argparse
import configparser
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.datasets.mitsubishi_depth_dataset import (
    MitsubishiDepthDataset,
    mitsubishi_collate_fn,
)
from gluefactory.utils.tensor import batch_to_device


# ─── Calibration ───────────────────────────────────────────

def parse_calib(calib_path):
    """calib_XXXX.ini 파싱 → K, R, t"""
    cp = configparser.ConfigParser()
    cp.read(str(calib_path))
    sec = "camera_0"

    k_str = cp.get(sec, "k_matrix").split(":")[-1]
    k_vals = [float(v) for v in k_str.split(",")]
    K = np.array(k_vals).reshape(3, 3)

    r_str = cp.get(sec, "r_matrix").split(":")[-1]
    r_vals = [float(v) for v in r_str.split(",")]
    R = np.array(r_vals).reshape(3, 3)

    t_str = cp.get(sec, "t_vector").split(":")[-1]
    t_vals = [float(v) for v in t_str.split(",")]
    t = np.array(t_vals).reshape(3, 1)

    clip_end = float(cp.get(sec, "clip_end"))

    return K, R, t, clip_end


def get_calib_path(img_path):
    """depth_raw_XXXX.png → calib_XXXX.ini"""
    stem = Path(img_path).stem  # depth_raw_0001
    idx = stem.replace("depth_raw_", "")
    calib_name = f"calib_{idx}.ini"
    return Path(img_path).parent / calib_name


# ─── 2D+Depth → 3D ────────────────────────────────────────

def pixel_to_3d(keypoints_2d, depth_map_raw, K, clip_end):
    """
    2D keypoints + raw depth map → 3D points (camera frame)
    keypoints_2d: (N, 2) float, (x, y) format
    depth_map_raw: (H, W) uint16 or float
    Returns: points_3d (N, 3), valid_mask (N,)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    N = keypoints_2d.shape[0]

    points_3d = np.zeros((N, 3), dtype=np.float64)
    valid_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        u, v = int(keypoints_2d[i, 0]), int(keypoints_2d[i, 1])
        u = np.clip(u, 0, depth_map_raw.shape[1] - 1)
        v = np.clip(v, 0, depth_map_raw.shape[0] - 1)

        d_raw = float(depth_map_raw[v, u])
        if d_raw <= 0:
            continue

        Z = d_raw / 65535.0 * clip_end
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        points_3d[i] = [X, Y, Z]
        valid_mask[i] = True

    return points_3d, valid_mask


def transform_to_world(points_cam, R, t):
    """camera frame → world frame: P_world = R^T @ (P_cam - t)"""
    return (R.T @ (points_cam.T - t)).T


# ─── RANSAC + SVD ──────────────────────────────────────────

def estimate_rigid_svd(src, dst):
    """SVD로 rigid transformation 추정: dst = R @ src + t"""
    centroid_src = src.mean(axis=0)
    centroid_dst = dst.mean(axis=0)
    src_c = src - centroid_src
    dst_c = dst - centroid_dst

    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_dst - R @ centroid_src
    return R, t


def ransac_rigid(src, dst, n_iter=1000, inlier_th=5.0):
    """RANSAC + SVD로 rigid transformation 추정"""
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

        transformed = (R @ src.T).T + t
        errors = np.linalg.norm(transformed - dst, axis=1)
        inliers = errors < inlier_th

        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_R, best_t = R, t

    # inlier로 최종 refine
    if best_inliers.sum() >= 3:
        best_R, best_t = estimate_rigid_svd(src[best_inliers], dst[best_inliers])

    return best_R, best_t, best_inliers


# ─── Model ─────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    cp = torch.load(checkpoint_path, map_location="cpu")
    conf = OmegaConf.create(cp["conf"])
    model = get_model(conf.model.name)(conf.model).to(device)
    model.load_state_dict(cp["model"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {cp['epoch']})")
    return model


@torch.no_grad()
def run_inference(model, batch, device):
    batch = batch_to_device(batch, device)
    pred = model(batch)
    return pred, batch


# ─── Visualization ─────────────────────────────────────────

def warp_depth_to_master(depth1_raw, K0, K1, R0, t0, R1, t1, clip_end, R_est, t_est):
    """
    Input depth map을 registration 결과를 적용하여 Master 카메라 시점으로 워핑.
    Input pixel → 3D cam → world → R_est 적용 → Master cam → Master pixel
    Returns: warped depth image (float32, 0~1 normalized), same size as depth1_raw
    """
    H, W = depth1_raw.shape
    fx1, fy1, cx1, cy1 = K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]
    fx0, fy0, cx0, cy0 = K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]

    # 모든 유효 픽셀의 좌표
    vs, us = np.where(depth1_raw > 0)
    d_raw = depth1_raw[vs, us].astype(np.float64)
    Z = d_raw / 65535.0 * clip_end
    X = (us - cx1) * Z / fx1
    Y = (vs - cy1) * Z / fy1
    pts_cam1 = np.stack([X, Y, Z], axis=1)  # (N, 3)

    # camera1 → world
    pts_world = (R1.T @ (pts_cam1.T - t1)).T

    # registration 적용 (input world → master world)
    pts_aligned = (R_est @ pts_world.T).T + t_est

    # master world → master camera
    pts_cam0 = (R0 @ pts_aligned.T + t0).T

    # master camera → master pixel
    Z0 = pts_cam0[:, 2]
    valid = Z0 > 0
    u0 = (pts_cam0[valid, 0] * fx0 / Z0[valid] + cx0).astype(np.int32)
    v0 = (pts_cam0[valid, 1] * fy0 / Z0[valid] + cy0).astype(np.int32)
    z0 = Z0[valid]

    # Master 이미지 크기 기준으로 렌더링
    H0, W0 = depth1_raw.shape  # 같은 크기라고 가정
    in_bounds = (u0 >= 0) & (u0 < W0) & (v0 >= 0) & (v0 < H0)
    u0, v0, z0 = u0[in_bounds], v0[in_bounds], z0[in_bounds]

    # z-buffer로 가장 가까운 점만 렌더링
    warped = np.zeros((H0, W0), dtype=np.float64)
    zbuf = np.full((H0, W0), np.inf)
    for i in range(len(u0)):
        if z0[i] < zbuf[v0[i], u0[i]]:
            zbuf[v0[i], u0[i]] = z0[i]
            warped[v0[i], u0[i]] = z0[i]

    # normalize to 0~1
    if warped.max() > 0:
        warped = warped / warped.max()
    return warped.astype(np.float32)


def visualize_registration(
    depth0_raw, depth1_raw, K0, K1, R0, t0, R1, t1, clip_end,
    pred, batch_idx, output_path, inlier_th=5.0, conf_th=0.0,
):
    """정합 결과 시각화 (3-panel 2D depth image: Master / Input / Overlayed)"""
    kp0 = pred["keypoints0"][batch_idx].cpu().numpy()
    kp1 = pred["keypoints1"][batch_idx].cpu().numpy()
    m0 = pred["matches0"][batch_idx].cpu().numpy()
    scores = pred["matching_scores0"][batch_idx].cpu().numpy()

    valid = (m0 > -1) & (scores > conf_th)
    mkp0 = kp0[valid]
    mkp1 = kp1[m0[valid]]
    n_matches = int(valid.sum())
    print(f"  Confidence threshold: {conf_th:.2f} → {n_matches} matches (filtered from {int((m0 > -1).sum())})")

    # 매칭된 keypoint → 3D (camera frame)
    pts0_cam, vmask0 = pixel_to_3d(mkp0, depth0_raw, K0, clip_end)
    pts1_cam, vmask1 = pixel_to_3d(mkp1, depth1_raw, K1, clip_end)
    both_valid = vmask0 & vmask1

    pts0_cam = pts0_cam[both_valid]
    pts1_cam = pts1_cam[both_valid]

    # camera → world
    pts0_world = transform_to_world(pts0_cam, R0, t0)
    pts1_world = transform_to_world(pts1_cam, R1, t1)

    n_3d = len(pts0_world)
    print(f"  3D correspondences: {n_3d} (from {n_matches} 2D matches)")

    if n_3d < 3:
        print("  Not enough 3D correspondences for registration. Skip.")
        return

    # RANSAC 정합
    R_est, t_est, inliers = ransac_rigid(pts1_world, pts0_world, inlier_th=inlier_th)
    n_inliers = int(inliers.sum())

    # 정합 오차
    pts1_aligned = (R_est @ pts1_world.T).T + t_est
    errors_before = np.linalg.norm(pts1_world - pts0_world, axis=1)
    errors_after = np.linalg.norm(pts1_aligned - pts0_world, axis=1)

    # Input depth를 Master 시점으로 워핑
    warped_input = warp_depth_to_master(
        depth1_raw, K0, K1, R0, t0, R1, t1, clip_end, R_est, t_est
    )

    # depth 이미지 normalize (0~1)
    d0_vis = depth0_raw.astype(np.float32) / 65535.0
    d1_vis = depth1_raw.astype(np.float32) / 65535.0

    # Overlay: Master(cyan) + Warped Input(red)
    H, W = d0_vis.shape
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    # Master → cyan 채널
    overlay[:, :, 1] = d0_vis  # G
    overlay[:, :, 2] = d0_vis  # B  (→ cyan)
    # Warped Input → red 채널
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], warped_input)  # R

    # Master → cyan 컬러 이미지
    master_color = np.zeros((H, W, 3), dtype=np.float32)
    master_color[:, :, 1] = d0_vis  # G
    master_color[:, :, 2] = d0_vis  # B  (→ cyan)

    # Input → red 컬러 이미지
    input_color = np.zeros((H, W, 3), dtype=np.float32)
    input_color[:, :, 0] = d1_vis  # R

    # ─── 3-panel 2D 시각화: Master / Input / Overlayed ───
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(np.clip(master_color, 0, 1))
    axes[0].set_title("Master", fontsize=13)
    axes[0].set_axis_off()

    axes[1].imshow(np.clip(input_color, 0, 1))
    axes[1].set_title("Input", fontsize=13)
    axes[1].set_axis_off()

    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title("Overlayed", fontsize=13)
    axes[2].set_axis_off()

    fig.suptitle(
        f"Matches: {n_matches}  |  3D: {n_3d}  |  "
        f"Inliers: {n_inliers}/{n_3d}  |  "
        f"Error: {errors_before.mean():.2f} → {errors_after.mean():.2f}",
        fontsize=11, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {output_path}")
    print(f"  Inliers: {n_inliers}/{n_3d}, Error: {errors_before.mean():.2f} → {errors_after.mean():.2f}")


# ─── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LightGlue 매칭 기반 3D 정합 시각화")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/registration")
    parser.add_argument("--inlier_th", type=float, default=5.0, help="RANSAC inlier threshold (world coord)")
    parser.add_argument("--ransac_iter", type=int, default=1000)
    parser.add_argument("--conf_th", type=float, default=0.0, help="Matching confidence threshold (0.0~1.0)")

    # 커스텀 이미지
    parser.add_argument("--img0", type=str, default=None)
    parser.add_argument("--img1", type=str, default=None)
    parser.add_argument("--calib0", type=str, default=None)
    parser.add_argument("--calib1", type=str, default=None)

    # 데이터셋 모드
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--indices", type=int, nargs="*", default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    model = load_model(args.checkpoint, device)

    base_dir = Path("gluefactory/datasets/mitsubishi")

    # --- 커스텀 이미지 ---
    if args.img0 and args.img1:
        if not args.calib0 or not args.calib1:
            print("Error: --calib0, --calib1 필요")
            return

        K0, R0, t0, clip_end = parse_calib(args.calib0)
        K1, R1, t1, _ = parse_calib(args.calib1)

        depth0_raw = cv2.imread(args.img0, cv2.IMREAD_UNCHANGED)
        depth1_raw = cv2.imread(args.img1, cv2.IMREAD_UNCHANGED)
        d0_norm = depth0_raw.astype(np.float32) / 65535.0
        d1_norm = depth1_raw.astype(np.float32) / 65535.0

        batch = {
            "view0": {
                "image": torch.from_numpy(d0_norm).unsqueeze(0).unsqueeze(0),
                "image_size": torch.tensor([[depth0_raw.shape[0], depth0_raw.shape[1]]]),
            },
            "view1": {
                "image": torch.from_numpy(d1_norm).unsqueeze(0).unsqueeze(0),
                "image_size": torch.tensor([[depth1_raw.shape[0], depth1_raw.shape[1]]]),
            },
            "gt_matches": torch.zeros(1, 1, 4),
        }
        pred, batch = run_inference(model, batch, device)
        out_name = f"reg_{Path(args.img0).stem}__{Path(args.img1).stem}.png"
        visualize_registration(
            depth0_raw, depth1_raw, K0, K1, R0, t0, R1, t1, clip_end,
            pred, 0, output_dir / out_name, args.inlier_th, args.conf_th,
        )
        return

    # --- 데이터셋 모드 ---
    import pandas as pd
    combo = pd.read_csv(base_dir / "outputs" / "combination.csv")
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

    dataset = MitsubishiDepthDataset(split=args.split)
    print(f"{args.split} dataset: {len(dataset)} pairs")

    if args.indices is not None:
        indices = args.indices
    else:
        indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
        indices = sorted(indices)

    print(f"Testing {len(indices)} pairs: {indices}")

    for idx in indices:
        print(f"\n--- Pair {idx} ---")
        row = combo_split.iloc[idx]

        # calibration 로드
        calib0_path = get_calib_path(base_dir / row["master_path"])
        calib1_path = get_calib_path(base_dir / row["input_path"])
        K0, R0, t0, clip_end = parse_calib(calib0_path)
        K1, R1, t1, _ = parse_calib(calib1_path)

        # raw depth 로드 (uint16)
        depth0_raw = cv2.imread(str(base_dir / row["master_path"]), cv2.IMREAD_UNCHANGED)
        depth1_raw = cv2.imread(str(base_dir / row["input_path"]), cv2.IMREAD_UNCHANGED)

        # 모델 추론
        sample = dataset[idx]
        batch = mitsubishi_collate_fn([sample])
        pred, batch = run_inference(model, batch, device)

        visualize_registration(
            depth0_raw, depth1_raw, K0, K1, R0, t0, R1, t1, clip_end,
            pred, 0, output_dir / f"reg_{args.split}_{idx:05d}.png",
            args.inlier_th, args.conf_th,
        )

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
