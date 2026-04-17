"""
FPFH + LightGlue 매칭 결과를 이용한 3D 정합(Registration) 시각화.

파이프라인:
    1. SuperPoint keypoint + 온라인 FPFH descriptor 추출
    2. LightGlue matcher로 매칭
    3. 매칭된 2D keypoint + depth → 3D 대응점 생성
    4. RANSAC + SVD로 rigid transformation (R, t) 추정
    5. 변환 적용 후 정합 결과 시각화

사용법:
    # 데이터셋 기반 테스트
    python test_registration_fpfh.py \
        --checkpoint outputs/training/Depth_fpfh/checkpoint_best.tar \
        --indices 0 10 50 90 \
        --output_dir results/registration/Depth_fpfh \
        --conf_th 0.1 \
        --inlier_th 3.0

    # 커스텀 이미지
    python test_registration_fpfh.py \
        --checkpoint outputs/training/Depth_fpfh/checkpoint_best.tar \
        --img0 /path/to/master.png --img1 /path/to/input.png \
        --calib0 /path/to/calib_master.ini --calib1 /path/to/calib_input.ini
"""

import argparse
import configparser
import torch
import numpy as np
import cv2
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.models.extractors.superpoint_open import SuperPoint
from gluefactory.utils.tensor import batch_to_device
from gluefactory.datasets.mitsubishi_depth_dataset import (
    MitsubishiDepthDataset,
    mitsubishi_collate_fn,
)


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
    stem = Path(img_path).stem
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


# ─── FPFH 추출 ─────────────────────────────────────────────

def load_superpoint(device, detection_threshold=0.003, max_num_keypoints=512):
    sp_conf = {
        "descriptor_dim": 256,
        "nms_radius": 3,
        "max_num_keypoints": max_num_keypoints * 2,
        "force_num_keypoints": False,
        "detection_threshold": detection_threshold,
        "remove_borders": 4,
        "channels": [64, 64, 128, 128, 256],
        "dense_outputs": None,
        "weights": None,
    }
    sp = SuperPoint(sp_conf).to(device)
    sp.eval()
    return sp


def filter_keypoints_by_depth(keypoints, scores, depth_map, max_num_keypoints):
    """depth > 0 인 keypoint만 남기고 zero-padding"""
    N = keypoints.shape[0]
    valid_mask = np.zeros(N, dtype=bool)
    for i in range(N):
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
        x = np.clip(x, 0, depth_map.shape[1] - 1)
        y = np.clip(y, 0, depth_map.shape[0] - 1)
        valid_mask[i] = depth_map[y, x] > 0

    valid_kp = keypoints[valid_mask]
    valid_sc = scores[valid_mask]

    if len(valid_sc) > 0:
        order = np.argsort(-valid_sc)
        valid_kp = valid_kp[order]
        valid_sc = valid_sc[order]

    n_valid = min(len(valid_kp), max_num_keypoints)
    out_kp = np.zeros((max_num_keypoints, 2), dtype=np.float32)
    out_sc = np.zeros(max_num_keypoints, dtype=np.float32)
    out_kp[:n_valid] = valid_kp[:n_valid]
    out_sc[:n_valid] = valid_sc[:n_valid]
    return out_kp, out_sc, n_valid


@torch.no_grad()
def extract_fpfh_online(sp_model, depth_map, device, fpfh_radius=10.0, fpfh_max_nn=100,
                         max_num_keypoints=512):
    """온라인으로 SuperPoint keypoints + depth 필터링 + FPFH descriptors 추출"""
    h, w = depth_map.shape
    img_tensor = torch.from_numpy(depth_map).unsqueeze(0).unsqueeze(0).to(device)
    sp_pred = sp_model({"image": img_tensor, "image_size": torch.tensor([[h, w]])})

    raw_kp = sp_pred["keypoints"][0].cpu().numpy()
    raw_sc = sp_pred["keypoint_scores"][0].cpu().numpy()

    kp, sc, n_valid = filter_keypoints_by_depth(raw_kp, raw_sc, depth_map, max_num_keypoints)

    fpfh = np.zeros((max_num_keypoints, 33), dtype=np.float32)
    if n_valid >= 3:
        valid_kp = kp[:n_valid]
        points_3d = []
        for i in range(n_valid):
            x, y = int(valid_kp[i, 0]), int(valid_kp[i, 1])
            x = np.clip(x, 0, depth_map.shape[1] - 1)
            y = np.clip(y, 0, depth_map.shape[0] - 1)
            z = depth_map[y, x]
            points_3d.append([float(x), float(y), float(z * 1000)])
        points_3d = np.array(points_3d, dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius * 2, max_nn=30)
        )
        fpfh_feat = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=fpfh_max_nn),
        )
        fpfh_raw = np.array(fpfh_feat.data).T.astype(np.float32)
        norms = np.linalg.norm(fpfh_raw, axis=1, keepdims=True)
        fpfh[:n_valid] = fpfh_raw / (norms + 1e-8)

    return {
        "keypoints": torch.from_numpy(kp).unsqueeze(0).to(device),
        "keypoint_scores": torch.from_numpy(sc).unsqueeze(0).to(device),
        "descriptors": torch.from_numpy(fpfh).unsqueeze(0).to(device),
    }


# ─── Model ─────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    cp = torch.load(checkpoint_path, map_location="cpu")
    conf = OmegaConf.create(cp["conf"])
    model = get_model(conf.model.name)(conf.model).to(device)
    model.load_state_dict(cp["model"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {cp['epoch']})")
    return model


def load_depth_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    if img.max() > 255:
        img = img / 65535.0
    else:
        img = img / 255.0
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


@torch.no_grad()
def run_inference(model, sp_model, depth0, depth1, device, fpfh_radius, fpfh_max_nn,
                  max_num_keypoints=512):
    """두 depth map에 대해 온라인 FPFH + LightGlue 매칭"""
    ext0 = extract_fpfh_online(sp_model, depth0, device, fpfh_radius, fpfh_max_nn, max_num_keypoints)
    ext1 = extract_fpfh_online(sp_model, depth1, device, fpfh_radius, fpfh_max_nn, max_num_keypoints)

    h0, w0 = depth0.shape
    h1, w1 = depth1.shape

    data = {
        "view0": {"image": torch.from_numpy(depth0).unsqueeze(0).unsqueeze(0).to(device),
                   "image_size": torch.tensor([[h0, w0]])},
        "view1": {"image": torch.from_numpy(depth1).unsqueeze(0).unsqueeze(0).to(device),
                   "image_size": torch.tensor([[h1, w1]])},
    }

    pred = {
        "keypoints0": ext0["keypoints"],
        "keypoint_scores0": ext0["keypoint_scores"],
        "descriptors0": ext0["descriptors"],
        "keypoints1": ext1["keypoints"],
        "keypoint_scores1": ext1["keypoint_scores"],
        "descriptors1": ext1["descriptors"],
    }

    match_pred = model.matcher({**data, **pred})
    pred.update(match_pred)

    return pred, data


# ─── Visualization ─────────────────────────────────────────

def warp_depth_to_master(depth1_raw, K0, K1, R0, t0, R1, t1, clip_end, R_est, t_est):
    """
    Input depth map을 registration 결과를 적용하여 Master 카메라 시점으로 워핑.
    """
    H, W = depth1_raw.shape
    fx1, fy1, cx1, cy1 = K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]
    fx0, fy0, cx0, cy0 = K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]

    vs, us = np.where(depth1_raw > 0)
    d_raw = depth1_raw[vs, us].astype(np.float64)
    Z = d_raw / 65535.0 * clip_end
    X = (us - cx1) * Z / fx1
    Y = (vs - cy1) * Z / fy1
    pts_cam1 = np.stack([X, Y, Z], axis=1)

    pts_world = (R1.T @ (pts_cam1.T - t1)).T
    pts_aligned = (R_est @ pts_world.T).T + t_est
    pts_cam0 = (R0 @ pts_aligned.T + t0).T

    Z0 = pts_cam0[:, 2]
    valid = Z0 > 0
    u0 = (pts_cam0[valid, 0] * fx0 / Z0[valid] + cx0).astype(np.int32)
    v0 = (pts_cam0[valid, 1] * fy0 / Z0[valid] + cy0).astype(np.int32)
    z0 = Z0[valid]

    H0, W0 = depth1_raw.shape
    in_bounds = (u0 >= 0) & (u0 < W0) & (v0 >= 0) & (v0 < H0)
    u0, v0, z0 = u0[in_bounds], v0[in_bounds], z0[in_bounds]

    warped = np.zeros((H0, W0), dtype=np.float64)
    zbuf = np.full((H0, W0), np.inf)
    for i in range(len(u0)):
        if z0[i] < zbuf[v0[i], u0[i]]:
            zbuf[v0[i], u0[i]] = z0[i]
            warped[v0[i], u0[i]] = z0[i]

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
    overlay[:, :, 1] = d0_vis  # G
    overlay[:, :, 2] = d0_vis  # B  (→ cyan)
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
    parser = argparse.ArgumentParser(description="FPFH + LightGlue 매칭 기반 3D 정합 시각화")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/registration_fpfh")
    parser.add_argument("--inlier_th", type=float, default=5.0, help="RANSAC inlier threshold (world coord)")
    parser.add_argument("--ransac_iter", type=int, default=1000)
    parser.add_argument("--conf_th", type=float, default=0.0, help="Matching confidence threshold (0.0~1.0)")

    # FPFH 파라미터
    parser.add_argument("--fpfh_radius", type=float, default=10.0)
    parser.add_argument("--fpfh_max_nn", type=int, default=100)
    parser.add_argument("--detection_threshold", type=float, default=0.003)
    parser.add_argument("--max_num_keypoints", type=int, default=512)

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
    sp_model = load_superpoint(device, args.detection_threshold, args.max_num_keypoints)

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

        pred, data = run_inference(model, sp_model, d0_norm, d1_norm, device,
                                   args.fpfh_radius, args.fpfh_max_nn, args.max_num_keypoints)
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

    print(f"Testing {len(indices)} pairs (online FPFH): {indices}")

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

        # 모델 추론 (온라인 FPFH)
        sample = dataset[idx]
        d0 = sample["view0"]["image"].squeeze(0).numpy()
        d1 = sample["view1"]["image"].squeeze(0).numpy()
        pred, data = run_inference(model, sp_model, d0, d1, device,
                                   args.fpfh_radius, args.fpfh_max_nn, args.max_num_keypoints)

        visualize_registration(
            depth0_raw, depth1_raw, K0, K1, R0, t0, R1, t1, clip_end,
            pred, 0, output_dir / f"reg_{args.split}_{idx:05d}.png",
            args.inlier_th, args.conf_th,
        )

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
