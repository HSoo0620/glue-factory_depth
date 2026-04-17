"""
학습된 SP+FPFH+LG 모델로 resample 데이터 3D Registration 테스트.

파이프라인:
    1. SuperPoint keypoint 추출 + Dense FPFH descriptor 계산
    2. 학습된 FPFH+LG로 매칭
    3. 매칭된 2D keypoint + depth → 3D 대응점 생성 (원본 5761 해상도 기준)
    4. RANSAC + SVD로 rigid transformation (R, t) 추정
    5. 변환 적용 후 정합 결과 시각화

사용법:
    python test_registration_resample_fpfh.py --experiment 0323_resample_sp_fpfh_lg --fpfh_radius 0.5 --indices 0 10 50 90
    python test_registration_resample_fpfh.py --conf_th 0.1 --inlier_th 5.0
"""

import argparse
import configparser
import torch
import numpy as np
import cv2
import open3d as o3d
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.utils.tensor import batch_to_device


# ─── Calibration ───────────────────────────────────────────

def parse_calib(calib_path):
    cp = configparser.ConfigParser()
    cp.read(str(calib_path))
    sec = "camera_0"

    k_vals = [float(v) for v in cp.get(sec, "k_matrix").split(":")[-1].split(",")]
    K = np.array(k_vals).reshape(3, 3)

    r_vals = [float(v) for v in cp.get(sec, "r_matrix").split(":")[-1].split(",")]
    R = np.array(r_vals).reshape(3, 3)

    t_vals = [float(v) for v in cp.get(sec, "t_vector").split(":")[-1].split(",")]
    t = np.array(t_vals).reshape(3, 1)

    clip_start = float(cp.get(sec, "clip_start"))
    clip_end = float(cp.get(sec, "clip_end"))
    return K, R, t, clip_start, clip_end


def get_calib_path(img_path):
    stem = Path(img_path).stem
    idx = stem.replace("depth_raw_", "")
    return Path(img_path).parent / f"calib_{idx}.ini"


# ─── SuperPoint ───────────────────────────────────────────

def load_superpoint(device, detection_threshold=0.001, max_num_keypoints=512):
    from gluefactory.models.extractors.superpoint_open import SuperPoint
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
    model = SuperPoint(sp_conf).to(device)
    model.eval()
    return model


# ─── FPFH ─────────────────────────────────────────────────

def compute_dense_fpfh(depth_real, K, fpfh_radius=1.5, fpfh_max_nn=100):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    vs, us = np.where(depth_real > 0)
    zs = depth_real[vs, us]
    xs = (us - cx) * zs / fx
    ys = (vs - cy) * zs / fy
    dense_pts = np.stack([xs, ys, zs], axis=1).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dense_pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius * 2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=fpfh_max_nn),
    )
    return pcd, np.array(fpfh.data).T.astype(np.float32)


def lookup_fpfh(keypoints, n_valid, depth_real, K, pcd, fpfh_dense, max_kp=512):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    fpfh_out = np.zeros((max_kp, 33), dtype=np.float32)

    if n_valid < 3:
        return fpfh_out

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for i in range(n_valid):
        u, v = int(keypoints[i, 0]), int(keypoints[i, 1])
        u = np.clip(u, 0, depth_real.shape[1] - 1)
        v = np.clip(v, 0, depth_real.shape[0] - 1)
        z = float(depth_real[v, u])
        if z <= 0:
            continue
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        _, idx, _ = kdtree.search_knn_vector_3d([x, y, z], 1)
        fpfh_out[i] = fpfh_dense[idx[0]]

    norms = np.linalg.norm(fpfh_out[:n_valid], axis=1, keepdims=True)
    fpfh_out[:n_valid] = fpfh_out[:n_valid] / (norms + 1e-8)
    return fpfh_out


# ─── 2D+Depth → 3D ────────────────────────────────────────

def pixel_to_3d(keypoints_2d, depth_map_raw, K, clip_start, clip_end):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    N = keypoints_2d.shape[0]

    points_3d = np.zeros((N, 3), dtype=np.float64)
    valid_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        u = int(np.clip(keypoints_2d[i, 0], 0, depth_map_raw.shape[1] - 1))
        v = int(np.clip(keypoints_2d[i, 1], 0, depth_map_raw.shape[0] - 1))

        d_raw = float(depth_map_raw[v, u])
        if d_raw <= 0:
            continue

        Z = clip_start + (d_raw / 65535.0) * (clip_end - clip_start)
        points_3d[i] = [(u - cx) * Z / fx, (v - cy) * Z / fy, Z]
        valid_mask[i] = True

    return points_3d, valid_mask


def transform_to_world(points_cam, R, t):
    return (R.T @ (points_cam.T - t)).T


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

def load_model(checkpoint_path, device):
    cp = torch.load(checkpoint_path, map_location="cpu")
    conf = OmegaConf.create(cp["conf"])
    model = get_model(conf.model.name)(conf.model).to(device)
    model.load_state_dict(cp["model"], strict=False)
    model.eval()
    epoch = cp["epoch"]
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {epoch})")
    return model, conf


@torch.no_grad()
def run_inference(model, batch, device):
    batch = batch_to_device(batch, device)
    pred = model(batch)
    return pred, batch


# ─── Visualization ─────────────────────────────────────────

def warp_depth_to_master(depth1_raw, K0, K1, R0, t0, R1, t1,
                         clip_start, clip_end, R_est, t_est):
    H, W = depth1_raw.shape
    fx1, fy1, cx1, cy1 = K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]
    fx0, fy0, cx0, cy0 = K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]

    vs, us = np.where(depth1_raw > 0)
    d_raw = depth1_raw[vs, us].astype(np.float64)
    Z = clip_start + (d_raw / 65535.0) * (clip_end - clip_start)
    pts_cam1 = np.stack([(us - cx1) * Z / fx1, (vs - cy1) * Z / fy1, Z], axis=1)

    pts_world = (R1.T @ (pts_cam1.T - t1)).T
    pts_aligned = (R_est @ pts_world.T).T + t_est
    pts_cam0 = (R0 @ pts_aligned.T + t0).T

    Z0 = pts_cam0[:, 2]
    valid = Z0 > 0
    u0 = (pts_cam0[valid, 0] * fx0 / Z0[valid] + cx0).astype(np.int32)
    v0 = (pts_cam0[valid, 1] * fy0 / Z0[valid] + cy0).astype(np.int32)
    z0 = Z0[valid]

    in_bounds = (u0 >= 0) & (u0 < W) & (v0 >= 0) & (v0 < H)
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


def visualize_registration(
    depth0_raw, depth1_raw, K0, K1, R0, t0, R1, t1,
    clip_start, clip_end,
    pred, batch_idx, output_path, inlier_th=5.0, conf_th=0.0,
    image_size=2880, orig_size=5761,
):
    kp0 = pred["keypoints0"][batch_idx].cpu().numpy()
    kp1 = pred["keypoints1"][batch_idx].cpu().numpy()
    m0 = pred["matches0"][batch_idx].cpu().numpy()
    scores = pred["matching_scores0"][batch_idx].cpu().numpy()

    valid = (m0 > -1) & (scores > conf_th)
    mkp0 = kp0[valid]
    mkp1 = kp1[m0[valid]]
    n_matches = int(valid.sum())
    print(f"  conf_th={conf_th:.2f} -> {n_matches} matches (total {int((m0 > -1).sum())})")

    scale = orig_size / image_size
    mkp0_orig = mkp0 * scale
    mkp1_orig = mkp1 * scale

    pts0_cam, vmask0 = pixel_to_3d(mkp0_orig, depth0_raw, K0, clip_start, clip_end)
    pts1_cam, vmask1 = pixel_to_3d(mkp1_orig, depth1_raw, K1, clip_start, clip_end)
    both_valid = vmask0 & vmask1
    pts0_world = transform_to_world(pts0_cam[both_valid], R0, t0)
    pts1_world = transform_to_world(pts1_cam[both_valid], R1, t1)
    n_3d = len(pts0_world)
    print(f"  3D correspondences: {n_3d}")

    if n_3d < 3:
        print("  Not enough 3D correspondences. Skip.")
        return

    R_est, t_est, inliers = ransac_rigid(pts1_world, pts0_world, inlier_th=inlier_th)
    n_inliers = int(inliers.sum())

    pts1_aligned = (R_est @ pts1_world.T).T + t_est
    err_before = np.linalg.norm(pts1_world - pts0_world, axis=1).mean()
    err_after = np.linalg.norm(pts1_aligned - pts0_world, axis=1).mean()

    warped_input = warp_depth_to_master(
        depth1_raw, K0, K1, R0, t0, R1, t1,
        clip_start, clip_end, R_est, t_est,
    )

    d0_vis = depth0_raw.astype(np.float32) / 65535.0
    d1_vis = depth1_raw.astype(np.float32) / 65535.0

    H, W = d0_vis.shape
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    overlay[:, :, 1] = d0_vis
    overlay[:, :, 2] = d0_vis
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], warped_input)

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
        f"[SP+FPFH+LG]  Matches: {n_matches}  |  3D: {n_3d}  |  "
        f"Inliers: {n_inliers}/{n_3d}  |  "
        f"Error: {err_before:.2f} -> {err_after:.2f}",
        fontsize=11, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}  |  inliers={n_inliers}/{n_3d}  err={err_before:.2f}->{err_after:.2f}")


# ─── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SP+FPFH+LG 기반 resample 3D Registration")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="0323_resample_sp_fpfh_lg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--image_size", type=int, default=2880)
    parser.add_argument("--fpfh_radius", type=float, default=0.5)
    parser.add_argument("--detection_threshold", type=float, default=0.001)
    parser.add_argument("--max_num_keypoints", type=int, default=512)
    parser.add_argument("--conf_th", type=float, default=0.0)
    parser.add_argument("--inlier_th", type=float, default=5.0)
    parser.add_argument("--ransac_iter", type=int, default=1000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir if args.output_dir else f"results/registration/{args.experiment}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.checkpoint is None:
        from gluefactory.utils.experiments import get_best_checkpoint
        cp_path = get_best_checkpoint(args.experiment)
        print(f"Auto-loaded best checkpoint: {cp_path}")
    else:
        cp_path = args.checkpoint

    model, conf = load_model(cp_path, device)
    sp_model = load_superpoint(device, args.detection_threshold, args.max_num_keypoints)
    image_size = args.image_size

    base_dir = Path("gluefactory/datasets/mitsubishi")
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

    print(f"{args.split}: {len(combo_split)} pairs (image_size={image_size}, fpfh_radius={args.fpfh_radius})")

    if args.indices is not None:
        indices = args.indices
    else:
        indices = np.random.choice(len(combo_split), min(args.num_samples, len(combo_split)), replace=False)
        indices = sorted(indices)

    print(f"Testing {len(indices)} pairs: {indices}")
    max_kp = args.max_num_keypoints

    for idx in indices:
        print(f"\n--- Pair {idx} ---")
        row = combo_split.iloc[idx]

        master_img_path = base_dir / row["master_path"]
        input_img_path = base_dir / row["input_path"]

        calib0_path = get_calib_path(master_img_path)
        calib1_path = get_calib_path(input_img_path)
        K0, R0, t0, clip_start0, clip_end0 = parse_calib(calib0_path)
        K1, R1, t1, clip_start1, clip_end1 = parse_calib(calib1_path)

        # 원본 이미지 로드
        master_raw = cv2.imread(str(master_img_path), cv2.IMREAD_UNCHANGED)
        input_raw = cv2.imread(str(input_img_path), cv2.IMREAD_UNCHANGED)
        orig_size = master_raw.shape[0]

        # 리사이즈
        master_resized = cv2.resize(master_raw, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        input_resized = cv2.resize(input_raw, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        scale = orig_size / image_size

        master_norm = master_resized.astype(np.float32) / 65535.0
        input_norm = input_resized.astype(np.float32) / 65535.0
        h, w = master_norm.shape

        # SuperPoint
        with torch.no_grad():
            sp0 = sp_model({"image": torch.from_numpy(master_norm).unsqueeze(0).unsqueeze(0).to(device),
                            "image_size": torch.tensor([[h, w]])})
            sp1 = sp_model({"image": torch.from_numpy(input_norm).unsqueeze(0).unsqueeze(0).to(device),
                            "image_size": torch.tensor([[h, w]])})

        kp0 = sp0["keypoints"][0].cpu().numpy()
        sc0 = sp0["keypoint_scores"][0].cpu().numpy()
        kp1 = sp1["keypoints"][0].cpu().numpy()
        sc1 = sp1["keypoint_scores"][0].cpu().numpy()

        # depth 필터
        from precompute_fpfh_resample import filter_keypoints_by_depth
        kp0_f, sc0_f, n0 = filter_keypoints_by_depth(kp0, sc0, master_resized, max_kp)
        kp1_f, sc1_f, n1 = filter_keypoints_by_depth(kp1, sc1, input_resized, max_kp)

        # FPFH on original resolution
        depth_real0 = master_raw.astype(np.float64) / 65535.0
        depth_real0 = clip_start0 + depth_real0 * (clip_end0 - clip_start0)
        depth_real0[master_raw == 0] = 0.0
        depth_real1 = input_raw.astype(np.float64) / 65535.0
        depth_real1 = clip_start1 + depth_real1 * (clip_end1 - clip_start1)
        depth_real1[input_raw == 0] = 0.0

        kp0_orig = kp0_f.copy()
        kp0_orig[:n0] *= scale
        kp1_orig = kp1_f.copy()
        kp1_orig[:n1] *= scale

        print(f"  Computing FPFH (radius={args.fpfh_radius})...")
        pcd0, fpfh0_dense = compute_dense_fpfh(depth_real0, K0, args.fpfh_radius)
        pcd1, fpfh1_dense = compute_dense_fpfh(depth_real1, K1, args.fpfh_radius)

        desc0 = lookup_fpfh(kp0_orig, n0, depth_real0, K0, pcd0, fpfh0_dense, max_kp)
        desc1 = lookup_fpfh(kp1_orig, n1, depth_real1, K1, pcd1, fpfh1_dense, max_kp)

        # batch
        batch = {
            "view0": {
                "image": torch.from_numpy(master_norm).unsqueeze(0).unsqueeze(0),
                "image_size": torch.tensor([[h, w]]),
                "keypoints": torch.from_numpy(kp0_f).unsqueeze(0).float(),
                "keypoint_scores": torch.from_numpy(sc0_f).unsqueeze(0).float(),
                "descriptors": torch.from_numpy(desc0).unsqueeze(0).float(),
            },
            "view1": {
                "image": torch.from_numpy(input_norm).unsqueeze(0).unsqueeze(0),
                "image_size": torch.tensor([[h, w]]),
                "keypoints": torch.from_numpy(kp1_f).unsqueeze(0).float(),
                "keypoint_scores": torch.from_numpy(sc1_f).unsqueeze(0).float(),
                "descriptors": torch.from_numpy(desc1).unsqueeze(0).float(),
            },
        }

        pred, batch = run_inference(model, batch, device)

        visualize_registration(
            master_raw, input_raw, K0, K1, R0, t0, R1, t1,
            clip_start0, clip_end0,
            pred, 0,
            output_dir / f"reg_{args.split}_{idx:05d}.png",
            args.inlier_th, args.conf_th,
            image_size=image_size, orig_size=orig_size,
        )

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
