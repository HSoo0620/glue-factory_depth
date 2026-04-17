"""
학습된 ISS(det)+SHOT(desc)+LG 모델로 resample_2 데이터 3D Registration 테스트.
Open3D registration_ransac_based_on_correspondence 사용.

RANSAC inlier 매칭 pair 기반 RMSE 산출 (GT CSV occluded 제외).

좌표계: (u*grid_dx, v*grid_dy, depth_real) — 물리 단위
  GRID_DX = GRID_DY = 0.05 mm/pixel
  GRID_DZ = 0.02           (calib grid_dz, RMSE 계산 시 Z 스케일 보정)
  depth_real: CLIP_START + (raw/65535) * (CLIP_END - CLIP_START)

RANSAC은 GRID_DZ 미적용 좌표계(기존)로 동작.
RMSE는 GRID_DZ 적용 3D 거리로 산출, GT CSV의 occluded pair 제외.

사용법:
    python test_registration_resample2_iss_shot_3D_ransac_rmse.py --experiment 0407_resample2_iss_shot352_lg --indices 0 10 50 90
"""

import argparse
import torch
import numpy as np
import cv2
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.datasets.mitsubishi_resample2_iss_shot_dataset import (
    MitsubishiResample2ISSSHOTDataset,
    resample2_iss_shot_collate_fn,
    CROP_X0,
    CROP_Y0,
    CROP_SIZE,
)
from gluefactory.utils.tensor import batch_to_device


# ─── 상수 (resample_2 공통) ──────────────────────────────────
CLIP_START = 0.1
CLIP_END   = 1000.0
ORIG_SIZE  = 5761
GRID_DX = 0.05  # mm/pixel (calib grid_dx)
GRID_DY = 0.05  # mm/pixel (calib grid_dy)
GRID_DZ = 0.02  # depth Z 스케일 (calib grid_dz) — RMSE 계산 전용
GT_MATCH_RADIUS_ORIG = 40.0  # gt_radius(20px@1751) × scale(2.0) → 원본 5761 기준

# ─── 2D+Depth → 3D (RANSAC용, GRID_DZ 미적용) ───────────────

def pixel_to_3d(keypoints_2d, depth_map_raw):
    """2D keypoint + raw depth → 3D 좌표 (u*GRID_DX, v*GRID_DY, depth_real).

    keypoints_2d: (N, 2) — 원본 해상도(5761) 기준 좌표
    depth_map_raw: (H, W) uint16 — 원본 해상도
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

        depth_real = CLIP_START + (d_raw / 65535.0) * (CLIP_END - CLIP_START)
        points_3d[i] = [u * GRID_DX, v * GRID_DY, depth_real]
        valid_mask[i] = True

    return points_3d, valid_mask


# ─── GT occluded 셋 로드 ─────────────────────────────────────

def build_occluded_set(csv_path):
    """GT CSV에서 occluded=True인 master (x, y) 좌표를 set으로 반환."""
    df = pd.read_csv(csv_path)
    occluded = df[df["occluded"] == True]
    return set(zip(occluded["master_x"].astype(int), occluded["master_y"].astype(int)))


def classify_matches(mkp0_orig, mkp1_orig, depth0_raw, depth1_raw,
                     gt_csv_path, match_radius=GT_MATCH_RADIUS_ORIG):
    """매칭 pair 분류.

    Returns:
      categories (N,) int:
        0 = red    — 포인트 미정의 (depth=0)
        1 = blue   — 정답 pair
        2 = green  — 정답 pair & 가려짐 (occluded)
        3 = purple — 포인트 정의 & 매칭 틀림
      gt_occluded_set: RMSE용 occluded master 좌표 set
    """
    N = len(mkp0_orig)
    categories = np.full(N, 3, dtype=int)  # default: purple

    H0, W0 = depth0_raw.shape
    H1, W1 = depth1_raw.shape

    df = pd.read_csv(gt_csv_path)
    gt_master_xy = df[["master_x", "master_y"]].values.astype(np.float64)
    gt_input_xy  = df[["input_x", "input_y"]].values.astype(np.float64)
    gt_occ       = df["occluded"].values
    tree = cKDTree(gt_master_xy)

    # occluded set (RMSE용)
    occ_df = df[df["occluded"] == True]
    gt_occluded_set = set(zip(occ_df["master_x"].astype(int),
                              occ_df["master_y"].astype(int)))

    for i in range(N):
        u0 = int(np.clip(mkp0_orig[i, 0], 0, W0 - 1))
        v0 = int(np.clip(mkp0_orig[i, 1], 0, H0 - 1))
        u1 = int(np.clip(mkp1_orig[i, 0], 0, W1 - 1))
        v1 = int(np.clip(mkp1_orig[i, 1], 0, H1 - 1))

        # depth=0 → 포인트 미정의
        if depth0_raw[v0, u0] == 0 or depth1_raw[v1, u1] == 0:
            categories[i] = 0
            continue

        # GT에서 가장 가까운 master 대응점
        dist_m, idx = tree.query(mkp0_orig[i])
        if dist_m > match_radius:
            continue  # GT 범위 밖 → purple 유지

        pred_dist = np.linalg.norm(mkp1_orig[i] - gt_input_xy[idx])
        if pred_dist <= match_radius:
            categories[i] = 2 if gt_occ[idx] else 1

    return categories, gt_occluded_set


# ─── Open3D RANSAC (correspondence 기반) ─────────────────────

def ransac_rigid_o3d(src, dst, inlier_th=5.0, ransac_n=3, max_iter=100000):
    """Open3D registration_ransac_based_on_correspondence.

    src: (N, 3) — input 포인트 (pts1)
    dst: (N, 3) — master 포인트 (pts0)
    inlier_th: correspondence 허용 거리
    """
    N = src.shape[0]
    if N < ransac_n:
        return np.eye(3), np.zeros(3), np.zeros(N, dtype=bool)

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src)

    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(dst)

    corres = o3d.utility.Vector2iVector(
        np.column_stack([np.arange(N), np.arange(N)])
    )

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        pcd_src, pcd_dst, corres,
        max_correspondence_distance=inlier_th,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=ransac_n,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iter, 0.999),
    )

    T = np.asarray(result.transformation)
    R = T[:3, :3]
    t = T[:3, 3]

    corres_set = np.asarray(result.correspondence_set)
    inlier_mask = np.zeros(N, dtype=bool)
    if len(corres_set) > 0:
        inlier_mask[corres_set[:, 0]] = True

    return R, t, inlier_mask


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


# ─── 좌표 변환: crop+resize → 원본 5761 ───────────────────

def kpts_to_orig(kpts, image_size):
    scale = CROP_SIZE / image_size
    kpts_orig = kpts * scale
    kpts_orig = kpts_orig.copy()
    kpts_orig[:, 0] += CROP_X0
    kpts_orig[:, 1] += CROP_Y0
    return kpts_orig


# ─── Visualization ─────────────────────────────────────────

def warp_depth_to_master(depth1_raw, R_est, t_est, out_shape=None):
    """Input depth를 추정된 R,t로 변환하여 master 좌표계에 투영.

    R_est, t_est는 (u*dx, v*dy, depth_real) 좌표계에서 추정된 값.
    """
    H_out, W_out = out_shape if out_shape is not None else depth1_raw.shape
    vs, us = np.where(depth1_raw > 0)
    d_raw = depth1_raw[vs, us].astype(np.float64)
    depth_real = CLIP_START + (d_raw / 65535.0) * (CLIP_END - CLIP_START)

    pts = np.stack([us.astype(np.float64) * GRID_DX,
                    vs.astype(np.float64) * GRID_DY,
                    depth_real], axis=1)

    pts_aligned = (R_est @ pts.T).T + t_est

    u0 = np.round(pts_aligned[:, 0] / GRID_DX).astype(np.int32)
    v0 = np.round(pts_aligned[:, 1] / GRID_DY).astype(np.int32)
    z0 = pts_aligned[:, 2]

    in_bounds = (u0 >= 0) & (u0 < W_out) & (v0 >= 0) & (v0 < H_out) & (z0 > 0)
    u0, v0, z0 = u0[in_bounds], v0[in_bounds], z0[in_bounds]

    warped = np.zeros((H_out, W_out), dtype=np.float64)
    zbuf   = np.full((H_out, W_out), np.inf)
    for i in range(len(u0)):
        if z0[i] < zbuf[v0[i], u0[i]]:
            zbuf[v0[i], u0[i]] = z0[i]
            warped[v0[i], u0[i]] = z0[i]

    if warped.max() > 0:
        warped = warped / warped.max()
    return warped.astype(np.float32)


def visualize_registration(
    depth0_raw, depth1_raw,
    pred, batch_idx, output_path, inlier_th=5.0, conf_th=0.0,
    image_size=1751, ransac_iter=100000, gt_csv_path=None,
):
    kp0    = pred["keypoints0"][batch_idx].cpu().numpy()
    kp1    = pred["keypoints1"][batch_idx].cpu().numpy()
    m0     = pred["matches0"][batch_idx].cpu().numpy()
    scores = pred["matching_scores0"][batch_idx].cpu().numpy()

    valid     = (m0 > -1) & (scores > conf_th)
    mkp0      = kp0[valid]
    mkp1      = kp1[m0[valid]]
    n_matches = int(valid.sum())
    print(f"  conf_th={conf_th:.2f} -> {n_matches} matches (total {int((m0 > -1).sum())})")

    mkp0_orig = kpts_to_orig(mkp0, image_size)
    mkp1_orig = kpts_to_orig(mkp1, image_size)

    # 매칭 분류 + occluded set 생성
    gt_occluded = None
    categories = None
    if gt_csv_path is not None:
        categories, gt_occluded = classify_matches(
            mkp0_orig, mkp1_orig, depth0_raw, depth1_raw, gt_csv_path
        )
        n_blue   = int((categories == 1).sum())
        n_green  = int((categories == 2).sum())
        n_purple = int((categories == 3).sum())
        n_red    = int((categories == 0).sum())
        print(f"  Match class: blue={n_blue} green={n_green} purple={n_purple} red={n_red}")

    pts0, vmask0 = pixel_to_3d(mkp0_orig, depth0_raw)
    pts1, vmask1 = pixel_to_3d(mkp1_orig, depth1_raw)
    both_valid   = vmask0 & vmask1
    pts0_valid   = pts0[both_valid]
    pts1_valid   = pts1[both_valid]
    n_3d = len(pts0_valid)
    print(f"  3D correspondences: {n_3d}")

    if n_3d < 3:
        print("  Not enough 3D correspondences. Skip.")
        return

    # 그리드 3D 좌표계 — pixel_to_3d: (u*dx, v*dy, depth_real)
    R_est, t_est, inliers = ransac_rigid_o3d(
        pts1_valid, pts0_valid, inlier_th=inlier_th, max_iter=ransac_iter
    )
    n_inliers = int(inliers.sum())

    pts1_aligned = (R_est @ pts1_valid.T).T + t_est
    err_before   = np.linalg.norm(pts1_valid  - pts0_valid, axis=1).mean()
    err_after    = np.linalg.norm(pts1_aligned - pts0_valid, axis=1).mean()
    print(f"  Inliers: {n_inliers}/{n_3d}  err: {err_before:.3f} -> {err_after:.3f} mm")

    # RANSAC inlier pair 기반 RMSE (가려지지 않은 pair만)
    # RMSE는 GRID_DZ 적용 3D 거리(mm)로 산출
    inlier_rmse = float("nan")
    n_rmse = 0
    if gt_occluded is not None:
        valid_orig_idx = np.where(both_valid)[0]
        sq_sum = 0.0
        for j in np.where(inliers)[0]:
            mx = int(round(mkp0_orig[valid_orig_idx[j], 0]))
            my = int(round(mkp0_orig[valid_orig_idx[j], 1]))
            if (mx, my) in gt_occluded:
                continue
            p0 = pts0_valid[j].copy()
            p1 = pts1_aligned[j].copy()
            p0[2] *= GRID_DZ
            p1[2] *= GRID_DZ
            sq_sum += float(np.sum((p1 - p0) ** 2))
            n_rmse += 1
        inlier_rmse = float(np.sqrt(sq_sum / n_rmse)) if n_rmse > 0 else float("nan")
        print(f"  Inlier RMSE (non-occluded, {n_rmse}pts): {inlier_rmse:.4f} mm")

    # warp & overlay
    warped_input = warp_depth_to_master(depth1_raw, R_est, t_est, out_shape=depth0_raw.shape)

    d0_vis     = depth0_raw.astype(np.float32) / 65535.0
    d1_raw_vis = depth1_raw.astype(np.float32) / 65535.0

    H, W = d0_vis.shape
    if d1_raw_vis.shape != (H, W):
        d1_vis = cv2.resize(d1_raw_vis, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        d1_vis = d1_raw_vis

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

    rmse_str = f"{inlier_rmse:.4f}" if not np.isnan(inlier_rmse) else "N/A"
    fig.suptitle(
        f"[ISS+SHOT+LG | O3D RANSAC]  Matches: {n_matches}  |  3D: {n_3d}  |  "
        f"Inliers: {n_inliers}/{n_3d}  |  "
        f"RMSE({n_rmse}): {rmse_str} mm",
        fontsize=10, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}  |  Inlier_RMSE={rmse_str}")

    # ─── 3D Scatter: master vs aligned input ──────────────────
    fig_3d = plt.figure(figsize=(12, 9))
    ax3 = fig_3d.add_subplot(111, projection="3d")

    inlier_idx  = np.where(inliers)[0]
    outlier_idx = np.where(~inliers)[0]

    # inlier 오차 계산
    errors_inlier = np.linalg.norm(
        pts1_aligned[inlier_idx] - pts0_valid[inlier_idx], axis=1
    )

    # master inlier (cyan), aligned input inlier (red)
    ax3.scatter(pts0_valid[inlier_idx, 0], pts0_valid[inlier_idx, 1],
                pts0_valid[inlier_idx, 2],
                c="cyan", s=10, alpha=0.7, label="Master (inlier)")
    ax3.scatter(pts1_aligned[inlier_idx, 0], pts1_aligned[inlier_idx, 1],
                pts1_aligned[inlier_idx, 2],
                c="red", s=10, alpha=0.7, label="Aligned Input (inlier)")

    # outlier (gray)
    if len(outlier_idx) > 0:
        ax3.scatter(pts0_valid[outlier_idx, 0], pts0_valid[outlier_idx, 1],
                    pts0_valid[outlier_idx, 2],
                    c="gray", s=3, alpha=0.3, label="Outlier")

    # 오차 연결선 (error magnitude → colormap)
    if len(errors_inlier) > 0:
        cmap = plt.cm.hot
        e_max = errors_inlier.max() if errors_inlier.max() > 0 else 1.0
        for k, j in enumerate(inlier_idx):
            ax3.plot(
                [pts0_valid[j, 0], pts1_aligned[j, 0]],
                [pts0_valid[j, 1], pts1_aligned[j, 1]],
                [pts0_valid[j, 2], pts1_aligned[j, 2]],
                c=cmap(errors_inlier[k] / e_max), alpha=0.5, linewidth=0.8,
            )

    ax3.set_xlabel("X [mm]")
    ax3.set_ylabel("Y [mm]")
    ax3.set_zlabel("Z [mm]")
    ax3.legend(fontsize=8, loc="upper left")
    ax3.set_title(
        f"3D Point Error  |  Inliers: {n_inliers}/{n_3d}  |  RMSE: {rmse_str} mm",
        fontsize=11,
    )

    path_3d = output_path.parent / (output_path.stem + "_3d.png")
    fig_3d.savefig(path_3d, dpi=150, bbox_inches="tight")
    plt.close(fig_3d)
    print(f"  Saved 3D: {path_3d}")

    # ─── 2D View: 모델 입력 뷰에 error 포인트 표시 ──────────────
    # crop + resize → 모델이 보는 1751px 뷰
    crop0 = depth0_raw[CROP_Y0:CROP_Y0 + CROP_SIZE, CROP_X0:CROP_X0 + CROP_SIZE]
    crop1 = depth1_raw[CROP_Y0:CROP_Y0 + CROP_SIZE, CROP_X0:CROP_X0 + CROP_SIZE]
    vis0 = cv2.resize(crop0.astype(np.float32) / 65535.0,
                       (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    vis1 = cv2.resize(crop1.astype(np.float32) / 65535.0,
                       (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    # inlier/outlier의 2D 좌표 (crop+resized 1751px 공간)
    mkp0_3d = mkp0[both_valid]   # 3D 유효한 매칭 키포인트
    mkp1_3d = mkp1[both_valid]
    mkp0_inlier  = mkp0_3d[inliers]
    mkp1_inlier  = mkp1_3d[inliers]
    mkp0_outlier = mkp0_3d[~inliers]
    mkp1_outlier = mkp1_3d[~inliers]

    # per-point 오차 (RANSAC 좌표계)
    errors_inlier_pts = np.linalg.norm(
        pts1_aligned[inliers] - pts0_valid[inliers], axis=1
    )

    fig_2d, (ax_m, ax_i) = plt.subplots(1, 2, figsize=(16, 8))

    ax_m.imshow(vis0, cmap="gray")
    if len(mkp0_outlier) > 0:
        ax_m.scatter(mkp0_outlier[:, 0], mkp0_outlier[:, 1],
                     c="gray", s=8, alpha=0.3, label="Outlier")
    sc_m = ax_m.scatter(mkp0_inlier[:, 0], mkp0_inlier[:, 1],
                        c=errors_inlier_pts, cmap="hot", s=20,
                        edgecolors="white", linewidths=0.3, label="Inlier")
    ax_m.set_title("Master", fontsize=12)
    ax_m.legend(fontsize=8, loc="upper right")
    ax_m.set_axis_off()

    ax_i.imshow(vis1, cmap="gray")
    if len(mkp1_outlier) > 0:
        ax_i.scatter(mkp1_outlier[:, 0], mkp1_outlier[:, 1],
                     c="gray", s=8, alpha=0.3, label="Outlier")
    ax_i.scatter(mkp1_inlier[:, 0], mkp1_inlier[:, 1],
                 c=errors_inlier_pts, cmap="hot", s=20,
                 edgecolors="white", linewidths=0.3, label="Inlier")
    ax_i.set_title("Input", fontsize=12)
    ax_i.legend(fontsize=8, loc="upper right")
    ax_i.set_axis_off()

    fig_2d.colorbar(sc_m, ax=[ax_m, ax_i], shrink=0.6, pad=0.02,
                    label="Error [mm]")
    fig_2d.suptitle(
        f"2D Inlier Error  |  Inliers: {n_inliers}/{n_3d}  |  "
        f"RMSE: {rmse_str} mm",
        fontsize=11,
    )
    fig_2d.tight_layout(rect=[0, 0, 0.92, 0.95])

    path_2d = output_path.parent / (output_path.stem + "_2d_err.png")
    fig_2d.savefig(path_2d, dpi=150, bbox_inches="tight")
    plt.close(fig_2d)
    print(f"  Saved 2D: {path_2d}")

    # ─── 2D Matching Lines: GT 기반 매칭 분류 ────────────────────
    if categories is not None:
        W_off = image_size
        canvas = np.zeros((image_size, image_size * 2), dtype=np.float32)
        canvas[:, :image_size] = vis0
        canvas[:, image_size:] = vis1

        fig_ml, ax_ml = plt.subplots(1, 1, figsize=(20, 10))
        ax_ml.imshow(canvas, cmap="gray")

        cat_cfg = [
            (1, "deepskyblue", "Correct"),
            (2, "limegreen", "Correct+Occluded"),
            (3, "purple",    "Wrong match"),
            (0, "red",       "Undefined depth"),
        ]
        for cat_id, color, label in cat_cfg:
            mask = (categories == cat_id)
            if not mask.any():
                continue
            idxs = np.where(mask)[0]
            for ii in idxs:
                ax_ml.plot(
                    [mkp0[ii, 0], mkp1[ii, 0] + W_off],
                    [mkp0[ii, 1], mkp1[ii, 1]],
                    c=color, alpha=0.5, linewidth=0.8,
                )
            ax_ml.plot([], [], c=color, linewidth=2,
                       label=f"{label} ({int(mask.sum())})")

        ax_ml.axvline(x=image_size, color="white", linewidth=0.5, alpha=0.5)
        ax_ml.legend(fontsize=10, loc="upper right")
        ax_ml.set_title(
            f"Matching Lines  |  Total: {n_matches}  |  "
            f"Blue: {int((categories==1).sum())}  "
            f"Green: {int((categories==2).sum())}  "
            f"Purple: {int((categories==3).sum())}  "
            f"Red: {int((categories==0).sum())}",
            fontsize=11,
        )
        ax_ml.set_axis_off()

        fig_ml.tight_layout()
        path_ml = output_path.parent / (output_path.stem + "_match.png")
        fig_ml.savefig(path_ml, dpi=150, bbox_inches="tight")
        plt.close(fig_ml)
        print(f"  Saved match: {path_ml}")


# ─── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ISS+SHOT+LG 3D Registration (O3D RANSAC + GT RMSE)"
    )
    parser.add_argument("--checkpoint",  type=str,   default=None)
    parser.add_argument("--experiment",  type=str,   default="0407_resample2_iss_shot352_lg")
    parser.add_argument("--device",      type=str,   default="cuda")
    parser.add_argument("--output_dir",  type=str,   default=None)
    parser.add_argument("--split",       type=str,   default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int,   default=5)
    parser.add_argument("--indices",     type=int,   nargs="*", default=None)
    parser.add_argument("--image_size",  type=int,   default=1751)
    parser.add_argument("--shot_radius", type=float, default=10.0)
    parser.add_argument("--conf_th",     type=float, default=0.0)
    parser.add_argument("--inlier_th",   type=float, default=5.0)
    parser.add_argument("--ransac_iter", type=int,   default=100000)
    args = parser.parse_args()

    output_dir = Path(
        args.output_dir if args.output_dir
        else f"results/registration/{args.experiment}_o3d_rmse"
    )
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
    val_end   = total - test_size

    if args.split == "train":
        combo_split = combo.iloc[:train_end]
    elif args.split == "val":
        combo_split = combo.iloc[train_end:val_end]
    else:
        combo_split = combo.iloc[val_end:]
    combo_split = combo_split.reset_index(drop=True)

    dataset = MitsubishiResample2ISSSHOTDataset(
        split=args.split, shot_radius=args.shot_radius, image_size=image_size
    )
    print(f"{args.split} dataset: {len(dataset)} pairs "
          f"(crop={CROP_SIZE}, image_size={image_size}, shot_r={args.shot_radius})")
    print(f"RANSAC: inlier_th={args.inlier_th}, max_iter={args.ransac_iter}")

    if args.indices is not None:
        indices = args.indices
    else:
        indices = np.random.choice(
            len(dataset), min(args.num_samples, len(dataset)), replace=False
        )
        indices = sorted(indices)

    print(f"Testing {len(indices)} pairs: {indices}")

    for idx in indices:
        print(f"\n--- Pair {idx} ---")
        row = combo_split.iloc[idx]

        master_fname    = Path(row["master_path"]).name
        input_fname     = Path(row["input_path"]).name
        master_img_path = base_dir / "dataset_resample_2" / master_fname
        input_img_path  = base_dir / "dataset_resample_2" / input_fname

        depth0_raw = cv2.imread(str(master_img_path), cv2.IMREAD_UNCHANGED)
        depth1_raw = cv2.imread(str(input_img_path),  cv2.IMREAD_UNCHANGED)

        csv_path = base_dir / row["csv_path"]

        sample = dataset[idx]
        batch  = resample2_iss_shot_collate_fn([sample])
        pred, batch = run_inference(model, batch, device)

        visualize_registration(
            depth0_raw, depth1_raw,
            pred, 0,
            output_dir / f"reg_{args.split}_{idx:05d}.png",
            inlier_th=args.inlier_th,
            conf_th=args.conf_th,
            image_size=image_size,
            ransac_iter=args.ransac_iter,
            gt_csv_path=csv_path,
        )

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
