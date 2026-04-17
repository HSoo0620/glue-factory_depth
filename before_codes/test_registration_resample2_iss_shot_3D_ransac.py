"""
학습된 ISS(det)+SHOT(desc)+LG 모델로 resample_2 데이터 3D Registration 테스트.
Open3D registration_ransac_based_on_correspondence 사용.

좌표계: (u*grid_dx, v*grid_dy, depth_real) — 물리 mm 단위
  GRID_DX = GRID_DY = 0.05 mm/pixel
  depth_real: CLIP_START + (raw/65535) * (CLIP_END - CLIP_START) [mm]

파이프라인:
    1. 학습된 ISS+SHOT+LG로 crop+resize된 depth image pair 매칭
    2. 매칭된 keypoint를 원본 5761 좌표로 복원 → (u*dx, v*dy, depth_real) 3D 좌표
    3. Open3D RANSAC based on correspondence로 rigid transformation 추정
    4. 변환 적용 후 정합 결과 시각화

사용법:
    python test_registration_resample2_iss_shot_3D_ransac.py --experiment 0407_resample2_iss_shot352_lg --indices 0 10 50 90
    python test_registration_resample2_iss_shot_3D_ransac.py --checkpoint outputs/training/0407_resample2_iss_shot352_lg/checkpoint_best.tar --conf_th 0.1 --inlier_th 5.0
"""

import argparse
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
CLIP_END = 1000.0
ORIG_SIZE = 5761
GRID_DX = 0.05  # 픽셀→물리 변환 [mm/pixel]
GRID_DY = 0.05


# ─── 2D+Depth → 3D (u*dx, v*dy, depth_real) ─────────────────

def pixel_to_3d(keypoints_2d, depth_map_raw):
    """2D keypoint + raw depth → 3D 좌표 (u*GRID_DX, v*GRID_DY, depth_real) [mm].

    keypoints_2d: (N, 2) — 원본 해상도(5761) 기준 좌표
    depth_map_raw: (H, W) uint16 — 원본 해상도
    반환:
        points_3d: (N, 3) float64 — mm 단위
        valid_mask: (N,) bool
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


# ─── Open3D RANSAC (correspondence 기반) ─────────────────────

def ransac_rigid_o3d(src, dst, inlier_th=5.0, ransac_n=3, max_iter=100000):
    """Open3D registration_ransac_based_on_correspondence.

    src: (N, 3) — input 포인트 (pts1)
    dst: (N, 3) — master 포인트 (pts0)
    inlier_th: correspondence 허용 거리 [mm]
    반환:
        R: (3, 3) rotation
        t: (3,) translation
        inlier_mask: (N,) bool — src 기준 inlier
    """
    N = src.shape[0]
    if N < ransac_n:
        return np.eye(3), np.zeros(3), np.zeros(N, dtype=bool)

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src)

    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(dst)

    # i번째 src ↔ i번째 dst 대응
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

    T = np.asarray(result.transformation)  # 4×4, src → dst
    R = T[:3, :3]
    t = T[:3, 3]

    # inlier mask 재구성 (src 인덱스 기준)
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
    """crop+resize 좌표를 원본 5761 좌표로 복원.

    kpts: (N, 2) — resize된 이미지 기준 좌표
    image_size: resize 후 크기 (e.g. 1751)
    반환: (N, 2) — 원본 5761 기준 좌표
    """
    scale = CROP_SIZE / image_size  # 3502 / 1751 = 2.0
    kpts_orig = kpts * scale
    kpts_orig = kpts_orig.copy()
    kpts_orig[:, 0] += CROP_X0
    kpts_orig[:, 1] += CROP_Y0
    return kpts_orig


# ─── Visualization ─────────────────────────────────────────

def warp_depth_to_master(depth1_raw, R_est, t_est, out_shape=None):
    """Input depth를 추정된 R,t로 변환하여 master 좌표계에 투영.

    out_shape: (H, W) — 출력 크기 (master 이미지 크기). None이면 depth1_raw와 동일.
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
    zbuf = np.full((H_out, W_out), np.inf)
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
    image_size=1751, ransac_iter=100000,
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

    # crop+resize 좌표 → 원본 5761 좌표로 복원
    mkp0_orig = kpts_to_orig(mkp0, image_size)
    mkp1_orig = kpts_to_orig(mkp1, image_size)

    # (u*dx, v*dy, depth_real) 3D 좌표 [mm]
    pts0, vmask0 = pixel_to_3d(mkp0_orig, depth0_raw)
    pts1, vmask1 = pixel_to_3d(mkp1_orig, depth1_raw)
    both_valid = vmask0 & vmask1
    pts0_valid = pts0[both_valid]
    pts1_valid = pts1[both_valid]
    n_3d = len(pts0_valid)
    print(f"  3D correspondences: {n_3d}")

    if n_3d < 3:
        print("  Not enough 3D correspondences. Skip.")
        return

    # Open3D RANSAC: src=pts1(input), dst=pts0(master)
    R_est, t_est, inliers = ransac_rigid_o3d(
        pts1_valid, pts0_valid, inlier_th=inlier_th, max_iter=ransac_iter
    )
    n_inliers = int(inliers.sum())

    pts1_aligned = (R_est @ pts1_valid.T).T + t_est
    err_before = np.linalg.norm(pts1_valid - pts0_valid, axis=1).mean()
    err_after  = np.linalg.norm(pts1_aligned - pts0_valid, axis=1).mean()
    print(f"  Inliers: {n_inliers}/{n_3d}  err: {err_before:.3f} -> {err_after:.3f} mm")

    # warp & overlay
    warped_input = warp_depth_to_master(depth1_raw, R_est, t_est, out_shape=depth0_raw.shape)

    d0_vis = depth0_raw.astype(np.float32) / 65535.0
    d1_raw_vis = depth1_raw.astype(np.float32) / 65535.0

    H, W = d0_vis.shape
    if d1_raw_vis.shape != (H, W):
        d1_vis = cv2.resize(d1_raw_vis, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        d1_vis = d1_raw_vis

    overlay = np.zeros((H, W, 3), dtype=np.float32)
    overlay[:, :, 1] = d0_vis   # Master -> cyan
    overlay[:, :, 2] = d0_vis
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], warped_input)  # Warped -> red

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
        f"[Resample2 ISS+SHOT+LG | O3D RANSAC]  Matches: {n_matches}  |  3D: {n_3d}  |  "
        f"Inliers: {n_inliers}/{n_3d}  |  "
        f"Error: {err_before:.3f} -> {err_after:.3f} mm",
        fontsize=11, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="학습된 ISS+SHOT+LG 기반 resample_2 3D Registration (O3D RANSAC)"
    )
    parser.add_argument("--checkpoint",   type=str,   default=None)
    parser.add_argument("--experiment",   type=str,   default="0407_resample2_iss_shot352_lg")
    parser.add_argument("--device",       type=str,   default="cuda")
    parser.add_argument("--output_dir",   type=str,   default=None)
    parser.add_argument("--split",        type=str,   default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--num_samples",  type=int,   default=5)
    parser.add_argument("--indices",      type=int,   nargs="*", default=None)
    parser.add_argument("--image_size",   type=int,   default=1751)
    parser.add_argument("--shot_radius",  type=float, default=10.0)
    parser.add_argument("--conf_th",      type=float, default=0.0)
    parser.add_argument("--inlier_th",    type=float, default=5.0,
                        help="RANSAC inlier distance threshold [mm]")
    parser.add_argument("--ransac_iter",  type=int,   default=100000,
                        help="RANSACConvergenceCriteria max_iteration")
    args = parser.parse_args()

    output_dir = Path(
        args.output_dir if args.output_dir
        else f"results/registration/{args.experiment}_o3d"
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
    print(f"RANSAC: inlier_th={args.inlier_th}mm, max_iter={args.ransac_iter}")

    if args.indices is not None:
        indices = args.indices
    else:
        indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
        indices = sorted(indices)

    print(f"Testing {len(indices)} pairs: {indices}")

    for idx in indices:
        print(f"\n--- Pair {idx} ---")
        row = combo_split.iloc[idx]

        master_fname = Path(row["master_path"]).name
        input_fname  = Path(row["input_path"]).name
        master_img_path = base_dir / "dataset_resample_2" / master_fname
        input_img_path  = base_dir / "dataset_resample_2" / input_fname

        depth0_raw = cv2.imread(str(master_img_path), cv2.IMREAD_UNCHANGED)
        depth1_raw = cv2.imread(str(input_img_path),  cv2.IMREAD_UNCHANGED)

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
        )

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
