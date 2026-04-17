"""Registration 시각화 — new_dataset ISS + FPFH/SHOT + LG.

visualize_registration.py 의 NEW ISS 버전.
구조·레이아웃·컬러 모두 동일하게 유지하고 다음만 교체:
  - 그리드 3D → camera-frame XYZ (new_dataset.coords.pixel_to_cam_xyz)
  - GT 를 pair CSV 의 SVD 가 아니라 scene config(camera_rt) 로 계산
      T_gt(cam_i → cam_m) = inv(T_world_from_cam_m) @ T_world_from_cam_i
  - keypoint 스케일 복원은 `/ resize_factor` (crop 상수 제거)
  - 데이터셋 = NewDatasetISSDescDataset (fpfh/shot cache 자동 선택)

사용법:
    python visualize_registration_new_iss.py \
        --checkpoint outputs/training/0413_new_iss_fpfh_v5_lg/checkpoint_best.tar \
        --descriptor_type fpfh --indices 0 10 50 90 --experiment 0413_new_iss_fpfh_v5_lg

    python visualize_registration_new_iss.py \
        --checkpoint outputs/training/0413_new_iss_shot_v5_lg/checkpoint_best.tar \
        --descriptor_type shot --indices 0 10 50 90 --experiment 0413_new_iss_shot_v5_lg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.datasets.new_dataset import (
    NewDatasetISSDescDataset, collate_fn_dynamic_pad,
)
from gluefactory.datasets.new_dataset import constants as NEW_C
from gluefactory.datasets.new_dataset.coords import (
    load_scene_config, pixel_to_cam_xyz,
)
from gluefactory.datasets.new_dataset.split import pair_filename_to_scene_ids
from gluefactory.utils.tensor import batch_to_device


# ─── Core math ────────────────────────────────

def pixel_to_cam_vec(keypoints_2d: np.ndarray, zmap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N,2) keypoints → (N,3) camera-frame xyz mm + valid mask (raw>0)."""
    H, W = zmap.shape
    u = np.clip(np.round(keypoints_2d[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(keypoints_2d[:, 1]).astype(int), 0, H - 1)
    raw = zmap[v, u].astype(np.float64)
    xyz = pixel_to_cam_xyz(u, v, raw)
    return xyz, raw > 0


def sample_camera_pcd(zmap: np.ndarray, n_pts: int = 30000, seed: int = 42) -> np.ndarray:
    """zmap 의 valid 픽셀 중 n_pts 샘플 → camera-frame (N,3) mm."""
    rng = np.random.RandomState(seed)
    vs, us = np.where(zmap > 0)
    n_valid = len(vs)
    if n_valid == 0:
        return np.zeros((0, 3), dtype=np.float64)
    idx = rng.choice(n_valid, min(n_pts, n_valid), replace=False)
    us_s, vs_s = us[idx].astype(np.float64), vs[idx].astype(np.float64)
    raw = zmap[vs[idx], us[idx]].astype(np.float64)
    return pixel_to_cam_xyz(us_s, vs_s, raw)


def rigid_transform_svd(P_src: np.ndarray, P_dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def ransac_rigid(src: np.ndarray, dst: np.ndarray, n_iter: int = 1000,
                 inlier_th: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def gt_transform_cam_i_to_cam_m(cfg_m, cfg_i) -> tuple[np.ndarray, np.ndarray]:
    """world = R_cam @ cam + t_cam 규약에서,
    cam_m = R_m.T @ (R_i @ cam_i + t_i - t_m)
          = (R_m.T @ R_i) @ cam_i + R_m.T @ (t_i - t_m)
    """
    R = cfg_m.R_cam.T @ cfg_i.R_cam
    t = cfg_m.R_cam.T @ (cfg_i.t_cam - cfg_m.t_cam)
    return R, t


# ─── Model helpers ────────────────────────────

def load_model(checkpoint_path: str, device: str):
    cp = torch.load(checkpoint_path, map_location="cpu")
    conf = OmegaConf.create(cp["conf"])
    model = get_model(conf.model.name)(conf.model).to(device)
    model.load_state_dict(cp["model"], strict=False)
    model.eval()
    print(f"Loaded: {checkpoint_path} (epoch {cp['epoch']})")
    return model, conf


@torch.no_grad()
def run_inference(model, batch, device):
    batch = batch_to_device(batch, device)
    return model(batch), batch


def extract_matches(pred, batch_idx: int, conf_th: float = 0.0):
    kp0 = pred["keypoints0"][batch_idx].cpu().numpy()
    kp1 = pred["keypoints1"][batch_idx].cpu().numpy()
    m0 = pred["matches0"][batch_idx].cpu().numpy()
    scores_key = "matching_scores0" if "matching_scores0" in pred else None
    if scores_key is not None:
        scores = pred[scores_key][batch_idx].cpu().numpy()
        valid = (m0 > -1) & (scores > conf_th)
    else:
        valid = m0 > -1
    return kp0[valid], kp1[m0[valid]], int(valid.sum())


# ─── Visualization (visualize_registration.py 와 동일 포맷) ──

def plot_alignment(pc_master, pc_input, pc_est, pc_gt, title, output_path,
                   n_inliers: int = 0, n_matches: int = 0, rmse_est: float | None = None):
    """3×3 시각화. X/Y/Z 축명만 mm 로 교체."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    s = 0.5
    alpha = 0.6

    labels_row = ["Before", "Estimated R,t", "GT R,t"]
    labels_col = ["Top-down (X, Y)", "Front (X, Z)", "Side (Y, Z)"]
    col_axes = [(0, 1), (0, 2), (1, 2)]

    C_MASTER = "lightskyblue"
    C_INPUT = "crimson"
    pairs = [
        (pc_master, pc_input, C_MASTER, C_INPUT),
        (pc_master, pc_est, C_MASTER, C_INPUT),
        (pc_master, pc_gt, C_MASTER, C_INPUT),
    ]

    for row, (pc_a, pc_b, ca, cb) in enumerate(pairs):
        for col, (xi, yi) in enumerate(col_axes):
            ax = axes[row, col]
            ax.scatter(pc_a[:, xi], pc_a[:, yi], s=s, c=ca, alpha=alpha, label="master")
            ax.scatter(pc_b[:, xi], pc_b[:, yi], s=s, c=cb, alpha=alpha, label="input")
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(labels_col[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(labels_row[row], fontsize=11, fontweight="bold")

    for row in range(3):
        axes[row, 0].invert_yaxis()
        axes[row, 1].invert_yaxis()
        axes[row, 2].invert_yaxis()

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
    for ax in axes:
        ax.invert_yaxis()
    fig.suptitle(f"Overlay: {title}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─── Main ─────────────────────────────────────

def resolve_cache_dir(args) -> Path:
    if args.descriptor_type == "fpfh":
        return (Path("gluefactory/datasets/new_dataset_cache")
                / f"cache_new_iss_fpfh_v{args.voxel_size}_r{args.fpfh_radius}")
    return (Path("gluefactory/datasets/new_dataset_cache")
            / f"cache_new_iss_shot352_{args.shot_version}")


def main():
    p = argparse.ArgumentParser(
        description="Registration visualization — new_dataset ISS + FPFH/SHOT + LG")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--experiment", type=str, default="0413_new_iss_fpfh_v5_lg")
    p.add_argument("--descriptor_type", type=str, default="fpfh", choices=["fpfh", "shot"])
    p.add_argument("--fpfh_radius", type=float, default=50.0)
    p.add_argument("--voxel_size", type=float, default=5.0)
    p.add_argument("--shot_version", type=str, default="v5")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--indices", type=int, nargs="*", default=None)
    p.add_argument("--resize_factor", type=float, default=NEW_C.RESIZE_FACTOR)
    p.add_argument("--conf_th", type=float, default=0.0)
    p.add_argument("--inlier_th", type=float, default=5.0,
                   help="RANSAC inlier threshold (mm)")
    p.add_argument("--ransac_iter", type=int, default=1000)
    p.add_argument("--n_sample_pts", type=int, default=15000,
                   help="시각화용 포인트 샘플 수")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    if args.checkpoint is None:
        from gluefactory.utils.experiments import get_best_checkpoint
        cp_path = str(get_best_checkpoint(args.experiment))
        print(f"Auto-loaded best checkpoint: {cp_path}")
    else:
        cp_path = args.checkpoint

    cp_stem = Path(cp_path).stem
    output_dir = Path(
        args.output_dir if args.output_dir
        else f"results/registration/{args.experiment}_{cp_stem}_vis")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _ = load_model(cp_path, device)
    cache_dir = resolve_cache_dir(args)
    dataset = NewDatasetISSDescDataset(
        split=args.split, cache_dir=cache_dir,
        resize_factor=args.resize_factor, val_ratio=0.05, seed=0,
    )
    print(f"{args.split} dataset: {len(dataset)} pairs  ({args.descriptor_type}, cache={cache_dir})")

    if args.indices is not None:
        indices = list(args.indices)
    else:
        indices = sorted(np.random.choice(
            len(dataset), min(args.num_samples, len(dataset)), replace=False).tolist())
    print(f"Visualizing {len(indices)} pairs: {indices}")

    for idx in indices:
        print(f"\n--- Pair {idx} ---")
        sample = dataset[idx]
        master_path = Path(sample["master_path"])
        input_path = Path(sample["input_path"])
        csv_path = Path(sample["csv_path"])
        m_id, i_id = pair_filename_to_scene_ids(csv_path.name)

        data_root = master_path.parent
        cfg_m = load_scene_config(m_id, data_root)
        cfg_i = load_scene_config(i_id, data_root)

        zmap_m = cv2.imread(str(master_path), cv2.IMREAD_UNCHANGED)
        zmap_i = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)

        batch = collate_fn_dynamic_pad([sample])
        pred, batch = run_inference(model, batch, device)

        mkp0, mkp1, n_match = extract_matches(pred, 0, args.conf_th)
        # resized 공간 → 원본 픽셀 공간으로 스케일 복원
        if args.resize_factor != 1.0:
            mkp0 = mkp0 / args.resize_factor
            mkp1 = mkp1 / args.resize_factor

        pts0, vm0 = pixel_to_cam_vec(mkp0, zmap_m)
        pts1, vm1 = pixel_to_cam_vec(mkp1, zmap_i)
        both = vm0 & vm1
        pts0_v, pts1_v = pts0[both], pts1[both]
        n_3d = len(pts0_v)
        if n_3d < 3:
            print(f"  Not enough 3D matches ({n_3d}). Skip.")
            continue

        # RANSAC: input cam → master cam
        R_est, t_est, inliers = ransac_rigid(
            pts1_v, pts0_v, n_iter=args.ransac_iter, inlier_th=args.inlier_th)
        n_inl = int(inliers.sum())
        print(f"  Matches={n_match}, valid-depth={n_3d}, inliers={n_inl}")

        # GT transform (cam_i → cam_m) from scene configs
        R_gt, t_gt = gt_transform_cam_i_to_cam_m(cfg_m, cfg_i)

        # 포인트 클라우드 샘플
        pc_master = sample_camera_pcd(zmap_m, args.n_sample_pts)
        pc_input = sample_camera_pcd(zmap_i, args.n_sample_pts)
        pc_est = (R_est @ pc_input.T).T + t_est
        pc_gt = (R_gt @ pc_input.T).T + t_gt

        # RMSE (T_est vs T_gt, 30K 샘플)
        P_30k = sample_camera_pcd(zmap_i, 30000, seed=123)
        P_est_30k = (R_est @ P_30k.T).T + t_est
        P_gt_30k = (R_gt @ P_30k.T).T + t_gt
        rmse = float(np.sqrt(np.mean(np.sum((P_est_30k - P_gt_30k) ** 2, axis=1))))
        print(f"  RMSE(est vs GT)={rmse:.4f} mm")

        # Display-only: flip Z sign so depth increases opposite of camera-frame convention.
        z_flip = np.array([1.0, 1.0, -1.0])
        pc_master_v = pc_master * z_flip
        pc_input_v = pc_input * z_flip
        pc_est_v = pc_est * z_flip
        pc_gt_v = pc_gt * z_flip

        pair_title = f"scene {m_id:04d} vs {i_id:04d}  ({args.descriptor_type})"
        plot_alignment(
            pc_master_v, pc_input_v, pc_est_v, pc_gt_v,
            pair_title,
            output_dir / f"align_{args.split}_{idx:05d}_{m_id:04d}_{i_id:04d}.png",
            n_inliers=n_inl, n_matches=n_match, rmse_est=rmse)

        plot_overlay_detail(
            pc_master_v, pc_est_v, pair_title,
            output_dir / f"overlay_{args.split}_{idx:05d}_{m_id:04d}_{i_id:04d}.png")

    print(f"\nDone! Output: {output_dir}/")


if __name__ == "__main__":
    main()
