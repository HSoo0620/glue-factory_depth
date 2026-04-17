"""
Registration 결과 시각화 — FLANN Baseline 포인트 클라우드 정합.

캐시된 ISS keypoints + descriptors를 FLANN으로 매칭하고,
RANSAC+SVD로 R,t를 구한 뒤 포인트 클라우드 정합을 시각화.

  Panel Row 0: Before (master=lightskyblue, input=crimson)
  Panel Row 1: After estimated transform
  Panel Row 2: After GT transform

사용법:
    python visualize_registration_flann.py --descriptor shot --indices 0 10 50 90
    python visualize_registration_flann.py --descriptor fpfh --indices 0 10 50 90
    python visualize_registration_flann.py --descriptor rops --ratio_th 0.8
"""

import argparse
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# --- 상수 ---
CROP_X0 = 1129
CROP_Y0 = 1081
CROP_SIZE = 3502
CLIP_START = 0.1
CLIP_END = 1000.0
GRID_DX = 0.05
GRID_DY = 0.05

DESCRIPTOR_CONFIG = {
    "shot":  {"key": "shot_descriptors",  "cache_dir": "iss_shot352_resample2_cache"},
    "fpfh":  {"key": "fpfh_descriptors",  "cache_dir": "iss_fpfh_resample2_cache_r5.0_xyz"},
    "rops":  {"key": "rops_descriptors",  "cache_dir": "iss_rops135_resample2_cache"},
}


# --- Core math ---

def pixel_to_grid3d(keypoints_2d, depth_map_raw):
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


def rigid_transform_svd(P_src, P_dst):
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
    df = pd.read_csv(gt_csv_path)
    valid = df["occluded"] == False
    master_xy = df.loc[valid, ["master_x", "master_y"]].values.astype(np.float64)
    input_xy = df.loc[valid, ["input_x", "input_y"]].values.astype(np.float64)
    H0, W0 = depth0_raw.shape
    H1, W1 = depth1_raw.shape
    pts_m, pts_i = [], []
    for i in range(len(master_xy)):
        mu = int(np.clip(master_xy[i, 0], 0, W0 - 1))
        mv = int(np.clip(master_xy[i, 1], 0, H0 - 1))
        iu = int(np.clip(input_xy[i, 0], 0, W1 - 1))
        iv = int(np.clip(input_xy[i, 1], 0, H1 - 1))
        d0, d1 = float(depth0_raw[mv, mu]), float(depth1_raw[iv, iu])
        if d0 <= 0 or d1 <= 0:
            continue
        dr0 = CLIP_START + (d0 / 65535.0) * (CLIP_END - CLIP_START)
        dr1 = CLIP_START + (d1 / 65535.0) * (CLIP_END - CLIP_START)
        pts_m.append([mu * GRID_DX, mv * GRID_DY, dr0])
        pts_i.append([iu * GRID_DX, iv * GRID_DY, dr1])
    pts_m, pts_i = np.array(pts_m), np.array(pts_i)
    if len(pts_m) < 3:
        return np.eye(3), np.zeros(3), len(pts_m)
    R_gt, t_gt = rigid_transform_svd(pts_i, pts_m)
    return R_gt, t_gt, len(pts_m)


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


def sample_point_cloud(depth_raw, n_pts=15000, seed=42):
    rng = np.random.RandomState(seed)
    vs, us = np.where(depth_raw > 0)
    n_valid = len(vs)
    if n_valid == 0:
        return np.zeros((0, 3), dtype=np.float64)
    idx = rng.choice(n_valid, min(n_pts, n_valid), replace=False)
    us_s, vs_s = us[idx].astype(np.float64), vs[idx].astype(np.float64)
    d_raw = depth_raw[vs[idx], us[idx]].astype(np.float64)
    depth_real = CLIP_START + (d_raw / 65535.0) * (CLIP_END - CLIP_START)
    return np.stack([us_s * GRID_DX, vs_s * GRID_DY, depth_real], axis=1)


# --- Cache & FLANN ---

def load_cache(cache_dir, depth_fname, desc_key):
    npz_name = Path(depth_fname).stem + ".npz"
    data = np.load(str(cache_dir / npz_name))
    n_valid = int(data["n_valid"])
    kp = data["keypoints"][:n_valid]
    desc = data[desc_key][:n_valid]
    return kp, desc


def flann_match(desc0, desc1, ratio_th=0.75):
    if len(desc0) < 2 or len(desc1) < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(desc0.astype(np.float32),
                               desc1.astype(np.float32), k=2)
    idx0, idx1 = [], []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < ratio_th * n.distance:
                idx0.append(m.queryIdx)
                idx1.append(m.trainIdx)
    return np.array(idx0, dtype=int), np.array(idx1, dtype=int)


def kpts_to_orig(kpts, image_size):
    scale = CROP_SIZE / image_size
    kpts_orig = kpts.copy().astype(np.float64)
    kpts_orig *= scale
    kpts_orig[:, 0] += CROP_X0
    kpts_orig[:, 1] += CROP_Y0
    return kpts_orig


# --- Visualization ---

def plot_alignment(pc_master, pc_input, pc_est, pc_gt, title, output_path,
                   n_inliers=0, n_matches=0, rmse_est=None):
    """3-row x 3-col alignment visualization."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    s = 0.5
    alpha = 0.6

    labels_row = ["Before", "Estimated R,t", "GT R,t"]
    labels_col = ["Top-down (u, v)", "Front (u, z)", "Side (v, z)"]
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
        info += f", RMSE={rmse_est:.4f}"
    fig.suptitle(info, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_overlay_detail(pc_master, pc_est, title, output_path):
    """Estimated alignment overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    s = 0.5
    alpha = 0.6
    col_axes = [(0, 1), (0, 2), (1, 2)]
    col_labels = ["Top-down (u, v)", "Front (u, z)", "Side (v, z)"]

    for col, (xi, yi) in enumerate(col_axes):
        ax = axes[col]
        ax.scatter(pc_master[:, xi], pc_master[:, yi], s=s, c="lightskyblue", alpha=alpha, label="master")
        ax.scatter(pc_est[:, xi], pc_est[:, yi], s=s, c="crimson", alpha=alpha, label="aligned")
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


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="FLANN Baseline Registration Visualization")
    parser.add_argument("--descriptor", type=str, default="shot",
                        choices=["shot", "fpfh", "rops"])
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--image_size", type=int, default=1751)
    parser.add_argument("--ratio_th", type=float, default=0.75)
    parser.add_argument("--inlier_th", type=float, default=5.0)
    parser.add_argument("--ransac_iter", type=int, default=1000)
    parser.add_argument("--n_sample_pts", type=int, default=15000,
                        help="시각화용 샘플 포인트 수 (기본 15K)")
    args = parser.parse_args()

    desc_conf = DESCRIPTOR_CONFIG[args.descriptor]
    cache_dir_name = args.cache_dir if args.cache_dir else desc_conf["cache_dir"]
    desc_key = desc_conf["key"]

    output_dir = Path(
        args.output_dir if args.output_dir
        else f"results/registration/iss_{args.descriptor}_flann_vis")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path("gluefactory/datasets/mitsubishi")
    cache_dir = base_dir / cache_dir_name

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

    title_prefix = f"ISS+{args.descriptor.upper()}+FLANN"
    print(f"Method: {title_prefix} (cache: {cache_dir_name})")
    print(f"Split: {args.split} ({len(combo_split)} pairs)")
    print(f"ratio_th={args.ratio_th}, inlier_th={args.inlier_th}")

    if args.indices is not None:
        indices = args.indices
    else:
        indices = sorted(np.random.choice(
            len(combo_split), min(args.num_samples, len(combo_split)),
            replace=False))
    print(f"Visualizing {len(indices)} pairs: {indices}")

    image_size = args.image_size

    for idx in indices:
        print(f"\n--- Pair {idx} ---")
        row = combo_split.iloc[idx]
        master_fname = Path(row["master_path"]).name
        input_fname = Path(row["input_path"]).name

        depth0_raw = cv2.imread(
            str(base_dir / "dataset_resample_2" / master_fname), cv2.IMREAD_UNCHANGED)
        depth1_raw = cv2.imread(
            str(base_dir / "dataset_resample_2" / input_fname), cv2.IMREAD_UNCHANGED)
        csv_path = base_dir / row["csv_path"]

        # GT transform
        R_gt, t_gt, n_gt = compute_gt_transform(csv_path, depth0_raw, depth1_raw)
        print(f"  GT: {n_gt} correspondences")

        # Cache load + FLANN matching
        kp0, desc0 = load_cache(cache_dir, master_fname, desc_key)
        kp1, desc1 = load_cache(cache_dir, input_fname, desc_key)

        idx0, idx1 = flann_match(desc0, desc1, ratio_th=args.ratio_th)
        if len(idx0) == 0:
            print(f"  No FLANN matches. Skip.")
            continue

        mkp0_crop = kp0[idx0]
        mkp1_crop = kp1[idx1]
        mkp0_orig = kpts_to_orig(mkp0_crop, image_size)
        mkp1_orig = kpts_to_orig(mkp1_crop, image_size)

        pts0, vm0 = pixel_to_grid3d(mkp0_orig, depth0_raw)
        pts1, vm1 = pixel_to_grid3d(mkp1_orig, depth1_raw)
        both = vm0 & vm1
        pts0_v, pts1_v = pts0[both], pts1[both]
        n_3d = len(pts0_v)
        n_match = len(idx0)

        if n_3d < 3:
            print(f"  Not enough 3D matches ({n_3d}). Skip.")
            continue

        R_est, t_est, inliers = ransac_rigid(
            pts1_v, pts0_v, n_iter=args.ransac_iter, inlier_th=args.inlier_th)
        n_inl = int(inliers.sum())
        print(f"  Matches={n_match}, 3D={n_3d}, inliers={n_inl}")

        # Point clouds
        pc_master = sample_point_cloud(depth0_raw, args.n_sample_pts)
        pc_input = sample_point_cloud(depth1_raw, args.n_sample_pts)

        pc_est = (R_est @ pc_input.T).T + t_est
        pc_gt = (R_gt @ pc_input.T).T + t_gt

        # RMSE
        P_30k = sample_point_cloud(depth1_raw, 30000)
        P_est_30k = (R_est @ P_30k.T).T + t_est
        P_gt_30k = (R_gt @ P_30k.T).T + t_gt
        rmse = float(np.sqrt(np.mean(np.sum((P_est_30k - P_gt_30k) ** 2, axis=1))))
        print(f"  RMSE={rmse:.4f}")

        pair_title = f"[{title_prefix}] {Path(master_fname).stem} vs {Path(input_fname).stem}"

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
