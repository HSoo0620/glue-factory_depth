"""
Rule-based (SIFT/ORB + FLANN) Registration 시각화 — 포인트 클라우드 정합.

visualize_registration.py 와 동일한 시각화:
  Panel 3x3: Before / Estimated / GT × Top-down / Front / Side
  + Overlay detail (1×3)

사용법:
    python visualize_registration_rule_based.py --method sift --indices 0 10 50 90
    python visualize_registration_rule_based.py --method orb --indices 0 10 50 90
    python visualize_registration_rule_based.py --method sift --ratio_th 0.8 --max_keypoints 4000
"""

import argparse
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ─── 상수 ──────────────────────────────────────
CROP_X0    = 1129
CROP_Y0    = 1081
CROP_SIZE  = 3502
CLIP_START = 0.1
CLIP_END   = 1000.0
GRID_DX    = 0.05
GRID_DY    = 0.05


# ─── Core math (visualize_registration.py 동일) ──

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
    input_xy  = df.loc[valid, ["input_x",  "input_y" ]].values.astype(np.float64)
    H0, W0 = depth0_raw.shape
    H1, W1 = depth1_raw.shape
    pts_m, pts_i = [], []
    for i in range(len(master_xy)):
        mu = int(np.clip(master_xy[i, 0], 0, W0 - 1))
        mv = int(np.clip(master_xy[i, 1], 0, H0 - 1))
        iu = int(np.clip(input_xy[i, 0],  0, W1 - 1))
        iv = int(np.clip(input_xy[i, 1],  0, H1 - 1))
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


def sample_point_cloud(depth_raw, n_pts=30000, seed=42):
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


# ─── Feature detection & matching ─────────────

def depth_to_uint8(depth_raw):
    mask = depth_raw > 0
    if not mask.any():
        return np.zeros_like(depth_raw, dtype=np.uint8)
    d_valid = depth_raw[mask].astype(np.float64)
    d_min = d_valid.min()
    d_max = np.percentile(d_valid, 99)
    result = np.zeros_like(depth_raw, dtype=np.uint8)
    normalized = np.clip(
        (depth_raw.astype(np.float64) - d_min) / (d_max - d_min + 1e-8), 0, 1)
    result[mask] = (normalized[mask] * 255).astype(np.uint8)
    return result


def create_detector(method, max_keypoints):
    if method == "sift":
        return cv2.SIFT_create(nfeatures=max_keypoints)
    elif method == "orb":
        return cv2.ORB_create(nfeatures=max_keypoints)
    raise ValueError(f"Unknown method: {method}")


def create_flann_matcher(method):
    if method == "sift":
        index_params = dict(algorithm=1, trees=5)
    elif method == "orb":
        index_params = dict(algorithm=6, table_number=6,
                            key_size=12, multi_probe_level=1)
    return cv2.FlannBasedMatcher(index_params, dict(checks=50))


def detect_and_match(img0_u8, img1_u8, detector, method, ratio_th=0.75):
    kp0, des0 = detector.detectAndCompute(img0_u8, None)
    kp1, des1 = detector.detectAndCompute(img1_u8, None)
    n_kp0 = len(kp0) if kp0 else 0
    n_kp1 = len(kp1) if kp1 else 0
    empty = np.zeros((0, 2), dtype=np.float64)

    if des0 is None or des1 is None or n_kp0 < 2 or n_kp1 < 2:
        return empty, empty, n_kp0, n_kp1

    matcher = create_flann_matcher(method)
    try:
        matches = matcher.knnMatch(des0, des1, k=2)
    except cv2.error:
        return empty, empty, n_kp0, n_kp1

    good = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < ratio_th * n.distance:
                good.append(m)

    if len(good) == 0:
        return empty, empty, n_kp0, n_kp1

    mkp0 = np.array([kp0[m.queryIdx].pt for m in good], dtype=np.float64)
    mkp1 = np.array([kp1[m.trainIdx].pt for m in good], dtype=np.float64)
    return mkp0, mkp1, n_kp0, n_kp1


# ─── Visualization (visualize_registration.py 동일) ──

def plot_alignment(pc_master, pc_input, pc_est, pc_gt, title, output_path,
                   n_inliers=0, n_matches=0, rmse_est=None):
    """3-row × 3-col 시각화.
    Row 0: Before  (master=skyblue, input=crimson)
    Row 1: Estimated (master=skyblue, aligned=crimson)
    Row 2: GT       (master=skyblue, gt_aligned=crimson)
    Col 0: Top-down (u-v)
    Col 1: Front    (u-z)
    Col 2: Side     (v-z)
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    s = 0.5
    alpha = 0.6

    labels_row = ["Before", "Estimated R,t", "GT R,t"]
    labels_col = ["Top-down (u, v)", "Front (u, z)", "Side (v, z)"]
    col_axes = [(0, 1), (0, 2), (1, 2)]

    C_MASTER = "lightskyblue"
    C_INPUT  = "crimson"
    pairs = [
        (pc_master, pc_input, C_MASTER, C_INPUT),
        (pc_master, pc_est,   C_MASTER, C_INPUT),
        (pc_master, pc_gt,    C_MASTER, C_INPUT),
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
    """Estimated alignment 확대 — overlay 한 장."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    s = 0.5
    alpha = 0.6
    col_axes = [(0, 1), (0, 2), (1, 2)]
    col_labels = ["Top-down (u, v)", "Front (u, z)", "Side (v, z)"]

    for col, (xi, yi) in enumerate(col_axes):
        ax = axes[col]
        ax.scatter(pc_master[:, xi], pc_master[:, yi], s=s, c="lightskyblue", alpha=alpha, label="master")
        ax.scatter(pc_est[:, xi],    pc_est[:, yi],    s=s, c="crimson",      alpha=alpha, label="aligned")
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

def main():
    parser = argparse.ArgumentParser(
        description="Rule-based (SIFT/ORB + FLANN) Registration 포인트 클라우드 시각화")
    parser.add_argument("--method", type=str, default="sift",
                        choices=["sift", "orb"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--max_keypoints", type=int, default=2000)
    parser.add_argument("--ratio_th", type=float, default=0.75,
                        help="Lowe's ratio test threshold")
    parser.add_argument("--inlier_th", type=float, default=5.0)
    parser.add_argument("--ransac_iter", type=int, default=1000)
    parser.add_argument("--n_sample_pts", type=int, default=15000,
                        help="시각화용 샘플 포인트 수")
    args = parser.parse_args()

    output_dir = Path(
        args.output_dir if args.output_dir
        else f"results/registration/{args.method}_flann_vis")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path("gluefactory/datasets/mitsubishi")
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

    detector = create_detector(args.method, args.max_keypoints)

    print(f"Method: {args.method.upper()} + FLANN")
    print(f"Split: {args.split} ({len(combo_split)} pairs)")
    print(f"max_keypoints={args.max_keypoints}, ratio_th={args.ratio_th}")
    print(f"RANSAC: inlier_th={args.inlier_th}, iter={args.ransac_iter}")

    if args.indices is not None:
        indices = args.indices
    else:
        indices = sorted(np.random.choice(
            len(combo_split), min(args.num_samples, len(combo_split)),
            replace=False))

    print(f"Visualizing {len(indices)} pairs: {indices}")

    for idx in indices:
        print(f"\n--- Pair {idx} ---")
        row = combo_split.iloc[idx]
        master_fname = Path(row["master_path"]).name
        input_fname  = Path(row["input_path"]).name

        depth0_raw = cv2.imread(
            str(base_dir / "dataset_resample_2" / master_fname), cv2.IMREAD_UNCHANGED)
        depth1_raw = cv2.imread(
            str(base_dir / "dataset_resample_2" / input_fname), cv2.IMREAD_UNCHANGED)
        csv_path = base_dir / row["csv_path"]

        # GT transform
        R_gt, t_gt, n_gt = compute_gt_transform(csv_path, depth0_raw, depth1_raw)
        print(f"  GT: {n_gt} correspondences")

        # Crop + uint8 → SIFT/ORB + FLANN
        crop0 = depth0_raw[CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]
        crop1 = depth1_raw[CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]
        img0_u8 = depth_to_uint8(crop0)
        img1_u8 = depth_to_uint8(crop1)

        mkp0_crop, mkp1_crop, n_kp0, n_kp1 = detect_and_match(
            img0_u8, img1_u8, detector, args.method, ratio_th=args.ratio_th)
        n_match = len(mkp0_crop)
        print(f"  Keypoints: {n_kp0}/{n_kp1}, matches={n_match}")

        if n_match < 3:
            print("  Not enough matches. Skip.")
            continue

        # Crop → 원본 좌표
        mkp0_orig = mkp0_crop.copy()
        mkp0_orig[:, 0] += CROP_X0
        mkp0_orig[:, 1] += CROP_Y0
        mkp1_orig = mkp1_crop.copy()
        mkp1_orig[:, 0] += CROP_X0
        mkp1_orig[:, 1] += CROP_Y0

        # 3D 변환 + RANSAC
        pts0, vm0 = pixel_to_grid3d(mkp0_orig, depth0_raw)
        pts1, vm1 = pixel_to_grid3d(mkp1_orig, depth1_raw)
        both = vm0 & vm1
        pts0_v, pts1_v = pts0[both], pts1[both]
        n_3d = len(pts0_v)

        if n_3d < 3:
            print(f"  Not enough 3D matches ({n_3d}). Skip.")
            continue

        R_est, t_est, inliers = ransac_rigid(
            pts1_v, pts0_v, n_iter=args.ransac_iter, inlier_th=args.inlier_th)
        n_inl = int(inliers.sum())
        print(f"  3D={n_3d}, inliers={n_inl}")

        # 포인트 클라우드 샘플링 + transform 적용
        pc_master = sample_point_cloud(depth0_raw, args.n_sample_pts)
        pc_input  = sample_point_cloud(depth1_raw, args.n_sample_pts)
        pc_est = (R_est @ pc_input.T).T + t_est
        pc_gt  = (R_gt  @ pc_input.T).T + t_gt

        # RMSE
        P_30k = sample_point_cloud(depth1_raw, 30000)
        P_est_30k = (R_est @ P_30k.T).T + t_est
        P_gt_30k  = (R_gt  @ P_30k.T).T + t_gt
        rmse = float(np.sqrt(np.mean(np.sum((P_est_30k - P_gt_30k) ** 2, axis=1))))
        print(f"  RMSE={rmse:.4f}")

        pair_title = (f"[{args.method.upper()}+FLANN] "
                      f"{Path(master_fname).stem} vs {Path(input_fname).stem}")

        # 3×3 정합 비교
        plot_alignment(
            pc_master, pc_input, pc_est, pc_gt,
            pair_title,
            output_dir / f"align_{args.split}_{idx:05d}.png",
            n_inliers=n_inl, n_matches=n_match, rmse_est=rmse)

        # Overlay 확대
        plot_overlay_detail(
            pc_master, pc_est, pair_title,
            output_dir / f"overlay_{args.split}_{idx:05d}.png")

    print(f"\nDone! Output: {output_dir}/")


if __name__ == "__main__":
    main()
