"""
ISS+SHOT/FPFH/RoPS FLANN Baseline Registration 평가.

학습 기반 matcher 없이, 캐시된 ISS keypoints + SHOT/FPFH/RoPS descriptors를
OpenCV FLANN(KDTree)으로 매칭하여 Registration baseline 성능을 평가한다.

사용법:
    python eval_registration_iss_shot_flann.py --descriptor shot --indices 0 10 50 90
    python eval_registration_iss_shot_flann.py --descriptor fpfh --indices 0 10 50 90
    python eval_registration_iss_shot_flann.py --descriptor rops --ratio_th 0.8
"""

import argparse
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# --- 상수 ---
CROP_X0 = 1129
CROP_Y0 = 1081
CROP_SIZE = 3502
CLIP_START = 0.1
CLIP_END = 1000.0
ORIG_SIZE = 5761
GRID_DX = 0.05
GRID_DY = 0.05


# --- Core math functions ---

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

    pts_master, pts_input = [], []
    for i in range(len(master_xy)):
        mu = int(np.clip(master_xy[i, 0], 0, W0 - 1))
        mv = int(np.clip(master_xy[i, 1], 0, H0 - 1))
        iu = int(np.clip(input_xy[i, 0], 0, W1 - 1))
        iv = int(np.clip(input_xy[i, 1], 0, H1 - 1))

        d0 = float(depth0_raw[mv, mu])
        d1 = float(depth1_raw[iv, iu])
        if d0 <= 0 or d1 <= 0:
            continue

        dr0 = CLIP_START + (d0 / 65535.0) * (CLIP_END - CLIP_START)
        dr1 = CLIP_START + (d1 / 65535.0) * (CLIP_END - CLIP_START)
        pts_master.append([mu * GRID_DX, mv * GRID_DY, dr0])
        pts_input.append([iu * GRID_DX, iv * GRID_DY, dr1])

    pts_master = np.array(pts_master)
    pts_input = np.array(pts_input)
    n_pts = len(pts_master)

    if n_pts < 3:
        return np.eye(3), np.zeros(3), n_pts

    R_gt, t_gt = rigid_transform_svd(pts_input, pts_master)
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


def compute_transform_rmse(P_src, R_est, t_est, R_gt, t_gt):
    P_est = (R_est @ P_src.T).T + t_est
    P_gt = (R_gt @ P_src.T).T + t_gt
    return float(np.sqrt(np.mean(np.sum((P_est - P_gt) ** 2, axis=1))))


# --- Descriptor config ---

DESCRIPTOR_CONFIG = {
    "shot":  {"key": "shot_descriptors",  "cache_dir": "iss_shot352_resample2_cache"},
    "fpfh":  {"key": "fpfh_descriptors",  "cache_dir": "iss_fpfh_resample2_cache_r5.0_xyz"},
    "rops":  {"key": "rops_descriptors",  "cache_dir": "iss_rops135_resample2_cache"},
}


# --- Cache loading & FLANN matching ---

def load_cache(cache_dir, depth_fname, desc_key="shot_descriptors"):
    """캐시 npz 로드. Returns: keypoints (n_valid, 2), descriptors (n_valid, D).
    depth_fname: 'depth_raw_XXXX.png' -> 'depth_raw_XXXX.npz'
    """
    npz_name = Path(depth_fname).stem + ".npz"
    npz_path = cache_dir / npz_name
    data = np.load(str(npz_path))
    n_valid = int(data["n_valid"])
    kp = data["keypoints"][:n_valid]
    desc = data[desc_key][:n_valid]
    return kp, desc


def flann_match(desc0, desc1, ratio_th=0.75):
    """OpenCV FLANN KDTree + Lowe's ratio test.
    Returns: idx0, idx1 (matched descriptor indices)
    """
    if len(desc0) < 2 or len(desc1) < 2:
        return np.array([], dtype=int), np.array([], dtype=int)

    desc0_f32 = desc0.astype(np.float32)
    desc1_f32 = desc1.astype(np.float32)

    index_params = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    matches = matcher.knnMatch(desc0_f32, desc1_f32, k=2)

    idx0, idx1 = [], []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < ratio_th * n.distance:
                idx0.append(m.queryIdx)
                idx1.append(m.trainIdx)

    return np.array(idx0, dtype=int), np.array(idx1, dtype=int)


def kpts_to_orig(kpts, image_size):
    """crop+resize 좌표(1751px) -> 원본(5761px)."""
    scale = CROP_SIZE / image_size
    kpts_orig = kpts.copy().astype(np.float64)
    kpts_orig *= scale
    kpts_orig[:, 0] += CROP_X0
    kpts_orig[:, 1] += CROP_Y0
    return kpts_orig


# --- Evaluation pipeline ---

def evaluate_one_direction(mkp_src_orig, mkp_dst_orig,
                           depth_src_raw, depth_dst_raw,
                           R_gt, t_gt, args, label="Forward"):
    """한 방향 Registration 평가."""
    n_matches = len(mkp_src_orig)
    if n_matches < 3:
        print(f"  {label}: not enough matches ({n_matches}). Skip.")
        return None

    pts_src, vm_src = pixel_to_grid3d(mkp_src_orig, depth_src_raw)
    pts_dst, vm_dst = pixel_to_grid3d(mkp_dst_orig, depth_dst_raw)
    both = vm_src & vm_dst
    pts_s, pts_d = pts_src[both], pts_dst[both]
    n_3d = len(pts_s)

    if n_3d < 3:
        print(f"  {label}: not enough 3D points ({n_3d}). Skip.")
        return None

    R_est, t_est, inliers = ransac_rigid(
        pts_s, pts_d, n_iter=args.ransac_iter, inlier_th=args.inlier_th)
    n_inl = int(inliers.sum())

    pts_aligned = (R_est @ pts_s.T).T + t_est
    err_before = np.linalg.norm(pts_s - pts_d, axis=1).mean()
    err_after = np.linalg.norm(pts_aligned - pts_d, axis=1).mean()

    P_src_cloud = sample_point_cloud(depth_src_raw, args.n_sample_pts)
    rmse = compute_transform_rmse(P_src_cloud, R_est, t_est, R_gt, t_gt)

    print(f"  {label}: matches={n_matches}, 3D={n_3d}, inliers={n_inl}")
    print(f"    err: {err_before:.3f} -> {err_after:.3f}, RMSE={rmse:.4f}")

    return {
        "rmse": rmse,
        "n_matches": n_matches,
        "n_3d": n_3d,
        "n_inliers": n_inl,
        "err_before": err_before,
        "err_after": err_after,
        "R_est": R_est,
        "t_est": t_est,
    }


def evaluate_pair(cache_dir, master_fname, input_fname,
                  depth0_raw, depth1_raw, gt_csv_path,
                  image_size, args, desc_key="shot_descriptors"):
    """단일 pair 양방향 평가 (FLANN baseline)."""
    # 캐시 로드
    kp0, desc0 = load_cache(cache_dir, master_fname, desc_key)
    kp1, desc1 = load_cache(cache_dir, input_fname, desc_key)
    n_kp0, n_kp1 = len(kp0), len(kp1)
    print(f"  Keypoints: view0={n_kp0}, view1={n_kp1}")

    # GT transform (input -> master)
    R_gt, t_gt, n_gt = compute_gt_transform(gt_csv_path, depth0_raw, depth1_raw)
    print(f"  GT SVD: {n_gt} non-occluded pairs used")

    # --- Forward: (view0=master, view1=input) ---
    idx0, idx1 = flann_match(desc0, desc1, ratio_th=args.ratio_th)
    mkp0_crop = kp0[idx0]  # master keypoints
    mkp1_crop = kp1[idx1]  # input keypoints
    mkp0_orig = kpts_to_orig(mkp0_crop, image_size)
    mkp1_orig = kpts_to_orig(mkp1_crop, image_size)

    fwd = evaluate_one_direction(
        mkp1_orig, mkp0_orig,  # input -> master
        depth1_raw, depth0_raw,
        R_gt, t_gt, args, label="Forward")

    if fwd is None:
        return None

    # --- Reverse: (view0=input, view1=master) ---
    idx1r, idx0r = flann_match(desc1, desc0, ratio_th=args.ratio_th)
    mkp1r_crop = kp1[idx1r]
    mkp0r_crop = kp0[idx0r]
    mkp1r_orig = kpts_to_orig(mkp1r_crop, image_size)
    mkp0r_orig = kpts_to_orig(mkp0r_crop, image_size)

    R_gt_rev = R_gt.T
    t_gt_rev = -R_gt.T @ t_gt

    rev = evaluate_one_direction(
        mkp0r_orig, mkp1r_orig,  # master -> input
        depth0_raw, depth1_raw,
        R_gt_rev, t_gt_rev, args, label="Reverse")

    if rev is None:
        return None

    rmse_bi = (fwd["rmse"] + rev["rmse"]) / 2.0
    print(f"  Bidirectional RMSE: {rmse_bi:.4f}")

    return {
        "rmse_fwd": fwd["rmse"],
        "rmse_rev": rev["rmse"],
        "rmse_bi": rmse_bi,
        "n_kp0": n_kp0,
        "n_kp1": n_kp1,
        "n_matches_fwd": fwd["n_matches"],
        "n_matches_rev": rev["n_matches"],
        "n_3d_fwd": fwd["n_3d"],
        "n_inliers_fwd": fwd["n_inliers"],
        "n_inliers_rev": rev["n_inliers"],
        "err_before_fwd": fwd["err_before"],
        "err_after_fwd": fwd["err_after"],
        "n_gt_pts": n_gt,
        "R_est_fwd": fwd["R_est"],
        "t_est_fwd": fwd["t_est"],
        "depth0_raw": depth0_raw,
        "depth1_raw": depth1_raw,
    }


# --- Visualization ---

def warp_depth_to_master(depth1_raw, R_est, t_est, out_shape=None):
    """Input depth를 추정된 R,t로 변환하여 master 좌표계에 투영."""
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


def visualize_registration(depth0_raw, depth1_raw, R_est, t_est,
                           output_path, n_matches, n_3d,
                           n_inliers, err_before, err_after,
                           title_prefix="ISS+SHOT+FLANN"):
    """3-panel 시각화: Master(cyan) / Input(red) / Overlay."""
    warped_input = warp_depth_to_master(
        depth1_raw, R_est, t_est, out_shape=depth0_raw.shape)

    H, W = depth0_raw.shape
    d0_vis = depth0_raw.astype(np.float32) / 65535.0
    d1_vis = depth1_raw.astype(np.float32) / 65535.0
    if d1_vis.shape != (H, W):
        d1_vis = cv2.resize(d1_vis, (W, H), interpolation=cv2.INTER_NEAREST)

    master_color = np.zeros((H, W, 3), dtype=np.float32)
    master_color[:, :, 1] = d0_vis
    master_color[:, :, 2] = d0_vis

    input_color = np.zeros((H, W, 3), dtype=np.float32)
    input_color[:, :, 0] = d1_vis

    overlay = np.zeros((H, W, 3), dtype=np.float32)
    overlay[:, :, 1] = d0_vis
    overlay[:, :, 2] = d0_vis
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], warped_input)

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
        f"[{title_prefix}]  Matches: {n_matches}  |  3D: {n_3d}  |  "
        f"Inliers: {n_inliers}/{n_3d}  |  "
        f"Error: {err_before:.2f} -> {err_after:.2f}",
        fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="ISS+SHOT/FPFH/RoPS FLANN Baseline Registration Eval")
    parser.add_argument("--cache_dir", type=str,
                        default="iss_shot352_resample2_cache",
                        help="Cache directory name under mitsubishi/")
    parser.add_argument("--descriptor", type=str, default="shot",
                        choices=["shot", "fpfh", "rops"],
                        help="Descriptor type")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--image_size", type=int, default=1751)
    parser.add_argument("--ratio_th", type=float, default=0.75,
                        help="Lowe's ratio test threshold")
    parser.add_argument("--inlier_th", type=float, default=5.0)
    parser.add_argument("--ransac_iter", type=int, default=1000)
    parser.add_argument("--n_sample_pts", type=int, default=30000)
    args = parser.parse_args()

    # Auto-set cache_dir from descriptor if not explicitly provided
    desc_conf = DESCRIPTOR_CONFIG[args.descriptor]
    if args.cache_dir == "iss_shot352_resample2_cache":  # default unchanged
        cache_dir_name = desc_conf["cache_dir"]
    else:
        cache_dir_name = args.cache_dir
    desc_key = desc_conf["key"]

    output_dir = Path(
        args.output_dir if args.output_dir
        else f"results/registration/iss_{args.descriptor}_flann_eval")
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

    print(f"Method: ISS+{args.descriptor.upper()}+FLANN (cache: {cache_dir_name})")
    print(f"Split: {args.split} ({len(combo_split)} pairs)")
    print(f"ratio_th={args.ratio_th}, image_size={args.image_size}")
    print(f"RANSAC: inlier_th={args.inlier_th}, iter={args.ransac_iter}")

    if args.indices is not None:
        indices = args.indices
    else:
        indices = sorted(np.random.choice(
            len(combo_split), min(args.num_samples, len(combo_split)),
            replace=False))

    print(f"Testing {len(indices)} pairs: {indices}")

    results = []
    for idx in tqdm(indices, desc=f"ISS+{args.descriptor.upper()}+FLANN"):
        print(f"\n--- Pair {idx} ---")
        row = combo_split.iloc[idx]

        master_fname = Path(row["master_path"]).name
        input_fname = Path(row["input_path"]).name
        depth0_raw = cv2.imread(
            str(base_dir / "dataset_resample_2" / master_fname),
            cv2.IMREAD_UNCHANGED)
        depth1_raw = cv2.imread(
            str(base_dir / "dataset_resample_2" / input_fname),
            cv2.IMREAD_UNCHANGED)
        csv_path = base_dir / row["csv_path"]

        res = evaluate_pair(
            cache_dir, master_fname, input_fname,
            depth0_raw, depth1_raw, csv_path,
            args.image_size, args, desc_key)

        if res is None:
            continue

        title_prefix = f"ISS+{args.descriptor.upper()}+FLANN"
        visualize_registration(
            res["depth0_raw"], res["depth1_raw"],
            res["R_est_fwd"], res["t_est_fwd"],
            output_dir / f"reg_{args.split}_{idx:05d}.png",
            n_matches=res["n_matches_fwd"],
            n_3d=res["n_3d_fwd"],
            n_inliers=res["n_inliers_fwd"],
            err_before=res["err_before_fwd"],
            err_after=res["err_after_fwd"],
            title_prefix=title_prefix)
        print(f"  Saved: reg_{args.split}_{idx:05d}.png")

        results.append({
            "pair_idx": idx,
            "master": master_fname,
            "input": input_fname,
            **{k: v for k, v in res.items()
               if k not in ("R_est_fwd", "t_est_fwd",
                             "depth0_raw", "depth1_raw",
                             "err_before_fwd", "err_after_fwd")},
        })

    if results:
        df = pd.DataFrame(results)
        csv_out = output_dir / f"rmse_{args.split}.csv"
        df.to_csv(csv_out, index=False)
        print(f"\n{'='*60}")
        print(f"Method: ISS+{args.descriptor.upper()}+FLANN")
        print(f"Results saved: {csv_out}")
        print(f"Mean Bidirectional RMSE: {df['rmse_bi'].mean():.4f}")
        print(df[["pair_idx", "n_kp0", "n_kp1", "n_matches_fwd",
                   "rmse_fwd", "rmse_rev", "rmse_bi"]].to_string(index=False))

    print(f"\nDone! Output: {output_dir}/")


if __name__ == "__main__":
    main()
