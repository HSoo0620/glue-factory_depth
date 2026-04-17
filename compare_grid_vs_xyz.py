"""
Grid vs XYZ RANSAC 직접 비교 스크립트.

동일한 매칭 결과에 대해 Grid 좌표와 XYZ 좌표에서 각각 RANSAC을 돌리고
RMSE, inlier 비율, residual 분포를 비교한다.

사용법:
    conda run -n LightGlue python compare_grid_vs_xyz.py \
        --experiment 0407_resample2_iss_shot352_lg --descriptor shot \
        --indices 0 10 30 50 70 90
"""

import argparse
import torch
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.utils.tensor import batch_to_device

# ─── 상수 ──────────────────────────────────────
CROP_X0    = 1129
CROP_Y0    = 1081
CROP_SIZE  = 3502
CLIP_START = 0.1
CLIP_END   = 1000.0
FX         = 8001.39
FY         = 8001.39
CX_ORIG    = 2880.5
CY_ORIG    = 2880.5
GRID_DX    = 0.05
GRID_DY    = 0.05


# ─── 좌표 변환 ─────────────────────────────────

def pixel_to_grid3d(kpts_orig, depth_raw):
    N = kpts_orig.shape[0]
    pts = np.zeros((N, 3), dtype=np.float64)
    valid = np.zeros(N, dtype=bool)
    H, W = depth_raw.shape
    for i in range(N):
        u = int(np.clip(kpts_orig[i, 0], 0, W - 1))
        v = int(np.clip(kpts_orig[i, 1], 0, H - 1))
        d = float(depth_raw[v, u])
        if d <= 0:
            continue
        Z = CLIP_START + (d / 65535.0) * (CLIP_END - CLIP_START)
        pts[i] = [u * GRID_DX, v * GRID_DY, Z]
        valid[i] = True
    return pts, valid


def pixel_to_xyz(kpts_orig, depth_raw):
    N = kpts_orig.shape[0]
    pts = np.zeros((N, 3), dtype=np.float64)
    valid = np.zeros(N, dtype=bool)
    H, W = depth_raw.shape
    for i in range(N):
        u = int(np.clip(kpts_orig[i, 0], 0, W - 1))
        v = int(np.clip(kpts_orig[i, 1], 0, H - 1))
        d = float(depth_raw[v, u])
        if d <= 0:
            continue
        Z = CLIP_START + (d / 65535.0) * (CLIP_END - CLIP_START)
        X = (u - CX_ORIG) * Z / FX
        Y = (v - CY_ORIG) * Z / FY
        pts[i] = [X, Y, Z]
        valid[i] = True
    return pts, valid


# ─── Core math ──────────────────────────────────

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


def ransac_rigid(src, dst, n_iter=1000, inlier_th=5.0, seed=42):
    """RANSAC with fixed seed for reproducibility."""
    rng = np.random.RandomState(seed)
    N = src.shape[0]
    if N < 3:
        return np.eye(3), np.zeros(3), np.zeros(N, dtype=bool), np.full(N, np.inf)

    best_inliers = np.zeros(N, dtype=bool)
    best_R, best_t = np.eye(3), np.zeros(3)
    best_errors = np.full(N, np.inf)

    for _ in range(n_iter):
        idx = rng.choice(N, 3, replace=False)
        try:
            R, t = rigid_transform_svd(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        errors = np.linalg.norm((R @ src.T).T + t - dst, axis=1)
        inliers = errors < inlier_th
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_R, best_t = R, t
            best_errors = errors

    if best_inliers.sum() >= 3:
        best_R, best_t = rigid_transform_svd(src[best_inliers], dst[best_inliers])
        best_errors = np.linalg.norm((best_R @ src.T).T + best_t - dst, axis=1)

    return best_R, best_t, best_inliers, best_errors


def sample_point_cloud_grid(depth_raw, n_pts=30000, seed=42):
    rng = np.random.RandomState(seed)
    vs, us = np.where(depth_raw > 0)
    idx = rng.choice(len(vs), min(n_pts, len(vs)), replace=False)
    d = depth_raw[vs[idx], us[idx]].astype(np.float64)
    Z = CLIP_START + (d / 65535.0) * (CLIP_END - CLIP_START)
    return np.stack([us[idx] * GRID_DX, vs[idx] * GRID_DY, Z], axis=1)


def sample_point_cloud_xyz(depth_raw, n_pts=30000, seed=42):
    rng = np.random.RandomState(seed)
    vs, us = np.where(depth_raw > 0)
    idx = rng.choice(len(vs), min(n_pts, len(vs)), replace=False)
    d = depth_raw[vs[idx], us[idx]].astype(np.float64)
    Z = CLIP_START + (d / 65535.0) * (CLIP_END - CLIP_START)
    X = (us[idx].astype(np.float64) - CX_ORIG) * Z / FX
    Y = (vs[idx].astype(np.float64) - CY_ORIG) * Z / FY
    return np.stack([X, Y, Z], axis=1)


def compute_rmse(P_src, R_est, t_est, R_gt, t_gt):
    P_est = (R_est @ P_src.T).T + t_est
    P_gt  = (R_gt  @ P_src.T).T + t_gt
    return float(np.sqrt(np.mean(np.sum((P_est - P_gt) ** 2, axis=1))))


# ─── Model helpers ──────────────────────────────

def load_model(cp_path, device):
    cp = torch.load(cp_path, map_location="cpu")
    conf = OmegaConf.create(cp["conf"])
    model = get_model(conf.model.name)(conf.model).to(device)
    model.load_state_dict(cp["model"], strict=False)
    model.eval()
    print("Loaded:", cp_path, "epoch", cp["epoch"])
    return model, conf


def kpts_to_orig(kpts, image_size):
    scale = CROP_SIZE / image_size
    out = kpts * scale
    out[:, 0] += CROP_X0
    out[:, 1] += CROP_Y0
    return out


def extract_matches(pred, batch_idx, conf_th=0.0):
    kp0 = pred["keypoints0"][batch_idx].cpu().numpy()
    kp1 = pred["keypoints1"][batch_idx].cpu().numpy()
    m0  = pred["matches0"][batch_idx].cpu().numpy()
    sc  = pred["matching_scores0"][batch_idx].cpu().numpy()
    ok  = (m0 > -1) & (sc > conf_th)
    return kp0[ok], kp1[m0[ok]], int(ok.sum())


def get_dataset_and_collate(descriptor):
    if descriptor == "rops":
        from gluefactory.datasets.mitsubishi_resample2_iss_rops_dataset import (
            MitsubishiResample2ISSRoPSDataset as DS,
            resample2_iss_rops_collate_fn as fn)
    elif descriptor == "fpfh":
        from gluefactory.datasets.mitsubishi_resample2_iss_fpfh_dataset import (
            MitsubishiResample2ISSFPFHDataset as DS,
            resample2_iss_fpfh_collate_fn as fn)
    else:
        from gluefactory.datasets.mitsubishi_resample2_iss_shot_dataset import (
            MitsubishiResample2ISSSHOTDataset as DS,
            resample2_iss_shot_collate_fn as fn)
    return DS, fn


# ─── Main ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("Grid vs XYZ RANSAC comparison")
    parser.add_argument("--experiment", default="0407_resample2_iss_shot352_lg")
    parser.add_argument("--descriptor", default="shot", choices=["shot", "rops", "fpfh"])
    parser.add_argument("--indices", type=int, nargs="*", default=[0, 10, 30, 50, 70, 90])
    parser.add_argument("--inlier_th", type=float, default=5.0)
    parser.add_argument("--ransac_iter", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=1751)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    from gluefactory.utils.experiments import get_best_checkpoint
    cp_path = get_best_checkpoint(args.experiment)
    model, conf = load_model(cp_path, device)
    image_size = args.image_size

    base_dir = Path("gluefactory/datasets/mitsubishi")
    combo = pd.read_csv(base_dir / "outputs_txt" / "combination.csv")
    total = len(combo)
    test_start = total - 100
    combo_test = combo.iloc[test_start:].reset_index(drop=True)

    DS, collate_fn = get_dataset_and_collate(args.descriptor)
    dataset = DS(split="test", image_size=image_size)
    print("Dataset:", len(dataset), "pairs")
    print("inlier_th =", args.inlier_th)
    print()

    # Header
    hdr = "%5s | %10s %6s %6s | %10s %6s %6s | %9s" % (
        "Pair", "Grid RMSE", "inl", "inl%",
        "XYZ RMSE", "inl", "inl%", "XYZ/Grid")
    print(hdr)
    print("-" * len(hdr))

    all_results = []

    for idx in args.indices:
        row = combo_test.iloc[idx]
        master_f = Path(row["master_path"]).name
        input_f  = Path(row["input_path"]).name
        csv_path = base_dir / row["csv_path"]

        depth0 = cv2.imread(str(base_dir / "dataset_resample_2" / master_f), cv2.IMREAD_UNCHANGED)
        depth1 = cv2.imread(str(base_dir / "dataset_resample_2" / input_f),  cv2.IMREAD_UNCHANGED)

        # Inference
        sample = dataset[idx]
        batch = collate_fn([sample])
        batch = batch_to_device(batch, device)
        with torch.no_grad():
            pred = model(batch)

        mkp0, mkp1, n_match = extract_matches(pred, 0)
        mkp0_orig = kpts_to_orig(mkp0, image_size)
        mkp1_orig = kpts_to_orig(mkp1, image_size)

        # ── Grid 좌표 ──
        g0, gv0 = pixel_to_grid3d(mkp0_orig, depth0)
        g1, gv1 = pixel_to_grid3d(mkp1_orig, depth1)
        g_both = gv0 & gv1
        g0v, g1v = g0[g_both], g1[g_both]

        # ── XYZ 좌표 ──
        x0, xv0 = pixel_to_xyz(mkp0_orig, depth0)
        x1, xv1 = pixel_to_xyz(mkp1_orig, depth1)
        x_both = xv0 & xv1
        x0v, x1v = x0[x_both], x1[x_both]

        n_3d = len(g0v)  # should be same as len(x0v)

        if n_3d < 3:
            print("%5d | SKIP (only %d 3D pts)" % (idx, n_3d))
            continue

        # ── GT transform (Grid) ──
        df_gt = pd.read_csv(csv_path)
        ok = df_gt["occluded"] == False
        m_xy = df_gt.loc[ok, ["master_x", "master_y"]].values.astype(np.float64)
        i_xy = df_gt.loc[ok, ["input_x",  "input_y"]].values.astype(np.float64)
        H0, W0 = depth0.shape
        H1, W1 = depth1.shape

        gt_master_g, gt_input_g = [], []
        gt_master_x, gt_input_x = [], []
        for j in range(len(m_xy)):
            mu = int(np.clip(m_xy[j,0], 0, W0-1))
            mv = int(np.clip(m_xy[j,1], 0, H0-1))
            iu = int(np.clip(i_xy[j,0], 0, W1-1))
            iv = int(np.clip(i_xy[j,1], 0, H1-1))
            d0, d1 = float(depth0[mv,mu]), float(depth1[iv,iu])
            if d0 <= 0 or d1 <= 0:
                continue
            Z0 = CLIP_START + (d0/65535.0)*(CLIP_END-CLIP_START)
            Z1 = CLIP_START + (d1/65535.0)*(CLIP_END-CLIP_START)
            gt_master_g.append([mu*GRID_DX, mv*GRID_DY, Z0])
            gt_input_g.append([iu*GRID_DX, iv*GRID_DY, Z1])
            gt_master_x.append([(mu-CX_ORIG)*Z0/FX, (mv-CY_ORIG)*Z0/FY, Z0])
            gt_input_x.append([(iu-CX_ORIG)*Z1/FX, (iv-CY_ORIG)*Z1/FY, Z1])

        Rgt_g, tgt_g = rigid_transform_svd(np.array(gt_input_g), np.array(gt_master_g))
        Rgt_x, tgt_x = rigid_transform_svd(np.array(gt_input_x), np.array(gt_master_x))

        # ── RANSAC (same seed) ──
        Rg, tg, inl_g, err_g = ransac_rigid(g1v, g0v, args.ransac_iter, args.inlier_th, seed=42)
        Rx, tx, inl_x, err_x = ransac_rigid(x1v, x0v, args.ransac_iter, args.inlier_th, seed=42)

        n_inl_g = int(inl_g.sum())
        n_inl_x = int(inl_x.sum())

        # ── RMSE ──
        Pg = sample_point_cloud_grid(depth1, 30000, seed=42)
        Px = sample_point_cloud_xyz(depth1, 30000, seed=42)
        rmse_g = compute_rmse(Pg, Rg, tg, Rgt_g, tgt_g)
        rmse_x = compute_rmse(Px, Rx, tx, Rgt_x, tgt_x)

        ratio = rmse_x / rmse_g if rmse_g > 0 else float('inf')
        pct_g = 100.0 * n_inl_g / n_3d
        pct_x = 100.0 * n_inl_x / n_3d

        print("%5d | %10.4f %6d %5.1f%% | %10.4f %6d %5.1f%% | %9.2f" % (
            idx, rmse_g, n_inl_g, pct_g, rmse_x, n_inl_x, pct_x, ratio))

        all_results.append({
            "idx": idx, "n_3d": n_3d,
            "rmse_grid": rmse_g, "inl_grid": n_inl_g, "pct_grid": pct_g,
            "rmse_xyz": rmse_x, "inl_xyz": n_inl_x, "pct_xyz": pct_x,
            "ratio": ratio,
        })

        # Per-axis residual analysis (XYZ space)
        aligned_x = (Rx @ x1v.T).T + tx
        dx = np.abs(aligned_x - x0v)
        print("       XYZ per-axis residual P50 (X=%.2f Y=%.2f Z=%.2f mm)" % (
            np.median(dx[:,0]), np.median(dx[:,1]), np.median(dx[:,2])))

        # Per-axis residual analysis (Grid space)
        aligned_g = (Rg @ g1v.T).T + tg
        dg = np.abs(aligned_g - g0v)
        print("       Grid per-axis residual P50 (gx=%.4f gy=%.4f gz=%.2f)" % (
            np.median(dg[:,0]), np.median(dg[:,1]), np.median(dg[:,2])))

        # Grid coordinate range analysis (첫 pair만)
        if idx == args.indices[0]:
            print("\n  [Coordinate Range Analysis - pair %d]" % idx)
            print("  Grid: x=[%.1f, %.1f]  y=[%.1f, %.1f]  z=[%.1f, %.1f]" % (
                g0v[:,0].min(), g0v[:,0].max(),
                g0v[:,1].min(), g0v[:,1].max(),
                g0v[:,2].min(), g0v[:,2].max()))
            print("  XYZ:  X=[%.1f, %.1f]  Y=[%.1f, %.1f]  Z=[%.1f, %.1f] mm" % (
                x0v[:,0].min(), x0v[:,0].max(),
                x0v[:,1].min(), x0v[:,1].max(),
                x0v[:,2].min(), x0v[:,2].max()))
            print("  inlier_th=%.1f -> Grid에서 gx 허용 pixel=%.0f, XYZ에서 X 허용 pixel(at meanZ)=%.0f" % (
                args.inlier_th,
                args.inlier_th / GRID_DX,
                args.inlier_th * FX / np.mean(g0v[:,2])))
            print()

    # ── Summary ──
    if all_results:
        print()
        print("=" * 60)
        rmse_g_avg = np.mean([r["rmse_grid"] for r in all_results])
        rmse_x_avg = np.mean([r["rmse_xyz"]  for r in all_results])
        inl_g_avg  = np.mean([r["pct_grid"]  for r in all_results])
        inl_x_avg  = np.mean([r["pct_xyz"]   for r in all_results])
        ratio_avg  = np.mean([r["ratio"]      for r in all_results])
        print("Mean Grid RMSE: %.4f   Mean XYZ RMSE: %.4f   Mean ratio: %.2f" % (
            rmse_g_avg, rmse_x_avg, ratio_avg))
        print("Mean Grid inlier%%: %.1f%%   Mean XYZ inlier%%: %.1f%%" % (
            inl_g_avg, inl_x_avg))


if __name__ == "__main__":
    main()
