"""Scanned roi13 × Synthetic master1: ISS+SHOT352+LG 매칭 & 정합.

Model : iss_shot_v1_20260415_dim352/checkpoint_best.tar
Scanned 전처리: floor removal → bilateral filter (vis_new 동일)
Descriptor: SHOT352 on-the-fly (voxel=1mm, nr=20mm, sr=40mm — 학습과 동일)

ISS 모드:
  --iss_mode mm   : mm 공간 ISS (학습과 동일, extract_iss_on_dense)
  --iss_mode uvd  : (u,v,depth_scaled) 픽셀 공간 ISS (등방성, 더 선택적)

두 방향 추론:
  A) view0=synthetic master, view1=scanned  (forward)
  B) view0=scanned, view1=synthetic master  (reverse)

Results → experiments/v1_20260415/results/scanned_vs_shot_dim352_{iss_mode}/

Usage:
    conda activate LightGlue
    cd /home/jhs/work/Registration/glue-factory_depth
    python experiments/v1_20260415/infer_scanned_vs_master_shot352.py --iss_mode uvd
    python experiments/v1_20260415/infer_scanned_vs_master_shot352.py --iss_mode mm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pybind_shot_linux"))
sys.path.insert(0, str(ROOT / "experiments/v1_20260415/precompute"))

import shot_module  # noqa: E402
import helpers as h  # noqa: E402
from gluefactory.models import get_model  # noqa: E402
from gluefactory.utils.tensor import batch_to_device  # noqa: E402
from gluefactory.datasets.new_dataset.iss_detection import (  # noqa: E402
    build_iss_pcd_uvd_scaled, detect_iss_keypoints, select_keypoints,
)

SCAN_PATH = ROOT / "gluefactory/datasets/scanned/roi13_zmap 1.png"
MASTER_PATH = ROOT / "gluefactory/datasets/scanned/blender_master1/master.png"
CHECKPOINT = ROOT / "outputs/training/iss_shot_v1_20260415_dim352/checkpoint_best.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BILAT_D = 5
BILAT_SIGMA_C = 100.0
BILAT_SIGMA_S = 3.0

RANSAC_ITER = 1000
INLIER_TH = 5.0
N_SAMPLE_PTS = 15000


# ═══════════════════════════════════════════════════════════════════
# Preprocessing (scanned only)
# ═══════════════════════════════════════════════════════════════════

def mask_scanned_table(canvas, band_fraction=0.05, bin_width=50):
    mask = canvas > 0
    if mask.sum() == 0:
        return canvas.copy(), {"applied": False}
    values = canvas[mask]
    bins = np.arange(0, 65536 + bin_width, bin_width)
    hist, edges = np.histogram(values, bins=bins)
    peak_bin = int(hist.argmax())
    peak_height = int(hist[peak_bin])
    threshold = peak_height * band_fraction
    left = peak_bin
    while left > 0 and hist[left - 1] > threshold:
        left -= 1
    right = peak_bin
    while right < len(hist) - 1 and hist[right + 1] > threshold:
        right += 1
    band_low = int(edges[left])
    band_high = int(edges[right + 1])
    out = canvas.copy()
    sel = (canvas >= band_low) & (canvas < band_high)
    out[sel] = 0
    info = {
        "applied": True,
        "peak_center": int((edges[peak_bin] + edges[peak_bin + 1]) / 2),
        "band_low": band_low, "band_high": band_high,
        "fraction_masked": float(sel.sum() / mask.sum()),
    }
    return out, info


def apply_bilateral(zmap_u16):
    f32 = zmap_u16.astype(np.float32)
    mask0 = zmap_u16 == 0
    filtered = cv2.bilateralFilter(f32, d=BILAT_D,
                                   sigmaColor=BILAT_SIGMA_C,
                                   sigmaSpace=BILAT_SIGMA_S)
    filtered[mask0] = 0.0
    return np.clip(filtered, 0, 65535).astype(np.uint16)


# ═══════════════════════════════════════════════════════════════════
# ISS + SHOT pipeline
# ═══════════════════════════════════════════════════════════════════

ERODE_BOUNDARY = 5


def zmap_to_pcd_mm(zmap, erode_boundary=ERODE_BOUNDARY):
    mask = (zmap > 0).astype(np.uint8)
    if erode_boundary > 0:
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=erode_boundary)
    vs, us = np.where(mask > 0)
    zs = zmap[vs, us].astype(np.float32)
    return np.column_stack([
        us.astype(np.float32) * h.LATERAL_MM,
        vs.astype(np.float32) * h.TRANSPORT_MM,
        zs * h.VERTICAL_MM,
    ]).astype(np.float32)


def _prepare_iss_mm(zmap, label, seed):
    """mm 공간 ISS (학습과 동일)."""
    pts_mm = zmap_to_pcd_mm(zmap)
    print(f"  [{label}] PCD: {len(pts_mm)} pts  "
          f"Z: {pts_mm[:, 2].min():.1f}–{pts_mm[:, 2].max():.1f} mm")

    pcd_vox = h.voxel_downsample(pts_mm, voxel_size=h.VOXEL_SIZE)
    vox_pts = np.asarray(pcd_vox.points, dtype=np.float32)

    kp_xyz_iss = h.extract_iss_on_dense(pts_mm)
    kp_xyz, _, kp_scores, n_iss, n_valid = h.subsample_or_pad_keypoints(
        kp_xyz_iss, pcd_vox, max_n=h.MAX_KEYPOINTS, seed=seed,
    )
    kp_uv = h.kp_xyz_mm_to_uv_padded(kp_xyz)
    return kp_uv, kp_scores, kp_xyz, vox_pts, n_iss, n_valid


def _prepare_iss_uvd(zmap, label, seed):
    """(u,v,depth_scaled) 픽셀 공간 ISS (등방성)."""
    rng = np.random.default_rng(seed)
    pcd_iss, erode_mask, _, _ = build_iss_pcd_uvd_scaled(
        zmap, erode_boundary=5,
    )
    iss_kp_3d = detect_iss_keypoints(
        pcd_iss, gamma_21=0.5, gamma_32=0.5, min_neighbors=5,
    )
    kp_uv, kp_scores, n_valid, kp_uv_orig = select_keypoints(
        iss_kp_3d, zmap, max_num_keypoints=h.MAX_KEYPOINTS,
        erode_mask=erode_mask, resize_factor=1.0, rng=rng,
    )
    n_iss = len(iss_kp_3d)

    H, W = zmap.shape
    kp_xyz = np.zeros((h.MAX_KEYPOINTS, 3), dtype=np.float32)
    for i in range(n_valid):
        u = int(round(float(kp_uv_orig[i, 0])))
        v = int(round(float(kp_uv_orig[i, 1])))
        u = max(0, min(u, W - 1))
        v = max(0, min(v, H - 1))
        raw = zmap[v, u]
        kp_xyz[i] = [u * h.LATERAL_MM, v * h.TRANSPORT_MM, raw * h.VERTICAL_MM]

    pts_mm = zmap_to_pcd_mm(zmap)
    pcd_vox = h.voxel_downsample(pts_mm, voxel_size=h.VOXEL_SIZE)
    vox_pts = np.asarray(pcd_vox.points, dtype=np.float32)

    return kp_uv, kp_scores, kp_xyz, vox_pts, n_iss, n_valid


def prepare_iss_shot(zmap, label="", seed=0, iss_mode="mm"):
    """ISS detection + SHOT352 on-the-fly.

    Returns: kp_uv (512,2), kp_scores (512,), descriptors (512,352),
             kp_xyz_mm (512,3), n_iss, n_valid
    """
    if iss_mode == "mm":
        kp_uv, kp_scores, kp_xyz, vox_pts, n_iss, n_valid = \
            _prepare_iss_mm(zmap, label, seed)
    else:
        kp_uv, kp_scores, kp_xyz, vox_pts, n_iss, n_valid = \
            _prepare_iss_uvd(zmap, label, seed)

    print(f"  [{label}] ISS({iss_mode}): {n_iss} detected → {n_valid} valid  "
          f"| Voxel: {len(vox_pts)} pts")

    result = shot_module.extract_shot_at_keypoints(
        vox_pts, kp_xyz.astype(np.float32),
        voxel_size=h.VOXEL_SIZE,
        normal_radius=h.NORMAL_RADIUS,
        shot_radius=h.SHOT_RADIUS,
    )
    descriptors = np.nan_to_num(result["descriptors"].astype(np.float32), nan=0.0)
    n_shot_valid = int(result["valid_mask"].sum())
    print(f"  [{label}] SHOT: {n_shot_valid}/{h.MAX_KEYPOINTS} valid descriptors")

    return kp_uv, kp_scores, descriptors, kp_xyz, n_iss, n_valid


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

def load_model(checkpoint_path, device):
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    conf = OmegaConf.create(cp["conf"])
    model = get_model(conf.model.name)(conf.model).to(device)
    miss, unexp = model.load_state_dict(cp["model"], strict=False)
    if miss or unexp:
        print(f"[warn] load: missing={len(miss)} unexpected={len(unexp)}")
    model.eval()
    print(f"Loaded: {Path(checkpoint_path).name} (epoch {cp.get('epoch', '?')})")
    return model


def make_view(zmap, kp_uv, kp_scores, descriptors, pad_h, pad_w):
    H, W = zmap.shape
    padded = np.zeros((pad_h, pad_w), dtype=zmap.dtype)
    padded[:H, :W] = zmap
    img = padded.astype(np.float32) / 65535.0
    return {
        "image": torch.from_numpy(img).unsqueeze(0),       # (1, pad_h, pad_w)
        "image_size": torch.tensor([pad_h, pad_w], dtype=torch.long),
        "keypoints": torch.from_numpy(kp_uv).float(),      # (512, 2)
        "keypoint_scores": torch.from_numpy(kp_scores).float(),
        "descriptors": torch.from_numpy(descriptors).float(),
    }


def collate_single_pair(view0, view1):
    return {
        "view0": {k: v.unsqueeze(0) for k, v in view0.items()},
        "view1": {k: v.unsqueeze(0) for k, v in view1.items()},
    }


@torch.no_grad()
def run_inference(model, batch, device):
    batch = batch_to_device(batch, device)
    pred = model(batch)
    return pred


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def visualize_matches(pred, view0_img, view1_img, output_path,
                      title_extra="",
                      view0_label="View 0 (Synthetic master)",
                      view1_label="View 1 (Scanned roi13)"):
    kp0 = pred["keypoints0"][0].cpu().numpy()
    kp1 = pred["keypoints1"][0].cpu().numpy()
    m0 = pred["matches0"][0].cpu().numpy()
    sc = pred["matching_scores0"][0].cpu().numpy()

    valid = m0 > -1
    mkp0 = kp0[valid]
    mkp1 = kp1[m0[valid]]
    scores = sc[valid]
    n_total = int(valid.sum())

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=100)
    for ax, img, kp, title in [
        (axes[0], view0_img, kp0, view0_label),
        (axes[1], view1_img, kp1, view1_label),
    ]:
        ax.imshow(img, cmap="gray")
        ax.scatter(kp[:, 0], kp[:, 1], c="royalblue", s=3, alpha=0.3, linewidths=0)
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()

    for i in range(n_total):
        line = matplotlib.patches.ConnectionPatch(
            xyA=(float(mkp0[i, 0]), float(mkp0[i, 1])),
            xyB=(float(mkp1[i, 0]), float(mkp1[i, 1])),
            coordsA=axes[0].transData, coordsB=axes[1].transData,
            axesA=axes[0], axesB=axes[1],
            color="skyblue", linewidth=0.8, alpha=0.6,
        )
        fig.add_artist(line)

    info = (f"KP: {kp0.shape[0]} / {kp1.shape[0]}  |  "
            f"Matches: {n_total}")
    if n_total > 0:
        info += (f"  |  Score: min={scores.min():.3f}  "
                 f"mean={scores.mean():.3f}  max={scores.max():.3f}")
    if title_extra:
        info += f"  |  {title_extra}"
    fig.suptitle(info, fontsize=11, y=0.02)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.2, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}  |  matches={n_total}")


# ═══════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════

def _rigid_svd(P_src, P_dst):
    cs, cd = P_src.mean(0), P_dst.mean(0)
    H_mat = (P_src - cs).T @ (P_dst - cd)
    U, _, Vt = np.linalg.svd(H_mat)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R, cd - R @ cs


def _ransac_rigid(src, dst, n_iter=RANSAC_ITER, inlier_th=INLIER_TH):
    N = src.shape[0]
    if N < 3:
        return np.eye(3), np.zeros(3), np.zeros(N, dtype=bool)
    best = np.zeros(N, dtype=bool)
    bR, bt = np.eye(3), np.zeros(3)
    for _ in range(n_iter):
        idx = np.random.choice(N, 3, replace=False)
        try:
            R, t = _rigid_svd(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        err = np.linalg.norm((R @ src.T).T + t - dst, axis=1)
        inl = err < inlier_th
        if inl.sum() > best.sum():
            best, bR, bt = inl.copy(), R, t
    if best.sum() >= 3:
        bR, bt = _rigid_svd(src[best], dst[best])
    return bR, bt, best


def _sample_pcd_mm(zmap, n_pts, seed=42):
    rng_ = np.random.RandomState(seed)
    vs, us = np.where(zmap > 0)
    if len(vs) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    idx = rng_.choice(len(vs), min(n_pts, len(vs)), replace=False)
    pts = np.column_stack([
        us[idx].astype(np.float64) * h.LATERAL_MM,
        vs[idx].astype(np.float64) * h.TRANSPORT_MM,
        zmap[vs[idx], us[idx]].astype(np.float64) * h.VERTICAL_MM,
    ])
    return pts


def registration_and_viz(pred, dst_zmap, src_zmap,
                         dst_kp_xyz, src_kp_xyz,
                         out_reg, out_overlay,
                         dst_label="master (synthetic)",
                         src_label="input (scan)"):
    """RANSAC rigid registration: src → dst. view0=dst, view1=src."""
    m0 = pred["matches0"][0].cpu().numpy()
    sc = pred["matching_scores0"][0].cpu().numpy()

    valid = (m0 > -1) & (sc > 0.0)
    idx0 = np.where(valid)[0]
    idx1 = m0[valid].astype(int)

    pts_dst = dst_kp_xyz[idx0]
    pts_src = src_kp_xyz[idx1]

    both = (np.linalg.norm(pts_dst, axis=1) > 0) & \
           (np.linalg.norm(pts_src, axis=1) > 0)
    p_dst = pts_dst[both]
    p_src = pts_src[both]
    n_3d = len(p_dst)
    print(f"  3D match pairs: {n_3d}")

    if n_3d < 3:
        print("  3D 매칭 부족으로 정합 생략.")
        return

    R_est, t_est, inl = _ransac_rigid(p_src, p_dst)
    n_inl = int(inl.sum())
    err_inl = np.linalg.norm((R_est @ p_src[inl].T).T + t_est - p_dst[inl], axis=1)
    print(f"  RANSAC: inliers={n_inl}/{n_3d}  "
          f"err_inl mean={err_inl.mean():.2f} mm  max={err_inl.max():.2f} mm")

    pc_dst = _sample_pcd_mm(dst_zmap, N_SAMPLE_PTS)
    pc_src = _sample_pcd_mm(src_zmap, N_SAMPLE_PTS)
    pc_est = (R_est @ pc_src.T).T + t_est

    zf = np.array([1.0, 1.0, -1.0])

    _plot_registration(pc_dst * zf, pc_src * zf, pc_est * zf,
                       n_inl, n_3d, out_reg,
                       dst_label=dst_label, src_label=src_label)
    _plot_overlay(pc_dst * zf, pc_est * zf, n_inl, n_3d, out_overlay,
                  dst_label=dst_label, src_label=src_label)


def _plot_registration(pc_dst, pc_src, pc_est,
                       n_inliers, n_matches, output_path,
                       dst_label="master (synthetic)",
                       src_label="input (scan)"):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    s, alpha = 0.5, 0.6
    labels_row = ["Before registration", "After (estimated R,t)"]
    labels_col = ["Top-down (X, Y)", "Front (X, Z)", "Side (Y, Z)"]
    col_axes = [(0, 1), (0, 2), (1, 2)]
    C_M, C_I = "lightskyblue", "crimson"
    pairs = [(pc_dst, pc_src), (pc_dst, pc_est)]

    for row, (pa, pb) in enumerate(pairs):
        for col, (xi, yi) in enumerate(col_axes):
            ax = axes[row, col]
            ax.scatter(pa[:, xi], pa[:, yi], s=s, c=C_M, alpha=alpha,
                       label=dst_label)
            ax.scatter(pb[:, xi], pb[:, yi], s=s, c=C_I, alpha=alpha,
                       label=src_label)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(labels_col[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(labels_row[row], fontsize=11, fontweight="bold")
    for row in range(2):
        for col in range(3):
            axes[row, col].invert_yaxis()

    title = (f"SHOT352 dim352  |  "
             f"matches={n_matches}  inliers={n_inliers}")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def _plot_overlay(pc_dst, pc_est, n_inliers, n_matches, output_path,
                  dst_label="master (synthetic)",
                  src_label="aligned scan"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    s, alpha = 0.5, 0.6
    cols = [(0, 1), (0, 2), (1, 2)]
    names = ["Top-down (X, Y)", "Front (X, Z)", "Side (Y, Z)"]
    for col, (xi, yi) in enumerate(cols):
        ax = axes[col]
        ax.scatter(pc_dst[:, xi], pc_dst[:, yi], s=s, c="lightskyblue",
                   alpha=alpha, label=dst_label)
        ax.scatter(pc_est[:, xi], pc_est[:, yi], s=s, c="crimson",
                   alpha=alpha, label=src_label)
        ax.set_aspect("equal")
        ax.set_title(names[col], fontsize=11)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.legend(markerscale=5, fontsize=9)
    for ax in axes:
        ax.invert_yaxis()
    title = (f"Overlay  |  matches={n_matches}  inliers={n_inliers}")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iss_mode", type=str, default="mm",
                        choices=["mm", "uvd"],
                        help="mm: mm 공간 ISS (학습 동일), uvd: 픽셀 공간 ISS (등방성)")
    parser.add_argument("--master", type=str, default=None,
                        help="Master zmap 경로 (기본: blender_master1/master.png)")
    parser.add_argument("--no_rotate", action="store_true",
                        help="Master 180도 회전 비활성화")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="결과 저장 폴더 (기본: 자동 생성)")
    args = parser.parse_args()

    master_path = Path(args.master) if args.master else MASTER_PATH

    if args.output_dir:
        out_dir = ROOT / args.output_dir
    else:
        out_dir = (ROOT / "experiments/v1_20260415/results"
                   / f"scanned_vs_shot_dim352_iss_{args.iss_mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 이미지 로드
    print("=" * 60)
    print(f"[1] 이미지 로드  (ISS mode: {args.iss_mode})")
    scan_raw = cv2.imread(str(SCAN_PATH), cv2.IMREAD_UNCHANGED)
    master_raw = cv2.imread(str(master_path), cv2.IMREAD_UNCHANGED)
    if scan_raw is None:
        raise FileNotFoundError(SCAN_PATH)
    if master_raw is None:
        raise FileNotFoundError(master_path)

    if not args.no_rotate:
        master_raw = cv2.rotate(master_raw, cv2.ROTATE_180)
    print(f"  scan:   {scan_raw.shape}  depth: "
          f"{scan_raw[scan_raw > 0].min()}–{scan_raw[scan_raw > 0].max()}")
    print(f"  master: {master_raw.shape}  depth: "
          f"{master_raw[master_raw > 0].min()}–{master_raw[master_raw > 0].max()}"
          f"  ({'회전 없음' if args.no_rotate else '180도 회전 적용'})"
          f"  [{master_path.name}]")

    # 2. Preprocess scanned (floor removal + bilateral)
    print("\n[2] Preprocessing scanned (floor mask + bilateral)")
    scan_masked, info = mask_scanned_table(scan_raw)
    if info.get("applied"):
        print(f"  floor mask: peak={info['peak_center']}  "
              f"band=[{info['band_low']}, {info['band_high']}]  "
              f"masked={info['fraction_masked'] * 100:.1f}%")
    scan_bilat = apply_bilateral(scan_masked)
    nz = scan_bilat[scan_bilat > 0]
    print(f"  after bilateral: {nz.size} valid px  "
          f"depth: {nz.min()}–{nz.max()}")

    # 3. ISS + SHOT for both
    print(f"\n[3] ISS({args.iss_mode}) + SHOT352 (voxel=1, nr=20, sr=40)")
    print("  --- Master ---")
    m_kp_uv, m_scores, m_desc, m_kp_xyz, m_n_iss, m_nv = \
        prepare_iss_shot(master_raw, label="master", seed=0, iss_mode=args.iss_mode)
    print("  --- Scanned ---")
    s_kp_uv, s_scores, s_desc, s_kp_xyz, s_n_iss, s_nv = \
        prepare_iss_shot(scan_bilat, label="scan", seed=0, iss_mode=args.iss_mode)

    # 4. Load model
    print(f"\n[4] Loading model: {CHECKPOINT.name}")
    model = load_model(str(CHECKPOINT), DEVICE)

    # 5. 공통 패딩 크기
    scan_h, scan_w = scan_bilat.shape
    master_h, master_w = master_raw.shape
    pad_h = max(scan_h, master_h)
    pad_w = max(scan_w, master_w)
    print(f"\n[5] 공통 패딩: {pad_h}×{pad_w}")

    view_master = make_view(master_raw, m_kp_uv, m_scores, m_desc, pad_h, pad_w)
    view_scan = make_view(scan_bilat, s_kp_uv, s_scores, s_desc, pad_h, pad_w)
    img_master_disp = master_raw.astype(np.float32) / 65535.0
    img_scan_disp = scan_bilat.astype(np.float32) / 65535.0

    # ── A) Forward: view0=synthetic, view1=scanned ──
    print("\n" + "=" * 60)
    print("[A] Forward 추론: view0=Synthetic master, view1=Scanned")
    batch_fwd = collate_single_pair(view_master, view_scan)
    pred_fwd = run_inference(model, batch_fwd, DEVICE)

    m0_fwd = pred_fwd["matches0"][0].cpu().numpy()
    print(f"  매칭 수: {int((m0_fwd > -1).sum())}")

    visualize_matches(
        pred_fwd, img_master_disp, img_scan_disp,
        out_dir / "match_forward.png",
        title_extra="SHOT352 dim352",
        view0_label="View 0 (Synthetic master)",
        view1_label="View 1 (Scanned roi13)",
    )
    registration_and_viz(
        pred_fwd, master_raw, scan_bilat,
        m_kp_xyz, s_kp_xyz,
        out_dir / "reg_forward.png",
        out_dir / "overlay_forward.png",
        dst_label="master (synthetic)", src_label="input (scan)",
    )

    # ── B) Reverse: view0=scanned, view1=synthetic ──
    print("\n" + "=" * 60)
    print("[B] Reverse 추론: view0=Scanned, view1=Synthetic master")
    batch_rev = collate_single_pair(view_scan, view_master)
    pred_rev = run_inference(model, batch_rev, DEVICE)

    m0_rev = pred_rev["matches0"][0].cpu().numpy()
    print(f"  매칭 수: {int((m0_rev > -1).sum())}")

    visualize_matches(
        pred_rev, img_scan_disp, img_master_disp,
        out_dir / "match_reverse.png",
        title_extra="SHOT352 dim352",
        view0_label="View 0 (Scanned roi13)",
        view1_label="View 1 (Synthetic master)",
    )
    registration_and_viz(
        pred_rev, scan_bilat, master_raw,
        s_kp_xyz, m_kp_xyz,
        out_dir / "reg_reverse.png",
        out_dir / "overlay_reverse.png",
        dst_label="input (scan)", src_label="master (synthetic)",
    )

    print(f"\nDone! → {out_dir}/")


if __name__ == "__main__":
    main()
