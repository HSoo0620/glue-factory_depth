# Scanned SHOT Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `test_scanned_vs_master_shot.py` — scanned depth vs master matching inference using the trained ISS+SHOT352+LightGlue model.

**Architecture:** Duplicate `test_scanned_vs_master.py` (FPFH-based), replacing FPFH descriptor extraction with precomputed SHOT352 `.bin` loading + KDTree lookup. ISS keypoint detection, preprocessing, registration, and visualization are reused unchanged.

**Tech Stack:** PyTorch, Open3D (ISS), scipy (KDTree), matplotlib, OmegaConf

**Spec:** `docs/superpowers/specs/2026-04-13-scanned-shot-inference-design.md`

---

### Task 1: Create `test_scanned_vs_master_shot.py`

**Files:**
- Create: `test_scanned_vs_master_shot.py`

This is the main and only task. The script is a targeted modification of `test_scanned_vs_master.py` with these changes:

1. Imports: `precompute_iss_shot352_resample2` instead of `precompute_iss_fpfh_resample2`
2. Scanned features: `load_shot352_bin()` + `lookup_shot352()` instead of runtime FPFH
3. Master features: `shot_descriptors` key from `iss_shot352_resample2_cache/`
4. Collate: `resample2_iss_shot_collate_fn`
5. Model: `test_resample2_iss_shot_0408.load_model`
6. Default `flip_scanned_depth=False`

- [ ] **Step 1: Create the script**

```python
"""
Inference script for matching real 3D line-scanner data against master depth
using the trained ISS+SHOT352+LightGlue model.

Usage:
    python test_scanned_vs_master_shot.py --run_both
    python test_scanned_vs_master_shot.py --master_indices 0 594 --scale_dz
    python test_scanned_vs_master_shot.py --help

Spec: docs/superpowers/specs/2026-04-13-scanned-shot-inference-design.md
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from precompute_iss_shot352_resample2 import (
    depth_crop_to_pcd,
    extract_iss_keypoints,
    select_keypoints,
    kp_crop_to_xyz,
    load_shot352_bin,
    lookup_shot352,
    CROP_X0,
    CROP_Y0,
    CROP_SIZE,
    CLIP_START,
    CLIP_END,
    SHOT_DIM,
)
from visualize_registration_flann import (
    pixel_to_grid3d,
    rigid_transform_svd,
    ransac_rigid,
    sample_point_cloud,
    GRID_DX,
    GRID_DY,
)
from test_resample2_iss_shot_0408 import load_model, run_inference
from gluefactory.datasets.mitsubishi_resample2_iss_shot_dataset import (
    resample2_iss_shot_collate_fn,
)
from gluefactory.utils.tensor import batch_to_device


# Scanned data spacing (mm)
SCANNED_DX = 0.056
SCANNED_DY = 0.056
SCANNED_DZ = 0.0085
# Master data spacing (mm)
MASTER_DZ = 0.02
# Default dz correction ratio (scanned_dz / master_dz)
DEFAULT_DZ_RATIO = SCANNED_DZ / MASTER_DZ  # ~ 0.425


def parse_args():
    p = argparse.ArgumentParser(
        description="Scanned depth inference against master viewpoints (SHOT352)."
    )
    p.add_argument("--scanned_path", type=str,
                   default="gluefactory/datasets/scanned/scanned_data1.png")
    p.add_argument("--scanned_bin_path", type=str,
                   default="gluefactory/datasets/Descriptor/output_shot_scanned/scanned_data1_shot352.bin")
    p.add_argument("--master_indices", type=int, nargs="+", default=[0, 73, 88, 102, 594])
    p.add_argument("--checkpoint", type=str,
                   default="outputs/training/0407_resample2_iss_shot352_lg/checkpoint_best.tar")
    p.add_argument("--cache_dir", type=str,
                   default="gluefactory/datasets/mitsubishi/iss_shot352_resample2_cache")
    p.add_argument("--master_bin_dir", type=str,
                   default="gluefactory/datasets/Descriptor/output_shot_0407")
    p.add_argument("--master_img_dir", type=str,
                   default="gluefactory/datasets/mitsubishi/dataset_resample_2")
    p.add_argument("--output_dir", type=str,
                   default="results/scanned/scanned_data1_shot")
    p.add_argument("--image_size", type=int, default=1751)
    p.add_argument("--max_num_keypoints", type=int, default=512)
    p.add_argument("--dz_ratio", type=float, default=DEFAULT_DZ_RATIO)
    p.add_argument("--run_both", action="store_true",
                   help="Run both raw and dzfix experiments in one invocation.")
    p.add_argument("--scale_dz", action="store_true",
                   help="Run only the dzfix experiment (ignored if --run_both is set).")
    p.add_argument("--flip_scanned_depth", action="store_true",
                   help="Invert scanned raw depth (raw' = 65535 - raw). "
                        "WARNING: precomputed .bin was built on non-flipped data, "
                        "so KDTree lookup and descriptor values will be mismatched.")
    p.add_argument("--mask_scanned_table", action="store_true",
                   help="Remove the flat bed/table surface from scanned via "
                        "histogram-peak detection on the raw depth values.")
    p.add_argument("--table_band_fraction", type=float, default=0.05)
    p.add_argument("--erode_boundary", type=int, default=5)
    p.add_argument("--rotate_master", type=int, default=0, choices=[0, 90, 180, 270],
                   help="Rotate master depth crop by N degrees CCW.")
    p.add_argument("--ransac_iter", type=int, default=1000)
    p.add_argument("--inlier_th", type=float, default=5.0)
    p.add_argument("--overlap_threshold", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# ── Preprocessing (unchanged from FPFH version) ─────────────────────────────

def mask_scanned_table(canvas, band_fraction=0.05, bin_width=50):
    """Zero out the flat table/bed surface in a scanned depth canvas."""
    mask = canvas > 0
    if mask.sum() == 0:
        return canvas.copy(), {"applied": False, "reason": "empty"}

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
    table_sel = (canvas >= band_low) & (canvas < band_high)
    out[table_sel] = 0

    info = {
        "applied": True,
        "peak_center": int((edges[peak_bin] + edges[peak_bin + 1]) / 2),
        "band_low": band_low,
        "band_high": band_high,
        "n_masked": int(table_sel.sum()),
        "n_kept": int(mask.sum() - table_sel.sum()),
        "fraction_masked": float(table_sel.sum() / mask.sum()),
    }
    return out, info


def preprocess_scanned(
    scanned_png_path,
    scale_dz=False,
    dz_ratio=DEFAULT_DZ_RATIO,
    target_crop_size=3502,
    target_image_size=1751,
    flip_depth=False,
    mask_table=False,
    table_band_fraction=0.05,
):
    """Load scanned PNG and produce crop + small image tensors."""
    img = cv2.imread(str(scanned_png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read scanned image: {scanned_png_path}")
    if img.dtype != np.uint16:
        print(f"WARNING: scanned dtype is {img.dtype}, forcing uint16")
        img = img.astype(np.uint16)

    orig_h, orig_w = img.shape[:2]
    scale_factor = target_crop_size / max(orig_h, orig_w)
    new_h = int(round(orig_h * scale_factor))
    new_w = int(round(orig_w * scale_factor))
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((target_crop_size, target_crop_size), dtype=np.uint16)
    pad_top = (target_crop_size - new_h) // 2
    pad_left = (target_crop_size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = scaled

    if flip_depth:
        flip_mask = canvas > 0
        canvas[flip_mask] = (65535 - canvas[flip_mask].astype(np.int32)).astype(np.uint16)

    if scale_dz:
        mask = canvas > 0
        scaled_vals = (canvas.astype(np.float64) * dz_ratio).clip(0, 65535)
        canvas = np.where(mask, scaled_vals.astype(np.uint16), np.uint16(0))

    table_info = {"applied": False}
    if mask_table:
        canvas, table_info = mask_scanned_table(
            canvas, band_fraction=table_band_fraction
        )

    n_valid = int((canvas > 0).sum())
    if n_valid < 1000:
        print(f"WARNING: only {n_valid} valid pixels in scanned crop")

    small = cv2.resize(
        canvas, (target_image_size, target_image_size),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.float32) / 65535.0

    meta = {
        "orig_shape": (orig_h, orig_w),
        "scaled_shape": (new_h, new_w),
        "pad_top": pad_top,
        "pad_left": pad_left,
        "scale_factor": scale_factor,
        "dz_scaled": scale_dz,
        "depth_flipped": flip_depth,
        "table_mask_info": table_info,
    }
    return canvas, small, meta


# ── SHOT-specific feature extraction ────────────────────────────────────────

def extract_scanned_features_shot(
    scanned_crop_raw,
    scanned_bin_path,
    fx=8001.39,
    fy=8001.39,
    cx=1751.0,
    cy=1751.0,
    max_num_keypoints=512,
    image_size=1751,
    erode_boundary=5,
):
    """Runtime ISS keypoint detection + precomputed SHOT352 lookup.

    1. ISS detection on (u,v,depth_scaled) — same as FPFH version
    2. kp_crop → XYZ conversion
    3. Load scanned .bin → KDTree lookup → SHOT descriptors
    """
    pcd_iss, _, depth_scale, d_min, erode_mask = depth_crop_to_pcd(
        scanned_crop_raw, erode_boundary=erode_boundary
    )

    iss_kp_3d = extract_iss_keypoints(pcd_iss)
    n_iss = len(iss_kp_3d)
    print(f"  ISS keypoints: {n_iss}")

    keypoints, scores, n_valid, kp_crop, _ = select_keypoints(
        iss_kp_3d, scanned_crop_raw, max_num_keypoints, image_size,
        depth_scale=depth_scale, depth_min=d_min, erode_mask=erode_mask,
    )

    kp_xyz = kp_crop_to_xyz(
        kp_crop, n_valid, scanned_crop_raw, fx, fy, cx, cy
    )

    pts_cloud, desc_cloud = load_shot352_bin(scanned_bin_path)
    shot = lookup_shot352(pts_cloud, desc_cloud, kp_xyz, n_valid)

    return {
        "keypoints": keypoints,
        "keypoint_scores": scores,
        "shot_descriptors": shot,
        "n_valid": int(n_valid),
        "n_iss": int(n_iss),
        "kp_crop": kp_crop,
    }


def load_master_features(master_idx, cache_dir, master_img_dir, image_size=1751):
    """Load cached master SHOT features and raw depth crop."""
    cache_dir = Path(cache_dir)
    master_img_dir = Path(master_img_dir)
    npz_path = cache_dir / f"depth_raw_{master_idx:04d}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Master cache not found: {npz_path}\n"
            f"Run: python precompute_iss_shot352_resample2.py"
        )
    data = np.load(npz_path)
    n_valid = int(data["n_valid"])
    keypoints = data["keypoints"]

    scale = CROP_SIZE / image_size
    kp_crop = np.zeros_like(keypoints)
    kp_crop[:n_valid] = keypoints[:n_valid] * scale

    features = {
        "keypoints": keypoints,
        "keypoint_scores": data["keypoint_scores"],
        "shot_descriptors": data["shot_descriptors"],
        "n_valid": n_valid,
        "n_iss": int(data["n_iss"]),
        "kp_crop": kp_crop,
    }

    img_path = master_img_dir / f"depth_raw_{master_idx:04d}.png"
    img_raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise FileNotFoundError(f"Master image not found: {img_path}")
    master_crop_raw = img_raw[CROP_Y0:CROP_Y0 + CROP_SIZE,
                               CROP_X0:CROP_X0 + CROP_SIZE]
    return features, master_crop_raw


def extract_master_features_shot(
    master_crop_raw,
    master_bin_path,
    fx=8001.39,
    fy=8001.39,
    cx=1751.5,
    cy=1799.5,
    max_num_keypoints=512,
    image_size=1751,
    erode_boundary=5,
):
    """Runtime ISS + SHOT lookup for rotated master (same flow as scanned)."""
    pcd_iss, _, depth_scale, d_min, erode_mask = depth_crop_to_pcd(
        master_crop_raw, erode_boundary=erode_boundary
    )
    iss_kp_3d = extract_iss_keypoints(pcd_iss)
    keypoints, scores, n_valid, kp_crop, _ = select_keypoints(
        iss_kp_3d, master_crop_raw, max_num_keypoints, image_size,
        depth_scale=depth_scale, depth_min=d_min, erode_mask=erode_mask,
    )
    kp_xyz = kp_crop_to_xyz(
        kp_crop, n_valid, master_crop_raw, fx, fy, cx, cy
    )
    pts_cloud, desc_cloud = load_shot352_bin(master_bin_path)
    shot = lookup_shot352(pts_cloud, desc_cloud, kp_xyz, n_valid)
    return {
        "keypoints": keypoints,
        "keypoint_scores": scores,
        "shot_descriptors": shot,
        "n_valid": int(n_valid),
        "n_iss": int(len(iss_kp_3d)),
        "kp_crop": kp_crop,
    }


def build_batch(
    master_feats, master_crop_raw,
    scanned_feats, scanned_crop_raw,
    image_size=1751,
    master_path="",
    scanned_path="",
):
    """Construct batch dict for the SHOT model."""
    master_small = cv2.resize(
        master_crop_raw, (image_size, image_size),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.float32) / 65535.0
    scanned_small = cv2.resize(
        scanned_crop_raw, (image_size, image_size),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.float32) / 65535.0

    def _view_dict(feats, small):
        return {
            "image": torch.from_numpy(small).unsqueeze(0),
            "image_size": torch.tensor([image_size, image_size]),
            "keypoints": torch.from_numpy(feats["keypoints"]).float(),
            "keypoint_scores": torch.from_numpy(feats["keypoint_scores"]).float(),
            "descriptors": torch.from_numpy(feats["shot_descriptors"]).float(),
        }

    sample = {
        "view0": _view_dict(master_feats, master_small),
        "view1": _view_dict(scanned_feats, scanned_small),
        "gt_matches": torch.zeros((0, 4), dtype=torch.float32),
        "csv_path": "",
        "master_path": master_path,
        "input_path": scanned_path,
    }

    batch = resample2_iss_shot_collate_fn([sample])
    return batch, master_small, scanned_small


# ── Registration & Visualization (unchanged from FPFH version) ──────────────

def compute_overlap_rate(pc_master, pc_aligned, threshold=1.0):
    if len(pc_master) == 0 or len(pc_aligned) == 0:
        return 0.0
    tree = cKDTree(pc_master)
    dists, _ = tree.query(pc_aligned, k=1)
    return float((dists < threshold).mean())


def estimate_registration(
    pred, master_crop_raw, scanned_crop_raw,
    image_size=1751,
    ransac_iter=1000,
    inlier_th=5.0,
    n_sample_pts=15000,
):
    kp0 = pred["keypoints0"][0].cpu().numpy()
    kp1 = pred["keypoints1"][0].cpu().numpy()
    m0 = pred["matches0"][0].cpu().numpy()

    valid = m0 > -1
    mkp0 = kp0[valid]
    mkp1 = kp1[m0[valid]]
    n_matches = int(valid.sum())

    scale = CROP_SIZE / image_size
    mkp0_crop = mkp0 * scale
    mkp1_crop = mkp1 * scale

    pts_master, vm = pixel_to_grid3d(mkp0_crop, master_crop_raw)
    pts_scanned, vi = pixel_to_grid3d(mkp1_crop, scanned_crop_raw)
    both_valid = vm & vi
    pts_master = pts_master[both_valid]
    pts_scanned = pts_scanned[both_valid]
    n_valid_matches = len(pts_master)

    if n_valid_matches < 3:
        print(f"  WARNING: only {n_valid_matches} valid matches, skipping RANSAC")
        R, t = np.eye(3), np.zeros(3)
        inliers = np.zeros(n_valid_matches, dtype=bool)
    else:
        R, t, inliers = ransac_rigid(
            pts_scanned, pts_master, n_iter=ransac_iter, inlier_th=inlier_th
        )

    pc_master = sample_point_cloud(master_crop_raw, n_pts=n_sample_pts)
    pc_scanned = sample_point_cloud(scanned_crop_raw, n_pts=n_sample_pts)

    if len(pc_scanned) > 0:
        pc_scanned_aligned = (R @ pc_scanned.T).T + t
    else:
        pc_scanned_aligned = pc_scanned

    return {
        "R": R,
        "t": t,
        "n_matches": n_matches,
        "n_valid_matches": n_valid_matches,
        "n_inliers": int(inliers.sum()),
        "pc_master": pc_master,
        "pc_scanned": pc_scanned,
        "pc_scanned_aligned": pc_scanned_aligned,
    }


def plot_matches_no_gt(pred, master_small, scanned_small, output_path, title):
    kp0 = pred["keypoints0"][0].cpu().numpy()
    kp1 = pred["keypoints1"][0].cpu().numpy()
    m0 = pred["matches0"][0].cpu().numpy()

    valid = m0 > -1
    mkp0 = kp0[valid]
    mkp1 = kp1[m0[valid]]
    n_total = int(valid.sum())

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
    for ax, img, kp, label in [
        (axes[0], master_small, kp0, "View 0 (Master)"),
        (axes[1], scanned_small, kp1, "View 1 (Scanned)"),
    ]:
        ax.imshow(img, cmap="gray")
        ax.scatter(kp[:, 0], kp[:, 1], c="royalblue", s=3, alpha=0.3, linewidths=0)
        ax.set_title(label, fontsize=14)
        ax.set_axis_off()

    for i in range(n_total):
        line = matplotlib.patches.ConnectionPatch(
            xyA=(float(mkp0[i, 0]), float(mkp0[i, 1])),
            xyB=(float(mkp1[i, 0]), float(mkp1[i, 1])),
            coordsA=axes[0].transData,
            coordsB=axes[1].transData,
            axesA=axes[0],
            axesB=axes[1],
            color="skyblue",
            linewidth=0.8,
            alpha=0.6,
        )
        fig.add_artist(line)

    info = f"{title}  |  matches={n_total}"
    fig.suptitle(info, fontsize=11, y=0.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_alignment_no_gt(
    pc_master, pc_scanned, pc_scanned_aligned,
    output_path, title,
    n_matches=0, n_inliers=0, overlap_rate=0.0,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    s = 0.5
    alpha = 0.6

    labels_row = ["Before", "Estimated R,t"]
    labels_col = ["Top-down (u, v)", "Front (u, z)", "Side (v, z)"]
    col_axes = [(0, 1), (0, 2), (1, 2)]

    C_MASTER = "lightskyblue"
    C_INPUT = "crimson"
    pairs = [
        (pc_master, pc_scanned),
        (pc_master, pc_scanned_aligned),
    ]

    for row, (pc_a, pc_b) in enumerate(pairs):
        for col, (xi, yi) in enumerate(col_axes):
            ax = axes[row, col]
            if len(pc_a) > 0:
                ax.scatter(pc_a[:, xi], pc_a[:, yi], s=s, c=C_MASTER, alpha=alpha, label="master")
            if len(pc_b) > 0:
                ax.scatter(pc_b[:, xi], pc_b[:, yi], s=s, c=C_INPUT, alpha=alpha, label="scanned")
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(labels_col[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(labels_row[row], fontsize=11, fontweight="bold")

    for row in range(2):
        for col in range(3):
            axes[row, col].invert_yaxis()

    info = (f"{title}  |  matches={n_matches}, inliers={n_inliers}, "
            f"overlap@1={overlap_rate:.3f}")
    fig.suptitle(info, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Experiment runner ────────────────────────────────────────────────────────

def run_single_experiment(
    args, model, conf, device, scale_dz, experiment_label,
):
    exp_dir = Path(args.output_dir) / experiment_label
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Experiment: {experiment_label} (scale_dz={scale_dz}, "
          f"flip_depth={args.flip_scanned_depth}, "
          f"mask_table={args.mask_scanned_table}, "
          f"erode_boundary={args.erode_boundary}, "
          f"rotate_master={args.rotate_master}) ===")
    scanned_crop_raw, _, meta = preprocess_scanned(
        Path(args.scanned_path),
        scale_dz=scale_dz,
        dz_ratio=args.dz_ratio,
        flip_depth=args.flip_scanned_depth,
        mask_table=args.mask_scanned_table,
        table_band_fraction=args.table_band_fraction,
    )
    print(f"  Scanned preprocessed: {meta}")

    scanned_feats = extract_scanned_features_shot(
        scanned_crop_raw,
        args.scanned_bin_path,
        max_num_keypoints=args.max_num_keypoints,
        image_size=args.image_size,
        erode_boundary=args.erode_boundary,
    )
    print(f"  Scanned features: n_valid={scanned_feats['n_valid']}, "
          f"n_iss={scanned_feats['n_iss']}")

    rows = []
    for master_idx in args.master_indices:
        print(f"\n  --- Master {master_idx:04d} ---")
        master_feats, master_crop_raw = load_master_features(
            master_idx, args.cache_dir, args.master_img_dir,
            image_size=args.image_size,
        )

        if args.rotate_master != 0:
            k = args.rotate_master // 90
            master_crop_raw = np.rot90(master_crop_raw, k=k)
            print(f"  Rotated master by {args.rotate_master} deg CCW -> re-extracting features")
            master_bin_path = str(
                Path(args.master_bin_dir) / f"depth_raw_{master_idx:04d}_shot352.bin"
            )
            master_feats = extract_master_features_shot(
                master_crop_raw,
                master_bin_path,
                max_num_keypoints=args.max_num_keypoints,
                image_size=args.image_size,
                erode_boundary=5,
            )
            print(f"  Rotated master features: n_valid={master_feats['n_valid']}, "
                  f"n_iss={master_feats['n_iss']}")

        batch, master_small, scanned_small = build_batch(
            master_feats, master_crop_raw,
            scanned_feats, scanned_crop_raw,
            image_size=args.image_size,
            master_path=str(Path(args.master_img_dir) / f"depth_raw_{master_idx:04d}.png"),
            scanned_path=str(args.scanned_path),
        )

        pred, batch = run_inference(model, batch, device)

        match_path = exp_dir / f"master{master_idx:04d}_match.png"
        plot_matches_no_gt(
            pred, master_small, scanned_small, match_path,
            title=f"{experiment_label} master={master_idx:04d}",
        )

        reg = estimate_registration(
            pred, master_crop_raw, scanned_crop_raw,
            image_size=args.image_size,
            ransac_iter=args.ransac_iter,
            inlier_th=args.inlier_th,
        )
        overlap = compute_overlap_rate(
            reg["pc_master"], reg["pc_scanned_aligned"],
            threshold=args.overlap_threshold,
        )

        reg_path = exp_dir / f"master{master_idx:04d}_reg.png"
        plot_alignment_no_gt(
            reg["pc_master"], reg["pc_scanned"], reg["pc_scanned_aligned"],
            reg_path,
            title=f"{experiment_label} master={master_idx:04d}",
            n_matches=reg["n_matches"],
            n_inliers=reg["n_inliers"],
            overlap_rate=overlap,
        )

        rows.append({
            "master_idx": master_idx,
            "experiment": experiment_label,
            "n_scanned_kp": scanned_feats["n_valid"],
            "n_master_kp": master_feats["n_valid"],
            "n_matches": reg["n_matches"],
            "n_inliers": reg["n_inliers"],
            "overlap_rate": round(overlap, 4),
        })

    return rows


def main():
    args = parse_args()
    print(f"Args: {vars(args)}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model, conf = load_model(args.checkpoint, device)

    if args.run_both:
        experiments = [(False, "raw"), (True, "dzfix")]
    elif args.scale_dz:
        experiments = [(True, "dzfix")]
    else:
        experiments = [(False, "raw")]

    all_rows = []
    for scale_dz, label in experiments:
        rows = run_single_experiment(args, model, conf, device, scale_dz, label)
        all_rows.extend(rows)

    csv_path = output_root / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["master_idx", "experiment", "n_scanned_kp",
                        "n_master_kp", "n_matches", "n_inliers", "overlap_rate"],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\nDone! Summary: {csv_path}")
    print(f"Total runs: {len(all_rows)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test (single master index)**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda activate LightGlue
python test_scanned_vs_master_shot.py --master_indices 0 --output_dir results/scanned/scanned_data1_shot_smoke
```

Expected: No errors, produces `results/scanned/scanned_data1_shot_smoke/raw/master0000_match.png` and `master0000_reg.png`.

- [ ] **Step 3: Full run (all indices, run_both)**

Run:
```bash
python test_scanned_vs_master_shot.py --run_both --output_dir results/scanned/scanned_data1_shot
```

Expected: `results/scanned/scanned_data1_shot/summary.csv` with 10 rows (5 master × 2 experiments).

- [ ] **Step 4: Commit**

```bash
git add test_scanned_vs_master_shot.py docs/superpowers/specs/2026-04-13-scanned-shot-inference-design.md docs/superpowers/plans/2026-04-13-scanned-shot-inference.md
git commit -m "feat: add scanned vs master inference script for SHOT352 descriptors"
```
