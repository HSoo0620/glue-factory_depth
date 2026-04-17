"""
Inference script for matching real 3D line-scanner data against master depth
using the trained ISS+FPFH+LightGlue model.

Usage:
    python test_scanned_vs_master.py --run_both
    python test_scanned_vs_master.py --master_indices 0 594 --scale_dz
    python test_scanned_vs_master.py --help

Spec: docs/superpowers/specs/2026-04-10-scanned-depth-matching-inference-design.md
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

from precompute_iss_fpfh_resample2 import (
    depth_crop_to_pcd,
    build_xyz_pcd,
    extract_iss_keypoints,
    select_keypoints,
    kp_crop_to_xyz,
    compute_fpfh_for_keypoints,
    CROP_X0,
    CROP_Y0,
    CROP_SIZE,
    CLIP_START,
    CLIP_END,
)
from visualize_registration_flann import (
    pixel_to_grid3d,
    rigid_transform_svd,
    ransac_rigid,
    sample_point_cloud,
    GRID_DX,
    GRID_DY,
)
from test_resample2_iss_fpfh_0406 import load_model, run_inference
from gluefactory.datasets.mitsubishi_resample2_iss_fpfh_dataset import (
    resample2_iss_fpfh_collate_fn,
)
from gluefactory.utils.tensor import batch_to_device


# Scanned data spacing (mm)
SCANNED_DX = 0.056
SCANNED_DY = 0.056
SCANNED_DZ = 0.0085
# Master data spacing (mm)
MASTER_DZ = 0.02
# Default dz correction ratio (scanned_dz / master_dz)
DEFAULT_DZ_RATIO = SCANNED_DZ / MASTER_DZ  # ≈ 0.425


def parse_args():
    p = argparse.ArgumentParser(
        description="Scanned depth inference against master viewpoints."
    )
    p.add_argument("--scanned_path", type=str,
                   default="gluefactory/datasets/scanned/scanned_data1.png")
    p.add_argument("--master_indices", type=int, nargs="+", default=[0, 594])
    p.add_argument("--checkpoint", type=str,
                   default="outputs/training/0406_resample2_iss_fpfh_xyz_lg/checkpoint_best.tar")
    p.add_argument("--cache_dir", type=str,
                   default="gluefactory/datasets/mitsubishi/iss_fpfh_resample2_cache_r5.0_xyz")
    p.add_argument("--master_img_dir", type=str,
                   default="gluefactory/datasets/mitsubishi/dataset_resample_2")
    p.add_argument("--output_dir", type=str,
                   default="results/scanned/scanned_data1")
    p.add_argument("--fpfh_radius", type=float, default=5.0)
    p.add_argument("--image_size", type=int, default=1751)
    p.add_argument("--max_num_keypoints", type=int, default=512)
    p.add_argument("--dz_ratio", type=float, default=DEFAULT_DZ_RATIO)
    p.add_argument("--run_both", action="store_true",
                   help="Run both raw and dzfix experiments in one invocation.")
    p.add_argument("--scale_dz", action="store_true",
                   help="Run only the dzfix experiment (ignored if --run_both is set).")
    p.add_argument("--flip_scanned_depth", action="store_true",
                   help="Invert scanned raw depth (raw' = 65535 - raw) on non-zero "
                        "pixels to align with master's depth convention.")
    p.add_argument("--mask_scanned_table", action="store_true",
                   help="Remove the flat bed/table surface from scanned via "
                        "histogram-peak detection on the raw depth values.")
    p.add_argument("--table_band_fraction", type=float, default=0.05,
                   help="Fraction of histogram peak height used to delimit the "
                        "table band (smaller = narrower band).")
    p.add_argument("--erode_boundary", type=int, default=5,
                   help="Morphological erosion iterations on scanned valid mask "
                        "before ISS detection. Reduce to 0-1 when "
                        "--mask_scanned_table is active so pyramid side "
                        "surfaces are preserved.")
    p.add_argument("--rotate_master", type=int, default=0, choices=[0, 90, 180, 270],
                   help="Rotate master depth crop by N degrees CCW before "
                        "feature extraction. Useful when scanned has a known "
                        "in-plane rotation w.r.t. master orientation.")
    p.add_argument("--ransac_iter", type=int, default=1000)
    p.add_argument("--inlier_th", type=float, default=5.0)
    p.add_argument("--overlap_threshold", type=float, default=1.0,
                   help="Grid-space distance for proxy overlap metric.")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def mask_scanned_table(canvas, band_fraction=0.05, bin_width=50):
    """Zero out the flat table/bed surface in a scanned depth canvas.

    Line-scanner data contains a large flat bed behind the object that appears
    as a dominant narrow spike in the raw-value histogram. Masters, being
    Blender renders, have no such bed — so the bed is a pure distribution-gap
    distractor for FPFH neighborhoods. This helper detects the tallest bin and
    expands outward until density drops below ``band_fraction * peak_height``.

    Args:
        canvas: (H, W) uint16 depth image. Zeros stay zero.
        band_fraction: expansion threshold as a fraction of peak height.
        bin_width: histogram bin width in raw units.

    Returns:
        masked: uint16 copy with table pixels set to 0.
        info: dict with peak_center, band_low, band_high, n_masked, fraction_masked.
    """
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
    """Load scanned PNG and produce crop + small image tensors.

    Method D: aspect-preserving scale so max dimension equals target_crop_size,
    then center-pad into a square canvas. Uses INTER_NEAREST throughout to
    preserve depth edge values.

    Pipeline: load → resize → pad → [flip] → [scale_dz] → [mask_table] → small.

    Args:
        scanned_png_path: path to uint16 depth PNG.
        scale_dz: if True, multiply non-zero raw values by dz_ratio.
        dz_ratio: scanned_dz / master_dz, default 0.425.
        target_crop_size: canvas side length, default 3502.
        target_image_size: small image side length, default 1751.
        flip_depth: if True, invert non-zero raw values (raw' = 65535 - raw)
            to flip the scanner's depth convention to match the master's.
            Applied after resize/pad and before optional scale_dz. Background
            zeros are preserved.
        mask_table: if True, zero out the dominant histogram peak (flat bed
            surface) via :func:`mask_scanned_table`.
        table_band_fraction: fraction of peak height used as the band-expansion
            cutoff for table detection.

    Returns:
        scanned_crop_raw: (target_crop_size, target_crop_size) uint16 canvas
        scanned_img_small: (target_image_size, target_image_size) float32 normalized
        meta: dict with orig_shape, scaled_shape, pad_left, pad_top, scale_factor,
            dz_scaled, depth_flipped, table_mask_info
    """
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
        # Invert scanner raw convention on non-zero pixels; leave background zeros
        # as zeros so ISS erode_boundary and depth_crop_to_pcd still treat them
        # as invalid.
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

    # Sanity check: warn if too few valid pixels
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


def extract_scanned_features(
    scanned_crop_raw,
    fx=8001.39,
    fy=8001.39,
    cx=1751.0,
    cy=1751.0,
    fpfh_radius=5.0,
    max_num_keypoints=512,
    image_size=1751,
    erode_boundary=5,
):
    """Runtime ISS+FPFH extraction on scanned crop.

    Uses the same sequence as precompute_iss_fpfh_resample2.main() but
    operates on an in-memory scanned canvas instead of looping over disk files.

    Returns a dict with the same keys as the cache npz, plus kp_crop.
    """
    pcd_iss, _, depth_scale, d_min, erode_mask = depth_crop_to_pcd(
        scanned_crop_raw, erode_boundary=erode_boundary
    )

    iss_kp_3d = extract_iss_keypoints(pcd_iss)
    n_iss = len(iss_kp_3d)
    print(f"  ISS keypoints: {n_iss}")

    keypoints, scores, n_valid, kp_crop, kp_3d = select_keypoints(
        iss_kp_3d, scanned_crop_raw, max_num_keypoints, image_size,
        depth_scale=depth_scale, depth_min=d_min, erode_mask=erode_mask,
    )

    pcd_xyz = build_xyz_pcd(
        scanned_crop_raw, fx, fy, cx, cy, mask=erode_mask
    )
    kp_xyz = kp_crop_to_xyz(
        kp_crop, n_valid, scanned_crop_raw, fx, fy, cx, cy
    )
    fpfh = compute_fpfh_for_keypoints(
        pcd_xyz, kp_xyz, n_valid, fpfh_radius=fpfh_radius
    )

    return {
        "keypoints": keypoints,
        "keypoint_scores": scores,
        "fpfh_descriptors": fpfh,
        "n_valid": int(n_valid),
        "n_iss": int(n_iss),
        "kp_crop": kp_crop,
    }


def load_master_features(master_idx, cache_dir, master_img_dir, image_size=1751):
    """Load cached master features and raw depth crop.

    The existing cache does NOT store kp_crop, so it's reconstructed as
    keypoints * (CROP_SIZE / image_size) = keypoints * 2 when image_size=1751.

    Returns:
        features: dict with keypoints, keypoint_scores, fpfh_descriptors,
                  n_valid, n_iss, kp_crop
        master_crop_raw: (CROP_SIZE, CROP_SIZE) uint16 crop of the master PNG
    """
    cache_dir = Path(cache_dir)
    master_img_dir = Path(master_img_dir)
    npz_path = cache_dir / f"depth_raw_{master_idx:04d}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Master cache not found: {npz_path}\n"
            f"Run: python precompute_iss_fpfh_resample2.py --fpfh_radius 5.0"
        )
    data = np.load(npz_path)
    n_valid = int(data["n_valid"])
    keypoints = data["keypoints"]

    # Reconstruct kp_crop: keypoints is in image_size scale, kp_crop is in CROP_SIZE scale
    scale = CROP_SIZE / image_size
    kp_crop = np.zeros_like(keypoints)
    kp_crop[:n_valid] = keypoints[:n_valid] * scale

    features = {
        "keypoints": keypoints,
        "keypoint_scores": data["keypoint_scores"],
        "fpfh_descriptors": data["fpfh_descriptors"],
        "n_valid": n_valid,
        "n_iss": int(data["n_iss"]),
        "kp_crop": kp_crop,
    }

    # Load the master image and apply fixed crop
    img_path = master_img_dir / f"depth_raw_{master_idx:04d}.png"
    img_raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise FileNotFoundError(f"Master image not found: {img_path}")
    master_crop_raw = img_raw[CROP_Y0:CROP_Y0 + CROP_SIZE,
                               CROP_X0:CROP_X0 + CROP_SIZE]
    return features, master_crop_raw


def build_batch(
    master_feats, master_crop_raw,
    scanned_feats, scanned_crop_raw,
    image_size=1751,
    master_path="",
    scanned_path="",
):
    """Construct the batch dict expected by the trained model.

    Critical: the cache key 'fpfh_descriptors' must be renamed to 'descriptors'
    to match what the model forward pass reads (see
    mitsubishi_resample2_iss_fpfh_dataset.py:155-165).
    """
    # Downsample crops to image_size for the 'image' tensor
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
            "image": torch.from_numpy(small).unsqueeze(0),  # (1, H, W)
            "image_size": torch.tensor([image_size, image_size]),
            "keypoints": torch.from_numpy(feats["keypoints"]).float(),
            "keypoint_scores": torch.from_numpy(feats["keypoint_scores"]).float(),
            "descriptors": torch.from_numpy(feats["fpfh_descriptors"]).float(),
        }

    sample = {
        "view0": _view_dict(master_feats, master_small),
        "view1": _view_dict(scanned_feats, scanned_small),
        "gt_matches": torch.zeros((0, 4), dtype=torch.float32),
        "csv_path": "",
        "master_path": master_path,
        "input_path": scanned_path,
    }

    batch = resample2_iss_fpfh_collate_fn([sample])
    return batch, master_small, scanned_small


def compute_overlap_rate(pc_master, pc_aligned, threshold=1.0):
    """Fraction of aligned points whose nearest master point is within threshold.

    Uses scipy cKDTree for efficient nearest-neighbor search.
    Returns 0.0 if either cloud is empty.
    """
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
    """Extract matches, compute RANSAC rigid transform in Grid space, and
    return both the transformation and sampled point clouds for visualization.

    Grid space uses existing pixel_to_grid3d helper from visualize_registration_flann.
    """
    kp0 = pred["keypoints0"][0].cpu().numpy()  # master
    kp1 = pred["keypoints1"][0].cpu().numpy()  # scanned
    m0 = pred["matches0"][0].cpu().numpy()

    valid = m0 > -1
    mkp0 = kp0[valid]
    mkp1 = kp1[m0[valid]]
    n_matches = int(valid.sum())

    # Rescale keypoints from image_size back to CROP_SIZE for pixel_to_grid3d
    scale = CROP_SIZE / image_size
    mkp0_crop = mkp0 * scale
    mkp1_crop = mkp1 * scale

    # Convert matched keypoints to Grid 3D
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
        # RANSAC: align scanned → master
        R, t, inliers = ransac_rigid(
            pts_scanned, pts_master, n_iter=ransac_iter, inlier_th=inlier_th
        )

    # Sample dense clouds for visualization
    pc_master = sample_point_cloud(master_crop_raw, n_pts=n_sample_pts)
    pc_scanned = sample_point_cloud(scanned_crop_raw, n_pts=n_sample_pts)

    # Apply transform to scanned cloud
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


def plot_matches_no_gt(
    pred, master_small, scanned_small, output_path, title,
):
    """Simplified matching visualization — single color, no GT comparison."""
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
    """2-row (Before / Estimated) × 3-col (Top-down / Front / Side) visualization."""
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


def run_single_experiment(
    args, model, conf, device, scale_dz, experiment_label,
):
    """Run one full experiment (raw or dzfix) across all master indices."""
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

    scanned_feats = extract_scanned_features(
        scanned_crop_raw,
        fpfh_radius=args.fpfh_radius,
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
            print(f"  Rotated master by {args.rotate_master}° CCW → re-extracting features")
            master_feats = extract_scanned_features(
                master_crop_raw,
                fpfh_radius=args.fpfh_radius,
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

    # Determine which experiments to run
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

    # Write summary.csv at the root (not inside raw/ or dzfix/)
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
