"""Compare erode=5 vs erode=0 on depth→PCD→ISS pipeline.

Measures:
    1. PCD point count
    2. ISS keypoint count
    3. Per-stage timing
    4. Keypoint distance to nearest zero-depth (invalid) pixel
       → quantifies how many keypoints sit on fragile boundary edges.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/v1_20260415/share"))

from depth_to_iss import (  # noqa: E402
    LATERAL_MM, TRANSPORT_MM, depth_to_pcd, detect_iss_keypoints,
)

SCAN = ROOT / "gluefactory/datasets/scanned/scanned_data1.png"


def run(zmap: np.ndarray, erode: int):
    t0 = time.perf_counter()
    pts = depth_to_pcd(zmap, erode_boundary=erode)
    t1 = time.perf_counter()
    kps = detect_iss_keypoints(pts)
    t2 = time.perf_counter()
    return pts, kps, t1 - t0, t2 - t1


def kp_boundary_px(kps_mm: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Return per-keypoint distance (px) to nearest zero-depth pixel."""
    H, W = dt.shape
    us = np.clip((kps_mm[:, 0] / LATERAL_MM).astype(np.int32), 0, W - 1)
    vs = np.clip((kps_mm[:, 1] / TRANSPORT_MM).astype(np.int32), 0, H - 1)
    return dt[vs, us]


def main():
    zmap = np.array(Image.open(SCAN))
    H, W = zmap.shape
    zero = (zmap == 0).astype(np.uint8)
    valid = (1 - zero).astype(np.uint8)
    dt_px = cv2.distanceTransform(valid, cv2.DIST_L2, 5)

    print(f"input     : {SCAN.name}  shape=({H}, {W})")
    print(f"valid px  : {int(valid.sum()):,d}  ({valid.mean() * 100:.1f}%)")
    print()

    # Run both configs
    pts5, kp5, tp5, ti5 = run(zmap, erode=5)
    pts0, kp0, tp0, ti0 = run(zmap, erode=0)

    # Per-kp boundary distance
    d5 = kp_boundary_px(kp5, dt_px)
    d0 = kp_boundary_px(kp0, dt_px)

    # Summary table
    print(f"{'metric':<30}  {'erode=5':>12}  {'erode=0':>12}  {'Δ':>10}")
    print("─" * 72)
    rows = [
        ("pcd_points",       len(pts5),  len(pts0)),
        ("iss_keypoints",    len(kp5),   len(kp0)),
        ("t_depth_to_pcd_s", tp5,        tp0),
        ("t_iss_detect_s",   ti5,        ti0),
        ("t_total_s",        tp5 + ti5,  tp0 + ti0),
    ]
    for name, a, b in rows:
        if isinstance(a, int):
            diff = b - a
            print(f"{name:<30}  {a:>12,d}  {b:>12,d}  {diff:>+10,d}")
        else:
            diff = b - a
            print(f"{name:<30}  {a:>12.3f}  {b:>12.3f}  {diff:>+10.3f}")

    print()
    print("Keypoint boundary proximity (distance to nearest zero-depth px)")
    print(f"{'threshold':<15}  {'erode=5':>12}  {'erode=0':>12}")
    print("─" * 45)
    for thr in [1, 3, 5, 10, 20]:
        n5 = int(np.sum(d5 <= thr))
        n0 = int(np.sum(d0 <= thr))
        p5 = n5 / len(kp5) * 100 if len(kp5) else 0.0
        p0 = n0 / len(kp0) * 100 if len(kp0) else 0.0
        print(f"≤{thr:>3}px       {n5:>5d} ({p5:>4.1f}%)  "
              f"{n0:>5d} ({p0:>4.1f}%)")

    print()
    print("Distance statistics (px)")
    for label, arr in [("erode=5", d5), ("erode=0", d0)]:
        if len(arr) == 0:
            continue
        print(f"  {label}: min={arr.min():.1f}  p5={np.percentile(arr, 5):.1f}  "
              f"median={np.median(arr):.1f}  mean={arr.mean():.1f}  "
              f"max={arr.max():.1f}")


if __name__ == "__main__":
    main()
