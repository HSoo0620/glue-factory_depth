"""
Dependencies
------------
numpy, open3d, opencv-python, pillow  (pillow only for the __main__ demo)

Example
-------
>>> import numpy as np
>>> from PIL import Image
>>> from depth_to_iss import depth_to_pcd, detect_iss_keypoints
>>> zmap = np.array(Image.open("scan.png"))          # (H, W) uint16
>>> pts_mm = depth_to_pcd(zmap)                       # (N, 3) float32 mm
>>> kps_mm = detect_iss_keypoints(pts_mm)             # (K, 3) float32 mm

"""

from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d

LATERAL_MM: float = 0.056       # X = u * LATERAL_MM
TRANSPORT_MM: float = 0.056     # Y = v * TRANSPORT_MM
VERTICAL_MM: float = 0.0085     # Z = raw_uint16 * VERTICAL_MM

ERODE_BOUNDARY_PX: int = 5

ISS_SALIENT_RADIUS_MULT: float = 6.0   # salient_r = K * avg_1nn_dist
ISS_NON_MAX_RADIUS_MULT: float = 2.0   # non_max_r = K * salient_r
ISS_GAMMA_21: float = 0.5              # λ₂ / λ₁ threshold
ISS_GAMMA_32: float = 0.5              # λ₃ / λ₂ threshold
ISS_MIN_NEIGHBORS: int = 5


def depth_to_pcd(
    zmap: np.ndarray,
    erode_boundary: int = ERODE_BOUNDARY_PX,
) -> np.ndarray:
    """Convert a single-channel depth map into a mm point cloud.

    Parameters
    ----------
    zmap : np.ndarray
        (H, W) uint16 depth map from the 3D line scanner.
        A value of 0 is treated as "no return" and discarded.
    erode_boundary : int, default=ERODE_BOUNDARY_PX
        Shrink the valid-depth mask by this many pixels before lifting,
        removing noisy fringe points on the silhouette boundary.

    Returns
    -------
    pts_mm : np.ndarray
        (N, 3) float32 array of (X, Y, Z) in millimetres.
        N equals the number of valid (non-zero, non-eroded) depth pixels.
    """
    mask = (zmap > 0).astype(np.uint8)
    if erode_boundary > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=erode_boundary)

    vs, us = np.where(mask > 0)
    zs = zmap[vs, us].astype(np.float32)

    return np.column_stack([
        us.astype(np.float32) * LATERAL_MM,
        vs.astype(np.float32) * TRANSPORT_MM,
        zs * VERTICAL_MM,
    ]).astype(np.float32)


def detect_iss_keypoints(pts_mm: np.ndarray) -> np.ndarray:
    """Detect ISS keypoints on an (N, 3) mm point cloud.

    Uses Open3D's `compute_iss_keypoints` with salient and non-maximum
    suppression radii scaled from the cloud's mean 1-NN distance.
    Matches the v1_20260415 training cache build exactly.

    Parameters
    ----------
    pts_mm : np.ndarray
        (N, 3) float point cloud in millimetres.

    Returns
    -------
    kp_xyz : np.ndarray
        (K, 3) float32 ISS keypoint coordinates in mm.
        Returns an empty (0, 3) array when no keypoints can be found.
    """
    if pts_mm.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_mm.astype(np.float64))

    distances = pcd.compute_nearest_neighbor_distance()
    if len(distances) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    avg_nn = float(np.mean(distances))
    salient_r = ISS_SALIENT_RADIUS_MULT * avg_nn
    non_max_r = ISS_NON_MAX_RADIUS_MULT * salient_r

    kp_pcd = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd,
        salient_radius=salient_r,
        non_max_radius=non_max_r,
        gamma_21=ISS_GAMMA_21,
        gamma_32=ISS_GAMMA_32,
        min_neighbors=ISS_MIN_NEIGHBORS,
    )
    return np.asarray(kp_pcd.points, dtype=np.float32)


def depth_to_iss_keypoints(
    zmap: np.ndarray,
    erode_boundary: int = ERODE_BOUNDARY_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: depth_to_pcd → detect_iss_keypoints.

    Returns
    -------
    pts_mm : (N, 3) float32 mm point cloud
    kp_xyz : (K, 3) float32 mm ISS keypoints
    """
    pts_mm = depth_to_pcd(zmap, erode_boundary=erode_boundary)
    kp_xyz = detect_iss_keypoints(pts_mm)
    return pts_mm, kp_xyz


if __name__ == "__main__":
    import argparse
    import sys
    import time
    from pathlib import Path

    from PIL import Image

    ap = argparse.ArgumentParser(
        description="Depth map → PCD → ISS keypoints demo.")
    ap.add_argument("zmap_png", type=str, help="Path to 16-bit depth PNG.")
    ap.add_argument("--no-erode", action="store_true",
                    help="Skip boundary erosion (debug).")
    args = ap.parse_args()

    zmap = np.array(Image.open(args.zmap_png))
    if zmap.ndim != 2:
        sys.exit(f"[err] expected (H, W) depth map, got shape {zmap.shape}")

    erode = 0 if args.no_erode else ERODE_BOUNDARY_PX

    t0 = time.perf_counter()
    pts = depth_to_pcd(zmap, erode_boundary=erode)
    t1 = time.perf_counter()
    kps = detect_iss_keypoints(pts)
    t2 = time.perf_counter()

    print(f"input      : {Path(args.zmap_png).name}  shape={zmap.shape}")
    print(f"erode_px   : {erode}")
    print(f"pcd points : {len(pts):>8,d}    ({t1 - t0:.3f}s)")
    print(f"keypoints  : {len(kps):>8,d}    ({t2 - t1:.3f}s)")
