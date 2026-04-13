"""ISS detection over (u, v, depth_scaled) PCD — same approach as
precompute_iss_fpfh_resample2.py:depth_crop_to_pcd, but:
  - no CROP (new dataset keeps full zmap)
  - no CLIP_START/CLIP_END (orthographic; raw uint16 used directly)
"""
from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d


def build_iss_pcd_uvd_scaled(zmap: np.ndarray, erode_boundary: int = 5):
    """Build a PCD in (u, v, depth_scaled) space for ISS detection.

    `depth_scaled = (raw - raw_min) * (u_range / raw_range)` so that XY
    and Z have similar scale (same idea as the existing Mitsubishi code).
    """
    mask = (zmap > 0).astype(np.uint8)
    if erode_boundary > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=erode_boundary)
    vs, us = np.where(mask > 0)
    if len(us) == 0:
        empty = o3d.geometry.PointCloud()
        return empty, mask, 1.0, 0.0

    us_f = us.astype(np.float64)
    vs_f = vs.astype(np.float64)
    raw = zmap[vs, us].astype(np.float64)

    u_range = max(us_f.max() - us_f.min(), 1.0)
    raw_min = float(raw.min())
    raw_range = float(raw.max() - raw.min()) if raw.max() > raw.min() else 1.0
    depth_scale = u_range / raw_range
    depth_scaled = (raw - raw_min) * depth_scale

    pts = np.stack([us_f, vs_f, depth_scaled], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd, mask, depth_scale, raw_min


def detect_iss_keypoints(pcd, gamma_21: float = 0.5, gamma_32: float = 0.5,
                         min_neighbors: int = 5) -> np.ndarray:
    """ISS keypoint detection. Hyperparameters match the existing Mitsubishi
    convention (salient_r = 6·nn_avg, non_max_r = 2·salient_r)."""
    if len(pcd.points) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    dists = pcd.compute_nearest_neighbor_distance()
    avg = float(np.mean(dists))
    salient_r = 6 * avg
    non_max_r = 2 * salient_r
    kp_pcd = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd,
        salient_radius=salient_r,
        non_max_radius=non_max_r,
        gamma_21=gamma_21,
        gamma_32=gamma_32,
        min_neighbors=min_neighbors,
    )
    return np.asarray(kp_pcd.points)  # (K, 3) in (u, v, depth_scaled)


def select_keypoints(iss_kp_3d: np.ndarray, zmap: np.ndarray,
                     max_num_keypoints: int = 512, erode_mask=None,
                     resize_factor: float = 0.5, rng=None):
    """Return (kp_resized(Nmax,2), scores(Nmax,), n_valid, kp_uv_orig(Nmax,2)).

    scores: ISS=1.0, random-fill=0.5, pad=0.0.
    kp_uv_orig are original-resolution (u,v) pixels. kp_resized = kp_uv_orig * resize_factor.
    If ISS >= Nmax, randomly subselect; otherwise ISS + random fill (prefer
    erode_mask; fall back to depth>0) up to Nmax; zero-pad if mask is too small.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_iss = int(len(iss_kp_3d))
    kp_uv_orig = np.zeros((max_num_keypoints, 2), dtype=np.float32)
    scores = np.zeros(max_num_keypoints, dtype=np.float32)

    if n_iss >= max_num_keypoints:
        idx = rng.choice(n_iss, max_num_keypoints, replace=False)
        sel = iss_kp_3d[idx, :2]
        kp_uv_orig[:] = sel.astype(np.float32)
        scores[:] = 1.0
        n_valid = max_num_keypoints
    else:
        if n_iss > 0:
            kp_uv_orig[:n_iss] = iss_kp_3d[:, :2].astype(np.float32)
            scores[:n_iss] = 1.0
        if erode_mask is not None:
            ys, xs = np.where(erode_mask > 0)
        else:
            ys, xs = np.where(zmap > 0)
        n_need = max_num_keypoints - n_iss
        n_rand = min(n_need, len(xs))
        if n_rand > 0:
            idx = rng.choice(len(xs), n_rand, replace=False)
            kp_uv_orig[n_iss:n_iss + n_rand, 0] = xs[idx].astype(np.float32)
            kp_uv_orig[n_iss:n_iss + n_rand, 1] = ys[idx].astype(np.float32)
            scores[n_iss:n_iss + n_rand] = 0.5
        n_valid = n_iss + n_rand

    kp_resized = kp_uv_orig * float(resize_factor)
    return kp_resized, scores, n_valid, kp_uv_orig
