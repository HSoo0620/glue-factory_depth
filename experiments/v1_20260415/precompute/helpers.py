"""v1 precompute 공통 헬퍼.

좌표계: 모두 mm. zmap PNG → (X=u·0.056, Y=v·0.056, Z=raw·0.0085) mm.
ISS는 voxel-downsampled cloud 위에서 검출 → keypoint indices 자동 매핑.
zero-pad target: (W=2432, H=3008). 2026-04-15 NAS 스캔 결과 W=2413 상수,
H=1556~2975 가변 → 32배수 round-up + H 여유 33px.
"""
import numpy as np
import open3d as o3d
import cv2
from PIL import Image
from pathlib import Path

LATERAL_MM = 0.056
TRANSPORT_MM = 0.056
VERTICAL_MM = 0.0085
PAD_W = 2432
PAD_H = 3008
VOXEL_SIZE = 1.0
NORMAL_RADIUS = 20.0
FPFH_RADIUS = 20.0
SHOT_RADIUS = 40.0
MAX_KEYPOINTS = 512
ISS_GAMMA_21 = 0.5
ISS_GAMMA_32 = 0.5
ISS_MIN_NEIGHBORS = 5


def load_zmap_to_pcd_mm(png_path):
    """16-bit depth PNG → (X,Y,Z) mm PCD + (u,v) 매핑.
    Returns:
        pts_mm: (N, 3) float32, mm
        uv:     (N, 2) float32, image pixel (u,v)
        zmap_shape: (H, W) original
    """
    img = np.array(Image.open(png_path))
    if img.dtype == np.uint8:
        img = img.astype(np.uint16) * 257
    H, W = img.shape
    mask = img > 0
    vs, us = np.where(mask)
    zs = img[mask].astype(np.float32)
    pts = np.column_stack([
        us.astype(np.float32) * LATERAL_MM,
        vs.astype(np.float32) * TRANSPORT_MM,
        zs * VERTICAL_MM,
    ]).astype(np.float32)
    uv = np.column_stack([us, vs]).astype(np.float32)
    return pts, uv, (H, W)


def voxel_downsample(pts_mm, voxel_size=VOXEL_SIZE):
    """Open3D VoxelGrid. (mm 단위) → P_vox."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_mm.astype(np.float64))
    pcd_vox = pcd.voxel_down_sample(voxel_size)
    return pcd_vox  # Open3D PointCloud


def extract_iss_on_voxel(pcd_vox):
    """ISS keypoint detection on voxel cloud (mm).
    Deprecated: ISS는 dense PCD 에서 수행해야 함. `extract_iss_on_dense` 사용 권장.
    Returns:
        kp_xyz: (K, 3) float32 mm (ISS keypoint 좌표)
    """
    distances = pcd_vox.compute_nearest_neighbor_distance()
    if len(distances) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    avg_dist = float(np.mean(distances))
    salient_radius = 6.0 * avg_dist
    non_max_radius = 2.0 * salient_radius
    kp_pcd = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd_vox,
        salient_radius=salient_radius,
        non_max_radius=non_max_radius,
        gamma_21=ISS_GAMMA_21,
        gamma_32=ISS_GAMMA_32,
        min_neighbors=ISS_MIN_NEIGHBORS,
    )
    return np.asarray(kp_pcd.points, dtype=np.float32)


def extract_iss_on_dense(pts_mm):
    """ISS keypoint detection on full-density (no voxel) mm PCD.
    voxel downsample은 descriptor 계산용, ISS detection은 dense 에서 수행.
    Returns:
        kp_xyz: (K, 3) float32 mm
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_mm.astype(np.float64))
    distances = pcd.compute_nearest_neighbor_distance()
    if len(distances) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    avg_dist = float(np.mean(distances))
    salient_radius = 6.0 * avg_dist
    non_max_radius = 2.0 * salient_radius
    kp_pcd = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd,
        salient_radius=salient_radius,
        non_max_radius=non_max_radius,
        gamma_21=ISS_GAMMA_21,
        gamma_32=ISS_GAMMA_32,
        min_neighbors=ISS_MIN_NEIGHBORS,
    )
    return np.asarray(kp_pcd.points, dtype=np.float32)


def map_kp_to_voxel_indices(kp_xyz, pcd_vox):
    """ISS keypoint XYZ를 P_vox 인덱스로 매핑 (KDTree 1-NN).
    Returns:
        indices: (K,) int64
        dists:   (K,) float32 (1-NN 거리, mm. 정상 ≈ 0)
    """
    from scipy.spatial import cKDTree
    vox_pts = np.asarray(pcd_vox.points, dtype=np.float32)
    if len(kp_xyz) == 0 or len(vox_pts) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    tree = cKDTree(vox_pts)
    d, idx = tree.query(kp_xyz, k=1)
    return idx.astype(np.int64), d.astype(np.float32)


def subsample_or_pad_keypoints(kp_xyz, pcd_vox, max_n=MAX_KEYPOINTS, seed=None):
    """K → 정확히 max_n 개로 subsample 또는 random 보충.
    Returns:
        kp_xyz_out:  (max_n, 3) float32 mm
        kp_indices:  (max_n,) int64 (P_vox 인덱스)
        kp_scores:   (max_n,) float32 (ISS=1.0, 보충=0.0)
        n_iss:       int (원본 ISS 검출 수)
        n_valid:     int (실제 채워진 수, 항상 max_n 또는 그보다 작은 값)
    """
    rng = np.random.default_rng(seed)
    vox_pts = np.asarray(pcd_vox.points, dtype=np.float32)
    K = len(kp_xyz)

    kp_xyz_out = np.zeros((max_n, 3), dtype=np.float32)
    kp_scores = np.zeros((max_n,), dtype=np.float32)
    kp_idx_out = np.zeros((max_n,), dtype=np.int64)

    if K >= max_n:
        sel = rng.choice(K, max_n, replace=False)
        kp_xyz_out[:] = kp_xyz[sel]
        idx, _ = map_kp_to_voxel_indices(kp_xyz_out, pcd_vox)
        kp_idx_out[:] = idx
        kp_scores[:] = 1.0
        return kp_xyz_out, kp_idx_out, kp_scores, K, max_n

    # K < max_n: ISS 모두 + 보충 (random from P_vox \ iss)
    iss_idx, _ = map_kp_to_voxel_indices(kp_xyz, pcd_vox)
    mask = np.ones(len(vox_pts), dtype=bool)
    mask[iss_idx] = False
    candidates = np.nonzero(mask)[0].astype(np.int64)
    n_need = max_n - K
    if len(candidates) >= n_need:
        rand_idx = candidates[rng.choice(len(candidates), n_need, replace=False)]
    else:
        # 극단적인 경우: 보충도 부족 → 가능한 만큼만, 나머지는 0 padding
        rand_idx = candidates
    kp_xyz_out[:K] = kp_xyz
    kp_idx_out[:K] = iss_idx
    kp_scores[:K] = 1.0
    n_filled = K + len(rand_idx)
    if len(rand_idx) > 0:
        kp_xyz_out[K:K + len(rand_idx)] = vox_pts[rand_idx]
        kp_idx_out[K:K + len(rand_idx)] = rand_idx
    # scores 보충분은 0.0 유지
    return kp_xyz_out, kp_idx_out, kp_scores, K, n_filled


def kp_xyz_mm_to_uv_padded(kp_xyz_mm, pad_w=PAD_W, pad_h=PAD_H):
    """mm 좌표 → 픽셀 (u, v). Zero-pad 좌표계 (원본과 동일, padding은 우/하단 추가).
    u = X / LATERAL_MM, v = Y / TRANSPORT_MM
    Returns:
        kp_uv: (K, 2) float32
    """
    u = kp_xyz_mm[:, 0] / LATERAL_MM
    v = kp_xyz_mm[:, 1] / TRANSPORT_MM
    return np.stack([u, v], axis=1).astype(np.float32)


def zero_pad_zmap(img, pad_w=PAD_W, pad_h=PAD_H):
    """zmap PNG (uint16) → zero-pad to (pad_h, pad_w). 우/하단에만 0 padding."""
    H, W = img.shape
    if H > pad_h or W > pad_w:
        raise ValueError(f"image ({H},{W}) exceeds pad target ({pad_h},{pad_w})")
    out = np.zeros((pad_h, pad_w), dtype=img.dtype)
    out[:H, :W] = img
    return out
