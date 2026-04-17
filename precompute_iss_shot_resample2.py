"""
Resample_2 depth 이미지에 대해 ISS keypoints + SHOT descriptor 계산.
기존 precompute_iss_fpfh_resample2.py에서 FPFH → SHOT 교체.

사용법:
    conda activate LightGlue
    python precompute_iss_shot_resample2.py                        # 기본값
    python precompute_iss_shot_resample2.py --shot_radius 10.0
    python precompute_iss_shot_resample2.py --max_images 2 --force # 스모크 테스트

출력:
    gluefactory/datasets/mitsubishi/iss_shot_resample2_cache_r{radius}/{image_stem}.npz
    npz 구성: keypoints(512,2), keypoint_scores(512,), shot_descriptors(512,336),
              n_valid(int), n_iss(int)
"""

import argparse
import json
import numpy as np
import cv2
import open3d as o3d
import pyshot
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm

CROP_X0 = 1129
CROP_Y0 = 1081
CROP_SIZE = 3502
CLIP_START = 0.1
CLIP_END = 1000.0

# ── ISS 함수 (precompute_iss_fpfh_resample2.py에서 그대로 복사) ──────────────

def depth_crop_to_pcd(depth_crop_raw, erode_boundary=5):
    """ISS 검출용 PCD: (u, v, depth_scaled) 공간."""
    mask = (depth_crop_raw > 0).astype(np.uint8)
    if erode_boundary > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=erode_boundary)
    vs, us = np.where(mask > 0)
    d_raw = depth_crop_raw[vs, us].astype(np.float64)
    depth_real = CLIP_START + (d_raw / 65535.0) * (CLIP_END - CLIP_START)
    us_f, vs_f = us.astype(np.float64), vs.astype(np.float64)
    u_range = us_f.max() - us_f.min() if len(us_f) > 1 else 1.0
    d_min = float(depth_real.min())
    d_range = float(depth_real.max() - depth_real.min()) if len(depth_real) > 1 else 1.0
    depth_scale = u_range / d_range if d_range > 0 else 1.0
    depth_scaled = (depth_real - d_min) * depth_scale
    pts = np.stack([us_f, vs_f, depth_scaled], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd, pts, depth_scale, d_min, mask


def extract_iss_keypoints(pcd, gamma_21=0.5, gamma_32=0.5, min_neighbors=5):
    """ISS keypoint 검출."""
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    salient_radius = 6 * avg_dist
    non_max_radius = 2 * salient_radius
    kp_pcd = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd, salient_radius=salient_radius, non_max_radius=non_max_radius,
        gamma_21=gamma_21, gamma_32=gamma_32, min_neighbors=min_neighbors,
    )
    return np.asarray(kp_pcd.points)  # (K, 3): u, v, depth_scaled


def select_keypoints(iss_kp_3d, depth_crop_raw, max_num_keypoints, image_size,
                     depth_scale=1.0, depth_min=0.0, erode_mask=None):
    """ISS keypoints max_num 제한/보충 + resized 좌표 변환."""
    scale = image_size / CROP_SIZE
    n_iss = len(iss_kp_3d)
    if n_iss >= max_num_keypoints:
        indices = np.random.choice(n_iss, max_num_keypoints, replace=False)
        selected = iss_kp_3d[indices]
        n_valid = max_num_keypoints; n_iss_used = max_num_keypoints
    else:
        n_need = max_num_keypoints - n_iss
        if erode_mask is not None:
            ys, xs = np.where(erode_mask > 0)
        else:
            ys, xs = np.where(depth_crop_raw > 0)
        rand_indices = np.random.choice(len(xs), min(n_need, len(xs)), replace=False)
        rand_u = xs[rand_indices].astype(np.float64)
        rand_v = ys[rand_indices].astype(np.float64)
        rand_d_raw = depth_crop_raw[ys[rand_indices], xs[rand_indices]].astype(np.float64)
        rand_depth = CLIP_START + (rand_d_raw / 65535.0) * (CLIP_END - CLIP_START)
        rand_depth_scaled = (rand_depth - depth_min) * depth_scale
        rand_pts = np.stack([rand_u, rand_v, rand_depth_scaled], axis=1)
        selected = np.vstack([iss_kp_3d, rand_pts])
        n_valid = len(selected); n_iss_used = n_iss

    kp_crop = np.zeros((max_num_keypoints, 2), dtype=np.float32)
    kp_crop[:n_valid, 0] = selected[:n_valid, 0]
    kp_crop[:n_valid, 1] = selected[:n_valid, 1]
    keypoints = np.zeros((max_num_keypoints, 2), dtype=np.float32)
    keypoints[:n_valid] = kp_crop[:n_valid] * scale
    keypoint_scores = np.zeros(max_num_keypoints, dtype=np.float32)
    keypoint_scores[:n_iss_used] = 1.0
    keypoint_scores[n_iss_used:n_valid] = 0.5
    kp_3d = np.zeros((max_num_keypoints, 3), dtype=np.float64)
    kp_3d[:n_valid] = selected[:n_valid]
    return keypoints, keypoint_scores, n_valid, kp_crop, kp_3d


def kp_crop_to_xyz(kp_crop, n_valid, depth_crop_raw, fx, fy, cx, cy):
    """ISS keypoint crop 좌표 (u,v) → 진짜 (X,Y,Z) 변환."""
    kp_xyz = np.zeros((len(kp_crop), 3), dtype=np.float64)
    h, w = depth_crop_raw.shape
    for i in range(n_valid):
        u_i = int(round(kp_crop[i, 0])); v_i = int(round(kp_crop[i, 1]))
        u_i = max(0, min(u_i, w-1)); v_i = max(0, min(v_i, h-1))
        d_raw = depth_crop_raw[v_i, u_i]
        if d_raw > 0:
            d = CLIP_START + (d_raw / 65535.0) * (CLIP_END - CLIP_START)
            kp_xyz[i, 0] = (u_i - cx) * d / fx
            kp_xyz[i, 1] = (v_i - cy) * d / fy
            kp_xyz[i, 2] = d
    return kp_xyz


# ── SHOT 전용 함수 ────────────────────────────────────────────────────────────

def build_grid_mesh(depth_raw, fx, fy, cx, cy, subsample=1, mask=None):
    """Depth image → XYZ grid mesh for pyshot.

    Args:
        subsample: 픽셀 stride (depth_raw[::s, ::s] 적용 전 이미지 기준 intrinsics 전달)
    Returns:
        verts: (N, 3) float64 XYZ in mm
        faces: (M, 3) int64 triangle indices
        pixel_to_vertex: (H_s, W_s) int32
    """
    if subsample > 1:
        depth_s = depth_raw[::subsample, ::subsample]
        mask_s  = mask[::subsample, ::subsample] if mask is not None else None
        fx_s = fx / subsample; fy_s = fy / subsample
        cx_s = cx / subsample; cy_s = cy / subsample
    else:
        depth_s = depth_raw; mask_s = mask
        fx_s, fy_s, cx_s, cy_s = fx, fy, cx, cy

    H, W = depth_s.shape
    valid = (depth_s > 0) if mask_s is None else ((depth_s > 0) & (mask_s > 0))

    pixel_to_vertex = np.full((H, W), -1, dtype=np.int32)
    ys, xs = np.where(valid)
    pixel_to_vertex[ys, xs] = np.arange(len(ys), dtype=np.int32)

    d_raw = depth_s[ys, xs].astype(np.float64)
    depth_real = CLIP_START + (d_raw / 65535.0) * (CLIP_END - CLIP_START)
    X = (xs.astype(np.float64) - cx_s) * depth_real / fx_s
    Y = (ys.astype(np.float64) - cy_s) * depth_real / fy_s
    verts = np.stack([X, Y, depth_real], axis=1)

    vg, ug = np.mgrid[0:H-1, 0:W-1]
    vg, ug = vg.ravel(), ug.ravel()
    i00 = pixel_to_vertex[vg,     ug    ]
    i10 = pixel_to_vertex[vg,     ug + 1]
    i01 = pixel_to_vertex[vg + 1, ug    ]
    i11 = pixel_to_vertex[vg + 1, ug + 1]
    m1  = (i00 >= 0) & (i10 >= 0) & (i01 >= 0)
    m2  = (i10 >= 0) & (i11 >= 0) & (i01 >= 0)
    tri1 = np.stack([i00[m1], i10[m1], i01[m1]], axis=1)
    tri2 = np.stack([i10[m2], i11[m2], i01[m2]], axis=1)
    faces = np.concatenate([tri1, tri2], axis=0).astype(np.int64) \
        if (m1.any() or m2.any()) else np.zeros((0, 3), dtype=np.int64)

    return verts, faces, pixel_to_vertex


def compute_shot_for_keypoints(verts, faces, kp_xyz, n_valid,
                               shot_radius=10.0, n_bins=20):
    """SHOT descriptor를 keypoint XYZ 위치에서 KDTree lookup.

    Returns:
        shot_out: (max_num, 336) float32, L2 normalized
    """
    max_n = kp_xyz.shape[0]
    n_features = 16 * (n_bins + 1)   # 336 for n_bins=20
    shot_out = np.zeros((max_n, n_features), dtype=np.float32)

    if n_valid < 4 or len(faces) == 0:
        return shot_out

    shot_all = pyshot.get_descriptors(
        verts.astype(np.float64),
        faces.astype(np.int64),
        radius=float(shot_radius),
        local_rf_radius=float(shot_radius),
        n_bins=int(n_bins),
        double_volumes_sectors=False,   # ← 반드시 False (True → 672D)
        use_normalization=True,
        min_neighbors=3,
    ).astype(np.float32)  # (N, 336)

    tree = cKDTree(verts)
    _, idxs = tree.query(kp_xyz[:n_valid])
    shot_out[:n_valid] = shot_all[idxs]

    return shot_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_num_keypoints", type=int,   default=512)
    parser.add_argument("--gamma_21",          type=float, default=0.5)
    parser.add_argument("--gamma_32",          type=float, default=0.5)
    parser.add_argument("--min_neighbors",     type=int,   default=5)
    parser.add_argument("--shot_radius",       type=float, default=10.0)
    parser.add_argument("--n_bins",            type=int,   default=20)
    parser.add_argument("--image_size",        type=int,   default=1751)
    parser.add_argument("--mesh_subsample",    type=int,   default=4,
                        help="Grid mesh subsampling stride (4=balanced, 1=full resolution)")
    parser.add_argument("--erode_boundary",    type=int,   default=5)
    parser.add_argument("--fx",  type=float, default=8001.39)
    parser.add_argument("--fy",  type=float, default=8001.39)
    parser.add_argument("--cx_orig", type=float, default=2880.5)
    parser.add_argument("--cy_orig", type=float, default=2880.5)
    parser.add_argument("--max_images", type=int, default=None,
                        help="처음 N개 이미지만 처리 (스모크 테스트용)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cx = args.cx_orig - CROP_X0   # 1751.5
    cy = args.cy_orig - CROP_Y0   # 1799.5

    base_dir  = Path("gluefactory/datasets/mitsubishi")
    img_dir   = base_dir / "dataset_resample_2"
    cache_dir = base_dir / f"iss_shot_resample2_cache_r{args.shot_radius}"
    cache_dir.mkdir(exist_ok=True)

    image_paths = sorted(img_dir.glob("depth_raw_*.png"))
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    print(f"Found {len(image_paths)} images")
    print(f"Config: shot_radius={args.shot_radius}mm, n_bins={args.n_bins} → {16*(args.n_bins+1)}D")
    print(f"        mesh_subsample={args.mesh_subsample}, image_size={args.image_size}")

    stats = {"iss_counts": [], "valid_counts": []}

    for img_path in tqdm(image_paths, desc="Precomputing ISS+SHOT"):
        out_path = cache_dir / f"{img_path.stem}.npz"
        if out_path.exists() and not args.force:
            continue

        img_raw  = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        img_crop = img_raw[CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]

        # ISS 검출용 PCD (u, v, depth_scaled)
        pcd_iss, _, depth_scale, d_min, erode_mask = depth_crop_to_pcd(
            img_crop, erode_boundary=args.erode_boundary)

        iss_kp_3d = extract_iss_keypoints(
            pcd_iss, gamma_21=args.gamma_21,
            gamma_32=args.gamma_32, min_neighbors=args.min_neighbors)

        keypoints, scores, n_valid, kp_crop, _ = select_keypoints(
            iss_kp_3d, img_crop, args.max_num_keypoints, args.image_size,
            depth_scale=depth_scale, depth_min=d_min, erode_mask=erode_mask)

        # SHOT lookup용 XYZ 변환
        kp_xyz = kp_crop_to_xyz(kp_crop, n_valid, img_crop, args.fx, args.fy, cx, cy)

        # Grid mesh (XYZ) + SHOT
        verts, faces, _ = build_grid_mesh(
            img_crop, args.fx, args.fy, cx, cy,
            subsample=args.mesh_subsample, mask=erode_mask)

        shot = compute_shot_for_keypoints(
            verts, faces, kp_xyz, n_valid,
            shot_radius=args.shot_radius, n_bins=args.n_bins)

        np.savez_compressed(
            out_path,
            keypoints=keypoints,
            keypoint_scores=scores,
            shot_descriptors=shot,
            n_valid=np.array(n_valid),
            n_iss=np.array(len(iss_kp_3d)),
        )
        stats["iss_counts"].append(len(iss_kp_3d))
        stats["valid_counts"].append(n_valid)

    if stats["iss_counts"]:
        ic = np.array(stats["iss_counts"]); vc = np.array(stats["valid_counts"])
        print(f"\n--- Stats ---")
        print(f"ISS keypoints: mean={ic.mean():.1f}, min={ic.min()}, max={ic.max()}")
        print(f"Total keypoints: mean={vc.mean():.1f}")

    print(f"Done! → {cache_dir}/")
    json.dump(
        {**vars(args), "cx_crop": cx, "cy_crop": cy},
        open(cache_dir / "precompute_config.json", "w"), indent=2)


if __name__ == "__main__":
    main()
