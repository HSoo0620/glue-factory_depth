"""
Resample_2 depth 이미지에 대해 ISS keypoints + Dense FPFH 계산 (Grid 좌표계).
FPFH를 Grid 좌표 (u*grid_dx, v*grid_dy, depth_real) 공간에서 계산.
ISS 검출은 기존과 동일하게 (u, v, depth_scaled) 공간에서 수행.

사용법:
    python precompute_iss_fpfh_resample2_grid.py
    python precompute_iss_fpfh_resample2_grid.py --fpfh_radius 10.0 --image_size 1751
    python precompute_iss_fpfh_resample2_grid.py --force

출력:
    gluefactory/datasets/mitsubishi/iss_fpfh_resample2_cache_r{radius}_grid/{image_stem}.npz
"""

import argparse
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from tqdm import tqdm


# 고정 crop 파라미터
CROP_X0 = 1129
CROP_Y0 = 1081
CROP_SIZE = 3502

# depth 변환 상수
CLIP_START = 0.1
CLIP_END = 1000.0

# Grid 좌표 변환 상수 (calib INI 기준)
GRID_DX = 0.05
GRID_DY = 0.05


def depth_crop_to_pcd(depth_crop_raw, erode_boundary=5):
    """crop된 depth에서 스케일 보정된 point cloud 생성 (ISS 검출용).

    경계 erosion으로 실루엣 경계 제거 + depth range를 u,v range에 맞춰 스케일링.

    Returns:
        pcd: Open3D PointCloud (depth 스케일 보정됨)
        pts: (N, 3) 보정된 좌표
        depth_scale: depth → pixel 변환 스케일
        depth_min: depth_real 최소값 (역변환용)
        erode_mask: erosion 적용된 마스크
    """
    mask = (depth_crop_raw > 0).astype(np.uint8)

    # 경계 erosion: 실루엣 경계 ISS 검출 방지
    if erode_boundary > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=erode_boundary)

    vs, us = np.where(mask > 0)
    d_raw = depth_crop_raw[vs, us].astype(np.float64)
    depth_real = CLIP_START + (d_raw / 65535.0) * (CLIP_END - CLIP_START)

    us_f = us.astype(np.float64)
    vs_f = vs.astype(np.float64)

    # Depth 스케일 보정: depth range를 u,v range에 맞춤
    u_range = us_f.max() - us_f.min() if len(us_f) > 1 else 1.0
    d_min = float(depth_real.min())
    d_range = float(depth_real.max() - depth_real.min()) if len(depth_real) > 1 else 1.0
    depth_scale = u_range / d_range if d_range > 0 else 1.0
    depth_scaled = (depth_real - d_min) * depth_scale

    pts = np.stack([us_f, vs_f, depth_scaled], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    return pcd, pts, depth_scale, d_min, mask


def build_grid_pcd(depth_crop_raw, mask=None):
    """depth crop → Grid 좌표 (u*grid_dx, v*grid_dy, depth_real) point cloud 생성 (FPFH 계산용).

    Args:
        mask: erosion 마스크. None이면 depth>0 영역 전체 사용.
    Returns:
        pcd: Open3D PointCloud (Grid 좌표)
    """
    if mask is None:
        mask = (depth_crop_raw > 0).astype(np.uint8)

    vs, us = np.where(mask > 0)
    d_raw = depth_crop_raw[vs, us].astype(np.float64)
    depth_real = CLIP_START + (d_raw / 65535.0) * (CLIP_END - CLIP_START)

    gx = us.astype(np.float64) * GRID_DX
    gy = vs.astype(np.float64) * GRID_DY
    gz = depth_real

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack([gx, gy, gz], axis=1))
    return pcd


def kp_crop_to_grid(kp_crop, n_valid, depth_crop_raw):
    """ISS keypoint의 crop 좌표 (u,v) → Grid 좌표 변환 (FPFH KDTree 조회용).

    Args:
        kp_crop: (max_num, 2) crop 좌표 (u,v)
        n_valid: 유효 keypoint 수
    Returns:
        kp_grid: (max_num, 3) Grid 좌표 (u*grid_dx, v*grid_dy, depth_real)
    """
    kp_grid = np.zeros((len(kp_crop), 3), dtype=np.float64)
    h, w = depth_crop_raw.shape
    for i in range(n_valid):
        u_i = int(round(kp_crop[i, 0]))
        v_i = int(round(kp_crop[i, 1]))
        u_i = max(0, min(u_i, w - 1))
        v_i = max(0, min(v_i, h - 1))
        d_raw = depth_crop_raw[v_i, u_i]
        if d_raw > 0:
            d = CLIP_START + (d_raw / 65535.0) * (CLIP_END - CLIP_START)
            kp_grid[i, 0] = u_i * GRID_DX
            kp_grid[i, 1] = v_i * GRID_DY
            kp_grid[i, 2] = d
    return kp_grid


def extract_iss_keypoints(pcd, gamma_21=0.5, gamma_32=0.5, min_neighbors=5):
    """ISS keypoint 검출. (u, v, depth_real) 좌표 반환."""
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    salient_radius = 6 * avg_dist
    non_max_radius = 2 * salient_radius

    keypoints_pcd = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd,
        salient_radius=salient_radius,
        non_max_radius=non_max_radius,
        gamma_21=gamma_21,
        gamma_32=gamma_32,
        min_neighbors=min_neighbors,
    )
    return np.asarray(keypoints_pcd.points)  # (K, 3) — u, v, depth_real


def select_keypoints(iss_kp_3d, depth_crop_raw, max_num_keypoints, image_size,
                     depth_scale=1.0, depth_min=0.0, erode_mask=None):
    """ISS keypoints를 max_num 개로 제한/보충하고, resized 좌표로 변환.

    Returns:
        keypoints: (max_num, 2) — resized 좌표
        keypoint_scores: (max_num,) — ISS=1.0, random=0.5, padding=0.0
        n_valid: 유효 keypoint 수 (ISS + random 포함)
        kp_crop: (max_num, 2) — crop 좌표 (FPFH lookup용)
    """
    scale = image_size / CROP_SIZE
    n_iss = len(iss_kp_3d)

    if n_iss >= max_num_keypoints:
        # 랜덤으로 max_num 개 선택
        indices = np.random.choice(n_iss, max_num_keypoints, replace=False)
        selected = iss_kp_3d[indices]
        n_valid = max_num_keypoints
        n_iss_used = max_num_keypoints
    else:
        # ISS keypoints 전부 + random 보충
        n_need = max_num_keypoints - n_iss

        # erode_mask 영역에서 random sampling (경계 제외)
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
        n_valid = len(selected)
        n_iss_used = n_iss

    # crop 좌표 (u, v) — FPFH lookup용
    kp_crop = np.zeros((max_num_keypoints, 2), dtype=np.float32)
    kp_crop[:n_valid, 0] = selected[:n_valid, 0]  # u
    kp_crop[:n_valid, 1] = selected[:n_valid, 1]  # v

    # resized 좌표
    keypoints = np.zeros((max_num_keypoints, 2), dtype=np.float32)
    keypoints[:n_valid] = kp_crop[:n_valid] * scale

    # scores: ISS keypoints = 1.0, random = 0.5, padding = 0.0
    keypoint_scores = np.zeros(max_num_keypoints, dtype=np.float32)
    keypoint_scores[:n_iss_used] = 1.0
    keypoint_scores[n_iss_used:n_valid] = 0.5

    # ISS keypoints의 3D 좌표도 보존 (FPFH lookup용)
    kp_3d = np.zeros((max_num_keypoints, 3), dtype=np.float64)
    kp_3d[:n_valid] = selected[:n_valid]

    return keypoints, keypoint_scores, n_valid, kp_crop, kp_3d


def compute_fpfh_for_keypoints(pcd, kp_grid, n_valid,
                               fpfh_radius=10.0, fpfh_max_nn=100):
    """Dense FPFH에서 keypoint 위치의 descriptor를 KDTree lookup (Grid 좌표)."""
    max_n = kp_grid.shape[0]
    fpfh_out = np.zeros((max_n, 33), dtype=np.float32)

    if n_valid < 3:
        return fpfh_out

    # Normal 추정
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=fpfh_radius * 2, max_nn=30
        )
    )

    # FPFH 계산
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=fpfh_max_nn),
    )
    fpfh_dense = np.array(fpfh.data).T.astype(np.float32)

    # KDTree로 keypoint 위치에서 lookup (Grid 좌표 직접 사용)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for i in range(n_valid):
        query = kp_grid[i]  # (u*grid_dx, v*grid_dy, depth_real)
        _, idx, _ = kdtree.search_knn_vector_3d(query, 1)
        fpfh_out[i] = fpfh_dense[idx[0]]

    # L2 normalize
    norms = np.linalg.norm(fpfh_out[:n_valid], axis=1, keepdims=True)
    fpfh_out[:n_valid] = fpfh_out[:n_valid] / (norms + 1e-8)

    return fpfh_out


def main():
    parser = argparse.ArgumentParser(description="Resample_2용 ISS + Dense FPFH precomputation (Grid 좌표)")
    parser.add_argument("--max_num_keypoints", type=int, default=512)
    parser.add_argument("--gamma_21", type=float, default=0.5)
    parser.add_argument("--gamma_32", type=float, default=0.5)
    parser.add_argument("--min_neighbors", type=int, default=5)
    parser.add_argument("--fpfh_radius", type=float, default=10.0,
                        help="FPFH 검색 반경 (Grid 좌표 기준)")
    parser.add_argument("--fpfh_max_nn", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=1751)
    parser.add_argument("--erode_boundary", type=int, default=5,
                        help="경계 erosion 반복 횟수 (실루엣 경계 ISS 제거)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_dir = Path("gluefactory/datasets/mitsubishi")
    img_dir = base_dir / "dataset_resample_2"
    cache_dir = base_dir / f"iss_fpfh_resample2_cache_r{args.fpfh_radius}_grid"
    cache_dir.mkdir(exist_ok=True)

    image_paths = sorted(img_dir.glob("depth_raw_*.png"))
    print(f"Found {len(image_paths)} resample_2 depth images")
    print(f"Config: max_kp={args.max_num_keypoints}, gamma_21={args.gamma_21}, "
          f"gamma_32={args.gamma_32}, fpfh_radius={args.fpfh_radius}")
    print(f"Grid: dx={GRID_DX}, dy={GRID_DY}")
    print(f"Crop: ({CROP_X0}, {CROP_Y0}), size={CROP_SIZE}")

    stats = {"total": 0, "iss_counts": [], "valid_counts": []}

    for img_path in tqdm(image_paths, desc="Precomputing ISS+FPFH (resample_2, grid)"):
        out_path = cache_dir / f"{img_path.stem}.npz"
        if out_path.exists() and not args.force:
            continue

        # 원본 이미지 로드 + 고정 crop
        img_raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        img_crop = img_raw[CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]

        # ISS 검출용 PCD (기존 방식: u,v,depth_scaled)
        pcd_iss, dense_pts, depth_scale, d_min, erode_mask = depth_crop_to_pcd(
            img_crop, erode_boundary=args.erode_boundary
        )

        # ISS keypoint 검출
        iss_kp_3d = extract_iss_keypoints(
            pcd_iss,
            gamma_21=args.gamma_21,
            gamma_32=args.gamma_32,
            min_neighbors=args.min_neighbors,
        )

        n_iss = len(iss_kp_3d)

        # keypoint 선택 (max_num 제한/보충) + 좌표 변환
        keypoints, scores, n_valid, kp_crop, kp_3d = select_keypoints(
            iss_kp_3d, img_crop, args.max_num_keypoints, args.image_size,
            depth_scale=depth_scale, depth_min=d_min, erode_mask=erode_mask,
        )

        # FPFH용 Grid PCD (u*grid_dx, v*grid_dy, depth_real)
        pcd_grid = build_grid_pcd(img_crop, mask=erode_mask)

        # ISS keypoint crop 좌표 → Grid 좌표 변환 (FPFH lookup용)
        kp_grid = kp_crop_to_grid(kp_crop, n_valid, img_crop)

        # FPFH 계산 (Grid PCD 기반)
        fpfh = compute_fpfh_for_keypoints(
            pcd_grid, kp_grid, n_valid,
            fpfh_radius=args.fpfh_radius,
            fpfh_max_nn=args.fpfh_max_nn,
        )

        # 저장 (keypoints는 resized 이미지 기준 좌표)
        np.savez_compressed(
            out_path,
            keypoints=keypoints,
            keypoint_scores=scores,
            fpfh_descriptors=fpfh,
            n_valid=np.array(n_valid),
            n_iss=np.array(n_iss),
        )

        stats["total"] += 1
        stats["iss_counts"].append(n_iss)
        stats["valid_counts"].append(n_valid)

    if stats["iss_counts"]:
        ic = np.array(stats["iss_counts"])
        vc = np.array(stats["valid_counts"])
        print(f"\n--- Stats ---")
        print(f"Processed: {stats['total']} images")
        print(f"ISS keypoints: mean={ic.mean():.1f}, min={ic.min()}, max={ic.max()}")
        print(f"Total keypoints (ISS+random): mean={vc.mean():.1f}, min={vc.min()}, max={vc.max()}")

    print(f"Done! Cached to {cache_dir}/")

    import json
    config_path = cache_dir / "precompute_config.json"
    json.dump({
        **vars(args),
        "coord_space": "grid",
        "grid_dx": GRID_DX,
        "grid_dy": GRID_DY,
    }, open(config_path, "w"), indent=2)


if __name__ == "__main__":
    main()
