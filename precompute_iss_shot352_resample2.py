"""
Resample_2 depth 이미지에 대해 ISS keypoints + C++ SHOT352 descriptor 계산.
pyshot 대신 precomputed C++ bins (output_shot_0407/) 사용.

ISS 검출: (u,v,depth_scaled) 공간 (기존과 동일)
SHOT352 : C++ bin 로드 → KDTree lookup (XYZ 공간)

좌표계 일치 근거:
  ISS kp (u_crop, v_crop) → XYZ: (u_crop - cx_crop) * Z / fx
  C++ cloud XYZ          : (u_orig - cx_orig) * Z / fx
  cx_crop = cx_orig - CROP_X0 = 2880.5 - 1129 = 1751.5  → 동일 좌표계

사용법:
    conda activate LightGlue
    cd /home/jhs/work/Registration/glue-factory_depth
    python precompute_iss_shot352_resample2.py                    # 전체
    python precompute_iss_shot352_resample2.py --max_images 3 --force  # 스모크 테스트

출력:
    gluefactory/datasets/mitsubishi/iss_shot352_resample2_cache/{stem}.npz
    npz: keypoints(512,2), keypoint_scores(512,), shot_descriptors(512,352),
         n_valid(int), n_iss(int)
"""

import argparse
import json
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm

CROP_X0   = 1129
CROP_Y0   = 1081
CROP_SIZE = 3502
CLIP_START = 0.1
CLIP_END   = 1000.0
SHOT_DIM   = 352


# ── C++ bin loader ────────────────────────────────────────────────────────────

def load_shot352_bin(bin_path):
    """C++ SHOT352 bin 로드.
    Returns:
        pts:  (N, 3) float32 XYZ in mm (full original image 기준)
        desc: (N, 352) float32, L2 normalized
    """
    with open(bin_path, 'rb') as f:
        N, D = np.frombuffer(f.read(8), dtype=np.uint32)
        N, D = int(N), int(D)
        assert D == SHOT_DIM, f"Expected D={SHOT_DIM}, got {D}"
        data = np.frombuffer(f.read(N * (3 + D) * 4), dtype=np.float32)
    data = data.reshape(N, 3 + D)
    pts  = data[:, :3]
    desc = data[:, 3:]
    valid = ~np.isnan(desc).any(axis=1)
    return pts[valid], desc[valid]


def lookup_shot352(pts_cloud, desc_cloud, kp_xyz, n_valid):
    """ISS keypoint XYZ → KDTree lookup in C++ cloud → SHOT352 descriptors.
    Returns:
        shot_out: (max_num, 352) float32
    """
    max_n = kp_xyz.shape[0]
    shot_out = np.zeros((max_n, SHOT_DIM), dtype=np.float32)
    if n_valid < 1 or len(pts_cloud) < 1:
        return shot_out
    tree = cKDTree(pts_cloud)
    _, idxs = tree.query(kp_xyz[:n_valid])
    shot_out[:n_valid] = desc_cloud[idxs]
    return shot_out


# ── ISS 함수 (precompute_iss_fpfh_resample2.py 와 동일) ──────────────────────

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
    d_min   = float(depth_real.min())
    d_range = float(depth_real.max() - depth_real.min()) if len(depth_real) > 1 else 1.0
    depth_scale = u_range / d_range if d_range > 0 else 1.0
    depth_scaled = (depth_real - d_min) * depth_scale
    pts = np.stack([us_f, vs_f, depth_scaled], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd, pts, depth_scale, d_min, mask


def extract_iss_keypoints(pcd, gamma_21=0.5, gamma_32=0.5, min_neighbors=5):
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
    scale  = image_size / CROP_SIZE
    n_iss  = len(iss_kp_3d)
    if n_iss >= max_num_keypoints:
        indices  = np.random.choice(n_iss, max_num_keypoints, replace=False)
        selected = iss_kp_3d[indices]
        n_valid  = max_num_keypoints; n_iss_used = max_num_keypoints
    else:
        n_need = max_num_keypoints - n_iss
        if erode_mask is not None:
            ys, xs = np.where(erode_mask > 0)
        else:
            ys, xs = np.where(depth_crop_raw > 0)
        rand_idx = np.random.choice(len(xs), min(n_need, len(xs)), replace=False)
        rand_u = xs[rand_idx].astype(np.float64)
        rand_v = ys[rand_idx].astype(np.float64)
        rand_d_raw   = depth_crop_raw[ys[rand_idx], xs[rand_idx]].astype(np.float64)
        rand_depth   = CLIP_START + (rand_d_raw / 65535.0) * (CLIP_END - CLIP_START)
        rand_scaled  = (rand_depth - depth_min) * depth_scale
        rand_pts     = np.stack([rand_u, rand_v, rand_scaled], axis=1)
        selected     = np.vstack([iss_kp_3d, rand_pts])
        n_valid      = len(selected); n_iss_used = n_iss

    kp_crop = np.zeros((max_num_keypoints, 2), dtype=np.float32)
    kp_crop[:n_valid, 0] = selected[:n_valid, 0]
    kp_crop[:n_valid, 1] = selected[:n_valid, 1]
    keypoints = np.zeros((max_num_keypoints, 2), dtype=np.float32)
    keypoints[:n_valid] = kp_crop[:n_valid] * scale
    keypoint_scores = np.zeros(max_num_keypoints, dtype=np.float32)
    keypoint_scores[:n_iss_used]      = 1.0
    keypoint_scores[n_iss_used:n_valid] = 0.5
    kp_3d = np.zeros((max_num_keypoints, 3), dtype=np.float64)
    kp_3d[:n_valid] = selected[:n_valid]
    return keypoints, keypoint_scores, n_valid, kp_crop, kp_3d


def kp_crop_to_xyz(kp_crop, n_valid, depth_crop_raw, fx, fy, cx, cy):
    """ISS keypoint crop 좌표 (u,v) → 진짜 (X,Y,Z) in mm."""
    kp_xyz = np.zeros((len(kp_crop), 3), dtype=np.float64)
    h, w   = depth_crop_raw.shape
    for i in range(n_valid):
        u_i = int(round(kp_crop[i, 0])); v_i = int(round(kp_crop[i, 1]))
        u_i = max(0, min(u_i, w - 1));   v_i = max(0, min(v_i, h - 1))
        d_raw = depth_crop_raw[v_i, u_i]
        if d_raw > 0:
            d = CLIP_START + (d_raw / 65535.0) * (CLIP_END - CLIP_START)
            kp_xyz[i, 0] = (u_i - cx) * d / fx
            kp_xyz[i, 1] = (v_i - cy) * d / fy
            kp_xyz[i, 2] = d
    return kp_xyz


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_num_keypoints", type=int,   default=512)
    parser.add_argument("--gamma_21",          type=float, default=0.5)
    parser.add_argument("--gamma_32",          type=float, default=0.5)
    parser.add_argument("--min_neighbors",     type=int,   default=5)
    parser.add_argument("--erode_boundary",    type=int,   default=5)
    parser.add_argument("--image_size",        type=int,   default=1751)
    parser.add_argument("--fx",  type=float, default=8001.39)
    parser.add_argument("--fy",  type=float, default=8001.39)
    parser.add_argument("--cx_orig", type=float, default=2880.5)
    parser.add_argument("--cy_orig", type=float, default=2880.5)
    parser.add_argument("--bin_dir", type=str,
                        default="gluefactory/datasets/Descriptor/output_shot_0407")
    parser.add_argument("--max_images", type=int, default=None,
                        help="처음 N개만 처리 (스모크 테스트용)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # cx/cy를 crop 기준으로 변환 (검증 완료: ISS kp XYZ = C++ cloud XYZ)
    cx = args.cx_orig - CROP_X0   # 1751.5
    cy = args.cy_orig - CROP_Y0   # 1799.5

    base_dir  = Path("gluefactory/datasets/mitsubishi")
    img_dir   = base_dir / "dataset_resample_2"
    bin_dir   = Path(args.bin_dir)
    cache_dir = base_dir / "iss_shot352_resample2_cache"
    cache_dir.mkdir(exist_ok=True)

    image_paths = sorted(img_dir.glob("depth_raw_*.png"))
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    print(f"Found {len(image_paths)} images")
    print(f"Config: SHOT352 (352D, L2-norm), C++ bins from {bin_dir.name}/")
    print(f"Cache → {cache_dir}/")

    stats = {"iss_counts": [], "valid_counts": []}

    for img_path in tqdm(image_paths, desc="Precomputing ISS+SHOT352"):
        out_path = cache_dir / f"{img_path.stem}.npz"
        if out_path.exists() and not args.force:
            continue

        # ── C++ SHOT352 bin 로드 ──
        bin_path = bin_dir / f"{img_path.stem}_shot352.bin"
        if not bin_path.exists():
            print(f"\n  [SKIP] bin not found: {bin_path.name}")
            continue
        pts_cloud, desc_cloud = load_shot352_bin(bin_path)

        # ── ISS 검출 (crop, u/v/depth_scaled 공간) ──
        img_raw  = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        img_crop = img_raw[CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]

        pcd_iss, _, depth_scale, d_min, erode_mask = depth_crop_to_pcd(
            img_crop, erode_boundary=args.erode_boundary)

        iss_kp_3d = extract_iss_keypoints(
            pcd_iss, gamma_21=args.gamma_21,
            gamma_32=args.gamma_32, min_neighbors=args.min_neighbors)

        keypoints, scores, n_valid, kp_crop, _ = select_keypoints(
            iss_kp_3d, img_crop, args.max_num_keypoints, args.image_size,
            depth_scale=depth_scale, depth_min=d_min, erode_mask=erode_mask)

        # ── keypoint (u_crop, v_crop) → XYZ ──
        kp_xyz = kp_crop_to_xyz(kp_crop, n_valid, img_crop, args.fx, args.fy, cx, cy)

        # ── C++ cloud KDTree lookup → SHOT352 ──
        shot = lookup_shot352(pts_cloud, desc_cloud, kp_xyz, n_valid)

        np.savez_compressed(
            out_path,
            keypoints        = keypoints,
            keypoint_scores  = scores,
            shot_descriptors = shot,      # (512, 352)
            n_valid          = np.array(n_valid),
            n_iss            = np.array(len(iss_kp_3d)),
        )
        stats["iss_counts"].append(len(iss_kp_3d))
        stats["valid_counts"].append(n_valid)

    if stats["iss_counts"]:
        ic = np.array(stats["iss_counts"]); vc = np.array(stats["valid_counts"])
        print(f"\n--- Stats ---")
        print(f"ISS keypoints : mean={ic.mean():.1f}, min={ic.min()}, max={ic.max()}")
        print(f"Total kp used : mean={vc.mean():.1f}")

    print(f"Done! → {cache_dir}/")
    json.dump(vars(args), open(cache_dir / "precompute_config.json", "w"), indent=2)


if __name__ == "__main__":
    main()
