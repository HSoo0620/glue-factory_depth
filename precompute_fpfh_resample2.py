"""
Resample_2 depth 이미지에 대해 SuperPoint keypoints + Dense FPFH 계산.
카메라 intrinsic 없이 (u, v, depth_real)을 직접 3D 좌표로 사용.
고정 crop (1129, 1081, 3502x3502) 적용 후 resize → SP keypoint 추출.

사용법:
    python precompute_fpfh_resample2.py
    python precompute_fpfh_resample2.py --fpfh_radius 5.0 --image_size 1751
    python precompute_fpfh_resample2.py --force

출력:
    gluefactory/datasets/mitsubishi/fpfh_resample2_cache_r{radius}/{image_stem}.npz
"""

import argparse
import torch
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


def load_superpoint(conf):
    from gluefactory.models.extractors.superpoint_open import SuperPoint
    model = SuperPoint(conf)
    model.eval()
    return model


def filter_keypoints_by_depth(keypoints, scores, depth_map, max_num_keypoints):
    N = keypoints.shape[0]
    valid_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
        x = np.clip(x, 0, depth_map.shape[1] - 1)
        y = np.clip(y, 0, depth_map.shape[0] - 1)
        valid_mask[i] = depth_map[y, x] > 0

    valid_kp = keypoints[valid_mask]
    valid_sc = scores[valid_mask]

    if len(valid_sc) > 0:
        order = np.argsort(-valid_sc)
        valid_kp = valid_kp[order]
        valid_sc = valid_sc[order]

    n_valid = min(len(valid_kp), max_num_keypoints)

    out_kp = np.zeros((max_num_keypoints, 2), dtype=np.float32)
    out_sc = np.zeros(max_num_keypoints, dtype=np.float32)

    out_kp[:n_valid] = valid_kp[:n_valid]
    out_sc[:n_valid] = valid_sc[:n_valid]

    return out_kp, out_sc, n_valid


def compute_fpfh_for_keypoints(depth_crop_raw, keypoints_crop, n_valid,
                               fpfh_radius=5.0, fpfh_max_nn=100):
    """crop된 depth에서 (u, v, depth_real)로 FPFH 계산.

    keypoints_crop: (max_n, 2) — crop 좌표계 (resize 전)
    depth_crop_raw: (H, W) uint16 — crop된 원본 depth
    """
    max_n = keypoints_crop.shape[0]
    fpfh_out = np.zeros((max_n, 33), dtype=np.float32)

    if n_valid < 3:
        return fpfh_out

    # Dense point cloud: (u, v, depth_real)
    vs, us = np.where(depth_crop_raw > 0)
    d_raw = depth_crop_raw[vs, us].astype(np.float64)
    depth_real = CLIP_START + (d_raw / 65535.0) * (CLIP_END - CLIP_START)
    dense_pts = np.stack([us.astype(np.float64), vs.astype(np.float64), depth_real], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dense_pts)

    # Normal 추정 (radius = fpfh_radius * 2)
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

    # KDTree로 keypoint 위치에서 lookup
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    valid_kp = keypoints_crop[:n_valid]
    for i in range(n_valid):
        u = int(np.clip(valid_kp[i, 0], 0, depth_crop_raw.shape[1] - 1))
        v = int(np.clip(valid_kp[i, 1], 0, depth_crop_raw.shape[0] - 1))
        d = float(depth_crop_raw[v, u])
        if d <= 0:
            continue
        depth_val = CLIP_START + (d / 65535.0) * (CLIP_END - CLIP_START)
        _, idx, _ = kdtree.search_knn_vector_3d([u, v, depth_val], 1)
        fpfh_out[i] = fpfh_dense[idx[0]]

    # L2 normalize
    norms = np.linalg.norm(fpfh_out[:n_valid], axis=1, keepdims=True)
    fpfh_out[:n_valid] = fpfh_out[:n_valid] / (norms + 1e-8)

    return fpfh_out


def main():
    parser = argparse.ArgumentParser(description="Resample_2용 SuperPoint + Dense FPFH precomputation")
    parser.add_argument("--max_num_keypoints", type=int, default=512)
    parser.add_argument("--detection_threshold", type=float, default=0.001)
    parser.add_argument("--nms_radius", type=int, default=3)
    parser.add_argument("--fpfh_radius", type=float, default=5.0)
    parser.add_argument("--fpfh_max_nn", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=1751)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_dir = Path("gluefactory/datasets/mitsubishi")
    img_dir = base_dir / "dataset_resample_2"
    cache_dir = base_dir / f"fpfh_resample2_cache_r{args.fpfh_radius}"
    cache_dir.mkdir(exist_ok=True)

    sp_conf = {
        "descriptor_dim": 256,
        "nms_radius": args.nms_radius,
        "max_num_keypoints": args.max_num_keypoints * 2,
        "force_num_keypoints": False,
        "detection_threshold": args.detection_threshold,
        "remove_borders": 4,
        "channels": [64, 64, 128, 128, 256],
        "dense_outputs": None,
        "weights": None,
    }

    device = args.device if torch.cuda.is_available() else "cpu"
    sp_model = load_superpoint(sp_conf).to(device)

    image_paths = sorted(img_dir.glob("depth_raw_*.png"))
    print(f"Found {len(image_paths)} resample_2 depth images")
    print(f"Config: max_kp={args.max_num_keypoints}, det_th={args.detection_threshold}, "
          f"fpfh_radius={args.fpfh_radius}, image_size={args.image_size}")
    print(f"Crop: ({CROP_X0}, {CROP_Y0}), size={CROP_SIZE}")

    stats = {"total": 0, "valid_counts": []}

    for img_path in tqdm(image_paths, desc="Precomputing FPFH (resample_2)"):
        out_path = cache_dir / f"{img_path.stem}.npz"
        if out_path.exists() and not args.force:
            continue

        # 원본 이미지 로드
        img_raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        # 고정 crop
        img_crop = img_raw[CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]

        # resize (SP 입력용)
        if args.image_size != CROP_SIZE:
            img_resized = cv2.resize(
                img_crop, (args.image_size, args.image_size),
                interpolation=cv2.INTER_NEAREST
            )
            scale = CROP_SIZE / args.image_size  # resized → crop 스케일
        else:
            img_resized = img_crop
            scale = 1.0

        # SuperPoint 입력용 정규화
        depth_norm = img_resized.astype(np.float32) / 65535.0
        h, w = depth_norm.shape[:2]

        # SuperPoint keypoint 추출 (resized 이미지에서)
        img_tensor = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            sp_pred = sp_model({"image": img_tensor, "image_size": torch.tensor([[h, w]])})

        raw_kp = sp_pred["keypoints"][0].cpu().numpy()
        raw_sc = sp_pred["keypoint_scores"][0].cpu().numpy()

        # depth > 0 필터링 (resized 이미지 기준)
        keypoints, scores, n_valid = filter_keypoints_by_depth(
            raw_kp, raw_sc, img_resized, args.max_num_keypoints
        )

        # keypoint를 crop 좌표계로 변환 (FPFH 계산용)
        kp_crop = keypoints.copy()
        kp_crop[:n_valid] *= scale

        # FPFH 계산: crop된 원본 depth에서 (u, v, depth_real)
        fpfh = compute_fpfh_for_keypoints(
            img_crop, kp_crop, n_valid,
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
        )

        stats["total"] += 1
        stats["valid_counts"].append(n_valid)

    if stats["valid_counts"]:
        vc = np.array(stats["valid_counts"])
        print(f"\n--- Stats ---")
        print(f"Processed: {stats['total']} images")
        print(f"Valid keypoints: mean={vc.mean():.1f}, min={vc.min()}, max={vc.max()}")

    print(f"Done! Cached to {cache_dir}/")

    import json
    config_path = cache_dir / "precompute_config.json"
    json.dump(vars(args), open(config_path, "w"), indent=2)


if __name__ == "__main__":
    main()
