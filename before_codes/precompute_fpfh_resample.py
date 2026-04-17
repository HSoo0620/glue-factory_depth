"""
Resample (5761×5761) depth 이미지에 대해 SuperPoint keypoints + Dense FPFH 계산.
이미지를 image_size로 리사이즈 후 SuperPoint 적용, 3D는 원본 해상도 기준 계산.

사용법:
    python precompute_fpfh_resample.py
    python precompute_fpfh_resample.py --fpfh_radius 1.5 --image_size 2880
    python precompute_fpfh_resample.py --force

출력:
    gluefactory/datasets/mitsubishi/fpfh_resample_cache_r{radius}/{image_stem}.npz
"""

import argparse
import torch
import numpy as np
import cv2
import open3d as o3d
import configparser
from pathlib import Path
from tqdm import tqdm


def parse_calib_ini(ini_path):
    config = configparser.ConfigParser()
    config.read(str(ini_path))
    section = config.sections()[0]

    k_vals = [float(v) for v in config[section]["k_matrix"].split(":")[3].split(",")]
    K = np.array(k_vals, dtype=np.float64).reshape(3, 3)

    clip_start = float(config[section]["clip_start"])
    clip_end = float(config[section]["clip_end"])

    return K, clip_start, clip_end


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


def compute_fpfh_for_keypoints(depth_map_real, keypoints, n_valid, K,
                               fpfh_radius=1.0, fpfh_max_nn=100):
    max_n = keypoints.shape[0]
    fpfh_out = np.zeros((max_n, 33), dtype=np.float32)

    if n_valid < 3:
        return fpfh_out

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Dense point cloud
    vs, us = np.where(depth_map_real > 0)
    zs = depth_map_real[vs, us]
    xs = (us - cx) * zs / fx
    ys = (vs - cy) * zs / fy
    dense_pts = np.stack([xs, ys, zs], axis=1).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dense_pts)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius * 2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=fpfh_max_nn),
    )
    fpfh_dense = np.array(fpfh.data).T.astype(np.float32)

    # keypoint → nearest neighbor lookup
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    valid_kp = keypoints[:n_valid]
    for i in range(n_valid):
        u, v = int(valid_kp[i, 0]), int(valid_kp[i, 1])
        u = np.clip(u, 0, depth_map_real.shape[1] - 1)
        v = np.clip(v, 0, depth_map_real.shape[0] - 1)
        z = float(depth_map_real[v, u])
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        _, idx, _ = kdtree.search_knn_vector_3d([x, y, z], 1)
        fpfh_out[i] = fpfh_dense[idx[0]]

    # L2 normalize
    norms = np.linalg.norm(fpfh_out[:n_valid], axis=1, keepdims=True)
    fpfh_out[:n_valid] = fpfh_out[:n_valid] / (norms + 1e-8)

    return fpfh_out


def main():
    parser = argparse.ArgumentParser(description="Resample 이미지용 SuperPoint + Dense FPFH precomputation")
    parser.add_argument("--max_num_keypoints", type=int, default=512)
    parser.add_argument("--detection_threshold", type=float, default=0.001)
    parser.add_argument("--nms_radius", type=int, default=3)
    parser.add_argument("--fpfh_radius", type=float, default=1.5)
    parser.add_argument("--fpfh_max_nn", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=2880)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_dir = Path("gluefactory/datasets/mitsubishi")
    img_dir = base_dir / "dataset_resample"
    cache_dir = base_dir / f"fpfh_resample_cache_r{args.fpfh_radius}"
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
    print(f"Found {len(image_paths)} resample depth images")
    print(f"Config: max_kp={args.max_num_keypoints}, det_th={args.detection_threshold}, "
          f"fpfh_radius={args.fpfh_radius}, image_size={args.image_size}")

    stats = {"total": 0, "valid_counts": []}

    for img_path in tqdm(image_paths, desc="Precomputing FPFH (resample)"):
        out_path = cache_dir / f"{img_path.stem}.npz"
        if out_path.exists() and not args.force:
            continue

        img_idx = img_path.stem.replace("depth_raw_", "")
        ini_path = img_path.parent / f"calib_{img_idx}.ini"
        if not ini_path.exists():
            print(f"  SKIP {img_path.name}: calib not found")
            continue

        K, clip_start, clip_end = parse_calib_ini(ini_path)

        # 원본 이미지 로드
        img_raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        orig_h, orig_w = img_raw.shape[:2]

        # 리사이즈
        if args.image_size != orig_h:
            img_resized = cv2.resize(
                img_raw, (args.image_size, args.image_size),
                interpolation=cv2.INTER_NEAREST
            )
            scale = orig_w / args.image_size  # resized → orig 스케일
        else:
            img_resized = img_raw
            scale = 1.0

        # SuperPoint 입력용 정규화
        depth_norm = img_resized.astype(np.float32) / 65535.0
        h, w = depth_norm.shape[:2]

        # SuperPoint keypoint 추출 (리사이즈된 이미지에서)
        img_tensor = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            sp_pred = sp_model({"image": img_tensor, "image_size": torch.tensor([[h, w]])})

        raw_kp = sp_pred["keypoints"][0].cpu().numpy()
        raw_sc = sp_pred["keypoint_scores"][0].cpu().numpy()

        # depth > 0 필터링 (리사이즈된 이미지 기준)
        keypoints, scores, n_valid = filter_keypoints_by_depth(
            raw_kp, raw_sc, img_resized, args.max_num_keypoints
        )

        # 3D FPFH: 원본 해상도 depth로 계산
        depth_real_orig = img_raw.astype(np.float64) / 65535.0
        depth_real_orig = clip_start + depth_real_orig * (clip_end - clip_start)
        depth_real_orig[img_raw == 0] = 0.0

        # keypoint 좌표를 원본 해상도로 스케일
        kp_orig = keypoints.copy()
        kp_orig[:n_valid] *= scale

        fpfh = compute_fpfh_for_keypoints(
            depth_real_orig, kp_orig, n_valid, K,
            fpfh_radius=args.fpfh_radius,
            fpfh_max_nn=args.fpfh_max_nn,
        )

        # 저장 (keypoints는 리사이즈된 이미지 기준 좌표)
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
