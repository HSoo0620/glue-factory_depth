"""
모든 depth 이미지에 대해 SuperPoint keypoints + FPFH descriptors를 미리 계산.
배경(depth=0) keypoint는 제거하고, 부족한 분은 zero-padding.

사용법:
    python precompute_fpfh.py
    python precompute_fpfh.py --max_num_keypoints 512 --detection_threshold 0.003
    python precompute_fpfh.py --fpfh_radius 10.0 --fpfh_max_nn 100
    python precompute_fpfh.py --force  # 기존 캐시 덮어쓰기

출력:
    gluefactory/datasets/mitsubishi/fpfh_cache_r{radius}/{image_stem}.npz
    각 .npz 파일에는 keypoints, keypoint_scores, fpfh_descriptors, n_valid 가 저장됨
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
    """calib_XXXX.ini에서 K(intrinsic), clip_start, clip_end 파싱"""
    config = configparser.ConfigParser()
    config.read(str(ini_path))
    section = config.sections()[0]

    # k_matrix=Matrix:3:3:v0,v1,...,v8
    k_vals = [float(v) for v in config[section]["k_matrix"].split(":")[3].split(",")]
    K = np.array(k_vals, dtype=np.float64).reshape(3, 3)

    clip_start = float(config[section]["clip_start"])
    clip_end = float(config[section]["clip_end"])

    return K, clip_start, clip_end


def load_superpoint(conf):
    """SuperPoint 모델 로드 (detector만 사용)"""
    from gluefactory.models.extractors.superpoint_open import SuperPoint
    model = SuperPoint(conf)
    model.eval()
    return model


def filter_keypoints_by_depth(keypoints, scores, depth_map, max_num_keypoints):
    """
    depth > 0 인 keypoint만 남기고, score 순으로 정렬 후
    max_num_keypoints까지 zero-padding.

    Returns:
        keypoints: (max_num_keypoints, 2) - 유효 keypoints + zero padding
        scores: (max_num_keypoints,) - 유효 scores + zero padding
        n_valid: int - 실제 유효 keypoint 수
    """
    N = keypoints.shape[0]
    valid_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
        x = np.clip(x, 0, depth_map.shape[1] - 1)
        y = np.clip(y, 0, depth_map.shape[0] - 1)
        valid_mask[i] = depth_map[y, x] > 0

    # depth > 0 인 keypoint만 추출
    valid_kp = keypoints[valid_mask]
    valid_sc = scores[valid_mask]

    # score 내림차순 정렬
    if len(valid_sc) > 0:
        order = np.argsort(-valid_sc)
        valid_kp = valid_kp[order]
        valid_sc = valid_sc[order]

    # max_num_keypoints로 자르기 / 패딩
    n_valid = min(len(valid_kp), max_num_keypoints)

    out_kp = np.zeros((max_num_keypoints, 2), dtype=np.float32)
    out_sc = np.zeros(max_num_keypoints, dtype=np.float32)

    out_kp[:n_valid] = valid_kp[:n_valid]
    out_sc[:n_valid] = valid_sc[:n_valid]

    return out_kp, out_sc, n_valid


def compute_fpfh_for_keypoints(depth_map_real, keypoints, n_valid, K,
                               fpfh_radius=10.0, fpfh_max_nn=100):
    """
    실제 depth + K(intrinsic)로 3D 역투영 후 FPFH descriptors (33-dim) 계산.
    n_valid 이후의 keypoint(=zero padding)는 zero descriptor.

    Args:
        depth_map_real: 실제 depth 값 (clip 적용된 미터/mm 단위)
        K: 3x3 카메라 intrinsic matrix
    """
    max_n = keypoints.shape[0]
    fpfh_out = np.zeros((max_n, 33), dtype=np.float32)

    if n_valid < 3:
        return fpfh_out

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 유효 keypoint → 3D 역투영
    valid_kp = keypoints[:n_valid]
    points_3d = []
    for i in range(n_valid):
        u, v = int(valid_kp[i, 0]), int(valid_kp[i, 1])
        u = np.clip(u, 0, depth_map_real.shape[1] - 1)
        v = np.clip(v, 0, depth_map_real.shape[0] - 1)
        z = float(depth_map_real[v, u])
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points_3d.append([x, y, z])

    points_3d = np.array(points_3d, dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius * 2, max_nn=30)
    )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=fpfh_max_nn),
    )
    fpfh_np = np.array(fpfh.data).T.astype(np.float32)  # (n_valid, 33)
    # L2 normalize
    norms = np.linalg.norm(fpfh_np, axis=1, keepdims=True)
    fpfh_np = fpfh_np / (norms + 1e-8)
    fpfh_out[:n_valid] = fpfh_np

    return fpfh_out


def main():
    parser = argparse.ArgumentParser(description="SuperPoint keypoints + FPFH precomputation")
    parser.add_argument("--max_num_keypoints", type=int, default=512)
    parser.add_argument("--detection_threshold", type=float, default=0.003)
    parser.add_argument("--nms_radius", type=int, default=3)
    parser.add_argument("--fpfh_radius", type=float, default=10.0)
    parser.add_argument("--fpfh_max_nn", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true", help="기존 캐시 덮어쓰기")
    args = parser.parse_args()

    base_dir = Path("gluefactory/datasets/mitsubishi")
    img_dir = base_dir / "dataset_depth"
    cache_dir = base_dir / f"fpfh_cache_r{args.fpfh_radius}"
    cache_dir.mkdir(exist_ok=True)

    # SuperPoint config - force_num_keypoints 끔 (직접 필터링)
    # max_num_keypoints는 넉넉하게 뽑고, 이후 depth 필터링
    sp_conf = {
        "descriptor_dim": 256,
        "nms_radius": args.nms_radius,
        "max_num_keypoints": args.max_num_keypoints * 2,  # 넉넉하게 뽑고 필터링
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
    print(f"Found {len(image_paths)} depth images")
    print(f"Config: max_kp={args.max_num_keypoints}, det_th={args.detection_threshold}, "
          f"fpfh_radius={args.fpfh_radius}, fpfh_max_nn={args.fpfh_max_nn}")

    stats = {"total": 0, "valid_counts": []}

    for img_path in tqdm(image_paths, desc="Precomputing FPFH"):
        out_path = cache_dir / f"{img_path.stem}.npz"
        if out_path.exists() and not args.force:
            continue

        # ini 파일에서 카메라 정보 로드
        img_idx = img_path.stem.replace("depth_raw_", "")
        ini_path = img_path.parent / f"calib_{img_idx}.ini"
        if not ini_path.exists():
            print(f"  SKIP {img_path.name}: calib_{img_idx}.ini not found")
            continue

        K, clip_start, clip_end = parse_calib_ini(ini_path)

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        # 정규화 [0,1] (SuperPoint 입력용)
        depth_norm = img / 65535.0
        # 실제 depth 복원 (3D 역투영용)
        depth_real = clip_start + depth_norm * (clip_end - clip_start)
        # 배경(원본 0)은 그대로 0 유지
        depth_real[img == 0] = 0.0

        h, w = depth_norm.shape[:2]

        # SuperPoint keypoint 추출 (force_num_keypoints=False)
        img_tensor = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            sp_pred = sp_model({"image": img_tensor, "image_size": torch.tensor([[h, w]])})

        raw_kp = sp_pred["keypoints"][0].cpu().numpy()
        raw_sc = sp_pred["keypoint_scores"][0].cpu().numpy()

        # depth > 0 필터링 + zero padding (원본 이미지 기준으로 배경 판별)
        keypoints, scores, n_valid = filter_keypoints_by_depth(
            raw_kp, raw_sc, img, args.max_num_keypoints
        )

        # FPFH 계산 (실제 depth + K로 3D 역투영)
        fpfh = compute_fpfh_for_keypoints(
            depth_real, keypoints, n_valid, K,
            fpfh_radius=args.fpfh_radius,
            fpfh_max_nn=args.fpfh_max_nn,
        )

        # 저장
        np.savez_compressed(
            out_path,
            keypoints=keypoints,          # (max_num_keypoints, 2)
            keypoint_scores=scores,       # (max_num_keypoints,)
            fpfh_descriptors=fpfh,        # (max_num_keypoints, 33)
            n_valid=np.array(n_valid),    # 유효 keypoint 수
        )

        stats["total"] += 1
        stats["valid_counts"].append(n_valid)

    if stats["valid_counts"]:
        vc = np.array(stats["valid_counts"])
        print(f"\n--- Stats ---")
        print(f"Processed: {stats['total']} images")
        print(f"Valid keypoints per image: "
              f"mean={vc.mean():.1f}, min={vc.min()}, max={vc.max()}, "
              f"median={np.median(vc):.0f}")
        print(f"Images with <50 valid kp: {(vc < 50).sum()}")

    print(f"Done! Cached to {cache_dir}/")

    import json
    config_path = cache_dir / "precompute_config.json"
    json.dump(vars(args), open(config_path, "w"), indent=2)
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
