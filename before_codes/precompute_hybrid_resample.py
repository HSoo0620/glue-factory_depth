"""
기존 FPFH 캐시(fpfh_resample_cache_r{radius})를 읽고,
동일한 keypoint 위치에서 SP descriptor(256D)를 추출하여 Hybrid 캐시 생성.

FPFH 재계산 없이 SP forward만 수행하므로 빠름.

사용법:
    python precompute_hybrid_resample.py
    python precompute_hybrid_resample.py --fpfh_radius 0.5 --image_size 2880
    python precompute_hybrid_resample.py --force

출력:
    gluefactory/datasets/mitsubishi/hybrid_resample_cache_r{radius}/{image_stem}.npz
"""

import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def load_superpoint(conf):
    from gluefactory.models.extractors.superpoint_open import SuperPoint
    model = SuperPoint(conf)
    model.eval()
    return model



def main():
    parser = argparse.ArgumentParser(description="기존 FPFH 캐시 + SP descriptor → Hybrid 캐시 생성")
    parser.add_argument("--max_num_keypoints", type=int, default=512)
    parser.add_argument("--detection_threshold", type=float, default=0.001)
    parser.add_argument("--nms_radius", type=int, default=3)
    parser.add_argument("--fpfh_radius", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=2880)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_dir = Path("gluefactory/datasets/mitsubishi")
    img_dir = base_dir / "dataset_resample"
    fpfh_cache_dir = base_dir / f"fpfh_resample_cache_r{args.fpfh_radius}"
    hybrid_cache_dir = base_dir / f"hybrid_resample_cache_r{args.fpfh_radius}"
    hybrid_cache_dir.mkdir(exist_ok=True)

    if not fpfh_cache_dir.exists():
        raise FileNotFoundError(
            f"FPFH cache not found: {fpfh_cache_dir}\n"
            f"Run: python precompute_fpfh_resample.py --fpfh_radius {args.fpfh_radius}"
        )

    # SP 모델 로드 (dense descriptor 출력 활성화)
    sp_conf = {
        "descriptor_dim": 256,
        "nms_radius": args.nms_radius,
        "max_num_keypoints": args.max_num_keypoints * 2,
        "force_num_keypoints": False,
        "detection_threshold": args.detection_threshold,
        "remove_borders": 4,
        "channels": [64, 64, 128, 128, 256],
        "dense_outputs": True,  # dense descriptor map 출력
        "weights": None,
    }

    device = args.device if torch.cuda.is_available() else "cpu"
    sp_model = load_superpoint(sp_conf).to(device)

    # grid_sample 유틸
    from gluefactory.models.extractors.superpoint_open import sample_descriptors

    fpfh_files = sorted(fpfh_cache_dir.glob("*.npz"))
    fpfh_files = [f for f in fpfh_files if f.stem.startswith("depth_raw_")]
    print(f"Found {len(fpfh_files)} FPFH cache files in {fpfh_cache_dir}")
    print(f"Output: {hybrid_cache_dir}/")

    stats = {"total": 0, "reused": 0}

    for fpfh_path in tqdm(fpfh_files, desc="Building Hybrid cache"):
        out_path = hybrid_cache_dir / fpfh_path.name
        if out_path.exists() and not args.force:
            stats["reused"] += 1
            continue

        # 기존 FPFH 캐시 로드
        fpfh_data = np.load(fpfh_path)
        keypoints = fpfh_data["keypoints"]          # (512, 2)
        scores = fpfh_data["keypoint_scores"]        # (512,)
        fpfh_desc = fpfh_data["fpfh_descriptors"]    # (512, 33)
        n_valid = int(fpfh_data["n_valid"])

        # 이미지 로드 → SP dense descriptor 계산
        img_path = img_dir / f"{fpfh_path.stem}.png"
        if not img_path.exists():
            print(f"  SKIP {fpfh_path.name}: image not found")
            continue

        img_raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        orig_h, orig_w = img_raw.shape[:2]

        if args.image_size != orig_h:
            img_resized = cv2.resize(
                img_raw, (args.image_size, args.image_size),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            img_resized = img_raw

        depth_norm = img_resized.astype(np.float32) / 65535.0
        h, w = depth_norm.shape[:2]

        img_tensor = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            sp_pred = sp_model({"image": img_tensor, "image_size": torch.tensor([[h, w]])})

        dense_desc = sp_pred["dense_descriptors"]  # (1, 256, H/8, W/8)

        # 캐시된 keypoint 위치에서 SP descriptor 샘플링
        # 주의: 캐시 keypoints는 SP forward 시 +0.5가 이미 적용됨
        # sample_descriptors 내부에서 다시 +0.5하므로, 여기서 -0.5 보정
        kp_for_sample = keypoints[:n_valid] - 0.5
        kp_tensor = torch.from_numpy(kp_for_sample).unsqueeze(0).float().to(device)
        sp_desc_sampled = sample_descriptors(kp_tensor, dense_desc, s=8)  # (1, 256, n_valid)
        sp_desc_sampled = sp_desc_sampled[0].T.cpu().numpy()  # (n_valid, 256)

        # 전체 배열 (zero-padded)
        sp_descriptors = np.zeros((keypoints.shape[0], 256), dtype=np.float32)
        sp_descriptors[:n_valid] = sp_desc_sampled

        # Hybrid 캐시 저장
        np.savez_compressed(
            out_path,
            keypoints=keypoints,
            keypoint_scores=scores,
            sp_descriptors=sp_descriptors,   # (512, 256) — NEW
            fpfh_descriptors=fpfh_desc,      # (512, 33)  — from existing cache
            n_valid=np.array(n_valid),
        )

        stats["total"] += 1

    print(f"\n--- Stats ---")
    print(f"Newly processed: {stats['total']} images")
    print(f"Skipped (already cached): {stats['reused']} images")
    print(f"Done! Cached to {hybrid_cache_dir}/")

    import json
    config_path = hybrid_cache_dir / "precompute_config.json"
    json.dump(vars(args), open(config_path, "w"), indent=2)


if __name__ == "__main__":
    main()
