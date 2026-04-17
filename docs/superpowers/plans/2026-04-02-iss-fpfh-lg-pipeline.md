# ISS + FPFH + LightGlue Pipeline (0402) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** resample_2 depth 데이터에서 ISS(3D detector) + FPFH(3D descriptor) + LightGlue(matcher) 파이프라인을 구현하고, ISS 기반 GT(`outputs_txt`)로 학습한다.

**Architecture:** 기존 SP+FPFH+LG(0331) 파이프라인과 동일 구조. detector를 SP→ISS로 교체하고, GT를 `outputs_resample_2`→`outputs_txt`로 변경. Precompute 캐시 방식으로 ISS keypoints + FPFH descriptors를 사전 계산하여 .npz로 저장. 학습/추론 시 캐시를 로드하여 LightGlue에 전달.

**Tech Stack:** Open3D (ISS, FPFH), PyTorch, LightGlue, NumPy, pandas

---

### Task 1: Precompute 스크립트 작성 (`precompute_iss_fpfh_resample2.py`)

**Files:**
- Create: `precompute_iss_fpfh_resample2.py`

기존 `precompute_fpfh_resample2.py`를 기반으로 SP keypoint 추출 부분을 ISS로 교체.

- [ ] **Step 1: 파일 생성**

`precompute_iss_fpfh_resample2.py`를 다음 내용으로 생성:

```python
"""
Resample_2 depth 이미지에 대해 ISS keypoints + Dense FPFH 계산.
카메라 intrinsic 없이 (u, v, depth_real)을 직접 3D 좌표로 사용.
고정 crop (1129, 1081, 3502x3502) 적용 후 ISS keypoint 추출.

사용법:
    python precompute_iss_fpfh_resample2.py
    python precompute_iss_fpfh_resample2.py --fpfh_radius 5.0 --image_size 1751
    python precompute_iss_fpfh_resample2.py --force

출력:
    gluefactory/datasets/mitsubishi/iss_fpfh_resample2_cache_r{radius}/{image_stem}.npz
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


def depth_crop_to_pcd(depth_crop_raw):
    """crop된 depth에서 (u, v, depth_real) point cloud 생성."""
    vs, us = np.where(depth_crop_raw > 0)
    d_raw = depth_crop_raw[vs, us].astype(np.float64)
    depth_real = CLIP_START + (d_raw / 65535.0) * (CLIP_END - CLIP_START)
    pts = np.stack([us.astype(np.float64), vs.astype(np.float64), depth_real], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd, pts


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


def select_keypoints(iss_kp_3d, depth_crop_raw, max_num_keypoints, image_size):
    """ISS keypoints를 max_num 개로 제한/보충하고, resized 좌표로 변환.

    Returns:
        keypoints: (max_num, 2) — resized 좌표
        keypoint_scores: (max_num,) — ISS는 score 없으므로 1.0 / 0.0
        n_valid: 유효 keypoint 수
        kp_crop: (max_num, 2) — crop 좌표 (FPFH lookup용)
    """
    scale = image_size / CROP_SIZE
    n_iss = len(iss_kp_3d)

    if n_iss >= max_num_keypoints:
        # 랜덤으로 max_num 개 선택
        indices = np.random.choice(n_iss, max_num_keypoints, replace=False)
        selected = iss_kp_3d[indices]
        n_valid = max_num_keypoints
    else:
        # ISS keypoints 전부 + random 보충
        selected = iss_kp_3d.copy()
        n_need = max_num_keypoints - n_iss

        # depth > 0인 위치에서 random sampling
        ys, xs = np.where(depth_crop_raw > 0)
        rand_indices = np.random.choice(len(xs), min(n_need, len(xs)), replace=False)
        rand_u = xs[rand_indices].astype(np.float64)
        rand_v = ys[rand_indices].astype(np.float64)
        rand_d_raw = depth_crop_raw[ys[rand_indices], xs[rand_indices]].astype(np.float64)
        rand_depth = CLIP_START + (rand_d_raw / 65535.0) * (CLIP_END - CLIP_START)
        rand_pts = np.stack([rand_u, rand_v, rand_depth], axis=1)

        selected = np.vstack([selected, rand_pts])
        n_valid = n_iss  # ISS keypoints만 valid로 표시

    # crop 좌표 (u, v) — FPFH lookup용
    kp_crop = np.zeros((max_num_keypoints, 2), dtype=np.float32)
    kp_crop[:len(selected), 0] = selected[:len(selected), 0]  # u
    kp_crop[:len(selected), 1] = selected[:len(selected), 1]  # v

    # resized 좌표
    keypoints = np.zeros((max_num_keypoints, 2), dtype=np.float32)
    keypoints[:len(selected)] = kp_crop[:len(selected)] * scale

    # scores: ISS keypoints = 1.0, random = 0.5, padding = 0.0
    keypoint_scores = np.zeros(max_num_keypoints, dtype=np.float32)
    keypoint_scores[:n_valid] = 1.0
    if n_iss < max_num_keypoints:
        keypoint_scores[n_valid:len(selected)] = 0.5

    total_valid = len(selected)

    return keypoints, keypoint_scores, total_valid, kp_crop


def compute_fpfh_for_keypoints(pcd, kp_crop, n_valid,
                               fpfh_radius=5.0, fpfh_max_nn=100):
    """Dense FPFH에서 keypoint 위치의 descriptor를 KDTree lookup."""
    max_n = kp_crop.shape[0]
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

    # KDTree로 keypoint 위치에서 lookup
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for i in range(n_valid):
        u, v = kp_crop[i, 0], kp_crop[i, 1]
        # pcd의 좌표는 (u, v, depth_real) — crop 좌표계 기준
        # keypoint의 3D 좌표를 직접 사용
        query = [float(u), float(v), 0.0]  # depth는 가장 가까운 점으로 검색
        _, idx, _ = kdtree.search_knn_vector_3d(query, 1)
        # 실제 depth를 사용해서 다시 검색
        nearest_pt = np.asarray(pcd.points)[idx[0]]
        _, idx2, _ = kdtree.search_knn_vector_3d(nearest_pt, 1)
        fpfh_out[i] = fpfh_dense[idx2[0]]

    # L2 normalize
    norms = np.linalg.norm(fpfh_out[:n_valid], axis=1, keepdims=True)
    fpfh_out[:n_valid] = fpfh_out[:n_valid] / (norms + 1e-8)

    return fpfh_out


def main():
    parser = argparse.ArgumentParser(description="Resample_2용 ISS + Dense FPFH precomputation")
    parser.add_argument("--max_num_keypoints", type=int, default=512)
    parser.add_argument("--gamma_21", type=float, default=0.5)
    parser.add_argument("--gamma_32", type=float, default=0.5)
    parser.add_argument("--min_neighbors", type=int, default=5)
    parser.add_argument("--fpfh_radius", type=float, default=5.0)
    parser.add_argument("--fpfh_max_nn", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=1751)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_dir = Path("gluefactory/datasets/mitsubishi")
    img_dir = base_dir / "dataset_resample_2"
    cache_dir = base_dir / f"iss_fpfh_resample2_cache_r{args.fpfh_radius}"
    cache_dir.mkdir(exist_ok=True)

    image_paths = sorted(img_dir.glob("depth_raw_*.png"))
    print(f"Found {len(image_paths)} resample_2 depth images")
    print(f"Config: max_kp={args.max_num_keypoints}, gamma_21={args.gamma_21}, "
          f"gamma_32={args.gamma_32}, fpfh_radius={args.fpfh_radius}, "
          f"image_size={args.image_size}")
    print(f"Crop: ({CROP_X0}, {CROP_Y0}), size={CROP_SIZE}")

    stats = {"total": 0, "iss_counts": [], "valid_counts": []}

    for img_path in tqdm(image_paths, desc="Precomputing ISS+FPFH (resample_2)"):
        out_path = cache_dir / f"{img_path.stem}.npz"
        if out_path.exists() and not args.force:
            continue

        # 원본 이미지 로드 + 고정 crop
        img_raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        img_crop = img_raw[CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]

        # point cloud 생성 (crop 좌표계)
        pcd, dense_pts = depth_crop_to_pcd(img_crop)

        # ISS keypoint 검출
        iss_kp_3d = extract_iss_keypoints(
            pcd,
            gamma_21=args.gamma_21,
            gamma_32=args.gamma_32,
            min_neighbors=args.min_neighbors,
        )

        n_iss = len(iss_kp_3d)

        # keypoint 선택 (max_num 제한/보충) + 좌표 변환
        keypoints, scores, n_valid, kp_crop = select_keypoints(
            iss_kp_3d, img_crop, args.max_num_keypoints, args.image_size
        )

        # FPFH 계산 (Dense FPFH → keypoint lookup)
        fpfh = compute_fpfh_for_keypoints(
            pcd, kp_crop, n_valid,
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
    json.dump(vars(args), open(config_path, "w"), indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 단일 이미지 테스트**

Run: `conda run -n LightGlue python3 precompute_iss_fpfh_resample2.py --force 2>&1 | head -20`

첫 몇 장에서 에러 없이 실행되는지 확인. 예상 출력:
```
Found 641 resample_2 depth images
Config: max_kp=512, gamma_21=0.5, gamma_32=0.5, fpfh_radius=5.0, image_size=1751
Crop: (1129, 1081), size=3502
Precomputing ISS+FPFH (resample_2):   0%| ...
```

- [ ] **Step 3: 캐시 파일 검증**

```bash
conda run -n LightGlue python3 -c "
import numpy as np
d = np.load('gluefactory/datasets/mitsubishi/iss_fpfh_resample2_cache_r5.0/depth_raw_0000.npz')
print('keys:', list(d.keys()))
print('keypoints:', d['keypoints'].shape, d['keypoints'].dtype)
print('scores:', d['keypoint_scores'].shape)
print('fpfh:', d['fpfh_descriptors'].shape)
print('n_valid:', d['n_valid'], 'n_iss:', d['n_iss'])
print('keypoints sample:', d['keypoints'][:3])
print('fpfh nonzero:', (d['fpfh_descriptors'].sum(axis=1) != 0).sum())
"
```

예상: keypoints=(512,2), scores=(512,), fpfh=(512,33), n_valid<=512

- [ ] **Step 4: 커밋**

```bash
git add precompute_iss_fpfh_resample2.py
git commit -m "feat: add ISS+FPFH precompute script for resample_2"
```

---

### Task 2: Dataset 클래스 작성 (`mitsubishi_resample2_iss_fpfh_dataset.py`)

**Files:**
- Create: `gluefactory/datasets/mitsubishi_resample2_iss_fpfh_dataset.py`

기존 `mitsubishi_resample2_fpfh_dataset.py`를 기반으로 GT를 `outputs_txt`로 변경.

- [ ] **Step 1: 파일 생성**

`gluefactory/datasets/mitsubishi_resample2_iss_fpfh_dataset.py`를 다음 내용으로 생성:

```python
"""
Resample_2 이미지 기반 ISS+FPFH 캐시 데이터셋.
precompute_iss_fpfh_resample2.py로 캐시 생성 후 사용.
GT: outputs_txt (ISS 기반 matching 결과).
고정 crop (1129, 1081, 3502x3502) + 선택적 리사이즈.
"""

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import cv2


# 고정 crop 파라미터
CROP_X0 = 1129
CROP_Y0 = 1081
CROP_SIZE = 3502


class MitsubishiResample2ISSFPFHDataset(Dataset):

    def __init__(self, split="train", cache_dir=None, fpfh_radius=5.0,
                 image_size=None, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path("./gluefactory/datasets/mitsubishi")
        self.image_dir = self.base_dir / "dataset_resample_2"
        self.output_dir = self.base_dir / "outputs_txt"
        self.image_size = image_size

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self.base_dir / f"iss_fpfh_resample2_cache_r{fpfh_radius}"

        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"ISS+FPFH cache not found: {self.cache_dir}\n"
                f"Run: python precompute_iss_fpfh_resample2.py --fpfh_radius {fpfh_radius}"
            )

        combo = pd.read_csv(self.output_dir / "combination.csv")

        total = len(combo)
        val_size = 100
        test_size = 100
        train_end = total - val_size - test_size
        val_end = total - test_size

        if split == "train":
            combo = combo.iloc[:train_end]
        elif split == "val":
            combo = combo.iloc[train_end:val_end]
        elif split == "test":
            combo = combo.iloc[val_end:]
        else:
            raise ValueError(split)

        self.combo = combo.reset_index(drop=True)

    def _extract_filename(self, path_str):
        """combination.csv의 경로에서 파일명만 추출"""
        return Path(path_str).name

    def _load_cache(self, img_path_str):
        stem = Path(img_path_str).stem
        npz_path = self.cache_dir / f"{stem}.npz"
        data = np.load(npz_path)
        return {
            "keypoints": data["keypoints"],
            "keypoint_scores": data["keypoint_scores"],
            "descriptors": data["fpfh_descriptors"],
        }

    def __len__(self):
        return len(self.combo)

    def __getitem__(self, idx):
        row = self.combo.iloc[idx]

        # 경로 변환 (상대경로에서 파일명 추출)
        master_fname = self._extract_filename(row["master_path"])
        input_fname = self._extract_filename(row["input_path"])
        master_path = self.image_dir / master_fname
        input_path = self.image_dir / input_fname

        # csv_path 변환 (outputs_txt/00000.csv → 로컬 경로)
        csv_fname = Path(row["csv_path"]).name
        csv_path = self.output_dir / csv_fname

        # 이미지 로드 + 고정 crop
        master_raw = cv2.imread(str(master_path), cv2.IMREAD_UNCHANGED)
        input_raw = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)

        master_crop = master_raw[CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]
        input_crop = input_raw[CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]

        # 리사이즈 (옵션)
        if self.image_size is not None and self.image_size != CROP_SIZE:
            master_crop = cv2.resize(
                master_crop, (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST
            )
            input_crop = cv2.resize(
                input_crop, (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST
            )
            output_size = self.image_size
        else:
            output_size = CROP_SIZE

        scale = output_size / CROP_SIZE

        master = master_crop.astype(np.float32) / 65535.0
        input_img = input_crop.astype(np.float32) / 65535.0
        h, w = master.shape[:2]

        # GT 로드 + crop offset + 스케일링
        corr = pd.read_csv(csv_path)
        valid = corr["occluded"] == False
        master_xy = corr.loc[valid, ["master_x", "master_y"]].values.astype(np.float32)
        input_xy = corr.loc[valid, ["input_x", "input_y"]].values.astype(np.float32)

        # crop offset 적용
        master_xy[:, 0] -= CROP_X0
        master_xy[:, 1] -= CROP_Y0
        input_xy[:, 0] -= CROP_X0
        input_xy[:, 1] -= CROP_Y0

        # 리사이즈 스케일링
        if scale != 1.0:
            master_xy *= scale
            input_xy *= scale

        # crop 범위 밖 GT 제거
        in_bounds = (
            (master_xy[:, 0] >= 0) & (master_xy[:, 0] < output_size) &
            (master_xy[:, 1] >= 0) & (master_xy[:, 1] < output_size) &
            (input_xy[:, 0] >= 0) & (input_xy[:, 0] < output_size) &
            (input_xy[:, 1] >= 0) & (input_xy[:, 1] < output_size)
        )
        master_xy = master_xy[in_bounds]
        input_xy = input_xy[in_bounds]
        gt_matches = np.concatenate([master_xy, input_xy], axis=1)

        # ISS+FPFH 캐시 로드
        cache0 = self._load_cache(row["master_path"])
        cache1 = self._load_cache(row["input_path"])

        data = {
            "view0": {
                "image": torch.from_numpy(master).unsqueeze(0),
                "image_size": torch.tensor([h, w]),
                "keypoints": torch.from_numpy(cache0["keypoints"]).float(),
                "keypoint_scores": torch.from_numpy(cache0["keypoint_scores"]).float(),
                "descriptors": torch.from_numpy(cache0["descriptors"]).float(),
            },
            "view1": {
                "image": torch.from_numpy(input_img).unsqueeze(0),
                "image_size": torch.tensor([h, w]),
                "keypoints": torch.from_numpy(cache1["keypoints"]).float(),
                "keypoint_scores": torch.from_numpy(cache1["keypoint_scores"]).float(),
                "descriptors": torch.from_numpy(cache1["descriptors"]).float(),
            },
            "gt_matches": torch.from_numpy(gt_matches).float(),
            "csv_path": str(csv_path),
            "master_path": str(master_path),
            "input_path": str(input_path),
        }

        return data


def resample2_iss_fpfh_collate_fn(batch):
    images0 = torch.stack([b["view0"]["image"] for b in batch], dim=0)
    images1 = torch.stack([b["view1"]["image"] for b in batch], dim=0)

    sizes0 = torch.stack([b["view0"]["image_size"] for b in batch], dim=0)
    sizes1 = torch.stack([b["view1"]["image_size"] for b in batch], dim=0)

    kp0 = torch.stack([b["view0"]["keypoints"] for b in batch], dim=0)
    kp1 = torch.stack([b["view1"]["keypoints"] for b in batch], dim=0)
    sc0 = torch.stack([b["view0"]["keypoint_scores"] for b in batch], dim=0)
    sc1 = torch.stack([b["view1"]["keypoint_scores"] for b in batch], dim=0)
    desc0 = torch.stack([b["view0"]["descriptors"] for b in batch], dim=0)
    desc1 = torch.stack([b["view1"]["descriptors"] for b in batch], dim=0)

    gt_list = [b["gt_matches"] for b in batch]
    gt_matches = pad_sequence(gt_list, batch_first=True, padding_value=0.0)

    return {
        "view0": {
            "image": images0,
            "image_size": sizes0,
            "keypoints": kp0,
            "keypoint_scores": sc0,
            "descriptors": desc0,
        },
        "view1": {
            "image": images1,
            "image_size": sizes1,
            "keypoints": kp1,
            "keypoint_scores": sc1,
            "descriptors": desc1,
        },
        "gt_matches": gt_matches,
        "csv_path": [b["csv_path"] for b in batch],
        "master_path": [b["master_path"] for b in batch],
        "input_path": [b["input_path"] for b in batch],
    }
```

- [ ] **Step 2: 데이터셋 로드 테스트**

```bash
conda run -n LightGlue python3 -c "
from gluefactory.datasets.mitsubishi_resample2_iss_fpfh_dataset import *
ds = MitsubishiResample2ISSFPFHDataset(split='train', image_size=1751)
print(f'Train size: {len(ds)}')
sample = ds[0]
print(f'view0 kp: {sample[\"view0\"][\"keypoints\"].shape}')
print(f'view0 desc: {sample[\"view0\"][\"descriptors\"].shape}')
print(f'gt_matches: {sample[\"gt_matches\"].shape}')
print(f'csv_path: {sample[\"csv_path\"]}')
"
```

예상: Train size=63800, kp=(512,2), desc=(512,33), gt_matches=(N,4) where N~4180

- [ ] **Step 3: 커밋**

```bash
git add gluefactory/datasets/mitsubishi_resample2_iss_fpfh_dataset.py
git commit -m "feat: add ISS+FPFH dataset class with outputs_txt GT"
```

---

### Task 3: Config 파일 작성 (`0402_resample2_iss_fpfh_lg.yaml`)

**Files:**
- Create: `gluefactory/configs/0402_resample2_iss_fpfh_lg.yaml`

- [ ] **Step 1: 파일 생성**

`gluefactory/configs/0402_resample2_iss_fpfh_lg.yaml`를 다음 내용으로 생성:

```yaml
data:
    name: resample2_iss_fpfh
    fpfh_radius: 5.0
    image_size: 1751  # 3502 / 2
    batch_size: 32    # 캐시 방식 → 큰 배치 가능
    num_workers: 4
model:
    name: two_view_pipeline
    filter_zero_depth: true
    extractor:
        name: extractors.superpoint_fpfh_cached
        descriptor_dim: 33
        trainable: False
    ground_truth:
        name: matchers.gt_pair_matcher
        gt_radius: 6
    matcher:
        name: matchers.lightglue
        input_dim: 33
        descriptor_dim: 36
        num_heads: 3
        filter_threshold: 0.1
        flash: false
        checkpointed: true
train:
    seed: 0
    epochs: 100
    best_key: match_recall
    best_key_mode: max
    log_every_iter: 1
    eval_every_iter: 500
    lr: 1e-4
    lr_schedule:
        start: 20
        type: exp
        on_epoch: true
        exp_div_10: 10
    plot: [5, 'gluefactory.visualization.visualize_batch.make_match_figures_depth']
```

- [ ] **Step 2: 커밋**

```bash
git add gluefactory/configs/0402_resample2_iss_fpfh_lg.yaml
git commit -m "feat: add config for ISS+FPFH+LG pipeline (0402)"
```

---

### Task 4: 학습 모듈 작성 (`train_resample2_iss_fpfh.py`)

**Files:**
- Create: `gluefactory/train_resample2_iss_fpfh.py`

기존 `gluefactory/train_resample2_fpfh.py`를 복사하고, 데이터셋/collate를 ISS 버전으로 교체.

- [ ] **Step 1: 파일 복사 및 수정**

`gluefactory/train_resample2_fpfh.py`를 `gluefactory/train_resample2_iss_fpfh.py`로 복사한 뒤, 다음 2줄을 변경:

변경 전 (line 38-39):
```python
from .datasets.mitsubishi_resample2_fpfh_dataset import MitsubishiResample2FPFHDataset
from .datasets.mitsubishi_resample2_fpfh_dataset import resample2_fpfh_collate_fn
```

변경 후:
```python
from .datasets.mitsubishi_resample2_iss_fpfh_dataset import MitsubishiResample2ISSFPFHDataset
from .datasets.mitsubishi_resample2_iss_fpfh_dataset import resample2_iss_fpfh_collate_fn
```

그리고 dataset 생성 부분 (line 290-291):

변경 전:
```python
    dataset = MitsubishiResample2FPFHDataset(split="train", fpfh_radius=fpfh_radius, image_size=image_size)
    val_dataset = MitsubishiResample2FPFHDataset(split="val", fpfh_radius=fpfh_radius, image_size=image_size)
```

변경 후:
```python
    dataset = MitsubishiResample2ISSFPFHDataset(split="train", fpfh_radius=fpfh_radius, image_size=image_size)
    val_dataset = MitsubishiResample2ISSFPFHDataset(split="val", fpfh_radius=fpfh_radius, image_size=image_size)
```

그리고 collate_fn (line 298):

변경 전:
```python
        collate_fn=resample2_fpfh_collate_fn,
```

변경 후:
```python
        collate_fn=resample2_iss_fpfh_collate_fn,
```

같은 변경을 val_loader (line 305)에도 적용:

변경 전:
```python
        collate_fn=resample2_fpfh_collate_fn,
```

변경 후:
```python
        collate_fn=resample2_iss_fpfh_collate_fn,
```

파일 상단 docstring도 변경:

```python
"""
Resample_2 ISS+FPFH 데이터용 학습 스크립트.
ISS(detector) + FPFH(descriptor) + LightGlue.
"""
```

- [ ] **Step 2: 커밋**

```bash
git add gluefactory/train_resample2_iss_fpfh.py
git commit -m "feat: add ISS+FPFH training module for resample_2"
```

---

### Task 5: 학습 셸 스크립트 작성 (`train_resample2_iss_fpfh_0402.sh`)

**Files:**
- Create: `train_resample2_iss_fpfh_0402.sh`

- [ ] **Step 1: 파일 생성**

`train_resample2_iss_fpfh_0402.sh`를 다음 내용으로 생성:

```bash
#!/bin/bash
# ISS(detector) + FPFH(descriptor) + LG 학습 (resample_2 데이터)
# ISS+FPFH 캐시 없으면 자동 생성

GPU_ID=${1:-0}
EXPERIMENT=${2:-"0402_resample2_iss_fpfh_lg"}
FPFH_RADIUS=${3:-5.0}
IMAGE_SIZE=${4:-1751}
GT_RADIUS=${5:-6}
BATCH_SIZE=${6:-32}
RESTORE=${7:-""}
CONF="gluefactory/configs/0402_resample2_iss_fpfh_lg.yaml"

RESTORE_FLAG=""
if [ "$RESTORE" = "--restore" ]; then
    RESTORE_FLAG="--restore"
    echo "=== RESUME training from last checkpoint ==="
fi

# ISS+FPFH 캐시 확인 및 생성
CACHE_DIR="gluefactory/datasets/mitsubishi/iss_fpfh_resample2_cache_r${FPFH_RADIUS}"
if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A $CACHE_DIR/*.npz 2>/dev/null)" ]; then
    echo "=== ISS+FPFH cache not found. Precomputing... ==="
    python3 precompute_iss_fpfh_resample2.py \
        --fpfh_radius "$FPFH_RADIUS" \
        --image_size "$IMAGE_SIZE"
fi

# 출력 디렉토리 생성
mkdir -p "outputs/training/${EXPERIMENT}"

echo "=== Training ISS(det)+FPFH(desc)+LG on resample_2 (fpfh_r=${FPFH_RADIUS}, image_size=${IMAGE_SIZE}, gt_radius=${GT_RADIUS}, batch=${BATCH_SIZE}) ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m gluefactory.train_resample2_iss_fpfh "$EXPERIMENT" \
    --conf "$CONF" \
    $RESTORE_FLAG \
    --mixed_precision float16 \
    data.fpfh_radius="$FPFH_RADIUS" \
    data.image_size="$IMAGE_SIZE" \
    data.batch_size="$BATCH_SIZE" \
    model.ground_truth.gt_radius="$GT_RADIUS" \
    2>&1 | tee "outputs/training/${EXPERIMENT}/train.log"
```

- [ ] **Step 2: 실행 권한 부여**

```bash
chmod +x train_resample2_iss_fpfh_0402.sh
```

- [ ] **Step 3: 커밋**

```bash
git add train_resample2_iss_fpfh_0402.sh
git commit -m "feat: add ISS+FPFH training shell script (0402)"
```

---

### Task 6: 테스트 스크립트 작성 (`test_resample2_iss_fpfh_0402.py`)

**Files:**
- Create: `test_resample2_iss_fpfh_0402.py`

기존 `test_resample2_fpfh_0331.py`를 기반으로 데이터셋을 ISS 버전으로 교체.

- [ ] **Step 1: 파일 생성**

`test_resample2_iss_fpfh_0402.py`를 다음 내용으로 생성:

```python
"""
학습된 ISS(det)+FPFH(desc)+LG 모델로 resample_2 데이터 매칭 테스트.

사용법:
    python test_resample2_iss_fpfh_0402.py --checkpoint outputs/training/0402_resample2_iss_fpfh_lg/checkpoint_best.tar --indices 0 10 50 90
    python test_resample2_iss_fpfh_0402.py --experiment 0402_resample2_iss_fpfh_lg --split test --num_samples 10
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.datasets.mitsubishi_resample2_iss_fpfh_dataset import (
    MitsubishiResample2ISSFPFHDataset,
    resample2_iss_fpfh_collate_fn,
    CROP_X0,
    CROP_Y0,
    CROP_SIZE,
)
from gluefactory.utils.tensor import batch_to_device


def load_model(checkpoint_path, device):
    cp = torch.load(checkpoint_path, map_location="cpu")
    conf = OmegaConf.create(cp["conf"])
    model = get_model(conf.model.name)(conf.model).to(device)
    model.load_state_dict(cp["model"], strict=False)
    model.eval()
    epoch = cp["epoch"]
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {epoch})")
    return model, conf


@torch.no_grad()
def run_inference(model, batch, device):
    batch = batch_to_device(batch, device)
    pred = model(batch)
    return pred, batch


def visualize_pair(pred, data, idx, output_path, csv_path=None, gt_radius=6,
                   image_size=1751):
    img0 = data["view0"]["image"][idx].cpu()
    img1 = data["view1"]["image"][idx].cpu()

    kp0 = pred["keypoints0"][idx].cpu().numpy()
    kp1 = pred["keypoints1"][idx].cpu().numpy()
    m0 = pred["matches0"][idx].cpu().numpy()

    valid = m0 > -1
    mkp0 = kp0[valid]
    mkp1 = kp1[m0[valid]]
    n_total = int(valid.sum())

    n_correct, n_wrong, n_occluded, n_no_gt = 0, 0, 0, 0
    gt_pos_total = 0
    colors = ["red"] * n_total

    if csv_path is not None and Path(csv_path).exists():
        corr = pd.read_csv(csv_path)
        scale = image_size / CROP_SIZE

        all_master_xy = corr[["master_x", "master_y"]].values.astype(np.float32)
        all_input_xy = corr[["input_x", "input_y"]].values.astype(np.float32)
        all_occluded = corr["occluded"].values.astype(bool)

        all_master_xy[:, 0] -= CROP_X0
        all_master_xy[:, 1] -= CROP_Y0
        all_input_xy[:, 0] -= CROP_X0
        all_input_xy[:, 1] -= CROP_Y0

        all_master_xy *= scale
        all_input_xy *= scale

        gt_pos_total = int((~all_occluded).sum())

        for i in range(n_total):
            kp0_pt = mkp0[i]
            kp1_pt = mkp1[i]

            dists_master = np.linalg.norm(all_master_xy - kp0_pt, axis=1)
            nearest_idx = np.argmin(dists_master)
            nearest_dist = dists_master[nearest_idx]

            if nearest_dist < gt_radius:
                gt_input_pt = all_input_xy[nearest_idx]
                is_occluded = all_occluded[nearest_idx]
                match_dist = np.linalg.norm(kp1_pt - gt_input_pt)

                if match_dist < gt_radius:
                    if is_occluded:
                        colors[i] = "limegreen"
                        n_occluded += 1
                    else:
                        colors[i] = "skyblue"
                        n_correct += 1
                else:
                    colors[i] = "purple"
                    n_wrong += 1
            else:
                colors[i] = "red"
                n_no_gt += 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

    for ax, img, kp, title in [
        (axes[0], img0, kp0, "View 0 (Master)"),
        (axes[1], img1, kp1, "View 1 (Input)"),
    ]:
        img_np = img.squeeze(0).numpy()
        ax.imshow(img_np, cmap="gray")
        ax.scatter(kp[:, 0], kp[:, 1], c="royalblue", s=3, alpha=0.3, linewidths=0)
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()

    for i in range(n_total):
        line = matplotlib.patches.ConnectionPatch(
            xyA=(float(mkp0[i, 0]), float(mkp0[i, 1])),
            xyB=(float(mkp1[i, 0]), float(mkp1[i, 1])),
            coordsA=axes[0].transData,
            coordsB=axes[1].transData,
            axesA=axes[0],
            axesB=axes[1],
            color=colors[i],
            linewidth=0.8,
            alpha=0.6,
        )
        fig.add_artist(line)

    info = f"Keypoints: {kp0.shape[0]} / {kp1.shape[0]}  |  Total matches: {n_total}"
    if csv_path is not None:
        recall = n_correct / max(gt_pos_total, 1) * 100
        info += (f"  |  Recall(cyan): {n_correct}/{gt_pos_total} ({recall:.1f}%)"
                 f"  |  Wrong(purple): {n_wrong}"
                 f"  |  Occluded(green): {n_occluded}"
                 f"  |  No-GT(red): {n_no_gt}")
    fig.suptitle(info, fontsize=10, y=0.02)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close(fig)
    recall = n_correct / max(gt_pos_total, 1) * 100 if gt_pos_total > 0 else 0.0
    print(f"  Saved: {output_path}  |  matches={n_total}"
          f"  recall={n_correct}/{gt_pos_total} ({recall:.1f}%)"
          f"  wrong={n_wrong}  occluded={n_occluded}  no-GT={n_no_gt}")


def main():
    parser = argparse.ArgumentParser(description="학습된 ISS+FPFH+LG로 resample_2 매칭 테스트")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="0402_resample2_iss_fpfh_lg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--image_size", type=int, default=1751)
    parser.add_argument("--fpfh_radius", type=float, default=5.0)
    parser.add_argument("--gt_radius", type=int, default=6)
    args = parser.parse_args()

    output_dir = Path(args.output_dir if args.output_dir else f"results/{args.experiment}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.checkpoint is None:
        from gluefactory.utils.experiments import get_best_checkpoint
        cp_path = get_best_checkpoint(args.experiment)
        print(f"Auto-loaded best checkpoint: {cp_path}")
    else:
        cp_path = args.checkpoint

    model, conf = load_model(cp_path, device)
    image_size = args.image_size

    dataset = MitsubishiResample2ISSFPFHDataset(
        split=args.split, fpfh_radius=args.fpfh_radius, image_size=image_size
    )
    print(f"{args.split} dataset: {len(dataset)} pairs "
          f"(crop={CROP_SIZE}, image_size={image_size}, fpfh_r={args.fpfh_radius})")

    if args.indices is not None:
        indices = args.indices
    else:
        indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
        indices = sorted(indices)

    print(f"Testing {len(indices)} pairs: {indices}")

    for idx in indices:
        sample = dataset[idx]
        batch = resample2_iss_fpfh_collate_fn([sample])
        pred, batch = run_inference(model, batch, device)
        csv_path = batch.get("csv_path", [None])[0]
        output_path = output_dir / f"{args.split}_{idx:05d}.png"
        visualize_pair(pred, batch, 0, output_path, csv_path=csv_path,
                       gt_radius=args.gt_radius, image_size=image_size)

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 커밋**

```bash
git add test_resample2_iss_fpfh_0402.py
git commit -m "feat: add ISS+FPFH test/visualization script (0402)"
```

---

### Task 7: 전체 통합 테스트

- [ ] **Step 1: 캐시 생성 (전체)**

```bash
conda run -n LightGlue python3 precompute_iss_fpfh_resample2.py --fpfh_radius 5.0 --image_size 1751
```

예상: 641개 이미지 처리, `iss_fpfh_resample2_cache_r5.0/` 디렉토리에 .npz 파일 생성

- [ ] **Step 2: 데이터셋 로드 테스트**

```bash
conda run -n LightGlue python3 -c "
from gluefactory.datasets.mitsubishi_resample2_iss_fpfh_dataset import *
for split in ['train', 'val', 'test']:
    ds = MitsubishiResample2ISSFPFHDataset(split=split, image_size=1751)
    sample = ds[0]
    print(f'{split}: {len(ds)} pairs, gt_matches={sample[\"gt_matches\"].shape}')
"
```

예상: train=63800, val=100, test=100

- [ ] **Step 3: 학습 시작 테스트 (1 iteration)**

```bash
bash train_resample2_iss_fpfh_0402.sh 0
```

학습이 시작되고 첫 iteration의 loss가 출력되는지 확인 후 Ctrl+C로 중단.

- [ ] **Step 4: 커밋 (모든 파일)**

```bash
git add -A
git commit -m "feat: complete ISS+FPFH+LG pipeline (0402) for resample_2"
```
