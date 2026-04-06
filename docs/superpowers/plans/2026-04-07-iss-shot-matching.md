# ISS + SHOT Feature Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FPFH(33D) descriptor를 SHOT(336D)으로 교체하는 ISS+SHOT+LightGlue 파이프라인 구현. 학습 전 discriminability 검증 필수.

**Architecture:** validate_shot.py로 SHOT 품질 검증(통과 시) → precompute_iss_shot_resample2.py로 캐시 생성 → 기존 ISS+FPFH 학습 구조를 그대로 재사용해 학습. 기존 파일은 수정하지 않음.

**Tech Stack:** Python, pyshot (`pyshot.get_descriptors`), Open3D (ISS detection), scipy.spatial.cKDTree, numpy (vectorized grid mesh), PyTorch/LightGlue (학습), conda env: `LightGlue`

---

## 파일 구조

```
신규 생성 (6개):
  validate_shot.py                                        — SHOT discriminability 검증
  precompute_iss_shot_resample2.py                        — ISS keypoint + SHOT 캐시 생성
  gluefactory/configs/0407_resample2_iss_shot_lg.yaml     — 학습 설정 (input_dim=336)
  gluefactory/datasets/mitsubishi_resample2_iss_shot_dataset.py  — 데이터셋
  gluefactory/train_resample2_iss_shot.py                 — 학습 루프
  train_resample2_iss_shot_0407.sh                        — 학습 실행 스크립트

기존 파일: 수정 없음
캐시 경로: gluefactory/datasets/mitsubishi/iss_shot_resample2_cache_r{radius}/
```

**pyshot 핵심 파라미터 (336D):**
```python
pyshot.get_descriptors(
    verts,           # (N, 3) float64, XYZ in mm
    faces,           # (M, 3) int32, triangle indices
    radius=10.0,
    local_rf_radius=10.0,
    n_bins=20,
    double_volumes_sectors=False,  # ← 반드시 False! True이면 672D
    use_normalization=True,
)  # → (N, 336)
```

---

## Task 1: validate_shot.py

**Files:**
- Create: `validate_shot.py`

- [ ] **Step 1: validate_shot.py 작성**

```python
"""
SHOT descriptor discriminability 검증.
GT 대응점 쌍의 positive cosine sim vs random pair sim 비교.
positive sim >> random sim 이어야 학습 가능.

사용법:
    conda activate LightGlue
    python validate_shot.py                                    # 기본값
    python validate_shot.py --n_pairs 3 --mesh_subsample 16   # 빠른 스모크 테스트
    python validate_shot.py --n_pairs 10 --mesh_subsample 8   # 표준 검증
"""

import argparse
import time
import numpy as np
import cv2
import pyshot
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

CROP_X0 = 1129
CROP_Y0 = 1081
CROP_SIZE = 3502
CLIP_START = 0.1
CLIP_END = 1000.0
FX = FY = 8001.39
CX_CROP = 2880.5 - CROP_X0   # 1751.5
CY_CROP = 2880.5 - CROP_Y0   # 1799.5


def build_grid_mesh(depth_raw, fx, fy, cx, cy, mask=None):
    """Depth image → XYZ grid mesh (verts, faces) for pyshot.
    서브샘플링은 호출 전 외부에서 적용 (depth_raw[::s,::s], intrinsics /= s).

    Args:
        depth_raw: (H, W) uint16 depth image (이미 서브샘플링 적용된 상태)
        fx, fy, cx, cy: intrinsics (서브샘플 기준으로 조정된 값)
        mask: (H, W) uint8 optional erosion mask
    Returns:
        verts: (N, 3) float64 XYZ in mm
        faces: (M, 3) int32 triangle indices
        pixel_to_vertex: (H, W) int32, -1 if invalid
    """
    H, W = depth_raw.shape
    if mask is None:
        valid = (depth_raw > 0)
    else:
        valid = (depth_raw > 0) & (mask > 0)

    pixel_to_vertex = np.full((H, W), -1, dtype=np.int32)
    ys, xs = np.where(valid)
    pixel_to_vertex[ys, xs] = np.arange(len(ys), dtype=np.int32)

    d_raw = depth_raw[ys, xs].astype(np.float64)
    depth_real = CLIP_START + (d_raw / 65535.0) * (CLIP_END - CLIP_START)
    X = (xs.astype(np.float64) - cx) * depth_real / fx
    Y = (ys.astype(np.float64) - cy) * depth_real / fy
    Z = depth_real
    verts = np.stack([X, Y, Z], axis=1)

    # Vectorized 2×2 block → 2 triangles
    vg, ug = np.mgrid[0:H-1, 0:W-1]
    vg, ug = vg.ravel(), ug.ravel()
    i00 = pixel_to_vertex[vg,     ug    ]
    i10 = pixel_to_vertex[vg,     ug + 1]
    i01 = pixel_to_vertex[vg + 1, ug    ]
    i11 = pixel_to_vertex[vg + 1, ug + 1]

    m1 = (i00 >= 0) & (i10 >= 0) & (i01 >= 0)
    m2 = (i10 >= 0) & (i11 >= 0) & (i01 >= 0)
    tri1 = np.stack([i00[m1], i10[m1], i01[m1]], axis=1)
    tri2 = np.stack([i10[m2], i11[m2], i01[m2]], axis=1)
    faces = np.concatenate([tri1, tri2], axis=0).astype(np.int32) if (m1.any() or m2.any()) \
        else np.zeros((0, 3), dtype=np.int32)

    return verts, faces, pixel_to_vertex


def compute_shot_all(verts, faces, shot_radius, n_bins=20):
    """SHOT descriptors for all mesh vertices. Returns (N, 336) float32."""
    return pyshot.get_descriptors(
        verts.astype(np.float64),
        faces.astype(np.int32),
        radius=float(shot_radius),
        local_rf_radius=float(shot_radius),
        n_bins=int(n_bins),
        double_volumes_sectors=False,   # ← 336D (True이면 672D)
        use_normalization=True,
        min_neighbors=3,
    ).astype(np.float32)


def process_pair(master_path, input_path, csv_path, shot_radius, n_bins, subsample):
    """한 이미지 쌍의 positive/random cosine sim 반환."""
    # 이미지 로드 + crop + subsample
    master_raw = cv2.imread(str(master_path), cv2.IMREAD_UNCHANGED)
    input_raw  = cv2.imread(str(input_path),  cv2.IMREAD_UNCHANGED)
    master_crop = master_raw[CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]
    input_crop  = input_raw [CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]
    if subsample > 1:
        master_crop = master_crop[::subsample, ::subsample]
        input_crop  = input_crop [::subsample, ::subsample]

    # Intrinsics adjusted for subsample
    fx_s = FX / subsample; fy_s = FY / subsample
    cx_s = CX_CROP / subsample; cy_s = CY_CROP / subsample

    # Grid mesh
    verts0, faces0, p2v0 = build_grid_mesh(master_crop, fx_s, fy_s, cx_s, cy_s)
    verts1, faces1, p2v1 = build_grid_mesh(input_crop,  fx_s, fy_s, cx_s, cy_s)

    if len(verts0) < 10 or len(verts1) < 10 or len(faces0) == 0 or len(faces1) == 0:
        print("  [SKIP] too few vertices or faces"); return None, None

    # SHOT for all vertices
    t0 = time.time()
    shot0 = compute_shot_all(verts0, faces0, shot_radius, n_bins)
    shot1 = compute_shot_all(verts1, faces1, shot_radius, n_bins)
    print(f"  SHOT computed: {len(verts0)} + {len(verts1)} verts in {time.time()-t0:.1f}s")

    # GT correspondences → subsampled crop 공간으로 변환
    corr = pd.read_csv(csv_path)
    gt_valid = corr["occluded"] == False
    mxy = corr.loc[gt_valid, ["master_x", "master_y"]].values.astype(np.float32)
    ixy = corr.loc[gt_valid, ["input_x",  "input_y" ]].values.astype(np.float32)
    mxy[:, 0] = (mxy[:, 0] - CROP_X0) / subsample
    mxy[:, 1] = (mxy[:, 1] - CROP_Y0) / subsample
    ixy[:, 0] = (ixy[:, 0] - CROP_X0) / subsample
    ixy[:, 1] = (ixy[:, 1] - CROP_Y0) / subsample

    H_s, W_s = master_crop.shape
    in_b = ((mxy[:,0] >= 0) & (mxy[:,0] < W_s) & (mxy[:,1] >= 0) & (mxy[:,1] < H_s) &
            (ixy[:,0] >= 0) & (ixy[:,0] < W_s) & (ixy[:,1] >= 0) & (ixy[:,1] < H_s))
    mxy, ixy = mxy[in_b], ixy[in_b]

    if len(mxy) < 10:
        print(f"  [SKIP] only {len(mxy)} GT pairs in-bounds"); return None, None

    # Positive sim: GT 위치에서 descriptor lookup → cosine sim
    pos_sims = []
    for i in range(len(mxy)):
        u0, v0 = int(round(mxy[i,0])), int(round(mxy[i,1]))
        u1, v1 = int(round(ixy[i,0])), int(round(ixy[i,1]))
        u0 = min(max(u0, 0), W_s-1); v0 = min(max(v0, 0), H_s-1)
        u1 = min(max(u1, 0), W_s-1); v1 = min(max(v1, 0), H_s-1)
        idx0 = p2v0[v0, u0]; idx1 = p2v1[v1, u1]
        if idx0 < 0 or idx1 < 0: continue
        d0, d1 = shot0[idx0], shot1[idx1]
        sim = float(np.dot(d0, d1) / (np.linalg.norm(d0) * np.linalg.norm(d1) + 1e-8))
        pos_sims.append(sim)

    if not pos_sims:
        print("  [SKIP] no valid positive pairs"); return None, None

    # Random sim
    rng = np.random.default_rng(42)
    n_rand = len(pos_sims)
    r0 = rng.integers(0, len(verts0), n_rand)
    r1 = rng.integers(0, len(verts1), n_rand)
    d0r = shot0[r0]; d1r = shot1[r1]
    n0 = np.linalg.norm(d0r, axis=1, keepdims=True) + 1e-8
    n1 = np.linalg.norm(d1r, axis=1, keepdims=True) + 1e-8
    rand_sims = ((d0r / n0) * (d1r / n1)).sum(axis=1)

    print(f"  positive pairs: {len(pos_sims)}, positive sim: {np.mean(pos_sims):.4f} ± {np.std(pos_sims):.4f}")
    print(f"  random sim:     {rand_sims.mean():.4f} ± {rand_sims.std():.4f}")
    return np.array(pos_sims), rand_sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs",       type=int,   default=10)
    parser.add_argument("--shot_radius",   type=float, default=10.0)
    parser.add_argument("--n_bins",        type=int,   default=20)
    parser.add_argument("--mesh_subsample",type=int,   default=8,
                        help="Grid subsampling stride. 8=fast/rough, 4=balanced, 1=full")
    args = parser.parse_args()

    base_dir   = Path("gluefactory/datasets/mitsubishi")
    output_dir = base_dir / "outputs_txt"
    image_dir  = base_dir / "dataset_resample_2"
    combo = pd.read_csv(output_dir / "combination.csv")
    pairs = combo.sample(n=min(args.n_pairs, len(combo)), random_state=42).reset_index(drop=True)

    print(f"=== SHOT Discriminability Validation ===")
    print(f"radius={args.shot_radius}mm, n_bins={args.n_bins} → {16*(args.n_bins+1)}D, subsample={args.mesh_subsample}")
    print(f"Pairs: {len(pairs)}")

    all_pos, all_rand = [], []
    for i, row in pairs.iterrows():
        master_path = image_dir / Path(row["master_path"]).name
        input_path  = image_dir / Path(row["input_path"]).name
        csv_path    = output_dir / Path(row["csv_path"]).name
        print(f"\n[{i+1}/{len(pairs)}] {master_path.name} ↔ {input_path.name}")
        pos, rand = process_pair(master_path, input_path, csv_path,
                                  args.shot_radius, args.n_bins, args.mesh_subsample)
        if pos is not None:
            all_pos.append(pos); all_rand.append(rand)

    if not all_pos:
        print("\n[FAIL] No valid pairs processed."); return

    all_pos  = np.concatenate(all_pos)
    all_rand = np.concatenate(all_rand)
    gap = all_pos.mean() - all_rand.mean()

    print(f"\n{'='*45}")
    print(f"RESULT over {len(pairs)} pairs:")
    print(f"  positive sim : {all_pos.mean():.4f} ± {all_pos.std():.4f}")
    print(f"  random sim   : {all_rand.mean():.4f} ± {all_rand.std():.4f}")
    print(f"  gap          : {gap:+.4f}")
    if gap > 0.05:
        print(f"  → PASS: SHOT is discriminative. Proceed to precompute.")
    else:
        print(f"  → FAIL: gap too small. Adjust radius or n_bins.")
    print('='*45)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 스모크 테스트 (1쌍, subsample=16, 빠름)**

```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda activate LightGlue
python validate_shot.py --n_pairs 1 --mesh_subsample 16
```

Expected output (약 30초 이내):
```
=== SHOT Discriminability Validation ===
radius=10.0mm, n_bins=20 → 336D, subsample=16
...
  SHOT computed: XXXXX + XXXXX verts in X.Xs
  positive pairs: NNN, positive sim: X.XXXX ± X.XXXX
  random sim:     X.XXXX ± X.XXXX
```
에러 없이 완료되면 PASS.

- [ ] **Step 3: 표준 검증 실행 (10쌍)**

```bash
python validate_shot.py --n_pairs 10 --mesh_subsample 8
```

Expected:
- `gap > 0.05` → "PASS: SHOT is discriminative." 확인
- gap ≤ 0.05이면 `--shot_radius 20.0` 또는 `--mesh_subsample 4`로 재시도

- [ ] **Step 4: 커밋**

```bash
git add validate_shot.py
git commit -m "feat: add SHOT discriminability validation script"
```

---

## Task 2: precompute_iss_shot_resample2.py

**Files:**
- Create: `precompute_iss_shot_resample2.py`

ISS 검출 함수들은 `precompute_iss_fpfh_resample2.py`에서 그대로 복사. FPFH 관련만 교체.

- [ ] **Step 1: precompute_iss_shot_resample2.py 작성**

```python
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
        faces: (M, 3) int32 triangle indices
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
    faces = np.concatenate([tri1, tri2], axis=0).astype(np.int32) \
        if (m1.any() or m2.any()) else np.zeros((0, 3), dtype=np.int32)

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
        faces.astype(np.int32),
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

        # FPFH lookup용 XYZ 변환
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
```

- [ ] **Step 2: 스모크 테스트 (2개 이미지)**

```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda activate LightGlue
python precompute_iss_shot_resample2.py --max_images 2 --force
```

Expected:
```
Found 2 images
Config: shot_radius=10.0mm, n_bins=20 → 336D
        mesh_subsample=4, image_size=1751
100%|████| 2/2 [XX:XX]
ISS keypoints: mean=XXXX.X ...
Done! → gluefactory/datasets/mitsubishi/iss_shot_resample2_cache_r10.0/
```

- [ ] **Step 3: 캐시 구조 검증**

```bash
python3 -c "
import numpy as np
from pathlib import Path
f = sorted(Path('gluefactory/datasets/mitsubishi/iss_shot_resample2_cache_r10.0').glob('*.npz'))[0]
d = np.load(f)
print('keypoints:       ', d['keypoints'].shape)       # (512, 2)
print('keypoint_scores: ', d['keypoint_scores'].shape) # (512,)
print('shot_descriptors:', d['shot_descriptors'].shape) # (512, 336)
print('n_valid:         ', int(d['n_valid']))
print('n_iss:           ', int(d['n_iss']))
assert d['shot_descriptors'].shape == (512, 336), 'WRONG SHAPE'
print('Shape OK')
"
```

Expected:
```
keypoints:        (512, 2)
keypoint_scores:  (512,)
shot_descriptors: (512, 336)
n_valid:          512
n_iss:            XXXX
Shape OK
```

- [ ] **Step 4: 전체 precompute 실행**

```bash
python precompute_iss_shot_resample2.py --shot_radius 10.0
```

완료 후 `.npz` 파일이 641개 생성되었는지 확인:
```bash
ls gluefactory/datasets/mitsubishi/iss_shot_resample2_cache_r10.0/*.npz | wc -l
# Expected: 641
```

- [ ] **Step 5: 커밋**

```bash
git add precompute_iss_shot_resample2.py
git commit -m "feat: add ISS+SHOT precompute script (grid mesh, 336D)"
```

---

## Task 3: Config 파일 + Dataset

**Files:**
- Create: `gluefactory/configs/0407_resample2_iss_shot_lg.yaml`
- Create: `gluefactory/datasets/mitsubishi_resample2_iss_shot_dataset.py`

- [ ] **Step 1: Config 작성**

파일: `gluefactory/configs/0407_resample2_iss_shot_lg.yaml`

```yaml
data:
    name: resample2_iss_shot
    shot_radius: 10.0
    image_size: 1751
    batch_size: 16       # 36D→336D로 모델 커짐, OOM 시 8로 줄일 것
    num_workers: 12

model:
    name: two_view_pipeline
    filter_zero_depth: true
    extractor:
        name: extractors.superpoint_fpfh_cached   # pass-through extractor 재사용
        descriptor_dim: 336
        trainable: False
    ground_truth:
        name: matchers.gt_pair_matcher
        gt_radius: 20
    matcher:
        name: matchers.lightglue
        input_dim: 336
        descriptor_dim: 336     # 줄이지 않음, input_proj = Identity
        num_heads: 4            # 336 / 4 = 84 (head_dim)
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

- [ ] **Step 2: Dataset 작성**

파일: `gluefactory/datasets/mitsubishi_resample2_iss_shot_dataset.py`

`mitsubishi_resample2_iss_fpfh_dataset.py`에서 다음만 변경:
- 클래스명: `MitsubishiResample2ISSFPFHDataset` → `MitsubishiResample2ISSSHOTDataset`
- `fpfh_radius` → `shot_radius`
- 캐시 경로: `iss_fpfh_resample2_cache_r{fpfh_radius}_xyz` → `iss_shot_resample2_cache_r{shot_radius}`
- `fpfh_descriptors` → `shot_descriptors`
- collate fn: `resample2_iss_fpfh_collate_fn` → `resample2_iss_shot_collate_fn`
- 에러 메시지: `Run: python precompute_iss_fpfh_resample2.py` → `Run: python precompute_iss_shot_resample2.py`

전체 코드:

```python
"""
Resample_2 이미지 기반 ISS+SHOT 캐시 데이터셋.
precompute_iss_shot_resample2.py로 캐시 생성 후 사용.
"""

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import cv2

CROP_X0 = 1129
CROP_Y0 = 1081
CROP_SIZE = 3502


class MitsubishiResample2ISSSHOTDataset(Dataset):

    def __init__(self, split="train", cache_dir=None, shot_radius=10.0,
                 image_size=None, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path("./gluefactory/datasets/mitsubishi")
        self.image_dir = self.base_dir / "dataset_resample_2"
        self.output_dir = self.base_dir / "outputs_txt"
        self.image_size = image_size

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self.base_dir / f"iss_shot_resample2_cache_r{shot_radius}"

        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"ISS+SHOT cache not found: {self.cache_dir}\n"
                f"Run: python precompute_iss_shot_resample2.py --shot_radius {shot_radius}"
            )

        combo = pd.read_csv(self.output_dir / "combination.csv")
        total = len(combo)
        val_size = 100; test_size = 100
        train_end = total - val_size - test_size
        val_end = total - test_size

        if split == "train":    combo = combo.iloc[:train_end]
        elif split == "val":    combo = combo.iloc[train_end:val_end]
        elif split == "test":   combo = combo.iloc[val_end:]
        else: raise ValueError(split)
        self.combo = combo.reset_index(drop=True)

    def _extract_filename(self, path_str):
        return Path(path_str).name

    def _load_cache(self, img_path_str):
        stem = Path(img_path_str).stem
        data = np.load(self.cache_dir / f"{stem}.npz")
        return {
            "keypoints":       data["keypoints"],
            "keypoint_scores": data["keypoint_scores"],
            "descriptors":     data["shot_descriptors"],   # ← shot_descriptors
        }

    def __len__(self):
        return len(self.combo)

    def __getitem__(self, idx):
        row = self.combo.iloc[idx]
        master_path = self.image_dir / self._extract_filename(row["master_path"])
        input_path  = self.image_dir / self._extract_filename(row["input_path"])
        csv_path    = self.output_dir / Path(row["csv_path"]).name

        master_raw = cv2.imread(str(master_path), cv2.IMREAD_UNCHANGED)
        input_raw  = cv2.imread(str(input_path),  cv2.IMREAD_UNCHANGED)
        master_crop = master_raw[CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]
        input_crop  = input_raw [CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]

        if self.image_size is not None and self.image_size != CROP_SIZE:
            master_crop = cv2.resize(master_crop, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            input_crop  = cv2.resize(input_crop,  (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            output_size = self.image_size
        else:
            output_size = CROP_SIZE

        scale = output_size / CROP_SIZE
        master = master_crop.astype(np.float32) / 65535.0
        input_img = input_crop.astype(np.float32) / 65535.0
        h, w = master.shape[:2]

        corr = pd.read_csv(csv_path)
        valid = corr["occluded"] == False
        master_xy = corr.loc[valid, ["master_x", "master_y"]].values.astype(np.float32)
        input_xy  = corr.loc[valid, ["input_x",  "input_y" ]].values.astype(np.float32)
        master_xy[:, 0] -= CROP_X0; master_xy[:, 1] -= CROP_Y0
        input_xy[:,  0] -= CROP_X0; input_xy[:,  1] -= CROP_Y0
        if scale != 1.0:
            master_xy *= scale; input_xy *= scale

        in_bounds = (
            (master_xy[:,0] >= 0) & (master_xy[:,0] < output_size) &
            (master_xy[:,1] >= 0) & (master_xy[:,1] < output_size) &
            (input_xy[:,0]  >= 0) & (input_xy[:,0]  < output_size) &
            (input_xy[:,1]  >= 0) & (input_xy[:,1]  < output_size)
        )
        master_xy = master_xy[in_bounds]; input_xy = input_xy[in_bounds]
        gt_matches = np.concatenate([master_xy, input_xy], axis=1)

        cache0 = self._load_cache(row["master_path"])
        cache1 = self._load_cache(row["input_path"])

        return {
            "view0": {
                "image":           torch.from_numpy(master).unsqueeze(0),
                "image_size":      torch.tensor([h, w]),
                "keypoints":       torch.from_numpy(cache0["keypoints"]).float(),
                "keypoint_scores": torch.from_numpy(cache0["keypoint_scores"]).float(),
                "descriptors":     torch.from_numpy(cache0["descriptors"]).float(),
            },
            "view1": {
                "image":           torch.from_numpy(input_img).unsqueeze(0),
                "image_size":      torch.tensor([h, w]),
                "keypoints":       torch.from_numpy(cache1["keypoints"]).float(),
                "keypoint_scores": torch.from_numpy(cache1["keypoint_scores"]).float(),
                "descriptors":     torch.from_numpy(cache1["descriptors"]).float(),
            },
            "gt_matches":   torch.from_numpy(gt_matches).float(),
            "csv_path":     str(csv_path),
            "master_path":  str(master_path),
            "input_path":   str(input_path),
        }


def resample2_iss_shot_collate_fn(batch):
    return {
        "view0": {
            "image":           torch.stack([b["view0"]["image"]           for b in batch]),
            "image_size":      torch.stack([b["view0"]["image_size"]      for b in batch]),
            "keypoints":       torch.stack([b["view0"]["keypoints"]       for b in batch]),
            "keypoint_scores": torch.stack([b["view0"]["keypoint_scores"] for b in batch]),
            "descriptors":     torch.stack([b["view0"]["descriptors"]     for b in batch]),
        },
        "view1": {
            "image":           torch.stack([b["view1"]["image"]           for b in batch]),
            "image_size":      torch.stack([b["view1"]["image_size"]      for b in batch]),
            "keypoints":       torch.stack([b["view1"]["keypoints"]       for b in batch]),
            "keypoint_scores": torch.stack([b["view1"]["keypoint_scores"] for b in batch]),
            "descriptors":     torch.stack([b["view1"]["descriptors"]     for b in batch]),
        },
        "gt_matches": pad_sequence([b["gt_matches"] for b in batch], batch_first=True, padding_value=0.0),
        "csv_path":    [b["csv_path"]   for b in batch],
        "master_path": [b["master_path"] for b in batch],
        "input_path":  [b["input_path"]  for b in batch],
    }
```

- [ ] **Step 3: Dataset 로딩 스모크 테스트**

```bash
conda activate LightGlue
cd /home/jhs/work/Registration/glue-factory_depth
python3 -c "
from gluefactory.datasets.mitsubishi_resample2_iss_shot_dataset import (
    MitsubishiResample2ISSSHOTDataset, resample2_iss_shot_collate_fn
)
from torch.utils.data import DataLoader

ds = MitsubishiResample2ISSSHOTDataset(split='train', shot_radius=10.0, image_size=1751)
print(f'Dataset size: {len(ds)}')
item = ds[0]
print('descriptors shape:', item['view0']['descriptors'].shape)  # (512, 336)
assert item['view0']['descriptors'].shape == (512, 336), 'WRONG'

loader = DataLoader(ds, batch_size=2, collate_fn=resample2_iss_shot_collate_fn)
batch = next(iter(loader))
print('batch descriptors:', batch['view0']['descriptors'].shape)  # (2, 512, 336)
print('OK')
"
```

Expected:
```
Dataset size: XXXX
descriptors shape: torch.Size([512, 336])
batch descriptors: torch.Size([2, 512, 336])
OK
```

- [ ] **Step 4: 커밋**

```bash
git add gluefactory/configs/0407_resample2_iss_shot_lg.yaml
git add gluefactory/datasets/mitsubishi_resample2_iss_shot_dataset.py
git commit -m "feat: add ISS+SHOT config and dataset (336D, gt_radius=20)"
```

---

## Task 4: 학습 루프 + 실행 스크립트

**Files:**
- Create: `gluefactory/train_resample2_iss_shot.py`
- Create: `train_resample2_iss_shot_0407.sh`

- [ ] **Step 1: 학습 루프 작성**

파일: `gluefactory/train_resample2_iss_shot.py`

`gluefactory/train_resample2_iss_fpfh.py`에서 4줄만 변경 (나머지 완전 동일):

```python
# ── 변경 1: import (line 36-37 교체) ──
from .datasets.mitsubishi_resample2_iss_shot_dataset import MitsubishiResample2ISSSHOTDataset
from .datasets.mitsubishi_resample2_iss_shot_dataset import resample2_iss_shot_collate_fn

# ── 변경 2: training() 함수 내 conf.data 읽기 (line 284 교체) ──
shot_radius = conf.data.get("shot_radius", 10.0)

# ── 변경 3: Dataset 인스턴스화 (line 288-289 교체) ──
dataset     = MitsubishiResample2ISSSHOTDataset(split="train", shot_radius=shot_radius, image_size=image_size)
val_dataset = MitsubishiResample2ISSSHOTDataset(split="val",   shot_radius=shot_radius, image_size=image_size)

# ── 변경 4: DataLoader collate_fn (line 296, 304 교체) ──
collate_fn=resample2_iss_shot_collate_fn,
```

전체 파일은 `gluefactory/train_resample2_iss_fpfh.py`를 복사하고 위 4개 변경 적용.

- [ ] **Step 2: 학습 스크립트 작성**

파일: `train_resample2_iss_shot_0407.sh`

```bash
#!/bin/bash
# ISS(detector) + SHOT(descriptor, 336D) + LightGlue 학습 (resample_2)
# 캐시 없으면 자동 생성

GPU_ID=${1:-3}
EXPERIMENT=${2:-"0407_resample2_iss_shot_lg"}
SHOT_RADIUS=${3:-10.0}
IMAGE_SIZE=${4:-1751}
GT_RADIUS=${5:-20}
BATCH_SIZE=${6:-16}
RESTORE=${7:-""}
CONF="gluefactory/configs/0407_resample2_iss_shot_lg.yaml"

RESTORE_FLAG=""
if [ "$RESTORE" = "--restore" ]; then
    RESTORE_FLAG="--restore"
    echo "=== RESUME training from last checkpoint ==="
fi

CACHE_DIR="gluefactory/datasets/mitsubishi/iss_shot_resample2_cache_r${SHOT_RADIUS}"
if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A $CACHE_DIR/*.npz 2>/dev/null)" ]; then
    echo "=== ISS+SHOT cache not found. Precomputing... ==="
    python3 precompute_iss_shot_resample2.py \
        --shot_radius "$SHOT_RADIUS" \
        --image_size  "$IMAGE_SIZE"
fi

mkdir -p "outputs/training/${EXPERIMENT}"

echo "=== Training ISS+SHOT+LG (shot_r=${SHOT_RADIUS}, img=${IMAGE_SIZE}, gt_r=${GT_RADIUS}, batch=${BATCH_SIZE}) ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m gluefactory.train_resample2_iss_shot "$EXPERIMENT" \
    --conf "$CONF" \
    $RESTORE_FLAG \
    --mixed_precision float16 \
    data.shot_radius="$SHOT_RADIUS" \
    data.image_size="$IMAGE_SIZE" \
    data.batch_size="$BATCH_SIZE" \
    model.ground_truth.gt_radius="$GT_RADIUS" \
    2>&1 | tee "outputs/training/${EXPERIMENT}/train.log"
```

- [ ] **Step 3: 학습 스모크 테스트 (1회 iteration 확인)**

```bash
conda activate LightGlue
cd /home/jhs/work/Registration/glue-factory_depth
CUDA_VISIBLE_DEVICES=3 python3 -m gluefactory.train_resample2_iss_shot \
    "smoke_test_shot" \
    --conf gluefactory/configs/0407_resample2_iss_shot_lg.yaml \
    data.shot_radius=10.0 \
    data.image_size=1751 \
    data.batch_size=2 \
    train.epochs=1 \
    train.eval_every_iter=999999 \
    train.log_every_iter=1 \
    2>&1 | head -30
```

Expected: 에러 없이 loss 출력:
```
[E 0 | it     0/XXXX] total=X.XXXX | nll_pos=X.XXXX | nll_neg=X.XXXX
```

OOM 발생 시: `data.batch_size=1`로 재시도.

- [ ] **Step 4: 커밋**

```bash
git add gluefactory/train_resample2_iss_shot.py
git add train_resample2_iss_shot_0407.sh
chmod +x train_resample2_iss_shot_0407.sh
git commit -m "feat: add ISS+SHOT training loop and script"
```

---

## Task 5: 학습 실행

**Files:** 없음 (Task 4 파일 실행)

- [ ] **Step 1: 학습 시작**

```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda activate LightGlue
bash train_resample2_iss_shot_0407.sh 3 0407_resample2_iss_shot_lg 10.0 1751 20 16
```

- [ ] **Step 2: 학습 진행 모니터링**

```bash
# 다른 터미널에서
tail -f /home/jhs/work/Registration/glue-factory_depth/outputs/training/0407_resample2_iss_shot_lg/train.log
```

E0에서 `match_recall > 0.0` 확인. 0.0이면 gt_radius 조정 필요.

- [ ] **Step 3: 결과 기록**

학습 완료 후 best checkpoint recall을 `docs/superpowers/specs/2026-04-07-iss-shot-matching-design.md`의 성능 섹션에 추가.
