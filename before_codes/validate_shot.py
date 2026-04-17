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
        faces.astype(np.int64),
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
