"""
C++ SHOT352 descriptor discriminability 검증.
precomputed output_shot_0407/ bins를 직접 읽어 GT 대응쌍 기반 검증.

Metrics:
  1. Positive vs Random cosine similarity (gap >= 0.05 목표)
  2. NN matching precision (3D 거리 기준, @5/10/20mm)
  3. Lowe's ratio test pass rate

C++ bin 특성:
  - full 5761x5761 이미지 기반 (crop 아님)
  - VoxelGrid voxelSize=1.0mm, shotRadius=10.0mm
  - 이미 L2 정규화 완료 (norm≈1.0) → dot product = cosine sim

사용법:
    conda activate LightGlue
    cd /home/jhs/work/Registration/glue-factory_depth
    python validate_shot352_cpp.py                   # 기본값 (10쌍)
    python validate_shot352_cpp.py --n_pairs 5       # 빠른 테스트
    python validate_shot352_cpp.py --n_pairs 20      # 더 많은 쌍
"""

import argparse
import time
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

# Dataset constants (원본 5761×5761 기준)
CLIP_START = 0.1
CLIP_END   = 1000.0
FX = FY    = 8001.389
CX = CY    = 2880.5

SHOT_DIM = 352


# ── Binary loader ─────────────────────────────────────────────────────────────

def load_shot352(bin_path):
    """C++ SHOT352 binary → points (N,3) float32, descriptors (N,352) float32."""
    with open(bin_path, 'rb') as f:
        N, D = np.frombuffer(f.read(8), dtype=np.uint32)
        N, D = int(N), int(D)
        assert D == SHOT_DIM, f"Expected D={SHOT_DIM}, got D={D}"
        raw = np.frombuffer(f.read(N * (3 + D) * 4), dtype=np.float32)
    data = raw.reshape(N, 3 + D)
    return data[:, :3], data[:, 3:]   # (N,3), (N,352)


# ── GT pair processing ────────────────────────────────────────────────────────

def read_depth_mm(depth_img, u, v):
    """원본 uint16 depth 이미지에서 (u,v) → depth in mm. 0 → None."""
    h, w = depth_img.shape
    u_c = int(round(u)); v_c = int(round(v))
    u_c = max(0, min(u_c, w - 1)); v_c = max(0, min(v_c, h - 1))
    d_raw = depth_img[v_c, u_c]
    if d_raw == 0:
        return None
    return CLIP_START + float(d_raw) / 65535.0 * (CLIP_END - CLIP_START)


def pixel_to_xyz(u, v, depth_img):
    """원본 이미지 픽셀 → XYZ in mm (camera coords). depth_img: uint16."""
    depth = read_depth_mm(depth_img, u, v)
    if depth is None:
        return None
    u_c = int(round(u)); v_c = int(round(v))
    return np.array([
        (u_c - CX) * depth / FX,
        (v_c - CY) * depth / FY,
        depth,
    ], dtype=np.float32)


def process_pair(master_path, input_path, csv_path, bin_dir, max_gt=500, rng_seed=42):
    """한 이미지 쌍 분석. None 반환 시 건너뜀."""
    master_stem = Path(master_path).stem
    input_stem  = Path(input_path).stem
    bin0 = bin_dir / f"{master_stem}_shot352.bin"
    bin1 = bin_dir / f"{input_stem}_shot352.bin"

    if not bin0.exists() or not bin1.exists():
        print(f"  [SKIP] bin not found")
        return None

    # ── Load C++ bins ──
    t0 = time.time()
    pts0, desc0 = load_shot352(bin0)
    pts1, desc1 = load_shot352(bin1)
    # Filter NaN
    valid0 = ~np.isnan(desc0).any(axis=1)
    valid1 = ~np.isnan(desc1).any(axis=1)
    pts0, desc0 = pts0[valid0], desc0[valid0]
    pts1, desc1 = pts1[valid1], desc1[valid1]
    print(f"  Loaded: {len(pts0)} + {len(pts1)} pts  [{time.time()-t0:.1f}s]")

    if len(pts0) < 100 or len(pts1) < 100:
        print("  [SKIP] too few valid points")
        return None

    # XYZ KDTree (for GT pixel → nearest C++ point)
    tree0 = cKDTree(pts0)
    tree1 = cKDTree(pts1)

    # ── Load images for depth lookup ──
    master_img = cv2.imread(str(master_path), cv2.IMREAD_UNCHANGED)
    input_img  = cv2.imread(str(input_path),  cv2.IMREAD_UNCHANGED)

    # ── GT correspondences ──
    corr = pd.read_csv(csv_path)
    gt = corr[corr["occluded"] == False].reset_index(drop=True)
    if len(gt) == 0:
        print("  [SKIP] no GT pairs")
        return None
    if len(gt) > max_gt:
        gt = gt.sample(n=max_gt, random_state=rng_seed).reset_index(drop=True)

    # ── Positive pairs ──
    pos_sims    = []
    gt_desc0    = []   # descriptor from master cloud at GT position
    gt_pts1     = []   # nearest point in input cloud at GT position
    gt_nn_dist0 = []   # 3D snap distance (GT pixel → nearest C++ pt, master)
    gt_nn_dist1 = []   # same for input

    for _, row in gt.iterrows():
        xyz0 = pixel_to_xyz(row["master_x"], row["master_y"], master_img)
        xyz1 = pixel_to_xyz(row["input_x"],  row["input_y"],  input_img)
        if xyz0 is None or xyz1 is None:
            continue
        d0, i0 = tree0.query(xyz0)
        d1, i1 = tree1.query(xyz1)
        da, db = desc0[i0], desc1[i1]
        # dot product = cosine sim (already L2-normalized)
        pos_sims.append(float(da @ db))
        gt_desc0.append(da)
        gt_pts1.append(pts1[i1])
        gt_nn_dist0.append(d0)
        gt_nn_dist1.append(d1)

    if len(pos_sims) < 5:
        print(f"  [SKIP] only {len(pos_sims)} valid positive pairs")
        return None

    pos_sims = np.array(pos_sims)
    gt_desc0 = np.array(gt_desc0)    # (M, 352)
    gt_pts1  = np.array(gt_pts1)     # (M, 3)
    snap_dist = (np.mean(gt_nn_dist0) + np.mean(gt_nn_dist1)) / 2

    print(f"  GT pairs: {len(pos_sims)}, GT→C++cloud snap dist: {snap_dist:.2f}mm (mean)")
    print(f"  Positive cosine sim: {pos_sims.mean():.4f} ± {pos_sims.std():.4f}")

    # ── Random pairs ──
    rng = np.random.default_rng(rng_seed)
    n_rand = min(len(pos_sims), 1000)
    r0 = rng.integers(0, len(desc0), n_rand)
    r1 = rng.integers(0, len(desc1), n_rand)
    rand_sims = (desc0[r0] * desc1[r1]).sum(axis=1)
    print(f"  Random cosine sim  : {rand_sims.mean():.4f} ± {rand_sims.std():.4f}")

    # ── NN matching precision (brute-force matmul, desc already normalized) ──
    # gt_desc0 (M,352) @ desc1.T (352,N) → sims (M,N)
    # Top-2 for ratio test
    sims_mat = gt_desc0 @ desc1.T          # (M, N1)
    top2_idx = np.argpartition(-sims_mat, kth=min(2, sims_mat.shape[1]-1), axis=1)[:, :2]
    # sort within top2
    for i in range(len(top2_idx)):
        if sims_mat[i, top2_idx[i, 0]] < sims_mat[i, top2_idx[i, 1]]:
            top2_idx[i] = top2_idx[i, ::-1]

    best_idx   = top2_idx[:, 0]
    second_idx = top2_idx[:, 1]
    sim_best   = sims_mat[np.arange(len(best_idx)), best_idx]
    sim_second = sims_mat[np.arange(len(best_idx)), second_idx]

    # Lowe ratio in cosine-dist space: sqrt((1-s1)/(1-s2)) < 0.8
    # (equivalent to L2 ratio for unit vectors)
    ratio = np.sqrt(np.clip(1 - sim_best, 1e-8, None) /
                    np.clip(1 - sim_second, 1e-8, None))
    ratio_pass = float((ratio < 0.8).mean())

    # 3D distance between NN-found point and GT reference point in input cloud
    nn_pts  = pts1[best_idx]               # (M, 3) found points
    nn_dist = np.linalg.norm(nn_pts - gt_pts1, axis=1)
    prec5   = float((nn_dist < 5.0 ).mean())
    prec10  = float((nn_dist < 10.0).mean())
    prec20  = float((nn_dist < 20.0).mean())

    print(f"  NN 3D dist: {nn_dist.mean():.1f}mm mean, {np.median(nn_dist):.1f}mm median")
    print(f"  NN prec @5mm:{prec5*100:.1f}% @10mm:{prec10*100:.1f}% @20mm:{prec20*100:.1f}%")
    print(f"  Lowe ratio pass (<0.8): {ratio_pass*100:.1f}%")

    return {
        "pos_sims":   pos_sims,
        "rand_sims":  rand_sims,
        "ratio_pass": ratio_pass,
        "prec5":      prec5,
        "prec10":     prec10,
        "prec20":     prec20,
        "nn_dist":    nn_dist,
        "snap_dist":  snap_dist,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs",  type=int, default=10,
                        help="검증할 이미지 쌍 수")
    parser.add_argument("--max_gt",   type=int, default=500,
                        help="쌍당 최대 GT 대응점 수")
    parser.add_argument("--bin_dir",  type=str,
                        default="gluefactory/datasets/Descriptor/output_shot_0407",
                        help="C++ SHOT352 .bin 디렉토리")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    base_dir   = Path("gluefactory/datasets/mitsubishi")
    image_dir  = base_dir / "dataset_resample_2"
    output_dir = base_dir / "outputs_txt"
    bin_dir    = Path(args.bin_dir)

    combo = pd.read_csv(output_dir / "combination.csv")
    pairs = combo.sample(n=min(args.n_pairs, len(combo)),
                         random_state=args.seed).reset_index(drop=True)

    print(f"=== C++ SHOT352 Discriminability Validation ===")
    print(f"Descriptor: SHOT352 (352D, L2-normalized)")
    print(f"Parameters: voxelSize=1.0mm, normalRadius=5.0mm, shotRadius=10.0mm")
    print(f"Pairs: {len(pairs)}, max_gt={args.max_gt}")
    print()

    all_results = []
    t_total = time.time()
    for i, row in pairs.iterrows():
        master_path = image_dir / Path(row["master_path"]).name
        input_path  = image_dir / Path(row["input_path"]).name
        csv_path    = output_dir / Path(row["csv_path"]).name
        print(f"[{i+1}/{len(pairs)}] {master_path.name} ↔ {input_path.name}")
        r = process_pair(master_path, input_path, csv_path,
                         bin_dir, args.max_gt, args.seed)
        if r is not None:
            all_results.append(r)

    print(f"\nTotal time: {time.time()-t_total:.1f}s")

    if not all_results:
        print("\n[FAIL] No valid pairs processed.")
        return

    all_pos  = np.concatenate([r["pos_sims"]  for r in all_results])
    all_rand = np.concatenate([r["rand_sims"] for r in all_results])
    all_nn   = np.concatenate([r["nn_dist"]   for r in all_results])
    gap      = float(all_pos.mean() - all_rand.mean())

    snap   = np.mean([r["snap_dist"]  for r in all_results])
    r_pass = np.mean([r["ratio_pass"] for r in all_results])
    p5     = np.mean([r["prec5"]      for r in all_results])
    p10    = np.mean([r["prec10"]     for r in all_results])
    p20    = np.mean([r["prec20"]     for r in all_results])

    print(f"\n{'='*55}")
    print(f"SUMMARY  ({len(all_results)} pairs, {len(all_pos)} positive pairs total)")
    print(f"  GT→C++ cloud snap dist : {snap:.2f}mm  (small = good coverage)")
    print()
    print(f"  Positive cosine sim    : {all_pos.mean():.4f} ± {all_pos.std():.4f}")
    print(f"  Random cosine sim      : {all_rand.mean():.4f} ± {all_rand.std():.4f}")
    print(f"  Gap (pos - rand)       : {gap:+.4f}  (>= 0.05 목표)")
    print()
    print(f"  NN 3D dist mean/median : {all_nn.mean():.1f}mm / {np.median(all_nn):.1f}mm")
    print(f"  NN precision @5mm      : {p5*100:.1f}%")
    print(f"  NN precision @10mm     : {p10*100:.1f}%")
    print(f"  NN precision @20mm     : {p20*100:.1f}%")
    print(f"  Lowe ratio pass (<0.8) : {r_pass*100:.1f}%")
    print()

    pass_gap  = gap >= 0.05
    pass_prec = p10 >= 0.20   # NN의 20% 이상이 10mm 이내
    pass_ratio = r_pass >= 0.20

    if pass_gap and pass_prec:
        verdict = "PASS"
        msg = "Proceed to precompute → training."
    elif pass_gap:
        verdict = "MARGINAL"
        msg = "Gap OK but NN precision low. Consider larger radius."
    else:
        verdict = "FAIL"
        msg = "Gap insufficient. SHOT not discriminative at radius=10.0mm."

    print(f"  → [{verdict}] {msg}")
    print('='*55)


if __name__ == "__main__":
    main()
