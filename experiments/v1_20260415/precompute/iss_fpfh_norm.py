"""ISS + FPFH L2-normalized (v1, 2026-04-15) precompute.

좌표계: mm. voxel=1mm, normal_r=20mm, fpfh_r=20mm. keypoints=512.
FPFH raw histogram 추출 후 L2 normalize 적용 (기존 iss_fpfh.py는 raw 그대로 저장).

입력: /mnt/aict_nas/.../dataset_output/zmap_*.png
출력: gluefactory/datasets/mitsubishi/iss_fpfh_v1_20260415_cache_vox1_nr20_fr20_norm/{stem}.npz

사용 (repo root 에서):
    conda activate LightGlue
    python experiments/v1_20260415/precompute/iss_fpfh_norm.py --max_images 1 --force   # 스모크
    python experiments/v1_20260415/precompute/iss_fpfh_norm.py                           # 전체 641
"""
import argparse
import json
import sys
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers as h  # noqa: E402

DATA_ROOT = Path("/mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output")
CACHE_DIR = Path(
    "/home/jhs/work/Registration/glue-factory_depth/gluefactory/datasets/mitsubishi/"
    "iss_fpfh_v1_20260415_cache_vox1_nr20_fr20_norm"
)

META = {
    "voxel": h.VOXEL_SIZE,
    "normal_r": h.NORMAL_RADIUS,
    "fpfh_r": h.FPFH_RADIUS,
    "lateral": h.LATERAL_MM,
    "transport": h.TRANSPORT_MM,
    "vertical": h.VERTICAL_MM,
    "pad": [h.PAD_W, h.PAD_H],
    "max_keypoints": h.MAX_KEYPOINTS,
    "dataset": "v1_20260415",
    "normalized": True,
}


def process_one(png_path, out_path, seed=0):
    pts, uv, shape = h.load_zmap_to_pcd_mm(png_path)
    if len(pts) < 100:
        print(f"  SKIP (too few points: {len(pts)})")
        return False
    pcd_vox = h.voxel_downsample(pts, voxel_size=h.VOXEL_SIZE)
    if len(pcd_vox.points) < 100:
        print(f"  SKIP (voxel too sparse: {len(pcd_vox.points)})")
        return False

    # ISS on dense PCD (voxel은 descriptor용)
    kp_xyz_iss = h.extract_iss_on_dense(pts)
    kp_xyz, kp_idx, kp_score, n_iss, n_valid = h.subsample_or_pad_keypoints(
        kp_xyz_iss, pcd_vox, max_n=h.MAX_KEYPOINTS, seed=seed)

    # Normals + FPFH (dense on voxel cloud)
    pcd_vox.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamRadius(radius=h.NORMAL_RADIUS))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_vox, o3d.geometry.KDTreeSearchParamRadius(radius=h.FPFH_RADIUS))
    fpfh_data = np.asarray(fpfh.data, dtype=np.float32)  # (33, M)

    # Index selection
    fpfh_kp = fpfh_data[:, kp_idx].T.astype(np.float32)  # (512, 33)

    # L2 normalize
    norms = np.linalg.norm(fpfh_kp, axis=1, keepdims=True)
    fpfh_kp = fpfh_kp / (norms + 1e-8)

    # (u, v) — mm → pixel
    kp_uv = h.kp_xyz_mm_to_uv_padded(kp_xyz)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        keypoints=kp_uv,
        keypoint_scores=kp_score,
        keypoints_xyz_mm=kp_xyz,
        descriptors=fpfh_kp,
        n_iss=np.int32(n_iss),
        n_valid=np.int32(n_valid),
        meta=np.frombuffer(json.dumps(META).encode("utf-8"), dtype=np.uint8),
    )
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pngs = sorted(DATA_ROOT.glob("zmap_*.png"))
    if args.max_images:
        pngs = pngs[:args.max_images]
    print(f"Cache dir : {CACHE_DIR}")
    print(f"Targets   : {len(pngs)} zmap")

    n_done = n_skip = 0
    for png in tqdm(pngs, desc="precompute FPFH (L2-norm)"):
        out = CACHE_DIR / f"{png.stem}.npz"
        if out.exists() and not args.force:
            n_skip += 1
            continue
        try:
            ok = process_one(png, out, seed=args.seed)
            if ok:
                n_done += 1
        except Exception as e:
            print(f"\n  ERROR on {png.name}: {e}")
            out.unlink(missing_ok=True)
    print(f"\nDone: {n_done} new, {n_skip} skipped, {len(pngs)} total target")


if __name__ == "__main__":
    main()
