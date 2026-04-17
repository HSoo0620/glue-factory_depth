"""ISS + SHOT352 (v1, 2026-04-15) precompute.

좌표계: mm. voxel=1mm, normal_r=20mm, shot_r=40mm. keypoints=512.
SHOT은 shot_module.extract_shot_at_keypoints (PCL setSearchSurface 분리) 사용
→ 진짜 keypoint-only.

출력: gluefactory/datasets/mitsubishi/iss_shot_v1_20260415_cache_vox1_nr20_sr40/{stem}.npz

사용 (repo root 에서):
    python experiments/v1_20260415/precompute/iss_shot.py --max_images 1 --force   # 스모크
    python experiments/v1_20260415/precompute/iss_shot.py                           # 전체 641
"""
import argparse
import json
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# pybind shot module path
SHOT_MOD_DIR = Path("/home/jhs/work/Registration/glue-factory_depth/pybind_shot_linux")
sys.path.insert(0, str(SHOT_MOD_DIR))
import shot_module  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers as h  # noqa: E402

DATA_ROOT = Path("/mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output")
CACHE_DIR = Path(
    "/home/jhs/work/Registration/glue-factory_depth/gluefactory/datasets/mitsubishi/"
    "iss_shot_v1_20260415_cache_vox1_nr20_sr40"
)

META = {
    "voxel": h.VOXEL_SIZE,
    "normal_r": h.NORMAL_RADIUS,
    "shot_r": h.SHOT_RADIUS,
    "lateral": h.LATERAL_MM,
    "transport": h.TRANSPORT_MM,
    "vertical": h.VERTICAL_MM,
    "pad": [h.PAD_W, h.PAD_H],
    "max_keypoints": h.MAX_KEYPOINTS,
    "dataset": "v1_20260415",
}


def process_one(png_path, out_path, seed=0):
    pts, uv, shape = h.load_zmap_to_pcd_mm(png_path)
    if len(pts) < 100:
        return False
    pcd_vox = h.voxel_downsample(pts, voxel_size=h.VOXEL_SIZE)
    if len(pcd_vox.points) < 100:
        return False

    # ISS on dense PCD (voxel은 descriptor용)
    kp_xyz_iss = h.extract_iss_on_dense(pts)
    kp_xyz, kp_idx, kp_score, n_iss, n_valid = h.subsample_or_pad_keypoints(
        kp_xyz_iss, pcd_vox, max_n=h.MAX_KEYPOINTS, seed=seed)

    # SHOT352 keypoint-only via pybind11 (P_vox dense as search surface)
    vox_pts = np.asarray(pcd_vox.points, dtype=np.float32)
    result = shot_module.extract_shot_at_keypoints(
        vox_pts, kp_xyz.astype(np.float32),
        voxel_size=h.VOXEL_SIZE,
        normal_radius=h.NORMAL_RADIUS,
        shot_radius=h.SHOT_RADIUS,
    )
    shot_kp = result["descriptors"]   # (512, 352)
    valid = result["valid_mask"]       # (512,)

    # (u, v)
    kp_uv = h.kp_xyz_mm_to_uv_padded(kp_xyz)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        keypoints=kp_uv,
        keypoint_scores=kp_score,
        keypoints_xyz_mm=kp_xyz,
        descriptors=shot_kp.astype(np.float32),
        valid_mask=valid.astype(bool),
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
    for png in tqdm(pngs, desc="precompute SHOT"):
        out = CACHE_DIR / f"{png.stem}.npz"
        if out.exists() and not args.force:
            n_skip += 1
            continue
        try:
            if process_one(png, out, seed=args.seed):
                n_done += 1
        except Exception as e:
            print(f"\n  ERROR on {png.name}: {e}")
            out.unlink(missing_ok=True)  # ⚠️ carry-over from Phase 4 code review
    print(f"\nDone: {n_done} new, {n_skip} skipped, {len(pngs)} total target")


if __name__ == "__main__":
    main()
