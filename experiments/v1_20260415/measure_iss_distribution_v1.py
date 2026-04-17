"""641장 zmap 전체에 대해 ISS keypoint 분포 측정.

목적: Phase 3 code-quality 리뷰에서 제기된 "ISS 44개 (max_n=512)" 이슈의
데이터셋 전체 분포 확인. 결과는 plan 부록 기록용.

사용: python measure_iss_distribution_v1.py
소요: ~8-10분 (8 workers, mp=spawn)
"""
import json
import numpy as np
from pathlib import Path
import multiprocessing as mp
import time

import precompute_helpers_v1_20260415 as h

NAS = Path("/mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output")
OUT_JSON = Path(__file__).parent / "iss_distribution_v1_20260415.json"
N_WORKERS = 8


def measure_one(png_path_str):
    p = Path(png_path_str)
    pts, _, (H, W) = h.load_zmap_to_pcd_mm(p)
    pcd_vox = h.voxel_downsample(pts)
    kp = h.extract_iss_on_voxel(pcd_vox)
    return {
        "name": p.name,
        "H": int(H),
        "W": int(W),
        "N_raw": int(len(pts)),
        "M_vox": int(len(pcd_vox.points)),
        "K_iss": int(len(kp)),
    }


def main():
    pngs = sorted(NAS.glob("zmap_*.png"))
    print(f"found {len(pngs)} files, launching {N_WORKERS} workers")
    t0 = time.time()
    with mp.get_context("spawn").Pool(N_WORKERS) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(
            measure_one, [str(p) for p in pngs], chunksize=4)):
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(pngs)} done ({time.time()-t0:.1f}s)")
    dt = time.time() - t0
    print(f"total time: {dt:.1f}s")

    Ks = np.array([r["K_iss"] for r in results])
    Ms = np.array([r["M_vox"] for r in results])

    summary = {
        "n_files": len(results),
        "total_seconds": round(dt, 1),
        "K_iss": {
            "min": int(Ks.min()),
            "p10": int(np.percentile(Ks, 10)),
            "p25": int(np.percentile(Ks, 25)),
            "median": int(np.median(Ks)),
            "mean": round(float(Ks.mean()), 1),
            "p75": int(np.percentile(Ks, 75)),
            "p90": int(np.percentile(Ks, 90)),
            "max": int(Ks.max()),
            "frac_ge_512": round(float((Ks >= 512).mean()), 3),
            "frac_ge_256": round(float((Ks >= 256).mean()), 3),
            "frac_ge_128": round(float((Ks >= 128).mean()), 3),
            "frac_ge_64":  round(float((Ks >= 64 ).mean()), 3),
            "frac_lt_64":  round(float((Ks <  64 ).mean()), 3),
        },
        "M_vox": {
            "min": int(Ms.min()),
            "median": int(np.median(Ms)),
            "max": int(Ms.max()),
        },
        "params": {
            "voxel_size": h.VOXEL_SIZE,
            "salient_radius_mul": 6.0,
            "non_max_radius_mul": 2.0,
            "gamma_21": h.ISS_GAMMA_21,
            "gamma_32": h.ISS_GAMMA_32,
            "min_neighbors": h.ISS_MIN_NEIGHBORS,
        },
        "bottom10_by_K": sorted(results, key=lambda r: r["K_iss"])[:10],
        "top10_by_K": sorted(results, key=lambda r: r["K_iss"], reverse=True)[:10],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nwritten: {OUT_JSON}")
    print("\n=== Summary ===")
    s = summary["K_iss"]
    print(f"K_iss: min={s['min']}, p10={s['p10']}, median={s['median']}, "
          f"mean={s['mean']}, p90={s['p90']}, max={s['max']}")
    print(f"  >=512: {s['frac_ge_512']:.1%}, >=256: {s['frac_ge_256']:.1%}, "
          f">=128: {s['frac_ge_128']:.1%}, >=64: {s['frac_ge_64']:.1%}, "
          f"<64: {s['frac_lt_64']:.1%}")


if __name__ == "__main__":
    main()
