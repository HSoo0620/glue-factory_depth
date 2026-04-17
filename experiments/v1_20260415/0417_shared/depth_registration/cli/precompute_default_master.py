"""기본 master 캐시 사전 계산.

사용:
  python -m depth_registration.cli.precompute_default_master --descriptor fpfh
  python -m depth_registration.cli.precompute_default_master --descriptor shot
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .. import params as P
from ..preprocessing import load_depth_raw, preprocess_master
from ..iss import detect_iss_mm
from ..cache import master_cache_key, save_master_cache
from ..descriptors.fpfh import compute_fpfh
from ..descriptors.shot import compute_shot


def _params_dict(descriptor: str) -> dict:
    base = {
        "voxel": P.VOXEL_MM, "normal_r": P.NORMAL_R_MM,
        "iss_salient_mult": P.ISS_SALIENT_MULT,
        "iss_nonmax_mult":  P.ISS_NONMAX_MULT,
        "iss_gamma_21": P.ISS_GAMMA_21, "iss_gamma_32": P.ISS_GAMMA_32,
        "iss_min_neighbors": P.ISS_MIN_NEIGHBORS,
        "max_keypoints": P.MAX_KEYPOINTS,
        "erode_boundary_px": P.ERODE_BOUNDARY_PX,
        "lateral_mm": P.LATERAL_MM, "transport_mm": P.TRANSPORT_MM,
        "vertical_mm": P.VERTICAL_MM,
    }
    if descriptor == "fpfh":
        base["fpfh_r"] = P.FPFH_R_MM
    else:
        base["shot_r"] = P.SHOT_R_MM
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptor", choices=["fpfh", "shot"], required=True)
    ap.add_argument("--master", type=str, default=str(P.DEFAULT_MASTER_PATH))
    ap.add_argument("--cache-dir", type=str, default=str(P.DEFAULT_CACHE_DIR))
    args = ap.parse_args()

    zmap = load_depth_raw(args.master)
    pts  = preprocess_master(zmap)
    kp   = detect_iss_mm(pts)
    desc_fn = compute_fpfh if args.descriptor == "fpfh" else compute_shot
    desc = desc_fn(pts, kp)
    params = _params_dict(args.descriptor)
    key = master_cache_key(zmap, args.descriptor, params)
    payload = dict(
        kp_mm=kp, desc=desc, pts_mm=pts,
        img_hash=key.split("_")[1], params_hash=key.split("_")[2],
        descriptor=args.descriptor,
        params_json=json.dumps(params, sort_keys=True, default=str),
        created_at=datetime.now(timezone.utc).isoformat(),
        version="0.1.0",
    )
    save_master_cache(key, payload, Path(args.cache_dir))
    print(f"saved: {Path(args.cache_dir) / (key + '.npz')}")
    print(f"kp={len(kp)} desc={desc.shape} pts={len(pts)}")


if __name__ == "__main__":
    main()
