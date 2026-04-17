"""Public API: register_pair / preload_model / unload_model."""
from __future__ import annotations
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, TypedDict, Union

import numpy as np
import torch

from . import params as P
from .preprocessing import load_depth_raw, preprocess_scanned, preprocess_master
from .iss import detect_iss_mm
from .descriptors.fpfh import compute_fpfh
from .descriptors.shot import compute_shot
from .cache import master_cache_key, load_master_cache, save_master_cache
from .matcher import load_lightglue, run_lightglue, mm_to_uv
from .registration import ransac_rigid

Descriptor = Literal["fpfh", "shot"]
ImageLike = Union[str, Path, np.ndarray]


class RegistrationResult(TypedDict):
    T: np.ndarray
    R: np.ndarray
    t: np.ndarray
    inlier_ratio: float
    num_matches: int
    num_inliers: int
    matches_scanned_xyz: np.ndarray
    matches_master_xyz: np.ndarray
    success: bool
    elapsed_ms: dict


# ─── 모델 상주 정책: 한 번에 하나 ─────────────────────────────
_ACTIVE: Optional[tuple] = None   # (desc, model, device)


def _get_model(descriptor: Descriptor, device: str) -> torch.nn.Module:
    global _ACTIVE
    if _ACTIVE is not None and _ACTIVE[0] == descriptor and _ACTIVE[2] == device:
        return _ACTIVE[1]
    if _ACTIVE is not None:
        _ACTIVE = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    model = load_lightglue(descriptor, device=device)
    _ACTIVE = (descriptor, model, device)
    return model


def preload_model(descriptor: Descriptor, device: str = "cuda") -> None:
    _get_model(descriptor, device)


def unload_model(descriptor: Optional[Descriptor] = None) -> None:
    global _ACTIVE
    if _ACTIVE is None:
        return
    if descriptor is None or _ACTIVE[0] == descriptor:
        _ACTIVE = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─── 헬퍼: descriptor 파라미터 dict (cache fingerprint 용) ──
def _params_dict(descriptor: Descriptor) -> dict:
    base = {
        "voxel": P.VOXEL_MM, "normal_r": P.NORMAL_R_MM,
        "iss_salient_mult": P.ISS_SALIENT_MULT,
        "iss_nonmax_mult":  P.ISS_NONMAX_MULT,
        "iss_gamma_21": P.ISS_GAMMA_21,
        "iss_gamma_32": P.ISS_GAMMA_32,
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


# ─── 메인 API ─────────────────────────────────────────────────
def register_pair(
    scanned: ImageLike,
    master: Optional[ImageLike] = None,
    descriptor: Descriptor = "fpfh",
    *,
    cache_dir: Optional[Union[str, Path]] = None,
    inlier_th: float = P.DEFAULT_INLIER_TH,
    ransac_iter: int = P.DEFAULT_RANSAC_ITER,
    success_min_inlier_ratio: float = 0.1,
    device: str = "cuda",
) -> RegistrationResult:
    if descriptor not in ("fpfh", "shot"):
        raise ValueError(f"unsupported descriptor: {descriptor!r}")
    cache_dir_p = Path(cache_dir) if cache_dir is not None else P.DEFAULT_CACHE_DIR
    master_in = master if master is not None else P.DEFAULT_MASTER_PATH

    t_all = time.perf_counter()
    elapsed = {}

    # 1) Scanned
    t0 = time.perf_counter()
    scanned_pts = preprocess_scanned(scanned)
    elapsed["preproc"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    scanned_kp = detect_iss_mm(scanned_pts)
    desc_fn = compute_fpfh if descriptor == "fpfh" else compute_shot
    scanned_desc = desc_fn(scanned_pts, scanned_kp)
    elapsed["descriptor_scanned"] = (time.perf_counter() - t0) * 1000

    # 2) Master (캐시)
    t0 = time.perf_counter()
    master_zmap = load_depth_raw(master_in)
    params = _params_dict(descriptor)
    key = master_cache_key(master_zmap, descriptor, params)
    cached = load_master_cache(key, cache_dir_p)
    if cached is not None:
        master_kp   = np.asarray(cached["kp_mm"],  dtype=np.float32)
        master_desc = np.asarray(cached["desc"],   dtype=np.float32)
    else:
        master_pts  = preprocess_master(master_zmap)
        master_kp   = detect_iss_mm(master_pts)
        master_desc = desc_fn(master_pts, master_kp)
        payload = dict(
            kp_mm=master_kp, desc=master_desc, pts_mm=master_pts,
            img_hash=key.split("_")[1], params_hash=key.split("_")[2],
            descriptor=descriptor,
            params_json=json.dumps(params, sort_keys=True, default=str),
            created_at=datetime.now(timezone.utc).isoformat(),
            version="0.1.0",
        )
        save_master_cache(key, payload, cache_dir_p)
    elapsed["descriptor_master"] = (time.perf_counter() - t0) * 1000

    # 3) LightGlue
    t0 = time.perf_counter()
    model = _get_model(descriptor, device)
    kp0_uv = mm_to_uv(scanned_kp)
    kp1_uv = mm_to_uv(master_kp)
    pred = run_lightglue(model, kp0_uv, scanned_desc, kp1_uv, master_desc,
                          device=device)
    elapsed["lightglue"] = (time.perf_counter() - t0) * 1000

    m = pred["matches0"]
    valid = m >= 0
    num_matches = int(valid.sum())
    if num_matches < 4:
        src_mm = np.zeros((0, 3), dtype=np.float32)
        dst_mm = np.zeros((0, 3), dtype=np.float32)
        T = np.eye(4)
        inlier_mask = np.zeros(0, dtype=bool)
        elapsed["ransac"] = 0.0
    else:
        src_mm = scanned_kp[valid]
        dst_mm = master_kp[m[valid]]
        t0 = time.perf_counter()
        T, inlier_mask = ransac_rigid(src_mm, dst_mm,
                                       n_iter=ransac_iter, inlier_th=inlier_th)
        elapsed["ransac"] = (time.perf_counter() - t0) * 1000

    num_inliers = int(inlier_mask.sum())
    inlier_ratio = num_inliers / max(num_matches, 1)
    success = (num_inliers >= 3) and (inlier_ratio >= success_min_inlier_ratio)

    elapsed["total"] = (time.perf_counter() - t_all) * 1000
    return RegistrationResult(
        T=T.astype(np.float64),
        R=T[:3, :3].astype(np.float64),
        t=T[:3, 3].astype(np.float64),
        inlier_ratio=float(inlier_ratio),
        num_matches=num_matches,
        num_inliers=num_inliers,
        matches_scanned_xyz=src_mm[inlier_mask].astype(np.float32),
        matches_master_xyz=dst_mm[inlier_mask].astype(np.float32),
        success=bool(success),
        elapsed_ms=elapsed,
    )
