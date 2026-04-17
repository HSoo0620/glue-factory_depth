"""Custom SVD RANSAC (test_registration_resample2_iss_shot.py 이식)."""
from __future__ import annotations
import numpy as np


def _svd_rigid(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    cs = src.mean(0)
    cd = dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cd - R @ cs
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def ransac_rigid(src_mm: np.ndarray, dst_mm: np.ndarray,
                 n_iter: int = 1000, inlier_th: float = 5.0,
                 seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(src_mm, dtype=np.float64)
    dst = np.asarray(dst_mm, dtype=np.float64)
    n = len(src)
    if n < 3:
        return np.eye(4), np.zeros(n, dtype=bool)

    rng = np.random.default_rng(seed)
    best_inlier = np.zeros(n, dtype=bool)
    best_count = 0

    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        try:
            T_try = _svd_rigid(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        proj = (T_try[:3, :3] @ src.T).T + T_try[:3, 3]
        err  = np.linalg.norm(proj - dst, axis=1)
        inlier = err < inlier_th
        cnt = int(inlier.sum())
        if cnt > best_count:
            best_count = cnt
            best_inlier = inlier

    if best_count < 3:
        return np.eye(4), np.zeros(n, dtype=bool)

    T_ref = _svd_rigid(src[best_inlier], dst[best_inlier])
    return T_ref, best_inlier
