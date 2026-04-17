"""Scanned/Master depth → mm point cloud 전처리.

기존 infer_scanned_vs_master_shot352.py / share/depth_to_iss.py 의 로직을 통합.
Spec §5 의 함수 시그니처를 따른다.
"""
from __future__ import annotations
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

from . import params as P

ImageLike = Union[str, Path, np.ndarray]


def load_depth_raw(img: ImageLike) -> np.ndarray:
    """uint16 (H, W) depth map 반환."""
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint16:
            raise ValueError(f"expected uint16 ndarray, got {img.dtype}")
        return img
    arr = np.array(Image.open(str(img)))
    if arr.dtype == np.uint8:
        arr = arr.astype(np.uint16) * 257
    if arr.ndim != 2:
        raise ValueError(f"expected (H, W) depth map, got shape {arr.shape}")
    return arr.astype(np.uint16)


def mask_scanned_table(zmap: np.ndarray,
                       band_fraction: float = P.FLOOR_BAND_FRACTION,
                       bin_width: int = P.FLOOR_BIN_WIDTH) -> np.ndarray:
    """Histogram peak(=floor) 주변 band 를 0 으로 마스킹."""
    out = zmap.copy()
    mask = out > 0
    if mask.sum() == 0:
        return out
    values = out[mask]
    bins = np.arange(0, 65536 + bin_width, bin_width)
    hist, _ = np.histogram(values, bins=bins)
    peak_bin = int(hist.argmax())
    peak_height = int(hist[peak_bin])
    threshold = peak_height * band_fraction
    left = peak_bin
    while left > 0 and hist[left - 1] > threshold:
        left -= 1
    right = peak_bin
    while right < len(hist) - 1 and hist[right + 1] > threshold:
        right += 1
    lo = bins[left]
    hi = bins[right + 1]
    band = (out >= lo) & (out <= hi)
    out[band] = 0
    return out


def apply_bilateral(zmap: np.ndarray,
                    d: int = P.BILAT_D,
                    sigma_c: float = P.BILAT_SIGMA_C,
                    sigma_s: float = P.BILAT_SIGMA_S) -> np.ndarray:
    """OpenCV bilateral (uint16 → float32 → uint16)."""
    z = zmap.astype(np.float32)
    out = cv2.bilateralFilter(z, d, sigma_c, sigma_s)
    return np.clip(out, 0, 65535).astype(np.uint16)


def zmap_to_pcd_mm(zmap: np.ndarray,
                   erode_boundary_px: int = P.ERODE_BOUNDARY_PX) -> np.ndarray:
    """uint16 zmap → (N, 3) float32 mm PCD."""
    mask = (zmap > 0).astype(np.uint8)
    if erode_boundary_px > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=erode_boundary_px)
    vs, us = np.where(mask > 0)
    zs = zmap[vs, us].astype(np.float32)
    return np.column_stack([
        us.astype(np.float32) * P.LATERAL_MM,
        vs.astype(np.float32) * P.TRANSPORT_MM,
        zs * P.VERTICAL_MM,
    ]).astype(np.float32)


def preprocess_scanned(img: ImageLike) -> np.ndarray:
    z = load_depth_raw(img)
    z = mask_scanned_table(z)
    z = apply_bilateral(z)
    return zmap_to_pcd_mm(z)


def preprocess_master(img: ImageLike) -> np.ndarray:
    z = load_depth_raw(img)
    return zmap_to_pcd_mm(z)
