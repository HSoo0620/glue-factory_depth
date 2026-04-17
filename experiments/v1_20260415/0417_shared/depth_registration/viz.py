"""선택 시각화 헬퍼. 핵심 API 와 분리 (spec §12)."""
from __future__ import annotations
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import params as P
from .preprocessing import load_depth_raw


def _to_display(z: np.ndarray) -> np.ndarray:
    m = z > 0
    if m.sum() == 0:
        return np.zeros_like(z, dtype=np.float32)
    lo, hi = np.percentile(z[m], [5, 95])
    out = np.zeros_like(z, dtype=np.float32)
    out[m] = np.clip((z[m] - lo) / max(hi - lo, 1.0), 0, 1)
    return out


def render_overlay_png(scanned, master, result, out_path=None) -> bytes:
    zs = _to_display(load_depth_raw(scanned))
    zm = _to_display(load_depth_raw(master))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(zm, cmap="gray"); axes[0].set_title("Master")
    axes[1].imshow(zs, cmap="gray"); axes[1].set_title("Scanned")
    axes[2].imshow(zm, cmap="gray", alpha=0.6)
    axes[2].imshow(zs, cmap="hot",  alpha=0.4)
    axes[2].set_title(f"Overlay (inliers={result['num_inliers']})")
    for a in axes: a.axis("off")
    buf = BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    data = buf.getvalue()
    if out_path is not None:
        Path(out_path).write_bytes(data)
    return data


def render_matches_png(scanned, master, result, out_path=None) -> bytes:
    zs = _to_display(load_depth_raw(scanned))
    zm = _to_display(load_depth_raw(master))
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(zs, cmap="gray"); ax[0].set_title("Scanned")
    ax[1].imshow(zm, cmap="gray"); ax[1].set_title("Master")
    if result["num_inliers"] > 0:
        u0 = result["matches_scanned_xyz"][:, 0] / P.LATERAL_MM
        v0 = result["matches_scanned_xyz"][:, 1] / P.TRANSPORT_MM
        u1 = result["matches_master_xyz"][:, 0] / P.LATERAL_MM
        v1 = result["matches_master_xyz"][:, 1] / P.TRANSPORT_MM
        ax[0].scatter(u0, v0, s=4, c="limegreen")
        ax[1].scatter(u1, v1, s=4, c="limegreen")
    for a in ax: a.axis("off")
    buf = BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    data = buf.getvalue()
    if out_path is not None:
        Path(out_path).write_bytes(data)
    return data
