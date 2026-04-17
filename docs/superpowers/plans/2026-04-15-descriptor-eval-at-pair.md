# Descriptor Evaluation at User-defined Pair — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scanned 와 Blender depth 이미지의 동일 물리 지점에서 SHOT(352D) / FPFH(33D) descriptor 를 뽑아 cross-domain discriminability 를 1 pair 단위로 시각 비교.

**Architecture:** `step1_bilateral_fps.py` 의 헬퍼(bilateral, ISS mm PCD, ISS detect) 를 복붙 재사용. 사용자 pixel pair 를 mm 로 올려 ISS keypoint 최근접으로 snap, pybind `shot_module` 과 Open3D FPFH 로 descriptor 를 추출해 NN lookup, SHOT·FPFH 각 독립 PNG 로 bar overlay 렌더.

**Tech Stack:** Python 3.10 (`LightGlue` conda env), numpy, scipy (cKDTree), opencv-python, open3d, matplotlib, repo-local `pybind_shot_linux/shot_module.so`.

**User rules honored (from MEMORY):**
- `git commit은 사용자가 직접 수행` → 각 Task 의 "commit" step 은 **사용자가 직접 실행** 하는 제안 명령으로 문서화 (자동 커밋 금지).
- `사용자 확인 없이 코드/설정 파일 수정 금지` → 각 Task 시작 전 사용자 확인 필요. 구현자는 Task 단위로 승인 요청.

**Spec reference:** `docs/superpowers/specs/2026-04-15-descriptor-eval-at-pair-design.md`

---

## File Structure

| 파일 | 역할 |
|---|---|
| `vis_new/Descriptor__분석/eval_descriptor_at_pair.py` | 단일 eval 스크립트 (helpers + main) |
| `vis_new/Descriptor__분석/test_snap.py` | `snap_to_iss` 최소 unit test |
| `vis_new/Descriptor__분석/eval_descriptor_at_pair_shot.png` | 실행 산출물 (SHOT) |
| `vis_new/Descriptor__분석/eval_descriptor_at_pair_fpfh.png` | 실행 산출물 (FPFH) |

---

## Task 1: Scaffold file (paths, constants, imports, empty `main`)

**Files:**
- Create: `vis_new/Descriptor__분석/eval_descriptor_at_pair.py`

- [ ] **Step 1: Make folder**

```bash
mkdir -p /home/jhs/work/Registration/glue-factory_depth/vis_new/Descriptor__분석
```

- [ ] **Step 2: Write scaffold**

```python
"""Descriptor evaluation at a user-defined pair (scanned vs blender).

Spec: docs/superpowers/specs/2026-04-15-descriptor-eval-at-pair-design.md

Usage:
    conda activate LightGlue
    cd /home/jhs/work/Registration/glue-factory_depth
    python vis_new/Descriptor__분석/eval_descriptor_at_pair.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vis_new.inference_scan_vs_train import mask_scanned_table

_SHOT_DIR = ROOT / "pybind_shot_linux"
if str(_SHOT_DIR) not in sys.path:
    sys.path.insert(0, str(_SHOT_DIR))
import shot_module  # noqa: E402  pybind11 built .so

# ---------- Paths ----------
SCANNED_PATH = ROOT / "gluefactory/datasets/scanned/roi13_zmap 1.png"
BLENDER_PATH = ROOT / "gluefactory/datasets/scanned/blender_master1/master.png"
OUT_DIR = ROOT / "vis_new/Descriptor__분석"

# ---------- Physical spacing ----------
LAT_MM = 0.056
VERT_MM = 0.0085

# ---------- Bilateral (scanned only) ----------
BILAT_DIAMETER = 5
BILAT_SIGMA_COLOR = 100.0
BILAT_SIGMA_SPACE = 3.0

# ---------- ISS ----------
GAMMA_21 = 0.5
GAMMA_32 = 0.5
MIN_NBRS = 5
ERODE_BOUNDARY = 5

# ---------- SHOT ----------
SHOT_VOXEL = 1.0
SHOT_NORMAL_R = 10.0
SHOT_RADIUS = 20.0

# ---------- FPFH ----------
FPFH_VOXEL = 1.0
FPFH_NORMAL_R = 10.0
FPFH_RADIUS = 20.0

# ---------- Snap ----------
SNAP_WARN_MM = 5.0

# ---------- User pair (EDIT HERE) ----------
# ((us, vs), (ub, vb))  — scanned pixel xy, blender(rot180) pixel xy
PAIRS: list[tuple[tuple[int, int], tuple[int, int]]] = [
    # ((1234, 567), (890, 123)),   # fill in after user provides coords
]


def main() -> None:
    if not PAIRS:
        print("[INFO] PAIRS is empty. Edit PAIRS at top of file with (us,vs),(ub,vb) pixel coords.")
        return
    raise NotImplementedError("wiring in later tasks")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke run — empty pairs path**

```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda run -n LightGlue python vis_new/Descriptor__분석/eval_descriptor_at_pair.py
```

Expected output:
```
[INFO] PAIRS is empty. Edit PAIRS at top of file with (us,vs),(ub,vb) pixel coords.
```

- [ ] **Step 4: Suggested commit (user runs)**

```bash
git add vis_new/Descriptor__분석/eval_descriptor_at_pair.py
git commit -m "feat(eval): scaffold descriptor-at-pair evaluator"
```

---

## Task 2: Copy shared helpers from step1 (bilateral, build_iss_pcd_mm, detect_iss_mm)

**Files:**
- Modify: `vis_new/Descriptor__분석/eval_descriptor_at_pair.py` (append helpers above `main`)

- [ ] **Step 1: Append three helpers above `main`**

```python
def apply_bilateral_depth(zmap_u16: np.ndarray) -> np.ndarray:
    """uint16 depth 에 bilateral filter. zero 픽셀은 보존."""
    f32 = zmap_u16.astype(np.float32)
    mask0 = zmap_u16 == 0
    filtered = cv2.bilateralFilter(
        f32, d=BILAT_DIAMETER,
        sigmaColor=BILAT_SIGMA_COLOR,
        sigmaSpace=BILAT_SIGMA_SPACE,
    )
    filtered[mask0] = 0.0
    return np.clip(filtered, 0, 65535).astype(np.uint16)


def build_iss_pcd_mm(zmap: np.ndarray, erode_boundary: int = ERODE_BOUNDARY):
    """uint16 zmap → (pcd_mm, eroded_mask)."""
    mask = (zmap > 0).astype(np.uint8)
    if erode_boundary > 0:
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=erode_boundary)
    vs, us = np.where(mask > 0)
    if len(us) == 0:
        return o3d.geometry.PointCloud(), mask
    xs = us.astype(np.float64) * LAT_MM
    ys = vs.astype(np.float64) * LAT_MM
    zs = zmap[vs, us].astype(np.float64) * VERT_MM
    pts = np.stack([xs, ys, zs], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd, mask


def detect_iss_mm(pcd: o3d.geometry.PointCloud):
    """ISS keypoints in mm space. Returns (kp_xyz_mm, salient_r_mm, avg_nn_mm)."""
    if len(pcd.points) == 0:
        return np.zeros((0, 3)), 0.0, 0.0
    dists = pcd.compute_nearest_neighbor_distance()
    avg = float(np.mean(dists))
    sr = 6.0 * avg
    nr = 2.0 * sr
    kp = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd, salient_radius=sr, non_max_radius=nr,
        gamma_21=GAMMA_21, gamma_32=GAMMA_32, min_neighbors=MIN_NBRS,
    )
    return np.asarray(kp.points), sr, avg
```

- [ ] **Step 2: Quick sanity — file still parses**

```bash
conda run -n LightGlue python -c "import ast; ast.parse(open('vis_new/Descriptor__분석/eval_descriptor_at_pair.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Suggested commit**

```bash
git add vis_new/Descriptor__분석/eval_descriptor_at_pair.py
git commit -m "feat(eval): port bilateral/ISS helpers from step1"
```

---

## Task 3: `snap_to_iss` + unit test

**Files:**
- Modify: `vis_new/Descriptor__분석/eval_descriptor_at_pair.py` (append `snap_to_iss`)
- Create: `vis_new/Descriptor__분석/test_snap.py`

- [ ] **Step 1: Write the failing test**

```python
# vis_new/Descriptor__분석/test_snap.py
"""Unit test for snap_to_iss (single responsibility, pure function)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vis_new.Descriptor__분석.eval_descriptor_at_pair import snap_to_iss  # noqa: E402


def test_snap_returns_nearest_point_and_distance():
    kp = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    q = np.array([1.0, 0.5, 0.0])

    snapped, dist = snap_to_iss(q, kp)

    np.testing.assert_array_equal(snapped, kp[0])
    assert pytest.approx(dist, rel=1e-9) == float(np.linalg.norm(q - kp[0]))


def test_snap_empty_keypoints_raises():
    q = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        snap_to_iss(q, np.zeros((0, 3)))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda run -n LightGlue python -m pytest vis_new/Descriptor__분석/test_snap.py -v
```

Expected: FAIL with `ImportError: cannot import name 'snap_to_iss'` (or similar import error).

- [ ] **Step 3: Implement `snap_to_iss`**

Append to `eval_descriptor_at_pair.py` (above `main`):

```python
def snap_to_iss(q_mm: np.ndarray, kp_mm: np.ndarray) -> tuple[np.ndarray, float]:
    """(3,) query → nearest in (N,3) kp_mm. Returns (snapped_xyz, dist_mm)."""
    if kp_mm.shape[0] == 0:
        raise ValueError("snap_to_iss: kp_mm is empty")
    tree = cKDTree(kp_mm)
    dist, idx = tree.query(q_mm, k=1)
    return kp_mm[idx], float(dist)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
conda run -n LightGlue python -m pytest vis_new/Descriptor__분석/test_snap.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Suggested commit**

```bash
git add vis_new/Descriptor__분석/eval_descriptor_at_pair.py vis_new/Descriptor__분석/test_snap.py
git commit -m "feat(eval): snap_to_iss with tests"
```

---

## Task 4: `compute_shot_desc_at` + `compute_fpfh_desc_at`

**Files:**
- Modify: `vis_new/Descriptor__분석/eval_descriptor_at_pair.py` (append two functions)

Rationale for no unit tests: heavy I/O, depends on real depth data and pybind SHOT module. Spec explicitly limits tests to `snap_to_iss`. End-to-end validation in Task 7 smoke run.

- [ ] **Step 1: Append `compute_shot_desc_at`**

```python
def compute_shot_desc_at(
    pcd_mm: o3d.geometry.PointCloud,
    query_xyz_mm: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Run pybind SHOT on full pcd, NN lookup at query_xyz_mm.

    Returns (desc (352,), n_shot_pts, lookup_dist_mm).
    """
    pts = np.asarray(pcd_mm.points, dtype=np.float32)
    r = shot_module.extract_shot(
        pts,
        voxel_size=float(SHOT_VOXEL),
        normal_radius=float(SHOT_NORMAL_R),
        shot_radius=float(SHOT_RADIUS),
    )
    pts_shot = r["points"]          # (M, 3) float32
    desc_shot = r["descriptors"]    # (M, 352) float32
    if pts_shot.shape[0] == 0:
        raise RuntimeError("SHOT returned 0 points — params likely invalid")
    tree = cKDTree(pts_shot)
    dist, idx = tree.query(query_xyz_mm, k=1)
    desc = np.asarray(desc_shot[idx], dtype=np.float64)
    assert desc.shape == (352,), f"SHOT desc shape {desc.shape}, expected (352,)"
    return desc, int(pts_shot.shape[0]), float(dist)
```

- [ ] **Step 2: Append `compute_fpfh_desc_at`**

```python
def compute_fpfh_desc_at(
    pcd_mm: o3d.geometry.PointCloud,
    query_xyz_mm: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Voxel-downsample → normals → FPFH, NN lookup at query_xyz_mm.

    Returns (desc (33,), n_fpfh_pts, lookup_dist_mm).
    """
    pcd_down = pcd_mm.voxel_down_sample(FPFH_VOXEL)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamRadius(radius=FPFH_NORMAL_R)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamRadius(radius=FPFH_RADIUS),
    )
    pts_down = np.asarray(pcd_down.points)
    if pts_down.shape[0] == 0:
        raise RuntimeError("FPFH: voxel-downsampled cloud is empty")
    tree = cKDTree(pts_down)
    dist, idx = tree.query(query_xyz_mm, k=1)
    desc = np.asarray(fpfh.data[:, idx], dtype=np.float64)
    assert desc.shape == (33,), f"FPFH desc shape {desc.shape}, expected (33,)"
    if not np.any(desc):
        print("[WARN] FPFH descriptor is all-zero — consider larger fpfh_radius",
              file=sys.stderr)
    return desc, int(pts_down.shape[0]), float(dist)
```

- [ ] **Step 3: Sanity parse**

```bash
conda run -n LightGlue python -c "import ast; ast.parse(open('vis_new/Descriptor__분석/eval_descriptor_at_pair.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Suggested commit**

```bash
git add vis_new/Descriptor__분석/eval_descriptor_at_pair.py
git commit -m "feat(eval): SHOT/FPFH descriptor-at-point functions"
```

---

## Task 5: `render_shot_figure` + `render_fpfh_figure` + `_stretch` display helper

**Files:**
- Modify: `vis_new/Descriptor__분석/eval_descriptor_at_pair.py`

- [ ] **Step 1: Append `_stretch` (display normalize, step1 과 동일)**

```python
def _stretch(img: np.ndarray) -> np.ndarray:
    valid = img[img > 0]
    if valid.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(valid, [5, 95])
    out = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    out[img == 0] = 0.0
    return out
```

- [ ] **Step 2: Append common ISS-context panel helper**

```python
def _draw_iss_context(
    ax,
    img_disp: np.ndarray,
    kp_uv: np.ndarray,
    picked_uv: tuple[float, float],
    title: str,
) -> None:
    ax.imshow(img_disp, cmap="gray", vmin=0, vmax=1)
    if kp_uv.shape[0] > 0:
        ax.scatter(kp_uv[:, 0], kp_uv[:, 1], c="cyan", s=4, alpha=0.5, edgecolors="none")
    ax.scatter([picked_uv[0]], [picked_uv[1]], c="yellow", s=150, marker="*",
               edgecolors="black", linewidths=1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
```

- [ ] **Step 3: Append `_cosine` helper**

```python
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
```

- [ ] **Step 4: Append `render_shot_figure`**

```python
def render_shot_figure(
    *,
    scanned_disp: np.ndarray,
    blender_disp: np.ndarray,
    kp_s_uv: np.ndarray,
    kp_b_uv: np.ndarray,
    picked_s_uv: tuple[float, float],
    picked_b_uv: tuple[float, float],
    desc_s: np.ndarray,
    desc_b: np.ndarray,
    snap_dist_s: float,
    snap_dist_b: float,
    pair_pixels: tuple[tuple[int, int], tuple[int, int]],
    out_path: Path,
) -> None:
    (us, vs), (ub, vb) = pair_pixels
    cos = _cosine(desc_s, desc_b)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.0])
    ax_s = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    _draw_iss_context(ax_s, scanned_disp, kp_s_uv, picked_s_uv,
                      f"scanned (bilat) + ISS, picked=({us},{vs})")
    _draw_iss_context(ax_b, blender_disp, kp_b_uv, picked_b_uv,
                      f"blender (rot180) + ISS, picked=({ub},{vb})")

    x = np.arange(352)
    ax_bar.bar(x, desc_s, color="red",  alpha=0.6, label="scanned", width=1.0)
    ax_bar.bar(x, desc_b, color="blue", alpha=0.6, label="blender", width=1.0)
    ax_bar.set_xlabel("SHOT bin index (0..351)")
    ax_bar.set_ylabel("bin value")
    ax_bar.legend(loc="upper right")
    ax_bar.set_xlim(-0.5, 351.5)

    fig.suptitle(
        f"SHOT @ pair ({us},{vs})↔({ub},{vb}) | cos={cos:.3f} | "
        f"snap: s={snap_dist_s:.2f}mm b={snap_dist_b:.2f}mm",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")
```

- [ ] **Step 5: Append `render_fpfh_figure`**

```python
def render_fpfh_figure(
    *,
    scanned_disp: np.ndarray,
    blender_disp: np.ndarray,
    kp_s_uv: np.ndarray,
    kp_b_uv: np.ndarray,
    picked_s_uv: tuple[float, float],
    picked_b_uv: tuple[float, float],
    desc_s: np.ndarray,
    desc_b: np.ndarray,
    snap_dist_s: float,
    snap_dist_b: float,
    pair_pixels: tuple[tuple[int, int], tuple[int, int]],
    out_path: Path,
) -> None:
    (us, vs), (ub, vb) = pair_pixels
    cos = _cosine(desc_s, desc_b)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.0])
    ax_s = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    _draw_iss_context(ax_s, scanned_disp, kp_s_uv, picked_s_uv,
                      f"scanned (bilat) + ISS, picked=({us},{vs})")
    _draw_iss_context(ax_b, blender_disp, kp_b_uv, picked_b_uv,
                      f"blender (rot180) + ISS, picked=({ub},{vb})")

    x = np.arange(33)
    ax_bar.bar(x, desc_s, color="red",  alpha=0.6, label="scanned", width=0.9)
    ax_bar.bar(x, desc_b, color="blue", alpha=0.6, label="blender", width=0.9)
    ax_bar.set_xlabel("FPFH bin index (0..32)")
    ax_bar.set_ylabel("bin value")
    ax_bar.legend(loc="upper right")
    ax_bar.set_xlim(-0.5, 32.5)

    fig.suptitle(
        f"FPFH @ pair ({us},{vs})↔({ub},{vb}) | cos={cos:.3f} | "
        f"snap: s={snap_dist_s:.2f}mm b={snap_dist_b:.2f}mm",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")
```

- [ ] **Step 6: Sanity parse**

```bash
conda run -n LightGlue python -c "import ast; ast.parse(open('vis_new/Descriptor__분석/eval_descriptor_at_pair.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Suggested commit**

```bash
git add vis_new/Descriptor__분석/eval_descriptor_at_pair.py
git commit -m "feat(eval): SHOT/FPFH figure renderers"
```

---

## Task 6: Wire up `main()` with depth-zero guard, logging, and pair loop

**Files:**
- Modify: `vis_new/Descriptor__분석/eval_descriptor_at_pair.py` (replace `main`)

- [ ] **Step 1: Replace `main` body**

```python
def _xyz_to_uv(xyz_mm: np.ndarray) -> np.ndarray:
    uv = np.zeros((xyz_mm.shape[0], 2), dtype=np.float64)
    uv[:, 0] = xyz_mm[:, 0] / LAT_MM
    uv[:, 1] = xyz_mm[:, 1] / LAT_MM
    return uv


def main() -> None:
    if not PAIRS:
        print("[INFO] PAIRS is empty. Edit PAIRS at top of file with (us,vs),(ub,vb) pixel coords.")
        return

    assert SCANNED_PATH.exists(), f"missing: {SCANNED_PATH}"
    assert BLENDER_PATH.exists(), f"missing: {BLENDER_PATH}"

    # --- Load + preprocess ---
    scanned_raw = cv2.imread(str(SCANNED_PATH), cv2.IMREAD_UNCHANGED)
    scanned_masked, _ = mask_scanned_table(scanned_raw)
    scanned_bilat = apply_bilateral_depth(scanned_masked)

    blender_raw = cv2.imread(str(BLENDER_PATH), cv2.IMREAD_UNCHANGED)
    blender_rot = cv2.rotate(blender_raw, cv2.ROTATE_180)

    # --- Build PCD + ISS ---
    pcd_s, _ = build_iss_pcd_mm(scanned_bilat)
    pcd_b, _ = build_iss_pcd_mm(blender_rot)
    kp_s_mm, sr_s, _ = detect_iss_mm(pcd_s)
    kp_b_mm, sr_b, _ = detect_iss_mm(pcd_b)

    print(f"[INFO] n_pcd_s={len(pcd_s.points)} n_pcd_b={len(pcd_b.points)}")
    print(f"[INFO] n_iss_s={kp_s_mm.shape[0]} (salient_r={sr_s:.3f}mm)")
    print(f"[INFO] n_iss_b={kp_b_mm.shape[0]} (salient_r={sr_b:.3f}mm)")

    # Display versions
    disp_s = _stretch(scanned_bilat)
    disp_b = _stretch(blender_rot)
    kp_s_uv = _xyz_to_uv(kp_s_mm)
    kp_b_uv = _xyz_to_uv(kp_b_mm)

    for i, ((us, vs), (ub, vb)) in enumerate(PAIRS):
        print(f"\n=== pair {i}: scanned=({us},{vs})  blender=({ub},{vb}) ===")

        H_s, W_s = scanned_bilat.shape
        H_b, W_b = blender_rot.shape
        assert 0 <= us < W_s and 0 <= vs < H_s, f"scanned pixel ({us},{vs}) out of bounds {W_s}x{H_s}"
        assert 0 <= ub < W_b and 0 <= vb < H_b, f"blender pixel ({ub},{vb}) out of bounds {W_b}x{H_b}"

        z_s = int(scanned_bilat[vs, us])
        z_b = int(blender_rot[vb, ub])
        if z_s == 0:
            raise ValueError(
                f"scanned pixel ({us},{vs}) has depth=0 — mask/floor 영역 가능성, 좌표 확인")
        if z_b == 0:
            raise ValueError(
                f"blender pixel ({ub},{vb}) has depth=0 — mask/floor 영역 가능성, 좌표 확인")

        q_s_mm = np.array([us * LAT_MM, vs * LAT_MM, z_s * VERT_MM])
        q_b_mm = np.array([ub * LAT_MM, vb * LAT_MM, z_b * VERT_MM])

        snap_s_xyz, snap_s_dist = snap_to_iss(q_s_mm, kp_s_mm)
        snap_b_xyz, snap_b_dist = snap_to_iss(q_b_mm, kp_b_mm)
        print(f"[INFO] snap_dist_s={snap_s_dist:.3f}mm  snap_dist_b={snap_b_dist:.3f}mm")
        if snap_s_dist > SNAP_WARN_MM:
            print(f"[WARN] scanned snap_dist={snap_s_dist:.2f}mm > {SNAP_WARN_MM}mm",
                  file=sys.stderr)
        if snap_b_dist > SNAP_WARN_MM:
            print(f"[WARN] blender snap_dist={snap_b_dist:.2f}mm > {SNAP_WARN_MM}mm",
                  file=sys.stderr)

        shot_s, n_shot_s, lookup_s_shot = compute_shot_desc_at(pcd_s, snap_s_xyz)
        shot_b, n_shot_b, lookup_b_shot = compute_shot_desc_at(pcd_b, snap_b_xyz)
        fpfh_s, n_fpfh_s, lookup_s_fpfh = compute_fpfh_desc_at(pcd_s, snap_s_xyz)
        fpfh_b, n_fpfh_b, lookup_b_fpfh = compute_fpfh_desc_at(pcd_b, snap_b_xyz)

        print(f"[INFO] n_shot_s={n_shot_s} n_shot_b={n_shot_b}  "
              f"shot_lookup_dist s={lookup_s_shot:.3f}mm b={lookup_b_shot:.3f}mm")
        print(f"[INFO] n_fpfh_s={n_fpfh_s} n_fpfh_b={n_fpfh_b}  "
              f"fpfh_lookup_dist s={lookup_s_fpfh:.3f}mm b={lookup_b_fpfh:.3f}mm")
        print(f"[INFO] SHOT : L2_s={np.linalg.norm(shot_s):.3f} L2_b={np.linalg.norm(shot_b):.3f}"
              f"  zero_bins_s={int(np.sum(shot_s == 0))} cos={_cosine(shot_s, shot_b):.4f}")
        print(f"[INFO] FPFH : L2_s={np.linalg.norm(fpfh_s):.3f} L2_b={np.linalg.norm(fpfh_b):.3f}"
              f"  zero_bins_s={int(np.sum(fpfh_s == 0))} cos={_cosine(fpfh_s, fpfh_b):.4f}")

        picked_s_uv = (snap_s_xyz[0] / LAT_MM, snap_s_xyz[1] / LAT_MM)
        picked_b_uv = (snap_b_xyz[0] / LAT_MM, snap_b_xyz[1] / LAT_MM)

        render_shot_figure(
            scanned_disp=disp_s, blender_disp=disp_b,
            kp_s_uv=kp_s_uv, kp_b_uv=kp_b_uv,
            picked_s_uv=picked_s_uv, picked_b_uv=picked_b_uv,
            desc_s=shot_s, desc_b=shot_b,
            snap_dist_s=snap_s_dist, snap_dist_b=snap_b_dist,
            pair_pixels=((us, vs), (ub, vb)),
            out_path=OUT_DIR / f"eval_descriptor_at_pair{i}_shot.png",
        )
        render_fpfh_figure(
            scanned_disp=disp_s, blender_disp=disp_b,
            kp_s_uv=kp_s_uv, kp_b_uv=kp_b_uv,
            picked_s_uv=picked_s_uv, picked_b_uv=picked_b_uv,
            desc_s=fpfh_s, desc_b=fpfh_b,
            snap_dist_s=snap_s_dist, snap_dist_b=snap_b_dist,
            pair_pixels=((us, vs), (ub, vb)),
            out_path=OUT_DIR / f"eval_descriptor_at_pair{i}_fpfh.png",
        )
```

Note: output file name suffix uses `pair{i}` so multi-pair (future) works without overwrite.
Spec originally said `eval_descriptor_at_pair_{shot,fpfh}.png`; with N=1 the files become
`eval_descriptor_at_pair0_shot.png` / `eval_descriptor_at_pair0_fpfh.png`. This deviation
from spec is small and intentional for forward compatibility.

- [ ] **Step 2: Sanity parse**

```bash
conda run -n LightGlue python -c "import ast; ast.parse(open('vis_new/Descriptor__분석/eval_descriptor_at_pair.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Empty-pair smoke (unchanged behavior)**

```bash
conda run -n LightGlue python vis_new/Descriptor__분석/eval_descriptor_at_pair.py
```

Expected: `[INFO] PAIRS is empty. ...`

- [ ] **Step 4: Suggested commit**

```bash
git add vis_new/Descriptor__분석/eval_descriptor_at_pair.py
git commit -m "feat(eval): wire main — load/ISS/snap/descriptors/render"
```

---

## Task 7: End-to-end smoke with user-provided pair

**Files:**
- Modify: `vis_new/Descriptor__분석/eval_descriptor_at_pair.py` (fill `PAIRS`)

- [ ] **Step 1: Fill in `PAIRS` with user-provided coords**

Replace `PAIRS` constant:
```python
PAIRS: list[tuple[tuple[int, int], tuple[int, int]]] = [
    ((US, VS), (UB, VB)),   # <-- 실제 값으로 치환 (사용자 제공)
]
```
(`US, VS, UB, VB` 는 사용자가 대화에서 알려주신 정수값으로 직접 치환)

- [ ] **Step 2: Run end-to-end**

```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda run -n LightGlue python vis_new/Descriptor__분석/eval_descriptor_at_pair.py 2>&1 | tee vis_new/Descriptor__분석/run.log
```

Expected stdout highlights:
- `[INFO] n_pcd_s=... n_pcd_b=...` (수만~수백만 점)
- `[INFO] n_iss_s=..., n_iss_b=...` (수백~수천 점)
- `[INFO] snap_dist_s=..., snap_dist_b=...` (ideally ≤ 5mm)
- `[INFO] SHOT : L2_s=... cos=0.xxxx`
- `[INFO] FPFH : L2_s=... cos=0.xxxx`
- `saved -> .../eval_descriptor_at_pair0_shot.png`
- `saved -> .../eval_descriptor_at_pair0_fpfh.png`

- [ ] **Step 3: Visual verification**

PNG 2장을 열어서 확인:
- **Row 0**: 두 이미지 모두에 picked ★ 가 object 내부 의도한 위치에 찍혔는지
- **Row 1**: bar overlay 가 합리적 — 모든 bin 0 이거나 NaN 안 나왔는지
- **Title**: cos_sim 값이 화면에 기록됐는지

```bash
xdg-open vis_new/Descriptor__분석/eval_descriptor_at_pair0_shot.png 2>/dev/null &
xdg-open vis_new/Descriptor__분석/eval_descriptor_at_pair0_fpfh.png 2>/dev/null &
```

- [ ] **Step 4: Suggested commit**

```bash
git add vis_new/Descriptor__분석/eval_descriptor_at_pair.py
git commit -m "chore(eval): populate PAIRS with user-provided coords"
```

주: `run.log` 과 PNG 산출물은 기본 gitignore 여부 확인 후 필요시 `git add -f` 로 별도 커밋.

---

## Self-review (writing-plans skill 요건)

**1. Spec coverage:**

| Spec 요건 | 담당 Task |
|---|---|
| mm 변환 + bilateral (scanned only) | Task 2 (helpers), Task 6 (main) |
| ISS keypoint 추출 | Task 2, Task 6 |
| User pair hardcoded | Task 1 (constant), Task 7 (populate) |
| ISS snap + warn threshold | Task 3 (function + test), Task 6 (wiring + warn) |
| SHOT descriptor (voxel=1, nr=10, sr=20) | Task 4, Task 6 |
| FPFH descriptor (voxel=1, nr=10, fr=20) | Task 4, Task 6 |
| Cosine similarity | Task 5 (_cosine), Task 6 (log) |
| Depth=0 조기 ValueError | Task 6 |
| Snap warn > 5mm | Task 6 |
| SHOT/FPFH 각 독립 PNG | Task 5 (renderers), Task 6 (call) |
| Figure Row0: ISS context + picked ★ | Task 5 (`_draw_iss_context`) |
| Figure Row1: bar overlay (red/blue α=0.6) | Task 5 (renderers) |
| Title with cos + snap_dist | Task 5 |
| 로그 print (n_pcd, n_iss, snap_dist, desc stats) | Task 6 |

전체 spec 섹션 모두 반영됨.

**2. Placeholder scan:** `US, VS, UB, VB` 는 Task 7 Step 1 에서 **사용자가 실값으로 치환해야 하는 자리** 로 의도적으로 남김 (사용자 입력 대기). 이외 "TBD / TODO / implement later" 없음.

**3. Type consistency:** `snap_to_iss` signature (Task 3) → `main` 호출부 (Task 6) 일치. `compute_shot_desc_at` / `compute_fpfh_desc_at` 리턴 `(desc, n, dist)` 3-tuple 과 main 언패킹 (`shot_s, n_shot_s, lookup_s_shot = ...`) 일치. Renderer 의 키워드 인자명도 main 호출부와 매치.
