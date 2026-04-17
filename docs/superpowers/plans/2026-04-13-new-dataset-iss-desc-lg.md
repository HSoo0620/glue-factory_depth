# New Dataset: ISS + FPFH/SHOT + LightGlue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an ISS detector + FPFH/SHOT descriptor + LightGlue matcher training/inference pipeline for the new feature matching dataset at `/mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output` (641 scenes / 64K pairs, variable-height zmaps). FPFH is runnable immediately; SHOT is config-switchable once the C++ bins arrive.

**Architecture:**
1. New package `gluefactory/datasets/new_dataset/` (dataset + coords + ISS detection + split + constants) — isolated from existing Mitsubishi code.
2. Precompute scripts produce `(512,D)` descriptor caches per scene; training/test reuses the existing `two_view_pipeline` / `SuperPointFPFHCached` / `LightGlue` / `gt_pair_matcher` stack.
3. Dynamic padding collate handles variable H (W=2413 fixed, H=1556~2975).

**Tech Stack:** Python 3, PyTorch, Open3D (ISS + FPFH + KDTree), OpenCV (image I/O + resize + erode), pandas/PyYAML, pytest, OmegaConf.

**Spec reference:** `docs/superpowers/specs/2026-04-13-new-dataset-iss-desc-lg-design.md` (read before each phase).

**Key conventions (mirror from existing code, spec §4, §7):**
- Camera/sensor frame: `X = u·LAT_MM, Y = v·LAT_MM, Z = raw·VERT_MM` (mm). `LAT_MM=0.056, VERT_MM=0.0085`.
- Resize factor = 0.5 (1/2, INTER_NEAREST). Keypoint cache stored in **resized** (u,v) pixels, float32.
- Descriptor caches: FPFH (33D), SHOT (352D). Padding zero-filled, `scores=0.0` for pad, `scores=0.5` for random-fill, `scores=1.0` for ISS.
- Scene split: 95/5 train/val, `seed=0`, no separate test.
- SHOT C++ bin file name: `<zmap_stem>_shot352.bin` (e.g., `zmap_0042_shot352.bin`), format `uint32 N, uint32 D=352, per-point [f32 x,y,z, f32[352]]` in camera frame mm.

**All work runs inside conda env `LightGlue` from `/home/jhs/work/Registration/glue-factory_depth/`.** Tests go in `tests/`.

---

## File Structure

### New files
| Path | Responsibility |
|---|---|
| `gluefactory/datasets/new_dataset/__init__.py` | Re-export dataset + collate_fn |
| `gluefactory/datasets/new_dataset/constants.py` | `LAT_MM, TRANS_MM, VERT_MM, RESIZE_FACTOR, DEFAULT_DATA_ROOT, CLIP_START?` (none — orthographic), `N_SCENES`, `DESC_DIMS={'fpfh':33,'shot':352}` |
| `gluefactory/datasets/new_dataset/coords.py` | `SceneConfig` dataclass, `load_scene_config`, `pixel_to_cam_xyz`, `cam_to_world_xyz`, `build_camera_frame_pcd` |
| `gluefactory/datasets/new_dataset/iss_detection.py` | `build_iss_pcd_uvd_scaled`, `detect_iss_keypoints`, `select_keypoints` |
| `gluefactory/datasets/new_dataset/split.py` | `scene_split`, `pair_filename_to_scene_ids`, `filter_pairs_by_scenes` |
| `gluefactory/datasets/new_dataset/dataset.py` | `NewDatasetISSDescDataset`, `collate_fn_dynamic_pad` |
| `precompute_new_iss_fpfh.py` | FPFH precompute script (Open3D) |
| `precompute_new_iss_shot.py` | SHOT352 precompute script (load C++ bin, KDTree lookup) |
| `gluefactory/train_new_iss_desc.py` | Training loop fork of `train_resample2_iss_fpfh.py` |
| `gluefactory/configs/0413_new_iss_fpfh_lg.yaml` | FPFH config (input_dim=33) |
| `gluefactory/configs/0413_new_iss_shot_lg.yaml` | SHOT config (input_dim=352) |
| `test_new_iss_desc.py` | Match visualization (4-color overlay) |
| `eval_registration_new_iss_desc.py` | Registration RMSE evaluation skeleton |
| `train_new_iss_fpfh.sh` | GPU/experiment/FPFH_RADIUS/BATCH/RESTORE launcher |
| `train_new_iss_shot.sh` | GPU/experiment/BATCH/RESTORE launcher |
| `tests/test_new_dataset_constants.py` | Unit tests for constants (minimal sanity) |
| `tests/test_new_dataset_coords.py` | Unit tests for coords.py |
| `tests/test_new_dataset_split.py` | Unit tests for split.py |
| `tests/test_new_dataset_iss_detection.py` | Unit tests for iss_detection.py |
| `tests/test_new_dataset_integration.py` | 1-scene end-to-end integration test |

### Files NOT touched (reuse only)
- `gluefactory/models/extractors/superpoint_fpfh_cached.py`
- `gluefactory/models/matchers/lightglue.py`
- `gluefactory/models/matchers/gt_pair_matcher.py`
- `gluefactory/models/two_view_pipeline.py`
- `gluefactory/visualization/visualize_batch.py::make_match_figures_depth`
- Any `mitsubishi_*` dataset file or `0402_*` config

---

## Task 1: Scaffold the package + constants

**Files:**
- Create: `gluefactory/datasets/new_dataset/__init__.py`
- Create: `gluefactory/datasets/new_dataset/constants.py`
- Create: `tests/test_new_dataset_constants.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_new_dataset_constants.py`:

```python
from pathlib import Path

from gluefactory.datasets.new_dataset import constants as C


def test_constants_values():
    assert C.LAT_MM == 0.056
    assert C.TRANS_MM == 0.056
    assert C.VERT_MM == 0.0085
    assert C.RESIZE_FACTOR == 0.5
    assert C.N_SCENES == 641
    assert C.DESC_DIMS == {"fpfh": 33, "shot": 352}
    assert isinstance(C.DEFAULT_DATA_ROOT, Path)
    assert str(C.DEFAULT_DATA_ROOT) == "/mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output"
```

- [ ] **Step 2: Run test — expect fail**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue pytest tests/test_new_dataset_constants.py -v`
Expected: FAIL with `ModuleNotFoundError: gluefactory.datasets.new_dataset`

- [ ] **Step 3: Create the package skeleton**

Create `gluefactory/datasets/new_dataset/__init__.py`:

```python
"""New dataset: ISS + FPFH/SHOT + LightGlue.

Public interface:
    from gluefactory.datasets.new_dataset import (
        NewDatasetISSDescDataset, collate_fn_dynamic_pad,
    )
"""

from .dataset import NewDatasetISSDescDataset, collate_fn_dynamic_pad

__all__ = ["NewDatasetISSDescDataset", "collate_fn_dynamic_pad"]
```

Create `gluefactory/datasets/new_dataset/constants.py`:

```python
"""Immutable constants for the new dataset pipeline."""
from pathlib import Path

LAT_MM = 0.056
TRANS_MM = 0.056
VERT_MM = 0.0085

RESIZE_FACTOR = 0.5

N_SCENES = 641

DESC_DIMS = {"fpfh": 33, "shot": 352}

DEFAULT_DATA_ROOT = Path("/mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output")
```

Note: `__init__.py` imports from `dataset.py`. The test imports `constants` directly (`from ... import constants as C`), so the package import only needs `constants.py` to exist as a submodule — but importing the package triggers `__init__.py`. Since `dataset.py` does not exist yet, the test file must import `constants` via `importlib` *or* `__init__.py` must tolerate the missing submodule.

Choose the robust approach: write `__init__.py` as lazy — keep it empty for now; populate it in Task 6.

**Replace** `__init__.py` content above with:

```python
"""New dataset: ISS + FPFH/SHOT + LightGlue.

Public interface is assembled progressively during implementation.
Final exports live in `dataset.py`.
"""
```

- [ ] **Step 4: Run test — expect pass**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue pytest tests/test_new_dataset_constants.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add gluefactory/datasets/new_dataset/__init__.py gluefactory/datasets/new_dataset/constants.py tests/test_new_dataset_constants.py
git commit -m "feat(new_dataset): scaffold package + immutable constants"
```

---

## Task 2: `coords.py` — scene config + frame conversions

**Files:**
- Create: `gluefactory/datasets/new_dataset/coords.py`
- Create: `tests/test_new_dataset_coords.py`

Reference values from spec §2.3 and `config_0000.yaml`:
- `camera_rt.rotation_matrix` = `R_cam` (3,3)
- `camera_rt.translation_mm` = `t_cam` (3,)
- Camera-frame XYZ = `(u·LAT_MM, v·LAT_MM, raw·VERT_MM)` (mm)
- World/object frame = `R_cam @ X_cam + t_cam`

- [ ] **Step 1: Write failing test**

Create `tests/test_new_dataset_coords.py`:

```python
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from gluefactory.datasets.new_dataset import constants as C
from gluefactory.datasets.new_dataset.coords import (
    SceneConfig,
    build_camera_frame_pcd,
    cam_to_world_xyz,
    load_scene_config,
    pixel_to_cam_xyz,
)

DATA_ROOT = C.DEFAULT_DATA_ROOT
HAS_DATA = DATA_ROOT.exists()


def test_pixel_to_cam_xyz_formula():
    u = np.array([0.0, 100.0, 200.0])
    v = np.array([0.0, 50.0, 100.0])
    raw = np.array([0, 1000, 65535], dtype=np.float64)
    xyz = pixel_to_cam_xyz(u, v, raw)
    assert xyz.shape == (3, 3)
    np.testing.assert_allclose(xyz[:, 0], u * C.LAT_MM)
    np.testing.assert_allclose(xyz[:, 1], v * C.LAT_MM)
    np.testing.assert_allclose(xyz[:, 2], raw * C.VERT_MM)


def test_cam_to_world_roundtrip():
    R = np.eye(3); t = np.array([1.0, 2.0, 3.0])
    cfg = SceneConfig(zmap_shape=(10, 10), R_cam=R, t_cam=t)
    cam = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    world = cam_to_world_xyz(cam, cfg)
    np.testing.assert_allclose(world, cam + t)


def test_cam_to_world_rotation():
    theta = np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0, 0, 1]])
    t = np.zeros(3)
    cfg = SceneConfig(zmap_shape=(10, 10), R_cam=R, t_cam=t)
    cam = np.array([[1.0, 0.0, 0.0]])
    world = cam_to_world_xyz(cam, cfg)
    np.testing.assert_allclose(world, [[0.0, 1.0, 0.0]], atol=1e-10)


@pytest.mark.skipif(not HAS_DATA, reason="dataset mount not available")
def test_load_scene_config_shape_and_dtype():
    cfg = load_scene_config(0, DATA_ROOT)
    assert cfg.R_cam.shape == (3, 3)
    assert cfg.t_cam.shape == (3,)
    assert cfg.zmap_shape[1] == 2413  # W fixed across 641 scenes
    # Rotation matrix should be orthonormal
    np.testing.assert_allclose(cfg.R_cam @ cfg.R_cam.T, np.eye(3), atol=1e-5)


@pytest.mark.skipif(not HAS_DATA, reason="dataset mount not available")
def test_build_camera_frame_pcd_smoke():
    import cv2
    zmap = cv2.imread(str(DATA_ROOT / "zmap_0000.png"), cv2.IMREAD_UNCHANGED)
    pcd, erode_mask = build_camera_frame_pcd(zmap, erode_boundary=5)
    assert isinstance(pcd, o3d.geometry.PointCloud)
    assert erode_mask.shape == zmap.shape
    n = len(pcd.points)
    assert n > 10_000
    pts = np.asarray(pcd.points)
    # Camera-frame mm sanity: X in [0, W·LAT_MM], Z >= 0
    assert pts[:, 0].min() >= 0 and pts[:, 0].max() <= zmap.shape[1] * C.LAT_MM + 1e-6
    assert pts[:, 2].min() >= 0
```

- [ ] **Step 2: Run test — expect fail**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue pytest tests/test_new_dataset_coords.py -v`
Expected: FAIL with `ModuleNotFoundError: gluefactory.datasets.new_dataset.coords`

- [ ] **Step 3: Implement `coords.py`**

Create `gluefactory/datasets/new_dataset/coords.py`:

```python
"""Coordinate frames for the new dataset.

Frames:
  - pixel  : (u, v, raw_uint16)
  - camera : (X, Y, Z) mm = (u·LAT_MM, v·LAT_MM, raw·VERT_MM)
  - world  : R_cam @ cam + t_cam  (pcd.ply frame)

pcd.ply is NOT consumed during precompute. It is only used by the
registration evaluation to compare two views in the world frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

from . import constants as C


@dataclass
class SceneConfig:
    zmap_shape: tuple[int, int]  # (H, W)
    R_cam: np.ndarray            # (3, 3) float64
    t_cam: np.ndarray            # (3,)   float64 (mm)


def load_scene_config(scene_id: int, data_root: Path) -> SceneConfig:
    yaml_path = Path(data_root) / f"config_{scene_id:04d}.yaml"
    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)
    rt = meta["camera_rt"]
    R = np.asarray(rt["rotation_matrix"], dtype=np.float64)
    t = np.asarray(rt["translation_mm"], dtype=np.float64)
    png_path = Path(data_root) / f"zmap_{scene_id:04d}.png"
    zmap = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if zmap is None:
        raise FileNotFoundError(png_path)
    return SceneConfig(zmap_shape=zmap.shape, R_cam=R, t_cam=t)


def pixel_to_cam_xyz(u, v, raw) -> np.ndarray:
    """(u, v, raw_uint16) → camera-frame (X, Y, Z) mm. Vectorized."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    return np.stack([u * C.LAT_MM, v * C.LAT_MM, raw * C.VERT_MM], axis=-1)


def cam_to_world_xyz(xyz_cam: np.ndarray, cfg: SceneConfig) -> np.ndarray:
    return xyz_cam @ cfg.R_cam.T + cfg.t_cam


def build_camera_frame_pcd(zmap: np.ndarray, erode_boundary: int = 0):
    """All valid pixels → camera-frame PCD + erosion mask.

    erode_boundary applied only to `mask`; if you need a PCD that matches
    a specific kp mask, pass erode_boundary > 0 and use the returned mask.
    """
    mask = (zmap > 0).astype(np.uint8)
    if erode_boundary > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=erode_boundary)
    vs, us = np.where(mask > 0)
    raw = zmap[vs, us]
    xyz = pixel_to_cam_xyz(us, vs, raw)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return pcd, mask
```

- [ ] **Step 4: Run test — expect pass**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue pytest tests/test_new_dataset_coords.py -v`
Expected: PASS (5 tests, or 3 passed + 2 skipped if no data mount).

- [ ] **Step 5: Commit**

```bash
git add gluefactory/datasets/new_dataset/coords.py tests/test_new_dataset_coords.py
git commit -m "feat(new_dataset): coords.py — camera/world frame conversions"
```

---

## Task 3: `split.py` — scene-based train/val split

**Files:**
- Create: `gluefactory/datasets/new_dataset/split.py`
- Create: `tests/test_new_dataset_split.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_new_dataset_split.py`:

```python
import pandas as pd
import pytest

from gluefactory.datasets.new_dataset.split import (
    filter_pairs_by_scenes,
    pair_filename_to_scene_ids,
    scene_split,
)


def test_scene_split_sizes_and_disjoint():
    train, val = scene_split(n_scenes=641, val_ratio=0.05, seed=0)
    assert len(train) + len(val) == 641
    assert len(val) == 32    # floor(641 * 0.05)
    assert len(train) == 609
    assert set(train).isdisjoint(set(val))
    assert set(train) | set(val) == set(range(641))


def test_scene_split_reproducible():
    a_train, a_val = scene_split(n_scenes=641, val_ratio=0.05, seed=0)
    b_train, b_val = scene_split(n_scenes=641, val_ratio=0.05, seed=0)
    assert a_train == b_train
    assert a_val == b_val


def test_scene_split_seed_sensitivity():
    a_train, _ = scene_split(n_scenes=641, val_ratio=0.05, seed=0)
    b_train, _ = scene_split(n_scenes=641, val_ratio=0.05, seed=1)
    assert a_train != b_train


def test_pair_filename_to_scene_ids():
    assert pair_filename_to_scene_ids("pair_0000_0001.csv") == (0, 1)
    assert pair_filename_to_scene_ids("pair_0042_0123.csv") == (42, 123)


def test_filter_pairs_by_scenes_both_endpoints_must_match():
    df = pd.DataFrame({
        "master_zmap_path": ["zmap_0000.png", "zmap_0000.png", "zmap_0001.png"],
        "input_zmap_path":  ["zmap_0001.png", "zmap_0002.png", "zmap_0003.png"],
        "csv_path":         ["pair_0000_0001.csv", "pair_0000_0002.csv", "pair_0001_0003.csv"],
    })
    kept = filter_pairs_by_scenes(df, scenes={0, 1})
    # row 0: (0,1) both in → keep
    # row 1: (0,2) 2 not in → drop
    # row 2: (1,3) 3 not in → drop
    assert len(kept) == 1
    assert kept.iloc[0]["csv_path"] == "pair_0000_0001.csv"
```

- [ ] **Step 2: Run test — expect fail**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue pytest tests/test_new_dataset_split.py -v`
Expected: FAIL with `ModuleNotFoundError: gluefactory.datasets.new_dataset.split`

- [ ] **Step 3: Implement `split.py`**

Create `gluefactory/datasets/new_dataset/split.py`:

```python
"""Scene-based train/val split. No data leakage: pairs whose master OR input
scene falls in val go to val; both endpoints must be in train for a pair
to be a training sample."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


_PAIR_RE = re.compile(r"^pair_(\d{4})_(\d{4})\.csv$")


def scene_split(n_scenes: int = 641, val_ratio: float = 0.05, seed: int = 0):
    """Deterministic shuffle → last floor(n·ratio) scenes go to val.

    Returns two sorted lists of scene ids (ints).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_scenes)
    n_val = int(n_scenes * val_ratio)
    val = sorted(perm[-n_val:].tolist())
    train = sorted(perm[:-n_val].tolist())
    return train, val


def pair_filename_to_scene_ids(pair_filename: str) -> tuple[int, int]:
    """'pair_0042_0123.csv' → (42, 123)."""
    name = Path(pair_filename).name
    m = _PAIR_RE.match(name)
    if m is None:
        raise ValueError(f"Unexpected pair filename: {pair_filename}")
    return int(m.group(1)), int(m.group(2))


def filter_pairs_by_scenes(combo_df: pd.DataFrame, scenes: set[int]) -> pd.DataFrame:
    """Keep rows where BOTH master and input scene id are in `scenes`."""
    scenes = set(int(s) for s in scenes)
    keep = []
    for csv_path in combo_df["csv_path"].values:
        mid, iid = pair_filename_to_scene_ids(csv_path)
        keep.append(mid in scenes and iid in scenes)
    return combo_df.loc[keep].reset_index(drop=True)
```

- [ ] **Step 4: Run test — expect pass**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue pytest tests/test_new_dataset_split.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add gluefactory/datasets/new_dataset/split.py tests/test_new_dataset_split.py
git commit -m "feat(new_dataset): split.py — scene-based 95/5 reproducible split"
```

---

## Task 4: `iss_detection.py` — ISS detector + keypoint selection

**Files:**
- Create: `gluefactory/datasets/new_dataset/iss_detection.py`
- Create: `tests/test_new_dataset_iss_detection.py`

Mirror the existing `(u, v, depth_scaled)` scheme from `precompute_iss_fpfh_resample2.py:depth_crop_to_pcd`. The new dataset is orthographic so there is no CROP offset and no `CLIP_START/CLIP_END` conversion — feed the raw uint16 directly (cast to float).

- [ ] **Step 1: Write failing test**

Create `tests/test_new_dataset_iss_detection.py`:

```python
import numpy as np
import open3d as o3d
import pytest

from gluefactory.datasets.new_dataset import constants as C
from gluefactory.datasets.new_dataset.iss_detection import (
    build_iss_pcd_uvd_scaled,
    detect_iss_keypoints,
    select_keypoints,
)

DATA_ROOT = C.DEFAULT_DATA_ROOT
HAS_DATA = DATA_ROOT.exists()


def test_build_iss_pcd_excludes_zero_depth():
    zmap = np.zeros((20, 20), dtype=np.uint16)
    zmap[5:15, 5:15] = 1000
    pcd, mask, depth_scale, depth_min = build_iss_pcd_uvd_scaled(
        zmap, erode_boundary=0
    )
    assert len(pcd.points) == 100  # 10x10 interior
    assert mask.sum() == 100
    assert depth_min >= 0


def test_select_keypoints_exceeds_max():
    iss = np.random.RandomState(0).rand(1000, 3) * 100.0
    zmap = np.ones((128, 128), dtype=np.uint16) * 500
    kp_resized, scores, n_valid, kp_uv_orig = select_keypoints(
        iss, zmap, max_num_keypoints=512, erode_mask=None,
        resize_factor=0.5, rng=np.random.default_rng(0)
    )
    assert kp_resized.shape == (512, 2)
    assert scores.shape == (512,)
    assert kp_uv_orig.shape == (512, 2)
    assert n_valid == 512
    assert (scores == 1.0).sum() == 512
    # resize scaling
    np.testing.assert_allclose(kp_resized, kp_uv_orig * 0.5, atol=1e-6)


def test_select_keypoints_below_max_random_fill():
    iss = np.random.RandomState(0).rand(100, 3) * 50.0
    # iss u,v must be inside zmap bounds and zmap > 0 for fill
    iss[:, 0] = np.clip(iss[:, 0], 0, 127)
    iss[:, 1] = np.clip(iss[:, 1], 0, 127)
    zmap = np.ones((128, 128), dtype=np.uint16) * 500
    kp_resized, scores, n_valid, kp_uv_orig = select_keypoints(
        iss, zmap, max_num_keypoints=512, erode_mask=None,
        resize_factor=0.5, rng=np.random.default_rng(0)
    )
    assert kp_resized.shape == (512, 2)
    assert n_valid == 512
    assert (scores == 1.0).sum() == 100
    assert (scores == 0.5).sum() == 412
    assert (scores == 0.0).sum() == 0


def test_select_keypoints_pad_when_mask_too_small():
    iss = np.random.RandomState(0).rand(10, 3) * 10.0
    zmap = np.zeros((128, 128), dtype=np.uint16)
    zmap[:5, :5] = 500  # only 25 fill candidates
    kp_resized, scores, n_valid, kp_uv_orig = select_keypoints(
        iss, zmap, max_num_keypoints=512, erode_mask=None,
        resize_factor=0.5, rng=np.random.default_rng(0)
    )
    # 10 ISS + 25 random = 35, rest padded with zero + scores=0
    assert n_valid == 35
    assert (scores == 1.0).sum() == 10
    assert (scores == 0.5).sum() == 25
    assert (scores == 0.0).sum() == 512 - 35
    # pad entries are zero
    np.testing.assert_allclose(kp_resized[35:], 0.0)


@pytest.mark.skipif(not HAS_DATA, reason="dataset mount not available")
def test_detect_iss_keypoints_smoke():
    import cv2
    zmap = cv2.imread(str(DATA_ROOT / "zmap_0000.png"), cv2.IMREAD_UNCHANGED)
    pcd, mask, _, _ = build_iss_pcd_uvd_scaled(zmap, erode_boundary=5)
    iss = detect_iss_keypoints(pcd)
    assert iss.ndim == 2 and iss.shape[1] == 3
    assert len(iss) > 10    # should be hundreds for a real scene
```

- [ ] **Step 2: Run test — expect fail**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue pytest tests/test_new_dataset_iss_detection.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `iss_detection.py`**

Create `gluefactory/datasets/new_dataset/iss_detection.py`:

```python
"""ISS detection over (u, v, depth_scaled) PCD — same approach as
precompute_iss_fpfh_resample2.py:depth_crop_to_pcd, but:
  - no CROP (new dataset keeps full zmap)
  - no CLIP_START/CLIP_END (orthographic; raw uint16 used directly)
"""
from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d


def build_iss_pcd_uvd_scaled(zmap: np.ndarray, erode_boundary: int = 5):
    """Build a PCD in (u, v, depth_scaled) space for ISS detection.

    `depth_scaled = (raw - raw_min) * (u_range / raw_range)` so that XY
    and Z have similar scale (same idea as the existing Mitsubishi code).
    """
    mask = (zmap > 0).astype(np.uint8)
    if erode_boundary > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=erode_boundary)
    vs, us = np.where(mask > 0)
    if len(us) == 0:
        empty = o3d.geometry.PointCloud()
        return empty, mask, 1.0, 0.0

    us_f = us.astype(np.float64)
    vs_f = vs.astype(np.float64)
    raw = zmap[vs, us].astype(np.float64)

    u_range = max(us_f.max() - us_f.min(), 1.0)
    raw_min = float(raw.min())
    raw_range = float(raw.max() - raw.min()) if raw.max() > raw.min() else 1.0
    depth_scale = u_range / raw_range
    depth_scaled = (raw - raw_min) * depth_scale

    pts = np.stack([us_f, vs_f, depth_scaled], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd, mask, depth_scale, raw_min


def detect_iss_keypoints(pcd, gamma_21: float = 0.5, gamma_32: float = 0.5,
                         min_neighbors: int = 5) -> np.ndarray:
    """ISS keypoint detection. Hyperparameters match the existing Mitsubishi
    convention (salient_r = 6·nn_avg, non_max_r = 2·salient_r)."""
    if len(pcd.points) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    dists = pcd.compute_nearest_neighbor_distance()
    avg = float(np.mean(dists))
    salient_r = 6 * avg
    non_max_r = 2 * salient_r
    kp_pcd = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd,
        salient_radius=salient_r,
        non_max_radius=non_max_r,
        gamma_21=gamma_21,
        gamma_32=gamma_32,
        min_neighbors=min_neighbors,
    )
    return np.asarray(kp_pcd.points)  # (K, 3) in (u, v, depth_scaled)


def select_keypoints(iss_kp_3d: np.ndarray, zmap: np.ndarray,
                     max_num_keypoints: int = 512, erode_mask=None,
                     resize_factor: float = 0.5, rng=None):
    """Return (kp_resized(Nmax,2), scores(Nmax,), n_valid, kp_uv_orig(Nmax,2)).

    scores: ISS=1.0, random-fill=0.5, pad=0.0.
    kp_uv_orig are original-resolution (u,v) pixels. kp_resized = kp_uv_orig * resize_factor.
    If ISS >= Nmax, randomly subselect; otherwise ISS + random fill (prefer
    erode_mask; fall back to depth>0) up to Nmax; zero-pad if mask is too small.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_iss = int(len(iss_kp_3d))
    kp_uv_orig = np.zeros((max_num_keypoints, 2), dtype=np.float32)
    scores = np.zeros(max_num_keypoints, dtype=np.float32)

    if n_iss >= max_num_keypoints:
        idx = rng.choice(n_iss, max_num_keypoints, replace=False)
        sel = iss_kp_3d[idx, :2]
        kp_uv_orig[:] = sel.astype(np.float32)
        scores[:] = 1.0
        n_valid = max_num_keypoints
    else:
        if n_iss > 0:
            kp_uv_orig[:n_iss] = iss_kp_3d[:, :2].astype(np.float32)
            scores[:n_iss] = 1.0
        if erode_mask is not None:
            ys, xs = np.where(erode_mask > 0)
        else:
            ys, xs = np.where(zmap > 0)
        n_need = max_num_keypoints - n_iss
        n_rand = min(n_need, len(xs))
        if n_rand > 0:
            idx = rng.choice(len(xs), n_rand, replace=False)
            kp_uv_orig[n_iss:n_iss + n_rand, 0] = xs[idx].astype(np.float32)
            kp_uv_orig[n_iss:n_iss + n_rand, 1] = ys[idx].astype(np.float32)
            scores[n_iss:n_iss + n_rand] = 0.5
        n_valid = n_iss + n_rand

    kp_resized = kp_uv_orig * float(resize_factor)
    return kp_resized, scores, n_valid, kp_uv_orig
```

- [ ] **Step 4: Run test — expect pass**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue pytest tests/test_new_dataset_iss_detection.py -v`
Expected: PASS (5 tests, or 4 passed + 1 skipped without mount)

- [ ] **Step 5: Commit**

```bash
git add gluefactory/datasets/new_dataset/iss_detection.py tests/test_new_dataset_iss_detection.py
git commit -m "feat(new_dataset): iss_detection.py — ISS detect + keypoint selection"
```

---

## Task 5: `precompute_new_iss_fpfh.py` — FPFH cache generator

**Files:**
- Create: `precompute_new_iss_fpfh.py`

Produces `cache_new_iss_fpfh_r{radius}/{zmap_stem}.npz` under `gluefactory/datasets/new_dataset_cache/`. Cache root is a **sibling** of the package — no writes inside the package. Uses `coords.build_camera_frame_pcd` for the FPFH PCD, `iss_detection.*` for keypoint selection, and `open3d.pipelines.registration.compute_fpfh_feature` with `radius=fpfh_radius` / `normal_radius=fpfh_normal_radius`.

- [ ] **Step 1: Write the script**

Create `precompute_new_iss_fpfh.py`:

```python
"""Precompute ISS keypoints + dense FPFH descriptors for the new dataset.

ISS uses (u, v, depth_scaled) detection (package: iss_detection);
FPFH uses camera-frame XYZ PCD (package: coords.build_camera_frame_pcd).
Keypoint XYZ is looked up into the dense FPFH via Open3D KDTree.

Cache layout:
    gluefactory/datasets/new_dataset_cache/
      cache_new_iss_fpfh_r{radius}/
        zmap_0000.npz, zmap_0001.npz, ...
        precompute_config.json

npz fields:
    keypoints          (512, 2) float32   # resized (u·0.5, v·0.5)
    keypoint_scores    (512,)   float32   # 1.0 / 0.5 / 0.0
    descriptors        (512, 33) float32  # L2-normalized
    n_valid            int
    n_iss              int

Usage:
    conda activate LightGlue
    cd /home/jhs/work/Registration/glue-factory_depth
    python precompute_new_iss_fpfh.py                          # full (641)
    python precompute_new_iss_fpfh.py --max_images 3 --force   # smoke test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

from gluefactory.datasets.new_dataset import constants as C
from gluefactory.datasets.new_dataset.coords import build_camera_frame_pcd, pixel_to_cam_xyz
from gluefactory.datasets.new_dataset.iss_detection import (
    build_iss_pcd_uvd_scaled, detect_iss_keypoints, select_keypoints,
)


def compute_fpfh_for_keypoints(pcd_xyz: o3d.geometry.PointCloud,
                               kp_xyz: np.ndarray, n_valid: int,
                               fpfh_radius: float, fpfh_normal_radius: float,
                               fpfh_max_nn: int = 100,
                               normal_max_nn: int = 30,
                               anomaly_dist_mm: float = 1.0):
    max_n = kp_xyz.shape[0]
    out = np.zeros((max_n, 33), dtype=np.float32)
    n_anomaly = 0
    if n_valid < 3 or len(pcd_xyz.points) == 0:
        return out, n_anomaly

    pcd_xyz.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=fpfh_normal_radius, max_nn=normal_max_nn
        )
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_xyz,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=fpfh_max_nn),
    )
    fpfh_dense = np.array(fpfh.data).T.astype(np.float32)  # (M, 33)
    kdtree = o3d.geometry.KDTreeFlann(pcd_xyz)

    pcd_pts = np.asarray(pcd_xyz.points)
    for i in range(n_valid):
        q = kp_xyz[i]
        _, idx, _ = kdtree.search_knn_vector_3d(q, 1)
        j = idx[0]
        dist = float(np.linalg.norm(pcd_pts[j] - q))
        if dist > anomaly_dist_mm:
            n_anomaly += 1
            continue  # leave as zero descriptor
        out[i] = fpfh_dense[j]

    n = np.linalg.norm(out[:n_valid], axis=1, keepdims=True)
    out[:n_valid] = out[:n_valid] / (n + 1e-8)
    return out, n_anomaly


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=str(C.DEFAULT_DATA_ROOT))
    p.add_argument("--cache_root", type=str,
                   default="gluefactory/datasets/new_dataset_cache")
    p.add_argument("--max_num_keypoints", type=int, default=512)
    p.add_argument("--gamma_21", type=float, default=0.5)
    p.add_argument("--gamma_32", type=float, default=0.5)
    p.add_argument("--min_neighbors", type=int, default=5)
    p.add_argument("--erode_boundary", type=int, default=5)
    p.add_argument("--fpfh_radius", type=float, default=10.0, help="mm")
    p.add_argument("--fpfh_normal_radius", type=float, default=20.0, help="mm (2·fpfh_radius)")
    p.add_argument("--fpfh_max_nn", type=int, default=100)
    p.add_argument("--anomaly_dist_mm", type=float, default=1.0)
    p.add_argument("--resize_factor", type=float, default=C.RESIZE_FACTOR)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data_root = Path(args.data_root)
    cache_dir = (Path(args.cache_root) / f"cache_new_iss_fpfh_r{args.fpfh_radius}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(data_root.glob("zmap_*.png"))
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    print(f"Found {len(image_paths)} zmaps in {data_root}")
    print(f"Cache → {cache_dir}")
    print(f"FPFH: r={args.fpfh_radius}mm, normal_r={args.fpfh_normal_radius}mm")

    rng = np.random.default_rng(args.seed)
    n_iss_list, n_valid_list, n_anom_list = [], [], []

    for img_path in tqdm(image_paths, desc="Precomputing ISS+FPFH"):
        out_path = cache_dir / f"{img_path.stem}.npz"
        if out_path.exists() and not args.force:
            continue

        zmap = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if zmap is None:
            print(f"  [SKIP] cannot read {img_path.name}")
            continue

        pcd_iss, erode_mask, _, _ = build_iss_pcd_uvd_scaled(
            zmap, erode_boundary=args.erode_boundary
        )
        iss_kp_3d = detect_iss_keypoints(
            pcd_iss, gamma_21=args.gamma_21, gamma_32=args.gamma_32,
            min_neighbors=args.min_neighbors,
        )

        kp_resized, scores, n_valid, kp_uv_orig = select_keypoints(
            iss_kp_3d, zmap, max_num_keypoints=args.max_num_keypoints,
            erode_mask=erode_mask, resize_factor=args.resize_factor, rng=rng,
        )

        pcd_xyz, _ = build_camera_frame_pcd(zmap, erode_boundary=args.erode_boundary)
        # keypoint XYZ: look up the raw zmap value at rounded (u,v) (original-res)
        h, w = zmap.shape
        kp_xyz = np.zeros((args.max_num_keypoints, 3), dtype=np.float64)
        for i in range(n_valid):
            u = int(round(float(kp_uv_orig[i, 0])))
            v = int(round(float(kp_uv_orig[i, 1])))
            u = max(0, min(u, w - 1))
            v = max(0, min(v, h - 1))
            raw = zmap[v, u]
            kp_xyz[i] = pixel_to_cam_xyz(u, v, raw)

        desc, n_anom = compute_fpfh_for_keypoints(
            pcd_xyz, kp_xyz, n_valid,
            fpfh_radius=args.fpfh_radius,
            fpfh_normal_radius=args.fpfh_normal_radius,
            fpfh_max_nn=args.fpfh_max_nn,
            anomaly_dist_mm=args.anomaly_dist_mm,
        )

        np.savez_compressed(
            out_path,
            keypoints=kp_resized,
            keypoint_scores=scores,
            descriptors=desc,
            n_valid=np.array(n_valid, dtype=np.int64),
            n_iss=np.array(len(iss_kp_3d), dtype=np.int64),
        )
        n_iss_list.append(len(iss_kp_3d))
        n_valid_list.append(n_valid)
        n_anom_list.append(n_anom)

    if n_iss_list:
        ii = np.asarray(n_iss_list); vv = np.asarray(n_valid_list); aa = np.asarray(n_anom_list)
        print(f"\n--- Stats ---")
        print(f"ISS    : mean={ii.mean():.1f}, min={ii.min()}, max={ii.max()}")
        print(f"n_valid: mean={vv.mean():.1f}, min={vv.min()}, max={vv.max()}")
        print(f"anomaly kp (>1mm from nearest PCD pt): mean={aa.mean():.2f} / 512")

    with open(cache_dir / "precompute_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Done! {cache_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on scene 0..2**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue python precompute_new_iss_fpfh.py --max_images 3 --force`
Expected: 3 `.npz` files produced in `gluefactory/datasets/new_dataset_cache/cache_new_iss_fpfh_r10.0/`; stats line prints ISS mean > 100, n_valid mean ≈ 512, anomaly mean < 5.

- [ ] **Step 3: Verify cache shapes**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue python -c "import numpy as np; d=np.load('gluefactory/datasets/new_dataset_cache/cache_new_iss_fpfh_r10.0/zmap_0000.npz'); print({k: (d[k].shape, d[k].dtype) for k in d.files})"`
Expected: `keypoints (512,2) float32, keypoint_scores (512,) float32, descriptors (512,33) float32, n_valid () int64, n_iss () int64`

- [ ] **Step 4: Commit**

```bash
git add precompute_new_iss_fpfh.py
git commit -m "feat(new_dataset): precompute_new_iss_fpfh.py — ISS+FPFH cache generator"
```

---

## Task 6: `dataset.py` — `NewDatasetISSDescDataset` + dynamic-pad collate

**Files:**
- Create: `gluefactory/datasets/new_dataset/dataset.py`
- Modify: `gluefactory/datasets/new_dataset/__init__.py` (wire public exports)
- Create: `tests/test_new_dataset_integration.py`

`__getitem__` returns:
```
{
  "view0": {"image": (1, H0', W0'), "image_size": (H0', W0'),
            "keypoints": (512,2), "keypoint_scores": (512,), "descriptors": (512,D)},
  "view1": {...},
  "gt_matches": (M, 4)  float32   # master_x, master_y, input_x, input_y (resized)
  "csv_path": str, "master_path": str, "input_path": str,
}
```
Image stored as `float32 / 65535.0` (matches existing Mitsubishi convention).

`collate_fn_dynamic_pad`: W=1207 fixed across all samples in this dataset (2413·0.5 rounded); pad H to `max(image_size[:,0])`. For 1/2 INTER_NEAREST of (H, W) with W=2413 → W_resized = `cv2.resize(..., (W//2, H//2))` gives 1206. Use `cv2.resize(..., (round(W*0.5), round(H*0.5)), INTER_NEAREST)` and **also save both (H_resized, W_resized) in image_size**; then pad BOTH H and W to batch max so the collate handles all cases robustly.

- [ ] **Step 1: Write failing integration test**

Create `tests/test_new_dataset_integration.py`:

```python
import numpy as np
import pytest
import torch
from pathlib import Path

from gluefactory.datasets.new_dataset import constants as C
from gluefactory.datasets.new_dataset import (
    NewDatasetISSDescDataset, collate_fn_dynamic_pad,
)

DATA_ROOT = C.DEFAULT_DATA_ROOT
CACHE_ROOT = Path("gluefactory/datasets/new_dataset_cache/cache_new_iss_fpfh_r10.0")
HAS_ENV = DATA_ROOT.exists() and CACHE_ROOT.exists() and any(CACHE_ROOT.glob("zmap_*.npz"))


@pytest.mark.skipif(not HAS_ENV, reason="data or FPFH cache missing")
def test_dataset_getitem_shapes():
    ds = NewDatasetISSDescDataset(
        split="train", cache_dir=str(CACHE_ROOT),
        data_root=str(DATA_ROOT), resize_factor=0.5, val_ratio=0.05, seed=0,
    )
    assert len(ds) > 0
    sample = ds[0]
    assert {"view0", "view1", "gt_matches", "csv_path",
            "master_path", "input_path"} <= set(sample.keys())
    for v in ("view0", "view1"):
        assert sample[v]["image"].ndim == 3 and sample[v]["image"].shape[0] == 1
        assert sample[v]["keypoints"].shape == (512, 2)
        assert sample[v]["keypoint_scores"].shape == (512,)
        assert sample[v]["descriptors"].shape == (512, 33)
        H, W = sample[v]["image_size"].tolist()
        assert 700 <= H <= 1500 and W == 1207   # H half of 1556..2975, W half of 2413


@pytest.mark.skipif(not HAS_ENV, reason="data or FPFH cache missing")
def test_collate_dynamic_pad_two_samples_diff_height():
    ds = NewDatasetISSDescDataset(
        split="train", cache_dir=str(CACHE_ROOT), data_root=str(DATA_ROOT),
        resize_factor=0.5, val_ratio=0.05, seed=0,
    )
    # Grab two samples; at least some pairs will differ in H.
    a, b = ds[0], ds[min(len(ds) - 1, 100)]
    batch = collate_fn_dynamic_pad([a, b])
    assert batch["view0"]["image"].shape[0] == 2
    assert batch["view0"]["image"].shape[1] == 1   # channel
    # Padded to max H, max W across both samples
    max_h = max(a["view0"]["image_size"][0].item(), b["view0"]["image_size"][0].item())
    max_w = max(a["view0"]["image_size"][1].item(), b["view0"]["image_size"][1].item())
    assert batch["view0"]["image"].shape[2] == max_h
    assert batch["view0"]["image"].shape[3] == max_w
    # image_size per-sample preserved
    assert batch["view0"]["image_size"].shape == (2, 2)
```

- [ ] **Step 2: Run test — expect fail**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue pytest tests/test_new_dataset_integration.py -v`
Expected: FAIL with import error from `gluefactory.datasets.new_dataset`.

- [ ] **Step 3: Implement `dataset.py`**

Create `gluefactory/datasets/new_dataset/dataset.py`:

```python
"""PyTorch Dataset + dynamic-pad collate for the new dataset.

Uses precomputed ISS+descriptor caches. GT comes from pair_####.csv.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import constants as C
from .split import filter_pairs_by_scenes, scene_split


class NewDatasetISSDescDataset(Dataset):
    """Returns per-pair dict; see module docstring for shape spec."""

    def __init__(self, split: str, cache_dir: str | Path,
                 data_root: str | Path | None = None,
                 resize_factor: float = C.RESIZE_FACTOR,
                 val_ratio: float = 0.05, seed: int = 0):
        self.data_root = Path(data_root) if data_root else C.DEFAULT_DATA_ROOT
        self.cache_dir = Path(cache_dir)
        self.resize_factor = float(resize_factor)

        if not self.cache_dir.exists() or not any(self.cache_dir.glob("zmap_*.npz")):
            raise FileNotFoundError(
                f"Cache not found or empty: {self.cache_dir}\n"
                f"Run: python precompute_new_iss_fpfh.py (or _shot)"
            )

        combo = pd.read_csv(self.data_root / "combination.csv")
        train_scenes, val_scenes = scene_split(
            n_scenes=C.N_SCENES, val_ratio=val_ratio, seed=seed
        )
        if split == "train":
            combo = filter_pairs_by_scenes(combo, set(train_scenes))
        elif split == "val":
            combo = filter_pairs_by_scenes(combo, set(val_scenes))
        else:
            raise ValueError(f"Unknown split: {split!r} (expected 'train' or 'val')")
        self.combo = combo.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.combo)

    def _load_cache(self, zmap_filename: str) -> dict:
        stem = Path(zmap_filename).stem
        npz = np.load(self.cache_dir / f"{stem}.npz")
        return {
            "keypoints": npz["keypoints"],
            "keypoint_scores": npz["keypoint_scores"],
            "descriptors": npz["descriptors"],
        }

    def _load_resized_zmap(self, zmap_filename: str) -> np.ndarray:
        path = self.data_root / zmap_filename
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        H, W = img.shape
        new_w = int(round(W * self.resize_factor))
        new_h = int(round(H * self.resize_factor))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return resized.astype(np.float32) / 65535.0

    def __getitem__(self, idx: int) -> dict:
        row = self.combo.iloc[idx]
        master_fname = row["master_zmap_path"]
        input_fname = row["input_zmap_path"]
        csv_path = self.data_root / row["csv_path"]

        m_img = self._load_resized_zmap(master_fname)
        i_img = self._load_resized_zmap(input_fname)

        c0 = self._load_cache(master_fname)
        c1 = self._load_cache(input_fname)

        corr = pd.read_csv(csv_path)
        valid = corr["occluded"].astype(int) == 0
        mxy = corr.loc[valid, ["master_x", "master_y"]].values.astype(np.float32)
        ixy = corr.loc[valid, ["input_x", "input_y"]].values.astype(np.float32)
        if self.resize_factor != 1.0:
            mxy *= self.resize_factor
            ixy *= self.resize_factor

        H_m, W_m = m_img.shape
        H_i, W_i = i_img.shape
        in_bounds = (
            (mxy[:, 0] >= 0) & (mxy[:, 0] < W_m) & (mxy[:, 1] >= 0) & (mxy[:, 1] < H_m) &
            (ixy[:, 0] >= 0) & (ixy[:, 0] < W_i) & (ixy[:, 1] >= 0) & (ixy[:, 1] < H_i)
        )
        mxy = mxy[in_bounds]; ixy = ixy[in_bounds]
        gt = np.concatenate([mxy, ixy], axis=1) if len(mxy) > 0 \
             else np.zeros((0, 4), dtype=np.float32)

        return {
            "view0": {
                "image": torch.from_numpy(m_img).unsqueeze(0),
                "image_size": torch.tensor([H_m, W_m], dtype=torch.long),
                "keypoints": torch.from_numpy(c0["keypoints"]).float(),
                "keypoint_scores": torch.from_numpy(c0["keypoint_scores"]).float(),
                "descriptors": torch.from_numpy(c0["descriptors"]).float(),
            },
            "view1": {
                "image": torch.from_numpy(i_img).unsqueeze(0),
                "image_size": torch.tensor([H_i, W_i], dtype=torch.long),
                "keypoints": torch.from_numpy(c1["keypoints"]).float(),
                "keypoint_scores": torch.from_numpy(c1["keypoint_scores"]).float(),
                "descriptors": torch.from_numpy(c1["descriptors"]).float(),
            },
            "gt_matches": torch.from_numpy(gt).float(),
            "csv_path": str(csv_path),
            "master_path": str(self.data_root / master_fname),
            "input_path": str(self.data_root / input_fname),
        }


def _pad_view(views: list[dict], max_h: int, max_w: int) -> dict:
    B = len(views)
    imgs = torch.zeros(B, 1, max_h, max_w, dtype=views[0]["image"].dtype)
    sizes = torch.zeros(B, 2, dtype=torch.long)
    for b, v in enumerate(views):
        h, w = v["image"].shape[-2:]
        imgs[b, :, :h, :w] = v["image"]
        sizes[b] = v["image_size"]
    kp = torch.stack([v["keypoints"] for v in views], dim=0)
    sc = torch.stack([v["keypoint_scores"] for v in views], dim=0)
    dc = torch.stack([v["descriptors"] for v in views], dim=0)
    return {"image": imgs, "image_size": sizes,
            "keypoints": kp, "keypoint_scores": sc, "descriptors": dc}


def collate_fn_dynamic_pad(batch: list[dict]) -> dict:
    """Zero-pad H and W to batch max for each view independently."""
    v0 = [b["view0"] for b in batch]
    v1 = [b["view1"] for b in batch]
    max_h0 = max(v["image"].shape[-2] for v in v0)
    max_w0 = max(v["image"].shape[-1] for v in v0)
    max_h1 = max(v["image"].shape[-2] for v in v1)
    max_w1 = max(v["image"].shape[-1] for v in v1)

    from torch.nn.utils.rnn import pad_sequence
    gt_list = [b["gt_matches"] for b in batch]
    gt = pad_sequence(gt_list, batch_first=True, padding_value=0.0)

    return {
        "view0": _pad_view(v0, max_h0, max_w0),
        "view1": _pad_view(v1, max_h1, max_w1),
        "gt_matches": gt,
        "csv_path": [b["csv_path"] for b in batch],
        "master_path": [b["master_path"] for b in batch],
        "input_path": [b["input_path"] for b in batch],
    }
```

- [ ] **Step 4: Update `__init__.py` to export the real public interface**

Replace `gluefactory/datasets/new_dataset/__init__.py` with:

```python
"""New dataset: ISS + FPFH/SHOT + LightGlue.

Public interface:
    from gluefactory.datasets.new_dataset import (
        NewDatasetISSDescDataset, collate_fn_dynamic_pad,
    )
"""

from .dataset import NewDatasetISSDescDataset, collate_fn_dynamic_pad

__all__ = ["NewDatasetISSDescDataset", "collate_fn_dynamic_pad"]
```

- [ ] **Step 5: Run integration test — expect pass**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue pytest tests/test_new_dataset_integration.py -v`
Expected: PASS (2 tests, or skipped if no cache/data).

- [ ] **Step 6: Run ALL new-dataset tests**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue pytest tests/test_new_dataset_*.py -v`
Expected: all pass (or appropriate skips if data unavailable).

- [ ] **Step 7: Commit**

```bash
git add gluefactory/datasets/new_dataset/dataset.py gluefactory/datasets/new_dataset/__init__.py tests/test_new_dataset_integration.py
git commit -m "feat(new_dataset): Dataset + dynamic-pad collate, integration test"
```

---

## Task 7: FPFH config + training script

**Files:**
- Create: `gluefactory/configs/0413_new_iss_fpfh_lg.yaml`
- Create: `gluefactory/train_new_iss_desc.py`

- [ ] **Step 1: Write the FPFH config**

Create `gluefactory/configs/0413_new_iss_fpfh_lg.yaml`:

```yaml
data:
    name: new_dataset_iss_desc
    descriptor_type: fpfh          # fpfh | shot
    fpfh_radius: 10.0
    resize_factor: 0.5
    val_ratio: 0.05
    seed: 0
    batch_size: 32
    num_workers: 12
model:
    name: two_view_pipeline
    filter_zero_depth: true
    extractor:
        name: extractors.superpoint_fpfh_cached
        descriptor_dim: 33
        trainable: False
    ground_truth:
        name: matchers.gt_pair_matcher
        gt_radius: 20
    matcher:
        name: matchers.lightglue
        input_dim: 33
        descriptor_dim: 36
        num_heads: 3
        filter_threshold: 0.1
        flash: false
        checkpointed: true
train:
    seed: 0
    epochs: 100
    best_key: match_recall
    best_key_mode: max
    log_every_iter: 1
    eval_every_iter: 500
    lr: 1.0e-4
    lr_schedule:
        start: 20
        type: exp
        on_epoch: true
        exp_div_10: 10
    plot: [5, 'gluefactory.visualization.visualize_batch.make_match_figures_depth']
```

- [ ] **Step 2: Create training script (fork of train_resample2_iss_fpfh.py)**

Create `gluefactory/train_new_iss_desc.py`. This is a near-verbatim copy of `gluefactory/train_resample2_iss_fpfh.py`; only the dataset imports and loader construction change. The diff is entirely in the `training(rank, conf, output_dir, args)` function's data-loader section.

Copy `gluefactory/train_resample2_iss_fpfh.py` first, then change 3 things:

1. **Imports** — replace:
```python
from .datasets.mitsubishi_resample2_iss_fpfh_dataset import MitsubishiResample2ISSFPFHDataset
from .datasets.mitsubishi_resample2_iss_fpfh_dataset import resample2_iss_fpfh_collate_fn
```
with:
```python
from .datasets.new_dataset import NewDatasetISSDescDataset, collate_fn_dynamic_pad
from .datasets.new_dataset import constants as NEW_C
```

2. **Dataset construction inside `training()`** — replace the block beginning `image_size = conf.data.get("image_size", None)` through the two `DataLoader(...)` calls with:

```python
descriptor_type = conf.data.get("descriptor_type", "fpfh")
if descriptor_type == "fpfh":
    fpfh_radius = conf.data.get("fpfh_radius", 10.0)
    cache_dir = Path("gluefactory/datasets/new_dataset_cache") / f"cache_new_iss_fpfh_r{fpfh_radius}"
elif descriptor_type == "shot":
    cache_dir = Path("gluefactory/datasets/new_dataset_cache") / "cache_new_iss_shot352"
else:
    raise ValueError(f"Unknown descriptor_type: {descriptor_type!r}")

resize_factor = conf.data.get("resize_factor", NEW_C.RESIZE_FACTOR)
val_ratio = conf.data.get("val_ratio", 0.05)
seed_split = conf.data.get("seed", 0)
batch_size = conf.data.get("batch_size", 4)
num_workers = conf.data.get("num_workers", 4)

dataset = NewDatasetISSDescDataset(
    split="train", cache_dir=cache_dir,
    resize_factor=resize_factor, val_ratio=val_ratio, seed=seed_split,
)
val_dataset = NewDatasetISSDescDataset(
    split="val", cache_dir=cache_dir,
    resize_factor=resize_factor, val_ratio=val_ratio, seed=seed_split,
)

train_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    collate_fn=collate_fn_dynamic_pad, pin_memory=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
    collate_fn=collate_fn_dynamic_pad, pin_memory=True,
)
```

3. **Add `from pathlib import Path`** at the top (next to existing `from pathlib import Path` line — if already present, skip).

All other lines (signal handler, optimizer, training loop, checkpointing, evaluation, CLI) stay identical. Full file must reference `__module_name__`, `logger`, `settings` from `gluefactory` just like the existing script.

Concrete procedure:

```bash
cp gluefactory/train_resample2_iss_fpfh.py gluefactory/train_new_iss_desc.py
```

Then edit `gluefactory/train_new_iss_desc.py` to apply the 3 changes above. Verify:

```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda run -n LightGlue python -c "import importlib; importlib.import_module('gluefactory.train_new_iss_desc')"
```
Expected: no error.

- [ ] **Step 3: Dry-run config parsing (no training)**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue python -c "from omegaconf import OmegaConf; c=OmegaConf.load('gluefactory/configs/0413_new_iss_fpfh_lg.yaml'); print(c.data.descriptor_type, c.model.matcher.input_dim, c.train.epochs)"`
Expected: `fpfh 33 100`

- [ ] **Step 4: Commit**

```bash
git add gluefactory/configs/0413_new_iss_fpfh_lg.yaml gluefactory/train_new_iss_desc.py
git commit -m "feat(new_dataset): FPFH config + training script fork"
```

---

## Task 8: FPFH shell launcher + 1-iteration smoke training

**Files:**
- Create: `train_new_iss_fpfh.sh`

- [ ] **Step 1: Write launcher**

Create `train_new_iss_fpfh.sh` (`chmod +x` after write):

```bash
#!/bin/bash
# ISS(detector) + FPFH(descriptor) + LG training on new dataset.
# Auto-builds FPFH cache if missing.

GPU_ID=${1:-0}
EXPERIMENT=${2:-"0413_new_iss_fpfh_lg"}
FPFH_RADIUS=${3:-10.0}
BATCH_SIZE=${4:-32}
GT_RADIUS=${5:-20}
RESTORE=${6:-""}
CONF="gluefactory/configs/0413_new_iss_fpfh_lg.yaml"

RESTORE_FLAG=""
if [ "$RESTORE" = "--restore" ]; then
    RESTORE_FLAG="--restore"
    echo "=== RESUME training from last checkpoint ==="
fi

CACHE_DIR="gluefactory/datasets/new_dataset_cache/cache_new_iss_fpfh_r${FPFH_RADIUS}"
if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A $CACHE_DIR/*.npz 2>/dev/null)" ]; then
    echo "=== FPFH cache not found. Precomputing... ==="
    python3 precompute_new_iss_fpfh.py --fpfh_radius "$FPFH_RADIUS"
fi

mkdir -p "outputs/training/${EXPERIMENT}"

echo "=== Training new_dataset ISS+FPFH+LG (fpfh_r=${FPFH_RADIUS}, batch=${BATCH_SIZE}, gt_radius=${GT_RADIUS}) ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m gluefactory.train_new_iss_desc "$EXPERIMENT" \
    --conf "$CONF" \
    $RESTORE_FLAG \
    --mixed_precision float16 \
    data.fpfh_radius="$FPFH_RADIUS" \
    data.batch_size="$BATCH_SIZE" \
    model.ground_truth.gt_radius="$GT_RADIUS" \
    2>&1 | tee "outputs/training/${EXPERIMENT}/train.log"
```

- [ ] **Step 2: Smoke training — 1 iteration, 2 scenes of cache**

Requires ≥3 cached scenes (built in Task 5 step 2). Override `epochs=1` and `eval_every_iter` via CLI dotlist to end quickly. Run from a shell with GPU available.

Run: `cd /home/jhs/work/Registration/glue-factory_depth && chmod +x train_new_iss_fpfh.sh && CUDA_VISIBLE_DEVICES=0 conda run -n LightGlue python3 -m gluefactory.train_new_iss_desc smoke_0413_new_iss_fpfh --conf gluefactory/configs/0413_new_iss_fpfh_lg.yaml --mixed_precision float16 data.batch_size=2 data.num_workers=2 train.epochs=1 train.eval_every_iter=100000 train.save_every_iter=100000`

Expected: logs first iteration `loss/total=...`, no `ModuleNotFoundError`, no NaN. Kill (`Ctrl-C`) after ~3 iterations.

(If cache only has 3 scenes, the dataloader sees few pairs — that's fine for smoke.)

- [ ] **Step 3: Commit**

```bash
git add train_new_iss_fpfh.sh
git commit -m "feat(new_dataset): FPFH launcher script"
```

---

## Task 9: SHOT precompute + config + launcher (no training yet)

**Files:**
- Create: `precompute_new_iss_shot.py`
- Create: `gluefactory/configs/0413_new_iss_shot_lg.yaml`
- Create: `train_new_iss_shot.sh`

SHOT C++ bins arrive separately; the rest of the pipeline (dataset, trainer) is descriptor-agnostic.

- [ ] **Step 1: Write SHOT precompute script**

Create `precompute_new_iss_shot.py`:

```python
"""Precompute ISS keypoints + SHOT352 descriptors (from C++ bins) for the new dataset.

Pipeline:
  1. ISS detection in (u, v, depth_scaled) — same as FPFH
  2. Keypoint XYZ in camera frame (mm) via pixel_to_cam_xyz
  3. KDTree lookup into C++ cloud (camera frame mm): X=u·0.056, Y=v·0.056, Z=raw·0.0085

C++ bin name: `<zmap_stem>_shot352.bin` (e.g., zmap_0042_shot352.bin).
Bin format: uint32 N, uint32 D=352, per-point [f32 x, y, z, f32[352]].
NaN descriptors are filtered during load.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from gluefactory.datasets.new_dataset import constants as C
from gluefactory.datasets.new_dataset.coords import pixel_to_cam_xyz
from gluefactory.datasets.new_dataset.iss_detection import (
    build_iss_pcd_uvd_scaled, detect_iss_keypoints, select_keypoints,
)

SHOT_DIM = 352


def load_shot352_bin(bin_path: Path):
    with open(bin_path, "rb") as f:
        N, D = np.frombuffer(f.read(8), dtype=np.uint32)
        N, D = int(N), int(D)
        assert D == SHOT_DIM, f"Expected D={SHOT_DIM}, got {D} in {bin_path}"
        data = np.frombuffer(f.read(N * (3 + D) * 4), dtype=np.float32)
    data = data.reshape(N, 3 + D)
    pts = data[:, :3]
    desc = data[:, 3:]
    valid = ~np.isnan(desc).any(axis=1)
    return pts[valid], desc[valid]


def lookup_shot352(pts_cloud: np.ndarray, desc_cloud: np.ndarray,
                   kp_xyz: np.ndarray, n_valid: int):
    out = np.zeros((kp_xyz.shape[0], SHOT_DIM), dtype=np.float32)
    if n_valid < 1 or len(pts_cloud) < 1:
        return out
    tree = cKDTree(pts_cloud)
    _, idxs = tree.query(kp_xyz[:n_valid])
    out[:n_valid] = desc_cloud[idxs]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=str(C.DEFAULT_DATA_ROOT))
    p.add_argument("--bin_dir", type=str, required=True,
                   help="Directory with <zmap_stem>_shot352.bin files")
    p.add_argument("--cache_root", type=str,
                   default="gluefactory/datasets/new_dataset_cache")
    p.add_argument("--max_num_keypoints", type=int, default=512)
    p.add_argument("--gamma_21", type=float, default=0.5)
    p.add_argument("--gamma_32", type=float, default=0.5)
    p.add_argument("--min_neighbors", type=int, default=5)
    p.add_argument("--erode_boundary", type=int, default=5)
    p.add_argument("--resize_factor", type=float, default=C.RESIZE_FACTOR)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data_root = Path(args.data_root)
    bin_dir = Path(args.bin_dir)
    cache_dir = Path(args.cache_root) / "cache_new_iss_shot352"
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(data_root.glob("zmap_*.png"))
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    print(f"Found {len(image_paths)} zmaps, bins in {bin_dir}, cache → {cache_dir}")

    rng = np.random.default_rng(args.seed)
    n_iss_list, n_valid_list, n_skip = [], [], 0

    for img_path in tqdm(image_paths, desc="Precomputing ISS+SHOT352"):
        out_path = cache_dir / f"{img_path.stem}.npz"
        if out_path.exists() and not args.force:
            continue

        bin_path = bin_dir / f"{img_path.stem}_shot352.bin"
        if not bin_path.exists():
            n_skip += 1
            continue
        pts_cloud, desc_cloud = load_shot352_bin(bin_path)

        zmap = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        pcd_iss, erode_mask, _, _ = build_iss_pcd_uvd_scaled(
            zmap, erode_boundary=args.erode_boundary
        )
        iss_kp_3d = detect_iss_keypoints(
            pcd_iss, gamma_21=args.gamma_21, gamma_32=args.gamma_32,
            min_neighbors=args.min_neighbors,
        )
        kp_resized, scores, n_valid, kp_uv_orig = select_keypoints(
            iss_kp_3d, zmap, max_num_keypoints=args.max_num_keypoints,
            erode_mask=erode_mask, resize_factor=args.resize_factor, rng=rng,
        )

        h, w = zmap.shape
        kp_xyz = np.zeros((args.max_num_keypoints, 3), dtype=np.float64)
        for i in range(n_valid):
            u = int(round(float(kp_uv_orig[i, 0])))
            v = int(round(float(kp_uv_orig[i, 1])))
            u = max(0, min(u, w - 1)); v = max(0, min(v, h - 1))
            kp_xyz[i] = pixel_to_cam_xyz(u, v, zmap[v, u])

        desc = lookup_shot352(pts_cloud, desc_cloud, kp_xyz, n_valid)

        np.savez_compressed(
            out_path,
            keypoints=kp_resized,
            keypoint_scores=scores,
            descriptors=desc,
            n_valid=np.array(n_valid, dtype=np.int64),
            n_iss=np.array(len(iss_kp_3d), dtype=np.int64),
        )
        n_iss_list.append(len(iss_kp_3d))
        n_valid_list.append(n_valid)

    if n_iss_list:
        ii = np.asarray(n_iss_list); vv = np.asarray(n_valid_list)
        print(f"\n--- Stats ---")
        print(f"ISS    : mean={ii.mean():.1f}, min={ii.min()}, max={ii.max()}")
        print(f"n_valid: mean={vv.mean():.1f}, min={vv.min()}, max={vv.max()}")
        print(f"Skipped (bin missing): {n_skip}")

    with open(cache_dir / "precompute_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Done! {cache_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write SHOT config**

Create `gluefactory/configs/0413_new_iss_shot_lg.yaml` (only differences from FPFH config: `descriptor_type`, `input_dim` 352, no `fpfh_radius`):

```yaml
data:
    name: new_dataset_iss_desc
    descriptor_type: shot
    resize_factor: 0.5
    val_ratio: 0.05
    seed: 0
    batch_size: 32
    num_workers: 12
model:
    name: two_view_pipeline
    filter_zero_depth: true
    extractor:
        name: extractors.superpoint_fpfh_cached
        descriptor_dim: 352
        trainable: False
    ground_truth:
        name: matchers.gt_pair_matcher
        gt_radius: 20
    matcher:
        name: matchers.lightglue
        input_dim: 352
        descriptor_dim: 36
        num_heads: 3
        filter_threshold: 0.1
        flash: false
        checkpointed: true
train:
    seed: 0
    epochs: 100
    best_key: match_recall
    best_key_mode: max
    log_every_iter: 1
    eval_every_iter: 500
    lr: 1.0e-4
    lr_schedule:
        start: 20
        type: exp
        on_epoch: true
        exp_div_10: 10
    plot: [5, 'gluefactory.visualization.visualize_batch.make_match_figures_depth']
```

- [ ] **Step 3: Write SHOT launcher**

Create `train_new_iss_shot.sh` (`chmod +x` after write):

```bash
#!/bin/bash
# ISS(detector) + SHOT352(descriptor) + LG training on new dataset.
# Requires C++ SHOT bins already generated by the external tool.

GPU_ID=${1:-0}
EXPERIMENT=${2:-"0413_new_iss_shot_lg"}
BIN_DIR=${3:-"gluefactory/datasets/new_dataset_shot_bins"}
BATCH_SIZE=${4:-32}
GT_RADIUS=${5:-20}
RESTORE=${6:-""}
CONF="gluefactory/configs/0413_new_iss_shot_lg.yaml"

RESTORE_FLAG=""
if [ "$RESTORE" = "--restore" ]; then
    RESTORE_FLAG="--restore"
    echo "=== RESUME training from last checkpoint ==="
fi

CACHE_DIR="gluefactory/datasets/new_dataset_cache/cache_new_iss_shot352"
if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A $CACHE_DIR/*.npz 2>/dev/null)" ]; then
    echo "=== SHOT cache not found. Precomputing from bins in $BIN_DIR ... ==="
    python3 precompute_new_iss_shot.py --bin_dir "$BIN_DIR"
fi

mkdir -p "outputs/training/${EXPERIMENT}"

echo "=== Training new_dataset ISS+SHOT+LG (batch=${BATCH_SIZE}, gt_radius=${GT_RADIUS}) ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m gluefactory.train_new_iss_desc "$EXPERIMENT" \
    --conf "$CONF" \
    $RESTORE_FLAG \
    --mixed_precision float16 \
    data.batch_size="$BATCH_SIZE" \
    model.ground_truth.gt_radius="$GT_RADIUS" \
    2>&1 | tee "outputs/training/${EXPERIMENT}/train.log"
```

- [ ] **Step 4: Syntax-check SHOT precompute**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue python -c "import importlib.util as u; s=u.spec_from_file_location('p','precompute_new_iss_shot.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print('ok')"`
Expected: `ok` (no import error). Full run will be deferred until C++ bins exist.

- [ ] **Step 5: Commit**

```bash
git add precompute_new_iss_shot.py gluefactory/configs/0413_new_iss_shot_lg.yaml train_new_iss_shot.sh
git commit -m "feat(new_dataset): SHOT precompute + config + launcher (awaits C++ bins)"
```

---

## Task 10: Match visualization (test_new_iss_desc.py)

**Files:**
- Create: `test_new_iss_desc.py`

Uses 4-color convention (skyblue=correct / purple=wrong / limegreen=occluded / red=no-GT), mirrored from `test_resample2_iss_fpfh_0406.py` but without CROP offsets.

- [ ] **Step 1: Write visualization script**

Create `test_new_iss_desc.py`:

```python
"""Visualize trained ISS+FPFH/SHOT+LG matches on the new dataset.

4-color convention (same as Mitsubishi):
  skyblue  = correct match (master-side within gt_radius AND input-side within gt_radius, not occluded)
  purple   = wrong         (master-side matched, input-side too far)
  limegreen= occluded      (master-side matched, input-side close, BUT occluded=1)
  red      = no-GT         (no master-side GT keypoint within gt_radius)

Usage:
    python test_new_iss_desc.py --experiment 0413_new_iss_fpfh_lg --num_samples 10
    python test_new_iss_desc.py --checkpoint outputs/training/0413_new_iss_fpfh_lg/checkpoint_best.tar --indices 0 5 10
    python test_new_iss_desc.py --experiment 0413_new_iss_shot_lg --descriptor_type shot --num_samples 5
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.datasets.new_dataset import (
    NewDatasetISSDescDataset, collate_fn_dynamic_pad,
)
from gluefactory.datasets.new_dataset import constants as NEW_C
from gluefactory.utils.tensor import batch_to_device


def load_model(checkpoint_path: str, device: str):
    cp = torch.load(checkpoint_path, map_location="cpu")
    conf = OmegaConf.create(cp["conf"])
    model = get_model(conf.model.name)(conf.model).to(device)
    model.load_state_dict(cp["model"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {cp['epoch']})")
    return model, conf


@torch.no_grad()
def run_inference(model, batch, device):
    batch = batch_to_device(batch, device)
    pred = model(batch)
    return pred, batch


def visualize_pair(pred, data, idx: int, output_path: Path,
                   csv_path: str | None = None, gt_radius: int = 20,
                   resize_factor: float = 0.5):
    img0 = data["view0"]["image"][idx].cpu()
    img1 = data["view1"]["image"][idx].cpu()
    h0, w0 = data["view0"]["image_size"][idx].tolist()
    h1, w1 = data["view1"]["image_size"][idx].tolist()
    # crop off padding for display
    img0_np = img0.squeeze(0).numpy()[:h0, :w0]
    img1_np = img1.squeeze(0).numpy()[:h1, :w1]

    kp0 = pred["keypoints0"][idx].cpu().numpy()
    kp1 = pred["keypoints1"][idx].cpu().numpy()
    m0 = pred["matches0"][idx].cpu().numpy()

    valid = m0 > -1
    mkp0 = kp0[valid]
    mkp1 = kp1[m0[valid]]
    n_total = int(valid.sum())

    n_correct = n_wrong = n_occluded = n_no_gt = 0
    gt_pos_total = 0
    colors = ["red"] * n_total

    if csv_path is not None and Path(csv_path).exists():
        corr = pd.read_csv(csv_path)
        all_m = corr[["master_x", "master_y"]].values.astype(np.float32) * resize_factor
        all_i = corr[["input_x", "input_y"]].values.astype(np.float32) * resize_factor
        all_occ = (corr["occluded"].astype(int) == 1).values
        gt_pos_total = int((~all_occ).sum())

        for i in range(n_total):
            dists = np.linalg.norm(all_m - mkp0[i], axis=1)
            j = int(np.argmin(dists))
            if dists[j] < gt_radius:
                d_input = float(np.linalg.norm(mkp1[i] - all_i[j]))
                is_occ = bool(all_occ[j])
                if d_input < gt_radius:
                    if is_occ:
                        colors[i] = "limegreen"; n_occluded += 1
                    else:
                        colors[i] = "skyblue"; n_correct += 1
                else:
                    colors[i] = "purple"; n_wrong += 1
            else:
                n_no_gt += 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
    for ax, img_np, kp, title in [
        (axes[0], img0_np, kp0, "View 0 (Master)"),
        (axes[1], img1_np, kp1, "View 1 (Input)"),
    ]:
        ax.imshow(img_np, cmap="gray")
        ax.scatter(kp[:, 0], kp[:, 1], c="royalblue", s=3, alpha=0.3, linewidths=0)
        ax.set_title(title, fontsize=14); ax.set_axis_off()

    for i in range(n_total):
        line = matplotlib.patches.ConnectionPatch(
            xyA=(float(mkp0[i, 0]), float(mkp0[i, 1])),
            xyB=(float(mkp1[i, 0]), float(mkp1[i, 1])),
            coordsA=axes[0].transData, coordsB=axes[1].transData,
            axesA=axes[0], axesB=axes[1],
            color=colors[i], linewidth=0.8, alpha=0.6,
        )
        fig.add_artist(line)

    info = f"Keypoints: {kp0.shape[0]} / {kp1.shape[0]}  |  matches: {n_total}"
    if csv_path is not None:
        recall = n_correct / max(gt_pos_total, 1) * 100
        info += (f"  |  Recall: {n_correct}/{gt_pos_total} ({recall:.1f}%)"
                 f"  |  Wrong: {n_wrong}  |  Occluded: {n_occluded}  |  No-GT: {n_no_gt}")
    fig.suptitle(info, fontsize=10, y=0.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close(fig)
    print(f"  {output_path.name}  |  matches={n_total}  correct={n_correct}/{gt_pos_total}"
          f"  wrong={n_wrong}  occluded={n_occluded}  no-GT={n_no_gt}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--experiment", type=str, default="0413_new_iss_fpfh_lg")
    p.add_argument("--descriptor_type", type=str, default="fpfh", choices=["fpfh", "shot"])
    p.add_argument("--fpfh_radius", type=float, default=10.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--indices", type=int, nargs="*", default=None)
    p.add_argument("--gt_radius", type=int, default=20)
    p.add_argument("--resize_factor", type=float, default=NEW_C.RESIZE_FACTOR)
    args = p.parse_args()

    output_dir = Path(args.output_dir if args.output_dir
                      else f"results/{args.experiment}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.checkpoint is None:
        from gluefactory.utils.experiments import get_best_checkpoint
        cp_path = str(get_best_checkpoint(args.experiment))
        print(f"Auto-loaded best checkpoint: {cp_path}")
    else:
        cp_path = args.checkpoint

    model, _ = load_model(cp_path, device)

    if args.descriptor_type == "fpfh":
        cache_dir = Path("gluefactory/datasets/new_dataset_cache") / f"cache_new_iss_fpfh_r{args.fpfh_radius}"
    else:
        cache_dir = Path("gluefactory/datasets/new_dataset_cache") / "cache_new_iss_shot352"

    dataset = NewDatasetISSDescDataset(
        split=args.split, cache_dir=cache_dir,
        resize_factor=args.resize_factor, val_ratio=0.05, seed=0,
    )
    print(f"{args.split} dataset: {len(dataset)} pairs")

    if args.indices is not None:
        indices = list(args.indices)
    else:
        rng = np.random.default_rng(0)
        indices = sorted(rng.choice(
            len(dataset), min(args.num_samples, len(dataset)), replace=False
        ).tolist())

    print(f"Testing {len(indices)} pairs: {indices}")
    for idx in indices:
        sample = dataset[idx]
        batch = collate_fn_dynamic_pad([sample])
        pred, batch = run_inference(model, batch, device)
        csv_path = batch.get("csv_path", [None])[0]
        out = output_dir / f"{args.split}_{idx:05d}.png"
        visualize_pair(pred, batch, 0, out, csv_path=csv_path,
                       gt_radius=args.gt_radius, resize_factor=args.resize_factor)
    print(f"\nDone! → {output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue python -c "import importlib.util as u; s=u.spec_from_file_location('p','test_new_iss_desc.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add test_new_iss_desc.py
git commit -m "feat(new_dataset): match visualization script (4-color overlay)"
```

---

## Task 11: Registration RMSE evaluation skeleton

**Files:**
- Create: `eval_registration_new_iss_desc.py`

Skeleton only — follow-up plan fleshes out Phase-5 analysis. The skeleton:
- loads a trained checkpoint
- iterates val pairs
- extracts predicted master/input matches (resized pixels)
- inverse-resizes back to original (u,v)
- maps each kp through `pixel_to_cam_xyz` → `cam_to_world_xyz` using scene YAML
- runs a custom SVD RANSAC (same function name used elsewhere: `ransac_rigid`) → `T_est`
- reads `camera_rt` of master/input → `T_gt_rel = T_input_world⁻¹ · T_master_world`
- prints per-pair RMSE and aggregate mean

- [ ] **Step 1: Write the skeleton**

Create `eval_registration_new_iss_desc.py`:

```python
"""Registration RMSE evaluation for new_dataset ISS+FPFH/SHOT+LG models.

Skeleton: wires up data loading, inference, coordinate conversion, and a
placeholder RANSAC+SVD call. Refinement (sampling strategy, threshold
tuning, forward/reverse RMSE computation) happens in a follow-up plan.

Usage (skeleton run, limited samples):
    python eval_registration_new_iss_desc.py --experiment 0413_new_iss_fpfh_lg \
        --descriptor_type fpfh --num_samples 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.models import get_model
from gluefactory.datasets.new_dataset import (
    NewDatasetISSDescDataset, collate_fn_dynamic_pad,
)
from gluefactory.datasets.new_dataset import constants as NEW_C
from gluefactory.datasets.new_dataset.coords import (
    cam_to_world_xyz, load_scene_config, pixel_to_cam_xyz,
)
from gluefactory.datasets.new_dataset.split import pair_filename_to_scene_ids
from gluefactory.utils.tensor import batch_to_device


def scene_world_transform(cfg) -> np.ndarray:
    """4x4 world-from-camera transform."""
    T = np.eye(4)
    T[:3, :3] = cfg.R_cam
    T[:3, 3] = cfg.t_cam
    return T


def rigid_transform_svd(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """src → dst. Returns 4x4 transform. Minimum 3 points."""
    assert src.shape == dst.shape and src.shape[0] >= 3
    sc = src.mean(0); dc = dst.mean(0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = dc - R @ sc
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def ransac_rigid(src: np.ndarray, dst: np.ndarray, n_iter: int = 1000,
                 inlier_th: float = 5.0, min_samples: int = 3,
                 rng=None) -> tuple[np.ndarray, np.ndarray]:
    """Vanilla SVD RANSAC. Returns (T 4x4, inlier_mask)."""
    if rng is None:
        rng = np.random.default_rng(0)
    N = src.shape[0]
    if N < min_samples:
        return np.eye(4), np.zeros(N, dtype=bool)
    best_T = np.eye(4); best_inl = np.zeros(N, dtype=bool); best_cnt = -1
    for _ in range(n_iter):
        idx = rng.choice(N, min_samples, replace=False)
        T = rigid_transform_svd(src[idx], dst[idx])
        proj = (src @ T[:3, :3].T) + T[:3, 3]
        d = np.linalg.norm(proj - dst, axis=1)
        inl = d < inlier_th
        if inl.sum() > best_cnt:
            best_cnt = int(inl.sum()); best_T = T; best_inl = inl
    if best_cnt >= min_samples:
        best_T = rigid_transform_svd(src[best_inl], dst[best_inl])
    return best_T, best_inl


def transform_rmse(T_a: np.ndarray, T_b: np.ndarray, n_samples: int = 30000,
                   extent_mm: float = 100.0, rng=None) -> float:
    """Symmetric RMSE: sample random points, apply T_a vs T_b, measure mm L2."""
    if rng is None:
        rng = np.random.default_rng(0)
    pts = (rng.random((n_samples, 3)) - 0.5) * (2 * extent_mm)
    a = pts @ T_a[:3, :3].T + T_a[:3, 3]
    b = pts @ T_b[:3, :3].T + T_b[:3, 3]
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


@torch.no_grad()
def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.checkpoint:
        cp_path = args.checkpoint
    else:
        from gluefactory.utils.experiments import get_best_checkpoint
        cp_path = str(get_best_checkpoint(args.experiment))
    cp = torch.load(cp_path, map_location="cpu")
    conf = OmegaConf.create(cp["conf"])
    model = get_model(conf.model.name)(conf.model).to(device)
    model.load_state_dict(cp["model"], strict=False)
    model.eval()

    if args.descriptor_type == "fpfh":
        cache_dir = Path("gluefactory/datasets/new_dataset_cache") / f"cache_new_iss_fpfh_r{args.fpfh_radius}"
    else:
        cache_dir = Path("gluefactory/datasets/new_dataset_cache") / "cache_new_iss_shot352"

    ds = NewDatasetISSDescDataset(
        split=args.split, cache_dir=cache_dir,
        resize_factor=args.resize_factor, val_ratio=0.05, seed=0,
    )
    rng = np.random.default_rng(0)
    idxs = list(range(min(args.num_samples, len(ds))))
    rmses = []

    for idx in tqdm(idxs, desc="Evaluating"):
        sample = ds[idx]
        batch = collate_fn_dynamic_pad([sample])
        batch = batch_to_device(batch, device)
        pred = model(batch)

        kp0 = pred["keypoints0"][0].cpu().numpy()
        kp1 = pred["keypoints1"][0].cpu().numpy()
        m0 = pred["matches0"][0].cpu().numpy()
        valid = m0 > -1
        if valid.sum() < 3:
            continue
        mkp0 = kp0[valid] / args.resize_factor  # back to original pixels
        mkp1 = kp1[m0[valid]] / args.resize_factor

        master_fname = Path(sample["master_path"]).name
        input_fname = Path(sample["input_path"]).name
        csv_fname = Path(sample["csv_path"]).name
        m_id, i_id = pair_filename_to_scene_ids(csv_fname)

        data_root = Path(sample["master_path"]).parent
        cfg_m = load_scene_config(m_id, data_root)
        cfg_i = load_scene_config(i_id, data_root)
        import cv2
        zmap_m = cv2.imread(sample["master_path"], cv2.IMREAD_UNCHANGED)
        zmap_i = cv2.imread(sample["input_path"], cv2.IMREAD_UNCHANGED)

        def kp_to_world(kp_uv, zmap, cfg):
            u = np.round(kp_uv[:, 0]).astype(int)
            v = np.round(kp_uv[:, 1]).astype(int)
            u = np.clip(u, 0, zmap.shape[1] - 1)
            v = np.clip(v, 0, zmap.shape[0] - 1)
            raw = zmap[v, u]
            cam = pixel_to_cam_xyz(u, v, raw)
            keep = raw > 0
            return cam_to_world_xyz(cam, cfg), keep

        w0, ok0 = kp_to_world(mkp0, zmap_m, cfg_m)
        w1, ok1 = kp_to_world(mkp1, zmap_i, cfg_i)
        ok = ok0 & ok1
        if ok.sum() < 3:
            continue
        T_est, inl = ransac_rigid(w0[ok], w1[ok], n_iter=args.ransac_iter,
                                  inlier_th=args.ransac_th, rng=rng)

        Tm = scene_world_transform(cfg_m)
        Ti = scene_world_transform(cfg_i)
        # T_gt maps master-world → input-world (same object frame reference)
        T_gt = Ti @ np.linalg.inv(Tm)

        rmse = transform_rmse(T_est, T_gt, rng=rng)
        rmses.append(rmse)
        print(f"  pair_{m_id:04d}_{i_id:04d}: inliers={int(inl.sum())}/{int(ok.sum())} "
              f"RMSE={rmse:.3f} mm")

    if rmses:
        arr = np.asarray(rmses)
        print(f"\n=== Summary over {len(rmses)} pairs ===")
        print(f"mean RMSE: {arr.mean():.3f} mm")
        print(f"median   : {np.median(arr):.3f} mm")
        print(f"max      : {arr.max():.3f} mm")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--experiment", type=str, default="0413_new_iss_fpfh_lg")
    p.add_argument("--descriptor_type", type=str, default="fpfh", choices=["fpfh", "shot"])
    p.add_argument("--fpfh_radius", type=float, default=10.0)
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--resize_factor", type=float, default=NEW_C.RESIZE_FACTOR)
    p.add_argument("--ransac_iter", type=int, default=1000)
    p.add_argument("--ransac_th", type=float, default=5.0)
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue python -c "import importlib.util as u; s=u.spec_from_file_location('p','eval_registration_new_iss_desc.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add eval_registration_new_iss_desc.py
git commit -m "feat(new_dataset): registration RMSE eval skeleton"
```

---

## Task 12: Full FPFH precompute + Phase-3 mini training

This is the validation phase. No new code — just running pipelines and checking spec §11 Phase 0–4 criteria.

- [ ] **Step 1: Full FPFH precompute**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue python precompute_new_iss_fpfh.py 2>&1 | tee logs/precompute_fpfh.log`
Expected: 641 `.npz` files in `gluefactory/datasets/new_dataset_cache/cache_new_iss_fpfh_r10.0/`. `n_valid == 512` for ≥95% of scenes (spec §15). Print stats at end.

If `n_valid == 512` ratio < 95%: investigate low-ISS scenes (zmap stats, erode aggressiveness). Adjust `--erode_boundary` or `--min_neighbors` and re-run.

- [ ] **Step 2: Phase-3 mini training**

Choose 10 scenes + restrict combo at runtime by creating a temp combination.csv sliced to first 100 pairs from those scenes — OR just run `epochs=2` on the full train split and watch loss.

Pragmatic path — 2 epochs on full split, small batch, fast eval:

Run:
```
cd /home/jhs/work/Registration/glue-factory_depth && mkdir -p outputs/training/mini_0413_new_iss_fpfh && CUDA_VISIBLE_DEVICES=0 conda run -n LightGlue python3 -m gluefactory.train_new_iss_desc mini_0413_new_iss_fpfh --conf gluefactory/configs/0413_new_iss_fpfh_lg.yaml --mixed_precision float16 data.batch_size=4 data.num_workers=4 train.epochs=2 train.eval_every_iter=500 2>&1 | tee outputs/training/mini_0413_new_iss_fpfh/train.log
```
Expected after 2 epochs: loss trends down, `val match_recall` logged (may still be low but should be non-NaN). If `positive pair ratio` is outside 5–20% (inspect `gt_pair_matcher` logs / `make_match_figures_depth`), tune `gt_radius`.

- [ ] **Step 3: Visualize 5 mini pairs**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && conda run -n LightGlue python test_new_iss_desc.py --experiment mini_0413_new_iss_fpfh --num_samples 5`
Expected: `results/mini_0413_new_iss_fpfh/val_*.png` exist, overlays rendered.

- [ ] **Step 4: Decide on full run / parameter tweaks**

Review mini-training log + overlays. If sane (loss monotonic down, recall > 0.05 at epoch-2), kick off full training:

Run: `cd /home/jhs/work/Registration/glue-factory_depth && ./train_new_iss_fpfh.sh 0 0413_new_iss_fpfh_lg 10.0 32 20` (long-running; monitor `outputs/training/0413_new_iss_fpfh_lg/train.log`).

Success checkpoint: val `match_recall > 0.1` after epoch 1 (spec §15). Final target fixed after smoke-test observation.

- [ ] **Step 5: Commit any config adjustments discovered in this phase**

If no changes, skip. Otherwise:
```bash
git add <changed files>
git commit -m "chore(new_dataset): tune hyperparameters from Phase-3 smoke results"
```

---

## Self-Review (write-plans step 7, done inline)

**1. Spec coverage:**
- §1 Goals → Tasks 5, 7, 8, 9, 10, 11 (precompute + train + test + eval).
- §2 Dataset analysis → Tasks 2 (`coords`), 3 (split).
- §3 Decisions → all. FPFH/SHOT unified (Tasks 7 + 9), ISS re-detection (Task 4), gt_radius (configs), 1/2 resize (constants + Task 6), dynamic padding (Task 6), scene-based split (Task 3), camera-frame standard (Task 2 + Tasks 5/9), new package (Task 1).
- §4 Hyperparameters → all baked into config YAML + precompute CLI defaults.
- §5 Data flow → Tasks 4 (detect), 5 (FPFH precompute), 9 (SHOT precompute), 6 (online), 11 (registration).
- §6 Directory structure → matches Tasks 1, 2, 3, 4, 5, 6, 7, 9, 10, 11 exactly.
- §7 Module interfaces → Tasks 2, 3, 4, 6 implement every signature.
- §8 Cache format → Tasks 5 + 9; npz fields match.
- §9 SHOT bin format → Task 9 parses `<stem>_shot352.bin`, camera-frame mm convention.
- §10 Edge cases → random fill (Task 4), zero-pad (Task 4), FileNotFoundError (Task 6), `occluded == 0` filter (Task 6), in_bounds filter (Task 6), FPFH anomaly (Task 5), reproducibility (Task 3), SKIP on bin missing (Task 9).
- §11 Validation phases → Tasks 5 step 2 (Phase 0), Task 6 (Phase 1), Task 8 step 2 (Phase 2), Task 12 steps 2–4 (Phase 3–4). Phase 5 (SHOT) is Task 9 — precompute smoke once bins arrive.
- §12 Tests → Tasks 1, 2, 3, 4, 6 each add a test file.
- §13 Rollback → `new_dataset/` is a fresh package; no Mitsubishi file modified; verified by the task list (no "Modify" entries for existing gluefactory files).
- §14 Paths → configs use `0413_new_iss_{fpfh,shot}_lg`, outputs auto-created by `train_new_iss_desc.py` and `test_new_iss_desc.py` under `outputs/training/...` and `results/...`.
- §15 Success criteria → Task 12 steps 1, 2, 4 gate on these.

**2. Placeholder scan:** No `TODO`, no "implement later", no "similar to Task N". Every code step contains complete code.

**3. Type consistency:**
- `select_keypoints` signature consistent between Task 4 definition, Task 5 caller, and Task 9 caller: `(iss_kp_3d, zmap, max_num_keypoints, erode_mask, resize_factor, rng) → (kp_resized, scores, n_valid, kp_uv_orig)`.
- `build_iss_pcd_uvd_scaled` returns `(pcd, mask, depth_scale, raw_min)` — 4-tuple; both callers unpack 4 values.
- `build_camera_frame_pcd` returns `(pcd, mask)` — 2-tuple; Task 5 unpacks `pcd_xyz, _`.
- `SceneConfig` fields used by Task 11: `R_cam`, `t_cam` — match Task 2 dataclass.
- `NewDatasetISSDescDataset(split, cache_dir, data_root, resize_factor, val_ratio, seed)` — Task 6 definition, Task 10/11 callers use same kwargs.
- `collate_fn_dynamic_pad` — same name everywhere.
- Cache directory naming: `cache_new_iss_fpfh_r{radius}` (Task 5 + config + launcher + test + eval) and `cache_new_iss_shot352` (Task 9 + config + launcher + test + eval) — consistent.

No gaps found. Plan is ready.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-13-new-dataset-iss-desc-lg.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task; review between tasks; fastest iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans; batched checkpoints.

Which approach?
