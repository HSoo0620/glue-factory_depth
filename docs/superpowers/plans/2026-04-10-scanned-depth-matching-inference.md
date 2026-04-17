# Scanned Depth Matching Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **User preference:** The user commits personally and asks to confirm before each code/config file modification. Agents executing this plan must obtain user confirmation before each Edit/Write and must NOT run `git commit`. Steps marked **[commit checkpoint]** are suggested moments for the user to create a commit; the agent should pause and report progress instead of committing.

**Goal:** Build a single-file inference script that runs the trained `0406_resample2_iss_fpfh_xyz_lg` model on real 3D line-scanner data (`scanned_data1.png`) against two master viewpoints (indices 0 and 594), with two experiments (raw vs dz-corrected), producing matching + registration visualizations and a proxy overlap CSV.

**Architecture:** One new standalone script `test_scanned_vs_master.py` (~350 lines) that imports existing feature extraction (`precompute_iss_fpfh_resample2.py`), registration (`visualize_registration_flann.py`), and model loading (`test_resample2_iss_fpfh_0406.py`) functions without modifying any existing file. Scanned features are computed at runtime (no cache), master features are loaded from the existing `iss_fpfh_resample2_cache_r5.0_xyz/` cache. Two experiments (`raw`, `dzfix`) are run in a single invocation via `--run_both`.

**Tech Stack:** Python 3, numpy, cv2, open3d, torch, pandas, matplotlib, scipy.spatial.cKDTree — all already present in the `LightGlue` conda environment.

**Spec:** `docs/superpowers/specs/2026-04-10-scanned-depth-matching-inference-design.md`

**Working directory:** `/home/jhs/work/Registration/glue-factory_depth/`

**Conda environment:** `LightGlue` (activate via `conda activate LightGlue`)

---

## File Structure

### New files (create)
| File | Responsibility |
|---|---|
| `test_scanned_vs_master.py` | Single entrypoint: preprocessing, feature extraction wrapper, master loading, batch building, inference, registration, visualization, CSV writing |
| `tests/test_scanned_preprocessor.py` | unittest for `preprocess_scanned` — shape, dtype, pad positioning, dz scaling invariants |
| `tests/test_overlap_rate.py` | unittest for `compute_overlap_rate` — synthetic point clouds with known overlap |

### Existing files (import only, NOT modified)
| File | Imported symbols |
|---|---|
| `precompute_iss_fpfh_resample2.py` | `depth_crop_to_pcd`, `build_xyz_pcd`, `extract_iss_keypoints`, `select_keypoints`, `kp_crop_to_xyz`, `compute_fpfh_for_keypoints`, `CROP_X0`, `CROP_Y0`, `CROP_SIZE`, `CLIP_START`, `CLIP_END` |
| `visualize_registration_flann.py` | `pixel_to_grid3d`, `rigid_transform_svd`, `ransac_rigid`, `sample_point_cloud`, `GRID_DX`, `GRID_DY` |
| `test_resample2_iss_fpfh_0406.py` | `load_model`, `run_inference` |
| `gluefactory/datasets/mitsubishi_resample2_iss_fpfh_dataset.py` | `resample2_iss_fpfh_collate_fn` |
| `gluefactory/utils/tensor.py` | `batch_to_device` |

### New output directories (created at runtime by script)
```
results/scanned/scanned_data1/
├── summary.csv
├── raw/          # experiment B: scanned raw as-is
│   ├── master0000_match.png
│   ├── master0000_reg.png
│   ├── master0594_match.png
│   └── master0594_reg.png
└── dzfix/        # experiment C: scanned raw × 0.425
    ├── master0000_match.png
    ├── master0000_reg.png
    ├── master0594_match.png
    └── master0594_reg.png
```

---

## Task 1: Script skeleton + CLI + imports

**Files:**
- Create: `test_scanned_vs_master.py`

- [ ] **Step 1.1: Create the skeleton file**

Write `test_scanned_vs_master.py` with imports, CLI argparse, and an empty `main()` function:

```python
"""
Inference script for matching real 3D line-scanner data against master depth
using the trained ISS+FPFH+LightGlue model.

Usage:
    python test_scanned_vs_master.py --run_both
    python test_scanned_vs_master.py --master_indices 0 594 --scale_dz
    python test_scanned_vs_master.py --help

Spec: docs/superpowers/specs/2026-04-10-scanned-depth-matching-inference-design.md
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from precompute_iss_fpfh_resample2 import (
    depth_crop_to_pcd,
    build_xyz_pcd,
    extract_iss_keypoints,
    select_keypoints,
    kp_crop_to_xyz,
    compute_fpfh_for_keypoints,
    CROP_X0,
    CROP_Y0,
    CROP_SIZE,
    CLIP_START,
    CLIP_END,
)
from visualize_registration_flann import (
    pixel_to_grid3d,
    rigid_transform_svd,
    ransac_rigid,
    sample_point_cloud,
    GRID_DX,
    GRID_DY,
)
from test_resample2_iss_fpfh_0406 import load_model, run_inference
from gluefactory.datasets.mitsubishi_resample2_iss_fpfh_dataset import (
    resample2_iss_fpfh_collate_fn,
)
from gluefactory.utils.tensor import batch_to_device


# Scanned data spacing (mm)
SCANNED_DX = 0.056
SCANNED_DY = 0.056
SCANNED_DZ = 0.0085
# Master data spacing (mm)
MASTER_DZ = 0.02
# Default dz correction ratio (scanned_dz / master_dz)
DEFAULT_DZ_RATIO = SCANNED_DZ / MASTER_DZ  # ≈ 0.425


def parse_args():
    p = argparse.ArgumentParser(
        description="Scanned depth inference against master viewpoints."
    )
    p.add_argument("--scanned_path", type=str,
                   default="gluefactory/datasets/scanned/scanned_data1.png")
    p.add_argument("--master_indices", type=int, nargs="+", default=[0, 594])
    p.add_argument("--checkpoint", type=str,
                   default="outputs/training/0406_resample2_iss_fpfh_xyz_lg/checkpoint_best.tar")
    p.add_argument("--cache_dir", type=str,
                   default="gluefactory/datasets/mitsubishi/iss_fpfh_resample2_cache_r5.0_xyz")
    p.add_argument("--master_img_dir", type=str,
                   default="gluefactory/datasets/mitsubishi/dataset_resample_2")
    p.add_argument("--output_dir", type=str,
                   default="results/scanned/scanned_data1")
    p.add_argument("--fpfh_radius", type=float, default=5.0)
    p.add_argument("--image_size", type=int, default=1751)
    p.add_argument("--max_num_keypoints", type=int, default=512)
    p.add_argument("--dz_ratio", type=float, default=DEFAULT_DZ_RATIO)
    p.add_argument("--run_both", action="store_true",
                   help="Run both raw and dzfix experiments in one invocation.")
    p.add_argument("--scale_dz", action="store_true",
                   help="Run only the dzfix experiment (ignored if --run_both is set).")
    p.add_argument("--ransac_iter", type=int, default=1000)
    p.add_argument("--inlier_th", type=float, default=5.0)
    p.add_argument("--overlap_threshold", type=float, default=1.0,
                   help="Grid-space distance for proxy overlap metric.")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Args: {vars(args)}")
    # Phases to be implemented in subsequent tasks
    raise NotImplementedError("main() not yet implemented")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.2: Verify the script parses and help works**

Run:
```bash
conda activate LightGlue && cd /home/jhs/work/Registration/glue-factory_depth && python test_scanned_vs_master.py --help
```

Expected: argparse help text prints without ImportError. If imports fail, fix the specific import path before continuing.

- [ ] **Step 1.3: Verify all symbols imported without error**

Run:
```bash
conda activate LightGlue && cd /home/jhs/work/Registration/glue-factory_depth && python -c "
from test_scanned_vs_master import (
    SCANNED_DX, SCANNED_DY, SCANNED_DZ, MASTER_DZ, DEFAULT_DZ_RATIO,
    depth_crop_to_pcd, build_xyz_pcd, extract_iss_keypoints,
    select_keypoints, kp_crop_to_xyz, compute_fpfh_for_keypoints,
    pixel_to_grid3d, rigid_transform_svd, ransac_rigid, sample_point_cloud,
    load_model, run_inference, resample2_iss_fpfh_collate_fn, batch_to_device,
)
print('All imports OK')
print(f'DEFAULT_DZ_RATIO = {DEFAULT_DZ_RATIO}')
"
```

Expected:
```
All imports OK
DEFAULT_DZ_RATIO = 0.425
```

- [ ] **Step 1.4 [commit checkpoint]**: Pause for user to commit. Suggested message: `feat(scanned): add script skeleton with CLI and imports`

---

## Task 2: `preprocess_scanned` + unit test

**Files:**
- Modify: `test_scanned_vs_master.py` (add function)
- Create: `tests/test_scanned_preprocessor.py`

- [ ] **Step 2.1: Write the failing unit test**

Create `tests/test_scanned_preprocessor.py`:

```python
"""Unit tests for preprocess_scanned — shape, dtype, pad positioning, dz scaling."""
import unittest
import tempfile
import numpy as np
import cv2
from pathlib import Path


def _write_synthetic_scanned(path, height=3515, width=2687, fill_value=29334):
    """Create a synthetic uint16 depth PNG with a bright rectangular 'object'."""
    img = np.zeros((height, width), dtype=np.uint16)
    # Center rectangle "object"
    img[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = fill_value
    cv2.imwrite(str(path), img)


class TestPreprocessScanned(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.scanned_path = Path(self.tmpdir.name) / "synthetic.png"
        _write_synthetic_scanned(self.scanned_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_output_shapes_and_dtypes(self):
        from test_scanned_vs_master import preprocess_scanned
        crop_raw, img_small, meta = preprocess_scanned(
            self.scanned_path, scale_dz=False
        )
        self.assertEqual(crop_raw.shape, (3502, 3502))
        self.assertEqual(crop_raw.dtype, np.uint16)
        self.assertEqual(img_small.shape, (1751, 1751))
        self.assertEqual(img_small.dtype, np.float32)

    def test_center_padding_positioning(self):
        from test_scanned_vs_master import preprocess_scanned
        crop_raw, _, meta = preprocess_scanned(self.scanned_path, scale_dz=False)
        # Left/right borders should be zero-padded after aspect-preserving scale
        self.assertTrue(meta["pad_left"] > 0)
        # Scaled content width should be less than canvas width
        self.assertLess(meta["scaled_shape"][1], 3502)
        # Scaled content height should equal canvas height
        self.assertEqual(meta["scaled_shape"][0], 3502)
        # Padded regions on the far left and far right are zero
        self.assertTrue(np.all(crop_raw[:, :10] == 0))
        self.assertTrue(np.all(crop_raw[:, -10:] == 0))

    def test_scale_factor(self):
        from test_scanned_vs_master import preprocess_scanned
        crop_raw, _, meta = preprocess_scanned(self.scanned_path, scale_dz=False)
        # 3502 / max(3515, 2687) = 3502 / 3515 ≈ 0.9963
        self.assertAlmostEqual(meta["scale_factor"], 3502 / 3515, places=4)

    def test_dz_scaling_applies_only_when_flag_set(self):
        from test_scanned_vs_master import preprocess_scanned
        crop_raw_raw, _, meta_raw = preprocess_scanned(
            self.scanned_path, scale_dz=False
        )
        crop_raw_fix, _, meta_fix = preprocess_scanned(
            self.scanned_path, scale_dz=True, dz_ratio=0.425
        )
        self.assertFalse(meta_raw["dz_scaled"])
        self.assertTrue(meta_fix["dz_scaled"])
        # Non-zero pixels in dzfix should be ~0.425× of the raw ones
        mask = crop_raw_raw > 0
        raw_vals = crop_raw_raw[mask].astype(np.float64)
        fix_vals = crop_raw_fix[mask].astype(np.float64)
        ratios = fix_vals / raw_vals
        self.assertTrue(np.all(np.abs(ratios - 0.425) < 0.01))
        # Zero pixels remain zero
        self.assertTrue(np.all(crop_raw_fix[~mask] == 0))

    def test_img_small_is_normalized(self):
        from test_scanned_vs_master import preprocess_scanned
        _, img_small, _ = preprocess_scanned(self.scanned_path, scale_dz=False)
        self.assertLessEqual(img_small.max(), 1.0)
        self.assertGreaterEqual(img_small.min(), 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2.2: Run the test to verify it fails**

Run:
```bash
conda activate LightGlue && cd /home/jhs/work/Registration/glue-factory_depth && python -m unittest tests.test_scanned_preprocessor -v 2>&1
```

Expected: `ImportError` or `AttributeError: module 'test_scanned_vs_master' has no attribute 'preprocess_scanned'`.

- [ ] **Step 2.3: Implement `preprocess_scanned`**

Add this function to `test_scanned_vs_master.py` before `main()`:

```python
def preprocess_scanned(
    scanned_png_path,
    scale_dz=False,
    dz_ratio=DEFAULT_DZ_RATIO,
    target_crop_size=3502,
    target_image_size=1751,
):
    """Load scanned PNG and produce crop + small image tensors.

    Method D: aspect-preserving scale so max dimension equals target_crop_size,
    then center-pad into a square canvas. Uses INTER_NEAREST throughout to
    preserve depth edge values.

    Args:
        scanned_png_path: path to uint16 depth PNG.
        scale_dz: if True, multiply non-zero raw values by dz_ratio.
        dz_ratio: scanned_dz / master_dz, default 0.425.
        target_crop_size: canvas side length, default 3502.
        target_image_size: small image side length, default 1751.

    Returns:
        scanned_crop_raw: (target_crop_size, target_crop_size) uint16 canvas
        scanned_img_small: (target_image_size, target_image_size) float32 normalized
        meta: dict with orig_shape, scaled_shape, pad_left, pad_top, scale_factor, dz_scaled
    """
    img = cv2.imread(str(scanned_png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read scanned image: {scanned_png_path}")
    if img.dtype != np.uint16:
        print(f"WARNING: scanned dtype is {img.dtype}, forcing uint16")
        img = img.astype(np.uint16)

    orig_h, orig_w = img.shape[:2]
    scale_factor = target_crop_size / max(orig_h, orig_w)
    new_h = int(round(orig_h * scale_factor))
    new_w = int(round(orig_w * scale_factor))
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((target_crop_size, target_crop_size), dtype=np.uint16)
    pad_top = (target_crop_size - new_h) // 2
    pad_left = (target_crop_size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = scaled

    if scale_dz:
        mask = canvas > 0
        scaled_vals = (canvas.astype(np.float64) * dz_ratio).clip(0, 65535)
        canvas = np.where(mask, scaled_vals.astype(np.uint16), np.uint16(0))

    # Sanity check: warn if too few valid pixels
    n_valid = int((canvas > 0).sum())
    if n_valid < 1000:
        print(f"WARNING: only {n_valid} valid pixels in scanned crop")

    small = cv2.resize(
        canvas, (target_image_size, target_image_size),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.float32) / 65535.0

    meta = {
        "orig_shape": (orig_h, orig_w),
        "scaled_shape": (new_h, new_w),
        "pad_top": pad_top,
        "pad_left": pad_left,
        "scale_factor": scale_factor,
        "dz_scaled": scale_dz,
    }
    return canvas, small, meta
```

- [ ] **Step 2.4: Run the test to verify it passes**

Run:
```bash
conda activate LightGlue && cd /home/jhs/work/Registration/glue-factory_depth && python -m unittest tests.test_scanned_preprocessor -v 2>&1
```

Expected: All 5 tests pass.

- [ ] **Step 2.5 [commit checkpoint]**: Pause for user. Suggested message: `feat(scanned): add preprocess_scanned with method D resize+pad and unit tests`

---

## Task 3: `extract_scanned_features` wrapper + smoke test

**Files:**
- Modify: `test_scanned_vs_master.py` (add function)

- [ ] **Step 3.1: Implement `extract_scanned_features`**

Add after `preprocess_scanned`:

```python
def extract_scanned_features(
    scanned_crop_raw,
    fx=8001.39,
    fy=8001.39,
    cx=1751.0,
    cy=1751.0,
    fpfh_radius=5.0,
    max_num_keypoints=512,
    image_size=1751,
    erode_boundary=5,
):
    """Runtime ISS+FPFH extraction on scanned crop.

    Uses the same sequence as precompute_iss_fpfh_resample2.main() but
    operates on an in-memory scanned canvas instead of looping over disk files.

    Returns a dict with the same keys as the cache npz, plus kp_crop.
    """
    pcd_iss, _, depth_scale, d_min, erode_mask = depth_crop_to_pcd(
        scanned_crop_raw, erode_boundary=erode_boundary
    )

    iss_kp_3d = extract_iss_keypoints(pcd_iss)
    n_iss = len(iss_kp_3d)
    print(f"  ISS keypoints: {n_iss}")

    keypoints, scores, n_valid, kp_crop, kp_3d = select_keypoints(
        iss_kp_3d, scanned_crop_raw, max_num_keypoints, image_size,
        depth_scale=depth_scale, depth_min=d_min, erode_mask=erode_mask,
    )

    pcd_xyz = build_xyz_pcd(
        scanned_crop_raw, fx, fy, cx, cy, mask=erode_mask
    )
    kp_xyz = kp_crop_to_xyz(
        kp_crop, n_valid, scanned_crop_raw, fx, fy, cx, cy
    )
    fpfh = compute_fpfh_for_keypoints(
        pcd_xyz, kp_xyz, n_valid, fpfh_radius=fpfh_radius
    )

    return {
        "keypoints": keypoints,
        "keypoint_scores": scores,
        "fpfh_descriptors": fpfh,
        "n_valid": int(n_valid),
        "n_iss": int(n_iss),
        "kp_crop": kp_crop,
    }
```

- [ ] **Step 3.2: Write smoke test script inline**

Run to verify the function produces expected shapes on the real scanned file:

```bash
conda activate LightGlue && cd /home/jhs/work/Registration/glue-factory_depth && python -c "
from test_scanned_vs_master import preprocess_scanned, extract_scanned_features
from pathlib import Path
crop, _, meta = preprocess_scanned(Path('gluefactory/datasets/scanned/scanned_data1.png'))
print(f'crop shape: {crop.shape}, meta: {meta}')
feats = extract_scanned_features(crop)
print(f'keypoints shape: {feats[\"keypoints\"].shape}')
print(f'fpfh shape: {feats[\"fpfh_descriptors\"].shape}')
print(f'n_valid: {feats[\"n_valid\"]}, n_iss: {feats[\"n_iss\"]}')
print(f'fpfh norm (first valid): {(feats[\"fpfh_descriptors\"][0] ** 2).sum() ** 0.5:.4f}')
"
```

Expected output:
- `keypoints shape: (512, 2)`
- `fpfh shape: (512, 33)`
- `n_valid > 0` (likely hundreds)
- `n_iss > 0`
- FPFH norm should be ≈ 1.0 for valid entries (L2 normalized)

If `n_valid == 0` or ISS detection fails, investigate the scanned crop visually before proceeding.

- [ ] **Step 3.3 [commit checkpoint]**: Suggested: `feat(scanned): add extract_scanned_features runtime wrapper`

---

## Task 4: `load_master_features` and `build_batch`

**Files:**
- Modify: `test_scanned_vs_master.py`

- [ ] **Step 4.1: Implement `load_master_features`**

Add after `extract_scanned_features`:

```python
def load_master_features(master_idx, cache_dir, master_img_dir, image_size=1751):
    """Load cached master features and raw depth crop.

    The existing cache does NOT store kp_crop, so it's reconstructed as
    keypoints * (CROP_SIZE / image_size) = keypoints * 2 when image_size=1751.

    Returns:
        features: dict with keypoints, keypoint_scores, fpfh_descriptors,
                  n_valid, n_iss, kp_crop
        master_crop_raw: (CROP_SIZE, CROP_SIZE) uint16 crop of the master PNG
    """
    cache_dir = Path(cache_dir)
    master_img_dir = Path(master_img_dir)
    npz_path = cache_dir / f"depth_raw_{master_idx:04d}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Master cache not found: {npz_path}\n"
            f"Run: python precompute_iss_fpfh_resample2.py --fpfh_radius 5.0"
        )
    data = np.load(npz_path)
    n_valid = int(data["n_valid"])
    keypoints = data["keypoints"]

    # Reconstruct kp_crop: keypoints is in image_size scale, kp_crop is in CROP_SIZE scale
    scale = CROP_SIZE / image_size
    kp_crop = np.zeros_like(keypoints)
    kp_crop[:n_valid] = keypoints[:n_valid] * scale

    features = {
        "keypoints": keypoints,
        "keypoint_scores": data["keypoint_scores"],
        "fpfh_descriptors": data["fpfh_descriptors"],
        "n_valid": n_valid,
        "n_iss": int(data["n_iss"]),
        "kp_crop": kp_crop,
    }

    # Load the master image and apply fixed crop
    img_path = master_img_dir / f"depth_raw_{master_idx:04d}.png"
    img_raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise FileNotFoundError(f"Master image not found: {img_path}")
    master_crop_raw = img_raw[CROP_Y0:CROP_Y0 + CROP_SIZE,
                               CROP_X0:CROP_X0 + CROP_SIZE]
    return features, master_crop_raw
```

- [ ] **Step 4.2: Implement `build_batch`**

Add after `load_master_features`:

```python
def build_batch(
    master_feats, master_crop_raw,
    scanned_feats, scanned_crop_raw,
    image_size=1751,
    master_path="",
    scanned_path="",
):
    """Construct the batch dict expected by the trained model.

    Critical: the cache key 'fpfh_descriptors' must be renamed to 'descriptors'
    to match what the model forward pass reads (see
    mitsubishi_resample2_iss_fpfh_dataset.py:155-165).
    """
    # Downsample crops to image_size for the 'image' tensor
    master_small = cv2.resize(
        master_crop_raw, (image_size, image_size),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.float32) / 65535.0
    scanned_small = cv2.resize(
        scanned_crop_raw, (image_size, image_size),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.float32) / 65535.0

    def _view_dict(feats, small):
        return {
            "image": torch.from_numpy(small).unsqueeze(0),  # (1, H, W)
            "image_size": torch.tensor([image_size, image_size]),
            "keypoints": torch.from_numpy(feats["keypoints"]).float(),
            "keypoint_scores": torch.from_numpy(feats["keypoint_scores"]).float(),
            "descriptors": torch.from_numpy(feats["fpfh_descriptors"]).float(),
        }

    sample = {
        "view0": _view_dict(master_feats, master_small),
        "view1": _view_dict(scanned_feats, scanned_small),
        "gt_matches": torch.zeros((0, 4), dtype=torch.float32),
        "csv_path": "",
        "master_path": master_path,
        "input_path": scanned_path,
    }

    batch = resample2_iss_fpfh_collate_fn([sample])
    return batch, master_small, scanned_small
```

- [ ] **Step 4.3: Smoke test — inline script**

```bash
conda activate LightGlue && cd /home/jhs/work/Registration/glue-factory_depth && python -c "
from test_scanned_vs_master import (
    preprocess_scanned, extract_scanned_features,
    load_master_features, build_batch,
)
from pathlib import Path
crop, _, _ = preprocess_scanned(Path('gluefactory/datasets/scanned/scanned_data1.png'))
scanned_feats = extract_scanned_features(crop)
master_feats, master_crop = load_master_features(
    0,
    'gluefactory/datasets/mitsubishi/iss_fpfh_resample2_cache_r5.0_xyz',
    'gluefactory/datasets/mitsubishi/dataset_resample_2',
)
print(f'master n_valid: {master_feats[\"n_valid\"]}')
print(f'master_crop shape: {master_crop.shape}')
batch, ms, ss = build_batch(master_feats, master_crop, scanned_feats, crop)
print(f'batch view0 keys: {list(batch[\"view0\"].keys())}')
print(f'batch view0 descriptors shape: {batch[\"view0\"][\"descriptors\"].shape}')
print(f'batch view1 descriptors shape: {batch[\"view1\"][\"descriptors\"].shape}')
print(f'master_small: {ms.shape}, scanned_small: {ss.shape}')
"
```

Expected:
- `master n_valid` > 0
- `master_crop shape: (3502, 3502)`
- `batch view0 keys: ['image', 'image_size', 'keypoints', 'keypoint_scores', 'descriptors']`
- `descriptors shape: (1, 512, 33)` (batched)
- `master_small: (1751, 1751), scanned_small: (1751, 1751)`

- [ ] **Step 4.4 [commit checkpoint]**: Suggested: `feat(scanned): add master loader and batch builder`

---

## Task 5: `compute_overlap_rate` + unit test, and `estimate_registration`

**Files:**
- Create: `tests/test_overlap_rate.py`
- Modify: `test_scanned_vs_master.py`

- [ ] **Step 5.1: Write failing unit test**

Create `tests/test_overlap_rate.py`:

```python
"""Unit tests for compute_overlap_rate — synthetic point clouds with known overlap."""
import unittest
import numpy as np


class TestComputeOverlapRate(unittest.TestCase):
    def test_fully_overlapping_clouds(self):
        from test_scanned_vs_master import compute_overlap_rate
        pc = np.random.RandomState(0).randn(100, 3)
        rate = compute_overlap_rate(pc, pc, threshold=1e-6)
        self.assertAlmostEqual(rate, 1.0)

    def test_fully_disjoint_clouds(self):
        from test_scanned_vs_master import compute_overlap_rate
        rng = np.random.RandomState(0)
        a = rng.randn(100, 3)
        b = rng.randn(100, 3) + 1000.0
        rate = compute_overlap_rate(a, b, threshold=1.0)
        self.assertAlmostEqual(rate, 0.0)

    def test_half_overlap(self):
        from test_scanned_vs_master import compute_overlap_rate
        rng = np.random.RandomState(0)
        master = rng.randn(100, 3)
        # 50 points identical to master, 50 points far away
        aligned = np.vstack([master[:50], rng.randn(50, 3) + 1000.0])
        rate = compute_overlap_rate(master, aligned, threshold=1e-6)
        self.assertAlmostEqual(rate, 0.5)

    def test_threshold_matters(self):
        from test_scanned_vs_master import compute_overlap_rate
        master = np.array([[0.0, 0.0, 0.0]])
        aligned = np.array([[0.5, 0.0, 0.0]])
        self.assertAlmostEqual(
            compute_overlap_rate(master, aligned, threshold=0.1), 0.0
        )
        self.assertAlmostEqual(
            compute_overlap_rate(master, aligned, threshold=1.0), 1.0
        )

    def test_empty_aligned_returns_zero(self):
        from test_scanned_vs_master import compute_overlap_rate
        master = np.random.randn(100, 3)
        aligned = np.zeros((0, 3))
        rate = compute_overlap_rate(master, aligned, threshold=1.0)
        self.assertEqual(rate, 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 5.2: Run test to verify it fails**

```bash
conda activate LightGlue && cd /home/jhs/work/Registration/glue-factory_depth && python -m unittest tests.test_overlap_rate -v 2>&1
```

Expected: ImportError / AttributeError (function not defined).

- [ ] **Step 5.3: Implement `compute_overlap_rate` and `estimate_registration`**

Add after `build_batch`:

```python
def compute_overlap_rate(pc_master, pc_aligned, threshold=1.0):
    """Fraction of aligned points whose nearest master point is within threshold.

    Uses scipy cKDTree for efficient nearest-neighbor search.
    Returns 0.0 if either cloud is empty.
    """
    if len(pc_master) == 0 or len(pc_aligned) == 0:
        return 0.0
    tree = cKDTree(pc_master)
    dists, _ = tree.query(pc_aligned, k=1)
    return float((dists < threshold).mean())


def estimate_registration(
    pred, master_crop_raw, scanned_crop_raw,
    image_size=1751,
    ransac_iter=1000,
    inlier_th=5.0,
    n_sample_pts=15000,
):
    """Extract matches, compute RANSAC rigid transform in Grid space, and
    return both the transformation and sampled point clouds for visualization.

    Grid space uses existing pixel_to_grid3d helper from visualize_registration_flann.
    """
    kp0 = pred["keypoints0"][0].cpu().numpy()  # master
    kp1 = pred["keypoints1"][0].cpu().numpy()  # scanned
    m0 = pred["matches0"][0].cpu().numpy()

    valid = m0 > -1
    mkp0 = kp0[valid]
    mkp1 = kp1[m0[valid]]
    n_matches = int(valid.sum())

    # Rescale keypoints from image_size back to CROP_SIZE for pixel_to_grid3d
    scale = CROP_SIZE / image_size
    mkp0_crop = mkp0 * scale
    mkp1_crop = mkp1 * scale

    # Convert matched keypoints to Grid 3D
    pts_master, vm = pixel_to_grid3d(mkp0_crop, master_crop_raw)
    pts_scanned, vi = pixel_to_grid3d(mkp1_crop, scanned_crop_raw)
    both_valid = vm & vi
    pts_master = pts_master[both_valid]
    pts_scanned = pts_scanned[both_valid]
    n_valid_matches = len(pts_master)

    if n_valid_matches < 3:
        print(f"  WARNING: only {n_valid_matches} valid matches, skipping RANSAC")
        R, t = np.eye(3), np.zeros(3)
        inliers = np.zeros(n_valid_matches, dtype=bool)
    else:
        # RANSAC: align scanned → master
        R, t, inliers = ransac_rigid(
            pts_scanned, pts_master, n_iter=ransac_iter, inlier_th=inlier_th
        )

    # Sample dense clouds for visualization
    pc_master = sample_point_cloud(master_crop_raw, n_pts=n_sample_pts)
    pc_scanned = sample_point_cloud(scanned_crop_raw, n_pts=n_sample_pts)

    # Apply transform to scanned cloud
    if len(pc_scanned) > 0:
        pc_scanned_aligned = (R @ pc_scanned.T).T + t
    else:
        pc_scanned_aligned = pc_scanned

    return {
        "R": R,
        "t": t,
        "n_matches": n_matches,
        "n_valid_matches": n_valid_matches,
        "n_inliers": int(inliers.sum()),
        "pc_master": pc_master,
        "pc_scanned": pc_scanned,
        "pc_scanned_aligned": pc_scanned_aligned,
    }
```

- [ ] **Step 5.4: Run unit tests to verify they pass**

```bash
conda activate LightGlue && cd /home/jhs/work/Registration/glue-factory_depth && python -m unittest tests.test_overlap_rate -v 2>&1
```

Expected: All 5 tests pass.

- [ ] **Step 5.5 [commit checkpoint]**: Suggested: `feat(scanned): add registration estimator and overlap metric with unit tests`

---

## Task 6: Visualization functions (GT-removed variants)

**Files:**
- Modify: `test_scanned_vs_master.py`

- [ ] **Step 6.1: Implement `plot_matches_no_gt`**

Add after `estimate_registration`:

```python
def plot_matches_no_gt(
    pred, master_small, scanned_small, output_path, title,
):
    """Simplified matching visualization — single color, no GT comparison."""
    kp0 = pred["keypoints0"][0].cpu().numpy()
    kp1 = pred["keypoints1"][0].cpu().numpy()
    m0 = pred["matches0"][0].cpu().numpy()

    valid = m0 > -1
    mkp0 = kp0[valid]
    mkp1 = kp1[m0[valid]]
    n_total = int(valid.sum())

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
    for ax, img, kp, label in [
        (axes[0], master_small, kp0, "View 0 (Master)"),
        (axes[1], scanned_small, kp1, "View 1 (Scanned)"),
    ]:
        ax.imshow(img, cmap="gray")
        ax.scatter(kp[:, 0], kp[:, 1], c="royalblue", s=3, alpha=0.3, linewidths=0)
        ax.set_title(label, fontsize=14)
        ax.set_axis_off()

    for i in range(n_total):
        line = matplotlib.patches.ConnectionPatch(
            xyA=(float(mkp0[i, 0]), float(mkp0[i, 1])),
            xyB=(float(mkp1[i, 0]), float(mkp1[i, 1])),
            coordsA=axes[0].transData,
            coordsB=axes[1].transData,
            axesA=axes[0],
            axesB=axes[1],
            color="skyblue",
            linewidth=0.8,
            alpha=0.6,
        )
        fig.add_artist(line)

    info = f"{title}  |  matches={n_total}"
    fig.suptitle(info, fontsize=11, y=0.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")
```

- [ ] **Step 6.2: Implement `plot_alignment_no_gt`**

Add after `plot_matches_no_gt`:

```python
def plot_alignment_no_gt(
    pc_master, pc_scanned, pc_scanned_aligned,
    output_path, title,
    n_matches=0, n_inliers=0, overlap_rate=0.0,
):
    """2-row (Before / Estimated) × 3-col (Top-down / Front / Side) visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    s = 0.5
    alpha = 0.6

    labels_row = ["Before", "Estimated R,t"]
    labels_col = ["Top-down (u, v)", "Front (u, z)", "Side (v, z)"]
    col_axes = [(0, 1), (0, 2), (1, 2)]

    C_MASTER = "lightskyblue"
    C_INPUT = "crimson"
    pairs = [
        (pc_master, pc_scanned),
        (pc_master, pc_scanned_aligned),
    ]

    for row, (pc_a, pc_b) in enumerate(pairs):
        for col, (xi, yi) in enumerate(col_axes):
            ax = axes[row, col]
            if len(pc_a) > 0:
                ax.scatter(pc_a[:, xi], pc_a[:, yi], s=s, c=C_MASTER, alpha=alpha, label="master")
            if len(pc_b) > 0:
                ax.scatter(pc_b[:, xi], pc_b[:, yi], s=s, c=C_INPUT, alpha=alpha, label="scanned")
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(labels_col[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(labels_row[row], fontsize=11, fontweight="bold")

    for row in range(2):
        for col in range(3):
            axes[row, col].invert_yaxis()

    info = (f"{title}  |  matches={n_matches}, inliers={n_inliers}, "
            f"overlap@1={overlap_rate:.3f}")
    fig.suptitle(info, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
```

- [ ] **Step 6.3: Smoke test — generate one match plot and one alignment plot from synthetic data**

```bash
conda activate LightGlue && cd /home/jhs/work/Registration/glue-factory_depth && python -c "
import numpy as np
import torch
from pathlib import Path
from test_scanned_vs_master import plot_matches_no_gt, plot_alignment_no_gt

# Synthetic pred with 5 matches
pred = {
    'keypoints0': torch.tensor([[[100, 100], [200, 200], [300, 300], [400, 400], [500, 500]]], dtype=torch.float32),
    'keypoints1': torch.tensor([[[110, 105], [210, 205], [310, 305], [410, 405], [510, 505]]], dtype=torch.float32),
    'matches0': torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long),
}
master_small = np.random.RandomState(0).rand(1751, 1751).astype(np.float32)
scanned_small = np.random.RandomState(1).rand(1751, 1751).astype(np.float32)

out_dir = Path('/tmp/scanned_viz_smoke')
out_dir.mkdir(exist_ok=True)

plot_matches_no_gt(pred, master_small, scanned_small, out_dir / 'match.png', 'smoke test')

pc_master = np.random.RandomState(2).randn(1000, 3) * 10
pc_scanned = pc_master + np.array([5, 5, 5])
pc_aligned = pc_master
plot_alignment_no_gt(
    pc_master, pc_scanned, pc_aligned,
    out_dir / 'reg.png', 'smoke test',
    n_matches=5, n_inliers=5, overlap_rate=1.0,
)
print('OK — check', out_dir)
"
```

Expected: two PNG files at `/tmp/scanned_viz_smoke/match.png` and `/tmp/scanned_viz_smoke/reg.png` with no errors.

- [ ] **Step 6.4 [commit checkpoint]**: Suggested: `feat(scanned): add GT-free visualization functions`

---

## Task 7: `main()` orchestration with `--run_both`

**Files:**
- Modify: `test_scanned_vs_master.py`

- [ ] **Step 7.1: Implement `run_single_experiment` helper**

Add before `main`:

```python
def run_single_experiment(
    args, model, conf, device, scale_dz, experiment_label,
):
    """Run one full experiment (raw or dzfix) across all master indices."""
    exp_dir = Path(args.output_dir) / experiment_label
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Experiment: {experiment_label} (scale_dz={scale_dz}) ===")
    scanned_crop_raw, _, meta = preprocess_scanned(
        Path(args.scanned_path),
        scale_dz=scale_dz,
        dz_ratio=args.dz_ratio,
    )
    print(f"  Scanned preprocessed: {meta}")

    scanned_feats = extract_scanned_features(
        scanned_crop_raw,
        fpfh_radius=args.fpfh_radius,
        max_num_keypoints=args.max_num_keypoints,
        image_size=args.image_size,
    )
    print(f"  Scanned features: n_valid={scanned_feats['n_valid']}, "
          f"n_iss={scanned_feats['n_iss']}")

    rows = []
    for master_idx in args.master_indices:
        print(f"\n  --- Master {master_idx:04d} ---")
        master_feats, master_crop_raw = load_master_features(
            master_idx, args.cache_dir, args.master_img_dir,
            image_size=args.image_size,
        )
        batch, master_small, scanned_small = build_batch(
            master_feats, master_crop_raw,
            scanned_feats, scanned_crop_raw,
            image_size=args.image_size,
            master_path=str(Path(args.master_img_dir) / f"depth_raw_{master_idx:04d}.png"),
            scanned_path=str(args.scanned_path),
        )

        pred, batch = run_inference(model, batch, device)

        match_path = exp_dir / f"master{master_idx:04d}_match.png"
        plot_matches_no_gt(
            pred, master_small, scanned_small, match_path,
            title=f"{experiment_label} master={master_idx:04d}",
        )

        reg = estimate_registration(
            pred, master_crop_raw, scanned_crop_raw,
            image_size=args.image_size,
            ransac_iter=args.ransac_iter,
            inlier_th=args.inlier_th,
        )
        overlap = compute_overlap_rate(
            reg["pc_master"], reg["pc_scanned_aligned"],
            threshold=args.overlap_threshold,
        )

        reg_path = exp_dir / f"master{master_idx:04d}_reg.png"
        plot_alignment_no_gt(
            reg["pc_master"], reg["pc_scanned"], reg["pc_scanned_aligned"],
            reg_path,
            title=f"{experiment_label} master={master_idx:04d}",
            n_matches=reg["n_matches"],
            n_inliers=reg["n_inliers"],
            overlap_rate=overlap,
        )

        rows.append({
            "master_idx": master_idx,
            "experiment": experiment_label,
            "n_scanned_kp": scanned_feats["n_valid"],
            "n_master_kp": master_feats["n_valid"],
            "n_matches": reg["n_matches"],
            "n_inliers": reg["n_inliers"],
            "overlap_rate": round(overlap, 4),
        })

    return rows
```

- [ ] **Step 7.2: Implement `main()`**

Replace the existing `main()` stub with:

```python
def main():
    args = parse_args()
    print(f"Args: {vars(args)}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model, conf = load_model(args.checkpoint, device)

    # Determine which experiments to run
    if args.run_both:
        experiments = [(False, "raw"), (True, "dzfix")]
    elif args.scale_dz:
        experiments = [(True, "dzfix")]
    else:
        experiments = [(False, "raw")]

    all_rows = []
    for scale_dz, label in experiments:
        rows = run_single_experiment(args, model, conf, device, scale_dz, label)
        all_rows.extend(rows)

    # Write summary.csv at the root (not inside raw/ or dzfix/)
    csv_path = output_root / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["master_idx", "experiment", "n_scanned_kp",
                        "n_master_kp", "n_matches", "n_inliers", "overlap_rate"],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\nDone! Summary: {csv_path}")
    print(f"Total runs: {len(all_rows)}")
```

- [ ] **Step 7.3: Smoke test — run with `--master_indices 0` only (single run, no dz scaling)**

```bash
conda activate LightGlue && cd /home/jhs/work/Registration/glue-factory_depth && python test_scanned_vs_master.py --master_indices 0 --output_dir /tmp/scanned_smoke 2>&1 | tail -30
```

Expected:
- No exceptions
- Prints ISS count, master loaded, "Saved: master0000_match.png", "Saved: master0000_reg.png"
- File created: `/tmp/scanned_smoke/raw/master0000_match.png`
- File created: `/tmp/scanned_smoke/raw/master0000_reg.png`
- File created: `/tmp/scanned_smoke/summary.csv` with one row

If it errors, debug before proceeding.

- [ ] **Step 7.4 [commit checkpoint]**: Suggested: `feat(scanned): add main orchestration with --run_both and summary.csv`

---

## Task 8: Phase 2 — Dry-run sanity check (master through scanned path)

**Purpose:** Verify the new script path produces reasonable matching when the input is a *known-good master image* routed through the scanned preprocessing path. This isolates "pipeline bug" vs "data issue" before running on real scanned data.

**Files:** No code changes — execution only.

- [ ] **Step 8.1: Run dry-run with master 594 fed as "scanned" input**

The idea: temporarily use `depth_raw_0594.png` as the scanned input path. Since the master file is 5761×5761 (much larger than scanned 2687×3515), the method D scale factor becomes `3502/5761 ≈ 0.608` instead of 0.9963, but the pipeline handles it the same way. Then match it against a different master (e.g., index 0).

```bash
conda activate LightGlue && cd /home/jhs/work/Registration/glue-factory_depth && python test_scanned_vs_master.py \
    --scanned_path gluefactory/datasets/mitsubishi/dataset_resample_2/depth_raw_0594.png \
    --master_indices 0 \
    --output_dir /tmp/scanned_dryrun 2>&1 | tail -40
```

- [ ] **Step 8.2: Inspect the dry-run output**

Check:
- `/tmp/scanned_dryrun/raw/master0000_match.png` — matches should follow object contours (not random)
- `/tmp/scanned_dryrun/raw/master0000_reg.png` — Before row should show the two clouds offset, Estimated row should show them partially overlapping
- `/tmp/scanned_dryrun/summary.csv` — `n_matches` should be > 10, `overlap_rate` > 0.05

Note: this is not a strict pass/fail — it's a sanity check. If matches are near zero, the pipeline has a bug. If matches are reasonable, the pipeline is working and any downstream failure with actual scanned data is due to data mismatch, not code.

Compare against the baseline `test_resample2_iss_fpfh_0406.py --indices <0, 594 pair index>` if you want a tighter reference (optional).

- [ ] **Step 8.3**: Document dry-run findings in terminal output (no file changes). Proceed only if reasonable.

---

## Task 9: Phase 3 — Real run on scanned data

**Purpose:** The actual experiment. Run both B (raw) and C (dzfix) against masters 0 and 594.

**Files:** No code changes — execution only.

- [ ] **Step 9.1: Run the full experiment**

```bash
conda activate LightGlue && cd /home/jhs/work/Registration/glue-factory_depth && python test_scanned_vs_master.py --run_both 2>&1 | tee results/scanned/scanned_data1/run.log
```

Expected:
- Console shows two "=== Experiment:" sections (raw, then dzfix)
- Four `Saved:` lines per experiment (2 masters × 2 viz types)
- Final line: `Done! Summary: results/scanned/scanned_data1/summary.csv` and `Total runs: 4`

- [ ] **Step 9.2: Inspect `summary.csv`**

```bash
cat results/scanned/scanned_data1/summary.csv
```

Expected layout:
```
master_idx,experiment,n_scanned_kp,n_master_kp,n_matches,n_inliers,overlap_rate
0,raw,...,...,...,...,...
594,raw,...,...,...,...,...
0,dzfix,...,...,...,...,...
594,dzfix,...,...,...,...,...
```

- [ ] **Step 9.3: Interpret results with the success criteria in §8 of the spec**

Report findings as a terminal summary:
1. **dz correction hypothesis**: compare `overlap_rate(dzfix)` vs `overlap_rate(raw)` for each master. If dzfix > raw, the dz mismatch was meaningful and correction helps.
2. **Viewpoint similarity hypothesis**: for each experiment, compare `overlap_rate(master=0)` vs `overlap_rate(master=594)`. The higher one indicates which master viewpoint is closer to the scanned object's pose.
3. **Visual check of 8 PNGs**: open `results/scanned/scanned_data1/raw/*.png` and `results/scanned/scanned_data1/dzfix/*.png` to visually confirm the numeric results.

- [ ] **Step 9.4**: Decide the next iteration based on results:
   - **Both experiments fail** (matches/overlap near zero across all combinations): revisit data assumptions (is scanned actually a different object? is dz ratio really 0.425?). Consider the rejected orthographic approach.
   - **dzfix clearly beats raw**: dz correction is validated. Extend to more masters to find the best viewpoint.
   - **Partial success (one master works, one doesn't)**: scanned viewpoint is close to the working master. Expand master indices around that index.
   - **Both work comparably**: dz difference was tolerable for this data; either approach is fine.

---

## Self-review

### Spec coverage

| Spec section | Covered by task |
|---|---|
| §2 Context — scanned file loaded from correct path | Task 3, Task 7 |
| §3.1 Scale interpretation A (dz only) | Task 2 (dz_ratio parameter) |
| §3.2 Virtual perspective B+C | Task 3 (uses existing `build_xyz_pcd` with master intrinsics), Task 7 (--run_both loops both) |
| §3.3 Size handling method D (scale + pad, INTER_NEAREST) | Task 2 |
| §3.4 Master indices [0, 594] | Task 1 (default), overridable via CLI |
| §3.5 Output structure with raw/dzfix nested + root summary.csv | Task 7 |
| §3.5 Proxy overlap metric at 1mm | Task 5 |
| §4.2 ScannedPreprocessor | Task 2 |
| §4.2 ScannedFeatureExtractor | Task 3 |
| §4.2 MasterLoader | Task 4 |
| §4.2 PairSampleBuilder | Task 4 |
| §4.2 LightGlueInference (import only) | Task 1 |
| §4.2 RegistrationEstimator | Task 5 |
| §4.2 Visualizer (no-GT variants) | Task 6 |
| §4.5 Key rename `fpfh_descriptors` → `descriptors` | Task 4 (`build_batch` `_view_dict`) |
| §5 Function signatures | Tasks 2–6 |
| §6 CLI | Task 1, Task 7 |
| §7 Error handling (missing files, overflow, too few pixels) | Task 2 (warn), Task 4 (FileNotFoundError), Task 5 (n_valid < 3 guard) |
| §8 Success criteria | Task 9.3 (reporting step) |
| §9 Phase 1/2/3 execution plan | Tasks 1–7 / 8 / 9 |

### Placeholder scan
Searched plan for "TBD", "TODO", "implement later", "handle edge cases", "similar to Task". None found. All code blocks are complete.

### Type consistency
- `preprocess_scanned` returns `(scanned_crop_raw, scanned_img_small, meta)` — 3-tuple. Used consistently in Task 3 (unpacks first element) and Task 7 (uses first + meta).
- `extract_scanned_features` returns dict with keys `keypoints, keypoint_scores, fpfh_descriptors, n_valid, n_iss, kp_crop` — matches `load_master_features` return dict (same keys). `build_batch._view_dict` reads `feats["fpfh_descriptors"]` and renames to `"descriptors"`. ✅
- `estimate_registration` returns dict with `R, t, n_matches, n_valid_matches, n_inliers, pc_master, pc_scanned, pc_scanned_aligned`. Task 7 reads `reg["pc_master"]`, `reg["pc_scanned_aligned"]`, `reg["n_matches"]`, `reg["n_inliers"]`. ✅
- `compute_overlap_rate(pc_master, pc_aligned, threshold)` — Task 5 test uses `threshold=` kwarg, Task 7 uses `threshold=args.overlap_threshold`. ✅
- `pc_master` in `plot_alignment_no_gt` vs same in `estimate_registration` — both are `np.ndarray` of shape `(N, 3)`. ✅
- `pred` dict access: `keypoints0`, `keypoints1`, `matches0` — used in Task 5 (`estimate_registration`) and Task 6 (`plot_matches_no_gt`) identically. These match the model output structure in `test_resample2_iss_fpfh_0406.py:visualize_pair`. ✅

### Inline fixes
None found on review.

---

## Execution notes

- **User confirmation**: Per the user's standing preference, any agent executing this plan must ask for confirmation before each code/config file Edit or Write. Reading files is fine without confirmation.
- **Commits**: The user commits personally. Agents must NOT run `git commit` or `git add` at any step. `[commit checkpoint]` markers are suggestions for the user to commit manually.
- **Conda environment**: Always activate `LightGlue` before running any Python command (`conda activate LightGlue`).
- **Working directory**: All commands assume `/home/jhs/work/Registration/glue-factory_depth/` as the current directory.
