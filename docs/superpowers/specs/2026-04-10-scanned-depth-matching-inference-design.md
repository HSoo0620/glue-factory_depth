# Scanned Depth Matching Inference Design

**Date**: 2026-04-10
**Status**: Approved, awaiting implementation
**Scope**: Single inference pipeline for comparing real-scanned depth data against Blender-rendered master depth using the existing trained ISS+FPFH+LightGlue model.

---

## 1. Goal

Run inference with the trained `0406_resample2_iss_fpfh_xyz_lg` model on a real 3D line-scanner output (`scanned_data1.png`) and visualize both keypoint matching and 3D registration against multiple master viewpoints. GT is unavailable, so evaluation is qualitative (visualization) plus a proxy overlap metric for relative comparison.

## 2. Context

### 2.1 Existing trained model
- **Experiment**: `0406_resample2_iss_fpfh_xyz_lg`
- **Checkpoint**: `outputs/training/0406_resample2_iss_fpfh_xyz_lg/checkpoint_best.tar` (ep81, recall 0.288 at gt_radius=20)
- **FPFH space**: true perspective `(X, Y, Z)` using camera intrinsics
  - `X = (u - cx) * depth_real / fx`, `Y = (v - cy) * depth_real / fy`, `Z = depth_real`
  - Master intrinsics: `fx = fy = 8001.39`, `cx_crop = 1751.5`, `cy_crop = 1799.5`
  - Radius `r = 5.0 mm` in true XYZ space
- **Registration space**: Grid `(u·0.05, v·0.05, depth_real)` — different from FPFH space, matches CLAUDE.md convention

### 2.2 Scanned data spec
- Path: `gluefactory/datasets/scanned/scanned_data1.png`
- Resolution: 2687×3515 (non-square), uint16 PNG
- Raw range: 0 to 37019, median 29334, zero (background) ratio 35.4%
- Reported physical spacing: `dx = dy = 0.056 mm`, `dz = 0.0085 mm`
- Acquisition: real 3D line scanner (triangulation-based, likely resampled to regular grid by firmware)
- No calib file; no camera intrinsics available

### 2.3 Master data spec (reference)
- Path: `gluefactory/datasets/mitsubishi/dataset_resample_2/depth_raw_XXXX.png`
- Resolution: 5761×5761, uint16 PNG
- calib: `fx = fy = 8001.39, cx = cy = 2880.5, clip_start = 0.1, clip_end = 1000.0`
- Physical spacing: `grid_dx = grid_dy = 0.05 mm, grid_dz = 0.02 mm`
- Fixed crop applied in pipeline: `(x0=1129, y0=1081, size=3502×3502)` then resized to 1751×1751 via INTER_NEAREST
- 641 total images

### 2.4 The mismatch
Scanned `dz = 0.0085 mm` vs master `grid_dz = 0.02 mm`. The ratio `0.0085 / 0.02 = 0.425` means scanned raw values sample depth ~2.35× more finely than master. Without correction, the scanned Z range is inflated relative to master.

Scanned `dx/dy = 0.056 mm` vs master `grid_dx/dy = 0.05 mm` is also different (scanner slightly coarser laterally), but this difference is treated as within acceptable tolerance and not corrected.

## 3. Approach Decision Log

### 3.1 Scale mismatch interpretation (Decision A)
The user's intent is dz-axis only: scanned samples depth more finely than master, so raw values must be scaled by `0.425` to align to master units. Lateral `dx/dy` differences are ignored.

### 3.2 FPFH computation space (Decision B+C)
Three options were considered:
- **Pure orthographic B**: `(u·dx, v·dy, raw·dz)` as the 3D space
- **Virtual perspective B'**: borrow master's intrinsics and use the same perspective projection formula
- **Both with/without dz correction**

Pure orthographic B was rejected because:
1. The trained model computes FPFH in *perspective* `(X, Y, Z)` space (`precompute_iss_fpfh_resample2.py:86-96`)
2. `CLAUDE.md` documents that a similar "fake 3D" experiment (`0402_resample2_iss_fpfh_lg`) achieved recall 0.0 because positive pair cosine similarity ≈ random similarity — the descriptors are non-discriminative in that space
3. Orthographic `(u·dx, v·dy, raw·dz)` is structurally identical to the failed "fake 3D" up to axis scaling
4. FPFH `radius = 5.0 mm` is defined in the true XYZ space; in orthographic pixel space the same numeric value has a different physical meaning

**Decision**: Use virtual perspective (B') — borrow master's intrinsics, compute FPFH via the same perspective formula on scanned data. Run two experiments:
- **Experiment B (raw)**: scanned raw values as-is (no dz correction)
- **Experiment C (dzfix)**: scanned raw values × 0.425 before any computation

Both experiments use identical infrastructure; the only difference is a single multiplication on the raw depth.

### 3.3 Size mismatch handling (Decision D)
Scanned is 2687×3515 (non-square) but the pipeline expects a 3502×3502 crop that is then resized to 1751×1751.

Options considered:
- **A — Zero pad after center**: simple but requires 13px vertical crop (3515 > 3502)
- **B — Non-uniform resize**: distorts dx ≠ dy, harmful to FPFH
- **C — Center crop to square**: loses ~25% of the object
- **D — Aspect-preserving scale + center pad**: scale by `3502 / 3515 ≈ 0.9963` to get 2677×3502, then center-pad horizontally to 3502×3502

**Decision**: Option D. It is virtually lossless (0.37% scale), preserves aspect ratio, keeps dx = dy, and places the content in a canvas compatible with existing pipeline offsets. `INTER_NEAREST` is mandatory to avoid introducing spurious intermediate depth values at silhouette boundaries.

### 3.4 Master selection
User-specified indices: `[0, 594]`. These represent near-extreme viewpoints in the 641-image dataset. Additional indices can be added via CLI override if needed.

### 3.5 Output structure
Nested directory layout under `results/scanned/scanned_data1/`. `summary.csv` lives at the root (outside the `raw/` and `dzfix/` subdirectories) so that both experiments append to the same file:
```
results/scanned/scanned_data1/
├── summary.csv
├── raw/
│   ├── master0000_match.png
│   ├── master0000_reg.png
│   ├── master0594_match.png
│   └── master0594_reg.png
└── dzfix/
    ├── master0000_match.png
    ├── master0000_reg.png
    ├── master0594_match.png
    └── master0594_reg.png
```

`summary.csv` schema: `master_idx, experiment, n_scanned_kp, n_master_kp, n_matches, n_inliers, overlap_rate`.

**Proxy metric**: `overlap_rate` = fraction of aligned scanned points whose nearest master point is within 1 unit of distance in the Grid coordinate space `(u·0.05, v·0.05, depth_real)`, computed via KDTree. The "1 unit" roughly corresponds to 1 mm in the Grid convention since `grid_dx = 0.05 mm` gives sub-pixel scale. This is **not** an absolute score — meaningful only for relative comparison across (experiment, master) combinations. See §5.5 for implementation.

## 4. Architecture

### 4.1 Single entrypoint
`test_scanned_vs_master.py` is a new standalone script (~350 lines) that:
- Imports existing feature-extraction and visualization functions without modifying any existing file
- Orchestrates preprocessing, feature extraction, inference, registration, and visualization
- Runs both experiments (B and C) in one invocation when `--run_both` is set

### 4.2 Component responsibilities

| Component | Type | Responsibility |
|---|---|---|
| `ScannedPreprocessor` | New | Load scanned PNG, apply method D resize+pad, optionally apply dz correction |
| `ScannedFeatureExtractor` | New (thin wrapper over existing funcs) | Run ISS+FPFH pipeline on scanned crop at runtime, return cache-compatible dict |
| `MasterLoader` | New (direct npz load) | Load existing cached master features and raw depth crop for the given index |
| `PairSampleBuilder` | New | Pack master+scanned into the batch dict shape expected by the model |
| `LightGlueInference` | Reused (import) | Load checkpoint and run model.forward (from `test_resample2_iss_fpfh_0406.py`) |
| `RegistrationEstimator` | Reused + proxy metric addition | Pixel→Grid3D + RANSAC SVD (from `visualize_registration_flann.py`), plus `compute_overlap_rate` |
| `Visualizer` | Reused with GT-removed variants | 2-row alignment plot (no GT row), simplified matching plot |

### 4.3 Data flow
```
scanned_data1.png (2687×3515 uint16)
    ↓  ScannedPreprocessor (method D ± dz × 0.425)
scanned_crop_raw (3502×3502)
    ↓  ScannedFeatureExtractor (ISS+FPFH runtime)
scanned_feats dict ←────────── MasterLoader ← depth_raw_{0000,0594}.npz
    ↓                                             ↓
PairSampleBuilder (view0=master, view1=scanned)
    ↓
LightGlueInference → pred (matches0, matching_scores0, ...)
    ↓
RegistrationEstimator → R, t, inliers, pc_master, pc_scanned_aligned, overlap_rate
    ↓
Visualizer → match PNG + reg PNG
    ↓
summary.csv row append
```

### 4.4 Module dependencies (imports)

From `precompute_iss_fpfh_resample2.py`:
`depth_crop_to_pcd`, `build_xyz_pcd`, `extract_iss_keypoints`, `select_keypoints`, `kp_crop_to_xyz`, `compute_fpfh_for_keypoints`, `CROP_X0`, `CROP_Y0`, `CROP_SIZE`, `CLIP_START`, `CLIP_END`

From `visualize_registration_flann.py`:
`pixel_to_grid3d`, `rigid_transform_svd`, `ransac_rigid`, `sample_point_cloud`, `GRID_DX`, `GRID_DY`

From `test_resample2_iss_fpfh_0406.py`:
`load_model`, `run_inference`

From `gluefactory.datasets.mitsubishi_resample2_iss_fpfh_dataset`:
`resample2_iss_fpfh_collate_fn`

From `gluefactory.utils.tensor`:
`batch_to_device`

### 4.5 Key invariant: `descriptors` key rename
The dataset class loads cache file `fpfh_descriptors` under the key `descriptors` before handing it to the model. The new `PairSampleBuilder` must do the same rename; passing the raw `fpfh_descriptors` key would break the model forward pass.

## 5. Function signatures

### 5.1 `preprocess_scanned`
```python
def preprocess_scanned(
    scanned_png_path: Path,
    scale_dz: bool = False,
    dz_ratio: float = 0.425,          # 0.0085 / 0.02
    target_crop_size: int = 3502,
    target_image_size: int = 1751,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns:
        scanned_crop_raw: (3502, 3502) uint16 — full-res canvas
        scanned_img_small: (1751, 1751) float32 — model image input
        meta: {orig_shape, scaled_shape, pad_left, pad_top, scale_factor, dz_scaled}
    """
```

Logic:
1. `cv2.imread(..., IMREAD_UNCHANGED)` → (3515, 2687) uint16
2. `scale_factor = target_crop_size / max(h, w) ≈ 0.9963`
3. `cv2.resize(..., (new_w, new_h), interpolation=INTER_NEAREST)`
4. Center-pad into `canvas = np.zeros((3502, 3502), uint16)`
5. If `scale_dz`: `canvas = np.where(canvas > 0, (canvas.astype(float64) * dz_ratio).clip(0, 65535).astype(uint16), 0)`
6. `scanned_img_small = cv2.resize(canvas, (1751, 1751), INTER_NEAREST).astype(float32) / 65535.0`

### 5.2 `extract_scanned_features`
```python
def extract_scanned_features(
    scanned_crop_raw: np.ndarray,      # (3502, 3502) uint16
    fx: float = 8001.39,
    fy: float = 8001.39,
    cx: float = 1751.0,                # scanned canvas center
    cy: float = 1751.0,
    fpfh_radius: float = 5.0,
    max_num_keypoints: int = 512,
    image_size: int = 1751,
    erode_boundary: int = 5,
) -> dict:
    """Returns: keypoints, keypoint_scores, fpfh_descriptors, n_valid, n_iss, kp_crop"""
```

Calls `depth_crop_to_pcd → extract_iss_keypoints → select_keypoints → build_xyz_pcd → kp_crop_to_xyz → compute_fpfh_for_keypoints` in sequence. Same order as `precompute_iss_fpfh_resample2.py:main`.

Note: `cx = cy = 1751.0` (canvas center), which differs from master's `1751.5 / 1799.5`. FPFH is translation-invariant so descriptor values are unaffected; this mismatch has no functional impact.

### 5.3 `load_master_features`
```python
def load_master_features(
    master_idx: int,
    cache_dir: Path = Path("gluefactory/datasets/mitsubishi/iss_fpfh_resample2_cache_r5.0_xyz"),
) -> tuple[dict, np.ndarray]:
    """Returns (features_dict, master_crop_raw)"""
```

Loads `depth_raw_{idx:04d}.npz` directly. Existing cache keys are `['keypoints', 'keypoint_scores', 'fpfh_descriptors', 'n_valid', 'n_iss']`. `kp_crop` is not stored — reconstruct as `keypoints[:n_valid] * (CROP_SIZE / image_size) = keypoints[:n_valid] * 2`.

Also loads the raw master PNG and applies the fixed crop `[CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]` for registration/visualization.

### 5.4 `build_batch`
```python
def build_batch(
    master_feats: dict, master_crop_raw: np.ndarray,
    scanned_feats: dict, scanned_crop_raw: np.ndarray,
    image_size: int = 1751,
) -> dict:
    """Returns the batch dict structure expected by resample2_iss_fpfh_collate_fn."""
```

Constructs per-view subdicts with keys `image`, `image_size`, `keypoints`, `keypoint_scores`, `descriptors` (renamed from `fpfh_descriptors`). Top-level includes `gt_matches = torch.zeros((0, 4))`, `csv_path = ""`, `master_path`, `input_path` (the scanned file path). Runs `resample2_iss_fpfh_collate_fn([sample])` to produce the final batched form.

### 5.5 `estimate_registration` and `compute_overlap_rate`
```python
def estimate_registration(
    pred: dict, master_crop_raw: np.ndarray, scanned_crop_raw: np.ndarray,
    image_size: int = 1751,
    ransac_iter: int = 1000, inlier_th: float = 5.0,
) -> dict:
    """Returns R, t, inliers, n_matches, pc_master, pc_scanned, pc_scanned_aligned."""

def compute_overlap_rate(
    pc_master: np.ndarray,          # (M, 3)
    pc_scanned_aligned: np.ndarray, # (N, 3)
    threshold_mm: float = 1.0,
) -> float:
    """Fraction of aligned scanned points within threshold of any master point."""
```

Registration uses the Grid coordinate system `(u·0.05, v·0.05, depth_real)`. Matches are derived from `pred["matches0"]`, converted to 3D via `pixel_to_grid3d`, then passed to `ransac_rigid`. `sample_point_cloud` is used to get dense clouds for visualization. KDTree (from `scipy.spatial.cKDTree` or `open3d.geometry.KDTreeFlann`) provides the nearest-neighbor search for `compute_overlap_rate`.

### 5.6 Visualization (GT-removed variants)
```python
def plot_matches_no_gt(
    pred, master_img_small, scanned_img_small, output_path, title,
): ...

def plot_alignment_no_gt(
    pc_master, pc_scanned, pc_scanned_aligned,
    output_path, title,
    n_matches=0, n_inliers=0, overlap_rate=0.0,
): ...
```

`plot_matches_no_gt`: simplified version of `test_resample2_iss_fpfh_0406.py:visualize_pair` with the `csv_path is None` branch only, all lines uniformly colored `skyblue`.

`plot_alignment_no_gt`: 2-row (Before, Estimated) × 3-column (Top-down, Front, Side) scatter plots. Same visual convention as `visualize_registration_flann.py:plot_alignment` but with the GT row removed.

## 6. CLI

```
python test_scanned_vs_master.py \
    --scanned_path gluefactory/datasets/scanned/scanned_data1.png \
    --master_indices 0 594 \
    --checkpoint outputs/training/0406_resample2_iss_fpfh_xyz_lg/checkpoint_best.tar \
    --output_dir results/scanned/scanned_data1 \
    --fpfh_radius 5.0 \
    --image_size 1751 \
    --dz_ratio 0.425 \
    --run_both          # runs both raw/ and dzfix/ in a single invocation
    # or
    --scale_dz          # runs only the dzfix experiment
```

When `--run_both` is set, the script loops over `[False, True]` for `scale_dz`, writes to `raw/` and `dzfix/` respectively, and appends rows to a single `summary.csv`.

## 7. Error handling

| Condition | Detection | Handling |
|---|---|---|
| scanned file missing | `Path.exists()` | FileNotFoundError, abort |
| scanned dtype ≠ uint16 | `arr.dtype` | Warn, force `astype(uint16)` |
| scanned height > target_crop_size | `scale_factor` calculation | Method D always shrinks to ≤ target; no action needed |
| dz-scaled raw overflow | `.clip(0, 65535)` | Silent saturation prevention |
| scanned valid pixels too few (< 1000) | `(raw > 0).sum()` | Warn; if < 100, skip |
| ISS returns zero keypoints | `n_iss == 0` | Existing `select_keypoints` path handles via random fill |
| FPFH with n_valid < 3 | `compute_fpfh_for_keypoints` check | Existing code returns zero descriptor |
| Master cache npz missing | `npz_path.exists()` | FileNotFoundError with instruction to run precompute |
| RANSAC inliers < 3 | `ransac_rigid` return | Identity transform, warning; Before-only visualization still produced |
| CUDA OOM | try/except around forward | Suggest `--device cpu` |

## 8. Success criteria (no GT available)

Three complementary signals, all three must be considered together:

1. **Matching density and coherence** (`*_match.png`): lines following object contours consistently indicate success; randomly scattered or locally clustered lines indicate failure. Expected: successful runs have tens to hundreds of matches.

2. **Registration alignment** (`*_reg.png`): in the "Estimated R,t" row, master (blue) and aligned scanned (red) should visually overlap across all three projections (Top-down, Front, Side). Partial overlap (e.g., only one view) suggests a plane fit, which is weak evidence. Before-row and Estimated-row should look clearly different.

3. **Proxy overlap rate** (`summary.csv`): absolute values are not meaningful; only relative comparisons matter.
   - Hypothesis 1: `overlap_rate(dzfix) > overlap_rate(raw)` would confirm the dz mismatch mattered.
   - Hypothesis 2: the larger of `overlap_rate(master=0)` vs `overlap_rate(master=594)` indicates which master viewpoint is closer to the scanned viewpoint.

## 9. Execution plan

### Phase 1: Implementation
1. `preprocess_scanned` + save intermediate PNG for visual sanity check
2. `extract_scanned_features` + log `n_iss, n_valid` to confirm ISS detection works on scanned data
3. `load_master_features` + inspect loaded dict keys and shapes
4. `build_batch` + single model forward pass succeeding (no match count check yet)
5. `plot_matches_no_gt` + first match visualization file created
6. `estimate_registration` + `compute_overlap_rate` + first reg visualization
7. `--run_both` loop + `summary.csv` population

### Phase 2: Dry-run sanity check (optional but recommended)
Run the new script replacing the scanned branch with a second *master* image (e.g., use `depth_raw_0594.png` as the "scanned" input, but load it through the ScannedPreprocessor path *without* dz correction). This exercises the new code path on known-good master data and should produce registration quality comparable to `test_resample2_iss_fpfh_0406.py` for the same pair. If the dry-run fails while the original test script passes on the same pair, the bug is in the new pipeline (not the data).

### Phase 3: Real inference
```
python test_scanned_vs_master.py --run_both
```
- Inspect 8 PNGs and `summary.csv`
- Compare `raw` vs `dzfix` to validate dz-correction hypothesis
- Compare `master0000` vs `master0594` to determine which viewpoint is closer
- Based on results, decide next iteration: expand master indices, re-tune `dz_ratio`, or revisit rejected ortho approach

## 10. Explicitly out of scope

- Formal scanned Dataset class (over-engineering for a single file)
- Scanned FPFH precompute script (runtime computation is simpler given the experiment toggles)
- Training or fine-tuning (inference only)
- Multiple scanned files in one invocation (extend if needed later)
- Pure orthographic experiment (already rejected — equivalent to documented "fake 3D" failure mode)
- Comparison with other descriptors (SHOT, ROPS)
- Quantitative metrics (RMSE, recall@k) — no GT exists
- Batch size > 1 — single pair per forward pass

## 11. Future work (if Phase 3 fails)

- If both `raw` and `dzfix` fail across both masters: revisit orthographic computation with a new dedicated FPFH model trained specifically on ortho data (significant effort, out of this spec)
- If only one master succeeds: expand master indices around the successful one, using an exploration-then-focus strategy
- If matches exist but registration fails: inspect RANSAC inlier count distribution, consider ICP refinement as a post-processing step
- If dz_ratio ≠ 0.425 is needed: make it a search parameter, sweep values and pick by overlap_rate
