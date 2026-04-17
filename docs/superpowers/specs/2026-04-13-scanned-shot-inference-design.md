# Scanned Data + SHOT Descriptor Inference

Date: 2026-04-13

## Goal

Duplicate `test_scanned_vs_master.py` (FPFH-based) to create `test_scanned_vs_master_shot.py` that uses precomputed SHOT352 descriptors for scanned-vs-master depth matching inference.

## Context

- Existing FPFH script computes descriptors at runtime via Open3D
- SHOT352 descriptors are precomputed externally (C++ PCL) and stored as `.bin` files
- Model: `0407_resample2_iss_shot352_lg` (352D input, best recall 0.547)

## Key Decision: flip_scanned_depth = False

The scanned SHOT `.bin` was computed on **original (non-flipped)** scanned data. Flipping depth at runtime would create a coordinate mismatch with the precomputed descriptors (both KDTree lookup XYZ and surface normals). Therefore `flip_scanned_depth` defaults to `False` for now. If needed, the `.bin` can be regenerated from flipped data later.

## Changes from FPFH Version

| Area | FPFH | SHOT |
|---|---|---|
| Scanned descriptor | Runtime Open3D FPFH (33D) | `.bin` load + KDTree lookup (352D) |
| Master cache | `iss_fpfh_resample2_cache_r5.0_xyz/` (`fpfh_descriptors`) | `iss_shot352_resample2_cache/` (`shot_descriptors`) |
| Model checkpoint | `0406_resample2_iss_fpfh_xyz_lg` | `0407_resample2_iss_shot352_lg` |
| Collate function | `resample2_iss_fpfh_collate_fn` | `resample2_iss_shot_collate_fn` |
| Imports | `precompute_iss_fpfh_resample2` | `precompute_iss_shot352_resample2` |
| flip_scanned_depth | True | False |

## Scanned Feature Extraction (SHOT)

1. `preprocess_scanned()` — unchanged (called with flip=False)
2. ISS keypoint detection — same functions from `precompute_iss_shot352_resample2`
3. `kp_crop_to_xyz()` — keypoint to XYZ conversion
4. `load_shot352_bin()` — load scanned `.bin` (points + descriptors)
5. `lookup_shot352()` — KDTree nearest-neighbor lookup for SHOT descriptors

## Master Feature Loading

Load from `iss_shot352_resample2_cache/{stem}.npz`:
- Key: `shot_descriptors` (512, 352) instead of `fpfh_descriptors`

## CLI Parameters

```
--scanned_path          gluefactory/datasets/scanned/scanned_data1.png
--scanned_bin_path      gluefactory/datasets/Descriptor/output_shot_scanned/scanned_data1_shot352.bin
--master_bin_dir        gluefactory/datasets/Descriptor/output_shot_0407
--checkpoint            outputs/training/0407_resample2_iss_shot352_lg/checkpoint_best.tar
--cache_dir             gluefactory/datasets/mitsubishi/iss_shot352_resample2_cache
--master_indices        0 73 88 102 594
--run_both              (raw + dzfix)
--output_dir            results/scanned/scanned_data1_shot
```

## Output

```
results/scanned/scanned_data1_shot/
├── raw/
│   ├── master0000_match.png, master0000_reg.png, ...
├── dzfix/
│   └── ...
└── summary.csv
```

## Unchanged Components

- `preprocess_scanned()`, `mask_scanned_table()`
- `estimate_registration()`, `compute_overlap_rate()`
- `plot_matches_no_gt()`, `plot_alignment_no_gt()`
- `run_single_experiment()` structure, `main()` structure
