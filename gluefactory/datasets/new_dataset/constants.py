"""Immutable constants for the new dataset pipeline."""
from pathlib import Path

LAT_MM = 0.056
TRANS_MM = 0.056
VERT_MM = 0.0085

RESIZE_FACTOR = 0.5

N_SCENES = 641

DESC_DIMS = {"fpfh": 33, "shot": 352}

DEFAULT_DATA_ROOT = Path("/mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output")
