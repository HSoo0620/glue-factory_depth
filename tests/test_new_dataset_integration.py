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
        # H half of 1556..2975; W=round(2413*0.5)=1206 (Python banker's rounding on 1206.5)
        assert 700 <= H <= 1500 and W == 1206


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
