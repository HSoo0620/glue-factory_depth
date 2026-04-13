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
