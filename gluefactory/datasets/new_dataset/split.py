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
