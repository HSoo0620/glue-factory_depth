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
            split_scenes = set(train_scenes)
        elif split == "val":
            split_scenes = set(val_scenes)
        else:
            raise ValueError(f"Unknown split: {split!r} (expected 'train' or 'val')")

        cached_scenes = {
            int(p.stem.split("_")[-1])
            for p in self.cache_dir.glob("zmap_*.npz")
        }
        usable = split_scenes & cached_scenes
        combo = filter_pairs_by_scenes(combo, usable)
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
