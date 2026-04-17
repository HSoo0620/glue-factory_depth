"""
Resample_2 이미지 기반 ISS+RoPS135 캐시 데이터셋.
precompute_iss_rops_resample2.py로 캐시 생성 후 사용.
"""

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import cv2

CROP_X0 = 1129
CROP_Y0 = 1081
CROP_SIZE = 3502


class MitsubishiResample2ISSRoPSDataset(Dataset):

    def __init__(self, split="train", cache_dir=None,
                 image_size=None, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path("./gluefactory/datasets/mitsubishi")
        self.image_dir = self.base_dir / "dataset_resample_2"
        self.output_dir = self.base_dir / "outputs_txt"
        self.image_size = image_size

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self.base_dir / "iss_rops135_resample2_cache"

        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"ISS+RoPS cache not found: {self.cache_dir}\n"
                f"Run: python precompute_iss_rops_resample2.py"
            )

        combo = pd.read_csv(self.output_dir / "combination.csv")
        total = len(combo)
        val_size = 100; test_size = 100
        train_end = total - val_size - test_size
        val_end = total - test_size

        if split == "train":    combo = combo.iloc[:train_end]
        elif split == "val":    combo = combo.iloc[train_end:val_end]
        elif split == "test":   combo = combo.iloc[val_end:]
        else: raise ValueError(split)
        self.combo = combo.reset_index(drop=True)

    def _extract_filename(self, path_str):
        return Path(path_str).name

    def _load_cache(self, img_path_str):
        stem = Path(img_path_str).stem
        data = np.load(self.cache_dir / f"{stem}.npz")
        return {
            "keypoints":       data["keypoints"],
            "keypoint_scores": data["keypoint_scores"],
            "descriptors":     data["rops_descriptors"],
        }

    def __len__(self):
        return len(self.combo)

    def __getitem__(self, idx):
        row = self.combo.iloc[idx]
        master_path = self.image_dir / self._extract_filename(row["master_path"])
        input_path  = self.image_dir / self._extract_filename(row["input_path"])
        csv_path    = self.output_dir / Path(row["csv_path"]).name

        master_raw = cv2.imread(str(master_path), cv2.IMREAD_UNCHANGED)
        input_raw  = cv2.imread(str(input_path),  cv2.IMREAD_UNCHANGED)
        master_crop = master_raw[CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]
        input_crop  = input_raw [CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]

        if self.image_size is not None and self.image_size != CROP_SIZE:
            master_crop = cv2.resize(master_crop, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            input_crop  = cv2.resize(input_crop,  (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            output_size = self.image_size
        else:
            output_size = CROP_SIZE

        scale = output_size / CROP_SIZE
        master = master_crop.astype(np.float32) / 65535.0
        input_img = input_crop.astype(np.float32) / 65535.0
        h, w = master.shape[:2]

        corr = pd.read_csv(csv_path)
        valid = corr["occluded"] == False
        master_xy = corr.loc[valid, ["master_x", "master_y"]].values.astype(np.float32)
        input_xy  = corr.loc[valid, ["input_x",  "input_y" ]].values.astype(np.float32)
        master_xy[:, 0] -= CROP_X0; master_xy[:, 1] -= CROP_Y0
        input_xy[:,  0] -= CROP_X0; input_xy[:,  1] -= CROP_Y0
        if scale != 1.0:
            master_xy *= scale; input_xy *= scale

        in_bounds = (
            (master_xy[:,0] >= 0) & (master_xy[:,0] < output_size) &
            (master_xy[:,1] >= 0) & (master_xy[:,1] < output_size) &
            (input_xy[:,0]  >= 0) & (input_xy[:,0]  < output_size) &
            (input_xy[:,1]  >= 0) & (input_xy[:,1]  < output_size)
        )
        master_xy = master_xy[in_bounds]; input_xy = input_xy[in_bounds]
        gt_matches = np.concatenate([master_xy, input_xy], axis=1)

        cache0 = self._load_cache(row["master_path"])
        cache1 = self._load_cache(row["input_path"])

        return {
            "view0": {
                "image":           torch.from_numpy(master).unsqueeze(0),
                "image_size":      torch.tensor([h, w]),
                "keypoints":       torch.from_numpy(cache0["keypoints"]).float(),
                "keypoint_scores": torch.from_numpy(cache0["keypoint_scores"]).float(),
                "descriptors":     torch.from_numpy(cache0["descriptors"]).float(),
            },
            "view1": {
                "image":           torch.from_numpy(input_img).unsqueeze(0),
                "image_size":      torch.tensor([h, w]),
                "keypoints":       torch.from_numpy(cache1["keypoints"]).float(),
                "keypoint_scores": torch.from_numpy(cache1["keypoint_scores"]).float(),
                "descriptors":     torch.from_numpy(cache1["descriptors"]).float(),
            },
            "gt_matches":   torch.from_numpy(gt_matches).float(),
            "csv_path":     str(csv_path),
            "master_path":  str(master_path),
            "input_path":   str(input_path),
        }


def resample2_iss_rops_collate_fn(batch):
    return {
        "view0": {
            "image":           torch.stack([b["view0"]["image"]           for b in batch]),
            "image_size":      torch.stack([b["view0"]["image_size"]      for b in batch]),
            "keypoints":       torch.stack([b["view0"]["keypoints"]       for b in batch]),
            "keypoint_scores": torch.stack([b["view0"]["keypoint_scores"] for b in batch]),
            "descriptors":     torch.stack([b["view0"]["descriptors"]     for b in batch]),
        },
        "view1": {
            "image":           torch.stack([b["view1"]["image"]           for b in batch]),
            "image_size":      torch.stack([b["view1"]["image_size"]      for b in batch]),
            "keypoints":       torch.stack([b["view1"]["keypoints"]       for b in batch]),
            "keypoint_scores": torch.stack([b["view1"]["keypoint_scores"] for b in batch]),
            "descriptors":     torch.stack([b["view1"]["descriptors"]     for b in batch]),
        },
        "gt_matches": pad_sequence([b["gt_matches"] for b in batch], batch_first=True, padding_value=0.0),
        "csv_path":    [b["csv_path"]   for b in batch],
        "master_path": [b["master_path"] for b in batch],
        "input_path":  [b["input_path"]  for b in batch],
    }
