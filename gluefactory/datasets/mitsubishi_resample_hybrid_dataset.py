"""
Resample 이미지 기반 Hybrid (SP+FPFH) 캐시 데이터셋.
precompute_hybrid_resample.py로 캐시 생성 후 사용.
SP descriptor(256D)와 FPFH descriptor(33D)를 concat하여 289D descriptor 반환.
"""

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import cv2


class MitsubishiResampleHybridDataset(Dataset):

    def __init__(self, split="train", cache_dir=None, fpfh_radius=0.5,
                 image_size=2880, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path("./gluefactory/datasets/mitsubishi")
        self.root = self.base_dir / "outputs"
        self.image_size = image_size

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self.base_dir / f"hybrid_resample_cache_r{fpfh_radius}"

        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"Hybrid cache not found: {self.cache_dir}\n"
                f"Run: python precompute_hybrid_resample.py --fpfh_radius {fpfh_radius}"
            )

        combo = pd.read_csv(self.root / "combination.csv")

        total = len(combo)
        val_size = 100
        test_size = 100
        train_end = total - val_size - test_size
        val_end = total - test_size

        if split == "train":
            combo = combo.iloc[:train_end]
        elif split == "val":
            combo = combo.iloc[train_end:val_end]
        elif split == "test":
            combo = combo.iloc[val_end:]
        else:
            raise ValueError(split)

        self.combo = combo.reset_index(drop=True)

    def _load_hybrid_cache(self, img_rel_path):
        stem = Path(img_rel_path).stem
        npz_path = self.cache_dir / f"{stem}.npz"
        data = np.load(npz_path)

        sp_desc = data["sp_descriptors"]       # (N, 256)
        fpfh_desc = data["fpfh_descriptors"]   # (N, 33)
        descriptors = np.concatenate([sp_desc, fpfh_desc], axis=1)  # (N, 289)

        return {
            "keypoints": data["keypoints"],             # (N, 2)
            "keypoint_scores": data["keypoint_scores"],  # (N,)
            "descriptors": descriptors,                  # (N, 289)
        }

    def __len__(self):
        return len(self.combo)

    def __getitem__(self, idx):
        row = self.combo.iloc[idx]

        master_path = self.base_dir / row["master_path"]
        input_path = self.base_dir / row["input_path"]
        csv_path = self.root / Path(row["csv_path"]).name

        # 이미지 로드 + 리사이즈
        master_raw = cv2.imread(str(master_path), cv2.IMREAD_UNCHANGED)
        input_raw = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)

        orig_h, orig_w = master_raw.shape[:2]

        if self.image_size != orig_h or self.image_size != orig_w:
            master_resized = cv2.resize(
                master_raw, (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST
            )
            input_resized = cv2.resize(
                input_raw, (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST
            )
            scale_x = self.image_size / orig_w
            scale_y = self.image_size / orig_h
        else:
            master_resized = master_raw
            input_resized = input_raw
            scale_x = 1.0
            scale_y = 1.0

        master = master_resized.astype(np.float32) / 65535.0
        input_img = input_resized.astype(np.float32) / 65535.0

        h, w = master.shape[:2]

        # GT 로드 + 스케일링
        corr = pd.read_csv(csv_path)
        valid = corr["occluded"] == False
        master_xy = corr.loc[valid, ["master_x", "master_y"]].values.astype(np.float32)
        input_xy = corr.loc[valid, ["input_x", "input_y"]].values.astype(np.float32)
        master_xy[:, 0] *= scale_x
        master_xy[:, 1] *= scale_y
        input_xy[:, 0] *= scale_x
        input_xy[:, 1] *= scale_y
        gt_matches = np.concatenate([master_xy, input_xy], axis=1)

        # Hybrid 캐시 로드 (keypoints는 이미 image_size 기준)
        cache0 = self._load_hybrid_cache(row["master_path"])
        cache1 = self._load_hybrid_cache(row["input_path"])

        data = {
            "view0": {
                "image": torch.from_numpy(master).unsqueeze(0),
                "image_size": torch.tensor([h, w]),
                "keypoints": torch.from_numpy(cache0["keypoints"]).float(),
                "keypoint_scores": torch.from_numpy(cache0["keypoint_scores"]).float(),
                "descriptors": torch.from_numpy(cache0["descriptors"]).float(),
            },
            "view1": {
                "image": torch.from_numpy(input_img).unsqueeze(0),
                "image_size": torch.tensor([h, w]),
                "keypoints": torch.from_numpy(cache1["keypoints"]).float(),
                "keypoint_scores": torch.from_numpy(cache1["keypoint_scores"]).float(),
                "descriptors": torch.from_numpy(cache1["descriptors"]).float(),
            },
            "gt_matches": torch.from_numpy(gt_matches).float(),
            "csv_path": str(csv_path),
        }

        return data


def resample_hybrid_collate_fn(batch):
    images0 = torch.stack([b["view0"]["image"] for b in batch], dim=0)
    images1 = torch.stack([b["view1"]["image"] for b in batch], dim=0)

    sizes0 = torch.stack([b["view0"]["image_size"] for b in batch], dim=0)
    sizes1 = torch.stack([b["view1"]["image_size"] for b in batch], dim=0)

    kp0 = torch.stack([b["view0"]["keypoints"] for b in batch], dim=0)
    kp1 = torch.stack([b["view1"]["keypoints"] for b in batch], dim=0)
    sc0 = torch.stack([b["view0"]["keypoint_scores"] for b in batch], dim=0)
    sc1 = torch.stack([b["view1"]["keypoint_scores"] for b in batch], dim=0)
    desc0 = torch.stack([b["view0"]["descriptors"] for b in batch], dim=0)
    desc1 = torch.stack([b["view1"]["descriptors"] for b in batch], dim=0)

    gt_list = [b["gt_matches"] for b in batch]
    gt_matches = pad_sequence(gt_list, batch_first=True, padding_value=0.0)

    return {
        "view0": {
            "image": images0,
            "image_size": sizes0,
            "keypoints": kp0,
            "keypoint_scores": sc0,
            "descriptors": desc0,
        },
        "view1": {
            "image": images1,
            "image_size": sizes1,
            "keypoints": kp1,
            "keypoint_scores": sc1,
            "descriptors": desc1,
        },
        "gt_matches": gt_matches,
        "csv_path": [b["csv_path"] for b in batch],
    }
