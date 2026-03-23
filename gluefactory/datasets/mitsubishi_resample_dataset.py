"""
Resample (5761×5761) depth 이미지를 사용하는 데이터셋.
이미지를 image_size로 리사이즈하고, GT 좌표를 비례 스케일링.

combination.csv의 경로가 dataset_resample/ 을 가리킴.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import cv2
import numpy as np
from pathlib import Path


class MitsubishiResampleDataset(Dataset):

    def __init__(self, split="train", image_size=2880, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path("./gluefactory/datasets/mitsubishi")
        self.root = self.base_dir / "outputs"
        self.image_size = image_size

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

    def __len__(self):
        return len(self.combo)

    def __getitem__(self, idx):
        row = self.combo.iloc[idx]

        master_path = self.base_dir / row["master_path"]
        input_path = self.base_dir / row["input_path"]
        csv_path = self.root / Path(row["csv_path"]).name

        # Load depth as uint16
        master_raw = cv2.imread(str(master_path), cv2.IMREAD_UNCHANGED)
        input_raw = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)

        orig_h, orig_w = master_raw.shape[:2]

        # Resize to target size
        if self.image_size != orig_h or self.image_size != orig_w:
            master_resized = cv2.resize(
                master_raw, (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST  # depth는 nearest로
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

        # Normalize to [0, 1]
        master = master_resized.astype(np.float32) / 65535.0
        input_img = input_resized.astype(np.float32) / 65535.0

        h, w = master.shape[:2]

        # GT 로드 + 좌표 스케일링
        corr = pd.read_csv(csv_path)
        valid = corr["occluded"] == False

        master_xy = corr.loc[valid, ["master_x", "master_y"]].values.astype(np.float32)
        input_xy = corr.loc[valid, ["input_x", "input_y"]].values.astype(np.float32)

        # 스케일링 적용
        master_xy[:, 0] *= scale_x
        master_xy[:, 1] *= scale_y
        input_xy[:, 0] *= scale_x
        input_xy[:, 1] *= scale_y

        gt_matches = np.concatenate([master_xy, input_xy], axis=1)

        data = {
            "view0": {
                "image": torch.from_numpy(master).unsqueeze(0),  # (1, H, W)
                "image_size": torch.tensor([h, w]),
            },
            "view1": {
                "image": torch.from_numpy(input_img).unsqueeze(0),
                "image_size": torch.tensor([h, w]),
            },
            "gt_matches": torch.from_numpy(gt_matches).float(),  # (M, 4)
            "csv_path": str(csv_path),
            "master_path": str(master_path),
            "input_path": str(input_path),
            "orig_size": orig_h,  # 원본 해상도 (스케일 복원용)
        }

        return data


def resample_collate_fn(batch):

    images0 = torch.stack([b["view0"]["image"] for b in batch], dim=0)
    images1 = torch.stack([b["view1"]["image"] for b in batch], dim=0)

    sizes0 = torch.stack([b["view0"]["image_size"] for b in batch], dim=0)
    sizes1 = torch.stack([b["view1"]["image_size"] for b in batch], dim=0)

    gt_list = [b["gt_matches"] for b in batch]
    gt_matches = pad_sequence(gt_list, batch_first=True, padding_value=0.0)

    return {
        "view0": {"image": images0, "image_size": sizes0},
        "view1": {"image": images1, "image_size": sizes1},
        "gt_matches": gt_matches,
        "csv_path": [b["csv_path"] for b in batch],
        "master_path": [b["master_path"] for b in batch],
        "input_path": [b["input_path"] for b in batch],
    }
