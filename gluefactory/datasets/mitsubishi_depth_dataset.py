import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import cv2
import numpy as np
from pathlib import Path


class MitsubishiDepthDataset(Dataset):

    def __init__(self, split="train"):

        self.root = Path("./gluefactory/datasets/mitsubishi/outputs")
        combo_path = self.root / "combination.csv"

        combo = pd.read_csv(combo_path)

        # 고정 seed로 셔플 → 재현 가능한 분할
        combo = combo.sample(frac=1, random_state=42).reset_index(drop=True)

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

        master_path = Path("./gluefactory/datasets/mitsubishi") / row["master_path"]
        input_path = Path("./gluefactory/datasets/mitsubishi") / row["input_path"]
        csv_path = self.root / Path(row["csv_path"]).name

        # Load depth as uint16, normalize to [0, 1] for SuperPoint
        master = cv2.imread(str(master_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        input_img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED).astype(np.float32)

        master = master / 65535.0
        input_img = input_img / 65535.0

        corr = pd.read_csv(csv_path)

        valid = corr["occluded"] == False

        master_xy = corr.loc[valid, ["master_x", "master_y"]].values.astype(np.float32)
        input_xy = corr.loc[valid, ["input_x", "input_y"]].values.astype(np.float32)

        gt_matches = np.concatenate([master_xy, input_xy], axis=1)

        h, w = master.shape[:2]

        data = {
            "view0": {
                "image": torch.from_numpy(master).unsqueeze(0),  # (1, H, W)
                "image_size": torch.tensor([h, w]),
            },
            "view1": {
                "image": torch.from_numpy(input_img).unsqueeze(0),  # (1, H, W)
                "image_size": torch.tensor([h, w]),
            },
            "gt_matches": torch.from_numpy(gt_matches).float(),  # (M, 4)
            "csv_path": str(csv_path),
            "master_path": str(master_path),
            "input_path": str(input_path),
        }

        return data


def mitsubishi_collate_fn(batch):

    images0 = torch.stack([b["view0"]["image"] for b in batch], dim=0)
    images1 = torch.stack([b["view1"]["image"] for b in batch], dim=0)

    sizes0 = torch.stack([b["view0"]["image_size"] for b in batch], dim=0)
    sizes1 = torch.stack([b["view1"]["image_size"] for b in batch], dim=0)

    # Pad gt_matches to the same length within the batch
    gt_list = [b["gt_matches"] for b in batch]
    gt_matches = pad_sequence(gt_list, batch_first=True, padding_value=0.0)  # (B, Mmax, 4)

    return {
        "view0": {"image": images0, "image_size": sizes0},
        "view1": {"image": images1, "image_size": sizes1},
        "gt_matches": gt_matches,
        "csv_path": [b["csv_path"] for b in batch],
        "master_path": [b["master_path"] for b in batch],
        "input_path": [b["input_path"] for b in batch],
    }
