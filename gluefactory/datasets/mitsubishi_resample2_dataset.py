"""
Resample_2 (5761x5761) depth 이미지를 사용하는 데이터셋.
고정 crop (1129, 1081) ~ (4631, 4583) -> 3502x3502 적용.
선택적으로 image_size로 리사이즈 가능.

combination.csv의 경로를 로컬 경로로 변환.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import cv2
import numpy as np
from pathlib import Path


# 고정 crop 파라미터 (전체 641장 nonzero bbox + 마진 기반)
CROP_X0 = 1129
CROP_Y0 = 1081
CROP_SIZE = 3502


class MitsubishiResample2Dataset(Dataset):

    def __init__(self, split="train", image_size=None, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path("./gluefactory/datasets/mitsubishi")
        self.image_dir = self.base_dir / "dataset_resample_2"
        self.output_dir = self.base_dir / "outputs_resample_2"
        self.image_size = image_size  # None이면 crop 크기(3502) 그대로

        combo = pd.read_csv(self.output_dir / "combination.csv")

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

    def _extract_filename(self, path_str):
        """combination.csv의 절대경로에서 파일명만 추출"""
        return Path(path_str).name

    def __getitem__(self, idx):
        row = self.combo.iloc[idx]

        # 경로 변환: 절대경로 -> 로컬 파일명
        master_fname = self._extract_filename(row["master_path"])
        input_fname = self._extract_filename(row["input_path"])
        master_path = self.image_dir / master_fname
        input_path = self.image_dir / input_fname

        # csv_path 변환: outputs_resample2/00000.csv -> outputs_resample_2/00000.csv
        csv_fname = Path(row["csv_path"]).name
        csv_path = self.output_dir / csv_fname

        # Load depth as uint16
        master_raw = cv2.imread(str(master_path), cv2.IMREAD_UNCHANGED)
        input_raw = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)

        # 고정 crop 적용
        master_crop = master_raw[CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]
        input_crop = input_raw[CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]

        # 리사이즈 (옵션)
        if self.image_size is not None and self.image_size != CROP_SIZE:
            master_crop = cv2.resize(
                master_crop, (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST
            )
            input_crop = cv2.resize(
                input_crop, (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST
            )
            output_size = self.image_size
        else:
            output_size = CROP_SIZE

        scale = output_size / CROP_SIZE

        # Normalize to [0, 1]
        master = master_crop.astype(np.float32) / 65535.0
        input_img = input_crop.astype(np.float32) / 65535.0

        h, w = master.shape[:2]

        # GT 로드 + crop offset + 스케일링
        corr = pd.read_csv(csv_path)
        valid = corr["occluded"] == False

        master_xy = corr.loc[valid, ["master_x", "master_y"]].values.astype(np.float32)
        input_xy = corr.loc[valid, ["input_x", "input_y"]].values.astype(np.float32)

        # crop offset 적용
        master_xy[:, 0] -= CROP_X0
        master_xy[:, 1] -= CROP_Y0
        input_xy[:, 0] -= CROP_X0
        input_xy[:, 1] -= CROP_Y0

        # 리사이즈 스케일링
        if scale != 1.0:
            master_xy *= scale
            input_xy *= scale

        # crop 범위 밖 GT 제거
        in_bounds = (
            (master_xy[:, 0] >= 0) & (master_xy[:, 0] < output_size) &
            (master_xy[:, 1] >= 0) & (master_xy[:, 1] < output_size) &
            (input_xy[:, 0] >= 0) & (input_xy[:, 0] < output_size) &
            (input_xy[:, 1] >= 0) & (input_xy[:, 1] < output_size)
        )
        master_xy = master_xy[in_bounds]
        input_xy = input_xy[in_bounds]

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
            "orig_size": CROP_SIZE,  # crop 후 원본 크기
        }

        return data


def resample2_collate_fn(batch):

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
