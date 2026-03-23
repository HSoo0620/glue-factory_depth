"""
Precomputed FPFH descriptor를 로드하는 depth 데이터셋.
SuperPoint keypoints + FPFH descriptors가 .npz로 캐시되어 있어야 함.
(precompute_fpfh.py로 생성)
"""

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import cv2


class MitsubishiDepthFPFHDataset(Dataset):

    def __init__(self, split="train", cache_dir=None, fpfh_radius=1.5):
        self.root = Path("./gluefactory/datasets/mitsubishi/outputs")
        self.base_dir = Path("./gluefactory/datasets/mitsubishi")
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self.base_dir / f"fpfh_cache_r{fpfh_radius}"

        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"FPFH cache not found: {self.cache_dir}\n"
                f"Run: python precompute_fpfh.py --fpfh_radius {fpfh_radius}"
            )

        combo = pd.read_csv(self.root / "combination.csv")

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

    def _load_fpfh_cache(self, img_rel_path):
        """이미지 경로로부터 캐시된 .npz 로드"""
        stem = Path(img_rel_path).stem  # e.g. depth_raw_0000
        npz_path = self.cache_dir / f"{stem}.npz"
        data = np.load(npz_path)
        return {
            "keypoints": data["keypoints"],           # (N, 2)
            "keypoint_scores": data["keypoint_scores"], # (N,)
            "descriptors": data["fpfh_descriptors"],   # (N, 33)
        }

    def __len__(self):
        return len(self.combo)

    def __getitem__(self, idx):
        row = self.combo.iloc[idx]

        master_path = self.base_dir / row["master_path"]
        input_path = self.base_dir / row["input_path"]
        csv_path = self.root / Path(row["csv_path"]).name

        # 이미지 로드 (시각화/GT용)
        master = cv2.imread(str(master_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        input_img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        master = master / 65535.0
        input_img = input_img / 65535.0

        # GT 로드
        corr = pd.read_csv(csv_path)
        valid = corr["occluded"] == False
        master_xy = corr.loc[valid, ["master_x", "master_y"]].values.astype(np.float32)
        input_xy = corr.loc[valid, ["input_x", "input_y"]].values.astype(np.float32)
        gt_matches = np.concatenate([master_xy, input_xy], axis=1)

        h, w = master.shape[:2]

        # Precomputed FPFH 로드
        cache0 = self._load_fpfh_cache(row["master_path"])
        cache1 = self._load_fpfh_cache(row["input_path"])

        data = {
            "view0": {
                "image": torch.from_numpy(master).unsqueeze(0),
                "image_size": torch.tensor([h, w]),
                # precomputed SP keypoints + FPFH descriptors
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


def mitsubishi_fpfh_collate_fn(batch):
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
