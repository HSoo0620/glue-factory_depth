import os
import pandas as pd
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class DepthPairDataset(Dataset):

    def __init__(self, root_dir):

        self.root = root_dir
        combo_path = os.path.join(root_dir, "./datasets/mitsubishi/outputs/combination.csv")
        self.combo = pd.read_csv(combo_path)

    def __len__(self):
        return len(self.combo)

    def load_depth(self, path):
        full = os.path.join(self.root, path)
        img = cv2.imread(full, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found: {full}")
        img = img.astype(np.float32)
        if img.max() > 0:
            # img = img / img.max()
            img = img / 50000
        img = torch.from_numpy(img)[None]  # [1,H,W]
        return img

    def load_gt(self, csv_path):

        full = os.path.join(self.root, csv_path)

        df = pd.read_csv(full)

        if df['occluded'].dtype == object:
            df['occluded'] = df['occluded'].map({'TRUE': True, 'FALSE': False})

        df = df[df["occluded"] == False]

        master = df[["master_x", "master_y"]].values
        inp = df[["input_x", "input_y"]].values

        gt = np.concatenate([master, inp], axis=1)

        gt = torch.tensor(gt).float()   # (M,4)

        return gt

    def __getitem__(self, idx):

        row = self.combo.iloc[idx]

        master_path = row["master_path"]
        input_path = row["input_path"]
        csv_path = row["csv_path"]

        img0 = self.load_depth(master_path)
        img1 = self.load_depth(input_path)

        gt_matches = self.load_gt(csv_path)

        data = {
            "view0": {
                "image": img0
            },
            "view1": {
                "image": img1
            },
            "gt_matches": gt_matches
        }

        return data