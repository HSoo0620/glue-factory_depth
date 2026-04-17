"""v1 (2026-04-15) ISS+FPFH dataset.

NAS dataset_output 기반.
- Zero-pad to (W=2432, H=3008).
- GT: pair_*.csv (occluded=False).
- Cache: iss_shot_v1_20260415_cache_vox1_nr20_sr40/{stem}.npz
- Split: gluefactory/datasets/mitsubishi/splits/v1_20260415_split.json
"""
import json
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from PIL import Image

PAD_W = 2432
PAD_H = 3008
ZMAP_RE = re.compile(r"zmap_(\d{4})\.png$")


def _zmap_id(path_str):
    m = ZMAP_RE.search(str(path_str))
    if not m:
        raise ValueError(f"cannot parse zmap_id from {path_str}")
    return int(m.group(1))


def _zero_pad(img_u16):
    H, W = img_u16.shape
    out = np.zeros((PAD_H, PAD_W), dtype=img_u16.dtype)
    out[:H, :W] = img_u16
    return out


class MitsubishiV1ISSSHOTDataset(Dataset):
    def __init__(self, split="train",
                 data_root="/mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output",
                 cache_dir=None,
                 split_json=None):
        self.data_root = Path(data_root)
        if cache_dir is None:
            cache_dir = ("/home/jhs/work/Registration/glue-factory_depth/"
                         "gluefactory/datasets/mitsubishi/"
                         "iss_shot_v1_20260415_cache_vox1_nr20_sr40")
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"cache not found: {self.cache_dir}")

        if split_json is None:
            split_json = ("/home/jhs/work/Registration/glue-factory_depth/"
                          "gluefactory/datasets/mitsubishi/splits/v1_20260415_split.json")
        s = json.loads(Path(split_json).read_text())
        self.split = split
        if split == "train":
            id_set = set(s["train_ids"])
        elif split == "val":
            id_set = set(s["val_ids"])
        elif split == "test":
            id_set = set(s["test_ids"])
        else:
            raise ValueError(split)
        self.id_set = id_set

        combo = pd.read_csv(self.data_root / "combination.csv")
        m_ids = combo["master_zmap_path"].map(_zmap_id)
        i_ids = combo["input_zmap_path"].map(_zmap_id)
        keep = m_ids.isin(id_set) & i_ids.isin(id_set)
        self.combo = combo[keep].reset_index(drop=True)

    def __len__(self):
        return len(self.combo)

    def _load_image(self, fname):
        img = np.array(Image.open(self.data_root / fname))
        if img.dtype == np.uint8:
            img = img.astype(np.uint16) * 257
        pad = _zero_pad(img)
        return pad.astype(np.float32) / 65535.0

    def _load_cache(self, fname):
        stem = Path(fname).stem
        d = np.load(self.cache_dir / f"{stem}.npz")
        return {
            "keypoints": d["keypoints"],
            "keypoint_scores": d["keypoint_scores"],
            "descriptors": d["descriptors"],
        }

    def _load_gt(self, csv_fname):
        df = pd.read_csv(self.data_root / csv_fname)
        valid = df["occluded"] == False
        m = df.loc[valid, ["master_x", "master_y"]].values.astype(np.float32)
        i = df.loc[valid, ["input_x", "input_y"]].values.astype(np.float32)
        return np.concatenate([m, i], axis=1)

    def __getitem__(self, idx):
        row = self.combo.iloc[idx]
        master_fname = row["master_zmap_path"]
        input_fname = row["input_zmap_path"]
        csv_fname = row["csv_path"]

        master = self._load_image(master_fname)
        inp = self._load_image(input_fname)
        c0 = self._load_cache(master_fname)
        c1 = self._load_cache(input_fname)
        gt = self._load_gt(csv_fname)

        return {
            "view0": {
                "image": torch.from_numpy(master).unsqueeze(0),
                "image_size": torch.tensor([PAD_H, PAD_W]),
                "keypoints": torch.from_numpy(c0["keypoints"]).float(),
                "keypoint_scores": torch.from_numpy(c0["keypoint_scores"]).float(),
                "descriptors": torch.from_numpy(c0["descriptors"]).float(),
            },
            "view1": {
                "image": torch.from_numpy(inp).unsqueeze(0),
                "image_size": torch.tensor([PAD_H, PAD_W]),
                "keypoints": torch.from_numpy(c1["keypoints"]).float(),
                "keypoint_scores": torch.from_numpy(c1["keypoint_scores"]).float(),
                "descriptors": torch.from_numpy(c1["descriptors"]).float(),
            },
            "gt_matches": torch.from_numpy(gt).float(),
            "csv_path": str(self.data_root / csv_fname),
            "master_path": str(self.data_root / master_fname),
            "input_path": str(self.data_root / input_fname),
        }


def v1_iss_shot_collate_fn(batch):
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
            "image": images0, "image_size": sizes0,
            "keypoints": kp0, "keypoint_scores": sc0, "descriptors": desc0,
        },
        "view1": {
            "image": images1, "image_size": sizes1,
            "keypoints": kp1, "keypoint_scores": sc1, "descriptors": desc1,
        },
        "gt_matches": gt_matches,
    }
