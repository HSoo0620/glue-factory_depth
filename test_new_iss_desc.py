"""Visualize trained ISS+FPFH/SHOT+LG matches on the new dataset.

4-color convention (same as Mitsubishi):
  skyblue  = correct match (master-side within gt_radius AND input-side within gt_radius, not occluded)
  purple   = wrong         (master-side matched, input-side too far)
  limegreen= occluded      (master-side matched, input-side close, BUT occluded=1)
  red      = no-GT         (no master-side GT keypoint within gt_radius)

Usage:
    python test_new_iss_desc.py --experiment 0413_new_iss_fpfh_lg --num_samples 10
    python test_new_iss_desc.py --checkpoint outputs/training/0413_new_iss_fpfh_lg/checkpoint_best.tar --indices 0 5 10
    python test_new_iss_desc.py --experiment 0413_new_iss_shot_lg --descriptor_type shot --num_samples 5
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.datasets.new_dataset import (
    NewDatasetISSDescDataset, collate_fn_dynamic_pad,
)
from gluefactory.datasets.new_dataset import constants as NEW_C
from gluefactory.utils.tensor import batch_to_device


def load_model(checkpoint_path: str, device: str):
    cp = torch.load(checkpoint_path, map_location="cpu")
    conf = OmegaConf.create(cp["conf"])
    model = get_model(conf.model.name)(conf.model).to(device)
    model.load_state_dict(cp["model"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {cp['epoch']})")
    return model, conf


@torch.no_grad()
def run_inference(model, batch, device):
    batch = batch_to_device(batch, device)
    pred = model(batch)
    return pred, batch


def visualize_pair(pred, data, idx: int, output_path: Path,
                   csv_path: str | None = None, gt_radius: int = 20,
                   resize_factor: float = 0.5):
    img0 = data["view0"]["image"][idx].cpu()
    img1 = data["view1"]["image"][idx].cpu()
    h0, w0 = data["view0"]["image_size"][idx].tolist()
    h1, w1 = data["view1"]["image_size"][idx].tolist()
    # crop off padding for display
    img0_np = img0.squeeze(0).numpy()[:h0, :w0]
    img1_np = img1.squeeze(0).numpy()[:h1, :w1]

    kp0 = pred["keypoints0"][idx].cpu().numpy()
    kp1 = pred["keypoints1"][idx].cpu().numpy()
    m0 = pred["matches0"][idx].cpu().numpy()

    valid = m0 > -1
    mkp0 = kp0[valid]
    mkp1 = kp1[m0[valid]]
    n_total = int(valid.sum())

    n_correct = n_wrong = n_occluded = n_no_gt = 0
    gt_pos_total = 0
    colors = ["red"] * n_total

    if csv_path is not None and Path(csv_path).exists():
        corr = pd.read_csv(csv_path)
        all_m = corr[["master_x", "master_y"]].values.astype(np.float32) * resize_factor
        all_i = corr[["input_x", "input_y"]].values.astype(np.float32) * resize_factor
        all_occ = (corr["occluded"].astype(int) == 1).values
        gt_pos_total = int((~all_occ).sum())

        for i in range(n_total):
            dists = np.linalg.norm(all_m - mkp0[i], axis=1)
            j = int(np.argmin(dists))
            if dists[j] < gt_radius:
                d_input = float(np.linalg.norm(mkp1[i] - all_i[j]))
                is_occ = bool(all_occ[j])
                if d_input < gt_radius:
                    if is_occ:
                        colors[i] = "limegreen"; n_occluded += 1
                    else:
                        colors[i] = "skyblue"; n_correct += 1
                else:
                    colors[i] = "purple"; n_wrong += 1
            else:
                n_no_gt += 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
    for ax, img_np, kp, title in [
        (axes[0], img0_np, kp0, "View 0 (Master)"),
        (axes[1], img1_np, kp1, "View 1 (Input)"),
    ]:
        ax.imshow(img_np, cmap="gray")
        ax.scatter(kp[:, 0], kp[:, 1], c="royalblue", s=3, alpha=0.3, linewidths=0)
        ax.set_title(title, fontsize=14); ax.set_axis_off()

    for i in range(n_total):
        line = matplotlib.patches.ConnectionPatch(
            xyA=(float(mkp0[i, 0]), float(mkp0[i, 1])),
            xyB=(float(mkp1[i, 0]), float(mkp1[i, 1])),
            coordsA=axes[0].transData, coordsB=axes[1].transData,
            axesA=axes[0], axesB=axes[1],
            color=colors[i], linewidth=0.8, alpha=0.6,
        )
        fig.add_artist(line)

    info = f"Keypoints: {kp0.shape[0]} / {kp1.shape[0]}  |  matches: {n_total}"
    if csv_path is not None:
        recall = n_correct / max(gt_pos_total, 1) * 100
        info += (f"  |  Recall: {n_correct}/{gt_pos_total} ({recall:.1f}%)"
                 f"  |  Wrong: {n_wrong}  |  Occluded: {n_occluded}  |  No-GT: {n_no_gt}")
    fig.suptitle(info, fontsize=10, y=0.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close(fig)
    print(f"  {output_path.name}  |  matches={n_total}  correct={n_correct}/{gt_pos_total}"
          f"  wrong={n_wrong}  occluded={n_occluded}  no-GT={n_no_gt}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--experiment", type=str, default="0413_new_iss_fpfh_lg")
    p.add_argument("--descriptor_type", type=str, default="fpfh", choices=["fpfh", "shot"])
    p.add_argument("--fpfh_radius", type=float, default=10.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--indices", type=int, nargs="*", default=None)
    p.add_argument("--gt_radius", type=int, default=20)
    p.add_argument("--resize_factor", type=float, default=NEW_C.RESIZE_FACTOR)
    args = p.parse_args()

    output_dir = Path(args.output_dir if args.output_dir
                      else f"results/{args.experiment}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.checkpoint is None:
        from gluefactory.utils.experiments import get_best_checkpoint
        cp_path = str(get_best_checkpoint(args.experiment))
        print(f"Auto-loaded best checkpoint: {cp_path}")
    else:
        cp_path = args.checkpoint

    model, _ = load_model(cp_path, device)

    if args.descriptor_type == "fpfh":
        cache_dir = Path("gluefactory/datasets/new_dataset_cache") / f"cache_new_iss_fpfh_r{args.fpfh_radius}"
    else:
        cache_dir = Path("gluefactory/datasets/new_dataset_cache") / "cache_new_iss_shot352"

    dataset = NewDatasetISSDescDataset(
        split=args.split, cache_dir=cache_dir,
        resize_factor=args.resize_factor, val_ratio=0.05, seed=0,
    )
    print(f"{args.split} dataset: {len(dataset)} pairs")

    if args.indices is not None:
        indices = list(args.indices)
    else:
        rng = np.random.default_rng(0)
        indices = sorted(rng.choice(
            len(dataset), min(args.num_samples, len(dataset)), replace=False
        ).tolist())

    print(f"Testing {len(indices)} pairs: {indices}")
    for idx in indices:
        sample = dataset[idx]
        batch = collate_fn_dynamic_pad([sample])
        pred, batch = run_inference(model, batch, device)
        csv_path = batch.get("csv_path", [None])[0]
        out = output_dir / f"{args.split}_{idx:05d}.png"
        visualize_pair(pred, batch, 0, out, csv_path=csv_path,
                       gt_radius=args.gt_radius, resize_factor=args.resize_factor)
    print(f"\nDone! → {output_dir}/")


if __name__ == "__main__":
    main()
