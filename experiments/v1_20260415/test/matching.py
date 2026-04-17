"""v1 (2026-04-15) ISS+FPFH/SHOT+LG 매칭 시각화.

4-color:
  skyblue  = correct (GT 존재, not occluded, input 맞음)
  purple   = wrong   (GT 존재, input 틀림)
  limegreen= occluded (GT occluded이지만 맞춤)
  red      = no-GT   (master kp 주변에 GT 없음)

사용법:
    python experiments/v1_20260415/test/matching.py --experiment iss_fpfh_v1_20260415_norm --descriptor_type fpfh --indices 0 10 20 30 
    python experiments/v1_20260415/test/matching.py --experiment iss_shot_v1_20260415_dim352 --descriptor_type shot --indices 0 10 20 30 

    """
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.utils.tensor import batch_to_device


def _get_dataset_and_collate(descriptor_type: str, split: str):
    if descriptor_type == "fpfh":
        from gluefactory.datasets.mitsubishi_v1_20260415_iss_fpfh_dataset import (
            MitsubishiV1ISSFPFHDataset as Cls, v1_iss_fpfh_collate_fn as fn, PAD_H, PAD_W)
    else:
        from gluefactory.datasets.mitsubishi_v1_20260415_iss_shot_dataset import (
            MitsubishiV1ISSSHOTDataset as Cls, v1_iss_shot_collate_fn as fn, PAD_H, PAD_W)
    return Cls(split=split), fn, PAD_H, PAD_W


def load_model(checkpoint_path: str, device: str, config_path: str | None = None):
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "conf" in cp:
        conf = OmegaConf.create(cp["conf"])
    elif config_path:
        conf = OmegaConf.load(config_path)
    else:
        raise RuntimeError("checkpoint has no 'conf' and --config not given")
    model = get_model(conf.model.name)(conf.model).to(device)
    miss, unexp = model.load_state_dict(cp["model"], strict=False)
    if miss or unexp:
        print(f"[warn] load_state_dict: missing={len(miss)} unexpected={len(unexp)}")
    model.eval()
    print(f"Loaded: {checkpoint_path} (epoch {cp.get('epoch', '?')})")
    return model


@torch.no_grad()
def run_inference(model, batch, device):
    batch = batch_to_device(batch, device)
    return model(batch), batch


def visualize_pair(pred, data, idx: int, output_path: Path,
                   csv_path: str | None = None, gt_radius: int = 20):
    img0 = data["view0"]["image"][idx].cpu()
    img1 = data["view1"]["image"][idx].cpu()
    h0, w0 = data["view0"]["image_size"][idx].tolist()
    h1, w1 = data["view1"]["image_size"][idx].tolist()
    img0_np = img0.squeeze(0).numpy()[:h0, :w0]
    img1_np = img1.squeeze(0).numpy()[:h1, :w1]

    kp0 = pred["keypoints0"][idx].cpu().numpy()
    kp1 = pred["keypoints1"][idx].cpu().numpy()
    m0 = pred["matches0"][idx].cpu().numpy()

    valid = (m0 > -1) & (m0 < kp1.shape[0])
    mkp0 = kp0[valid]
    mkp1 = kp1[m0[valid]]
    n_total = int(valid.sum())

    n_correct = n_wrong = n_occluded = n_no_gt = 0
    gt_pos_total = 0
    colors = ["red"] * n_total

    if csv_path is not None and Path(csv_path).exists():
        corr = pd.read_csv(csv_path)
        all_m = corr[["master_x", "master_y"]].values.astype(np.float32)
        all_i = corr[["input_x", "input_y"]].values.astype(np.float32)
        all_occ = corr["occluded"].values.astype(bool)
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
        info += (f"  |  Recall: {n_correct}  |  Wrong: {n_wrong}"
                 f"  |  Occluded: {n_occluded}  |  No-GT: {n_no_gt}")
    fig.suptitle(info, fontsize=13, y=0.01)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.2, dpi=150)
    plt.close(fig)
    print(f"  {output_path.name}  |  matches={n_total}  recall={n_correct}"
          f"  wrong={n_wrong}  occluded={n_occluded}  no-GT={n_no_gt}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--experiment", type=str, default="iss_fpfh_v1_20260415")
    p.add_argument("--descriptor_type", type=str, default="fpfh", choices=["fpfh", "shot"])
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--indices", type=int, nargs="*", default=None)
    p.add_argument("--gt_radius", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cp_path = args.checkpoint or f"outputs/training/{args.experiment}/checkpoint_best.tar"
    exp_name = Path(cp_path).parent.name
    output_dir = Path(args.output_dir if args.output_dir
                      else f"experiments/v1_20260415/results/{exp_name}/matching")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(cp_path, device, config_path=args.config)

    dataset, collate_fn, PAD_H, PAD_W = _get_dataset_and_collate(
        args.descriptor_type, args.split)
    print(f"{args.split} dataset: {len(dataset)} pairs")

    if args.indices is not None:
        indices = list(args.indices)
    else:
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(
            len(dataset), min(args.num_samples, len(dataset)), replace=False
        ).tolist())

    print(f"Testing {len(indices)} pairs: {indices}")
    for idx in indices:
        sample = dataset[idx]
        csv_path = sample.get("csv_path", None)
        batch = collate_fn([sample])
        pred, batch = run_inference(model, batch, device)
        out = output_dir / f"{args.split}_{idx:05d}.png"
        visualize_pair(pred, batch, 0, out, csv_path=csv_path,
                       gt_radius=args.gt_radius)
    print(f"\nDone! → {output_dir}/")


if __name__ == "__main__":
    main()
