"""v1 (2026-04-15) ISS+SHOT+LG 매칭 시각화.

학습된 ISS(det)+SHOT(desc)+LG 모델로 v1_20260415 테스트 페어를 매칭하고 4색 시각화.
- skyblue : correct (GT 존재, not occluded)
- limegreen : occluded-correct (GT occluded이지만 맞춤)
- purple : wrong (GT 존재하지만 틀림)
- red : no GT (master kp 주변에 GT가 없음)

주의: v1 데이터셋은 zero-pad된 full 이미지 (W=2432, H=3008) 좌표계를 사용.
Crop/Resize 없음. SHOT descriptor는 PCL 기본 unit-sphere 정규화를 사용 (별도 처리 없음).

사용법:
    python experiments/v1_20260415/test/matching_shot.py \
        --weights outputs/training/iss_shot_v1_20260415_dim352/checkpoint_best.tar --indices 0 10 20 30
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

from gluefactory.datasets.mitsubishi_v1_20260415_iss_shot_dataset import (
    MitsubishiV1ISSSHOTDataset,
    v1_iss_shot_collate_fn,
    PAD_H,
    PAD_W,
)
from gluefactory.models import get_model
from gluefactory.utils.tensor import batch_to_device


DEFAULT_WEIGHTS = "outputs/training/iss_shot_v1_20260415/checkpoint_best.tar"
DEFAULT_OUTPUT_DIR = None
DEFAULT_CONFIG = "gluefactory/configs/iss_shot_v1_20260415_lg.yaml"


def load_model(checkpoint_path, device, config_path=DEFAULT_CONFIG):
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "conf" in cp:
        conf = OmegaConf.create(cp["conf"])
    else:
        conf = OmegaConf.load(config_path)
    model = get_model(conf.model.name)(conf.model).to(device)
    miss, unexp = model.load_state_dict(cp["model"], strict=False)
    if miss or unexp:
        print(f"[warn] load_state_dict: missing={len(miss)} unexpected={len(unexp)}")
    model.eval()
    epoch = cp.get("epoch", "?")
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {epoch})")
    return model, conf


@torch.no_grad()
def run_inference(model, batch, device):
    batch = batch_to_device(batch, device)
    pred = model(batch)
    return pred, batch


def visualize_pair(pred, data, idx, output_path, csv_path=None, gt_radius=20):
    """4색 매칭 시각화. v1 좌표계 = zero-pad 픽셀 (2432, 3008). 스케일/오프셋 없음."""
    img0 = data["view0"]["image"][idx].cpu()
    img1 = data["view1"]["image"][idx].cpu()

    kp0 = pred["keypoints0"][idx].cpu().numpy()
    kp1 = pred["keypoints1"][idx].cpu().numpy()
    m0 = pred["matches0"][idx].cpu().numpy()

    valid = (m0 > -1) & (m0 < kp1.shape[0])
    mkp0 = kp0[valid]
    mkp1 = kp1[m0[valid]]
    n_total = int(valid.sum())

    n_correct, n_wrong, n_occluded, n_no_gt = 0, 0, 0, 0
    gt_pos_total = 0
    colors = ["red"] * n_total

    if csv_path is not None and Path(csv_path).exists():
        corr = pd.read_csv(csv_path)

        all_master_xy = corr[["master_x", "master_y"]].values.astype(np.float32)
        all_input_xy = corr[["input_x", "input_y"]].values.astype(np.float32)
        all_occluded = corr["occluded"].values.astype(bool)
        # v1: no crop, no resize — GT coords are already in zero-pad pixel space.

        gt_pos_total = int((~all_occluded).sum())

        for i in range(n_total):
            kp0_pt = mkp0[i]
            kp1_pt = mkp1[i]

            dists_master = np.linalg.norm(all_master_xy - kp0_pt, axis=1)
            nearest_idx = int(np.argmin(dists_master))
            nearest_dist = float(dists_master[nearest_idx])

            if nearest_dist < gt_radius:
                gt_input_pt = all_input_xy[nearest_idx]
                is_occluded = bool(all_occluded[nearest_idx])
                match_dist = float(np.linalg.norm(kp1_pt - gt_input_pt))

                if match_dist < gt_radius:
                    if is_occluded:
                        colors[i] = "limegreen"
                        n_occluded += 1
                    else:
                        colors[i] = "skyblue"
                        n_correct += 1
                else:
                    colors[i] = "purple"
                    n_wrong += 1
            else:
                colors[i] = "red"
                n_no_gt += 1
    else:
        # Fallback: 2-color viz from gt_matches0 (if available)
        if "gt_matches0" in pred:
            gt_m0 = pred["gt_matches0"][idx].cpu().numpy()
            pred_tgt = m0[valid]
            gt_tgt = gt_m0[valid]
            for i in range(n_total):
                if gt_tgt[i] == pred_tgt[i] and gt_tgt[i] >= 0:
                    colors[i] = "skyblue"
                    n_correct += 1
                elif gt_tgt[i] == -1:
                    colors[i] = "red"
                    n_no_gt += 1
                else:
                    colors[i] = "purple"
                    n_wrong += 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 10), dpi=100)
    for ax, img, kp, title in [
        (axes[0], img0, kp0, "View 0 (Master)"),
        (axes[1], img1, kp1, "View 1 (Input)"),
    ]:
        img_np = img.squeeze(0).numpy()
        ax.imshow(img_np, cmap="gray")
        ax.scatter(kp[:, 0], kp[:, 1], c="royalblue", s=3, alpha=0.3, linewidths=0)
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()

    for i in range(n_total):
        line = matplotlib.patches.ConnectionPatch(
            xyA=(float(mkp0[i, 0]), float(mkp0[i, 1])),
            xyB=(float(mkp1[i, 0]), float(mkp1[i, 1])),
            coordsA=axes[0].transData,
            coordsB=axes[1].transData,
            axesA=axes[0],
            axesB=axes[1],
            color=colors[i],
            linewidth=0.8,
            alpha=0.6,
        )
        fig.add_artist(line)

    info = f"Keypoints: {kp0.shape[0]} / {kp1.shape[0]}  |  Total matches: {n_total}"
    if csv_path is not None and Path(csv_path).exists():
        info += (
            f"  |  Recall(cyan): {n_correct}"
            f"  |  Wrong(purple): {n_wrong}"
            f"  |  Occluded(green): {n_occluded}"
            f"  |  No-GT(red): {n_no_gt}"
        )
    fig.suptitle(info, fontsize=13, y=0.01)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.2, dpi=150)
    plt.close(fig)

    print(
        f"  Saved: {output_path}  |  matches={n_total}"
        f"  recall={n_correct}  wrong={n_wrong}"
        f"  occluded={n_occluded}  no-GT={n_no_gt}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="v1_20260415 ISS+SHOT+LG 매칭 시각화"
    )
    parser.add_argument("--experiment", type=str, default="iss_shot_v1_20260415",
                        help="Experiment name (auto: outputs/training/{name}/checkpoint_best.tar)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Checkpoint path (.tar), overrides --experiment")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="Fallback config path if checkpoint has no 'conf'")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--n_pairs", type=int, default=20,
                        help="Number of random pairs to visualize")
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="Explicit pair indices (overrides --n_pairs)")
    parser.add_argument("--gt_radius", type=int, default=20,
                        help="GT match radius in pixels (zero-pad coord)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] requested device={args.device} but CUDA unavailable, falling back to cpu")
        device = "cpu"
    else:
        device = args.device

    weights = args.weights or f"outputs/training/{args.experiment}/checkpoint_best.tar"
    exp_name = Path(weights).parent.name
    output_dir = Path(args.output_dir if args.output_dir
                      else f"experiments/v1_20260415/results/{exp_name}/matching")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, conf = load_model(weights, device, config_path=args.config)

    dataset = MitsubishiV1ISSSHOTDataset(split=args.split)
    print(
        f"{args.split} dataset: {len(dataset)} pairs "
        f"(pad_w={PAD_W}, pad_h={PAD_H})"
    )

    if args.indices is not None and len(args.indices) > 0:
        indices = args.indices
    else:
        rng = np.random.default_rng(args.seed)
        n = min(args.n_pairs, len(dataset))
        indices = sorted(rng.choice(len(dataset), n, replace=False).tolist())

    print(f"Testing {len(indices)} pairs: {indices}")

    for idx in indices:
        sample = dataset[idx]
        # Pull csv_path before collate drops it.
        csv_path = sample.get("csv_path", None)
        batch = v1_iss_shot_collate_fn([sample])
        pred, batch = run_inference(model, batch, device)
        output_path = output_dir / f"{args.split}_{idx:05d}.png"
        visualize_pair(
            pred, batch, 0, output_path,
            csv_path=csv_path, gt_radius=args.gt_radius,
        )

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
