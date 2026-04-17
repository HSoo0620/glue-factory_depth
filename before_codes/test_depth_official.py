"""
공식 pretrained SuperPoint + LightGlue로 depth-depth 매칭 테스트.
학습 없이 공식 가중치를 그대로 사용하여 baseline 성능을 확인.

사용법:
    # 기본 (test split, 10개 샘플)
    python test_registration_resample.py --checkpoint outputs/training/resample_sp_lg/checkpoint_best.tar --indices 0 10 50 90

    # 샘플 수/인덱스 지정
    python test_depth_official.py --num_samples 20
    python test_depth_official.py --indices 0 10 50 90

    # 출력 경로 지정
    python test_depth_official.py --output_dir results/official_sp_lg
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.datasets.mitsubishi_depth_dataset import (
    MitsubishiDepthDataset,
    mitsubishi_collate_fn,
)
from gluefactory.utils.tensor import batch_to_device


def build_official_model(device, max_num_keypoints=512):
    """공식 pretrained SuperPoint + LightGlue 모델 구성"""
    conf = {
        "name": "two_view_pipeline",
        "extractor": {
            "name": "gluefactory_nonfree.superpoint",
            "max_num_keypoints": max_num_keypoints,
            "force_num_keypoints": True,
            "detection_threshold": 0.0,
            "nms_radius": 3,
            "trainable": False,
        },
        "ground_truth": {
            "name": "matchers.gt_pair_matcher",
            "gt_radius": 3,
        },
        "matcher": {
            "name": "matchers.lightglue",
            "weights": "superpoint",  # 공식 pretrained weights 자동 다운로드
            "input_dim": 256,
            "descriptor_dim": 256,
            "filter_threshold": 0.1,
            "flash": False,
        },
    }
    conf = OmegaConf.create(conf)
    model = get_model("two_view_pipeline")(conf).to(device)
    model.eval()
    print("Loaded official pretrained SuperPoint + LightGlue")
    return model


@torch.no_grad()
def run_inference(model, batch, device):
    batch = batch_to_device(batch, device)
    pred = model(batch)
    return pred, batch


def visualize_pair(pred, data, idx, output_path, csv_path=None, gt_radius=3):
    """
    4색 시각화 (test_depth.py와 동일):
        하늘색: 정답 pair
        보라색: GT 정의됐지만 틀림
        녹색: 정답인데 가려짐
        빨간색: GT 미정의 매칭
    """
    img0 = data["view0"]["image"][idx].cpu()
    img1 = data["view1"]["image"][idx].cpu()

    kp0 = pred["keypoints0"][idx].cpu().numpy()
    kp1 = pred["keypoints1"][idx].cpu().numpy()
    m0 = pred["matches0"][idx].cpu().numpy()

    valid = m0 > -1
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
        gt_pos_total = int((~all_occluded).sum())

        for i in range(n_total):
            kp0_pt = mkp0[i]
            kp1_pt = mkp1[i]

            dists_master = np.linalg.norm(all_master_xy - kp0_pt, axis=1)
            nearest_idx = np.argmin(dists_master)
            nearest_dist = dists_master[nearest_idx]

            if nearest_dist < gt_radius:
                gt_input_pt = all_input_xy[nearest_idx]
                is_occluded = all_occluded[nearest_idx]
                match_dist = np.linalg.norm(kp1_pt - gt_input_pt)

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

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

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

    info = f"[Official SP+LG]  Keypoints: {kp0.shape[0]} / {kp1.shape[0]}  |  Total matches: {n_total}"
    if csv_path is not None:
        recall = n_correct / max(gt_pos_total, 1) * 100
        info += (f"  |  Recall(cyan): {n_correct}/{gt_pos_total} ({recall:.1f}%)"
                 f"  |  Wrong(purple): {n_wrong}"
                 f"  |  Occluded(green): {n_occluded}"
                 f"  |  No-GT(red): {n_no_gt}")
    fig.suptitle(info, fontsize=11, y=0.02)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close(fig)
    recall = n_correct / max(gt_pos_total, 1) * 100 if gt_pos_total > 0 else 0.0
    print(f"  Saved: {output_path}  |  matches={n_total}"
          f"  recall(cyan)={n_correct}/{gt_pos_total} ({recall:.1f}%)"
          f"  wrong(purple)={n_wrong}"
          f"  occluded(green)={n_occluded}"
          f"  no-GT(red)={n_no_gt}")


def main():
    parser = argparse.ArgumentParser(description="Official SP+LG로 depth 매칭 테스트")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/official_sp_lg")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--max_num_keypoints", type=int, default=512)
    parser.add_argument("--gt_radius", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    model = build_official_model(device, args.max_num_keypoints)

    dataset = MitsubishiDepthDataset(split=args.split)
    print(f"{args.split} dataset: {len(dataset)} pairs")

    if args.indices is not None:
        indices = args.indices
    else:
        indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
        indices = sorted(indices)

    print(f"Testing {len(indices)} pairs: {indices}")

    for idx in indices:
        sample = dataset[idx]
        batch = mitsubishi_collate_fn([sample])
        pred, batch = run_inference(model, batch, device)
        csv_path = batch.get("csv_path", [None])[0]
        output_path = output_dir / f"{args.split}_{idx:05d}.png"
        visualize_pair(pred, batch, 0, output_path, csv_path=csv_path, gt_radius=args.gt_radius)

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
