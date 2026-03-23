"""
GT 없이 학습된 LightGlue 모델로 depth pair 매칭 테스트 및 시각화.

사용법:
    # 데이터셋 기반 (GT 무시, 매칭 결과만 시각화)
    python test_depth_no_gt.py \
        --checkpoint outputs/training/Depth_only_sp/checkpoint_best.tar

    # 커스텀 이미지 2장 직접 지정
    python test_depth_no_gt.py \
        --checkpoint outputs/training/Depth_only_sp/checkpoint_best.tar \
        --img0 /path/to/master.png --img1 /path/to/input.png

    # 커스텀 이미지 디렉토리 (master_*.png + input_*.png 페어)
    python test_depth_no_gt.py \
        --checkpoint outputs/training/Depth_only_sp/checkpoint_best.tar \
        --img_dir /path/to/pairs/

    # 데이터셋 기반 옵션
    python test_depth_no_gt.py \
        --checkpoint outputs/training/Depth_only_sp/checkpoint_best.tar \
        --num_samples 20 --indices 0 5 10
"""

import argparse
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.utils.tensor import batch_to_device


def load_model(checkpoint_path, device):
    cp = torch.load(checkpoint_path, map_location="cpu")
    conf = OmegaConf.create(cp["conf"])
    model = get_model(conf.model.name)(conf.model).to(device)
    model.load_state_dict(cp["model"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {cp['epoch']})")
    return model


def load_image(path):
    """depth 이미지 로드 → (1, H, W) float32 [0, 1]"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = img.astype(np.float32)
    # uint16 depth → [0, 1], uint8 → [0, 1]
    if img.max() > 255:
        img = img / 65535.0
    else:
        img = img / 255.0
    # grayscale로 변환 (컬러인 경우)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(img).unsqueeze(0)  # (1, H, W)


def make_batch(img0_tensor, img1_tensor):
    """이미지 텐서 2장 → 모델 입력 batch (GT 없음)"""
    h0, w0 = img0_tensor.shape[1], img0_tensor.shape[2]
    h1, w1 = img1_tensor.shape[1], img1_tensor.shape[2]
    return {
        "view0": {
            "image": img0_tensor.unsqueeze(0),       # (1, 1, H, W)
            "image_size": torch.tensor([[h0, w0]]),   # (1, 2)
        },
        "view1": {
            "image": img1_tensor.unsqueeze(0),
            "image_size": torch.tensor([[h1, w1]]),
        },
    }


@torch.no_grad()
def run_inference(model, batch, device):
    batch = batch_to_device(batch, device)
    pred = model(batch)
    return pred, batch


def visualize_pair(pred, data, idx, output_path, label=""):
    img0 = data["view0"]["image"][idx].cpu()
    img1 = data["view1"]["image"][idx].cpu()

    kp0 = pred["keypoints0"][idx].cpu()
    kp1 = pred["keypoints1"][idx].cpu()
    m0 = pred["matches0"][idx].cpu()
    scores0 = pred["matching_scores0"][idx].cpu()

    valid = m0 > -1
    mkp0 = kp0[valid]
    mkp1 = kp1[m0[valid]]
    mscores = scores0[valid]
    n_total = valid.sum().item()

    # confidence 기반 색상: 높을수록 진한 파랑
    score_np = mscores.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

    for ax, img, kp, title in [
        (axes[0], img0, kp0, "View 0 (Master)"),
        (axes[1], img1, kp1, "View 1 (Input)"),
    ]:
        img_np = img.squeeze(0).numpy()
        ax.imshow(img_np, cmap="gray")
        ax.scatter(
            kp[:, 0].numpy(), kp[:, 1].numpy(),
            c="royalblue", s=3, alpha=0.3, linewidths=0,
        )
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()

    # 매칭 라인: confidence에 따라 alpha 조절
    for i in range(len(mkp0)):
        alpha = 0.3 + 0.7 * score_np[i]  # 낮은 confidence는 연하게
        line = matplotlib.patches.ConnectionPatch(
            xyA=(mkp0[i, 0].item(), mkp0[i, 1].item()),
            xyB=(mkp1[i, 0].item(), mkp1[i, 1].item()),
            coordsA=axes[0].transData,
            coordsB=axes[1].transData,
            axesA=axes[0],
            axesB=axes[1],
            color="dodgerblue",
            linewidth=0.8,
            alpha=alpha,
        )
        fig.add_artist(line)

    # 통계
    avg_score = score_np.mean() if n_total > 0 else 0
    info = f"Keypoints: {kp0.shape[0]} / {kp1.shape[0]}  |  Matches: {n_total}  |  Avg confidence: {avg_score:.3f}"
    if label:
        info = f"[{label}]  " + info
    fig.suptitle(info, fontsize=12, y=0.02)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}  |  matches={n_total}  avg_conf={avg_score:.3f}")


def main():
    parser = argparse.ArgumentParser(description="GT 없이 depth pair 매칭 테스트")
    parser.add_argument("--checkpoint", type=str, required=True, help="체크포인트 경로")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/test_no_gt")

    # 모드 1: 커스텀 이미지 2장
    parser.add_argument("--img0", type=str, default=None, help="Master 이미지 경로")
    parser.add_argument("--img1", type=str, default=None, help="Input 이미지 경로")

    # 모드 2: 커스텀 이미지 디렉토리 (master_*.png / input_*.png)
    parser.add_argument("--img_dir", type=str, default=None,
                        help="이미지 페어 디렉토리. master_*.png + input_*.png 패턴")

    # 모드 3: 기존 데이터셋 (기본)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--indices", type=int, nargs="*", default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    model = load_model(args.checkpoint, device)

    # --- 모드 1: 커스텀 이미지 2장 ---
    if args.img0 and args.img1:
        print(f"Custom pair: {args.img0} <-> {args.img1}")
        img0 = load_image(args.img0)
        img1 = load_image(args.img1)
        batch = make_batch(img0, img1)
        pred, batch = run_inference(model, batch, device)
        out_name = f"custom_{Path(args.img0).stem}__{Path(args.img1).stem}.png"
        visualize_pair(pred, batch, 0, output_dir / out_name, label="custom")
        print(f"\nDone! Result saved to {output_dir / out_name}")
        return

    # --- 모드 2: 커스텀 디렉토리 ---
    if args.img_dir:
        img_dir = Path(args.img_dir)
        masters = sorted(img_dir.glob("master_*"))
        if not masters:
            print(f"No master_* files found in {img_dir}")
            return
        print(f"Found {len(masters)} master images in {img_dir}")
        for mp in masters:
            # master_001.png -> input_001.png
            ip = img_dir / mp.name.replace("master_", "input_", 1)
            if not ip.exists():
                print(f"  Skip {mp.name}: no matching {ip.name}")
                continue
            img0 = load_image(mp)
            img1 = load_image(ip)
            batch = make_batch(img0, img1)
            pred, batch = run_inference(model, batch, device)
            out_name = f"{mp.stem}__{ip.stem}.png"
            visualize_pair(pred, batch, 0, output_dir / out_name, label=mp.stem)
        print(f"\nDone! Results saved to {output_dir}/")
        return

    # --- 모드 3: 기존 데이터셋 (GT 무시) ---
    from gluefactory.datasets.mitsubishi_depth_dataset import (
        MitsubishiDepthDataset,
        mitsubishi_collate_fn,
    )
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
        # GT 제거하고 batch 구성
        batch = make_batch(sample["view0"]["image"], sample["view1"]["image"])
        pred, batch = run_inference(model, batch, device)
        output_path = output_dir / f"{args.split}_{idx:05d}.png"
        visualize_pair(pred, batch, 0, output_path, label=f"{args.split}#{idx}")

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
