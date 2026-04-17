"""
학습된 LightGlue 모델로 depth pair 매칭 테스트 및 결과 시각화.

사용법:
    # checkpoint_best.tar로 테스트 (기본)
    python test_depth.py

    # 특정 체크포인트 지정
    python test_depth.py --checkpoint outputs/training/Depth_only_sp_pt/checkpoint_best.tar \
        --indices 0 10 50 90 \
        --output_dir results/matchings/Depth_only_sp_pt \
        --save_point_pairs

    # 테스트 샘플 수, 저장 경로 지정
    python test_depth.py --num_samples 20 --output_dir results/

    # 특정 pair 인덱스 지정
    python test_depth.py --indices 0 5 10 50 70 90
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


def load_model(checkpoint_path, device):
    cp = torch.load(checkpoint_path, map_location="cpu")
    conf = OmegaConf.create(cp["conf"])
    model = get_model(conf.model.name)(conf.model).to(device)
    model.load_state_dict(cp["model"], strict=False)
    model.eval()
    epoch = cp["epoch"]
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  epoch: {epoch}")
    if cp.get("eval"):
        for k, v in cp["eval"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
    return model


@torch.no_grad()
def run_inference(model, batch, device):
    batch = batch_to_device(batch, device)
    pred = model(batch)
    # GT 생성 (loss 호출 시 gt_pair_matcher가 실행됨)
    if hasattr(model, "ground_truth") and model.conf.ground_truth.name:
        gt_pred = model.ground_truth({**batch, **pred})
        pred.update({f"gt_{k}": v for k, v in gt_pred.items()})
    return pred, batch


def visualize_pair(pred, data, idx, output_path, csv_path=None, gt_radius=3):
    """
    한 쌍의 매칭 결과를 시각화.
    색상 분류:
        하늘색: 정답 pair (GT에 정의 + 비가림 + 정확히 매칭)
        보라색: GT에 정의된 포인트인데 매칭이 잘못됨
        녹색:   정답 pair를 잘 맞췄는데 GT에서 가려짐(occluded)으로 표시
        빨간색: GT에 포인트 정의가 없는데 매칭됨
    """
    img0 = data["view0"]["image"][idx].cpu()  # (1, H, W)
    img1 = data["view1"]["image"][idx].cpu()

    kp0 = pred["keypoints0"][idx].cpu().numpy()
    kp1 = pred["keypoints1"][idx].cpu().numpy()
    m0 = pred["matches0"][idx].cpu().numpy()

    valid = m0 > -1
    mkp0 = kp0[valid]
    mkp1 = kp1[m0[valid]]
    n_total = int(valid.sum())

    # 색상 분류
    n_correct, n_wrong, n_occluded, n_no_gt = 0, 0, 0, 0
    gt_pos_total = 0
    colors = ["red"] * n_total  # 기본: 빨간색 (GT 미정의)

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
                        colors[i] = "limegreen"   # 녹색: 맞췄지만 가려짐
                        n_occluded += 1
                    else:
                        colors[i] = "skyblue"      # 하늘색: 정답 pair
                        n_correct += 1
                else:
                    colors[i] = "purple"           # 보라색: 정의됐지만 틀림
                    n_wrong += 1
            else:
                colors[i] = "red"                  # 빨간색: GT 미정의
                n_no_gt += 1

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

    for ax, img, kp, title in [
        (axes[0], img0, kp0, "View 0 (Master)"),
        (axes[1], img1, kp1, "View 1 (Input)"),
    ]:
        img_np = img.squeeze(0).numpy()
        ax.imshow(img_np, cmap="gray")
        ax.scatter(
            kp[:, 0], kp[:, 1],
            c="royalblue", s=3, alpha=0.3, linewidths=0,
        )
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()

    # 매칭 라인 그리기
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

    # 통계 텍스트
    info = f"Keypoints: {kp0.shape[0]} / {kp1.shape[0]}  |  Total matches: {n_total}"
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


def save_point_pairs_csv(pred, data, batch_idx, dataset_idx, output_dir, split):
    """원본 CSV 정보 + 모델 추론 point pair를 CSV로 저장"""
    # 원본 CSV 읽기
    csv_path = data["csv_path"][batch_idx]
    master_path = data["master_path"][batch_idx]
    input_path = data["input_path"][batch_idx]
    original_csv = pd.read_csv(csv_path)

    # 모델 추론 매칭 결과 추출
    kp0 = pred["keypoints0"][batch_idx].cpu()
    kp1 = pred["keypoints1"][batch_idx].cpu()
    m0 = pred["matches0"][batch_idx].cpu()
    scores0 = pred["matching_scores0"][batch_idx].cpu()

    valid = m0 > -1
    mkp0 = kp0[valid].numpy()
    mkp1 = kp1[m0[valid]].numpy()
    mscores = scores0[valid].numpy()

    # GT 정보가 있으면 correct 여부도 저장
    if "gt_matches0" in pred:
        gtm0 = pred["gt_matches0"][batch_idx].cpu()
        correct = (gtm0[valid] == m0[valid]).numpy()
    else:
        correct = np.full(len(mkp0), False)

    pred_df = pd.DataFrame({
        "pred_master_x": mkp0[:, 0],
        "pred_master_y": mkp0[:, 1],
        "pred_input_x": mkp1[:, 0],
        "pred_input_y": mkp1[:, 1],
        "matching_score": mscores,
        "gt_correct": correct,
    })

    # 저장
    out_path = output_dir / f"{split}_{dataset_idx:05d}_point_pairs.csv"
    with open(out_path, "w") as f:
        f.write(f"# master_path: {master_path}\n")
        f.write(f"# input_path: {input_path}\n")
        f.write(f"# original_csv: {csv_path}\n")
        f.write(f"#\n")
        f.write(f"# === Original GT Points (from CSV) ===\n")
        original_csv.to_csv(f, index=False)
        f.write(f"\n# === Model Predicted Point Pairs ===\n")
        pred_df.to_csv(f, index=False)

    print(f"  Point pairs saved: {out_path}  |  pred_matches={len(pred_df)}  gt_points={len(original_csv)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="체크포인트 경로. 미지정 시 sp_depth의 best checkpoint 자동 로드",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="sp_depth",
        help="실험 이름 (checkpoint 미지정 시 사용)",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--output_dir", type=str, default="results/test_depth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_point_pairs", action="store_true",
                        help="매칭된 point pair를 CSV로 저장 (원본 CSV 정보 + 모델 추론 결과)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"

    if args.checkpoint is None:
        from gluefactory.utils.experiments import get_best_checkpoint
        cp_path = get_best_checkpoint(args.experiment)
        print(f"Auto-loaded best checkpoint: {cp_path}")
    else:
        cp_path = args.checkpoint

    model = load_model(cp_path, device)

    dataset = MitsubishiDepthDataset(split=args.split)
    print(f"{args.split} dataset: {len(dataset)} pairs")

    if args.indices is not None:
        indices = args.indices
    else:
        indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
        indices = sorted(indices)

    print(f"Testing {len(indices)} pairs: {indices}")

    if args.save_point_pairs:
        pairs_dir = output_dir / "point_pairs"
        pairs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Point pairs will be saved to {pairs_dir}/")

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        batch = mitsubishi_collate_fn([sample])
        pred, batch = run_inference(model, batch, device)
        csv_path = batch.get("csv_path", [None])[0]
        output_path = output_dir / f"{args.split}_{idx:05d}.png"
        visualize_pair(pred, batch, 0, output_path, csv_path=csv_path, gt_radius=3)

        if args.save_point_pairs:
            save_point_pairs_csv(pred, batch, 0, idx, pairs_dir, args.split)

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
