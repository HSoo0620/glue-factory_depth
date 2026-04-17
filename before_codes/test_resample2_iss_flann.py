"""
ISS+FLANN Baseline 매칭 시각화 — cached descriptors + FLANN KDTree.

학습 기반 matcher 없이 FLANN으로 매칭한 결과를 시각화한다.
4색 라인: skyblue(correct) / purple(wrong) / limegreen(occluded) / red(no-GT)

사용법:
    python test_resample2_iss_flann.py --descriptor shot --indices 0 10 50 90
    python test_resample2_iss_flann.py --descriptor fpfh --ratio_th 0.8
    python test_resample2_iss_flann.py --descriptor rops --indices 0 10 50 90
"""

import argparse
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# --- 상수 ---
CROP_X0 = 1129
CROP_Y0 = 1081
CROP_SIZE = 3502
CLIP_START = 0.1
CLIP_END = 1000.0

DESCRIPTOR_CONFIG = {
    "shot":  {"key": "shot_descriptors",  "cache_dir": "iss_shot352_resample2_cache"},
    "fpfh":  {"key": "fpfh_descriptors",  "cache_dir": "iss_fpfh_resample2_cache_r5.0_xyz"},
    "rops":  {"key": "rops_descriptors",  "cache_dir": "iss_rops135_resample2_cache"},
}


# --- Cache & FLANN ---

def load_cache(cache_dir, depth_fname, desc_key):
    """캐시 npz 로드. Returns: keypoints (n_valid, 2), descriptors (n_valid, D)."""
    npz_name = Path(depth_fname).stem + ".npz"
    data = np.load(str(cache_dir / npz_name))
    n_valid = int(data["n_valid"])
    kp = data["keypoints"][:n_valid]
    desc = data[desc_key][:n_valid]
    return kp, desc


def flann_match(desc0, desc1, ratio_th=0.75):
    """OpenCV FLANN KDTree + Lowe's ratio test. Returns: idx0, idx1."""
    if len(desc0) < 2 or len(desc1) < 2:
        return np.array([], dtype=int), np.array([], dtype=int)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(desc0.astype(np.float32),
                               desc1.astype(np.float32), k=2)

    idx0, idx1 = [], []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < ratio_th * n.distance:
                idx0.append(m.queryIdx)
                idx1.append(m.trainIdx)
    return np.array(idx0, dtype=int), np.array(idx1, dtype=int)


def depth_to_display(depth_raw_crop):
    """Depth crop -> displayable float [0,1]. Dataset과 동일한 정규화 (/ 65535)."""
    return depth_raw_crop.astype(np.float32) / 65535.0


# --- Visualization ---

def visualize_pair(kp0, kp1, mkp0, mkp1, img0, img1,
                   output_path, csv_path=None, gt_radius=20,
                   image_size=1751, title_prefix="ISS+SHOT+FLANN"):
    """매칭 결과 시각화 (4색 라인)."""
    n_total = len(mkp0)

    n_correct, n_wrong, n_occluded, n_no_gt = 0, 0, 0, 0
    gt_pos_total = 0
    colors = ["red"] * n_total

    if csv_path is not None and Path(csv_path).exists():
        corr = pd.read_csv(csv_path)
        scale = image_size / CROP_SIZE

        all_master_xy = corr[["master_x", "master_y"]].values.astype(np.float32)
        all_input_xy = corr[["input_x", "input_y"]].values.astype(np.float32)
        all_occluded = corr["occluded"].values.astype(bool)

        all_master_xy[:, 0] -= CROP_X0
        all_master_xy[:, 1] -= CROP_Y0
        all_input_xy[:, 0] -= CROP_X0
        all_input_xy[:, 1] -= CROP_Y0

        all_master_xy *= scale
        all_input_xy *= scale

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
        ax.imshow(img, cmap="gray")
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

    info = f"[{title_prefix}]  Keypoints: {kp0.shape[0]} / {kp1.shape[0]}  |  Total matches: {n_total}"
    if csv_path is not None:
        recall = n_correct / max(gt_pos_total, 1) * 100
        info += (f"  |  Recall(cyan): {n_correct}/{gt_pos_total} ({recall:.1f}%)"
                 f"  |  Wrong(purple): {n_wrong}"
                 f"  |  Occluded(green): {n_occluded}"
                 f"  |  No-GT(red): {n_no_gt}")
    fig.suptitle(info, fontsize=10, y=0.02)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close(fig)
    recall = n_correct / max(gt_pos_total, 1) * 100 if gt_pos_total > 0 else 0.0
    print(f"  Saved: {output_path}  |  matches={n_total}"
          f"  recall={n_correct}/{gt_pos_total} ({recall:.1f}%)"
          f"  wrong={n_wrong}  occluded={n_occluded}  no-GT={n_no_gt}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="ISS+FLANN Baseline 매칭 시각화")
    parser.add_argument("--descriptor", type=str, default="shot",
                        choices=["shot", "fpfh", "rops"])
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Override cache dir (default: auto from descriptor)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--image_size", type=int, default=1751)
    parser.add_argument("--ratio_th", type=float, default=0.75)
    parser.add_argument("--gt_radius", type=int, default=20)
    args = parser.parse_args()

    desc_conf = DESCRIPTOR_CONFIG[args.descriptor]
    cache_dir_name = args.cache_dir if args.cache_dir else desc_conf["cache_dir"]
    desc_key = desc_conf["key"]

    output_dir = Path(
        args.output_dir if args.output_dir
        else f"results/iss_{args.descriptor}_flann")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path("gluefactory/datasets/mitsubishi")
    cache_dir = base_dir / cache_dir_name

    combo = pd.read_csv(base_dir / "outputs_txt" / "combination.csv")
    total = len(combo)
    val_size, test_size = 100, 100
    train_end = total - val_size - test_size
    val_end = total - test_size

    if args.split == "train":
        combo_split = combo.iloc[:train_end]
    elif args.split == "val":
        combo_split = combo.iloc[train_end:val_end]
    else:
        combo_split = combo.iloc[val_end:]
    combo_split = combo_split.reset_index(drop=True)

    title_prefix = f"ISS+{args.descriptor.upper()}+FLANN"
    print(f"Method: {title_prefix} (cache: {cache_dir_name})")
    print(f"Split: {args.split} ({len(combo_split)} pairs)")
    print(f"ratio_th={args.ratio_th}, gt_radius={args.gt_radius}")

    if args.indices is not None:
        indices = args.indices
    else:
        indices = sorted(np.random.choice(
            len(combo_split), min(args.num_samples, len(combo_split)),
            replace=False))

    print(f"Testing {len(indices)} pairs: {indices}")

    for idx in indices:
        print(f"\n--- Pair {idx} ---")
        row = combo_split.iloc[idx]
        master_fname = Path(row["master_path"]).name
        input_fname = Path(row["input_path"]).name

        # 캐시 로드 + FLANN 매칭
        kp0, desc0 = load_cache(cache_dir, master_fname, desc_key)
        kp1, desc1 = load_cache(cache_dir, input_fname, desc_key)

        idx0, idx1 = flann_match(desc0, desc1, ratio_th=args.ratio_th)
        mkp0 = kp0[idx0] if len(idx0) > 0 else np.zeros((0, 2))
        mkp1 = kp1[idx1] if len(idx1) > 0 else np.zeros((0, 2))

        # Depth 이미지 로드 (crop + display 변환)
        depth0_raw = cv2.imread(
            str(base_dir / "dataset_resample_2" / master_fname),
            cv2.IMREAD_UNCHANGED)
        depth1_raw = cv2.imread(
            str(base_dir / "dataset_resample_2" / input_fname),
            cv2.IMREAD_UNCHANGED)

        crop0 = depth0_raw[CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]
        crop1 = depth1_raw[CROP_Y0:CROP_Y0+CROP_SIZE, CROP_X0:CROP_X0+CROP_SIZE]

        # Resize to image_size for display (match keypoint coordinates)
        img0 = cv2.resize(depth_to_display(crop0),
                          (args.image_size, args.image_size),
                          interpolation=cv2.INTER_NEAREST)
        img1 = cv2.resize(depth_to_display(crop1),
                          (args.image_size, args.image_size),
                          interpolation=cv2.INTER_NEAREST)

        csv_path = str(base_dir / row["csv_path"])
        output_path = output_dir / f"{args.split}_{idx:05d}.png"

        visualize_pair(kp0, kp1, mkp0, mkp1, img0, img1,
                       output_path, csv_path=csv_path,
                       gt_radius=args.gt_radius,
                       image_size=args.image_size,
                       title_prefix=title_prefix)

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
