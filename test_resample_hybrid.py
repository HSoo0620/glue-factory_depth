"""
학습된 SP+Hybrid(SP+FPFH)+LG 모델로 resample 데이터 매칭 테스트.

사용법:
    python test_resample_hybrid.py --experiment 0324_resample_sp_hybrid_lg --fpfh_radius 0.5 --indices 0 10 50 90 --gt_radius 11
"""

import argparse
import torch
import numpy as np
import pandas as pd
import cv2
import open3d as o3d
import configparser
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
    return model, conf


@torch.no_grad()
def run_inference(model, batch, device):
    batch = batch_to_device(batch, device)
    pred = model(batch)
    return pred, batch


def parse_calib_ini(ini_path):
    config = configparser.ConfigParser()
    config.read(str(ini_path))
    section = config.sections()[0]
    k_vals = [float(v) for v in config[section]["k_matrix"].split(":")[3].split(",")]
    K = np.array(k_vals, dtype=np.float64).reshape(3, 3)
    clip_start = float(config[section]["clip_start"])
    clip_end = float(config[section]["clip_end"])
    return K, clip_start, clip_end


def load_superpoint(device, detection_threshold=0.001, max_num_keypoints=512):
    from gluefactory.models.extractors.superpoint_open import SuperPoint
    sp_conf = {
        "descriptor_dim": 256,
        "nms_radius": 3,
        "max_num_keypoints": max_num_keypoints * 2,
        "force_num_keypoints": False,
        "detection_threshold": detection_threshold,
        "remove_borders": 4,
        "channels": [64, 64, 128, 128, 256],
        "dense_outputs": True,
        "weights": None,
    }
    model = SuperPoint(sp_conf).to(device)
    model.eval()
    return model


def compute_dense_fpfh(depth_real, K, fpfh_radius=1.5, fpfh_max_nn=100):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    vs, us = np.where(depth_real > 0)
    zs = depth_real[vs, us]
    xs = (us - cx) * zs / fx
    ys = (vs - cy) * zs / fy
    dense_pts = np.stack([xs, ys, zs], axis=1).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dense_pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius * 2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=fpfh_max_nn),
    )
    return pcd, np.array(fpfh.data).T.astype(np.float32)


def lookup_fpfh(keypoints, n_valid, depth_real, K, pcd, fpfh_dense, max_kp=512):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    fpfh_out = np.zeros((max_kp, 33), dtype=np.float32)

    if n_valid < 3:
        return fpfh_out

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for i in range(n_valid):
        u, v = int(keypoints[i, 0]), int(keypoints[i, 1])
        u = np.clip(u, 0, depth_real.shape[1] - 1)
        v = np.clip(v, 0, depth_real.shape[0] - 1)
        z = float(depth_real[v, u])
        if z <= 0:
            continue
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        _, idx, _ = kdtree.search_knn_vector_3d([x, y, z], 1)
        fpfh_out[i] = fpfh_dense[idx[0]]

    norms = np.linalg.norm(fpfh_out[:n_valid], axis=1, keepdims=True)
    fpfh_out[:n_valid] = fpfh_out[:n_valid] / (norms + 1e-8)
    return fpfh_out


def sample_sp_descriptors(sp_pred, keypoints, n_valid, max_kp=512):
    """SP dense descriptor에서 keypoint 위치의 descriptor를 샘플링."""
    from gluefactory.models.extractors.superpoint_open import sample_descriptors
    dense_desc = sp_pred["dense_descriptors"]  # (1, 256, H/8, W/8)
    kp_for_sample = keypoints[:n_valid] - 0.5
    kp_tensor = torch.from_numpy(kp_for_sample).unsqueeze(0).float().to(dense_desc.device)
    sp_desc = sample_descriptors(kp_tensor, dense_desc, s=8)  # (1, 256, n_valid)
    sp_desc = sp_desc[0].T.cpu().numpy()  # (n_valid, 256)

    sp_out = np.zeros((max_kp, 256), dtype=np.float32)
    sp_out[:n_valid] = sp_desc
    return sp_out


def visualize_pair(pred, data, idx, output_path, csv_path=None, gt_radius=3,
                   orig_size=5761, image_size=2880):
    img0 = data["view0"]["image"][idx].cpu()
    img1 = data["view1"]["image"][idx].cpu()

    kp0 = pred["keypoints0"][idx].cpu().numpy()
    kp1 = pred["keypoints1"][idx].cpu().numpy()
    m0 = pred["matches0"][idx].cpu().numpy()
    scores = pred["matching_scores0"][idx].cpu().numpy()

    valid = m0 > -1
    mkp0 = kp0[valid]
    mkp1 = kp1[m0[valid]]
    n_total = int(valid.sum())

    n_correct, n_wrong, n_occluded, n_no_gt = 0, 0, 0, 0
    gt_pos_total = 0
    colors = ["red"] * n_total

    if csv_path is not None and Path(csv_path).exists():
        corr = pd.read_csv(csv_path)
        scale = image_size / orig_size
        all_master_xy = corr[["master_x", "master_y"]].values.astype(np.float32) * scale
        all_input_xy = corr[["input_x", "input_y"]].values.astype(np.float32) * scale
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

    info = f"[SP+Hybrid+LG]  Keypoints: {kp0.shape[0]} / {kp1.shape[0]}  |  Total matches: {n_total}"
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
          f"  recall={n_correct}/{gt_pos_total} ({recall:.1f}%)")
    return n_correct, gt_pos_total


def main():
    parser = argparse.ArgumentParser(description="SP+Hybrid(SP+FPFH)+LG resample 매칭 테스트")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="0324_resample_sp_hybrid_lg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--image_size", type=int, default=2880)
    parser.add_argument("--fpfh_radius", type=float, default=0.5)
    parser.add_argument("--detection_threshold", type=float, default=0.001)
    parser.add_argument("--max_num_keypoints", type=int, default=512)
    parser.add_argument("--gt_radius", type=int, default=11)
    args = parser.parse_args()

    output_dir = Path(args.output_dir if args.output_dir else f"results/{args.experiment}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.checkpoint is None:
        from gluefactory.utils.experiments import get_best_checkpoint
        cp_path = get_best_checkpoint(args.experiment)
    else:
        cp_path = args.checkpoint

    model, conf = load_model(cp_path, device)
    gt_radius = conf.model.ground_truth.get("gt_radius", args.gt_radius)

    sp_model = load_superpoint(device, args.detection_threshold, args.max_num_keypoints)

    base_dir = Path("gluefactory/datasets/mitsubishi")
    combo = pd.read_csv(base_dir / "outputs" / "combination.csv")
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

    print(f"{args.split}: {len(combo_split)} pairs (image_size={args.image_size})")

    if args.indices is not None:
        indices = args.indices
    else:
        indices = np.random.choice(len(combo_split), min(args.num_samples, len(combo_split)), replace=False)
        indices = sorted(indices)

    total_correct, total_gt = 0, 0

    for idx in indices:
        row = combo_split.iloc[idx]
        master_path = base_dir / row["master_path"]
        input_path = base_dir / row["input_path"]
        csv_path = base_dir / "outputs" / Path(row["csv_path"]).name

        # 이미지 로드 + 리사이즈
        master_raw = cv2.imread(str(master_path), cv2.IMREAD_UNCHANGED)
        input_raw = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        orig_h = master_raw.shape[0]

        master_resized = cv2.resize(master_raw, (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST)
        input_resized = cv2.resize(input_raw, (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST)
        scale = orig_h / args.image_size

        master_norm = master_resized.astype(np.float32) / 65535.0
        input_norm = input_resized.astype(np.float32) / 65535.0
        h, w = master_norm.shape

        # calib
        img_idx = master_path.stem.replace("depth_raw_", "")
        K0, clip_start, clip_end = parse_calib_ini(master_path.parent / f"calib_{img_idx}.ini")
        img_idx1 = input_path.stem.replace("depth_raw_", "")
        K1, clip_start1, clip_end1 = parse_calib_ini(input_path.parent / f"calib_{img_idx1}.ini")

        # SuperPoint (keypoints + dense descriptors)
        with torch.no_grad():
            sp0 = sp_model({"image": torch.from_numpy(master_norm).unsqueeze(0).unsqueeze(0).to(device),
                            "image_size": torch.tensor([[h, w]])})
            sp1 = sp_model({"image": torch.from_numpy(input_norm).unsqueeze(0).unsqueeze(0).to(device),
                            "image_size": torch.tensor([[h, w]])})

        kp0 = sp0["keypoints"][0].cpu().numpy()
        sc0 = sp0["keypoint_scores"][0].cpu().numpy()
        kp1 = sp1["keypoints"][0].cpu().numpy()
        sc1 = sp1["keypoint_scores"][0].cpu().numpy()

        # depth 필터
        max_kp = args.max_num_keypoints
        from precompute_fpfh_resample import filter_keypoints_by_depth
        kp0_f, sc0_f, n0 = filter_keypoints_by_depth(kp0, sc0, master_resized, max_kp)
        kp1_f, sc1_f, n1 = filter_keypoints_by_depth(kp1, sc1, input_resized, max_kp)

        # SP descriptor 샘플링 (256D)
        sp_desc0 = sample_sp_descriptors(sp0, kp0_f, n0, max_kp)
        sp_desc1 = sample_sp_descriptors(sp1, kp1_f, n1, max_kp)

        # FPFH on original resolution (33D)
        depth_real0 = master_raw.astype(np.float64) / 65535.0
        depth_real0 = clip_start + depth_real0 * (clip_end - clip_start)
        depth_real0[master_raw == 0] = 0.0
        depth_real1 = input_raw.astype(np.float64) / 65535.0
        depth_real1 = clip_start1 + depth_real1 * (clip_end1 - clip_start1)
        depth_real1[input_raw == 0] = 0.0

        kp0_orig = kp0_f.copy()
        kp0_orig[:n0] *= scale
        kp1_orig = kp1_f.copy()
        kp1_orig[:n1] *= scale

        pcd0, fpfh0_dense = compute_dense_fpfh(depth_real0, K0, args.fpfh_radius)
        pcd1, fpfh1_dense = compute_dense_fpfh(depth_real1, K1, args.fpfh_radius)

        fpfh_desc0 = lookup_fpfh(kp0_orig, n0, depth_real0, K0, pcd0, fpfh0_dense, max_kp)
        fpfh_desc1 = lookup_fpfh(kp1_orig, n1, depth_real1, K1, pcd1, fpfh1_dense, max_kp)

        # Hybrid: concat SP(256D) + FPFH(33D) = 289D
        desc0 = np.concatenate([sp_desc0, fpfh_desc0], axis=1)  # (512, 289)
        desc1 = np.concatenate([sp_desc1, fpfh_desc1], axis=1)  # (512, 289)

        # batch
        batch = {
            "view0": {
                "image": torch.from_numpy(master_norm).unsqueeze(0).unsqueeze(0),
                "image_size": torch.tensor([[h, w]]),
                "keypoints": torch.from_numpy(kp0_f).unsqueeze(0).float(),
                "keypoint_scores": torch.from_numpy(sc0_f).unsqueeze(0).float(),
                "descriptors": torch.from_numpy(desc0).unsqueeze(0).float(),
            },
            "view1": {
                "image": torch.from_numpy(input_norm).unsqueeze(0).unsqueeze(0),
                "image_size": torch.tensor([[h, w]]),
                "keypoints": torch.from_numpy(kp1_f).unsqueeze(0).float(),
                "keypoint_scores": torch.from_numpy(sc1_f).unsqueeze(0).float(),
                "descriptors": torch.from_numpy(desc1).unsqueeze(0).float(),
            },
        }

        pred, batch = run_inference(model, batch, device)
        output_path = output_dir / f"{args.split}_{idx:05d}.png"
        n_correct, n_gt = visualize_pair(pred, batch, 0, output_path, csv_path=str(csv_path),
                                         gt_radius=gt_radius, orig_size=orig_h, image_size=args.image_size)
        total_correct += n_correct
        total_gt += n_gt

    if total_gt > 0:
        print(f"\n=== Aggregate Recall: {total_correct}/{total_gt} ({total_correct/total_gt*100:.1f}%) ===")
    print(f"Done! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
