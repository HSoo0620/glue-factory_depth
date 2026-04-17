"""
FPFH descriptor로 학습된 LightGlue 모델 테스트.
추론 시에는 온라인으로 SuperPoint keypoint + FPFH descriptor를 계산.

사용법:
    # 데이터셋 기반 테스트
    python test_depth_fpfh.py \
        --checkpoint outputs/training/0313_fpfh_radius_5.0/checkpoint_best.tar \
        --indices 0 10 50 90 \
        --output_dir results/matchings/0313_fpfh_radius_5.0 \
        --fpfh_radius 5.0

    # 커스텀 이미지 2장 (calib ini 필요)
    python test_depth_fpfh.py \
        --checkpoint outputs/training/Depth_fpfh/checkpoint_best.tar \
        --img0 /path/to/master.png --calib0 /path/to/calib_master.ini \
        --img1 /path/to/input.png  --calib1 /path/to/calib_input.ini

    # 옵션
    python test_depth_fpfh.py \
        --checkpoint outputs/training/Depth_fpfh/checkpoint_best.tar \
        --num_samples 20 --fpfh_radius 1.5
"""

import argparse
import configparser
import torch
import numpy as np
import cv2
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.models.extractors.superpoint_open import SuperPoint
from gluefactory.datasets.mitsubishi_depth_dataset import (
    MitsubishiDepthDataset,
)


def parse_calib_ini(ini_path):
    """calib_XXXX.ini에서 K(intrinsic), clip_start, clip_end 파싱 (precompute_fpfh.py와 동일)"""
    config = configparser.ConfigParser()
    config.read(str(ini_path))
    section = config.sections()[0]

    k_vals = [float(v) for v in config[section]["k_matrix"].split(":")[3].split(",")]
    K = np.array(k_vals, dtype=np.float64).reshape(3, 3)

    clip_start = float(config[section]["clip_start"])
    clip_end = float(config[section]["clip_end"])

    return K, clip_start, clip_end


def get_ini_path(img_path):
    """depth 이미지 경로에서 대응하는 calib ini 경로 반환"""
    img_path = Path(img_path)
    img_idx = img_path.stem.replace("depth_raw_", "")
    return img_path.parent / f"calib_{img_idx}.ini"


def load_depth_image(img_path, ini_path):
    """
    uint16 depth 이미지 로드.
    Returns:
        depth_norm: [0, 1] 정규화 (SuperPoint 입력용, 학습 dataset과 동일)
        depth_real: 실제 depth 값 (3D 역투영용, clip_start/clip_end 적용)
        K: 카메라 intrinsic matrix
    """
    raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    if raw is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    K, clip_start, clip_end = parse_calib_ini(ini_path)

    depth_norm = raw / 65535.0  # [0, 1], 학습 dataset과 동일
    depth_real = clip_start + depth_norm * (clip_end - clip_start)
    depth_real[raw == 0] = 0.0  # 배경(원본 0)은 그대로 0 유지

    return depth_norm, depth_real, K


def load_model(checkpoint_path, device):
    cp = torch.load(checkpoint_path, map_location="cpu")
    conf = OmegaConf.create(cp["conf"])
    model = get_model(conf.model.name)(conf.model).to(device)
    model.load_state_dict(cp["model"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {cp['epoch']})")
    return model, conf


def load_superpoint(device, detection_threshold=0.001, max_num_keypoints=512):
    sp_conf = {
        "descriptor_dim": 256,
        "nms_radius": 3,
        "max_num_keypoints": max_num_keypoints * 2,  # 넉넉하게 뽑고 depth 필터링
        "force_num_keypoints": False,
        "detection_threshold": detection_threshold,
        "remove_borders": 4,
        "channels": [64, 64, 128, 128, 256],
        "dense_outputs": None,
        "weights": None,
    }
    sp = SuperPoint(sp_conf).to(device)
    sp.eval()
    return sp


def filter_keypoints_by_depth(keypoints, scores, depth_map, max_num_keypoints):
    """depth > 0 인 keypoint만 남기고 zero-padding (precompute와 동일 로직)"""
    N = keypoints.shape[0]
    valid_mask = np.zeros(N, dtype=bool)
    for i in range(N):
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
        x = np.clip(x, 0, depth_map.shape[1] - 1)
        y = np.clip(y, 0, depth_map.shape[0] - 1)
        valid_mask[i] = depth_map[y, x] > 0

    valid_kp = keypoints[valid_mask]
    valid_sc = scores[valid_mask]

    if len(valid_sc) > 0:
        order = np.argsort(-valid_sc)
        valid_kp = valid_kp[order]
        valid_sc = valid_sc[order]

    n_valid = min(len(valid_kp), max_num_keypoints)
    out_kp = np.zeros((max_num_keypoints, 2), dtype=np.float32)
    out_sc = np.zeros(max_num_keypoints, dtype=np.float32)
    out_kp[:n_valid] = valid_kp[:n_valid]
    out_sc[:n_valid] = valid_sc[:n_valid]
    return out_kp, out_sc, n_valid


@torch.no_grad()
def extract_fpfh_online(sp_model, depth_norm, depth_real, K, device,
                         fpfh_radius=1.0, fpfh_max_nn=100, max_num_keypoints=512):
    """
    온라인으로 SuperPoint keypoints + depth 필터링 + FPFH descriptors 추출.
    Dense point cloud 전체에서 FPFH를 계산한 뒤, 각 keypoint 위치의 descriptor를 lookup.
    (precompute_fpfh_dense.py와 동일한 방식)

    Args:
        depth_norm: [0, 1] 정규화된 depth (SuperPoint 입력용)
        depth_real: 실제 depth 값 (clip_start/clip_end 적용, 3D 역투영용)
        K: 카메라 intrinsic matrix (3x3)
    """
    h, w = depth_norm.shape
    img_tensor = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(0).to(device)
    sp_pred = sp_model({"image": img_tensor, "image_size": torch.tensor([[h, w]])})

    raw_kp = sp_pred["keypoints"][0].cpu().numpy()
    raw_sc = sp_pred["keypoint_scores"][0].cpu().numpy()

    # depth > 0 필터링 + score 순 정렬 + zero padding
    kp, sc, n_valid = filter_keypoints_by_depth(raw_kp, raw_sc, depth_norm, max_num_keypoints)

    # Dense FPFH 계산 후 keypoint lookup
    fpfh = np.zeros((max_num_keypoints, 33), dtype=np.float32)
    if n_valid >= 3:
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # 1) Dense point cloud 생성 (nonzero 픽셀 전체)
        vs, us = np.where(depth_real > 0)
        zs = depth_real[vs, us]
        xs = (us - cx) * zs / fx
        ys = (vs - cy) * zs / fy
        dense_pts = np.stack([xs, ys, zs], axis=1).astype(np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dense_pts)

        # 2) Dense cloud에서 normals + FPFH 계산
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius * 2, max_nn=30)
        )
        fpfh_feat = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=fpfh_max_nn),
        )
        fpfh_dense = np.array(fpfh_feat.data).T.astype(np.float32)

        # 3) 각 keypoint → dense cloud에서 nearest neighbor lookup
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        valid_kp = kp[:n_valid]
        for i in range(n_valid):
            u, v = int(valid_kp[i, 0]), int(valid_kp[i, 1])
            u = np.clip(u, 0, depth_real.shape[1] - 1)
            v = np.clip(v, 0, depth_real.shape[0] - 1)
            z = float(depth_real[v, u])
            x_3d = (u - cx) * z / fx
            y_3d = (v - cy) * z / fy
            _, idx, _ = kdtree.search_knn_vector_3d([x_3d, y_3d, z], 1)
            fpfh[i] = fpfh_dense[idx[0]]

        # L2 normalize
        norms = np.linalg.norm(fpfh[:n_valid], axis=1, keepdims=True)
        fpfh[:n_valid] = fpfh[:n_valid] / (norms + 1e-8)

    return {
        "keypoints": torch.from_numpy(kp).unsqueeze(0).to(device),        # (1, N, 2)
        "keypoint_scores": torch.from_numpy(sc).unsqueeze(0).to(device),   # (1, N)
        "descriptors": torch.from_numpy(fpfh).unsqueeze(0).to(device),     # (1, N, 33)
    }


@torch.no_grad()
def run_inference(model, sp_model, depth_norm0, depth_real0, K0,
                  depth_norm1, depth_real1, K1,
                  device, fpfh_radius, fpfh_max_nn, max_num_keypoints=512):
    """두 depth map에 대해 온라인 FPFH + LightGlue 매칭"""
    ext0 = extract_fpfh_online(sp_model, depth_norm0, depth_real0, K0, device,
                                fpfh_radius, fpfh_max_nn, max_num_keypoints)
    ext1 = extract_fpfh_online(sp_model, depth_norm1, depth_real1, K1, device,
                                fpfh_radius, fpfh_max_nn, max_num_keypoints)

    h0, w0 = depth_norm0.shape
    h1, w1 = depth_norm1.shape

    data = {
        "view0": {"image": torch.from_numpy(depth_norm0).unsqueeze(0).unsqueeze(0).to(device),
                   "image_size": torch.tensor([[h0, w0]])},
        "view1": {"image": torch.from_numpy(depth_norm1).unsqueeze(0).unsqueeze(0).to(device),
                   "image_size": torch.tensor([[h1, w1]])},
    }

    pred = {
        "keypoints0": ext0["keypoints"],
        "keypoint_scores0": ext0["keypoint_scores"],
        "descriptors0": ext0["descriptors"],
        "keypoints1": ext1["keypoints"],
        "keypoint_scores1": ext1["keypoint_scores"],
        "descriptors1": ext1["descriptors"],
    }

    # LightGlue matcher만 실행 (extractor skip)
    match_pred = model.matcher({**data, **pred})
    pred.update(match_pred)

    return pred, data


def visualize_pair(pred, data, output_path, label="", csv_path=None, gt_radius=3):
    """
    한 쌍의 매칭 결과를 시각화.
    색상 분류:
        하늘색: 정답 pair (GT에 정의 + 비가림 + 정확히 매칭)
        보라색: GT에 정의된 포인트인데 매칭이 잘못됨
        녹색:   정답 pair를 잘 맞췄는데 GT에서 가려짐(occluded)으로 표시
        빨간색: GT에 포인트 정의가 없는데 매칭됨
    """
    import pandas as pd

    img0 = data["view0"]["image"][0].cpu()
    img1 = data["view1"]["image"][0].cpu()

    kp0 = pred["keypoints0"][0].cpu().numpy()
    kp1 = pred["keypoints1"][0].cpu().numpy()
    m0 = pred["matches0"][0].cpu().numpy()

    valid = m0 > -1
    valid_idx = np.where(valid)[0]
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
            kp0_pt = mkp0[i]  # 매칭된 view0 keypoint
            kp1_pt = mkp1[i]  # 매칭된 view1 keypoint

            # view0 keypoint에서 가장 가까운 GT master point 찾기
            dists_master = np.linalg.norm(all_master_xy - kp0_pt, axis=1)
            nearest_idx = np.argmin(dists_master)
            nearest_dist = dists_master[nearest_idx]

            if nearest_dist < gt_radius:
                # GT에 정의된 포인트
                gt_input_pt = all_input_xy[nearest_idx]
                is_occluded = all_occluded[nearest_idx]
                match_dist = np.linalg.norm(kp1_pt - gt_input_pt)

                if match_dist < gt_radius:
                    # 정확히 매칭됨
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
    if label:
        info = f"[{label}]  " + info
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
    parser = argparse.ArgumentParser(description="FPFH + LightGlue 테스트 (온라인 FPFH)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/test_fpfh")

    # FPFH 파라미터
    parser.add_argument("--fpfh_radius", type=float, default=1.0)
    parser.add_argument("--fpfh_max_nn", type=int, default=100)
    parser.add_argument("--detection_threshold", type=float, default=0.001)
    parser.add_argument("--max_num_keypoints", type=int, default=512)

    # 커스텀 이미지 (calib ini 필수)
    parser.add_argument("--img0", type=str, default=None)
    parser.add_argument("--calib0", type=str, default=None, help="img0에 대응하는 calib ini 경로")
    parser.add_argument("--img1", type=str, default=None)
    parser.add_argument("--calib1", type=str, default=None, help="img1에 대응하는 calib ini 경로")

    # 데이터셋 모드
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--indices", type=int, nargs="*", default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    model, conf = load_model(args.checkpoint, device)
    sp_model = load_superpoint(device, args.detection_threshold, args.max_num_keypoints)

    # --- 커스텀 이미지 ---
    if args.img0 and args.img1:
        if not args.calib0 or not args.calib1:
            raise ValueError("커스텀 이미지 모드에서는 --calib0, --calib1 (calib ini 경로)이 필요합니다.")
        print(f"Custom pair: {args.img0} <-> {args.img1}")
        d0_norm, d0_real, K0 = load_depth_image(args.img0, args.calib0)
        d1_norm, d1_real, K1 = load_depth_image(args.img1, args.calib1)
        pred, data = run_inference(model, sp_model,
                                   d0_norm, d0_real, K0,
                                   d1_norm, d1_real, K1,
                                   device, args.fpfh_radius, args.fpfh_max_nn,
                                   args.max_num_keypoints)
        out_name = f"custom_{Path(args.img0).stem}__{Path(args.img1).stem}.png"
        visualize_pair(pred, data, output_dir / out_name, label="custom_fpfh")
        return

    # --- 데이터셋 모드 ---
    dataset = MitsubishiDepthDataset(split=args.split)
    print(f"{args.split} dataset: {len(dataset)} pairs")

    if args.indices is not None:
        indices = args.indices
    else:
        indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
        indices = sorted(indices)

    print(f"Testing {len(indices)} pairs (online FPFH): {indices}")

    for idx in indices:
        sample = dataset[idx]

        # calib ini에서 K, clip_start, clip_end 로드 후 실제 depth 복원
        master_path = Path(sample["master_path"])
        input_path = Path(sample["input_path"])

        d0_norm, d0_real, K0 = load_depth_image(master_path, get_ini_path(master_path))
        d1_norm, d1_real, K1 = load_depth_image(input_path, get_ini_path(input_path))

        pred, data = run_inference(model, sp_model,
                                   d0_norm, d0_real, K0,
                                   d1_norm, d1_real, K1,
                                   device, args.fpfh_radius, args.fpfh_max_nn,
                                   args.max_num_keypoints)

        csv_path = sample.get("csv_path", None)
        gt_radius = conf.model.ground_truth.get("gt_radius", 3)

        visualize_pair(pred, data, output_dir / f"{args.split}_{idx:05d}.png",
                       label=f"fpfh#{idx}", csv_path=csv_path, gt_radius=gt_radius)

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
