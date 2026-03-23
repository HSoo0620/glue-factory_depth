"""
FPFH descriptor가 어떻게 추출되는지 단계별로 시각화.

사용법:
    # 특정 이미지 1장
    python visualize_fpfh.py --image gluefactory/datasets/mitsubishi/dataset_depth/depth_raw_0000.png

    # 캐시된 .npz 사용 (precompute 이후)
    python visualize_fpfh.py --image gluefactory/datasets/mitsubishi/dataset_depth/depth_raw_0000.png --use_cache

    # 파라미터 조정
    python visualize_fpfh.py --image path/to/depth.png --fpfh_radius 15.0 --max_num_keypoints 256

시각화 내용:
    1. 원본 depth 이미지 + SuperPoint keypoints
    2. keypoint 위치의 3D point cloud (x, y, depth)
    3. 추정된 normals
    4. FPFH histogram (33-bin) 샘플
    5. 전체 keypoint별 FPFH heatmap
"""

import argparse
import torch
import numpy as np
import cv2
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path


def load_superpoint(device, detection_threshold=0.003, max_num_keypoints=512):
    from gluefactory.models.extractors.superpoint_open import SuperPoint
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


def compute_fpfh_with_intermediates(depth_map, keypoints, fpfh_radius=10.0, fpfh_max_nn=100):
    """FPFH 계산 + 중간 결과(3D points, normals) 반환"""
    N = keypoints.shape[0]

    points_3d = []
    valid_mask = []
    for i in range(N):
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
        x = np.clip(x, 0, depth_map.shape[1] - 1)
        y = np.clip(y, 0, depth_map.shape[0] - 1)
        z = depth_map[y, x]
        points_3d.append([float(x), float(y), float(z * 1000)])
        valid_mask.append(z > 0)

    points_3d = np.array(points_3d, dtype=np.float64)
    valid_mask = np.array(valid_mask)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius * 2, max_nn=30)
    )
    normals = np.asarray(pcd.normals)

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=fpfh_max_nn),
    )
    fpfh_np = np.array(fpfh.data).T.astype(np.float32)  # (N, 33)
    # L2 normalize
    norms = np.linalg.norm(fpfh_np, axis=1, keepdims=True)
    fpfh_np = fpfh_np / (norms + 1e-8)

    return {
        "points_3d": points_3d,
        "normals": normals,
        "valid_mask": valid_mask,
        "fpfh": fpfh_np,
    }


def visualize(depth_map, keypoints, kp_scores, intermediates, output_dir, stem, sample_indices=None):
    points_3d = intermediates["points_3d"]
    normals = intermediates["normals"]
    valid_mask = intermediates["valid_mask"]
    fpfh = intermediates["fpfh"]
    N = keypoints.shape[0]

    if sample_indices is None:
        # 점수 높은 keypoint 중 유효한 것 8개 선택
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) > 8:
            scores_valid = kp_scores[valid_idx]
            top8 = np.argsort(scores_valid)[-8:]
            sample_indices = valid_idx[top8]
        else:
            sample_indices = valid_idx

    # =========================================================
    # Figure 1: 4-panel overview
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    # --- (1) Depth image + keypoints ---
    ax = axes[0, 0]
    ax.imshow(depth_map, cmap="inferno")
    sc = ax.scatter(
        keypoints[:, 0], keypoints[:, 1],
        c=kp_scores, cmap="viridis", s=8, alpha=0.7, edgecolors="white", linewidths=0.3,
    )
    # 샘플 keypoint 번호 표시
    for i, idx in enumerate(sample_indices):
        ax.annotate(
            f"{idx}", (keypoints[idx, 0], keypoints[idx, 1]),
            color="red", fontsize=7, fontweight="bold",
            ha="center", va="bottom", xytext=(0, 4), textcoords="offset points",
        )
    fig.colorbar(sc, ax=ax, label="keypoint score", shrink=0.7)
    ax.set_title(f"1. Depth + SuperPoint keypoints (N={N})", fontsize=13)
    ax.set_axis_off()

    # --- (2) 3D point cloud (x, y, z) color = depth ---
    ax = axes[0, 1]
    z_vals = points_3d[:, 2]
    sc2 = ax.scatter(
        points_3d[:, 0], points_3d[:, 1],
        c=z_vals, cmap="plasma", s=10, alpha=0.7,
    )
    for idx in sample_indices:
        ax.annotate(
            f"{idx}", (points_3d[idx, 0], points_3d[idx, 1]),
            color="red", fontsize=7, fontweight="bold",
            ha="center", va="bottom", xytext=(0, 4), textcoords="offset points",
        )
    fig.colorbar(sc2, ax=ax, label="z (depth x1000)", shrink=0.7)
    ax.set_title("2. Keypoint → 3D points (x, y, z=depth*1000)", fontsize=13)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    # --- (3) Normals ---
    ax = axes[1, 0]
    ax.imshow(depth_map, cmap="gray", alpha=0.3)
    # normal을 RGB로 표현 (nx, ny, nz → r, g, b)
    norm_rgb = (normals + 1) / 2  # [-1,1] → [0,1]
    norm_rgb = np.clip(norm_rgb, 0, 1)
    ax.scatter(
        points_3d[:, 0], points_3d[:, 1],
        c=norm_rgb, s=10, alpha=0.8,
    )
    # quiver로 normal 방향 표시 (샘플만)
    scale = 15
    for idx in sample_indices:
        ax.arrow(
            points_3d[idx, 0], points_3d[idx, 1],
            normals[idx, 0] * scale, normals[idx, 1] * scale,
            head_width=3, head_length=2, fc="red", ec="red", alpha=0.8,
        )
        ax.annotate(
            f"{idx}", (points_3d[idx, 0], points_3d[idx, 1]),
            color="yellow", fontsize=7, fontweight="bold",
            ha="center", va="bottom", xytext=(0, 6), textcoords="offset points",
        )
    ax.set_title("3. Estimated normals (color=normal direction, arrow=sample)", fontsize=13)
    ax.set_axis_off()

    # --- (4) FPFH heatmap (all keypoints) ---
    ax = axes[1, 1]
    im = ax.imshow(fpfh.T, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_xlabel("Keypoint index")
    ax.set_ylabel("FPFH bin (33 bins)")
    ax.set_title("4. FPFH descriptors heatmap (all keypoints)", fontsize=13)
    fig.colorbar(im, ax=ax, label="FPFH value", shrink=0.7)
    # 샘플 위치 표시
    for idx in sample_indices:
        ax.axvline(x=idx, color="cyan", linewidth=0.5, alpha=0.5)

    fig.suptitle(f"FPFH Pipeline Overview — {stem}", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path1 = output_dir / f"{stem}_overview.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path1}")

    # =========================================================
    # Figure 2: 샘플 keypoint별 FPFH histogram 상세
    # =========================================================
    n_samples = len(sample_indices)
    cols = min(4, n_samples)
    rows = (n_samples + cols - 1) // cols
    fig2, axes2 = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_samples == 1:
        axes2 = np.array([axes2])
    axes2 = axes2.flatten()

    # FPFH 33 bins: 11 bins each for (alpha, phi, theta)
    bin_labels = (
        [f"a{i}" for i in range(11)] +
        [f"p{i}" for i in range(11)] +
        [f"t{i}" for i in range(11)]
    )

    for i, idx in enumerate(sample_indices):
        ax = axes2[i]
        hist = fpfh[idx]
        colors = (["#e74c3c"] * 11 + ["#2ecc71"] * 11 + ["#3498db"] * 11)
        ax.bar(range(33), hist, color=colors, alpha=0.8, width=0.8)
        ax.set_xticks(range(33))
        ax.set_xticklabels(bin_labels, rotation=90, fontsize=6)
        ax.set_title(
            f"KP #{idx}  pos=({keypoints[idx,0]:.0f},{keypoints[idx,1]:.0f})  "
            f"z={points_3d[idx,2]:.1f}",
            fontsize=10,
        )
        ax.set_ylabel("FPFH value")

        # 구간 구분선
        ax.axvline(x=10.5, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(x=21.5, color="gray", linewidth=0.5, linestyle="--")

    # 빈 subplot 숨김
    for i in range(n_samples, len(axes2)):
        axes2[i].set_visible(False)

    fig2.suptitle(
        f"FPFH Histograms — red: α(angle1)  green: φ(angle2)  blue: θ(angle3)\n{stem}",
        fontsize=13, y=1.02,
    )
    fig2.tight_layout()
    path2 = output_dir / f"{stem}_histograms.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # =========================================================
    # Figure 3: FPFH descriptor 유사도 (similarity matrix)
    # =========================================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))
    # L2 normalize 후 cosine similarity
    fpfh_norm = fpfh / (np.linalg.norm(fpfh, axis=1, keepdims=True) + 1e-8)
    sim = fpfh_norm @ fpfh_norm.T
    im3 = ax3.imshow(sim, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0)
    ax3.set_title(f"FPFH Cosine Similarity Matrix (N={N})", fontsize=13)
    ax3.set_xlabel("Keypoint index")
    ax3.set_ylabel("Keypoint index")
    fig3.colorbar(im3, ax=ax3, label="Cosine similarity", shrink=0.8)
    fig3.tight_layout()
    path3 = output_dir / f"{stem}_similarity.png"
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {path3}")


def main():
    parser = argparse.ArgumentParser(description="FPFH 추출 과정 시각화")
    parser.add_argument("--image", type=str, required=True, help="Depth 이미지 경로")
    parser.add_argument("--use_cache", action="store_true", help="precompute된 .npz 캐시 사용")
    parser.add_argument("--output_dir", type=str, default="results/fpfh_viz")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_num_keypoints", type=int, default=512)
    parser.add_argument("--detection_threshold", type=float, default=0.003)
    parser.add_argument("--fpfh_radius", type=float, default=10.0)
    parser.add_argument("--fpfh_max_nn", type=int, default=100)
    parser.add_argument("--sample_indices", type=int, nargs="*", default=None,
                        help="시각화할 keypoint 인덱스 (미지정 시 자동 선택)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(args.image)
    stem = img_path.stem
    device = args.device if torch.cuda.is_available() else "cpu"

    # 이미지 로드
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    if img.max() > 255:
        depth_map = img / 65535.0
    else:
        depth_map = img / 255.0
    if depth_map.ndim == 3:
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)

    if args.use_cache:
        # 캐시 로드
        cache_dir = Path("gluefactory/datasets/mitsubishi/fpfh_cache")
        npz = np.load(cache_dir / f"{stem}.npz")
        keypoints = npz["keypoints"]
        kp_scores = npz["keypoint_scores"]
        print(f"Loaded cache: {cache_dir / stem}.npz")
        # 중간 결과는 다시 계산 (normals 필요)
        intermediates = compute_fpfh_with_intermediates(
            depth_map, keypoints, args.fpfh_radius, args.fpfh_max_nn
        )
    else:
        # SuperPoint + depth 필터링 + FPFH 온라인 계산
        sp = load_superpoint(device, args.detection_threshold, args.max_num_keypoints)
        h, w = depth_map.shape
        img_tensor = torch.from_numpy(depth_map).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            sp_pred = sp({"image": img_tensor, "image_size": torch.tensor([[h, w]])})
        raw_kp = sp_pred["keypoints"][0].cpu().numpy()
        raw_sc = sp_pred["keypoint_scores"][0].cpu().numpy()
        n_raw = len(raw_kp)

        # depth > 0 필터링
        valid_mask = np.array([
            depth_map[
                np.clip(int(raw_kp[i, 1]), 0, h - 1),
                np.clip(int(raw_kp[i, 0]), 0, w - 1),
            ] > 0
            for i in range(n_raw)
        ])
        keypoints = raw_kp[valid_mask]
        kp_scores = raw_sc[valid_mask]
        # score 순 정렬
        if len(kp_scores) > 0:
            order = np.argsort(-kp_scores)
            keypoints = keypoints[order][:args.max_num_keypoints]
            kp_scores = kp_scores[order][:args.max_num_keypoints]

        print(f"SuperPoint: {n_raw} raw → {len(keypoints)} after depth>0 filtering")

        intermediates = compute_fpfh_with_intermediates(
            depth_map, keypoints, args.fpfh_radius, args.fpfh_max_nn
        )

    print(f"FPFH: {intermediates['fpfh'].shape} descriptors computed")
    print(f"Valid points (z>0): {intermediates['valid_mask'].sum()}/{len(keypoints)}")

    visualize(
        depth_map, keypoints, kp_scores, intermediates,
        output_dir, stem, sample_indices=args.sample_indices,
    )

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
