# Registration RMSE Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform 기반 RMSE 평가 스크립트로 ISS+SHOT, ISS+FPFH 방법론 간 Registration 성능을 공정 비교

**Architecture:** 모델 추론 → matching pairs → Grid 3D 변환 → RANSAC T_est → GT CSV SVD T_gt → 포인트 클라우드 샘플링 RMSE. 양방향(forward+reverse) 평균. 시각화는 overlay 1장으로 간소화.

**Tech Stack:** PyTorch, Open3D, NumPy, pandas, matplotlib

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `eval_registration_iss_shot.py` | ISS+SHOT+LG 평가 (자체 완결 스크립트) |
| Create | `eval_registration_iss_fpfh.py` | ISS+FPFH+LG 평가 (SHOT 버전에서 import만 변경) |
| Create | `tests/test_registration_eval.py` | 핵심 수학 함수 단위 테스트 |

---

### Task 1: Core math function tests

**Files:**
- Create: `tests/test_registration_eval.py`

- [ ] **Step 1: Write test for pixel_to_grid3d**

```python
# tests/test_registration_eval.py
import unittest
import numpy as np


CLIP_START = 0.1
CLIP_END = 1000.0
GRID_DX = 0.05
GRID_DY = 0.05
GRID_DZ = 0.02


def pixel_to_grid3d(keypoints_2d, depth_map_raw, clip_start=CLIP_START,
                    clip_end=CLIP_END, grid_dx=GRID_DX, grid_dy=GRID_DY,
                    grid_dz=GRID_DZ):
    """2D keypoint + raw depth -> Grid 3D (u*dx, v*dy, depth_real*dz)."""
    N = keypoints_2d.shape[0]
    points_3d = np.zeros((N, 3), dtype=np.float64)
    valid_mask = np.zeros(N, dtype=bool)
    H, W = depth_map_raw.shape
    for i in range(N):
        u = int(np.clip(keypoints_2d[i, 0], 0, W - 1))
        v = int(np.clip(keypoints_2d[i, 1], 0, H - 1))
        d_raw = float(depth_map_raw[v, u])
        if d_raw <= 0:
            continue
        depth_real = clip_start + (d_raw / 65535.0) * (clip_end - clip_start)
        points_3d[i] = [u * grid_dx, v * grid_dy, depth_real * grid_dz]
        valid_mask[i] = True
    return points_3d, valid_mask


class TestPixelToGrid3D(unittest.TestCase):
    def test_basic_conversion(self):
        depth_raw = np.zeros((100, 100), dtype=np.uint16)
        depth_raw[50, 30] = 32768  # roughly mid-range
        kp = np.array([[30.0, 50.0]])
        pts, mask = pixel_to_grid3d(kp, depth_raw)
        self.assertTrue(mask[0])
        self.assertAlmostEqual(pts[0, 0], 30 * 0.05, places=5)   # X
        self.assertAlmostEqual(pts[0, 1], 50 * 0.05, places=5)   # Y
        depth_real = 0.1 + (32768 / 65535.0) * 999.9
        self.assertAlmostEqual(pts[0, 2], depth_real * 0.02, places=5)  # Z

    def test_zero_depth_invalid(self):
        depth_raw = np.zeros((100, 100), dtype=np.uint16)
        kp = np.array([[10.0, 10.0]])
        pts, mask = pixel_to_grid3d(kp, depth_raw)
        self.assertFalse(mask[0])
        np.testing.assert_array_equal(pts[0], [0, 0, 0])

    def test_multiple_points(self):
        depth_raw = np.full((100, 100), 65535, dtype=np.uint16)
        kp = np.array([[0.0, 0.0], [99.0, 99.0]])
        pts, mask = pixel_to_grid3d(kp, depth_raw)
        self.assertTrue(mask.all())
        depth_real_max = 0.1 + 1.0 * 999.9  # = 1000.0
        self.assertAlmostEqual(pts[1, 0], 99 * 0.05, places=5)
        self.assertAlmostEqual(pts[1, 2], depth_real_max * 0.02, places=5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && python -m pytest tests/test_registration_eval.py -v`
Expected: PASS (함수가 테스트 파일 내에 정의되어 있으므로 바로 통과 확인)

- [ ] **Step 3: Write test for rigid_transform_svd**

`tests/test_registration_eval.py`에 추가:

```python
def rigid_transform_svd(P_src, P_dst):
    """SVD로 rigid transform 추정: P_dst = R @ P_src + t"""
    c_src = P_src.mean(axis=0)
    c_dst = P_dst.mean(axis=0)
    H = (P_src - c_src).T @ (P_dst - c_dst)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c_dst - R @ c_src
    return R, t


class TestRigidTransformSVD(unittest.TestCase):
    def test_identity(self):
        pts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        R, t = rigid_transform_svd(pts, pts)
        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=10)
        np.testing.assert_array_almost_equal(t, np.zeros(3), decimal=10)

    def test_known_rotation_translation(self):
        np.random.seed(42)
        pts_src = np.random.randn(50, 3) * 10
        angle = np.pi / 6  # 30 degrees around Z
        R_true = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])
        t_true = np.array([5.0, -3.0, 2.0])
        pts_dst = (R_true @ pts_src.T).T + t_true

        R_est, t_est = rigid_transform_svd(pts_src, pts_dst)
        np.testing.assert_array_almost_equal(R_est, R_true, decimal=10)
        np.testing.assert_array_almost_equal(t_est, t_true, decimal=10)

    def test_reflection_handling(self):
        pts_src = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                           dtype=np.float64)
        pts_dst = pts_src.copy()
        R, t = rigid_transform_svd(pts_src, pts_dst)
        self.assertGreater(np.linalg.det(R), 0)  # proper rotation
```

- [ ] **Step 4: Write test for compute_transform_rmse**

```python
def compute_transform_rmse(P_src, R_est, t_est, R_gt, t_gt):
    """T_est vs T_gt를 동일 소스 포인트에 적용하여 RMSE 산출."""
    P_est = (R_est @ P_src.T).T + t_est
    P_gt = (R_gt @ P_src.T).T + t_gt
    return float(np.sqrt(np.mean(np.sum((P_est - P_gt) ** 2, axis=1))))


class TestComputeTransformRMSE(unittest.TestCase):
    def test_identical_transforms(self):
        pts = np.random.randn(100, 3)
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        rmse = compute_transform_rmse(pts, R, t, R, t)
        self.assertAlmostEqual(rmse, 0.0, places=10)

    def test_known_error(self):
        pts = np.array([[0, 0, 0]], dtype=np.float64)
        R = np.eye(3)
        t_est = np.array([1.0, 0.0, 0.0])
        t_gt = np.array([0.0, 0.0, 0.0])
        rmse = compute_transform_rmse(pts, R, t_est, R, t_gt)
        self.assertAlmostEqual(rmse, 1.0, places=10)

    def test_multiple_points(self):
        pts = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float64)
        R = np.eye(3)
        t_est = np.array([1.0, 0.0, 0.0])
        t_gt = np.array([0.0, 0.0, 0.0])
        # Both points have 1.0 error in X → RMSE = 1.0
        rmse = compute_transform_rmse(pts, R, t_est, R, t_gt)
        self.assertAlmostEqual(rmse, 1.0, places=10)
```

- [ ] **Step 5: Run all tests**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && python -m pytest tests/test_registration_eval.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_registration_eval.py
git commit -m "test: add unit tests for registration RMSE eval core functions"
```

---

### Task 2: eval_registration_iss_shot.py

**Files:**
- Create: `eval_registration_iss_shot.py`

- [ ] **Step 1: Write imports, constants, pixel_to_grid3d**

```python
"""
ISS+SHOT+LG Registration 평가 — Transform 기반 RMSE.

Grid 3D 좌표계: (u*grid_dx, v*grid_dy, depth_real*grid_dz)
T_gt: GT CSV 비-occluded 대응점 SVD
RMSE: 포인트 클라우드 30K 샘플, T_est vs T_gt
양방향: (master,input) + (input,master) 평균

사용법:
    python eval_registration_iss_shot.py --experiment 0407_resample2_iss_shot352_lg --indices 0 10 50 90
"""

import argparse
import torch
import numpy as np
import cv2
import open3d as o3d
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.datasets.mitsubishi_resample2_iss_shot_dataset import (
    MitsubishiResample2ISSSHOTDataset,
    resample2_iss_shot_collate_fn,
    CROP_X0,
    CROP_Y0,
    CROP_SIZE,
)
from gluefactory.utils.tensor import batch_to_device

# ─── 상수 ──────────────────────────────────────
CLIP_START = 0.1
CLIP_END = 1000.0
ORIG_SIZE = 5761
GRID_DX = 0.05
GRID_DY = 0.05
GRID_DZ = 0.02


def pixel_to_grid3d(keypoints_2d, depth_map_raw):
    """2D keypoint + raw depth -> Grid 3D (u*dx, v*dy, depth_real*dz).
    keypoints_2d: (N, 2) 원본 해상도(5761) 기준
    depth_map_raw: (H, W) uint16
    """
    N = keypoints_2d.shape[0]
    points_3d = np.zeros((N, 3), dtype=np.float64)
    valid_mask = np.zeros(N, dtype=bool)
    H, W = depth_map_raw.shape
    for i in range(N):
        u = int(np.clip(keypoints_2d[i, 0], 0, W - 1))
        v = int(np.clip(keypoints_2d[i, 1], 0, H - 1))
        d_raw = float(depth_map_raw[v, u])
        if d_raw <= 0:
            continue
        depth_real = CLIP_START + (d_raw / 65535.0) * (CLIP_END - CLIP_START)
        points_3d[i] = [u * GRID_DX, v * GRID_DY, depth_real * GRID_DZ]
        valid_mask[i] = True
    return points_3d, valid_mask
```

- [ ] **Step 2: Write rigid_transform_svd, compute_gt_transform**

```python
def rigid_transform_svd(P_src, P_dst):
    """SVD로 rigid transform 추정: P_dst ≈ R @ P_src + t"""
    c_src = P_src.mean(axis=0)
    c_dst = P_dst.mean(axis=0)
    H = (P_src - c_src).T @ (P_dst - c_dst)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c_dst - R @ c_src
    return R, t


def compute_gt_transform(gt_csv_path, depth0_raw, depth1_raw):
    """GT CSV 비-occluded 대응점으로 T_gt(input->master) SVD 추정.
    Returns: R_gt (3,3), t_gt (3,), n_gt_used (int)
    """
    df = pd.read_csv(gt_csv_path)
    valid = df["occluded"] == False
    master_xy = df.loc[valid, ["master_x", "master_y"]].values.astype(np.float64)
    input_xy = df.loc[valid, ["input_x", "input_y"]].values.astype(np.float64)

    H0, W0 = depth0_raw.shape
    H1, W1 = depth1_raw.shape

    pts_master = []
    pts_input = []
    for i in range(len(master_xy)):
        mu = int(np.clip(master_xy[i, 0], 0, W0 - 1))
        mv = int(np.clip(master_xy[i, 1], 0, H0 - 1))
        iu = int(np.clip(input_xy[i, 0], 0, W1 - 1))
        iv = int(np.clip(input_xy[i, 1], 0, H1 - 1))

        d0 = float(depth0_raw[mv, mu])
        d1 = float(depth1_raw[iv, iu])
        if d0 <= 0 or d1 <= 0:
            continue

        dr0 = CLIP_START + (d0 / 65535.0) * (CLIP_END - CLIP_START)
        dr1 = CLIP_START + (d1 / 65535.0) * (CLIP_END - CLIP_START)
        pts_master.append([mu * GRID_DX, mv * GRID_DY, dr0 * GRID_DZ])
        pts_input.append([iu * GRID_DX, iv * GRID_DY, dr1 * GRID_DZ])

    pts_master = np.array(pts_master)
    pts_input = np.array(pts_input)
    R_gt, t_gt = rigid_transform_svd(pts_input, pts_master)
    return R_gt, t_gt, len(pts_master)
```

- [ ] **Step 3: Write ransac_rigid_o3d, sample_point_cloud, compute_transform_rmse**

```python
def ransac_rigid_o3d(src, dst, inlier_th=5.0, ransac_n=3, max_iter=100000):
    """Open3D RANSAC correspondence 기반 rigid transform.
    src -> dst 변환 추정.
    """
    N = src.shape[0]
    if N < ransac_n:
        return np.eye(3), np.zeros(3), np.zeros(N, dtype=bool), 0

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src)
    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(dst)
    corres = o3d.utility.Vector2iVector(
        np.column_stack([np.arange(N), np.arange(N)])
    )
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        pcd_src, pcd_dst, corres,
        max_correspondence_distance=inlier_th,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=ransac_n,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iter, 0.999),
    )
    T = np.asarray(result.transformation)
    R, t = T[:3, :3], T[:3, 3]
    corres_set = np.asarray(result.correspondence_set)
    inlier_mask = np.zeros(N, dtype=bool)
    if len(corres_set) > 0:
        inlier_mask[corres_set[:, 0]] = True
    return R, t, inlier_mask, int(inlier_mask.sum())


def sample_point_cloud(depth_raw, n_pts=30000):
    """Depth map에서 유효 픽셀 n_pts개 균일 샘플링 -> Grid 3D."""
    vs, us = np.where(depth_raw > 0)
    n_valid = len(vs)
    if n_valid == 0:
        return np.zeros((0, 3), dtype=np.float64)
    idx = np.random.choice(n_valid, min(n_pts, n_valid), replace=False)
    us_s, vs_s = us[idx], vs[idx]
    d_raw = depth_raw[vs_s, us_s].astype(np.float64)
    depth_real = CLIP_START + (d_raw / 65535.0) * (CLIP_END - CLIP_START)
    return np.stack([us_s * GRID_DX, vs_s * GRID_DY, depth_real * GRID_DZ], axis=1)


def compute_transform_rmse(P_src, R_est, t_est, R_gt, t_gt):
    """T_est vs T_gt를 동일 소스 포인트에 적용하여 RMSE 산출."""
    P_est = (R_est @ P_src.T).T + t_est
    P_gt = (R_gt @ P_src.T).T + t_gt
    return float(np.sqrt(np.mean(np.sum((P_est - P_gt) ** 2, axis=1))))
```

- [ ] **Step 4: Write model loading, inference, coordinate conversion, batch swap**

```python
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
    return model(batch), batch


def kpts_to_orig(kpts, image_size):
    """crop+resize 좌표(1751px) -> 원본(5761px)."""
    scale = CROP_SIZE / image_size
    kpts_orig = kpts * scale
    kpts_orig[:, 0] += CROP_X0
    kpts_orig[:, 1] += CROP_Y0
    return kpts_orig


def extract_matches(pred, batch_idx, conf_th=0.0):
    """모델 예측에서 매칭 pair 추출. Returns: mkp0, mkp1 (1751px 공간)."""
    kp0 = pred["keypoints0"][batch_idx].cpu().numpy()
    kp1 = pred["keypoints1"][batch_idx].cpu().numpy()
    m0 = pred["matches0"][batch_idx].cpu().numpy()
    scores = pred["matching_scores0"][batch_idx].cpu().numpy()
    valid = (m0 > -1) & (scores > conf_th)
    return kp0[valid], kp1[m0[valid]], int(valid.sum())


def swap_batch(batch):
    """view0 <-> view1 swap for reverse direction."""
    return {
        "view0": batch["view1"],
        "view1": batch["view0"],
        "gt_matches": batch.get("gt_matches"),
        "csv_path": batch.get("csv_path"),
        "master_path": batch.get("master_path"),
        "input_path": batch.get("input_path"),
    }
```

- [ ] **Step 5: Write overlay visualization**

```python
def save_overlay(depth0_raw, depth1_raw, R_est, t_est, output_path):
    """Master(cyan) + warped input(red) overlay 저장."""
    H, W = depth0_raw.shape
    vs, us = np.where(depth1_raw > 0)
    d_raw = depth1_raw[vs, us].astype(np.float64)
    depth_real = CLIP_START + (d_raw / 65535.0) * (CLIP_END - CLIP_START)
    pts = np.stack([us * GRID_DX, vs * GRID_DY, depth_real * GRID_DZ], axis=1)
    pts_aligned = (R_est @ pts.T).T + t_est

    u0 = np.round(pts_aligned[:, 0] / GRID_DX).astype(np.int32)
    v0 = np.round(pts_aligned[:, 1] / GRID_DY).astype(np.int32)
    z0 = pts_aligned[:, 2]

    in_bounds = (u0 >= 0) & (u0 < W) & (v0 >= 0) & (v0 < H) & (z0 > 0)
    u0, v0, z0 = u0[in_bounds], v0[in_bounds], z0[in_bounds]

    warped = np.zeros((H, W), dtype=np.float64)
    zbuf = np.full((H, W), np.inf)
    for i in range(len(u0)):
        if z0[i] < zbuf[v0[i], u0[i]]:
            zbuf[v0[i], u0[i]] = z0[i]
            warped[v0[i], u0[i]] = z0[i]
    if warped.max() > 0:
        warped = (warped / warped.max()).astype(np.float32)

    d0_vis = depth0_raw.astype(np.float32) / 65535.0
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    overlay[:, :, 1] = d0_vis   # master = cyan (G+B)
    overlay[:, :, 2] = d0_vis
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], warped)  # warped input = red

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(np.clip(overlay, 0, 1))
    ax.set_axis_off()
    ax.set_title("Master(cyan) + Warped Input(red)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 6: Write evaluate_pair — single pair forward+reverse evaluation**

```python
def evaluate_pair(model, dataset, collate_fn, idx, combo_row,
                  base_dir, device, image_size, args):
    """단일 pair 양방향 평가. Returns: result dict."""
    master_fname = Path(combo_row["master_path"]).name
    input_fname = Path(combo_row["input_path"]).name
    depth0_raw = cv2.imread(
        str(base_dir / "dataset_resample_2" / master_fname), cv2.IMREAD_UNCHANGED)
    depth1_raw = cv2.imread(
        str(base_dir / "dataset_resample_2" / input_fname), cv2.IMREAD_UNCHANGED)
    csv_path = base_dir / combo_row["csv_path"]

    # GT transform (input -> master) from CSV SVD
    R_gt, t_gt, n_gt = compute_gt_transform(csv_path, depth0_raw, depth1_raw)
    print(f"  GT SVD: {n_gt} non-occluded pairs used")

    # ─── Forward: (master=view0, input=view1) ───
    sample = dataset[idx]
    batch_fwd = collate_fn([sample])
    pred_fwd, batch_fwd = run_inference(model, batch_fwd, device)

    mkp0_fwd, mkp1_fwd, n_match_fwd = extract_matches(pred_fwd, 0, args.conf_th)
    mkp0_orig = kpts_to_orig(mkp0_fwd, image_size)
    mkp1_orig = kpts_to_orig(mkp1_fwd, image_size)

    pts0, vm0 = pixel_to_grid3d(mkp0_orig, depth0_raw)
    pts1, vm1 = pixel_to_grid3d(mkp1_orig, depth1_raw)
    both = vm0 & vm1
    pts0_v, pts1_v = pts0[both], pts1[both]
    n_3d_fwd = len(pts0_v)

    if n_3d_fwd < 3:
        print(f"  Forward: not enough 3D points ({n_3d_fwd}). Skip.")
        return None

    R_est_fwd, t_est_fwd, _, n_inl_fwd = ransac_rigid_o3d(
        pts1_v, pts0_v, inlier_th=args.inlier_th, max_iter=args.ransac_iter)

    P_input = sample_point_cloud(depth1_raw, args.n_sample_pts)
    rmse_fwd = compute_transform_rmse(P_input, R_est_fwd, t_est_fwd, R_gt, t_gt)
    print(f"  Forward: matches={n_match_fwd}, 3D={n_3d_fwd}, "
          f"inliers={n_inl_fwd}, RMSE={rmse_fwd:.4f}")

    # ─── Reverse: (input=view0, master=view1) ───
    batch_rev = swap_batch(batch_fwd)
    pred_rev, batch_rev = run_inference(model, batch_rev, device)

    mkp0_rev, mkp1_rev, n_match_rev = extract_matches(pred_rev, 0, args.conf_th)
    mkp0_rev_orig = kpts_to_orig(mkp0_rev, image_size)  # input 좌표
    mkp1_rev_orig = kpts_to_orig(mkp1_rev, image_size)  # master 좌표

    pts0_rev, vm0r = pixel_to_grid3d(mkp0_rev_orig, depth1_raw)   # input depth
    pts1_rev, vm1r = pixel_to_grid3d(mkp1_rev_orig, depth0_raw)   # master depth
    both_r = vm0r & vm1r
    pts0_rv, pts1_rv = pts0_rev[both_r], pts1_rev[both_r]
    n_3d_rev = len(pts0_rv)

    if n_3d_rev < 3:
        print(f"  Reverse: not enough 3D points ({n_3d_rev}). Skip.")
        return None

    R_est_rev, t_est_rev, _, n_inl_rev = ransac_rigid_o3d(
        pts1_rv, pts0_rv, inlier_th=args.inlier_th, max_iter=args.ransac_iter)

    # T_gt reverse = inverse(T_gt forward)
    R_gt_rev = R_gt.T
    t_gt_rev = -R_gt.T @ t_gt

    P_master = sample_point_cloud(depth0_raw, args.n_sample_pts)
    rmse_rev = compute_transform_rmse(P_master, R_est_rev, t_est_rev, R_gt_rev, t_gt_rev)
    print(f"  Reverse: matches={n_match_rev}, 3D={n_3d_rev}, "
          f"inliers={n_inl_rev}, RMSE={rmse_rev:.4f}")

    rmse_bi = (rmse_fwd + rmse_rev) / 2.0
    print(f"  Bidirectional RMSE: {rmse_bi:.4f}")

    return {
        "pair_idx": idx,
        "master": master_fname,
        "input": input_fname,
        "rmse_fwd": rmse_fwd,
        "rmse_rev": rmse_rev,
        "rmse_bi": rmse_bi,
        "n_matches_fwd": n_match_fwd,
        "n_matches_rev": n_match_rev,
        "n_inliers_fwd": n_inl_fwd,
        "n_inliers_rev": n_inl_rev,
        "n_gt_pts": n_gt,
        "R_est_fwd": R_est_fwd,
        "t_est_fwd": t_est_fwd,
        "depth0_raw": depth0_raw,
        "depth1_raw": depth1_raw,
    }
```

- [ ] **Step 7: Write main function**

```python
def main():
    parser = argparse.ArgumentParser(
        description="ISS+SHOT+LG Registration Eval (Transform RMSE)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--experiment", type=str,
                        default="0407_resample2_iss_shot352_lg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--image_size", type=int, default=1751)
    parser.add_argument("--shot_radius", type=float, default=10.0)
    parser.add_argument("--conf_th", type=float, default=0.0)
    parser.add_argument("--inlier_th", type=float, default=5.0)
    parser.add_argument("--ransac_iter", type=int, default=100000)
    parser.add_argument("--n_sample_pts", type=int, default=30000)
    args = parser.parse_args()

    output_dir = Path(
        args.output_dir if args.output_dir
        else f"results/registration/{args.experiment}_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.checkpoint is None:
        from gluefactory.utils.experiments import get_best_checkpoint
        cp_path = get_best_checkpoint(args.experiment)
        print(f"Auto-loaded best checkpoint: {cp_path}")
    else:
        cp_path = args.checkpoint

    model, conf = load_model(cp_path, device)
    image_size = args.image_size

    base_dir = Path("gluefactory/datasets/mitsubishi")
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

    dataset = MitsubishiResample2ISSSHOTDataset(
        split=args.split, shot_radius=args.shot_radius, image_size=image_size)
    collate_fn = resample2_iss_shot_collate_fn
    print(f"{args.split} dataset: {len(dataset)} pairs")
    print(f"RANSAC: inlier_th={args.inlier_th}, iter={args.ransac_iter}")
    print(f"Sample pts: {args.n_sample_pts}")

    if args.indices is not None:
        indices = args.indices
    else:
        indices = sorted(np.random.choice(
            len(dataset), min(args.num_samples, len(dataset)), replace=False))

    print(f"Testing {len(indices)} pairs: {indices}")

    results = []
    for idx in indices:
        print(f"\n--- Pair {idx} ---")
        row = combo_split.iloc[idx]
        res = evaluate_pair(
            model, dataset, collate_fn, idx, row,
            base_dir, device, image_size, args)
        if res is None:
            continue

        # Overlay (forward only)
        save_overlay(
            res["depth0_raw"], res["depth1_raw"],
            res["R_est_fwd"], res["t_est_fwd"],
            output_dir / f"overlay_{args.split}_{idx:05d}.png")

        results.append({k: v for k, v in res.items()
                        if k not in ("R_est_fwd", "t_est_fwd",
                                     "depth0_raw", "depth1_raw")})

    # CSV 저장
    if results:
        df = pd.DataFrame(results)
        csv_out = output_dir / f"rmse_{args.split}.csv"
        df.to_csv(csv_out, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved: {csv_out}")
        print(f"Mean Bidirectional RMSE: {df['rmse_bi'].mean():.4f}")
        print(df[["pair_idx", "rmse_fwd", "rmse_rev", "rmse_bi"]].to_string(index=False))

    print(f"\nDone! Output: {output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run lint check**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && python -c "import ast; ast.parse(open('eval_registration_iss_shot.py').read()); print('OK')" `
Expected: `OK`

- [ ] **Step 9: Commit**

```bash
git add eval_registration_iss_shot.py
git commit -m "feat: add ISS+SHOT registration eval with transform-based RMSE"
```

---

### Task 3: eval_registration_iss_fpfh.py

**Files:**
- Create: `eval_registration_iss_fpfh.py`

- [ ] **Step 1: Copy ISS+SHOT version and change imports/defaults**

`eval_registration_iss_shot.py`를 복사한 뒤 다음 부분만 변경:

```python
# 변경 1: docstring
"""
ISS+FPFH+LG Registration 평가 — Transform 기반 RMSE.
...
사용법:
    python eval_registration_iss_fpfh.py --experiment 0402_resample2_iss_fpfh_xyz_lg --indices 0 10 50 90
"""

# 변경 2: import
from gluefactory.datasets.mitsubishi_resample2_iss_fpfh_dataset import (
    MitsubishiResample2ISSFPFHDataset,
    resample2_iss_fpfh_collate_fn,
    CROP_X0,
    CROP_Y0,
    CROP_SIZE,
)

# 변경 3: argparse defaults
parser.add_argument("--experiment", type=str,
                    default="0402_resample2_iss_fpfh_xyz_lg")
# shot_radius → fpfh_radius
parser.add_argument("--fpfh_radius", type=float, default=5.0)

# 변경 4: dataset 생성
dataset = MitsubishiResample2ISSFPFHDataset(
    split=args.split, fpfh_radius=args.fpfh_radius, image_size=image_size)
collate_fn = resample2_iss_fpfh_collate_fn
```

나머지 모든 함수(pixel_to_grid3d, rigid_transform_svd, compute_gt_transform, ransac_rigid_o3d, sample_point_cloud, compute_transform_rmse, extract_matches, kpts_to_orig, swap_batch, save_overlay, evaluate_pair, main)는 동일.

- [ ] **Step 2: Run lint check**

Run: `cd /home/jhs/work/Registration/glue-factory_depth && python -c "import ast; ast.parse(open('eval_registration_iss_fpfh.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add eval_registration_iss_fpfh.py
git commit -m "feat: add ISS+FPFH registration eval with transform-based RMSE"
```

---

### Task 4: Integration verification

- [ ] **Step 1: Run ISS+SHOT eval on 1-2 test pairs**

```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda activate LightGlue
python eval_registration_iss_shot.py --experiment 0407_resample2_iss_shot352_lg --indices 0 --split test
```

Expected: 터미널에 forward/reverse/bidirectional RMSE 출력, `results/registration/` 하위에 overlay PNG + CSV 생성

- [ ] **Step 2: Run ISS+FPFH eval on same pairs**

```bash
python eval_registration_iss_fpfh.py --experiment 0402_resample2_iss_fpfh_xyz_lg --indices 0 --split test
```

Expected: 동일 형식 결과 출력

- [ ] **Step 3: Verify output CSV**

```bash
cat results/registration/*_eval/rmse_test.csv
```

Expected: pair_idx, master, input, rmse_fwd, rmse_rev, rmse_bi, n_matches_fwd/rev, n_inliers_fwd/rev 컬럼 확인

- [ ] **Step 4: Commit integration result**

```bash
git add -A results/  # or just confirm clean state
git commit -m "test: verify registration eval integration on test pairs"
```
