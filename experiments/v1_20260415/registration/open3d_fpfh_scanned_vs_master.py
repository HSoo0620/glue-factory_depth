"""Open3D 순수 RANSAC FPFH baseline: scanned(roi13) vs master(blender_master1).

학습 모델 없이 Open3D 내장 feature-matching RANSAC 만으로 정합.
결과: fitness/inlier_rmse, matplotlib 3-view PNG (before/after, overlay),
      result.json (파라미터 + metric + elapsed).

Usage:
    conda activate LightGlue
    cd /home/jhs/work/Registration/glue-factory_depth
    python experiments/v1_20260415/registration/open3d_fpfh_scanned_vs_master.py --param_mode v1
    python experiments/v1_20260415/registration/open3d_fpfh_scanned_vs_master.py --param_mode tutorial
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[3]
V1_DIR = ROOT / "experiments/v1_20260415"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V1_DIR))

from infer_scanned_vs_master_shot352 import (  # noqa: E402
    mask_scanned_table,
    apply_bilateral,
    zmap_to_pcd_mm,
    _sample_pcd_mm,
)

SCAN_PATH = ROOT / "gluefactory/datasets/scanned/roi13_zmap 1.png"
MASTER_PATH = ROOT / "gluefactory/datasets/scanned/blender_master1/master.png"

N_SAMPLE_PTS = 15000

PARAM_PRESETS = {
    "v1":       dict(voxel=1.0, normal_radius=20.0, fpfh_radius=20.0),
    "tutorial": dict(voxel=5.0, normal_radius=10.0, fpfh_radius=25.0),
}


def preprocess_point_cloud(pts_mm: np.ndarray, voxel: float, normal_r: float,
                           fpfh_r: float):
    """mm PCD → (downsampled PCD, FPFH feature).

    Open3D KDTreeSearchParamHybrid: 반지름 + max_nn 제한.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_mm.astype(np.float64))
    pcd_down = pcd.voxel_down_sample(voxel)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_r, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_r, max_nn=100))
    return pcd_down, fpfh


def execute_global_registration(src_down, dst_down, src_fpfh, dst_fpfh,
                                distance_threshold: float):
    """RANSAC feature-matching 전역 정합.

    open3d_fpfh_func.py 기준 파라미터 그대로:
    - PointToPoint (scale=False)
    - n_ransac=3
    - CorrespondenceCheckerBasedOnEdgeLength(0.9) + Distance(distance_th)
    - convergence: (max_iter=100000, confidence=0.999)
    """
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, dst_down, src_fpfh, dst_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )


def resolve_params(args) -> dict:
    """param_mode preset + 개별 override → 최종 파라미터 dict.

    distance_threshold 규칙: --distance_threshold 가 주어지면 그 값을 그대로 사용.
    주어지지 않으면 (override 반영 후) voxel * 1.5 로 재계산.
    """
    preset = PARAM_PRESETS[args.param_mode]
    voxel = args.voxel if args.voxel is not None else preset["voxel"]
    normal_r = args.normal_radius if args.normal_radius is not None else preset["normal_radius"]
    fpfh_r = args.fpfh_radius if args.fpfh_radius is not None else preset["fpfh_radius"]
    if args.distance_threshold is not None:
        distance_th = float(args.distance_threshold)
    else:
        distance_th = float(voxel) * 1.5
    return dict(
        voxel=float(voxel),
        normal_radius=float(normal_r),
        fpfh_radius=float(fpfh_r),
        distance_threshold=distance_th,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--param_mode", type=str, default="v1",
                   choices=list(PARAM_PRESETS.keys()))
    p.add_argument("--voxel", type=float, default=None)
    p.add_argument("--normal_radius", type=float, default=None)
    p.add_argument("--fpfh_radius", type=float, default=None)
    p.add_argument("--distance_threshold", type=float, default=None)
    p.add_argument("--master", type=str, default=None,
                   help="Master zmap 경로 (기본: blender_master1/master.png)")
    p.add_argument("--no_rotate", action="store_true",
                   help="Master 180도 회전 비활성화")
    p.add_argument("--output_dir", type=str, default=None,
                   help="결과 저장 폴더 (기본: 자동 생성)")
    p.add_argument("--seed", type=int, default=42,
                   help="PCD sampling seed")
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    params = resolve_params(args)
    print(f"params: {params}")
    # TODO(Task 4): 본격적인 실행 플로우


if __name__ == "__main__":
    main()
