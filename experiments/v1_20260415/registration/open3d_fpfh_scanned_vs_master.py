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


def _plot_registration(pc_dst, pc_src, pc_est,
                       title_line: str, output_path: Path,
                       dst_label="master (synthetic)",
                       src_label="input (scan)"):
    """2×3 grid: before (dst vs src) / after (dst vs est) × top/front/side.

    title_line: suptitle 한 줄 (예: "Open3D FPFH v1 | fitness=0.45 | ...").
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    s, alpha = 0.5, 0.6
    labels_row = ["Before registration", "After (estimated R,t)"]
    labels_col = ["Top-down (X, Y)", "Front (X, Z)", "Side (Y, Z)"]
    col_axes = [(0, 1), (0, 2), (1, 2)]
    C_M, C_I = "lightskyblue", "crimson"
    pairs = [(pc_dst, pc_src), (pc_dst, pc_est)]

    for row, (pa, pb) in enumerate(pairs):
        for col, (xi, yi) in enumerate(col_axes):
            ax = axes[row, col]
            ax.scatter(pa[:, xi], pa[:, yi], s=s, c=C_M, alpha=alpha,
                       label=dst_label)
            ax.scatter(pb[:, xi], pb[:, yi], s=s, c=C_I, alpha=alpha,
                       label=src_label)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(labels_col[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(labels_row[row], fontsize=11, fontweight="bold")
    for row in range(2):
        for col in range(3):
            axes[row, col].invert_yaxis()

    fig.suptitle(title_line, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def _plot_overlay(pc_dst, pc_est, title_line: str, output_path: Path,
                  dst_label="master (synthetic)",
                  src_label="aligned scan"):
    """1×3 overlay: aligned-est 와 master 겹쳐 그리기."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    s, alpha = 0.5, 0.6
    cols = [(0, 1), (0, 2), (1, 2)]
    names = ["Top-down (X, Y)", "Front (X, Z)", "Side (Y, Z)"]
    for col, (xi, yi) in enumerate(cols):
        ax = axes[col]
        ax.scatter(pc_dst[:, xi], pc_dst[:, yi], s=s, c="lightskyblue",
                   alpha=alpha, label=dst_label)
        ax.scatter(pc_est[:, xi], pc_est[:, yi], s=s, c="crimson",
                   alpha=alpha, label=src_label)
        ax.set_aspect("equal")
        ax.set_title(names[col], fontsize=11)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.legend(markerscale=5, fontsize=9)
    for ax in axes:
        ax.invert_yaxis()
    fig.suptitle(title_line, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


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

    master_path = Path(args.master) if args.master else MASTER_PATH
    rotate_master = not args.no_rotate

    if args.output_dir:
        out_dir = ROOT / args.output_dir
    else:
        out_dir = (V1_DIR / "results"
                   / f"scanned_vs_open3d_fpfh_{args.param_mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"[1] 입력 로드 (param_mode={args.param_mode}, params={params})")
    import cv2
    scan_raw = cv2.imread(str(SCAN_PATH), cv2.IMREAD_UNCHANGED)
    master_raw = cv2.imread(str(master_path), cv2.IMREAD_UNCHANGED)
    if scan_raw is None:
        raise FileNotFoundError(SCAN_PATH)
    if master_raw is None:
        raise FileNotFoundError(master_path)
    if rotate_master:
        master_raw = cv2.rotate(master_raw, cv2.ROTATE_180)
    print(f"  scan:   {scan_raw.shape}")
    print(f"  master: {master_raw.shape}  "
          f"({'rotated 180°' if rotate_master else 'no rotate'})  "
          f"[{master_path.name}]")

    print("\n[2] Preprocessing scanned (floor mask + bilateral)")
    scan_masked, info = mask_scanned_table(scan_raw)
    if info.get("applied"):
        print(f"  floor mask: peak={info['peak_center']}  "
              f"band=[{info['band_low']}, {info['band_high']}]  "
              f"masked={info['fraction_masked'] * 100:.1f}%")
    scan_bilat = apply_bilateral(scan_masked)

    print("\n[3] zmap → mm PCD (erode=5)")
    src_pts_mm = zmap_to_pcd_mm(scan_bilat)
    dst_pts_mm = zmap_to_pcd_mm(master_raw)
    print(f"  scan:   {len(src_pts_mm)} pts")
    print(f"  master: {len(dst_pts_mm)} pts")

    print("\n[4] Open3D preprocess + RANSAC FPFH (timed)")
    t0 = time.perf_counter()
    src_down, src_fpfh = preprocess_point_cloud(
        src_pts_mm, params["voxel"], params["normal_radius"], params["fpfh_radius"])
    dst_down, dst_fpfh = preprocess_point_cloud(
        dst_pts_mm, params["voxel"], params["normal_radius"], params["fpfh_radius"])
    n_src_down = len(src_down.points)
    n_dst_down = len(dst_down.points)
    print(f"  downsample: src={n_src_down} pts, dst={n_dst_down} pts "
          f"(voxel={params['voxel']}mm)")

    result = execute_global_registration(
        src_down, dst_down, src_fpfh, dst_fpfh, params["distance_threshold"])
    elapsed_s = time.perf_counter() - t0

    fitness = float(result.fitness)
    n_corr = int(len(result.correspondence_set))
    T = np.asarray(result.transformation, dtype=np.float64)
    R_est = T[:3, :3]
    t_est = T[:3, 3]
    print(f"  fitness={fitness:.4f}  n_corr={n_corr}  elapsed={elapsed_s:.2f}s")

    print("\n[5] Visualization (15k sample per cloud)")
    pc_src = _sample_pcd_mm(scan_bilat, N_SAMPLE_PTS, seed=args.seed)
    pc_dst = _sample_pcd_mm(master_raw, N_SAMPLE_PTS, seed=args.seed)
    pc_est = (R_est @ pc_src.T).T + t_est

    zf = np.array([1.0, 1.0, -1.0])
    title_reg = (f"Open3D FPFH ({args.param_mode})  |  "
                 f"voxel={params['voxel']:.2g} nr={params['normal_radius']:.2g} "
                 f"fr={params['fpfh_radius']:.2g} dth={params['distance_threshold']:.2g}  |  "
                 f"fitness={fitness:.3f} corr={n_corr}  "
                 f"elapsed={elapsed_s:.1f}s")
    _plot_registration(pc_dst * zf, pc_src * zf, pc_est * zf,
                       title_reg, out_dir / "reg_open3d_fpfh.png",
                       dst_label="master (synthetic)",
                       src_label="input (scan)")
    _plot_overlay(pc_dst * zf, pc_est * zf,
                  f"Overlay  |  {title_reg.split('|', 1)[1].strip()}",
                  out_dir / "overlay_open3d_fpfh.png",
                  dst_label="master (synthetic)",
                  src_label="aligned scan")

    print("\n[6] Save result.json")
    result_json = {
        "param_mode": args.param_mode,
        "voxel": params["voxel"],
        "normal_radius": params["normal_radius"],
        "fpfh_radius": params["fpfh_radius"],
        "distance_threshold": params["distance_threshold"],
        "fitness": fitness,
        "n_correspondences": n_corr,
        "n_src_down": n_src_down,
        "n_dst_down": n_dst_down,
        "elapsed_s": elapsed_s,
        "master_path": str(master_path),
        "scan_path": str(SCAN_PATH),
        "rotate_master": rotate_master,
        "transformation": T.tolist(),
    }
    json_path = out_dir / "result.json"
    with open(json_path, "w") as f:
        json.dump(result_json, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\nDone! → {out_dir}/")


if __name__ == "__main__":
    main()
