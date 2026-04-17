# Open3D FPFH Scanned vs Master Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** scanned(roi13) vs master(blender_master1) 쌍에 Open3D 순수 RANSAC FPFH global registration을 돌려 fitness/inlier_rmse, matplotlib 3-view PNG 2장, total elapsed time을 얻는 단일 스크립트를 만든다.

**Architecture:** `experiments/v1_20260415/registration/` 하위에 단일 스크립트 추가. 전처리(`mask_scanned_table`, `apply_bilateral`)와 zmap→mm PCD 변환(`zmap_to_pcd_mm`)은 기존 `infer_scanned_vs_master_shot352.py`에서 import 재사용. 시각화 함수(`_plot_registration`, `_plot_overlay`)는 해당 파일에 "SHOT352 dim352"가 하드코딩돼 있으므로 **로컬에 복사**하고 `title_line` 파라미터로 일반화 — SHOT 스크립트는 건드리지 않음 (spec 섹션 "컴포넌트 4" 보완).

**Tech Stack:** Python 3.10, Open3D, OpenCV, numpy, matplotlib, argparse. conda env `LightGlue`.

**Spec reference:** `docs/superpowers/specs/2026-04-17-open3d-fpfh-scanned-vs-master-design.md`

---

## File Structure

**Create:**
- `experiments/v1_20260415/registration/open3d_fpfh_scanned_vs_master.py` — 단일 실행 스크립트 (~280 LOC)
- `tests/test_open3d_fpfh_baseline.py` — `resolve_params` 유닛 테스트

**Read-only (import source):**
- `experiments/v1_20260415/infer_scanned_vs_master_shot352.py` — 전처리·zmap→PCD 함수 재사용 (수정 없음)

**Output (실행 시 생성):**
- `experiments/v1_20260415/results/scanned_vs_open3d_fpfh_{param_mode}/`
  - `reg_open3d_fpfh.png`
  - `overlay_open3d_fpfh.png`
  - `result.json`

---

## Task 1: 스크립트 스켈레톤 + CLI + param 해석 + 유닛 테스트

**Files:**
- Create: `experiments/v1_20260415/registration/open3d_fpfh_scanned_vs_master.py`
- Create: `tests/test_open3d_fpfh_baseline.py`

- [ ] **Step 1: 테스트 작성 (실패 상태)**

`tests/test_open3d_fpfh_baseline.py`:

```python
"""Open3D FPFH baseline 스크립트: param 해석 유닛 테스트."""
import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments/v1_20260415/registration"))

import open3d_fpfh_scanned_vs_master as mod  # noqa: E402


def _ns(**kw):
    default = dict(
        param_mode="v1",
        voxel=None,
        normal_radius=None,
        fpfh_radius=None,
        distance_threshold=None,
    )
    default.update(kw)
    return argparse.Namespace(**default)


def test_resolve_params_v1_defaults():
    p = mod.resolve_params(_ns(param_mode="v1"))
    assert p["voxel"] == 1.0
    assert p["normal_radius"] == 20.0
    assert p["fpfh_radius"] == 20.0
    assert p["distance_threshold"] == pytest.approx(1.5)


def test_resolve_params_tutorial_defaults():
    p = mod.resolve_params(_ns(param_mode="tutorial"))
    assert p["voxel"] == 5.0
    assert p["normal_radius"] == 10.0
    assert p["fpfh_radius"] == 25.0
    assert p["distance_threshold"] == pytest.approx(7.5)


def test_resolve_params_override_voxel_recomputes_distance():
    p = mod.resolve_params(_ns(param_mode="v1", voxel=2.0))
    assert p["voxel"] == 2.0
    assert p["normal_radius"] == 20.0
    assert p["fpfh_radius"] == 20.0
    assert p["distance_threshold"] == pytest.approx(3.0)


def test_resolve_params_override_distance_threshold_explicit():
    p = mod.resolve_params(_ns(param_mode="v1", distance_threshold=4.0))
    assert p["voxel"] == 1.0
    assert p["distance_threshold"] == 4.0


def test_resolve_params_override_all_individual():
    p = mod.resolve_params(_ns(
        param_mode="tutorial",
        voxel=3.0, normal_radius=7.0, fpfh_radius=15.0, distance_threshold=2.5,
    ))
    assert p["voxel"] == 3.0
    assert p["normal_radius"] == 7.0
    assert p["fpfh_radius"] == 15.0
    assert p["distance_threshold"] == 2.5
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda activate LightGlue
pytest tests/test_open3d_fpfh_baseline.py -v
```

Expected: ModuleNotFoundError (`open3d_fpfh_scanned_vs_master` 모듈 없음).

- [ ] **Step 3: 스크립트 스켈레톤 작성**

`experiments/v1_20260415/registration/open3d_fpfh_scanned_vs_master.py`:

```python
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

# SHOT 스크립트에서 전처리·zmap→PCD 함수 재사용 (SHOT 모델 로드 없이 모듈 최상위 부수효과 없음 확인 완료)
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
```

- [ ] **Step 4: 테스트 재실행 (통과 확인)**

```bash
pytest tests/test_open3d_fpfh_baseline.py -v
```

Expected: 5 tests passed.

- [ ] **Step 5: 커밋**

```bash
git add experiments/v1_20260415/registration/open3d_fpfh_scanned_vs_master.py \
        tests/test_open3d_fpfh_baseline.py
git commit -m "$(cat <<'EOF'
feat(v1): Open3D FPFH baseline 스크립트 스켈레톤 + param 해석

param_mode {v1, tutorial} preset + 개별 override 로직. 유닛 테스트 5개.
Task 1 of plan 2026-04-17-open3d-fpfh-scanned-vs-master.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Open3D 핵심 함수 (preprocess + RANSAC)

**Files:**
- Modify: `experiments/v1_20260415/registration/open3d_fpfh_scanned_vs_master.py`

이 두 함수는 Open3D thin wrapper라 pytest는 비용 대비 효용 낮음. 대신 Task 5에서 end-to-end 실행으로 확인.

- [ ] **Step 1: `preprocess_point_cloud` 추가**

Task 1의 스크립트 상단(constants 아래, `resolve_params` 위)에 추가:

```python
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
```

- [ ] **Step 2: `execute_global_registration` 추가**

`preprocess_point_cloud` 아래에 추가:

```python
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
```

- [ ] **Step 3: 스모크 import 테스트**

```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda activate LightGlue
python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'experiments/v1_20260415/registration')
import open3d_fpfh_scanned_vs_master as m
print('preprocess_point_cloud:', m.preprocess_point_cloud)
print('execute_global_registration:', m.execute_global_registration)
"
```

Expected: 두 함수 객체 출력, import 에러 없음.

- [ ] **Step 4: 커밋**

```bash
git add experiments/v1_20260415/registration/open3d_fpfh_scanned_vs_master.py
git commit -m "$(cat <<'EOF'
feat(v1): Open3D preprocess_point_cloud + RANSAC global reg 함수

open3d_fpfh_func.py 의 튜토리얼 로직을 파라미터 주입 가능한 형태로 이식.
Task 2 of plan 2026-04-17-open3d-fpfh-scanned-vs-master.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: 로컬 시각화 헬퍼 (title 커스터마이즈)

**Files:**
- Modify: `experiments/v1_20260415/registration/open3d_fpfh_scanned_vs_master.py`

SHOT 스크립트의 `_plot_registration` / `_plot_overlay`는 제목이 "SHOT352 dim352"로 하드코딩 → 그대로 쓰면 baseline 출력 오해. 로컬에 `title_line` 파라미터를 받는 버전으로 복사.

- [ ] **Step 1: `_plot_registration` 로컬 버전 추가**

Task 2에서 추가한 `execute_global_registration` 아래에 추가:

```python
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
```

- [ ] **Step 2: `_plot_overlay` 로컬 버전 추가**

바로 아래에 추가:

```python
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
```

- [ ] **Step 3: 기존 유닛 테스트 + import 스모크 재실행 (회귀 없는지 확인)**

```bash
pytest tests/test_open3d_fpfh_baseline.py -v
python -c "
import sys; sys.path.insert(0, 'experiments/v1_20260415/registration')
import open3d_fpfh_scanned_vs_master as m
assert callable(m._plot_registration); assert callable(m._plot_overlay)
print('ok')
"
```

Expected: 테스트 5 passed, `ok` 출력.

- [ ] **Step 4: 커밋**

```bash
git add experiments/v1_20260415/registration/open3d_fpfh_scanned_vs_master.py
git commit -m "$(cat <<'EOF'
feat(v1): Open3D FPFH baseline — 로컬 시각화 헬퍼 (title 커스텀)

SHOT 스크립트의 plot 함수는 "SHOT352 dim352" 제목이 하드코딩돼 있어
baseline 실행 시 오해 소지. title_line 파라미터를 받는 로컬 버전으로 복제.
Task 3 of plan 2026-04-17-open3d-fpfh-scanned-vs-master.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: 메인 플로우 통합

**Files:**
- Modify: `experiments/v1_20260415/registration/open3d_fpfh_scanned_vs_master.py`

Task 1의 `main()` TODO 자리에 실제 플로우 구현. 타이머는 Open3D 연산(preprocess + RANSAC)만 포함 — 전처리(floor mask + bilateral)와 zmap→PCD 변환은 타이머 외부.

- [ ] **Step 1: `main()` 완전 구현으로 교체**

Task 1에서 작성한 `main()` 함수 전체를 아래로 교체 (`TODO(Task 4)` 부분 포함 전체):

```python
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
    import cv2  # 지역 import — SHOT 스크립트에서도 cv2 사용 중
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
    inlier_rmse = float(result.inlier_rmse)
    n_corr = int(len(result.correspondence_set))
    T = np.asarray(result.transformation, dtype=np.float64)
    R_est = T[:3, :3]
    t_est = T[:3, 3]
    print(f"  fitness={fitness:.4f}  inlier_rmse={inlier_rmse:.3f} mm  "
          f"n_corr={n_corr}  elapsed={elapsed_s:.2f}s")

    print("\n[5] Visualization (15k sample per cloud)")
    pc_src = _sample_pcd_mm(scan_bilat, N_SAMPLE_PTS, seed=args.seed)
    pc_dst = _sample_pcd_mm(master_raw, N_SAMPLE_PTS, seed=args.seed)
    pc_est = (R_est @ pc_src.T).T + t_est

    zf = np.array([1.0, 1.0, -1.0])  # SHOT 스크립트와 동일 — Z 반전(시각화용)
    title_reg = (f"Open3D FPFH ({args.param_mode})  |  "
                 f"voxel={params['voxel']:.2g} nr={params['normal_radius']:.2g} "
                 f"fr={params['fpfh_radius']:.2g} dth={params['distance_threshold']:.2g}  |  "
                 f"fitness={fitness:.3f} rmse={inlier_rmse:.2f}mm corr={n_corr}  "
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
        "inlier_rmse": inlier_rmse,
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
```

- [ ] **Step 2: 기존 유닛 테스트 재확인 (`main` 변경이 resolve_params 테스트를 깨면 안 됨)**

```bash
pytest tests/test_open3d_fpfh_baseline.py -v
```

Expected: 5 tests passed.

- [ ] **Step 3: 커밋**

```bash
git add experiments/v1_20260415/registration/open3d_fpfh_scanned_vs_master.py
git commit -m "$(cat <<'EOF'
feat(v1): Open3D FPFH baseline — 메인 플로우 통합

load → preprocess scanned → zmap→mm PCD → Open3D preprocess + RANSAC (timed)
→ 15k sample → 3-view PNG × 2 → result.json. Timer 범위는 Open3D 단계만.
Task 4 of plan 2026-04-17-open3d-fpfh-scanned-vs-master.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: End-to-end 실행 및 검증

**Files:**
- Modify: 없음 (실행만)

스크립트 완성 여부를 실제 데이터로 검증. 두 모드 모두 실행 → 출력 파일 존재 및 JSON 키 완비 확인.

- [ ] **Step 1: `--param_mode v1` 실행**

```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda activate LightGlue
python experiments/v1_20260415/registration/open3d_fpfh_scanned_vs_master.py --param_mode v1
```

Expected stdout 말미: `Done! → .../scanned_vs_open3d_fpfh_v1/`
그리고 `fitness`, `inlier_rmse`, `n_corr`, `elapsed=XX.XXs` 가 출력됐는지 눈으로 확인.

- [ ] **Step 2: 출력 파일 3개 존재 확인**

```bash
ls -la experiments/v1_20260415/results/scanned_vs_open3d_fpfh_v1/
```

Expected: `reg_open3d_fpfh.png`, `overlay_open3d_fpfh.png`, `result.json` 3개 파일.

- [ ] **Step 3: result.json 스키마 검증**

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("experiments/v1_20260415/results/scanned_vs_open3d_fpfh_v1/result.json")
d = json.loads(p.read_text())
required = {"param_mode","voxel","normal_radius","fpfh_radius","distance_threshold",
            "fitness","inlier_rmse","n_correspondences","n_src_down","n_dst_down",
            "elapsed_s","master_path","scan_path","rotate_master","transformation"}
missing = required - d.keys()
assert not missing, f"missing keys: {missing}"
assert d["param_mode"] == "v1"
assert d["voxel"] == 1.0 and d["normal_radius"] == 20.0 and d["fpfh_radius"] == 20.0
assert len(d["transformation"]) == 4 and len(d["transformation"][0]) == 4
assert d["elapsed_s"] > 0
print("OK: v1 result.json schema valid.")
print(f"  fitness={d['fitness']:.4f}  inlier_rmse={d['inlier_rmse']:.3f}  "
      f"n_corr={d['n_correspondences']}  elapsed={d['elapsed_s']:.2f}s")
PY
```

Expected: `OK: v1 result.json schema valid.` + 한 줄 metric 요약.

- [ ] **Step 4: `--param_mode tutorial` 실행 + 동일 검증**

```bash
python experiments/v1_20260415/registration/open3d_fpfh_scanned_vs_master.py --param_mode tutorial
ls -la experiments/v1_20260415/results/scanned_vs_open3d_fpfh_tutorial/
python - <<'PY'
import json
from pathlib import Path
p = Path("experiments/v1_20260415/results/scanned_vs_open3d_fpfh_tutorial/result.json")
d = json.loads(p.read_text())
assert d["param_mode"] == "tutorial"
assert d["voxel"] == 5.0 and d["normal_radius"] == 10.0 and d["fpfh_radius"] == 25.0
assert d["distance_threshold"] == 7.5
print("OK: tutorial result.json schema valid.")
print(f"  fitness={d['fitness']:.4f}  inlier_rmse={d['inlier_rmse']:.3f}  "
      f"n_corr={d['n_correspondences']}  elapsed={d['elapsed_s']:.2f}s")
PY
```

Expected: `OK: tutorial result.json schema valid.` + metric 요약.

- [ ] **Step 5: PNG 육안 확인 (사용자에게 리포트)**

사용자에게 두 PNG 파일 경로를 보고하고 시각적 품질 확인 요청:

- `experiments/v1_20260415/results/scanned_vs_open3d_fpfh_v1/reg_open3d_fpfh.png`
- `experiments/v1_20260415/results/scanned_vs_open3d_fpfh_v1/overlay_open3d_fpfh.png`
- `experiments/v1_20260415/results/scanned_vs_open3d_fpfh_tutorial/reg_open3d_fpfh.png`
- `experiments/v1_20260415/results/scanned_vs_open3d_fpfh_tutorial/overlay_open3d_fpfh.png`

두 모드의 fitness/inlier_rmse/elapsed 비교 표를 stdout 요약으로 함께 보고.

- [ ] **Step 6: 커밋 (결과 파일은 커밋하지 않음 — results/는 재현 가능)**

결과 PNG/JSON은 results/ 하위라 git 추적 대상이 아닌지 확인. 추가된 코드 파일은 Task 4까지 이미 커밋됨. 이 태스크는 **검증만** 하므로 별도 커밋 없음.

```bash
git status
```

Expected: `nothing to commit, working tree clean` (또는 results/ 하위 untracked만).

---

## Self-Review (완료)

**Spec coverage**:
- 목표 1 (fitness/inlier_rmse) — Task 4 Step 1 (stdout) + Task 5 Step 3 (result.json) ✓
- 목표 2 (matplotlib 3-view PNG) — Task 3 + Task 4 Step 1 ✓
- 목표 3 (total elapsed) — Task 4 Step 1 (Open3D 구간 타이머) ✓
- Non-goal (GT RMSE 없음, ICP 없음, 단계별 타이밍 없음) — 플랜에 해당 작업 없음 ✓
- 범위 (master 180° 회전, scanned 전처리) — Task 4 Step 1 ✓
- 아키텍처 컴포넌트 1~5 — Task 1 + Task 2 + Task 3 + Task 4 ✓
- CLI 플래그 전부 — Task 1 Step 3 `build_arg_parser` ✓
- 에러 처리 (FileNotFoundError, degenerate fitness=0 케이스) — Task 4 Step 1 `raise FileNotFoundError` + 시각화는 identity transform을 자연스럽게 처리 (T=eye 일 때 pc_est == pc_src, 그대로 시각화됨) ✓

**Placeholder scan**: TODO/TBD 없음. 모든 step에 구체 코드/명령/기대 출력 포함.

**Type consistency**:
- `resolve_params` 반환 dict 키: `voxel`, `normal_radius`, `fpfh_radius`, `distance_threshold` — 테스트와 main 사용처 일치 ✓
- `preprocess_point_cloud` 반환 (pcd_down, fpfh) — main에서 언팩 일치 ✓
- `execute_global_registration` 반환 `result` (Open3D `RegistrationResult`) — main에서 `.fitness`, `.inlier_rmse`, `.correspondence_set`, `.transformation` 접근 일치 ✓
- `_plot_registration` / `_plot_overlay` signature — main 호출 시 인자 수/타입 일치 ✓
- SHOT 스크립트에서 import 하는 `mask_scanned_table`, `apply_bilateral`, `zmap_to_pcd_mm`, `_sample_pcd_mm` — 실제 정의 확인 완료 ✓
