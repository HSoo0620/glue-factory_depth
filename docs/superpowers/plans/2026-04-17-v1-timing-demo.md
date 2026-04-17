# v1_20260415 Scanned×2 Inference Timing Demo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 2장의 scanned zmap → matching → R,t end-to-end (cold-start) 시간을 stage 별로 측정하는 CLI demo 2개 작성 (SHOT352, FPFH33 `_norm`).

**Architecture:** 공용 `StageRecord` timer helper 하나 + descriptor 별 demo 얇은 wrapper 2개. 기존 `infer_scanned_vs_master_shot352.py` 와 `precompute/helpers.py` 함수들은 import 로 재사용. 기존 파일 수정 없음.

**Tech Stack:** Python 3, `time.perf_counter()`, numpy, torch (CUDA sync), PIL, open3d (FPFH), pybind11 SHOT extension (SHOT), LightGlue matcher.

**Spec:** `docs/superpowers/specs/2026-04-17-v1-timing-demo-design.md`

---

## File Structure

| 파일 | 역할 | 라인 수 예상 |
|---|---|---|
| `experiments/v1_20260415/profile/_timer.py` | `StageRecord` (timer context + print_table) | ~45 |
| `experiments/v1_20260415/profile/timing_demo_shot.py` | SHOT352 demo entry | ~130 |
| `experiments/v1_20260415/profile/timing_demo_fpfh.py` | FPFH33 (_norm) demo entry | ~160 |
| `tests/test_v1_profile_timer.py` | StageRecord 단위 테스트 | ~55 |

**수정 파일:** 없음.

---

### Task 1: Create `_timer.py` with TDD

**Files:**
- Test: `tests/test_v1_profile_timer.py`
- Create: `experiments/v1_20260415/profile/_timer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_v1_profile_timer.py`:

```python
import contextlib
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/v1_20260415/profile"))

from _timer import StageRecord  # noqa: E402


def test_timer_records_single_block():
    rec = StageRecord()
    with rec.timer("block_a"):
        time.sleep(0.02)
    assert len(rec.records) == 1
    name, elapsed = rec.records[0]
    assert name == "block_a"
    assert 0.01 <= elapsed <= 1.0


def test_timer_records_multiple_blocks_in_order():
    rec = StageRecord()
    with rec.timer("a"):
        time.sleep(0.005)
    with rec.timer("b"):
        time.sleep(0.005)
    names = [n for n, _ in rec.records]
    assert names == ["a", "b"]


def test_print_table_4stage_summary():
    rec = StageRecord()
    rec.records = [
        ("load", 0.1),
        ("preprocess", 0.2),
        ("descriptor", 0.5),
        ("matching", 0.05),
        ("registration", 0.1),
    ]
    groups = {
        "Pre":   ["load", "preprocess"],
        "Desc":  ["descriptor"],
        "Match": ["matching"],
        "Reg":   ["registration"],
    }
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rec.print_table(groups, detailed=False)
    out = buf.getvalue()
    assert "Pre" in out and "Desc" in out
    assert "Match" in out and "Reg" in out
    assert "Total" in out
    assert "0.300" in out       # Pre sum
    assert "0.950" in out       # Total


def test_print_table_detailed_includes_individual():
    rec = StageRecord()
    rec.records = [("load", 0.123), ("descriptor", 0.456)]
    groups = {"Pre": ["load"], "Desc": ["descriptor"]}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rec.print_table(groups, detailed=True)
    out = buf.getvalue()
    assert "load" in out and "descriptor" in out
    assert "0.123" in out and "0.456" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda run -n LightGlue python -m pytest tests/test_v1_profile_timer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named '_timer'`.

- [ ] **Step 3: Implement `_timer.py`**

Create `experiments/v1_20260415/profile/_timer.py`:

```python
"""Stage-level timing utility for v1_20260415 inference demo."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class StageRecord:
    def __init__(self) -> None:
        self.records: list[tuple[str, float]] = []

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.records.append((name, time.perf_counter() - t0))

    def _sum_by_names(self, names: list[str]) -> float:
        bucket: dict[str, float] = {}
        for n, t in self.records:
            bucket[n] = bucket.get(n, 0.0) + t
        return sum(bucket.get(n, 0.0) for n in names)

    def print_table(self, groups: dict[str, list[str]],
                    detailed: bool = False) -> None:
        total = sum(t for _, t in self.records)
        if detailed:
            for name, t in self.records:
                print(f"[{name:<18}]  {t:>7.3f}s")
            print("─" * 30)
        for gname, members in groups.items():
            s = self._sum_by_names(members)
            print(f"[{gname:<12}]  {s:>7.3f}s")
        print("─" * 30)
        print(f"[{'Total':<12}]  {total:>7.3f}s")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda run -n LightGlue python -m pytest tests/test_v1_profile_timer.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Hand-off to user for commit**

Do not run git commands. User commits manually when satisfied with the task.

---

### Task 2: Create `timing_demo_shot.py`

**Files:**
- Create: `experiments/v1_20260415/profile/timing_demo_shot.py`

- [ ] **Step 1: Write full demo file**

Create `experiments/v1_20260415/profile/timing_demo_shot.py`:

```python
"""v1_20260415 SHOT352 scanned×2 inference timing demo.

실제 3D line scanner 배포 시나리오: 2장의 scanned zmap → matching → R,t.
Cold-start 1회 실행. Stage 별 시간을 stdout 에 4-stage 요약 또는 9-stage 세부로 출력.

Usage:
    conda activate LightGlue
    cd /home/jhs/work/Registration/glue-factory_depth
    python experiments/v1_20260415/profile/timing_demo_shot.py
    python experiments/v1_20260415/profile/timing_demo_shot.py --detailed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pybind_shot_linux"))
sys.path.insert(0, str(ROOT / "experiments/v1_20260415"))
sys.path.insert(0, str(ROOT / "experiments/v1_20260415/precompute"))
sys.path.insert(0, str(Path(__file__).parent))

from _timer import StageRecord  # noqa: E402
import infer_scanned_vs_master_shot352 as infer  # noqa: E402
from gluefactory.utils.tensor import batch_to_device  # noqa: E402

LATERAL_MM = 0.056
TRANSPORT_MM = 0.056
VERTICAL_MM = 0.0085

DEFAULT_SCAN0 = ROOT / "gluefactory/datasets/scanned/scanned_data1.png"
DEFAULT_SCAN1 = ROOT / "gluefactory/datasets/scanned/roi13_zmap 1.png"
DEFAULT_CKPT = (
    ROOT / "outputs/training/iss_shot_v1_20260415_dim352_0417/checkpoint_best.tar"
)

GROUPS_4STAGE = {
    "Preprocess":   ["load_scan0", "load_scan1",
                     "preprocess_scan0", "preprocess_scan1"],
    "Descriptor":   ["iss_desc_scan0", "iss_desc_scan1"],
    "Matching":     ["lg_forward", "extract_matches"],
    "Registration": ["registration"],
}


def preprocess(zmap):
    masked, _ = infer.mask_scanned_table(zmap)
    return infer.apply_bilateral(masked)


def pixel_to_3d_mm(kp_uv, zmap_pre):
    H, W = zmap_pre.shape
    us = np.clip(kp_uv[:, 0].astype(int), 0, W - 1)
    vs = np.clip(kp_uv[:, 1].astype(int), 0, H - 1)
    d = zmap_pre[vs, us].astype(np.float64)
    pts = np.stack([us * LATERAL_MM, vs * TRANSPORT_MM, d * VERTICAL_MM], axis=1)
    return pts, d > 0


def main():
    ap = argparse.ArgumentParser(description="SHOT352 scanned×2 timing demo")
    ap.add_argument("--scan0", type=str, default=str(DEFAULT_SCAN0))
    ap.add_argument("--scan1", type=str, default=str(DEFAULT_SCAN1))
    ap.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--detailed", action="store_true")
    args = ap.parse_args()

    for p in (args.scan0, args.scan1, args.checkpoint):
        if not Path(p).exists():
            sys.exit(f"[err] path not found: {p}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        sys.stderr.write("[warn] CUDA not available, falling back to cpu\n")
        device = "cpu"

    print("=== SHOT352 Timing (scanned×2) ===")
    print(f"input : {Path(args.scan0).name} + {Path(args.scan1).name}")
    print(f"device: {device}")
    print(f"ckpt  : {Path(args.checkpoint).parent.name}/{Path(args.checkpoint).name}")
    print()

    model = infer.load_model(args.checkpoint, device)

    rec = StageRecord()

    with rec.timer("load_scan0"):
        zmap0 = np.array(Image.open(args.scan0))
    with rec.timer("load_scan1"):
        zmap1 = np.array(Image.open(args.scan1))

    with rec.timer("preprocess_scan0"):
        zmap0_pre = preprocess(zmap0)
    with rec.timer("preprocess_scan1"):
        zmap1_pre = preprocess(zmap1)

    with rec.timer("iss_desc_scan0"):
        kp0_uv, kp0_score, desc0, kp0_xyz, _, _ = \
            infer.prepare_iss_shot(zmap0_pre, label="scan0", seed=0, iss_mode="mm")
    with rec.timer("iss_desc_scan1"):
        kp1_uv, kp1_score, desc1, kp1_xyz, _, _ = \
            infer.prepare_iss_shot(zmap1_pre, label="scan1", seed=0, iss_mode="mm")

    if len(kp0_uv) == 0 or len(kp1_uv) == 0:
        sys.exit("[err] no ISS keypoints detected")

    pad_h, pad_w = zmap0_pre.shape

    if device == "cuda":
        torch.cuda.synchronize()
    with rec.timer("lg_forward"):
        view0 = infer.make_view(zmap0_pre, kp0_uv, kp0_score, desc0, pad_h, pad_w)
        view1 = infer.make_view(zmap1_pre, kp1_uv, kp1_score, desc1, pad_h, pad_w)
        batch = infer.collate_single_pair(view0, view1)
        batch = batch_to_device(batch, device)
        pred = model(batch)
        if device == "cuda":
            torch.cuda.synchronize()

    with rec.timer("extract_matches"):
        m0 = pred["matches0"][0].cpu().numpy()
        valid = (m0 > -1) & (m0 < kp1_uv.shape[0])
        mkp0 = kp0_uv[valid]
        mkp1 = kp1_uv[m0[valid]]
        n_match = int(valid.sum())

    pts0, vm0 = pixel_to_3d_mm(mkp0, zmap0_pre)
    pts1, vm1 = pixel_to_3d_mm(mkp1, zmap1_pre)
    both = vm0 & vm1
    n_3d = int(both.sum())

    n_inl = 0
    R = np.eye(3)
    t = np.zeros(3)
    skip_reg = n_3d < 3
    if skip_reg:
        print(f"[warn] <3 3D matches ({n_3d}), skip registration")
    else:
        with rec.timer("registration"):
            R, t, inl = infer._ransac_rigid(pts1[both], pts0[both])
            n_inl = int(inl.sum())

    print()
    rec.print_table(GROUPS_4STAGE, detailed=args.detailed)
    print()
    if skip_reg:
        print(f"n_matches = {n_match}, n_3d = {n_3d}  (registration skipped)")
    else:
        print(f"R = {R.tolist()}")
        print(f"t = {t.tolist()}")
        print(f"n_matches = {n_match}, n_inliers = {n_inl}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run (기본 출력)**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda run -n LightGlue python experiments/v1_20260415/profile/timing_demo_shot.py
```

Expected stdout:
- Header line `=== SHOT352 Timing (scanned×2) ===`
- `input:`, `device:`, `ckpt:` lines
- 4 stage lines: `[Preprocess  ]`, `[Descriptor  ]`, `[Matching    ]`, `[Registration]`
- `[Total       ]` line
- `R = [[...]]`, `t = [...]`, `n_matches = N, n_inliers = M`
- Exit code 0.

If `[err] path not found:` appears for ckpt (즉 `_0417` 재학습이 아직 미완), verify with:
```bash
ls /home/jhs/work/Registration/glue-factory_depth/outputs/training/iss_shot_v1_20260415_dim352_0417/checkpoint_best.tar
```
If file truly missing, run with explicit pre-_0417 fallback:
```bash
conda run -n LightGlue python experiments/v1_20260415/profile/timing_demo_shot.py \
    --checkpoint outputs/training/iss_shot_v1_20260415_dim352/checkpoint_best.tar
```
Report to user that `_0417` ckpt missing (plan doesn't fix this — it's a training dependency).

- [ ] **Step 3: Smoke run with `--detailed`**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda run -n LightGlue python experiments/v1_20260415/profile/timing_demo_shot.py --detailed
```

Expected: 9 individual timing lines (`load_scan0`, `load_scan1`, `preprocess_scan0`, `preprocess_scan1`, `iss_desc_scan0`, `iss_desc_scan1`, `lg_forward`, `extract_matches`, `registration`) + divider + 4-stage summary + Total.

- [ ] **Step 4: Edge case — missing file**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda run -n LightGlue python experiments/v1_20260415/profile/timing_demo_shot.py --scan0 /tmp/does_not_exist.png
```

Expected stdout: `[err] path not found: /tmp/does_not_exist.png`. Exit code 1.

- [ ] **Step 5: Hand-off to user for commit**

Do not run git commands.

---

### Task 3: Create `timing_demo_fpfh.py`

**Files:**
- Create: `experiments/v1_20260415/profile/timing_demo_fpfh.py`

- [ ] **Step 1: Write full demo file**

Create `experiments/v1_20260415/profile/timing_demo_fpfh.py`:

```python
"""v1_20260415 FPFH33 (_norm) scanned×2 inference timing demo.

FPFH descriptor: voxel=1mm, normal_r=20mm, fpfh_r=20mm, L2-normalized.
_norm variant ckpt 전제 → 추론 시에도 L2 normalize 적용 (train cache 분포와 동일).

Usage:
    conda activate LightGlue
    cd /home/jhs/work/Registration/glue-factory_depth
    python experiments/v1_20260415/profile/timing_demo_fpfh.py
    python experiments/v1_20260415/profile/timing_demo_fpfh.py --detailed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pybind_shot_linux"))
sys.path.insert(0, str(ROOT / "experiments/v1_20260415"))
sys.path.insert(0, str(ROOT / "experiments/v1_20260415/precompute"))
sys.path.insert(0, str(Path(__file__).parent))

from _timer import StageRecord  # noqa: E402
import infer_scanned_vs_master_shot352 as infer  # noqa: E402
import helpers as h  # noqa: E402
from gluefactory.utils.tensor import batch_to_device  # noqa: E402

LATERAL_MM = 0.056
TRANSPORT_MM = 0.056
VERTICAL_MM = 0.0085

DEFAULT_SCAN0 = ROOT / "gluefactory/datasets/scanned/scanned_data1.png"
DEFAULT_SCAN1 = ROOT / "gluefactory/datasets/scanned/roi13_zmap 1.png"
DEFAULT_CKPT = (
    ROOT / "outputs/training/iss_fpfh_v1_20260415_norm_0417/checkpoint_best.tar"
)

GROUPS_4STAGE = {
    "Preprocess":   ["load_scan0", "load_scan1",
                     "preprocess_scan0", "preprocess_scan1"],
    "Descriptor":   ["iss_desc_scan0", "iss_desc_scan1"],
    "Matching":     ["lg_forward", "extract_matches"],
    "Registration": ["registration"],
}


def preprocess(zmap):
    masked, _ = infer.mask_scanned_table(zmap)
    return infer.apply_bilateral(masked)


def zmap_array_to_pts_mm(zmap):
    vs, us = np.where(zmap > 0)
    d = zmap[vs, us].astype(np.float64)
    return np.stack(
        [us * LATERAL_MM, vs * TRANSPORT_MM, d * VERTICAL_MM], axis=1)


def prepare_iss_fpfh(zmap_pre, seed=0):
    """ISS detection + FPFH(33) on-the-fly with L2 normalize (_norm variant).

    Returns: kp_uv (M,2), kp_score (M,), desc (M,33) float32, kp_xyz_mm (M,3)
    """
    pts = zmap_array_to_pts_mm(zmap_pre)
    pcd_vox = h.voxel_downsample(pts)                 # VOXEL_SIZE=1.0mm
    kp_xyz_iss = h.extract_iss_on_dense(pts)
    kp_xyz, kp_idx, kp_score, _, _ = h.subsample_or_pad_keypoints(
        kp_xyz_iss, pcd_vox, max_n=h.MAX_KEYPOINTS, seed=seed)
    pcd_vox.estimate_normals(                         # NORMAL_RADIUS=20.0mm
        search_param=o3d.geometry.KDTreeSearchParamRadius(h.NORMAL_RADIUS))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_vox, o3d.geometry.KDTreeSearchParamRadius(h.FPFH_RADIUS))  # 20.0mm
    desc = np.asarray(fpfh.data, dtype=np.float32)[:, kp_idx].T
    desc = desc / (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12)
    kp_uv = h.kp_xyz_mm_to_uv_padded(kp_xyz)
    return kp_uv, kp_score, desc.astype(np.float32), kp_xyz


def pixel_to_3d_mm(kp_uv, zmap_pre):
    H, W = zmap_pre.shape
    us = np.clip(kp_uv[:, 0].astype(int), 0, W - 1)
    vs = np.clip(kp_uv[:, 1].astype(int), 0, H - 1)
    d = zmap_pre[vs, us].astype(np.float64)
    pts = np.stack([us * LATERAL_MM, vs * TRANSPORT_MM, d * VERTICAL_MM], axis=1)
    return pts, d > 0


def main():
    ap = argparse.ArgumentParser(description="FPFH33 scanned×2 timing demo")
    ap.add_argument("--scan0", type=str, default=str(DEFAULT_SCAN0))
    ap.add_argument("--scan1", type=str, default=str(DEFAULT_SCAN1))
    ap.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--detailed", action="store_true")
    args = ap.parse_args()

    for p in (args.scan0, args.scan1, args.checkpoint):
        if not Path(p).exists():
            sys.exit(f"[err] path not found: {p}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        sys.stderr.write("[warn] CUDA not available, falling back to cpu\n")
        device = "cpu"

    print("=== FPFH33 (_norm) Timing (scanned×2) ===")
    print(f"input : {Path(args.scan0).name} + {Path(args.scan1).name}")
    print(f"device: {device}")
    print(f"ckpt  : {Path(args.checkpoint).parent.name}/{Path(args.checkpoint).name}")
    print()

    model = infer.load_model(args.checkpoint, device)

    rec = StageRecord()

    with rec.timer("load_scan0"):
        zmap0 = np.array(Image.open(args.scan0))
    with rec.timer("load_scan1"):
        zmap1 = np.array(Image.open(args.scan1))

    with rec.timer("preprocess_scan0"):
        zmap0_pre = preprocess(zmap0)
    with rec.timer("preprocess_scan1"):
        zmap1_pre = preprocess(zmap1)

    with rec.timer("iss_desc_scan0"):
        kp0_uv, kp0_score, desc0, kp0_xyz = prepare_iss_fpfh(zmap0_pre, seed=0)
    with rec.timer("iss_desc_scan1"):
        kp1_uv, kp1_score, desc1, kp1_xyz = prepare_iss_fpfh(zmap1_pre, seed=0)

    if len(kp0_uv) == 0 or len(kp1_uv) == 0:
        sys.exit("[err] no ISS keypoints detected")

    pad_h, pad_w = zmap0_pre.shape

    if device == "cuda":
        torch.cuda.synchronize()
    with rec.timer("lg_forward"):
        view0 = infer.make_view(zmap0_pre, kp0_uv, kp0_score, desc0, pad_h, pad_w)
        view1 = infer.make_view(zmap1_pre, kp1_uv, kp1_score, desc1, pad_h, pad_w)
        batch = infer.collate_single_pair(view0, view1)
        batch = batch_to_device(batch, device)
        pred = model(batch)
        if device == "cuda":
            torch.cuda.synchronize()

    with rec.timer("extract_matches"):
        m0 = pred["matches0"][0].cpu().numpy()
        valid = (m0 > -1) & (m0 < kp1_uv.shape[0])
        mkp0 = kp0_uv[valid]
        mkp1 = kp1_uv[m0[valid]]
        n_match = int(valid.sum())

    pts0, vm0 = pixel_to_3d_mm(mkp0, zmap0_pre)
    pts1, vm1 = pixel_to_3d_mm(mkp1, zmap1_pre)
    both = vm0 & vm1
    n_3d = int(both.sum())

    n_inl = 0
    R = np.eye(3)
    t = np.zeros(3)
    skip_reg = n_3d < 3
    if skip_reg:
        print(f"[warn] <3 3D matches ({n_3d}), skip registration")
    else:
        with rec.timer("registration"):
            R, t, inl = infer._ransac_rigid(pts1[both], pts0[both])
            n_inl = int(inl.sum())

    print()
    rec.print_table(GROUPS_4STAGE, detailed=args.detailed)
    print()
    if skip_reg:
        print(f"n_matches = {n_match}, n_3d = {n_3d}  (registration skipped)")
    else:
        print(f"R = {R.tolist()}")
        print(f"t = {t.tolist()}")
        print(f"n_matches = {n_match}, n_inliers = {n_inl}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run (기본 출력)**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda run -n LightGlue python experiments/v1_20260415/profile/timing_demo_fpfh.py
```

Expected: Task 2 Step 2 와 동일한 포맷. Header 는 `=== FPFH33 (_norm) Timing (scanned×2) ===`, ckpt name `iss_fpfh_v1_20260415_norm_0417`. Exit code 0.

(`_0417` 재학습 미완이면 Task 2 Step 2 와 동일하게 pre-_0417 fallback 사용: `--checkpoint outputs/training/iss_fpfh_v1_20260415_norm/checkpoint_best.tar`.)

- [ ] **Step 3: Smoke run with `--detailed`**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda run -n LightGlue python experiments/v1_20260415/profile/timing_demo_fpfh.py --detailed
```

Expected: Task 2 Step 3 와 동일 포맷. 9 stage 이름이 동일 (`load_scan0` … `registration`).

- [ ] **Step 4: Hand-off to user for commit**

Do not run git commands.

---

### Task 4: Final Back-to-Back Verification

- [ ] **Step 1: Run both demos in sequence**

Run:
```bash
cd /home/jhs/work/Registration/glue-factory_depth
conda run -n LightGlue python experiments/v1_20260415/profile/timing_demo_shot.py
conda run -n LightGlue python experiments/v1_20260415/profile/timing_demo_fpfh.py
```

Expected: 둘 다 정상 완료. 각각 4-stage 테이블 + R, t + match/inlier 카운트 출력. Exit code 0.

- [ ] **Step 2: 숫자 기록 (수동)**

두 실행의 `[Total]` 값과 가장 큰 stage 를 기록. 이 plan 의 범위 밖이지만 후속 최적화 근거가 됨:

| descriptor | Total (s) | Max stage | Max time (s) |
|---|---|---|---|
| SHOT352 | ? | ? | ? |
| FPFH33_norm | ? | ? | ? |

- [ ] **Step 3: Hand-off to user for commit**

Do not run git commands. 사용자가 모든 task 완료 확인 후 본인 이름으로 커밋.

---

## Notes

- `infer._ransac_rigid` 는 underscore prefix (Python convention 상 private) 지만 spec 결정에 따라 직접 import 해 재사용. 추후 core helper 로 승격 시 이름 정리 (Future Work).
- v1 FPFH `_norm` ckpt 는 cache 자체가 L2-normalized → 추론 `desc` 도 반드시 L2 normalize (Task 3 `prepare_iss_fpfh` 안에 포함). non-norm ckpt 로 돌리려면 해당 한 줄을 제거하고 `--checkpoint` 를 `iss_fpfh_v1_20260415/checkpoint_best.tar` 로 변경.
- `_0417` suffix ckpt 는 padding bug fix 후 재학습본. 재학습 도중이라면 pre-_0417 variant 로 fallback 해도 demo 동작은 동일 (숫자만 약간 다름).
