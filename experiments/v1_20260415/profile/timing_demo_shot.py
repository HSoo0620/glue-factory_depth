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
from datetime import datetime
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
DEFAULT_CSV = Path(__file__).parent / "logs/timing.csv"

STAGE_ORDER = [
    "load_scan0", "load_scan1",
    "preprocess_scan0", "preprocess_scan1",
    "iss_desc_scan0", "iss_desc_scan1",
    "lg_forward", "extract_matches",
    "registration",
]

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
    ap.add_argument("--csv", type=str, default=str(DEFAULT_CSV),
                    help="CSV 누적 저장 경로 ('' 빈 문자열이면 저장 안함)")
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

    pad_h = max(zmap0_pre.shape[0], zmap1_pre.shape[0])
    pad_w = max(zmap0_pre.shape[1], zmap1_pre.shape[1])

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

    if args.csv:
        try:
            ckpt_rel = str(Path(args.checkpoint).resolve().relative_to(ROOT))
        except ValueError:
            ckpt_rel = args.checkpoint
        metadata = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "descriptor": "SHOT352",
            "scan0": Path(args.scan0).name,
            "scan1": Path(args.scan1).name,
            "ckpt": ckpt_rel,
            "device": device,
            "n_matches": n_match,
            "n_inliers": n_inl,
            "skip_reg": int(skip_reg),
        }
        rec.save_csv(args.csv, metadata, stage_order=STAGE_ORDER)
        print(f"\n[csv] appended → {args.csv}")


if __name__ == "__main__":
    main()
