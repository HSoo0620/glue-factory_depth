"""SHOT Phase 3 수치 일치 검증 (Windows ↔ Linux).

동일 master PNG + v1 파라미터로 계산한 SHOT descriptor 가 OS 간 일치하는지 확인.

사용 (두 단계):

1) Linux 에서 reference 생성 (이미 수행되어 있으면 생략 가능):
   python examples/verify_shot_phase3.py --produce-reference

2) Windows 에서 비교:
   python examples/verify_shot_phase3.py [--tol 1e-3]

기본 reference 경로: `tests/fixtures/shot_master_linux_reference.npy`
  (gitignore 된 파일이라 플랫폼 간 공유는 직접 복사/드라이브로 전달)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from depth_registration.descriptors.shot import compute_shot  # noqa: E402
from depth_registration.iss import detect_iss_mm  # noqa: E402
from depth_registration.params import DEFAULT_MASTER_PATH  # noqa: E402
from depth_registration.preprocessing import load_depth_raw, preprocess_master  # noqa: E402

REF_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "shot_master_linux_reference.npy"
KP_PATH  = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "shot_master_linux_kp.npy"


def compute_master_shot() -> tuple[np.ndarray, np.ndarray]:
    z = load_depth_raw(DEFAULT_MASTER_PATH)
    pts = preprocess_master(z)
    kp = detect_iss_mm(pts, seed=0)
    desc = compute_shot(pts, kp)
    return desc, kp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--produce-reference", action="store_true",
                    help="Linux reference 파일을 새로 생성하여 저장")
    ap.add_argument("--ref", type=Path, default=REF_PATH,
                    help=f"Reference .npy 경로 (default: {REF_PATH})")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="허용 평균 L2 거리 (default: 1e-3)")
    args = ap.parse_args()

    desc, kp = compute_master_shot()
    print(f"computed: desc={desc.shape}, kp={kp.shape}")

    if args.produce_reference:
        args.ref.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.ref, desc)
        np.save(args.ref.with_name("shot_master_linux_kp.npy"), kp)
        print(f"saved reference: {args.ref}")
        return 0

    if not args.ref.exists():
        print(f"[error] reference not found: {args.ref}")
        print("Linux 측에서 `python examples/verify_shot_phase3.py --produce-reference` "
              "로 생성한 .npy 를 해당 경로에 배치하세요.")
        return 2

    ref = np.load(args.ref).astype(np.float32)
    if ref.shape != desc.shape:
        print(f"[error] shape mismatch: ref={ref.shape}, current={desc.shape}")
        return 3

    # ISS 는 결정적(seed=0)이지만 floating point 로 keypoint 순서가 플랫폼 간
    # 다를 수 있어, keypoint 집합이 같다는 전제 하에 row-wise 비교를 시도.
    dist = np.linalg.norm(desc - ref, axis=1)
    mean_l2 = float(dist.mean())
    max_l2  = float(dist.max())
    nonzero = int((np.linalg.norm(desc, axis=1) > 1e-6).sum())
    print(f"pairwise L2: mean={mean_l2:.6f}, max={max_l2:.6f}, "
          f"valid_rows={nonzero}/{len(desc)}")

    if mean_l2 <= args.tol:
        print(f"[PASS] mean L2 {mean_l2:.6f} ≤ {args.tol}")
        return 0
    else:
        print(f"[FAIL] mean L2 {mean_l2:.6f} > {args.tol} — "
              "ISS keypoint 순서 차이 가능성. keypoint_xyz 를 비교해 진단 필요.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
