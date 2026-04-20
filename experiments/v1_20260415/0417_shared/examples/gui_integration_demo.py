"""GUI 통합 참고 예제 — 실제 애플리케이션에 이식할 때 이 파일을 뼈대로 쓰면 된다.

구조
----
- on_app_start()        : 앱 기동 시 1회. 모델 프리로드 + (선택) warmup.
- on_register_clicked() : "정합" 버튼 클릭 시마다 호출. 결과를 UI 친화적 dict 로 반환.
- on_app_exit()         : 앱 종료 시 1회. GPU 메모리 해제.

프레임워크 중립이라 PyQt, Tkinter, CLI 어디에 붙여도 그대로 동작한다.
자세한 레퍼런스/트러블슈팅은 ../docs/USAGE.md.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Union

import numpy as np

# 이 예제를 `python examples/gui_integration_demo.py` 로 직접 실행하기 위한 path 주입.
# 실제 GUI 앱에서는 `depth_registration` 이 import path 에 이미 있을 것이므로 불필요.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from depth_registration import (  # noqa: E402
    preload_model,
    register_pair,
    unload_model,
)

ScannedInput = Union[str, Path, np.ndarray]


# ─────────────────────────────────────────────────────────────────────────────
# 앱 기동 시 1회
# ─────────────────────────────────────────────────────────────────────────────
def on_app_start(descriptor: str = "fpfh", device: str = "cuda") -> None:
    """LightGlue 모델을 GPU 에 미리 올려 첫 호출의 cold-start 지연을 없앤다.

    왜 앞에서 해야 하나:
        - 첫 register_pair 호출은 ckpt 파일 IO + state_dict 로드로 1~2초 걸린다.
        - 버튼을 누른 순간 이 지연이 발생하면 사용자에게는 "멈춘" 것처럼 보임.
        - 앱 시작 로딩 스피너가 이미 떠 있는 시점으로 이 비용을 옮기는 게 자연스럽다.

    한 번에 하나만 상주:
        - `preload_model("fpfh")` 후 `preload_model("shot")` 를 호출하면 FPFH 는
          자동 언로드된다. 두 descriptor 를 동시에 메모리에 올리지 않는다.
    """
    preload_model(descriptor, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# 버튼 클릭 시마다
# ─────────────────────────────────────────────────────────────────────────────
def on_register_clicked(
    scanned: ScannedInput,
    *,
    descriptor: str = "fpfh",
    device: str = "cuda",
) -> dict[str, Any]:
    """한 번의 정합 호출을 감싸 UI 친화적 dict 로 변환한다.

    입력
    ----
    scanned : 파일 경로(str/Path) 또는 `np.ndarray[uint16] (H, W)`.
              센서에서 depth 를 메모리로 바로 받는 경우 PNG 저장을 생략할 수 있다.
    descriptor : "fpfh" (빠르고 기본) 또는 "shot" (느리지만 inlier 많음).
    device : "cuda" 권장. VRAM 부족 시 "cpu".

    반환
    ----
    UI 가 바로 파싱하기 좋은 평문 dict. 실패/예외도 같은 스키마로 반환하므로
    caller 쪽에서 try/except 를 두지 않고 `ok` 필드만 봐도 된다.
    """
    try:
        r = register_pair(scanned, descriptor=descriptor, device=device)
    except FileNotFoundError as e:
        # ckpt 미배치 / 잘못된 scanned 경로 등
        return {"ok": False, "reason": f"file_not_found: {e}"}
    except Exception as e:  # noqa: BLE001
        # 기타 예외는 전부 UI 에는 간단한 메시지로, 로그에는 traceback 으로.
        traceback.print_exc()
        return {"ok": False, "reason": f"exception: {type(e).__name__}: {e}"}

    if not r["success"]:
        # success 판정 기준: num_inliers >= 3 AND inlier_ratio >= 0.1.
        # 여기서는 caller 가 status bar 에 띄우기 좋게 수치를 같이 돌려준다.
        return {
            "ok": False,
            "reason": "low_inlier_ratio",
            "num_matches": r["num_matches"],
            "num_inliers": r["num_inliers"],
            "inlier_ratio": r["inlier_ratio"],
            "elapsed_ms": r["elapsed_ms"],
        }

    # T 를 JSON 직렬화 가능한 list-of-list 로 변환.
    # 3D 뷰어에 직접 적용할 거라면 r["T"] (ndarray) 를 그대로 넘기는 게 빠름.
    return {
        "ok": True,
        "T": r["T"].tolist(),
        "R": r["R"].tolist(),
        "t": r["t"].tolist(),
        "num_matches": r["num_matches"],
        "num_inliers": r["num_inliers"],
        "inlier_ratio": r["inlier_ratio"],
        "elapsed_ms": r["elapsed_ms"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 앱 종료 시
# ─────────────────────────────────────────────────────────────────────────────
def on_app_exit() -> None:
    """GPU 메모리를 해제한다. 프로세스 종료 시엔 OS 가 정리하지만,
    동일 프로세스에서 다른 CUDA 작업을 이어 돌린다면 명시적으로 호출해두는 편이 안전."""
    unload_model()


# ─────────────────────────────────────────────────────────────────────────────
# (PyQt 시나리오 참고 스니펫)
# ─────────────────────────────────────────────────────────────────────────────
# PyQt 의 메인 스레드에서 register_pair 를 직접 호출하면 UI 가 수 초간 멈춘다.
# 워커 스레드로 분리하는 최소 골격:
#
#   from PyQt5.QtCore import QThread, pyqtSignal
#
#   class RegisterWorker(QThread):
#       finished_ok  = pyqtSignal(dict)
#       finished_err = pyqtSignal(dict)
#
#       def __init__(self, scanned_path: str):
#           super().__init__()
#           self.scanned_path = scanned_path
#
#       def run(self):
#           res = on_register_clicked(self.scanned_path)
#           (self.finished_ok if res["ok"] else self.finished_err).emit(res)
#
#   # 주의: preload_model / register_pair / unload_model 은 동일 스레드에서
#   # 호출되어야 안전하다. 즉 워커를 매 호출마다 새로 만들지 말고 하나 유지하거나,
#   # 모든 호출을 "등록 전용 스레드" 한 곳에 모아 직렬화할 것.


# ─────────────────────────────────────────────────────────────────────────────
# 직접 실행 (스모크)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    on_app_start(descriptor="fpfh")

    # 테스트용 scanned 는 repo 동봉 샘플. 실제 앱에서는 파일 다이얼로그 결과 등.
    sample = str(Path(__file__).resolve().parent.parent
                 / "tests" / "fixtures" / "sample_scanned.png")
    result = on_register_clicked(sample)

    # UI 가 받을 법한 형태로 요약 출력.
    if result["ok"]:
        ms = result["elapsed_ms"]
        print(f"[OK]  inliers={result['num_inliers']}/{result['num_matches']} "
              f"(ratio={result['inlier_ratio']:.2%}), total={ms['total']:.0f} ms")
        print("T =")
        for row in result["T"]:
            print("   ", [f"{v:+.4f}" for v in row])
    else:
        print(f"[FAIL] reason={result['reason']}, detail={result}")

    on_app_exit()
