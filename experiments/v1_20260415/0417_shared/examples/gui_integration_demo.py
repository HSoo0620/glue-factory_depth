"""GUI 측 통합 예시 (약 10줄).

앱 시작 시 `on_app_start()` 한 번, 버튼 클릭마다 `on_register_clicked(path)`.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from depth_registration import preload_model, register_pair  # noqa: E402


def on_app_start() -> None:
    preload_model("fpfh", device="cuda")


def on_register_clicked(scanned_path: str) -> dict:
    r = register_pair(scanned_path, descriptor="fpfh")
    if not r["success"]:
        return {
            "ok": False,
            "reason": "low inlier ratio",
            "num_matches": r["num_matches"],
            "num_inliers": r["num_inliers"],
        }
    return {
        "ok": True,
        "T": r["T"].tolist(),
        "num_matches": r["num_matches"],
        "num_inliers": r["num_inliers"],
        "inlier_ratio": r["inlier_ratio"],
        "elapsed_ms": r["elapsed_ms"],
    }


if __name__ == "__main__":
    on_app_start()
    print(on_register_clicked("tests/fixtures/sample_scanned.png"))
