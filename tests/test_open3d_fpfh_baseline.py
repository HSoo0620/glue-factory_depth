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
