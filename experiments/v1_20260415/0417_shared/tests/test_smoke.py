import numpy as np
import pytest
from pathlib import Path

from depth_registration import register_pair, unload_model
from depth_registration.params import DEFAULT_MASTER_PATH


FIXTURE_SCAN = Path(__file__).parent / "fixtures" / "sample_scanned.png"


def _has_pcl() -> bool:
    try:
        from depth_registration.descriptors.shot import _lazy_import_shot
        _lazy_import_shot()
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _clear_models():
    yield
    unload_model()


def test_fpfh_self_matching():
    """master ↔ master: scanned 경로엔 floor-mask/bilateral 이 적용되지만
    synthetic master 는 floor 가 없어 실질 no-op → T ≈ I.
    """
    r = register_pair(DEFAULT_MASTER_PATH, DEFAULT_MASTER_PATH, descriptor="fpfh")
    assert r["success"], f"self-match should succeed; result={r}"
    assert r["num_inliers"] >= 20
    # bilateral 로 인한 미세한 drift 허용 (3 mm)
    assert np.allclose(r["T"], np.eye(4), atol=3.0)


def test_fpfh_sample_scanned():
    r = register_pair(str(FIXTURE_SCAN), descriptor="fpfh")
    assert r["num_matches"] > 0


@pytest.mark.skipif(not _has_pcl(), reason="PCL / shot_module unavailable")
def test_shot_self_matching():
    r = register_pair(DEFAULT_MASTER_PATH, DEFAULT_MASTER_PATH, descriptor="shot")
    assert r["success"]
    assert r["num_inliers"] >= 20
    assert np.allclose(r["T"], np.eye(4), atol=3.0)
