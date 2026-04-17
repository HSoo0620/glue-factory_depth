import numpy as np
from pathlib import Path

from depth_registration.cache import (
    master_cache_key, load_master_cache, save_master_cache,
)


def _payload(k_dim: int = 33):
    return dict(
        kp_mm  = np.random.default_rng(0).random((50, 3)).astype(np.float32),
        desc   = np.random.default_rng(1).random((50, k_dim)).astype(np.float32),
        pts_mm = np.random.default_rng(2).random((1000, 3)).astype(np.float32),
        img_hash    = "deadbeef",
        params_hash = "cafebabe",
        descriptor  = "fpfh",
        params_json = "{}",
        created_at  = "2026-04-17T00:00:00Z",
        version     = "0.1.0",
    )


def test_master_cache_key_deterministic():
    z = np.zeros((10, 10), dtype=np.uint16)
    params = {"a": 1, "b": 2.0}
    k1 = master_cache_key(z, "fpfh", params)
    k2 = master_cache_key(z, "fpfh", params)
    assert k1 == k2
    k3 = master_cache_key(z, "fpfh", {"a": 1, "b": 3.0})
    assert k1 != k3


def test_roundtrip(tmp_path):
    p = _payload()
    save_master_cache("key1", p, tmp_path)
    loaded = load_master_cache("key1", tmp_path)
    assert loaded is not None
    np.testing.assert_array_equal(loaded["kp_mm"], p["kp_mm"])
    np.testing.assert_array_equal(loaded["desc"],  p["desc"])
    assert str(loaded["descriptor"]) == "fpfh"


def test_load_miss_returns_none(tmp_path):
    assert load_master_cache("nonexistent", tmp_path) is None


def test_corrupted_npz_returns_none_and_removes(tmp_path):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a real npz")
    assert load_master_cache("bad", tmp_path) is None
    assert not bad.exists(), "corrupted file should be removed"
