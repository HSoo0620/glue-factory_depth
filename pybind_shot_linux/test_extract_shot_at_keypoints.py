"""extract_shot_at_keypoints 단위 검증."""
import numpy as np
import pytest
import shot_module


def make_test_cloud(n=20000, seed=0):
    """n-pt cloud in [0, 100] mm box."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(0, 100, (n, 3)).astype(np.float32))


def test_signature_and_shapes():
    pts = make_test_cloud()
    kps = pts[:512].copy()
    r = shot_module.extract_shot_at_keypoints(
        pts, kps,
        voxel_size=1.0, normal_radius=20.0, shot_radius=40.0)
    assert r["descriptors"].shape == (512, 352)
    assert r["valid_mask"].shape == (512,)
    assert r["num_keypoints"] == 512
    assert r["num_input"] == len(pts)


def test_valid_mask_majority_true():
    """대부분의 keypoint에서 SHOT 계산 성공해야 함 (>50%)."""
    pts = make_test_cloud()
    kps = pts[:512].copy()
    r = shot_module.extract_shot_at_keypoints(
        pts, kps, 1.0, 20.0, 40.0)
    valid_ratio = r["valid_mask"].mean()
    assert valid_ratio > 0.5, f"valid ratio too low: {valid_ratio:.2%}"


def test_keypoint_only_matches_dense_lookup():
    """SHOT(setSearchSurface=P_vox, setInputCloud=keypoints) 결과는
    SHOT(setInputCloud=P_vox)의 동일 keypoint 위치 lookup과
    cosine similarity >= 0.99 (수치적 동일성)."""
    pts = make_test_cloud(n=30000, seed=1)
    dense = shot_module.extract_shot(
        pts, voxel_size=1.0, normal_radius=20.0, shot_radius=40.0)
    dense_pts = dense["points"]
    dense_desc = dense["descriptors"]
    rng = np.random.default_rng(42)
    sel = rng.choice(len(dense_pts), 50, replace=False)
    kps = dense_pts[sel]
    expected = dense_desc[sel]
    new = shot_module.extract_shot_at_keypoints(
        pts, kps, voxel_size=1.0, normal_radius=20.0, shot_radius=40.0)
    actual = new["descriptors"]
    valid = new["valid_mask"]
    a = actual[valid]; b = expected[valid]
    sim = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    median_sim = float(np.median(sim))
    print(f"median cosine sim: {median_sim:.4f}")
    assert median_sim >= 0.99, f"too low: {median_sim:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
