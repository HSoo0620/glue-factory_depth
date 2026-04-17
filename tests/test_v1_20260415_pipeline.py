"""v1 (2026-04-15) precompute/dataset 단위 테스트."""
import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import precompute_helpers_v1_20260415 as h


SAMPLE_PNG = "/mnt/aict_nas/snuailab_datasets/Feature_Matching/dataset_output/zmap_0000.png"


def test_load_zmap_units():
    pts, uv, shape = h.load_zmap_to_pcd_mm(SAMPLE_PNG)
    assert pts.dtype == np.float32
    assert pts.shape[1] == 3
    assert uv.shape[1] == 2
    assert len(pts) == len(uv)
    # mm 좌표 sanity: X = uv[0]*0.056
    np.testing.assert_allclose(pts[:, 0], uv[:, 0] * 0.056, rtol=1e-5)
    np.testing.assert_allclose(pts[:, 1], uv[:, 1] * 0.056, rtol=1e-5)


def test_voxel_downsample():
    pts, _, _ = h.load_zmap_to_pcd_mm(SAMPLE_PNG)
    pcd_vox = h.voxel_downsample(pts, voxel_size=1.0)
    M = len(pcd_vox.points)
    assert M < len(pts), "voxel must reduce point count"
    assert M > 100, f"too sparse: {M}"
    print(f"  N={len(pts)} -> M={M} (1mm voxel)")


def test_iss_extraction():
    pts, _, _ = h.load_zmap_to_pcd_mm(SAMPLE_PNG)
    pcd_vox = h.voxel_downsample(pts)
    kp_xyz = h.extract_iss_on_voxel(pcd_vox)
    assert kp_xyz.shape[1] == 3
    assert len(kp_xyz) > 0
    print(f"  ISS keypoints: {len(kp_xyz)}")


def test_subsample_pad_to_512():
    pts, _, _ = h.load_zmap_to_pcd_mm(SAMPLE_PNG)
    pcd_vox = h.voxel_downsample(pts)
    kp_xyz = h.extract_iss_on_voxel(pcd_vox)
    out_xyz, out_idx, out_score, n_iss, n_valid = h.subsample_or_pad_keypoints(
        kp_xyz, pcd_vox, max_n=512, seed=0)
    assert out_xyz.shape == (512, 3)
    assert out_idx.shape == (512,)
    assert out_score.shape == (512,)
    # 인덱스가 P_vox 범위 내
    assert (out_idx >= 0).all() and (out_idx < len(pcd_vox.points)).all()
    # 인덱스가 가리키는 점이 out_xyz와 일치
    vox_pts = np.asarray(pcd_vox.points, dtype=np.float32)
    np.testing.assert_allclose(vox_pts[out_idx[:n_valid]], out_xyz[:n_valid], rtol=1e-4)


def test_kp_xyz_to_uv_round_trip():
    """mm → (u,v) → mm 왕복 일치."""
    rng = np.random.default_rng(0)
    kp = rng.uniform(0, 100, (10, 3)).astype(np.float32)
    uv = h.kp_xyz_mm_to_uv_padded(kp)
    np.testing.assert_allclose(uv[:, 0] * 0.056, kp[:, 0], rtol=1e-5)
    np.testing.assert_allclose(uv[:, 1] * 0.056, kp[:, 1], rtol=1e-5)


def test_zero_pad_zmap():
    from PIL import Image
    img = np.array(Image.open(SAMPLE_PNG))
    pad = h.zero_pad_zmap(img)
    assert pad.shape == (3008, 2432)
    assert pad.dtype == img.dtype
    # 원본 영역 보존
    H, W = img.shape
    np.testing.assert_array_equal(pad[:H, :W], img)
    # 패딩 영역 0
    assert (pad[H:, :] == 0).all()
    assert (pad[:, W:] == 0).all()


def test_dataset_fpfh_loads_one_pair():
    from gluefactory.datasets.mitsubishi_v1_20260415_iss_fpfh_dataset import (
        MitsubishiV1ISSFPFHDataset, v1_iss_fpfh_collate_fn,
    )
    ds = MitsubishiV1ISSFPFHDataset(split="val")
    assert len(ds) > 0
    print(f"  val pairs: {len(ds)}")
    item = ds[0]
    assert item["view0"]["image"].shape == (1, 3008, 2432)
    assert item["view0"]["keypoints"].shape == (512, 2)
    assert item["view0"]["descriptors"].shape == (512, 33)
    assert item["view1"]["descriptors"].shape == (512, 33)
    assert item["gt_matches"].shape[1] == 4
    batch = v1_iss_fpfh_collate_fn([ds[0], ds[1]])
    assert batch["view0"]["image"].shape == (2, 1, 3008, 2432)


def test_dataset_shot_loads_one_pair():
    from gluefactory.datasets.mitsubishi_v1_20260415_iss_shot_dataset import (
        MitsubishiV1ISSSHOTDataset, v1_iss_shot_collate_fn,
    )
    ds = MitsubishiV1ISSSHOTDataset(split="val")
    assert len(ds) > 0
    item = ds[0]
    assert item["view0"]["descriptors"].shape == (512, 352)


def test_split_no_leakage():
    from gluefactory.datasets.mitsubishi_v1_20260415_iss_fpfh_dataset import (
        MitsubishiV1ISSFPFHDataset, _zmap_id,
    )
    train = MitsubishiV1ISSFPFHDataset(split="train")
    val = MitsubishiV1ISSFPFHDataset(split="val")
    test = MitsubishiV1ISSFPFHDataset(split="test")
    train_ids = set()
    val_ids = set()
    test_ids = set()
    for ds, s in [(train, train_ids), (val, val_ids), (test, test_ids)]:
        for i in range(len(ds.combo)):
            row = ds.combo.iloc[i]
            s.add(_zmap_id(row["master_zmap_path"]))
            s.add(_zmap_id(row["input_zmap_path"]))
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    print(f"  train_ids: {len(train_ids)}, val_ids: {len(val_ids)}, test_ids: {len(test_ids)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
