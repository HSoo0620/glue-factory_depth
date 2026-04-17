import numpy as np

from depth_registration.preprocessing import (
    load_depth_raw, mask_scanned_table, apply_bilateral,
    zmap_to_pcd_mm, preprocess_scanned, preprocess_master,
)


def _synthetic_zmap(h: int = 64, w: int = 64, floor_val: int = 1000,
                    object_val: int = 3000) -> np.ndarray:
    z = np.full((h, w), floor_val, dtype=np.uint16)
    z[20:40, 20:40] = object_val
    z[:5, :]  = 0
    z[-5:, :] = 0
    return z


def test_load_depth_raw_accepts_ndarray_and_path(tmp_path):
    z = _synthetic_zmap()
    assert load_depth_raw(z).dtype == np.uint16
    from PIL import Image
    p = tmp_path / "z.png"
    Image.fromarray(z).save(p)
    assert load_depth_raw(str(p)).shape == z.shape


def test_mask_scanned_table_removes_floor_band():
    z = _synthetic_zmap()
    out = mask_scanned_table(z)
    # floor(1000) 근처는 마스킹, object(3000) 은 보존
    assert out[25, 25] == 3000
    assert out[50, 50] == 0


def test_apply_bilateral_preserves_shape_and_dtype():
    z = _synthetic_zmap().astype(np.uint16)
    out = apply_bilateral(z)
    assert out.shape == z.shape
    assert out.dtype == z.dtype


def test_zmap_to_pcd_mm_shape_and_units():
    z = _synthetic_zmap()
    pts = zmap_to_pcd_mm(z, erode_boundary_px=0)
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert pts.dtype == np.float32
    # 최대 Z 값: 3000 * 0.0085 ≈ 25.5 mm
    assert abs(pts[:, 2].max() - 3000 * 0.0085) < 1e-3


def test_preprocess_scanned_returns_mm_pcd():
    z = _synthetic_zmap()
    pts = preprocess_scanned(z)
    assert pts.ndim == 2 and pts.shape[1] == 3 and pts.dtype == np.float32


def test_preprocess_master_skips_floor_mask():
    z = _synthetic_zmap()
    pts_m = preprocess_master(z)
    pts_s = preprocess_scanned(z)
    # master 는 floor 제거 없이 floor 픽셀도 포함되므로 더 많은 점
    assert pts_m.shape[0] >= pts_s.shape[0]
