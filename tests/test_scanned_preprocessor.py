"""Unit tests for preprocess_scanned — shape, dtype, pad positioning, dz scaling."""
import unittest
import tempfile
import numpy as np
import cv2
from pathlib import Path


def _write_synthetic_scanned(path, height=3515, width=2687, fill_value=29334):
    """Create a synthetic uint16 depth PNG with a bright rectangular 'object'."""
    img = np.zeros((height, width), dtype=np.uint16)
    # Center rectangle "object"
    img[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = fill_value
    cv2.imwrite(str(path), img)


class TestPreprocessScanned(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.scanned_path = Path(self.tmpdir.name) / "synthetic.png"
        _write_synthetic_scanned(self.scanned_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_output_shapes_and_dtypes(self):
        from test_scanned_vs_master import preprocess_scanned
        crop_raw, img_small, meta = preprocess_scanned(
            self.scanned_path, scale_dz=False
        )
        self.assertEqual(crop_raw.shape, (3502, 3502))
        self.assertEqual(crop_raw.dtype, np.uint16)
        self.assertEqual(img_small.shape, (1751, 1751))
        self.assertEqual(img_small.dtype, np.float32)

    def test_center_padding_positioning(self):
        from test_scanned_vs_master import preprocess_scanned
        crop_raw, _, meta = preprocess_scanned(self.scanned_path, scale_dz=False)
        # Left/right borders should be zero-padded after aspect-preserving scale
        self.assertTrue(meta["pad_left"] > 0)
        # Scaled content width should be less than canvas width
        self.assertLess(meta["scaled_shape"][1], 3502)
        # Scaled content height should equal canvas height
        self.assertEqual(meta["scaled_shape"][0], 3502)
        # Padded regions on the far left and far right are zero
        self.assertTrue(np.all(crop_raw[:, :10] == 0))
        self.assertTrue(np.all(crop_raw[:, -10:] == 0))

    def test_scale_factor(self):
        from test_scanned_vs_master import preprocess_scanned
        crop_raw, _, meta = preprocess_scanned(self.scanned_path, scale_dz=False)
        # 3502 / max(3515, 2687) = 3502 / 3515 ≈ 0.9963
        self.assertAlmostEqual(meta["scale_factor"], 3502 / 3515, places=4)

    def test_dz_scaling_applies_only_when_flag_set(self):
        from test_scanned_vs_master import preprocess_scanned
        crop_raw_raw, _, meta_raw = preprocess_scanned(
            self.scanned_path, scale_dz=False
        )
        crop_raw_fix, _, meta_fix = preprocess_scanned(
            self.scanned_path, scale_dz=True, dz_ratio=0.425
        )
        self.assertFalse(meta_raw["dz_scaled"])
        self.assertTrue(meta_fix["dz_scaled"])
        # Non-zero pixels in dzfix should be ~0.425× of the raw ones
        mask = crop_raw_raw > 0
        raw_vals = crop_raw_raw[mask].astype(np.float64)
        fix_vals = crop_raw_fix[mask].astype(np.float64)
        ratios = fix_vals / raw_vals
        self.assertTrue(np.all(np.abs(ratios - 0.425) < 0.01))
        # Zero pixels remain zero
        self.assertTrue(np.all(crop_raw_fix[~mask] == 0))

    def test_img_small_is_normalized(self):
        from test_scanned_vs_master import preprocess_scanned
        _, img_small, _ = preprocess_scanned(self.scanned_path, scale_dz=False)
        self.assertLessEqual(img_small.max(), 1.0)
        self.assertGreaterEqual(img_small.min(), 0.0)


if __name__ == "__main__":
    unittest.main()
