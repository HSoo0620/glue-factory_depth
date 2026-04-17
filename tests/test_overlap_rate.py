"""Unit tests for compute_overlap_rate — synthetic point clouds with known overlap."""
import unittest
import numpy as np


class TestComputeOverlapRate(unittest.TestCase):
    def test_fully_overlapping_clouds(self):
        from test_scanned_vs_master import compute_overlap_rate
        pc = np.random.RandomState(0).randn(100, 3)
        rate = compute_overlap_rate(pc, pc, threshold=1e-6)
        self.assertAlmostEqual(rate, 1.0)

    def test_fully_disjoint_clouds(self):
        from test_scanned_vs_master import compute_overlap_rate
        rng = np.random.RandomState(0)
        a = rng.randn(100, 3)
        b = rng.randn(100, 3) + 1000.0
        rate = compute_overlap_rate(a, b, threshold=1.0)
        self.assertAlmostEqual(rate, 0.0)

    def test_half_overlap(self):
        from test_scanned_vs_master import compute_overlap_rate
        rng = np.random.RandomState(0)
        master = rng.randn(100, 3)
        # 50 points identical to master, 50 points far away
        aligned = np.vstack([master[:50], rng.randn(50, 3) + 1000.0])
        rate = compute_overlap_rate(master, aligned, threshold=1e-6)
        self.assertAlmostEqual(rate, 0.5)

    def test_threshold_matters(self):
        from test_scanned_vs_master import compute_overlap_rate
        master = np.array([[0.0, 0.0, 0.0]])
        aligned = np.array([[0.5, 0.0, 0.0]])
        self.assertAlmostEqual(
            compute_overlap_rate(master, aligned, threshold=0.1), 0.0
        )
        self.assertAlmostEqual(
            compute_overlap_rate(master, aligned, threshold=1.0), 1.0
        )

    def test_empty_aligned_returns_zero(self):
        from test_scanned_vs_master import compute_overlap_rate
        master = np.random.randn(100, 3)
        aligned = np.zeros((0, 3))
        rate = compute_overlap_rate(master, aligned, threshold=1.0)
        self.assertEqual(rate, 0.0)


if __name__ == "__main__":
    unittest.main()
