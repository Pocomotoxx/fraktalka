"""Deterministic property tests for the intrinsic diagnostics."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraktalka import diagnostics as dg
from fraktalka.stats import kendall_tau


class DFATests(unittest.TestCase):
    def test_white_noise_alpha_near_half(self):
        rng = np.random.default_rng(0)
        alpha = dg.dfa_exponent(rng.normal(size=4000))
        self.assertTrue(0.35 < alpha < 0.65, f"white-noise DFA alpha={alpha}")

    def test_random_walk_alpha_above_one(self):
        rng = np.random.default_rng(1)
        walk = np.cumsum(rng.normal(size=4000))
        alpha = dg.dfa_exponent(walk)
        self.assertGreater(alpha, 1.1, f"random-walk DFA alpha={alpha}")

    def test_short_series_returns_nan(self):
        self.assertTrue(np.isnan(dg.dfa_exponent(np.arange(3))))


class SpectralSlopeTests(unittest.TestCase):
    def test_smoother_field_has_larger_slope_than_white_noise(self):
        rng = np.random.default_rng(2)
        white = rng.normal(size=(64, 64))
        # A smooth low-frequency field: white noise blurred by a cumulative sum.
        smooth = np.cumsum(np.cumsum(rng.normal(size=(64, 64)), axis=0), axis=1)
        self.assertGreater(dg.spectral_slope(smooth), dg.spectral_slope(white))

    def test_slope_is_finite_on_a_matrix(self):
        rng = np.random.default_rng(3)
        self.assertTrue(np.isfinite(dg.spectral_slope(rng.normal(size=(32, 40)))))


class SSRateTests(unittest.TestCase):
    def test_identical_layers_are_maximally_self_similar(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(16, 8))
        self.assertAlmostEqual(dg.ss_rate([a, a.copy()]), 1.0, places=5)

    def test_single_layer_is_nan(self):
        self.assertTrue(np.isnan(dg.ss_rate([np.ones((4, 4))])))


class BaselineTests(unittest.TestCase):
    def test_norm_orderings(self):
        small = np.ones((4, 4)) * 0.1
        big = np.ones((4, 4)) * 10.0
        self.assertLess(dg.frobenius_norm(small), dg.frobenius_norm(big))
        self.assertLess(dg.spectral_norm(small), dg.spectral_norm(big))


class KendallTests(unittest.TestCase):
    def test_perfect_and_reversed(self):
        a = np.array([1.0, 2, 3, 4, 5])
        self.assertAlmostEqual(kendall_tau(a, a), 1.0, places=6)
        self.assertAlmostEqual(kendall_tau(a, a[::-1]), -1.0, places=6)

    def test_length_mismatch_is_nan(self):
        self.assertTrue(np.isnan(kendall_tau(np.array([1.0, 2]), np.array([1.0]))))


if __name__ == "__main__":
    unittest.main()
