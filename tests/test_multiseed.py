"""Smoke tests for the multi-seed statistical confirmation (skipped without sklearn)."""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraktalka.stats import binom_two_sided, sign_test

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


class SignTestStats(unittest.TestCase):
    def test_binom_two_sided_bounds(self):
        self.assertAlmostEqual(binom_two_sided(5, 10), 1.0, places=6)   # exactly balanced
        self.assertLess(binom_two_sided(10, 10), 0.01)                  # all one way
        self.assertEqual(binom_two_sided(0, 0), 1.0)

    def test_sign_test_consistent_is_significant(self):
        import numpy as np
        st = sign_test(np.array([0.2] * 12 + [-0.1]))  # 12 pos, 1 neg
        self.assertLess(st["p"], 0.05)
        self.assertEqual((st["pos"], st["neg"]), (12, 1))


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn not installed")
class MultiseedSmoke(unittest.TestCase):
    def test_runs_and_reports(self):
        from fraktalka.eval.multiseed import format_report, run_multiseed

        r = run_multiseed(n_seeds=3, n_models=8)
        self.assertEqual(r["n_seeds"], 3)
        self.assertIn("per_signal", r)
        self.assertIn("paired_vs_baseline", r)
        self.assertIn("honesty", format_report(r).lower())


if __name__ == "__main__":
    unittest.main()
