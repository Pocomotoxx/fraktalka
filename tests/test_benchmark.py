"""Standard-benchmark smoke tests. Skipped when scikit-learn is absent (the core
library stays numpy-only; the benchmark needs the dataset)."""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn not installed")
class BenchmarkTests(unittest.TestCase):
    def test_run_benchmark_is_deterministic_and_well_formed(self):
        from fraktalka.eval.benchmark import run_benchmark

        r1 = run_benchmark(seed=0, n_models=8)
        r2 = run_benchmark(seed=0, n_models=8)
        self.assertEqual(r1["tau_vs_gap"], r2["tau_vs_gap"])
        self.assertEqual(r1["dataset"], "sklearn-digits")
        self.assertIn("best_fractal_any", r1)
        self.assertIn("dfa_traj_proj[FRACTAL-TRAJ]", r1["tau_vs_gap"])

    def test_gap_has_real_spread(self):
        from fraktalka.eval.benchmark import run_benchmark

        r = run_benchmark(seed=1, n_models=12)
        self.assertGreater(r["gap_std"], 0.02, "benchmark must produce a spread of gaps")

    def test_assurance_never_verified(self):
        from fraktalka.eval.benchmark import run_benchmark

        self.assertNotEqual(run_benchmark(seed=0, n_models=8)["assurance_level"], "VERIFIED")


if __name__ == "__main__":
    unittest.main()
