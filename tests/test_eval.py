"""Smoke tests for the falsification harness: it must run, be deterministic, and
report an honest, bounded result (never claim VERIFIED)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraktalka.eval.generalization import format_report, run


class EvalHarnessTests(unittest.TestCase):
    def test_run_is_deterministic_and_well_formed(self):
        r1 = run(seed=0, n_models=8)
        r2 = run(seed=0, n_models=8)
        self.assertEqual(r1["tau_vs_gap"], r2["tau_vs_gap"])  # deterministic
        self.assertEqual(r1["n_models"], 8)
        self.assertIn("dfa_traj_proj[FRACTAL-TRAJ]", r1["tau_vs_gap"])  # the trajectory signal
        self.assertIn("spectral_slope_static[FRACTAL]", r1["tau_vs_gap"])
        self.assertIn("frobenius_norm[baseline]", r1["tau_vs_gap"])

    def test_trajectory_signal_is_defined(self):
        # With the longer training schedule, the trajectory DFA must be computable
        # (not all-NaN) for at least most models.
        r = run(seed=0, n_models=8)
        name, tau = r["best_trajectory"]
        self.assertNotEqual(name, "none", "trajectory signal never computed")

    def test_taus_are_bounded(self):
        r = run(seed=1, n_models=8)
        for name, tau in r["tau_vs_gap"].items():
            if tau == tau:  # not NaN
                self.assertGreaterEqual(tau, -1.0 - 1e-9, name)
                self.assertLessEqual(tau, 1.0 + 1e-9, name)

    def test_assurance_is_never_verified(self):
        r = run(seed=0, n_models=8)
        self.assertNotEqual(r["assurance_level"], "VERIFIED")

    def test_report_states_it_is_not_proof(self):
        text = format_report(run(seed=0, n_models=8)).lower()
        self.assertIn("never a proof", text)
        self.assertIn("unverifiable", text)

    def test_models_have_measurable_gap_variance(self):
        # The experiment is only meaningful if the models actually differ; guard it.
        r = run(seed=2, n_models=16)
        self.assertGreater(r["gap_std"], 0.0)


if __name__ == "__main__":
    unittest.main()
