"""fraktalka — validation-free structural diagnostics for representations.

Layer 0 (this release): the diagnostics and the falsification harness that tests
whether an intrinsic fractal/spectral signal predicts real generalization better
than simple baselines. A signal is a signal, never a proof; assurance is never
raised to VERIFIED.
"""

from . import diagnostics, stats

__all__ = ["diagnostics", "stats"]
__version__ = "0.0.1"
