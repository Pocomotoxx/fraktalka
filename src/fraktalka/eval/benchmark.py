"""The standard-benchmark version of the make-or-break test.

Same question as the synthetic run, on a real, recognised dataset (scikit-learn's
handwritten digits): does an intrinsic fractal signal — trajectory or static — predict
the generalization gap better than weight-norm baselines? We deliberately create a wide
range of gaps by varying capacity, regularization, training-set size, and label noise,
so there is real spread to rank (the synthetic pilot's gaps were too uniform to tell).

Requires scikit-learn (only for the dataset). Run:
    python -m fraktalka.eval.benchmark [--seed N] [--models N]

Still a signal, never a proof; assurance never reaches VERIFIED.
"""

from __future__ import annotations

import numpy as np

from ..diagnostics import ss_rate  # noqa: F401  (kept for parity/imports)
from .generalization import _model_signals, aggregate, format_report
from .tiny_mlp import TinyMLP


def load_digits_split(rng: np.random.Generator, n_test: int = 500):
    from sklearn.datasets import load_digits  # lazy: dataset dependency only

    d = load_digits()
    x = (d.data.astype(float) / 16.0)  # pixels 0..16 -> 0..1
    y = d.target.astype(int)
    perm = rng.permutation(len(x))
    x, y = x[perm], y[perm]
    x_te, y_te = x[:n_test], y[:n_test]          # fixed clean test = ground truth
    x_pool, y_pool = x[n_test:], y[n_test:]      # training pool
    return (x_pool, y_pool), (x_te, y_te), x.shape[1], int(y.max()) + 1


def _inject_label_noise(y: np.ndarray, noise: float, n_classes: int, rng: np.random.Generator):
    if noise <= 0:
        return y
    y = y.copy()
    flip = rng.random(len(y)) < noise
    y[flip] = rng.integers(0, n_classes, size=int(flip.sum()))
    return y


def run_benchmark(seed: int = 0, n_models: int = 30) -> dict:
    rng = np.random.default_rng(seed)
    (x_pool, y_pool), (x_te, y_te), dim, n_classes = load_digits_split(rng)

    gaps: list[float] = []
    signal_rows: list[dict] = []
    for _ in range(n_models):
        hidden = int(rng.choice([16, 32, 64, 128, 256]))
        epochs = int(rng.choice([15, 30, 60]))
        wd = float(rng.choice([0.0, 1e-4, 1e-3, 1e-2]))
        noise = float(rng.choice([0.0, 0.1, 0.2, 0.4]))          # spreads the gap
        n_train = int(rng.choice([128, 256, 512]))               # small data -> bigger gap

        idx = rng.permutation(len(x_pool))[:n_train]
        x_tr = x_pool[idx]
        y_tr = _inject_label_noise(y_pool[idx], noise, n_classes, rng)

        model = TinyMLP(dim, hidden, n_classes, rng)
        model.train(x_tr, y_tr, epochs=epochs, lr=0.1, weight_decay=wd, rng=rng,
                    record_trajectory=True)

        gap = model.accuracy(x_tr, y_tr) - model.accuracy(x_te, y_te)
        gaps.append(gap)
        signal_rows.append(_model_signals(model, x_tr[:64]))

    result = aggregate(signal_rows, np.array(gaps), n_models)
    result["dataset"] = "sklearn-digits"
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="fraktalka standard-benchmark harness (digits)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", type=int, default=30)
    args = parser.parse_args()
    result = run_benchmark(seed=args.seed, n_models=args.models)
    print("STANDARD BENCHMARK: scikit-learn handwritten digits (1797x64, 10 classes)")
    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
