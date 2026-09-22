"""The make-or-break experiment.

Train several real (tiny) models with genuinely different generalization gaps, then
ask the only question that matters for this project: does an intrinsic fractal /
spectral signal computed from the weights alone — no held-out data — predict the
real generalization gap better than simple baselines (weight norms)?

We measure each signal's Kendall tau against the *measured* gap (the test set is used
ONLY to establish ground truth, never by the signals). The fractal signal earns its
place only if it beats the baselines. If it does not, that is the honest result, and
it is better to see it here than after building a product on top of it.

Scope: this default run is small and synthetic — a demonstration of the harness and a
first datapoint, not a verdict on real networks. Assurance never reaches VERIFIED; the
conclusion is UNVERIFIABLE beyond this setup. Point the same harness at standard
benchmarks to say more.
"""

from __future__ import annotations

import numpy as np

from .. import diagnostics as dg
from ..stats import kendall_tau
from .tiny_mlp import TinyMLP, make_classification


# Each candidate signal maps a trained model to a scalar, from its weights/activations
# only. `higher_is_more_gap` is our prior on the sign; the report shows the raw tau.
def _model_signals(model: TinyMLP, x_probe: np.ndarray) -> dict[str, float]:
    slope = np.nanmean([dg.spectral_slope(model.W1), dg.spectral_slope(model.W2)])
    fro = dg.frobenius_norm(model.W1) + dg.frobenius_norm(model.W2)
    spec = dg.spectral_norm(model.W1) + dg.spectral_norm(model.W2)
    mabs = dg.mean_abs(model.W1) + dg.mean_abs(model.W2)
    ss = dg.ss_rate(model.activations(x_probe))
    return {
        "spectral_slope[FRACTAL]": float(slope),
        "ss_rate[FRACTAL]": float(ss),
        "frobenius_norm[baseline]": float(fro),
        "spectral_norm[baseline]": float(spec),
        "mean_abs[baseline]": float(mabs),
    }


def run(seed: int = 0, n_models: int = 24, dim: int = 20, n_classes: int = 4) -> dict:
    rng = np.random.default_rng(seed)
    gaps: list[float] = []
    signal_rows: list[dict[str, float]] = []

    for _ in range(n_models):
        hidden = int(rng.choice([8, 16, 32, 64, 128]))
        epochs = int(rng.choice([3, 8, 20]))
        wd = float(rng.choice([0.0, 1e-3, 1e-2]))
        noise = float(rng.choice([0.0, 0.1, 0.3]))
        n_train = int(rng.choice([80, 160, 320]))

        x_tr, y_tr = make_classification(n_train, dim, n_classes, noise, rng)
        x_te, y_te = make_classification(400, dim, n_classes, 0.0, rng)  # clean test = ground truth

        model = TinyMLP(dim, hidden, n_classes, rng)
        model.train(x_tr, y_tr, epochs=epochs, lr=0.1, weight_decay=wd, rng=rng)

        gap = model.accuracy(x_tr, y_tr) - model.accuracy(x_te, y_te)  # generalization gap
        gaps.append(gap)
        signal_rows.append(_model_signals(model, x_tr[:64]))

    gaps_arr = np.array(gaps)
    names = list(signal_rows[0].keys())
    taus = {}
    for name in names:
        col = np.array([row[name] for row in signal_rows])
        # A signal predicts the gap by its ordering; we score |tau| for ranking power.
        taus[name] = kendall_tau(col, gaps_arr)

    fractal = {k: v for k, v in taus.items() if "FRACTAL" in k}
    baseline = {k: v for k, v in taus.items() if "baseline" in k}
    best_fractal = max(fractal, key=lambda k: abs(fractal[k]))
    best_baseline = max(baseline, key=lambda k: abs(baseline[k]))
    beats = abs(fractal[best_fractal]) > abs(baseline[best_baseline])

    return {
        "n_models": n_models,
        "gap_mean": float(gaps_arr.mean()),
        "gap_std": float(gaps_arr.std()),
        "tau_vs_gap": taus,
        "best_fractal": (best_fractal, float(fractal[best_fractal])),
        "best_baseline": (best_baseline, float(baseline[best_baseline])),
        "fractal_beats_baseline": bool(beats),
        "assurance_level": "DECLARED",  # never VERIFIED: a pilot signal, not proof
    }


def format_report(result: dict) -> str:
    lines = [
        "fraktalka - Layer 0 make-or-break: does an intrinsic fractal signal predict",
        "the generalization gap better than weight-norm baselines?",
        "",
        f"models: {result['n_models']}   gap mean+/-std: "
        f"{result['gap_mean']:.3f} +/- {result['gap_std']:.3f}",
        "",
        "Kendall tau vs measured generalization gap (|tau| = ranking power):",
    ]
    for name, tau in sorted(result["tau_vs_gap"].items(), key=lambda kv: -abs(kv[1])):
        val = "nan" if tau != tau else f"{tau:+.3f}"
        lines.append(f"  {name:<28} tau = {val}   |tau| = "
                     f"{'nan' if tau != tau else f'{abs(tau):.3f}'}")
    bf_name, bf = result["best_fractal"]
    bb_name, bb = result["best_baseline"]
    lines += [
        "",
        f"best fractal : {bf_name} (|tau|={abs(bf):.3f})",
        f"best baseline: {bb_name} (|tau|={abs(bb):.3f})",
        "",
        ("VERDICT (assurance=DECLARED, pilot only, UNVERIFIABLE beyond this synthetic run):"),
        (f"  fractal beats baseline: {result['fractal_beats_baseline']}"),
        ("  -> If False on real benchmarks too, the core thesis fails; that is the point"),
        ("     of testing it first. A signal is a signal, never a proof."),
    ]
    return "\n".join(lines)
