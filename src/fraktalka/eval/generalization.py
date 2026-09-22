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
    # Static (final-weight) signals.
    slope = np.nanmean([dg.spectral_slope(model.W1), dg.spectral_slope(model.W2)])
    fro = dg.frobenius_norm(model.W1) + dg.frobenius_norm(model.W2)
    spec = dg.spectral_norm(model.W1) + dg.spectral_norm(model.W2)
    mabs = dg.mean_abs(model.W1) + dg.mean_abs(model.W2)
    ss = dg.ss_rate(model.activations(x_probe))
    # Trajectory (training-path) signals — the quantity the d_F claim is really about.
    traj = getattr(model, "trajectory", None) or {}
    dfa_proj = dg.dfa_exponent(traj["projection"]) if "projection" in traj else float("nan")
    dfa_upd = dg.dfa_exponent(traj["update_norms"]) if "update_norms" in traj else float("nan")
    return {
        "dfa_traj_proj[FRACTAL-TRAJ]": float(dfa_proj),
        "dfa_traj_updates[FRACTAL-TRAJ]": float(dfa_upd),
        "spectral_slope_static[FRACTAL]": float(slope),
        "ss_rate_static[FRACTAL]": float(ss),
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
        # Enough epochs that every training path is long enough for DFA on the trajectory.
        epochs = int(rng.choice([12, 25, 50]))
        wd = float(rng.choice([0.0, 1e-3, 1e-2]))
        noise = float(rng.choice([0.0, 0.1, 0.3]))
        n_train = int(rng.choice([128, 256]))

        x_tr, y_tr = make_classification(n_train, dim, n_classes, noise, rng)
        x_te, y_te = make_classification(400, dim, n_classes, 0.0, rng)  # clean test = ground truth

        model = TinyMLP(dim, hidden, n_classes, rng)
        model.train(x_tr, y_tr, epochs=epochs, lr=0.1, weight_decay=wd, rng=rng,
                    record_trajectory=True)

        gap = model.accuracy(x_tr, y_tr) - model.accuracy(x_te, y_te)  # generalization gap
        gaps.append(gap)
        signal_rows.append(_model_signals(model, x_tr[:64]))

    gaps_arr = np.array(gaps)
    names = list(signal_rows[0].keys())
    taus = {}
    for name in names:
        col = np.array([row[name] for row in signal_rows])
        keep = ~np.isnan(col)  # a signal is scored only where it is defined
        taus[name] = kendall_tau(col[keep], gaps_arr[keep]) if keep.sum() >= 3 else float("nan")

    def _best(group: dict) -> tuple[str, float]:
        usable = {k: v for k, v in group.items() if v == v}  # drop NaN
        if not usable:
            return ("none", float("nan"))
        k = max(usable, key=lambda k: abs(usable[k]))
        return (k, float(usable[k]))

    traj = {k: v for k, v in taus.items() if "FRACTAL-TRAJ" in k}
    static = {k: v for k, v in taus.items() if "FRACTAL" in k and "TRAJ" not in k}
    baseline = {k: v for k, v in taus.items() if "baseline" in k}
    best_traj, best_static, best_base = _best(traj), _best(static), _best(baseline)

    def _abs(x):
        return abs(x) if x == x else -1.0

    return {
        "n_models": n_models,
        "gap_mean": float(gaps_arr.mean()),
        "gap_std": float(gaps_arr.std()),
        "tau_vs_gap": taus,
        "best_trajectory": best_traj,
        "best_static": best_static,
        "best_baseline": best_base,
        "trajectory_beats_baseline": bool(_abs(best_traj[1]) > _abs(best_base[1])),
        "trajectory_beats_static": bool(_abs(best_traj[1]) > _abs(best_static[1])),
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
    for name, tau in sorted(result["tau_vs_gap"].items(), key=lambda kv: -(abs(kv[1]) if kv[1] == kv[1] else -1)):
        val = "nan" if tau != tau else f"{tau:+.3f}"
        amp = "nan" if tau != tau else f"{abs(tau):.3f}"
        lines.append(f"  {name:<34} tau = {val}   |tau| = {amp}")

    def _fmt(pair):
        name, v = pair
        return f"{name} (|tau|={'nan' if v != v else f'{abs(v):.3f}'})"

    lines += [
        "",
        f"best TRAJECTORY signal: {_fmt(result['best_trajectory'])}",
        f"best static signal    : {_fmt(result['best_static'])}",
        f"best baseline         : {_fmt(result['best_baseline'])}",
        "",
        "VERDICT (assurance=DECLARED, pilot only, UNVERIFIABLE beyond this synthetic run):",
        f"  trajectory d_F beats baseline: {result['trajectory_beats_baseline']}",
        f"  trajectory d_F beats static  : {result['trajectory_beats_static']}",
        "  -> This is the fair test of the 'd_F predicts generalization' claim: it is",
        "     about the training TRAJECTORY, not the static final weights. Whichever way",
        "     it falls on real benchmarks is the answer. A signal is a signal, never a proof.",
    ]
    return "\n".join(lines)
