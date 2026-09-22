"""Multi-seed statistical confirmation of the benchmark result.

One benchmark on a few seeds cannot tell a real predictor from a lucky one. This runs
the digits benchmark across many seeds, collects each signal's per-seed Kendall tau,
and asks two questions with an exact, dependency-free sign test:

1. Is a signal *sign-consistent* across seeds (a real predictor, not sign-flipping)?
2. Does a fractal signal beat a fixed baseline *reliably* (paired, per seed)?

Honesty on selection: `ss_rate` and the trajectory DFA were noticed as strong in an
earlier exploratory run on the same data. Testing them again here is therefore
partly post-hoc; a clean confirmation would pre-register them and replicate on a
fresh dataset. We report all signals and flag this, rather than quietly cherry-picking.

Requires scikit-learn (dataset). Run:
    python -m fraktalka.eval.multiseed --seeds 30 --models 30
"""

from __future__ import annotations

import numpy as np

from ..stats import sign_test
from .benchmark import run_benchmark

# The baseline each fractal signal is compared against (fixed in advance of the run).
_FIXED_BASELINE = "spectral_norm[baseline]"
_FRACTAL_OF_INTEREST = ["ss_rate_static[FRACTAL]", "dfa_traj_proj[FRACTAL-TRAJ]",
                        "dfa_traj_updates[FRACTAL-TRAJ]"]


def run_multiseed(n_seeds: int = 30, n_models: int = 30, start_seed: int = 100) -> dict:
    seeds = list(range(start_seed, start_seed + n_seeds))
    per_signal: dict[str, list[float]] = {}
    for s in seeds:
        taus = run_benchmark(seed=s, n_models=n_models)["tau_vs_gap"]
        for name, tau in taus.items():
            per_signal.setdefault(name, []).append(tau)

    summary = {}
    for name, vals in per_signal.items():
        arr = np.array(vals, dtype=float)
        finite = arr[~np.isnan(arr)]
        st = sign_test(arr)
        summary[name] = {
            "mean_tau": float(np.mean(finite)) if finite.size else float("nan"),
            "std_tau": float(np.std(finite)) if finite.size else float("nan"),
            "mean_abs_tau": float(np.mean(np.abs(finite))) if finite.size else float("nan"),
            "sign_consistent_p": st["p"], "pos": st["pos"], "neg": st["neg"],
        }

    # Paired: does each fractal-of-interest beat the fixed baseline, per seed, on |tau|?
    base = np.array(per_signal.get(_FIXED_BASELINE, []), dtype=float)
    paired = {}
    for name in _FRACTAL_OF_INTEREST:
        if name not in per_signal:
            continue
        fr = np.array(per_signal[name], dtype=float)
        diff = np.abs(fr) - np.abs(base)  # >0 where fractal has more ranking power
        st = sign_test(diff)
        paired[name] = {"wins": st["pos"], "losses": st["neg"], "p": st["p"],
                        "mean_diff": float(np.nanmean(diff))}

    return {"n_seeds": n_seeds, "n_models": n_models, "fixed_baseline": _FIXED_BASELINE,
            "per_signal": summary, "paired_vs_baseline": paired}


def format_report(result: dict) -> str:
    lines = [
        f"fraktalka - multi-seed confirmation ({result['n_seeds']} seeds x "
        f"{result['n_models']} models, digits)",
        "",
        "Per-signal Kendall tau across seeds (sign test: is it a consistent predictor?):",
        f"  {'signal':<34} {'mean':>7} {'std':>6} {'|mean|':>7} {'+/-':>7} {'sign-p':>7}",
    ]
    for name, s in sorted(result["per_signal"].items(), key=lambda kv: -kv[1]["mean_abs_tau"]):
        ratio = f"{s['pos']}/{s['neg']}"
        lines.append(f"  {name:<34} {s['mean_tau']:>7.3f} {s['std_tau']:>6.3f} "
                     f"{s['mean_abs_tau']:>7.3f} {ratio:>7} {s['sign_consistent_p']:>7.3f}")
    lines += ["", f"Paired sign test vs {result['fixed_baseline']} (does fractal beat it per seed?):"]
    for name, p in result["paired_vs_baseline"].items():
        verdict = "reliable" if p["p"] < 0.05 and p["wins"] > p["losses"] else "not reliable"
        lines.append(f"  {name:<34} wins {p['wins']:>2} / losses {p['losses']:>2}  "
                     f"mean_diff={p['mean_diff']:+.3f}  p={p['p']:.3f}  -> {verdict}")
    lines += [
        "",
        "NOTE (honesty): ss_rate and trajectory DFA were noticed post-hoc on the same",
        "data, so these tests are partly confirmatory-on-selected; a fresh pre-registered",
        "replication would be needed to be sure. Assurance is DECLARED, never VERIFIED.",
        "A signal is a signal, never a proof.",
    ]
    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="fraktalka multi-seed confirmation")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--models", type=int, default=30)
    args = parser.parse_args()
    print(format_report(run_multiseed(n_seeds=args.seeds, n_models=args.models)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
