# fraktalka

**Validation-free structural diagnostics for representations** — a small, honest
research library. The question it exists to answer, before anything is built on top:

> Does an **intrinsic** fractal / spectral signal, computed from a model's own weights
> (no held-out labels, no test set), predict its **generalization** better than simple
> baselines?

If yes, there is something here. If no, that is the honest result — and it is better
to find out in the first commit than after building a product on it. So this **Layer 0**
release is *eval-first*: the diagnostics plus a falsification harness, nothing else.

A signal is a signal, **never a proof**. Assurance is never raised to `VERIFIED`.

## Status (honest)

The harness works and is deterministic, and it now runs the **fair** test of the core
claim: the "fractal dimension predicts generalization" idea is about the *training
trajectory* (the sequence of weight updates), not the static final weights — so the
harness computes a **trajectory** signal (DFA of a 1D projection of the optimisation
path) alongside static-weight signals and baselines.

**Synthetic pilot:** inconclusive — every signal had weak ranking power (|Kendall tau|
< ~0.25) and the winner flipped across seeds, because the synthetic gaps were too
uniform to rank.

**Standard benchmark (scikit-learn digits, wide gap range via label noise + subsampling):**
early and **cautiously encouraging**. A *fractal-family* signal beats the weight-norm
baselines in most seeds, with |tau| up to ~0.5 — clearly stronger than the synthetic
pilot. The standout is **static self-similarity (`ss_rate`)**, consistently among the
best predictors; the trajectory DFA is strongest in some seeds and weak in others.
**But which fractal signal wins is seed-dependent, and this is one small benchmark.**
**Multi-seed confirmation (30 seeds x 30 models, exact sign test):** the strongest
signal is **static self-similarity (`ss_rate`)** — the *same sign in all 30/30 seeds*
(|mean tau| ~= 0.32) and it beats the strongest weight-norm baseline in **29/30 seeds
(p < 0.001)**. The trajectory DFA signals also reliably beat the baseline (p < 0.001 and
p = 0.04). Notably, the *raw spectral slope d_F was noise* (sign-test p = 0.36): if
anything carries here it is **self-similarity, not the Fourier fractal dimension** the
source material emphasised. The data named a different carrier than the hype.

**Two honest caveats that keep this a hypothesis, not a result:**
1. `ss_rate` and the trajectory DFA were noticed *post-hoc on the same data*, so this is
   confirmation-on-selected; the significance is inflated by the selection. A clean
   confirmation needs a **fresh, pre-registered replication on a different dataset**.
2. One dataset (digits), one model family (a tiny MLP), one gap mechanism (label noise +
   subsampling). Generalising beyond this needs more.

So: a **statistically significant, baseline-beating signal exists here** — but it is
selected and singular. **Promising, not proven.** Run:

```bash
pip install numpy scikit-learn
PYTHONPATH=src python -m fraktalka.eval.benchmark --seed 0 --models 40
PYTHONPATH=src python -m fraktalka.eval.multiseed --seeds 30 --models 30
```

## Run it

```bash
pip install numpy
PYTHONPATH=src python -m fraktalka.eval --seed 0 --models 24
PYTHONPATH=src python -m unittest discover -s tests
```

The eval prints, for each candidate signal, its Kendall tau against the *measured*
generalization gap (the test set establishes ground truth only; the signals never see
it), and whether the fractal signal beats the weight-norm baselines.

## What's here (Layer 0)

| Path | What |
|---|---|
| `src/fraktalka/diagnostics.py` | intrinsic signals: `spectral_slope` (2D radial power spectrum), `dfa_exponent` (1D DFA), `ss_rate` (cross-layer self-similarity), and norm baselines |
| `src/fraktalka/eval/` | the falsification harness: train real tiny models (recording the training trajectory), measure the gap, correlate each signal — trajectory d_F, static-weight signals, and baselines — against it |
| `src/fraktalka/stats.py` | `kendall_tau` (dependency-free) |
| `tests/` | deterministic property tests (DFA of white noise ~0.5, of a random walk >1, etc.) + harness smoke |

Roadmap (only if Layer 0 earns it): Matryoshka embeddings + coarse-to-fine retrieval;
a Lévy-Fourier optimizer; a drift monitor; an API used by other systems as an external
substrate.

## Discipline (borrowed, not the files)

This repo carries the *method* of the KRISTA/FRR systems, not their Markdown: outputs
are labelled as signals (`DECLARED`), never proofs; the README states its own limits;
results take a verdict shape (`ACHIEVED` only when earned, else `PARTIAL`/`UNVERIFIABLE`);
and the load-bearing claim is pre-registered and tested, not asserted.

See [docs/LINEAGE.md](docs/LINEAGE.md) for where each idea comes from.

## Not a promise

fraktalka does not promise that intrinsic signals predict generalization. It gives an
honest, reusable way to find out — and reports the answer whichever way it falls.
