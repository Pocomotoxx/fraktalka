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

The harness works and is deterministic. The **first pilot** (small, synthetic MLPs) is
**inconclusive**: no signal — fractal or baseline — strongly predicts the generalization
gap, and which one "wins" flips between random seeds. This is expected at this scale
(the pilot's gap variance is low, so there is little to rank) and is exactly why the
claim is not yet made. A real answer needs standard benchmarks with a wide range of
generalization gaps. **The thesis is a hypothesis until this harness says otherwise.**

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
| `src/fraktalka/eval/` | the falsification harness: train real tiny models, measure the gap, correlate each signal against it |
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
