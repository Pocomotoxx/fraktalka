# Lineage

Where each idea in fraktalka comes from, honestly. None of the ingredients is novel by
itself; the only thing that could be is whether the *intrinsic signal actually predicts
generalization/drift* — which is why the repo tests that first and claims nothing until
it does.

## Ingredients (prior art, adapted)

- **Fractal / spectral dimension as a generalization proxy** — the seed idea (from the
  user's Agent-OS spec/prototype): a weight tensor's spectral structure may correlate
  with the generalization gap. This is a research thread, correlational and
  dataset-dependent; fraktalka treats it as a hypothesis to falsify, not a fact. Note
  that the prototype's `d_F = (5 - beta)/2` is a 2D-image convention; fraktalka reports
  the raw spectral slope (rank-invariant) and, for 1D series, uses DFA instead.
- **Detrended Fluctuation Analysis (DFA)** — Peng et al. (1994); the standard,
  trend-robust way to estimate a scaling exponent of a 1D series.
- **Matryoshka Representation Learning (MRL)** — Kusupati et al. (2022); coarse-to-fine
  truncatable embeddings. Planned for a later layer, using a provider that supports it,
  not reinvented.
- **Lévy-driven / heavy-tailed optimization** — a known research thread; planned as an
  experimental optimizer only if the diagnostics earn a product around them.
- **Self-similarity across layers (SS-rate)** — a Gram-matrix similarity proxy;
  implemented here as a candidate signal.

## Discipline (from KRISTA / FRR)

The *method*, not the Markdown: evidence-level labelling (a signal is `DECLARED`, never
`VERIFIED`), no-overclaim documentation, verdict-shaped results, and pre-registering the
load-bearing claim before testing it. These sibling systems govern epistemics of
outputs; fraktalka is a separate, composable substrate they may reference.

## What fraktalka adds

Only the integration and — the point — a rigorous, honest **falsification harness** that
measures whether the intrinsic signal beats baselines at predicting real generalization,
and reports the answer whichever way it falls.
