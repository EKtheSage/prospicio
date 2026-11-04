---
type: Reference
title: chainladder-python's VotingChainladder
description: chainladder-python 0.10.1's VotingChainladder weights several methods' ultimates by origin and divides by the weights' sum, so weights that do not sum to 1 are rescaled; the parity reference for prospicio_reserving's UltimateExhibit::select, which refuses them instead; 156 GenIns values match to 1e-9 in CI.
resource: https://github.com/casact/chainladder-python/blob/v0.10.1/chainladder/workflow/voting.py
tags: [reserving, chainladder-python, parity, selection, exhibits]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-02T12:00:00Z }
verified: { by: process:ci, at: 2026-11-02T12:00:00Z }
sources:
  - id: voting
    resource: https://github.com/casact/chainladder-python/blob/v0.10.1/chainladder/workflow/voting.py
    title: chainladder-python 0.10.1, chainladder/workflow/voting.py
  - id: script
    resource: ../validation/scripts/reserving_exhibits_python.py
    title: reserving_exhibits_python.py, which writes validation/reference/reserving_exhibits_python.csv
  - id: test
    resource: ../validation/tests/reserving_exhibits.rs
    title: The parity test, voting_chainladder_parity
---

# What it is

`VotingChainladder(estimators=[(name, method), ...], weights=...,
default_weighting=...)` fits each method on the same triangle and
`sample_weight` (the exposure, shared by every method) and returns, per
origin, the weighted average of their ultimates. prospicio's
`UltimateExhibit::select` with `MethodWeights::ByOrigin` reproduces it
(`docs/design/reserving-v02.md`, decision 34).[^voting]

# Conventions that matter for parity

- **Weights of shape (origin, n_estimators).** A numpy array or a list of
  rows, one row per origin and one weight per method in the estimators'
  order. A dict maps an origin to a row; origins it leaves out, and
  `weights=None`, take `default_weighting` (by default 1 for every method,
  an equal blend).[^voting]
- **Rescaled by the sum.** The ultimate is the weighted sum divided by the
  sum of the origin's weights, so `(2, 1, 1)` is the same as
  `(0.5, 0.25, 0.25)`. prospicio refuses weights that do not sum to 1
  within 1e-9 (decision 34, item 8, open).[^voting]
- **Missing ultimates.** The weighted sum multiplies every method's
  ultimate, so a NaN ultimate with weight 0 still gives NaN; prospicio
  skips a method with weight 0. Read from the source, not run.[^voting]
- **The reference run.** The chain ladder, Bornhuetter-Ferguson with
  apriori 0.75 and Cape Cod on GenIns paid with the latest premium as
  `sample_weight`: phased by origin, a blend of (0.5, 0.25, 0.25) and the
  BF alone, each origin's method ultimates, selected ultimate and reserve,
  and the totals. All 156 values match prospicio to 1e-9.[^script][^test]

[^voting]: chainladder-python 0.10.1, `chainladder/workflow/voting.py`.
[^script]: `validation/scripts/reserving_exhibits_python.py`.
[^test]: `validation/tests/reserving_exhibits.rs`, `voting_chainladder_parity`, and the Python and R tests.
