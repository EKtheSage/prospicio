---
type: Reference Implementation
title: Selected development patterns in chainladder-python and R ChainLadder
description: chainladder-python 0.10.1's DevelopmentConstant (style ldf or cdf, no blend) is the parity reference for prospicio_reserving's SelectedPattern; a partial pattern gets factors of 1, a cdf pattern's level at the oldest age stays on as a tail unless a TailConstant replaces it (prospicio does the same since 2026-10-15), MackChainladder rejects its output, and on a latest diagonal it matches the full triangle but gives an unlisted age a factor of 1. R ChainLadder 0.2.21's MackChainLadder takes no factors.
resource: https://github.com/casact/chainladder-python
tags: [reserving, development, pattern, chain-ladder, mack, parity, python, r]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-15T12:00:00-07:00 }
sources:
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 12, selected development patterns
  - id: script
    resource: ../validation/scripts/reserving_patterns_python.py
    title: chainladder-python reference generator for selected patterns
  - id: test
    resource: ../validation/tests/reserving_patterns.rs
    title: Validation test of selected patterns against chainladder-python and R
  - id: diagonal
    resource: ../validation/tests/reserving_latest_diagonal.rs
    title: Validation test of a latest diagonal with a selected pattern, against chainladder-python (from validation/scripts/reserving_latest_diagonal_python.py)
---

# Use

`DevelopmentConstant(patterns, style, callable_axis)` replaces the
estimated development of a triangle by a pattern given by age, as
age-to-age factors (`style="ldf"`) or factors to ultimate (`style="cdf"`).
It is the reference for `prospicio_reserving::SelectedPattern` with a
weight of 1 on RAA and GenIns with the chain ladder, and on the CAS
workers' compensation triangle with the four expected-loss methods.[^script][^test]
Checked on 2026-10-14 with chainladder-python 0.10.1 and R ChainLadder
0.2.21.[^design]

# Behaviour

* **No blend.** A pattern replaces the estimate; there is no weight. The
  parity script blends itself and passes the blended factors, so its
  blended rows check how blended factors reach each method, not the blend.
* **Partial patterns.** An age the pattern does not give gets a factor of
  1, not the estimate. prospicio keeps the estimate before the first and
  after the last given age, and interpolates a cumulative pattern's
  skipped ages (`ln CDF` linear in age), so partial and gapped patterns are
  not comparable and are checked by hand.
* **The level of a cdf pattern.** With `style="cdf"` the factors are the
  ratios of consecutive factors to ultimate, and the value at the oldest
  age stays on as one more factor past it (a tail) when no tail follows.
  `TailConstant(tail=t)` after it replaces that level by `t`. prospicio
  does the same (decided by the user on 2026-10-15): with no tail, a
  cumulative pattern to the oldest age keeps its level there as a constant
  tail (`ChainLadderFit::pattern_tail`), and with a tail the level is
  rescaled to it (`pattern_rescale`). The two agree on all three cases
  (level 1, level 1.03 with no tail, level 1.03 then `TailConstant(1.05)`),
  checked at 1e-9 relative on RAA, GenIns and the four expected-loss
  methods on the CAS workers' compensation triangle. Before 2026-10-15
  prospicio rescaled the level to 100% with no tail and differed for a
  level above 1.
* **Per segment.** `callable_axis` gives a pattern per index level through
  a callable; prospicio applies one pattern to every segment.
* **On a latest diagonal.** `DevelopmentConstant` runs on a triangle's
  `latest_diagonal` (one valuation, one cell per origin), and the
  generator records the chain ladder on GenIns and Bornhuetter-Ferguson,
  Benktander and Cape Cod on the CAS workers' compensation triangle there,
  which prospicio's `LatestDiagonal` reproduces. With `style="ldf"` the pattern needs a factor of 1 at the oldest
  age, or the oldest origin's ultimate is NaN; prospicio's `LatestDiagonal`
  reads that factor as running one spacing past the oldest age, so the
  same pattern gives the same ultimates (decision 16 of
  `reserving-v02.md`).[^diagonal] The rest was a hand probe of 0.10.1 on
  2026-10-23, not kept in a script. chainladder-python has no ragged diagonal: a
  triangle has one valuation, so each origin's age is the valuation less
  its start; an age the pattern does not give (18 months against an
  annual pattern on a half-year grain) silently gets a factor of 1, with
  no interpolation or regrain, and a diagonal with a gap in its origins
  drops an origin. `Development().fit(full).transform(diagonal)` applies
  a fitted development to the diagonal (the oldest origin is NaN with no
  tail); there is no `predict`. So there
  is no reference for off-grid ages, and prospicio's log-linear reading
  of them is checked by hand.
* **Mack.** `MackChainladder().fit(DevelopmentConstant(...).fit_transform(tri))`
  fails with "Triangle not compatible with MackChainladder", so there is no
  reference for Mack with a selected pattern.
* **R ChainLadder.** `MackChainLadder(Triangle, weights, alpha, est.sigma,
  tail, tail.se, tail.sigma, mse.method)` takes no factors; `weights` only
  drops link ratios. The R parity of Mack with a pattern therefore selects
  the triangle's own factors, where R's process and parameter risks are the
  targets (fully selected: R's process risk and no parameter risk; weight
  `w`: `(1 - w)` times R's parameter risk).

[^design]: Design note, reserving v0.2, decision 12, selected development patterns
[^script]: chainladder-python reference generator for selected patterns
[^test]: Validation test of selected patterns against chainladder-python and R
[^diagonal]: Validation test of a latest diagonal with a selected pattern, against chainladder-python (from validation/scripts/reserving_latest_diagonal_python.py)
