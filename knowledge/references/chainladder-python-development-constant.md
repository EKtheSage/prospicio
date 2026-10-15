---
type: Reference Implementation
title: Selected development patterns in chainladder-python and R ChainLadder
description: chainladder-python 0.10.1's DevelopmentConstant (style ldf or cdf, no blend) is the parity reference for prospicio_reserving's SelectedPattern; a partial pattern gets factors of 1, a cdf pattern's level at the oldest age stays on as a tail unless a TailConstant replaces it (prospicio does the same since 2026-10-15), and MackChainladder rejects its output. R ChainLadder 0.2.21's MackChainLadder takes no factors.
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
