---
type: Finding
title: A bootstrap's development pattern per simulation falls somewhere in about half of RAA's ODP simulations, so the cash-flow engine cannot take it unchanged
description: Each simulation's cumulative shares from the ODP or Mack bootstrap's pseudo factors fall at some age in 54% (ODP) and 22% (Mack) of 5,000 simulations on RAA and 10% and 2.5% on GenIns, because a pseudo factor at an old age can be below 1; the engine needs a non-decreasing paid pattern, so DevelopmentPatterns::monthly refuses by default and holds the running maximum on request. The Bayesian models' patterns never fall. Adding the pattern columns to the simulated row leaves every fit's reserves bit for bit unchanged.
tags: [reserving, bootstrap, payment-pattern, cashflow, simulation]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-28T12:00:00Z }
sources:
  - id: note
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 28, a development pattern per simulation
  - id: test
    resource: ../validation/tests/reserving_pattern_draws.rs
    title: The bootstraps' pattern draws, run by CI's Rust job
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/600
    title: Issue 600, paid patterns that vary by simulation in the cash-flow engine
---

# What was found

A simulation's development pattern is the cumulative share of the ultimate
reached at each age under the factors that simulation drew:
`1 / (f_k f_{k+1} ... f_{n-1} tail)`. In the bootstraps the factors are
pseudo factors, refitted on a resampled triangle, and at the old ages of a
nearly closed triangle a pseudo factor can be below 1. The share then falls
at that age.

Share of simulations whose pattern falls at some age (5,000 simulations,
seed 1, default settings, measured by hand on 2026-10-28):

| Triangle | ODP bootstrap | Mack bootstrap |
|---|---|---|
| RAA | 54.2% | 22.4% |
| GenIns | 9.6% | 2.5% |

The cash-flow engine (`PatternDraws`) refuses a paid pattern that falls,
so `DevelopmentPatterns::monthly` refuses one by default, naming how many
simulations fall and the first, and `Monotone::RunningMax` holds each
share at the largest before it, capped at 1. On RAA's ODP bootstrap the
default refuses almost every run of more than a few simulations.

Mack's bootstrap draws its factors from normal distributions, and on RAA
it draws a negative first factor in 6 of 5,000 simulations at seed 600 and
8 at seed 1, which makes the share at 12 months negative (-12 and -20 in
the first such simulations). The default refuses that too, and
`RunningMax` gives a share that is not positive the next positive share,
so nothing is paid until then.

`BayesOdp`, `BayesBf` and `BayesBfNormalised` build their patterns from
positive increments (`exp` of the column effects or the log shares), so
their patterns never fall.

# Other facts confirmed

* Appending the pattern columns to the simulated row draws no random
  number, so the reserves of the ODP and Mack bootstraps (one fit, a tail
  with a standard error, rank-correlated and synchronized segments),
  `BayesOdp` (one fit and rank-correlated segments), `BayesBf` and
  `BayesBfNormalised` hashed bit for bit equal to `main`'s before the
  change (checked by hand; the CI tests check the patterns, not the
  hashes).
* A simulation's reserve for the youngest origin correlates with its
  factor to ultimate at about 0.39 in the ODP bootstrap without process
  error and 0.28 in Mack's bootstrap with it (2,000 simulations on RAA),
  and with the next simulation's at under 0.1; the test asserts 0.3 and
  0.2.
* Mack's bootstrap draws a constant tail's factor with the tail's sigma,
  so its share at the oldest age varies about `1 / tail`; the ODP
  bootstrap keeps it fixed at exactly `1 / tail`.
