---
type: Finding
title: A bootstrap's development pattern per simulation falls somewhere in about half of RAA's ODP simulations, so the cash-flow engine cannot take it unchanged
description: Each simulation's cumulative shares from the ODP or Mack bootstrap's pseudo factors fall at some age in 54% (ODP) and 22% (Mack) of 5,000 simulations on RAA and 10% and 2.5% on GenIns, because a pseudo factor at an old age can be below 1; the engine needs a non-decreasing paid pattern, so DevelopmentPatterns::monthly refuses by default and holds the running maximum on request. BayesOdp's, BayesBf's and BayesBfNormalised's patterns never fall; Meyers' CRC, CSR and CAY patterns (exp(beta_d g)) fall somewhere in 98% to 99% of the simulations on Meyers' group 353 paid data (nearly all on incurred) and pass 1 in 70% to 86%. Compartmental's patterns (the ODE's expected cumulative paid over that at the square's last age) never fall nor pass 1 in models 1 and 2 on company 337, as the paid curve rises. Adding the pattern columns to the simulated row leaves every fit's reserves bit for bit unchanged.
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
  - id: meyers-test
    resource: ../crates/prospicio-reserving/src/bayes/meyers.rs
    title: Meyers' unit test each_simulations_pattern_is_its_posterior_draws, run by CI's bayes job
  - id: meyers-issue
    resource: https://github.com/EKtheSage/prospicio/issues/678
    title: Issue 678, a development pattern per simulation from Meyers' CRC, CSR and CAY
  - id: compartmental-test
    resource: ../validation/tests/reserving_bayes_pattern_draws.rs
    title: compartmental_patterns_are_the_paid_curve_and_never_fall, run by CI's bayes job (model 2's compartmental_model_2_patterns_never_fall under RISK_RS_FULL_BAYES)
  - id: compartmental-issue
    resource: https://github.com/EKtheSage/prospicio/issues/679
    title: Issue 679, a development pattern per simulation from Compartmental
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
which that simulation then pays from its first age: on RAA the share
at 24 months, about 0.4, is paid by 12 months.

`BayesOdp`, `BayesBf` and `BayesBfNormalised` build their patterns from
positive increments (`exp` of the column effects or the log shares), so
their patterns never fall.

Meyers' models are different (#678): their pattern is `exp(beta_d g)`,
the median value at each age over the median at the oldest age
(`reserving-v02.md` decision 28 item 7), and nothing in the model keeps
`beta_d` rising or below 0. On Meyers' group 353 Commercial Auto data
(the upper triangle, `Meyers`' default chains, 2,000 simulations, seed
21; measured by hand from Python on 2026-10-28):

| Column | Model | A share falls | A share passes 1 | Largest share |
|---|---|---|---|---|
| paid | CRC | 97.9% | 71.0% | 1.356 |
| paid | CSR | 99.1% | 74.8% | 1.244 |
| paid | CAY | 98.2% | 70.0% | 1.290 |
| incurred | CRC | 99.6% | 81.8% | 1.163 |
| incurred | CSR | 99.8% | 85.0% | 1.137 |
| incurred | CAY | 99.7% | 86.4% | 1.192 |

The falls are mostly from 60 months on, where the columns barely move
and each late `beta_d` rests on one or two origins: on paid CRC, 422,
246, 701, 893, 1,068 and 818 of the 2,000 simulations fall into 60, 72,
84, 96, 108 and 120 months, and at most 13 into the younger ages. The
mean pattern itself falls: on paid CRC it is 0.9932, 0.9961 and 0.9933
at 84, 96 and 108 months. A share above 1 must fall to the last, which
is exactly 1. The default `Monotone::Refuse` therefore refuses nearly
every Meyers fit, and `RunningMax`, which also caps each share at 1, is
the way to use one. The unit test
`each_simulations_pattern_is_its_posterior_draws` (short chains, CRC and
CSR on paid) prints the same picture: 98.7% and 98.4% fall.

`Compartmental`'s patterns do not fall (#679). A simulation's pattern is
the expected cumulative paid of the ODE for its posterior draw at each
age of the square over that at the square's last age (`reserving-v02.md`
decision 28 item 7). Model 1's is `PD(t) / PD(t_last)` at the drawn
rates, and `PD' = k_p OS >= 0`; model 2's is a running sum of
incremental means, each `RLR_t RRF_t (PD_t(t) - PD_t(t - 1))` with that
age's parameters, all positive. On company 337 (the papers' training
triangle, outstanding = incurred - paid), no share falls or passes 1 in
any of 4,000 simulations of model 1 at the default sampler with 2 chains
of 400 and 400 draws (seed 679), or of model 2 at its defaults (seed 681,
under `RISK_RS_FULL_BAYES`), nor in 500 of each from short chains (the
unit test `each_simulations_pattern_is_its_posterior_draws`). Model 1's
mean pattern is 0.197, 0.455, 0.646, 0.775, 0.860, 0.915, 0.951, 0.975,
0.990 and 1 at 12 to 120 months; model 2's is 0.249, 0.556, 0.750,
0.866, 0.928, 0.963, 0.981, 0.991, 0.997 and 1. The mean
paid of 1997 (no observed cell) at each age over its mean at 120 months
is within 0.02 of model 1's mean pattern. Only a cumulative paid response
with development effects on `RLR` or `RRF`, which no preset uses, could
fall.

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
* Meyers' ultimates with the patterns are bit for bit those of the same
  predictive simulation without them (a unit test replays it without the
  pattern columns and compares the bits), and Meyers' parity tests pass
  unchanged.
* The same holds for `Compartmental`'s future paid and outstanding cells,
  models 1 and 2 (a unit test replays the simulation without the pattern
  columns and compares the bits), and its parity tests pass unchanged.
* Mack's bootstrap draws a constant tail's factor with the tail's sigma,
  so its share at the oldest age varies about `1 / tail`; the ODP
  bootstrap keeps it fixed at exactly `1 / tail`.
