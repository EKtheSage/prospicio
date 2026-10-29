---
type: Finding
title: Compartmental's paid pattern per posterior draw leaves 2% of the ODE's limit past the square, and no simulation's share fell in the short fits checked
description: The paid curve of each posterior draw's population rates over its value at the square's last age ends at exactly 1 there; on company 337 the mean paid at ten years is 0.981 of the ODE's limit, which the pattern does not include. In short fits of model 1 (500 simulations) every share rises with age, and in model 2 with its development-age effects none of 200 simulations fell, so the cash-flow engine takes them without a running maximum there. Computing the pattern draws no random number.
tags: [reserving, compartmental, payment-pattern, cashflow, simulation]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-28T20:00:00Z }
sources:
  - id: note
    resource: ../docs/design/reserving-compartmental.md
    title: Design note, compartmental reserving, decision 14, a paid pattern per posterior draw
  - id: pattern-note
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 28, a development pattern per simulation
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/679
    title: Issue 679, a development pattern per simulation from Compartmental
---

# What was found

`CompartmentalFit.patterns` is, for each simulation, the paid curve of the
one-compartment model (`PD(t) = 1 - e^(-k_p t) - k_p g(t)`, decision 2) at
the posterior draw's population rates, over the same curve at the square's
last age. The reserves stop at that age, so the pattern reaches exactly 1
there and carries no tail share.

Measured by the unit tests on company 337's training triangle with short
samplers (2 chains of 100 kept draws for model 1, 500 simulations, seed 21;
2 chains of 60 for model 2, 200 simulations, seed 6), on 2026-10-28:

* The mean over the simulations of the paid curve at ten years is 0.9814 of
  the ODE's limit of 1 per unit of `RLR RRF`. The normaliser at the square's
  last age therefore moves the early shares by about 2% against the limit's;
  the test asserts only that it lies between 0.9 and 1.
* Model 1's shares rise with age in all 500 simulations (the curve is
  increasing in `t`).
* Model 2's shares, built from the cumulative sum of increments at each
  age's rates times the development-age effects that simulation's reserves
  used (at 120 months, which has no observed cell on company 337, the
  effect the simulation drew), fell at no age in 200 simulations. The development effects could make a share fall, and
  `DevelopmentPatterns::monthly` refuses a falling share by default
  (decision 28 item 5), so a user of a longer model 2 fit should expect to
  meet `Monotone::RunningMax` if the effects are large; this short fit did
  not show it.

# Other facts confirmed

* Adding the pattern columns to the simulated row draws no random number: a
  replay of each simulation's cells from its stream, without the pattern,
  equals the fit's paid cells and reserves bit for bit (unit test
  `a_draws_pattern_is_the_paid_curve_of_its_rates`).
* Over segments, a rank correlation moves the pattern with its reserves:
  every row of a correlated segment's reserves is a row of the same fit
  without the correlation, and its pattern is that row's pattern (unit test
  `segment_patterns_follow_their_reserves_through_a_rank_correlation`).
