---
type: Finding
title: Meyers' development pattern per posterior draw falls somewhere in nearly every simulation on group 353, so the cash-flow engine needs the running maximum
description: A posterior draw's pattern from Meyers' CRC, CSR or CAY is the ratio of the expected cumulative values to the one at the oldest age, exp(beta_d + sigma_d^2 / 2) over exp(sigma_m^2 / 2). On group 353 (paid, 2,000 simulations) it falls at some age in 1,977 (CRC), 1,976 (CSR) and 1,964 (CAY) of them, mostly over the last four steps, because the late beta_d are all near 0; DevelopmentPatterns::monthly with the default refuses, so a caller passes Monotone::RunningMax. The reserves and ultimates are bit for bit unchanged.
tags: [reserving, meyers, bayes, payment-pattern, cashflow, simulation]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-28T18:00:00Z }
sources:
  - id: note
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving, decision 27, a development pattern per posterior draw for Meyers' models
  - id: test
    resource: ../crates/prospicio-reserving/src/bayes/meyers.rs
    title: each_simulations_pattern_is_its_posterior_draws, run by CI's bayes job (slow-tests)
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/678
    title: Issue 678, a development pattern per simulation from Meyers' CRC, CSR and CAY
---

# What was found

A simulation's pattern is the posterior draw's expected cumulative value at
each age over the one at the oldest age, for an origin with nothing observed:
`exp(beta_d g + sigma_d^2 / 2) / exp(sigma_m^2 / 2)`, with `g = 1` for CRC
and CAY and `(1 - gamma)^n` for CSR (the origin after the last of `n`).
`logelr`, `alpha` and the premium cancel. CAY's `rho` does not enter.

On group 353's paid triangle (10 origins, `tune = 300`, `draws = 300`, four
chains, 2,000 simulations, seed 21):

| Model | Simulations whose pattern falls at some age | Worst fall (share of the share) |
|---|---|---|
| CRC | 1,977 | 18.5% |
| CSR | 1,976 | 14.7% |
| CAY | 1,964 | 25.0% |

The falls are at the late steps: CRC's count of simulations falling at each
of the nine steps (age 1 to 2, up to age 9 to 10) is 4, 1, 19, 527, 223,
717, 940, 1,004, 869. The late `beta_d` are all near 0, so the share is flat
near 1 over the last four or five ages and moves up and down with the
draw. The mean pattern dips by less than 1% (CRC 0.9964, 0.9960, 0.9969 at
ages 7, 8 and 9, then 1 at age 10).

# What follows

`DevelopmentPatterns::monthly` with its default `Monotone::Refuse` refuses
Meyers' patterns almost always, so a caller passes `Monotone::RunningMax`
(`monotone="running_max"` in Python, `monotone = "running_max"` in R), which
holds each share at the largest before it. The falls are noise in a pattern
that has nearly finished. No pattern was made monotone inside the
fit: `reserving-v02.md` decision 28 item 5 leaves that choice to the caller.

Adding the pattern columns draws no random number: the test replays each
simulation's stream (the first uniform picks the posterior draw, the rest are
the ultimates') and the ultimates equal the replay bit for bit for CRC and CSR
(to 1e-12 for CAY), so the fits' reserves are unchanged.
