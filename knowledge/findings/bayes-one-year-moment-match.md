---
type: Finding
title: Moment matching rescues the flagged simulations of the Bayesian one-year view
description: Moving the posterior draws of a flagged simulation by affine maps to the weighted moments (Paananen et al. 2021, as loo_moment_match, with the split estimate) brings 99.99% to 100% of them to Pareto k at most 0.7 on RAA and GenIns under BayesOdp and BayesBf, within 0.04 to 0.06 sd of their NUTS refits where reweighting alone is 0.16 to 0.24 off, at about 1 ms of wall time each against about 26 ms per refit.
tags: [bayes, reserving, one-year, psis, importance-sampling, moment-matching, loo, raa, genins]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-15T00:00:00-07:00 }
sources:
  - id: test
    resource: ../validation/tests/reserving_bayes_one_year_moment_match.rs
    title: Moment matching against the refits on the same years, and the seed sweep
  - id: engine
    resource: ../crates/prospicio-reserving/src/bayes/moment_match.rs
    title: The moment matching steps, the split estimate and their unit tests
  - id: loo
    resource: https://github.com/stan-dev/loo/blob/master/R/loo_moment_matching.R
    title: R loo's loo_moment_match, the implementation followed
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decision 18, open decision 12)
---

# Finding

In the one-year view of a Bayesian reserving model, a simulation whose
Pareto `k` is above 0.7 used to be refitted with NUTS. Moment matching
first moves the opening posterior draws by affine maps (a shift to the
weighted mean, a scaling to the weighted variances, a map to the weighted
covariance), keeping each step that lowers `k`, and estimates the closing
reserve by `loo`'s split estimate; only a simulation whose `k` stays above
0.7 is refitted.[^engine] The steps follow `loo`'s source, not the paper's
summary: the scaling uses `loo`'s `S / (S - 1)` factor and the weighted
covariance `cov.wt`'s `1 - Σ w²` divisor.[^loo]

Over seeds 0 to 19, 2,000 simulations each, every setting on the same
years (mean, seed-to-seed sd in brackets):[^test]

| | `BayesOdp` RAA | `BayesOdp` GenIns | `BayesBf` GenIns (cv 0.2) |
|---|---|---|---|
| flagged | 268 (14) | 531 (15) | 255 (17) |
| rescued | 100% | 99.99% | 100% |
| rms gap to the refit, matched (sd of the total) | 0.058 (0.008) | 0.036 (0.007) | 0.038 (0.007) |
| the same simulations, reweighted only | 0.240 (0.045) | 0.155 (0.012) | 0.186 (0.021) |
| total sd over `RefitAll`'s, matched | 0.999 (0.006) | 1.001 (0.008) | 0.999 (0.008) |
| wall time, `RefitFlagged` / `MomentMatchThenRefit` (16 threads) | 7.7 s / 0.50 s | 13.6 s / 0.73 s | 6.9 s / 0.48 s |

* A rescued simulation is as close to its refit as an unflagged
  simulation's reweighting is to its own (0.035 to 0.056 sd), and the
  rescued simulations' spread and mean match the refits' (sd ratio
  1.000 to 1.002, mean gap within 0.006 sd of 0).
* The posteriors of these log-linear ODP models are close enough to
  normal that an affine map nearly always suffices: of about 21,000
  flagged simulations over the sweep, one or two (on GenIns) fell through
  to a refit.
* At 2,000 simulations the total's sd under `ReweightOnly` is within
  Monte Carlo error of `RefitAll`'s (ratio 1.004, sd 0.009 to 0.013);
  reweighting is noisier per simulation, not visibly biased at this
  size.
* The wall times were taken while other builds shared the machine;
  the ratio is steady, about 1 ms per matched simulation against 25 to
  28 ms per refit.

A unit test on a conjugate normal model (six parameters, `k` about 5)
lands within 0.05 of the exact posterior mean on every parameter (0.029
at the test's seed) where reweighting alone is 0.25 off.[^engine]

Matching's affine maps act on the scale the posterior draws are stored
on, so that scale must be unconstrained and the model's log posterior
must be the density of the draws on it. `BayesOdp` and `BayesBf` store
an affine map of their sampling scale and need no Jacobian. `Meyers`
and `Compartmental` store natural-scale values (`tanh`, square roots,
exponentials): a move can leave the support there, the ratio turns NaN
and counts as a zero weight, and matching falls back to refits without
saying so. A Jacobian left out cancels in the check that the year's log
likelihood is the year's change to the log posterior; the score
identity (the mean finite-difference score over the posterior draws is
zero) catches it, at `z` 18 for draws stored as `exp` against 1.1 to 1.7
for the two models.[^engine]

Checked by CI at 1,000 simulations and seed 0 (rescued at least 0.99,
gap below 0.11 and 0.07), not yet `verified` in this bundle's sense
until it has run there.
