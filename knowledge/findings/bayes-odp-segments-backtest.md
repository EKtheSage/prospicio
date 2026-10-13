---
type: Finding
title: The Bayesian ODP over segments and in the diagonal backtest
description: BayesOdp::fit_segments with a derived seed per segment makes twin lines' totals uncorrelated (sd 0.010 over 70 seeds, as for independent draws), rank correlation reaches its Spearman targets within 0.009; in the diagonal backtest BayesOdp agrees with the ODP GLM on the CLRD lines and gives about 1% higher means on GenIns; Newton's method needed a scale-aware rounding floor on lines summed over companies.
tags: [bayes, reserving, odp, segments, dependence, backtest, clrd, genins]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-12T20:00:00-07:00 }
sources:
  - id: test
    resource: ../validation/tests/reserving_bayes_segments.rs
    title: The segment and backtest tests and their seed sweep
  - id: odp
    resource: ../crates/prospicio-reserving/src/bayes/odp.rs
    title: BayesOdp::fit_segments and TriangleModel for BayesOdp
  - id: laplace
    resource: ../crates/prospicio-reserving/src/bayes/laplace.rs
    title: The posterior mode's Newton method
  - id: py
    resource: ../python/tests/test_reserving_bayes_segments.py
    title: The Python binding tests of fit_segments
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decisions 3, 8 and 11)
---

# Finding

All numbers at `BayesOdp`'s defaults (4 chains of 1,000 draws, 10,000
simulations; 4,000 in the backtest), measured over seeds 0 to 69 in a
release build (the ignored `seed_sweep` test with `SWEEP_SEEDS=70`).[^test]

* **Seeds per segment.** Each segment samples its own posterior and calls
  `PosteriorDraws::predictive`, which gives simulation `i` stream `i` of
  its seed: with one seed for every segment, simulation `i` of each would
  pick the same posterior draw position and use the same Gamma uniforms.
  With segment `k` on tags `2 + 2k` and `3 + 2k` of the derived-seed
  scheme, twin lines (CLRD wkcomp under two names) have a Spearman
  correlation of their totals with mean -0.0003 and sd 0.010 over seeds
  (largest 0.027), as `1/sqrt(10,000)` predicts for independent draws, and
  mean totals within 0.29% of each other.[^odp]
* **Rank correlation** through the bootstraps' Iman–Conover reordering
  reaches Spearman targets 0.6, 0.3 and 0.4 between comauto, ppauto and
  wkcomp within 0.009 (sd at most 0.0029).[^test]
* **Backtest against the ODP GLM**, latest three diagonals: on comauto and
  wkcomp the two models' A/E agree within 0.0011 per split, their mean
  cell CRPS within 1.9% and log densities within 0.11 on average. On
  GenIns the Bayesian A/E is lower by 0.007, 0.013 and 0.010 per split on
  average over seeds, and by 0.004 to 0.019 across seeds and splits (its means
  about 1% higher: the posterior mean of `exp(eta)` over the young origins'
  wide posteriors, where the GLM forecasts the MLE), and its mean log
  density lower by 0.013 on average (sd 0.0044 over seeds 0 to 69, 0.0051
  over 40 to 69 alone). Across seeds both models' 90% intervals cover 0.69
  to 0.73 of comauto's held-out cells and 0.41 to 0.46 of wkcomp's: the ODP model is
  overconfident on lines summed over companies, whatever the sampler.
  Single cells' log densities differ between the models by up to 2 at
  some seeds, so only means are compared.[^test]
* **Newton at the rounding floor.** On CLRD ppauto (increments near
  `10^7`) the mode's Newton method ran out of iterations: `b[10]` rests on
  one small cell, rounding kept its Newton step near `1e-6` (far below its
  posterior sd), a full step lowered the log posterior by one ulp, and the
  halving shrank the step until it no longer moved `theta`. Stopping when a
  full step lowers `f` by no more than `1e-12 |f|` fixes it; GenIns and
  RAA draws are unchanged bit for bit.[^laplace]
* **Through the bindings** (Python `BayesOdp.fit_segments`, R
  `bayes_odp_segments()`), comauto and wkcomp at the default chains pass
  every diagnostic at seed 5, twin wkcomp lines' totals stay within
  `4.5/sqrt(n_sims)` of uncorrelated and a 0.5 Spearman target is met
  within 0.03 at 4,000 simulations; the nine Python tests take about 5 s
  with a release build.[^py]
* Runtime: the seed sweep takes about 6.4 s a seed in a release build
  (two twin fits, three rank-correlated fits, three backtests of three
  splits); the three CI tests take about 32 s in a debug build.[^test]

Not yet run by CI.
