---
type: Finding
title: Compartmental model 1 against the paper's code in Stan and its printed posterior
description: Compartmental with the paper's preset matches Gesmann and Morris' model 1 reconstructed in Stan on company 337 at every one of 200 seeds (170 rows, tolerances at least 4.25 times its own Monte Carlo error) and the paper's printed m1fit; the defaults keep 2,000 draws per chain because 1,000 left 9 of 30 seeds with an R-hat or ESS warning, and 2,000 still warn at 6 of 200 seeds, each just over a threshold.
tags: [bayes, reserving, compartmental, stan, brms, clrd, validation]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-13T22:00:00-07:00 }
sources:
  - id: test
    resource: ../validation/tests/reserving_compartmental.rs
    title: The parity tests and the seed sweep
  - id: model
    resource: ../crates/prospicio-reserving/src/bayes/compartmental.rs
    title: prospicio_reserving::bayes::Compartmental
  - id: csv
    resource: ../validation/reference/reserving_compartmental_stan.csv
    title: Stan summaries and the paper's printed numbers
  - id: script
    resource: ../validation/scripts/reserving_compartmental_stan.py
    title: The reference generator and its tolerances
  - id: bindings
    resource: ../python/tests/test_reserving_compartmental.py
    title: The Python binding tests (R's are R/prospicio/tests/test-reserving-compartmental.R)
  - id: design
    resource: ../docs/design/reserving-compartmental.md
    title: Design note, compartmental reserving (decisions 5 and 7)
---

# Finding

* Company 337, the paper's training triangle (90 observations, 1997 with
  premium only); ours at the defaults with seed 0, Stan 4 x 25,000 draws of
  the reconstruction, the paper's printed `m1fit` (mean and sd):[^test][^csv]

  | | Ours | Stan | Printed |
  |---|---|---|---|
  | `b_ker` | -5.395 (0.833) | -5.392 (0.830) | -5.40 (0.81) |
  | `b_kp` | -8.485 (0.356) | -8.488 (0.358) | -8.49 (0.36) |
  | `b_RLR` | 1.446 (0.260) | 1.442 (0.259) | 1.44 (0.26) |
  | `b_RRF` | -1.357 (0.414) | -1.358 (0.416) | -1.36 (0.42) |
  | `tau_RLR` | 0.643 (0.150) | 0.642 (0.152) | 0.64 (0.15) |
  | `tau_RRF` | 0.739 (0.220) | 0.737 (0.224) | 0.74 (0.22) |
  | `rho` | 0.533 (0.259) | 0.534 (0.260) | 0.54 (0.26) |
  | `log_sigma_os` | 8.249 (0.160) | 8.252 (0.161) | 8.25 (0.16) |
  | `log_sigma_pd` | 6.671 (0.204) | 6.671 (0.201) | 6.66 (0.20) |
  | total reserve | 173,390 (10,167) | 173,341 (10,076) | |
  | 1997 reserve | 31,376 (6,171) | 31,321 (6,125) | |
  | elpd_loo | -810.7 | -811.0 | |

  `k_er` is about 1.76 and `k_p` 0.43, against prior medians 3 and 1.
* Monte Carlo error of the defaults (4 chains of 1,000 warm-up and 2,000
  kept draws, 10,000 simulations), the seed-to-seed sd over seeds 0 to 199
  with `s` Stan's posterior sd: at most `0.032 s` on a parameter mean
  (`RLR[1991]`, `tau_RRF`), 2.3% on a parameter sd, `0.016 s` on a reserve
  mean, 1.4% on a reserve sd, `0.042 s` on a reserve percentile, `0.025 s`
  on a test-diagonal mean and 0.35 on elpd_loo. The youngest origin's
  reserve is normal given a draw (not log-normal as in Meyers' models), so
  its sd is not heavy-tailed and keeps its row. Every tolerance is at least
  4.25 times its row's spread, and seeds 0 to 199 pass all 170 rows, using
  at most 0.73 of a tolerance.[^test][^script]
* The test diagonal's means are taken from the posterior draws (the
  expected outstanding and paid of each cell, with the population's mean
  ratios for 1997), not from the predictive cells: the cells carry the
  likelihood's noise (`sigma_os` about 3,900), whose simulation error on a
  small cell's mean (outstanding 1988 at age 10, about 1,550, posterior sd
  165) is larger than the tolerance.[^test]
* Diagnostics at the defaults over the 200 seeds: 6 seeds raise a warning
  (R-hat 1.010 to 1.014 on `tau_RRF`, `rho` or `RLR[1991]`, a tail ESS of
  391, one divergent transition), no draw has `k_er <= k_p`, the median fit
  takes 5.5 s in release (4 chains in parallel). With 1,000 kept draws 9 of
  30 seeds warned; 2,000 warm-up draws as well did not help (1 of 40).
  The sampler does not report tree depths, so the share of draws at the
  maximum depth is not measured; Stan had none at depth 15.[^test][^design]
* Both starts converge at seed 0 (the population least squares and the
  prior centre, `o = 0`, 5 to 9 prior sds from the posterior's rates), and
  `RateOrder::Ordered` gives the same posterior means within the Stan
  rows' tolerances.[^test]
* Through the Python and R bindings (release builds), the default fit at
  seed 0 with 2,000 simulations passes every diagnostic and gives the Rust
  core's posterior (`b_kp` -8.485, elpd_loo -810.7) and a total reserve of
  173,457 (sd 10,260), against 173,390 with 10,000 simulations; it takes
  about 5 s, and the 17 Python tests about 11 s in all.[^bindings]
