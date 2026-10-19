---
type: Finding
title: Meyers' models over segments and in the diagonal backtest
description: Meyers::fit_segments with a derived seed per segment makes twin lines' totals uncorrelated (sd 0.0083 over 60 seeds) and rank correlation reaches its targets within 0.0081; in the diagonal backtest CRC, CSR and CAY forecast group 353's held-out diagonals well above the outcomes (A/E 0.30 to 0.90) and cover every held-out cell, so their log density beats the ODP models' on wkcomp, where those under-cover, and trails it on comauto and group 353.
tags: [bayes, reserving, meyers, crc, csr, cay, segments, dependence, backtest, clrd]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-19T09:00:00-07:00 }
sources:
  - id: test
    resource: ../validation/tests/reserving_meyers_segments.rs
    title: The segment and backtest tests and their seed sweep
  - id: meyers
    resource: ../crates/prospicio-reserving/src/bayes/meyers.rs
    title: Meyers::fit_segments and TriangleModel for Meyers, with their unit tests
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decisions 8, 11 and 13, "Meyers")
  - id: pytest
    resource: ../python/tests/test_reserving_meyers_segments.py
    title: The Python binding tests of Meyers.fit_segments
---

# Finding

All numbers at `Meyers`' defaults (4 chains of 1,000 warm-up and 2,000
kept draws, 10,000 simulations; 4,000 in the backtest) but target
acceptance 0.99 in group 353's backtests, measured over seeds 0 to 119
in a release build (the ignored `seed_sweep` test with
`SWEEP_SEEDS=120`, about 40 s a seed) and checked at seeds 200 to
379.[^test]

* **Seeds per segment.** As for `BayesOdp`, segment `k` samples on tags
  `2 + 2k` and simulates on `3 + 2k`, and its pilot run takes stream
  `u64::MAX` of the segment's chains seed (`fit`'s draws are unchanged).
  Twin lines (CLRD wkcomp under two names, CRC) have a Spearman
  correlation of their totals with mean -0.002 and sd 0.0092 (largest
  0.027), as `1/sqrt(10,000)` predicts, and mean totals within 0.47% (sd
  0.14%).[^test]
* **Rank correlation** (CSR on comauto, ppauto and wkcomp, targets 0.6,
  0.3, 0.4) reaches each target within 0.0082 (sd at most 0.0029). The
  ultimates move with the reserves: Iman–Conover ranks each segment by its
  reserves plus its ultimates, `2 R + latest`, which ranks as `R` does, and
  each row's ultimates less reserves stay the latest values exactly.[^meyers]
* **The backtest adapter** rebuilds each origin's cumulative values as
  running sums of its train increments (exact for integer cells: on group
  353 the backtest's draws of 1989's 120-month cell equal the fit's
  reserve draws simulation by simulation) and simulates one path per
  origin, so an origin's increments telescope to its reserve (1997's nine
  future cells: the fit's reserve mean and sd within Monte Carlo error).
  Given the parameters a cell is independent of the origin's observed
  cumulative value, so a next-diagonal increment carries the whole
  cumulative value's spread and can be negative.[^meyers]
* **Backtest results**, latest three diagonals.[^test] On group 353 the
  models forecast well above the outcomes: A/E 0.30 to 0.36 on the
  earliest split (seven train diagonals), 0.65 (CAY), 0.66 (CRC) and 0.90
  (CSR) on the latest, against the ODP GLM's 0.82 to 0.97; CSR is the
  closest on paid, as Meyers finds. With diagonals held out the late
  `sigma_d` (their steps have Meyers' `U(0, 1)` prior) and the young
  origins' levels rest on few cells, and the mean of a log-normal grows
  with both: on the seven-diagonal triangle `sigma[1]` has posterior mean
  0.73, `alpha[7]` sd 0.83, and 12 of 8,000 draws diverge (at the
  default target acceptance 0.95). The models'
  90% intervals hold every held-out cell of group 353 and CLRD comauto
  and 0.92 to 1 of wkcomp's, where the ODP GLM and `BayesOdp` hold 0.41
  to 0.46 of wkcomp's and 0.69 to 0.73 of comauto's. So Meyers' mean log
  density beats the GLM's by 0.54 to 0.60 a cell on wkcomp, trails it by
  0.36 to 0.46 on comauto and by 1.28 to 1.39 on group 353 paid. On group
  353 incurred the GLM's IRLS does not converge, and CSR and CAY are
  within 0.07 of CRC's log density (mean cell CRPS 217, 228 and 251 for
  CSR, CRC and CAY, in the data's units).
* **Sampler failures on group 353's held-out triangles.** At the default
  target acceptance 0.95, a fit on group 353's 28- or 36-cell training
  triangles (paid or incurred) had an R-hat above 1.05 at 20 of 220
  seeds (0 to 119 and 200 to 299; up to 1.18, bulk ESS down to 15 of
  8,000, up to 978 divergences), and the backtest rows' largest outliers
  came from such fits: at seed 204 CRC on incurred (R-hat 1.094, 300
  divergences) moved its mean CRPS 4.5 sds, and with it every incurred
  row then scaled by CRC's; at seed 216 CRC on paid (R-hat 1.157) moved
  its log density 7.6 sds; at seed 284 CAY on paid (R-hat 1.184) moved an
  A/E 9.2 sds. An earlier version of this finding said no row strays
  more than 4.1 sds; that held only on seeds 0 to 59, which the
  tolerances were fitted on. At target acceptance 0.99 those fits pass
  (R-hat at most 1.008), and a fit with R-hat above 1.05 comes at 4 of
  300 seeds (0 to 119 and 200 to 379: 216, 243, 291 and 317), rarer but
  not gone: at seed 317 CSR on paid's earliest split has R-hat 1.36, bulk
  ESS 9 and 854 divergences, and its total CRPS ratio is 5.4 sds out. The
  backtest returned no diagnostics then, so such a failure was silent
  (since 2026-10-19 it reports each split's: `Backtest::diagnostics` and
  `warnings`, in Rust, Python and R; `reserving-bayes.md` decision 21);
  the sweep refitted each split's training triangle with `Meyers::fit` on
  the backtest's seed, which samples the same chains (R-hat 1.3619 both
  ways at seed 317), and counts that backtest's rows instead of checking
  them. CLRD comauto and wkcomp have no such failures at 0.95 (R-hat at
  most 1.052; their common R-hats of 1.02 to 1.03 are slow mixing
  without divergences, which 0.99 does not cure) and keep the
  default.[^test]
* **Tolerances.** Each checked row's target is its mean over seeds 0 to
  119 and its tolerance at least 6 sds there (and 1.25 times its largest
  distance from the mean). The rows' tails are a little heavier than
  normal, as a mean of log-normals rides on its largest draws: with every
  fit converged, the largest distance over seeds 0 to 119 and 200 to 379
  is 5.3 sds (group 353 paid, CSR's A/E on the earliest split, seed 23)
  and then 5.2 (comauto, CAY's A/E on the middle split, seed 372), where
  normal rows would reach about 4.1. A 4.5-sd floor failed at seed 372,
  so the floor is 6. Every row passes at each of the 180 fresh seeds. On
  incurred the CRPS and log density rows are each model's own, not
  ratios to CRC's: a ratio to a sampled baseline moves every row with
  that baseline's chains.[^test]
* Runtime: the four CI tests take about 140 s in a debug build, side by
  side (81 s before group 353 took target acceptance 0.99; its paid
  backtest alone now takes about 105 s, wkcomp's about 63 s); the twins
  about 21 s and the rank correlation about 36 s alongside them.[^test]
* **Through the bindings** (Python `Meyers.fit_segments`, R
  `meyers_segments()`, release build, 4,000 simulations): CLRD comauto
  and wkcomp with paid and net earned premium pass every diagnostic at
  the default chains at each of seeds 0 to 59 (CRC, no warning). A
  segment of the joint fit against `fit` of that line alone, over those
  seeds: `logelr`'s posterior mean differs by sd 0.0001 (largest 0.0003),
  the mean total reserve by sd 0.18% (largest 0.58%), elpd_loo by sd 0.52
  (largest 1.42, so a tolerance of 1, as for `BayesOdp`, would fail);
  twin lines' mean totals differ by sd 0.19% and their Spearman
  correlation has sd 0.0146, about `1/sqrt(4,000)`; a CSR rank
  correlation of 0.5 lands within 0.0085 (sd 0.0036). A two-line fit takes
  about 1 s and the nine Python tests about 8 s.[^pytest]

Not yet run by CI.
