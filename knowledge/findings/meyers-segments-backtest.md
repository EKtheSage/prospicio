---
type: Finding
title: Meyers' models over segments and in the diagonal backtest
description: Meyers::fit_segments with a derived seed per segment makes twin lines' totals uncorrelated (sd 0.0083 over 60 seeds) and rank correlation reaches its targets within 0.0081; in the diagonal backtest CRC, CSR and CAY forecast group 353's held-out diagonals well above the outcomes (A/E 0.30 to 0.90) and cover every held-out cell, so their log density beats the ODP models' on wkcomp, where those under-cover, and trails it on comauto and group 353.
tags: [bayes, reserving, meyers, crc, csr, cay, segments, dependence, backtest, clrd]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-13T08:00:00-07:00 }
sources:
  - id: test
    resource: ../validation/tests/reserving_meyers_segments.rs
    title: The segment and backtest tests and their seed sweep
  - id: meyers
    resource: ../crates/prospicio-reserving/src/bayes/meyers.rs
    title: Meyers::fit_segments and TriangleModel for Meyers, with their unit tests
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decisions 8 and 11, "Meyers")
---

# Finding

All numbers at `Meyers`' defaults (4 chains of 1,000 warm-up and 2,000
kept draws, 10,000 simulations; 4,000 in the backtest), measured over
seeds 0 to 59 in a release build (the ignored `seed_sweep` test with
`SWEEP_SEEDS=60`, about 23 s a seed).[^test]

* **Seeds per segment.** As for `BayesOdp`, segment `k` samples on tags
  `2 + 2k` and simulates on `3 + 2k`, and its pilot run takes stream
  `u64::MAX` of the segment's chains seed (`fit`'s draws are unchanged).
  Twin lines (CLRD wkcomp under two names, CRC) have a Spearman
  correlation of their totals with mean -0.003 and sd 0.0083 (largest
  0.022), as `1/sqrt(10,000)` predicts, and mean totals within 0.47% (sd
  0.15%).[^test]
* **Rank correlation** (CSR on comauto, ppauto and wkcomp, targets 0.6,
  0.3, 0.4) reaches each target within 0.0081 (sd at most 0.0030). The
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
  0.73, `alpha[7]` sd 0.83, and 12 of 8,000 draws diverge. The models'
  90% intervals hold every held-out cell of group 353 and CLRD comauto
  and 0.92 to 1 of wkcomp's, where the ODP GLM and `BayesOdp` hold 0.41
  to 0.46 of wkcomp's and 0.69 to 0.73 of comauto's. So Meyers' mean log
  density beats the GLM's by 0.54 to 0.60 a cell on wkcomp, trails it by
  0.36 to 0.46 on comauto and by 1.28 to 1.39 on group 353 paid. On group
  353 incurred the GLM's IRLS does not converge, and CSR and CAY are
  within 0.07 of CRC's log density.
* No checked row is heavy-tailed: over 60 seeds none strays more than 4.1
  sds from its mean, so every row keeps a tolerance of about 4 sds.[^test]
* Runtime: the four CI tests take about 81 s in a debug build (two
  backtests in parallel dominate); the twins about 11 s, the rank
  correlation about 20 s.[^test]

Not yet run by CI.
