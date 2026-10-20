---
type: Finding
title: The compartmental model over segments and in the diagonal backtest, paid and outstanding
description: Compartmental::fit_segments on a derived seed per segment makes twin segments independent and rank correlation reaches its targets with each segment's reserves, paid cells and outstanding cells reordered as one row; in the diagonal backtest on company 337 model 1 forecasts the held-out outstanding diagonals well (A/E 1.08, 1.02 and 0.98, every 90% interval holds its cell) but the paid increments badly (A/E 1.35, 1.07 and 1.16, mean cell CRPS 3.5 and total CRPS 1.8 times the ODP GLM's), so the outstanding total stays within 8% of its forecast at every split and the backtest cannot tell a case reserve strength shift from faster settlement.
tags: [bayes, reserving, compartmental, gesmann-morris, segments, dependence, backtest, outstanding, clrd]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-19T18:00:00-07:00 }
sources:
  - id: test
    resource: ../validation/tests/reserving_compartmental_segments.rs
    title: The segment and backtest tests, their tolerances and the seed sweep
  - id: model
    resource: ../crates/prospicio-reserving/src/bayes/compartmental.rs
    title: Compartmental::fit_segments
  - id: candidate
    resource: ../crates/prospicio-reserving/src/bayes/compartmental/backtest.rs
    title: CompartmentalCandidate and the outstanding backtest
  - id: design
    resource: ../docs/design/reserving-compartmental.md
    title: Design note, compartmental reserving (decision 8, and Risks)
  - id: pytest
    resource: ../python/tests/test_reserving_compartmental_segments.py
    title: The Python binding tests (R's are R/prospicio/tests/test-reserving-compartmental-segments.R)
---

# Finding

All numbers are model 1 (Gesmann and Morris, normal, cumulative paid and
outstanding amounts) at its defaults (4 chains of 1,000 warm-up and 2,000
kept draws, target acceptance 0.99, 10,000 simulations; 4,000 in the
backtest) on company 337 workers' compensation as the papers fit it
(`clrd-wkcomp-337`: calendar years up to 1996, 1997 with its premium only,
outstanding is incurred less paid), in a release build at seed 0. Each
checked row's target is its mean over a 30-seed sweep and its tolerance
at least six times its seed-to-seed sd, confirmed at seeds 100 to
103.[^test]

* **Seeds and dependence.** As for `Meyers`, segment `k` samples on
  `Stream::SegmentChains(k)` and simulates on `Stream::SegmentPredictive(k)`,
  so `fit`'s draws are unchanged. Twin segments (company 337 under two
  names) have a Spearman correlation of their reserves within
  `4.5 / sqrt(n_sims)` of zero and mean reserves within 0.6%. A rank
  correlation (three copies, targets 0.6, 0.3 and 0.4) is reached within
  0.010, 0.015 and 0.018, the tolerances of the sweep.[^test] The
  Iman-Conover reorder moves a segment's reserves, its future paid cells
  and its future outstanding cells as one row, ranked on the reserves, so
  every simulation's reserve is still its paid at the last age less the
  latest paid, exactly, in every segment (a unit test, and the Python test
  `test_a_reserve_is_the_paid_at_the_last_age_less_the_latest`).[^model]
* **The backtest holds out the latest three calendar diagonals** of the
  45-cell training triangle (1994, 1995 and 1996, trained on the cells
  before each), so 5, 6 and 7 held-out cells are scored and 2 a split are
  left out for lack of a training cell of their origin or age. The paid
  and the outstanding table score the same cells, from one NUTS fit a
  split.[^candidate]
* **Paid increments**, against the ODP GLM and `BayesOdp` (A/E on the
  diagonal total, then mean cell CRPS and total CRPS as ratios to the
  GLM's, then the mean log density a cell less the GLM's):

  | Model | A/E, splits 0 to 2 | cell CRPS | total CRPS | log density | 90% coverage |
  |---|---|---|---|---|---|
  | ODP GLM | 0.89, 0.92, 1.02 | 1 | 1 | 0 | 0.87 |
  | `BayesOdp` | 0.89, 0.92, 1.02 | 1.01 | 1.02 | 0.003 | 0.87 |
  | Compartmental 1 | 1.35, 1.07, 1.16 | 3.5 | 1.8 | -1.15 | 0.89 |

  The compartmental forecast of the paid increments is worse than the
  chain ladder's family on every metric except coverage, which it buys
  with wide intervals. This is a forecast of single next-diagonal
  increments from curves fitted to cumulative values with one `sigma_pd`
  for every cell (an inference from the model's form, not a measured
  cause), the adapter simulating one path per origin from a posterior
  draw as for `Meyers`.[^test]
* **Held-out outstanding.** The same split and scoring code applied to
  the outstanding cells: A/E on the diagonal's total outstanding 1.08,
  1.02 and 0.98, mean cell CRPS 1,866 and total CRPS 3,655 (USD
  thousands), mean log density -9.56 a cell, and the 90% interval holds
  every held-out cell (coverage 1.00). No other model gives an
  outstanding, so there is no comparison; the A/E says the level of the
  held-out outstanding diagonals is within 8% of the forecast at each of
  the three splits.[^test]
* **What it says about 1996 and 1997.** The design note's Risks
  section records a change in case reserve strength at company 337 in
  1996 and 1997 and expected the backtest to measure it. The backtest
  can look at 1996 only: the 1997 diagonal is the papers' test set and not
  in the 45-cell triangle. At the last split (1996 held out, the model
  fitted to cells up to 1995) the outstanding comes in 2.4% below the
  forecast (A/E 0.976) while the paid increments come in 16% above it
  (1.16); at the first two splits the outstanding is 8% and 2% above its
  forecast. So the outstanding total moves by less than 10% against its
  forecast at every split and no shift stands out in 1996. That is weak
  evidence either way: the paid and outstanding A/Es are on different
  scales (increments against levels), the outstanding here includes bulk
  and IBNR, and nothing in the table separates a strength change from
  faster settlement. The note's Risks line keeps its caution, and a
  backtest on a triangle that includes 1997 would be the test.[^design]
* **Model 2** (log-normal, loss ratios, incremental paid, the heavy
  sampler) runs its twins and a one-diagonal backtest only under
  `RISK_RS_FULL_BAYES`: the 9-origin training triangle has fewer cells
  than its 87 parameters once a diagonal is held out, so the arm uses the
  55-cell upper triangle to 1997 and holds out the 1997 diagonal. Its
  scores are reported and checked only for sanity.[^test]
* **Cost.** A backtest of three splits with three models takes about 10 s
  in release (the paid and outstanding tables share the fits), the twin
  check 15 s and the rank-correlation check 23 s; the Rust validation file
  takes 48 s. The Python binding tests take about 45 s and the R ones 77
  s.[^pytest]

[^test]: [validation/tests/reserving_compartmental_segments.rs](../validation/tests/reserving_compartmental_segments.rs), release build, seed 0 (the rows at `--nocapture`).
[^model]: [Compartmental::fit_segments](../crates/prospicio-reserving/src/bayes/compartmental.rs) and its unit tests.
[^candidate]: [CompartmentalCandidate](../crates/prospicio-reserving/src/bayes/compartmental/backtest.rs).
[^design]: [Design note, compartmental reserving](../docs/design/reserving-compartmental.md), decision 8 and Risks.
[^pytest]: [python/tests/test_reserving_compartmental_segments.py](../python/tests/test_reserving_compartmental_segments.py); R's `test-reserving-compartmental-segments.R` runs the same checks.
