---
type: Finding
title: On simulated sub-annual books Compartmental is calibrated at every grain in its own world, but model 1's interval narrows with the grain when noise persists, and annual origins with earned premium fail at every grain
description: On books simulated with known parameters and read at annual, quarterly and monthly development, model 1 in its own world covers the total reserve about equally at every grain (82 to 84 of 90) with Meyers' D passing, and finer cells cut the sd from 2,636 to 1,707. When payment noise stays in cumulative paid the sd still falls (1,762 to 1,069) while the error does not (RMSE 1,623 to 2,033) and coverage falls from 88 to 56. Annual origins with earned premium fail D at all three grains with rates pulled towards a swap; quarterly origins pass. The paper's rate preset stays sensible and matters less as the grain falls. Model 2 with one development level per age mixes badly on monthly cells, where levels by development year do not; by year is badly wrong when effects move within a year. Without a seasonal term model 1's mean moves up to 0.6 sd at mid-year and its sd widens about 50%.
tags: [bayes, reserving, compartmental, grain, quarterly, monthly, simulated, calibration, seasonality, nuts]
status: draft
generated: { by: claude-code/local-session, at: 2026-11-06T06:00:00-08:00 }
sources:
  - id: harness
    resource: ../validation/tests/reserving_compartmental_subannual.rs
    title: The worlds, the panels (ignored), prior_bands, zero_cells and export_books
  - id: grouped
    resource: ../crates/prospicio-reserving/src/bayes/compartmental/subannual_study.rs
    title: The test-only grouped development levels and grouped_against_per_age (ignored)
  - id: summary
    resource: ../validation/scripts/compartmental_subannual_summary.py
    title: Prints the tables from the panels' CSVs
  - id: design
    resource: ../docs/design/reserving-compartmental.md
    title: Design note, decision 14 and open decisions 9 to 13
  - id: grain
    resource: compartmental-grain.md
    title: Compartmental on quarterly development, the parity behind this study
  - id: calibration
    resource: meyers-calibration-compartmental.md
    title: Model 1 and model 2 on the Schedule P panel
---

# Facts

Run on 2026-11-05 and 06 (Windows, release build, 4 threads a fit, two fits at
a time, other heavy work on the machine for part of the run, so times are
rough). Not marked verified: no CI job runs the panels, only three cheap
checks of the books and one of the grouped levels. `target/` is not committed;
every figure below is reproduced by the commands at the end. NUTS draws at a
fixed seed differ by platform, so the last digits are this machine's. The
choices these results raise are in decision 14.[^design] The parity under them
is in the grain finding.[^grain]

Book `b` of a world is `StreamRng::new(20261105, stream * 10,000 + b)`, its
fit seed `1,000 + b`, in every arm, so an arm's books are the same books at
each grain, and `season` and `iid` (or `earned` and `seasonq`) are the same
books with and without the factor.[^harness] Panels run a shorter sampler than
the default: 500 warm-up and 500 draws a chain, target acceptance 0.9 for
model 1 and 0.99 for model 2, 4,000 simulations.

## Calibration of the total reserve

Coverage is of the 90% interval (binomial standard error about 4 points at 50
books), D is Meyers' Kolmogorov-Smirnov distance of the ultimate percentile
with its 5% critical value, the RMSE is of the mean against the outcome, and
sd is the median. Grains: `aa`, `aq`, `am` are annual origins with annual,
quarterly and monthly development; `qq`, `qm` quarterly origins.[^harness][^summary]

| World | Model | Grain | Cells | Books | Coverage | D (critical) | RMSE | sd |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `iid` | model 1 | aa | 110 | 50 | 84 | 12.1 (19.2) | 2,853 | 2,636 |
| `iid` | model 1 | aq | 440 | 50 | 82 | 12.4 (19.2) | 2,103 | 1,949 |
| `iid` | model 1 | am | 1,320 | 50 | 84 | 5.5 (19.2) | 1,716 | 1,707 |
| `persist` | model 1 | aa | 110 | 50 | 88 | 10.7 (19.2) | 1,623 | 1,762 |
| `persist` | model 1 | aq | 440 | 50 | 70 | 16.3 (19.2) | 1,903 | 1,271 |
| `persist` | model 1 | am | 1,320 | 50 | 56 | 19.1 (19.2) | 2,033 | 1,069 |
| `persist` | model 1, incremental | aa | 110 | 50 | 98 | 28.1 (19.2) | 1,510 | 3,356 |
| `persist` | model 1, incremental | aq | 440 | 50 | 100 | 19.9 (19.2) | 2,151 | 3,564 |
| `persist` | model 1, incremental | am | 1,320 | 50 | 100 | 11.0 (19.2) | 2,731 | 3,605 |
| `earned` | model 1 | aa | 110 | 50 | 100 | 39.9 (19.2) | 7,423 | 9,961 |
| `earned` | model 1 | aq | 440 | 50 | 98 | 23.9 (19.2) | 4,058 | 5,758 |
| `earned` | model 1 | am | 1,320 | 50 | 92 | 37.4 (19.2) | 4,860 | 4,728 |
| `earned` | model 1 | qq | 1,640 | 50 | 92 | 17.8 (19.2) | 2,007 | 2,293 |
| `earned` | model 1 | qm | 4,920 | 20 | 95 | 21.9 (30.4) | 1,586 | 1,964 |
| `m2y` | model 2 | aa | 110 | 50 | 86 | 13.8 (19.2) | 5,527 | 5,478 |
| `m2y` | model 2 | aq | 440 | 50 | 88 | 18.1 (19.2) | 4,882 | 4,362 |
| `m2y` | model 2 | am | 1,320 | 20 | 80 | 35.3 (30.4) | 5,259 | 4,135 |
| `m2q` | model 2 | aa | 110 | 50 | 96 | 7.8 (19.2) | 5,761 | 5,893 |
| `m2q` | model 2 | aq | 440 | 50 | 82 | 11.2 (19.2) | 5,156 | 4,378 |
| `m2qo` | model 2 | qq | 1,640 | 20 | 80 | 16.7 (30.4) | 2,960 | 3,032 |

* **In its own world** (`iid`) model 1 covers about the same at every grain
  (84, 82, 84; about two standard errors under 90) and D passes, while the sd
  and the RMSE fall together.[^harness][^summary]
* **Persistent noise** (`persist`): the sd falls from 1,762 to 1,069, the RMSE
  rises from 1,623 to 2,033 and coverage falls from 88 to 56; `k_p`'s central
  90% holds the truth in 80% of books annually and 18% monthly. Model 1 with
  incremental paid keeps the sd (3,356 to 3,605), covers 98 to 100 and is too
  wide (the sd of the standardised errors is 0.48 to 0.75). The same model's
  intervals were already narrow on annual Schedule P triangles.[^calibration]
* **Earned premium** (`earned`): annual origins fail D at all three grains,
  with `k_er` at 1.92, 1.22 and 1.11 against 3.32, and 45 of 50 monthly fits
  have draws with `k_er <= k_p`. Quarterly origins of the same books pass: D
  17.8, coverage 92, RMSE 2,007, `k_er` 3.30. Every book's sum over quarters is
  the annual origin, so the misfit is the model's single start at `t = 0`.
* **Model 2** covers 86 and 88 on `m2y` and 96 and 82 on `m2q`, annual and
  quarterly. The world's `tau_dev` of 1 sits three scales into the preset's
  half-t, so read grains against each other. On the same first 20 `m2y` books
  all three grains cover 80, with the mean 0.41, 0.49 and 0.55 sd high.
* **`sigma_pd` depends on the grain** under incremental paid, about as
  `sqrt(n)` for `n` months, and the preset's `log σ` prior does not: 8% of annual
  `m2y` books cover the truth of 0.088, 92% of quarterly ones. A prior of log sd
  1 changes the reserve little (coverage 85 against 80, RMSE 5,515 against 5,549,
  the same 20 books).

## Sampler

| Arm | Cells | Sampler | Books | Seconds | R-hat | Share above 1.01 | Bulk ESS | Books with divergences |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| model 1, aa | 110 | study | 50 | 2 | 1.017 | 90% | 184 | 46 |
| model 1, aa | 110 | default | 5 | 8 | 1.007 | 20% | 637 | 0 |
| model 1, aq | 440 | study | 50 | 7 | 1.022 | 90% | 178 | 45 |
| model 1, aq | 440 | default | 5 | 32 | 1.007 | 0% | 782 | 0 |
| model 1, am | 1,320 | study | 50 | 33 | 1.020 | 90% | 198 | 43 |
| model 1, am | 1,320 | default | 5 | 163 | 1.005 | 0% | 908 | 0 |
| model 1, qq | 1,640 | study | 50 | 34 | 1.056 | 100% | 63 | 27 |
| model 1, qq | 1,640 | default | 3 | 195 | 1.008 | 33% | 308 | 0 |
| model 1, qm | 4,920 | study | 20 | 149 | 1.039 | 100% | 94 | 8 |
| model 2, aa | 110 | study | 50 | 6 | 1.024 | 92% | 151 | 0 |
| model 2, aa | 110 | default | 5 | 52 | 1.001 | 0% | 2744 | 0 |
| model 2, aq | 440 | study | 50 | 32 | 1.034 | 100% | 99 | 0 |
| model 2, aq | 440 | default | 3 | 255 | 1.003 | 0% | 1302 | 0 |
| model 2, am | 1,320 | study | 20 | 128 | 1.037 | 100% | 119 | 0 |
| model 2, am | 1,320 | default | 1 | 2,850 | 1.542 | 100% | 7 | 0 |
| model 2, qq | 1,640 | study | 20 | 91 | 1.066 | 100% | 63 | 0 |

Under the default sampler model 1 converges on every arm measured: R-hat at
most 1.011, no divergence, bulk ESS at least 204. Decision 13's 32 s at 440
cells is reproduced (32 s), and the time grows a little faster than the cells:
8, 32, 163 and 195 s at 110, 440, 1,320 and 1,640 cells, about the power 1.2.
Model 2 takes 52 s, 255 s and 2,850 s at 110, 440 and 1,320 cells, the last
steps at about the square of the cells, with 541 parameters. The panels' R-hat
of 1.02 to 1.06 is the short run's: on the same books the default sampler moves
the reserve's mean by at most 0.045 sd and its sd by under 1%, so the panels'
reserves stand, and their parameter coverage carries the short run's noise.
The short run does leave strays: on `iid` monthly one book of 50 has R-hat 1.60
and 357 divergences. Model 2's default sampler is enough on 110 and 440 cells
(R-hat at most 1.007, ESS at least 861). On monthly `m2y` one book of 20 (book
0) stays at R-hat 1.62 under the short run and 1.54 after the default's 48
minutes, with ESS 7 and 1,322 warnings, yet its reserve moves 0.005 sd between
the two runs: the trouble is in the rates' parameters, not the total. Quarterly
origins cost about the same cells for cells (`earned`, 1,640 cells: 34 s short,
195 s default; model 2 `m2qo`: 91 s short, not run at the default), and
monthly development on quarterly origins (4,920 cells) took 149 s short.

## Rate priors

| World | Grain | Prior | `k_er` (truth) | `k_er` covered | Reserve coverage | RMSE | sd |
|---|---|---|---:|---:|---:|---:|---:|
| `iid` | aa | preset | 3.25 (3.32) | 90% | 84 | 2,853 | 2,636 |
| `iid` | aa | ratewide | 3.40 (3.32) | 85% | 85 | 2,990 | 2,567 |
| `iid` | aq | preset | 3.31 (3.32) | 90% | 82 | 2,103 | 1,949 |
| `iid` | aq | ratewide | 3.34 (3.32) | 85% | 95 | 2,073 | 1,920 |
| `iid` | am | preset | 3.31 (3.32) | 88% | 84 | 1,716 | 1,707 |
| `conflict` | aa | preset | 1.78 (1.75) | 90% | 85 | 3,456 | 3,314 |
| `conflict` | aa | ratewide | 1.76 (1.75) | 95% | 90 | 3,733 | 3,380 |
| `conflict` | aq | preset | 1.76 (1.75) | 85% | 95 | 2,100 | 2,463 |
| `conflict` | aq | ratewide | 1.76 (1.75) | 90% | 95 | 2,133 | 2,508 |
| `conflict` | am | preset | 1.76 (1.75) | 80% | 90 | 1,788 | 1,992 |
| `conflict` | am | ratewide | 1.76 (1.75) | 85% | 90 | 1,807 | 1,966 |
| `fast` | aa | preset | 5.20 (8.0) | 0% | 90 | 2,386 | 2,024 |
| `fast` | aa | ratewide | 11.66 (8.0) | 100% | 85 | 2,216 | 1,933 |
| `fast` | aq | preset | 7.73 (8.0) | 65% | 90 | 1,712 | 1,641 |
| `fast` | aq | ratewide | 8.04 (8.0) | 100% | 90 | 1,706 | 1,635 |
| `fast` | am | preset | 7.93 (8.0) | 85% | 95 | 1,416 | 1,551 |
| `fast` | am | ratewide | 8.01 (8.0) | 90% | 95 | 1,414 | 1,553 |

Prior predictive under the preset (test `prior_bands`): paid at 0.3% to 0.8%
of premium at month 1, 2.5% to 5.9% at month 3, 18% to 40% at month 12;
`ratewide` (log sd 1 on both rates) 4.6% to 57% at month 12. The reserve does
not depend on the choice where the truth is at the preset or at company 337's
rates, and finer cells pull the preset to the truth on `fast`. Under `ratewide`
5 of 20 annual `fast` fits have rate-swap draws, under the preset none. Model
2's rate effects sit on the scale of the prior's sd, so `ratewide` also widens
them tenfold there; on the same 20 `m2y` quarterly books the reserve is no
different (coverage 90 against 80, RMSE 5,564 against 5,549).

## Model 2's development effects

| World, grain | Form | Parameters | Books | Seconds | R-hat | Bulk ESS | Coverage | D | sd | elpd_loo |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m2y`, aq | per age | 221 | 30 | 22 | 1.032 | 105 | 87 | 29.5 | 4,324 | 1,930.8 |
| `m2y`, aq | by year | 101 | 30 | 21 | 1.035 | 110 | 87 | 24.3 | 4,311 | 1,952.2 |
| `m2q`, aq | per age | 221 | 30 | 34 | 1.029 | 120 | 83 | 12.4 | 4,371 | 1,918.4 |
| `m2q`, aq | by year | 101 | 30 | 9 | 1.014 | 225 | 97 | 8.9 | 6,933 | 1,557.3 |
| `m2season`, aq | per age | 221 | 20 | 28 | 1.059 | 67 | 80 | 28.9 | 4,789 | 1,910.2 |
| `m2season`, aq | by year | 101 | 20 | 22 | 1.038 | 93 | 95 | 17.8 | 5,941 | 1,840.4 |
| `m2y`, am | per age | 541 | 10 | 126 | 1.048 | 114 | 80 | 45.6 | 4,484 | 6,131.9 |
| `m2y`, am | by year | 101 | 10 | 89 | 1.038 | 101 | 70 | 48.5 | 4,322 | 6,199.2 |

The by-year form is the built density with its age-to-level map changed, so
the two forms differ in nothing else, and its per-age fits reproduce the
panels' to the digit.[^grouped] Where effects move by year the reserves agree
(mean absolute difference 0.10 sd) and by-year wins elpd in 30 of 30 books;
where they move by quarter it loses 361 of elpd in 30 of 30 and the sd is 59%
wider; on the seasonal book it loses 70 in 20 of 20. On 10 monthly books it
takes 89 s against 126 s, with R-hat at most 1.07 against 1.62, wins elpd by 67
in 10 of 10 and keeps the reserve within 0.14 sd; D on 10 books says nothing.

## Seasonality

| Pair | Valuation | Mean less outcome, sd (without the factor) | Reserve sd ratio | `sigma_pd` ratio | Coverage | D (critical) |
|---|---|---:|---:|---:|---:|---:|
| model 1, aa | year end | -0.51 (-0.28) | 1.02 | 1.00 | 80 | 35.1 (30.4) |
| model 1, aq | year end | -0.67 (-0.23) | 1.54 | 1.62 | 90 | 38.0 (30.4) |
| model 1, aq | end of Q3 | -0.32 (-0.31) | 1.50 | 1.64 | 95 | 22.7 (30.4) |
| model 1, aq | end of Q2 | 0.27 (-0.33) | 1.47 | 1.64 | 95 | 16.6 (30.4) |
| model 1, qq | year end | -0.94 (-0.25) | 1.17 | 1.22 | 90 | 58.7 (30.4) |
| model 2, aq | year end | 0.41 (0.49) | 1.06 | 1.00 | 80 | 28.9 (30.4) |
| model 2, aq | end of Q2 | 0.18 (0.04) | 0.99 | 1.01 | 75 | 22.6 (30.4) |

The world's factors are 1.3, 1, 0.7 and 1 by calendar quarter, and they raise
mean future paid by about 3% since early-year quarters weigh more. Each pair is
the same 20 books with and without the factor. Model 1 reads the wiggle as
noise on sub-annual cells (`sigma_pd` up 62%, the sd up about 50%); its mean
moves 0.6 sd at the end of Q2, where paid runs ahead of the trend, and not at
the end of Q3, where the factors' cumulative deviation is 0. On quarterly
origins the mean is 0.94 sd low and D fails. Model 2's per-age levels on annual
origins take the season up, since age and calendar quarter coincide there.

## Zero increments

Under a Poisson number of payments spread over `m2y`'s pattern (200 books), the
share of books with an empty cell, which model 2's log-normal likelihood on
increments refuses: at 500 payments an origin 99% annually and every book on
quarterly and monthly cells; at 2,000, 64% annually and every other book; at
10,000, 15%, 90% and 100%.[^harness]

## Not run

Panels of 50 on the costliest arms: `m2y` monthly, `m2qo` and `earned`
on quarterly origins with monthly development have 20 books each, and every
default-sampler arm a few books (model 2 monthly one, `m2qo` and `earned` `qm`
none). The rate and seasonal arms have 20 books, as the tables say. Levels by
development year ran only under the short sampler, on annual origins. Not
run at all: monthly origins, semiannual grains, the one-year view and the
backtest on these books, paid and incurred or one response alone, prototypes of
a seasonal term, of correlated residuals or of earned exposure, and a Stan
reference for any arm.

## Reproducing

An arm is `world:grain:model:prior:sampler`. Each panel appends to
`target/compartmental-subannual/<arm>.csv` and skips books already there:

```text
SUBANNUAL_ARMS=persist:am:m1:preset:study SUBANNUAL_BOOKS=50 RAYON_NUM_THREADS=4 \
  cargo test --release -p prospicio-validation --features slow-bayes \
  --test reserving_compartmental_subannual panel -- --ignored --nocapture
SUBANNUAL_EXPORT=m2y:aq SUBANNUAL_BOOKS=30 \
  cargo test --release -p prospicio-validation --features slow-bayes \
  --test reserving_compartmental_subannual export_books -- --ignored
SUBANNUAL_GROUPED=m2y_aq SUBANNUAL_BOOKS=30 cargo test --release \
  -p prospicio-reserving --features bayes --lib grouped_against_per_age \
  -- --ignored --nocapture
uv run --no-project python validation/scripts/compartmental_subannual_summary.py
```

The tests `prior_bands` and `zero_cells` run the same way. The summary script
prints every arm in the directory; 50 books unless a table says otherwise.
