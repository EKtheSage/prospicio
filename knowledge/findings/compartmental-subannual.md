---
type: Finding
title: On simulated sub-annual books Compartmental is calibrated at every grain in its own world, but model 1's interval narrows with the grain when noise persists, and annual origins with earned premium fail at every grain
description: On books simulated with known parameters and read at annual, quarterly and monthly development, model 1 in its own world covers the total reserve about equally at every grain (82 to 84 of 90) with Meyers' D passing, and finer cells cut the sd from 2,636 to 1,707. When payment noise stays in cumulative paid (at a stated persistence, PERSIST_PHI 25 and PERSIST_RHO 0.95) the sd still falls (1,762 to 1,069) while the error does not (RMSE 1,623 to 2,033) and coverage falls from 88 to 56. Annual origins with earned premium fail D at all three grains with rates pulled towards a swap; quarterly origins of the same books pass narrowly (the cause is an inference). The paper's rate preset stays sensible and matters less as the grain falls. Model 2 with one development level per age mixes badly on monthly cells, where levels by development year do not; by year is badly wrong when effects move within a year. Without a seasonal term, on a strong season (factors 1.3, 1, 0.7, 1), the mean is up to 0.94 sd low and Meyers' D fails at year end (58.7 against a critical 30.4 for model 1 on quarterly origins, 43.9 for model 2 on quarterly origins, 35.1 and 38.0 for model 1 on annual origins) while the sd widens by up to 54%; model 2's annual-origin fits escape only because every annual origin in the study starts in January.
tags: [bayes, reserving, compartmental, grain, quarterly, monthly, simulated, calibration, seasonality, nuts]
status: draft
generated: { by: claude-code/local-session, at: 2026-11-06T06:00:00-08:00 }
sources:
  - id: harness
    resource: ../validation/tests/reserving_compartmental_subannual.rs
    title: The worlds, the panels (ignored), prior_bands, zero_cells and export_books
  - id: grouped
    resource: ../crates/prospicio-reserving/src/bayes/compartmental/subannual_study.rs
    title: The study's grouped_against_per_age (ignored), now through the public DevelopmentLevels setting
  - id: summary
    resource: ../validation/scripts/compartmental_subannual_summary.py
    title: Prints the tables from the panels' CSVs
  - id: runner
    resource: ../validation/scripts/compartmental_subannual_run.py
    title: Every arm, its books and its command, and a runner
  - id: summary-output
    resource: ../validation/reference/compartmental_subannual_summary.txt
    title: The summary script's output on the original run, all books
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
checks of the books. The settings built from them (see "What was decided and
built") have unit tests that CI runs, listed there; the numbers in the tables
are still the panels' own. `target/` is not committed: the raw results are about 60 CSVs (50 MB) there,
and the tables below are the summary script's output on them, kept in
`validation/reference/` (see Reproducing). NUTS draws at a
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
  and the RMSE fall together.[^harness][^summary] `tau_RRF`, true 0.2, has
  posterior means of 0.10 to 0.18 and is covered in 64% to 86% of books at every
  grain: the preset's half-t of scale 0.1 pulls it down, which may be part of
  the gap (an inference, not tested).
* **Persistent noise** (`persist`), at the harness's own settings: independent
  monthly payment noise with variance `PERSIST_PHI = 25` times the month's mean
  payment (an sd of about 2% of a 60,000 ultimate at month 120), summed into
  cumulative paid, and an AR(1) on outstanding with `PERSIST_RHO = 0.95` a
  month. The two numbers were chosen by hand, not fitted to any book, and every
  figure in this bullet is at these values: with less persistence the fall is
  smaller (untested). Under them the sd falls from 1,762 to 1,069, the RMSE rises
  from 1,623 to 2,033 and coverage falls from 88 to 56 (annual, quarterly,
  monthly development); `k_p`'s central 90% holds the truth in 80% of books
  annually and 18% monthly. Model 1 with incremental paid keeps the sd (3,356 to
  3,605) and covers 98 to 100, too wide (the sd of the standardised errors is
  0.48 to 0.75), but it is not better throughout: its RMSE is 1,510 annually
  (cumulative: 1,623) and 2,151 and 2,731 at quarterly and monthly development
  (cumulative: 1,903 and 2,033), and it fails D at annual (28.1 against a
  critical 19.2) and narrowly at quarterly (19.9) because its percentiles bunch
  in the middle; monthly D is 11.0. The same model's intervals were already
  narrow on annual Schedule P triangles.[^calibration]
* **Earned premium** (`earned`): annual origins fail D at all three grains,
  with `k_er` at 1.92, 1.22 and 1.11 against 3.32, and 45 of 50 monthly fits
  have draws with `k_er <= k_p`. Quarterly origins of the same books pass: D
  17.8 against a critical 19.2, coverage 92, RMSE 2,007, `k_er` 3.30. These are
  the short-sampler panels of 50 books; the default sampler ran on 3
  quarterly-origin books (the reserve's mean moved by at most 0.038 sd) and on
  no annual-origin `earned` book, so "annual fails, quarterly passes" is a
  short-sampler result, and quarterly D is under its critical value by 1.4
  points. Every annual origin is the sum of four quarterly ones, which suggests
  the misfit is the model's single start at `t = 0` (an inference, not tested).
  The comparison changes more than the start: the quarterly fit has 40 origins
  against 10, so it estimates the effects' variance from four times as many, and
  an annual origin's curve is a mixture of four curves with different effects.
  Neither is separated from the start here; a test would fit the annual books
  with the exposure spread over the year (open decision 13 (b)).
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
the reserve's mean by at most 0.045 sd and, on every converged arm, its sd by
under 1%, so the panels' reserves stand, and their parameter coverage carries
the short run's noise. The short run's divergences are not rare. Of 50 books, 46, 45 and 43 of the `iid` fits
(annual, quarterly and monthly development), 48, 45 and 34 of the `persist` fits,
50 and 48 of the `earned` fits on annual origins at quarterly and monthly
development and 27 of the quarterly-origin `earned` fits have at least one
divergence (8 of 20 on `earned` `qm`); the median `iid` or `persist` fit has 2 to
3 and the worst 357. Model 2's short fits have none. The default sampler had none in
its 18 model 1 fits. Its rows rest on 5, 5, 5, 3, 5, 3 and 1 books (the sampler
table's Books column), 27 fits in all, too few for a rate, so the claim that the
short run's reserves stand rests on those 27 fits' mean differences. The short run does leave strays: on `iid` monthly one
book of 50 has R-hat 1.60 and 357 divergences. How the short run's trouble grows
with cells and parameters shows in its share of fits with R-hat above 1.05: 4
of 50 at 110 cells and 5 of 50 at 1,320 for model 1 on annual origins, 28 of 50
on quarterly origins (80 more effects); for model 2 5 of 50, 19 of 50 and 8 of
20 at 110, 440 and 1,320 cells, and 11 of 20 on `m2qo`. Model 2's default
sampler is enough on 110 and 440 cells (R-hat at most 1.007, ESS at least 861).
On monthly `m2y` book 0 stays at R-hat 1.62 under the short run and 1.54 after
the default's 48 minutes, with ESS 7 and 1,322 warnings, yet its reserve moves
0.005 sd between the two runs: the trouble is in the rates' parameters, not the
total. It is the one monthly book run at the default. Quarterly
origins cost about the same cells for cells (`earned`, 1,640 cells: 34 s short,
195 s default; model 2 `m2qo`: 91 s short, not run at the default), and
monthly development on quarterly origins (4,920 cells) took 149 s short.

## Rate priors

All rows are the same first 20 books of their world, so that a preset row and
a `ratewide` row compare like for like (the preset's 50-book `iid` rows are in
the calibration table; `iid` monthly has no `ratewide` arm). The standard error
of a coverage at 20 books is 7 points.

| World | Grain | Prior | `k_er` (truth) | `k_er` covered | Reserve coverage | RMSE | sd |
|---|---|---|---:|---:|---:|---:|---:|
| `iid` | aa | preset | 3.29 (3.32) | 95% | 80 | 3,037 | 2,589 |
| `iid` | aa | ratewide | 3.40 (3.32) | 85% | 85 | 2,990 | 2,567 |
| `iid` | aq | preset | 3.33 (3.32) | 85% | 90 | 2,105 | 1,919 |
| `iid` | aq | ratewide | 3.34 (3.32) | 85% | 95 | 2,073 | 1,920 |
| `iid` | am | preset | 3.32 (3.32) | 90% | 90 | 1,537 | 1,693 |
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

On these 20 `iid` books the preset's D at quarterly development is 31.4
against a critical 30.4 (12.4 on all 50), so D on 20 books is noisy.

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

The by-year form has the density of the per-age form with its age-to-level
map changed, and its per-age fits reproduce the panels' to the digit.[^grouped]
The study's regrouping, though, drew the effect of an age with no observed cell
once for each age, as the per-age form does, whereas the built setting
(`DevelopmentLevels::ByYear`) draws it once for each year, so the ages of one
year share it in the predictive. The fits' posteriors are the same and the
predictive draws of unobserved cells differ; the panels were not rerun, so the
table below is the study's regrouping. Where effects move by year the reserves agree
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
| model 2, qq | year end | -0.74 (0.27) | 1.26 | 1.60 | 95 | 43.9 (30.4) |

The world's factors are 1.3, 1, 0.7 and 1 by calendar quarter, and they raise
mean future paid by about 3% since early-year quarters weigh more. Each pair is
the same 20 books with and without the factor. The worst case first: model 1 on
quarterly origins at year end has its mean 0.94 sd under the outcome (0.25
without the factor) and D of 58.7 against a critical 30.4. D also fails at year
end for model 1 on annual origins, with annual and quarterly development (35.1
and 38.0), and for model 2 on quarterly origins (43.9; the mean is 0.74 sd under
the outcome against 0.27 over without the factor, a shift of 1.01 sd). It
passes at the end of Q2 and Q3 (16.6 to 22.7) and for model 2 on annual origins
(28.9 and 22.6). Coverage holds at 75 to 95 because the sd widens, so the error
lies in the centre. Model 1 reads the wiggle as noise on sub-annual cells
(`sigma_pd` up 62%, the sd up to 54%, and 2% with annual development); against
the plain book its mean moves 0.6 sd at the end of Q2, where paid runs ahead of
the trend, and not at the end of Q3, where the factors' cumulative deviation is
0. Model 2's per-age levels take the season up on annual origins only because
of this study's layout: every annual origin starts in January, so calendar
quarter and age are the same thing there (`sigma_pd` ratio 1.00, the mean within
0.14 sd of the plain book's). That is an artifact of the layout, and real earned
books do not have it. With origins starting in all four quarters (`m2seasonq`,
`m2qo` with the factor, 20 books) the levels cannot absorb the season:
`sigma_pd` rises 60%, the sd 26% and D is 43.9.

## Zero increments

Under a Poisson number of payments spread over `m2y`'s pattern (200 books), the
share of books with an empty cell, which model 2's log-normal likelihood on
increments refuses: at 500 payments an origin 99% annually and every book on
quarterly and monthly cells; at 2,000, 64% annually and every quarterly and monthly book; at
10,000, 15%, 90% and 100% (annual, quarterly, monthly).[^harness]

## What was decided and built

The user accepted the recommendations of open decisions 9 to 13 on 2026-11-06
(design note, decision 14, "As built").[^design]

* Rate priors: the preset, per year on every grain, said in the help.
* Levels: `DevelopmentLevels::{PerAge, ByYear}`, `PerAge` by default and
  unchanged. A year is `(age_months - 1) / 12` counted from each origin's own
  start, so it does not depend on the calendar month an origin starts in, on
  every grain. At annual grain the two settings are the same model. Model 1
  with `ByYear` is an error.
* Seasonality: no term; the help says what the study found. The calendar-quarter
  factor is not built.
* A warning on a fit with cumulative paid on a development grain finer than a
  year, asking for a look at the residuals, on calibrated fits too. Incremental
  paid is not made the default. Correlated residuals are not built.
* Earned exposure: the help says to use quarterly origins; the setting that
  spreads the exposure over the origin's period is not built.

CI runs the debug unit tests of the levels (the count of years, their
independence of the origin's calendar month, one shared effect for a year no
cell observes, the gradient against central differences, a cell's parameters
following its level in the one-year view, the error for model 1, the prior
predictive drawing the fit's levels) and of the
warning (it fires for cumulative paid on quarterly cells and not for
incremental paid or at annual grain), and the `bayes` job runs the release
tests (the one-year conformance with levels by year, annual identity of the
two settings at a fixed seed, a quarterly fit, the diagonal backtest). None of
the study's numbers is among them.

Building it found two things the study did not. `Parameters::at` in the
one-year view indexed development effects by age, which is wrong for levels
by year and was fixed. And on a triangle whose newest development year opens
in the coming year, the Pareto k of the reweighted simulations is large and
`gpd_fit` returned NaN for 18 of 200 simulations on low-noise data (none with a
30% wobble); the conformance check's replay comparison is now NaN-aware.
#782 found the cause by dumping those log ratios (the triangle of
`the_one_year_view_by_development_year_conforms` with an 8% wobble, model 2
by development year, 2 chains of 100 and 100). They were not ties: the 200
log ratios spanned thousands of units (the smallest -6,900 to -23,000 after
shifting the largest to 0), so the 41st largest, the tail's cutoff, sat at
-1,400 to -7,700, and `exp` underflowed to 0 there and for 16 to 37 of the
40 tail ratios. With a quarter or more of the exceedances 0, Zhang and
Stephens' grid divided by a zero quartile. The NaN k also set every tail
weight to the largest one (`NaN.min(0.0)` is 0). `psis` now gives such a
tail an infinite k and the raw weights; across the conformance check's
views every one of the 181 NaN values became infinite and no other k
moved. In the default view of 200 simulations 18 have an infinite k, all
flagged as before. Model 2's default (`RefitFlagged` with the light refit)
refits them as it did; under `MomentMatchThenRefit` the 18 now reach
moment matching, which rescues none of them (nor any of the other 182
flagged), and they are refitted.
Neither affected the study's numbers, which ran the per-age form and the
test-only regrouping outside the one-year view.

## Not run

Panels of 50 on the costliest arms: `m2y` monthly, `m2qo` and `earned`
on quarterly origins with monthly development have 20 books each, and every
default-sampler arm a few books (model 2 monthly one, `m2qo` and `earned` `qm`
none). The rate and seasonal arms (`m2seasonq` too) have 20 books, as the tables say. Levels by
development year ran only under the short sampler, on annual origins (the
built setting has been run on one quarterly triangle, in the unit tests). Not
run at all: monthly origins, semiannual grains, the one-year view and the
backtest on these books, paid and incurred or one response alone, prototypes of
a seasonal term, of correlated residuals or of earned exposure, and a Stan
reference for any arm.

## Reproducing

The raw results are the CSVs in `target/compartmental-subannual/`, which is
gitignored: a clean checkout has none. Two files in `validation/reference/`
hold what they say: `compartmental_subannual_summary.txt` (the summary script
on all books) and `compartmental_subannual_summary_first20.txt` (the same with
`SUBANNUAL_SUMMARY_BOOKS=20`, which every comparison on "the same first 20
books" reads).[^summary] `validation/scripts/compartmental_subannual_run.py`
lists every arm, its number of books and the command that ran it
(`--list`), and runs them (`--run --jobs 2`, resumable since a panel skips
books already in its CSV). A full run is 1,657 fits, 13.0 hours of summed fit
time (12.4 for the first 1,637 and 0.6 for `m2seasonq`, measured as the sum of
the CSVs' `seconds`) and, at two fits at a time and 4 threads each, 6 to 7
hours of wall time (the original queues ran 23:14 on 2026-11-05 to 05:49 on
2026-11-06, with other work on the machine). An arm is
`world:grain:model:prior:sampler`. Each panel appends to
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
prints every arm in the directory, the default-sampler pairs and the R-hat
shares; 50 books unless a table says otherwise. Comparisons on "the same first
20 books" are its output under `SUBANNUAL_SUMMARY_BOOKS=20`.
