---
type: Reference
title: Meyers, Stochastic Loss Reserving Using Bayesian MCMC Models (CAS Monograph 8), reproduced in prospicio
description: The guide reserving-meyers.md recomputes Monograph 8's printed results on group 353 (Mack and the ODP bootstrap with the outcome's percentile, CRC, CSR and CAY with parameters, totals, percentiles and elpd_loo); every printed value agrees within the combined Monte Carlo error, measured over 20 seeds. Facts kept here are the outcome percentiles, the seed-to-seed spread, Table 5.1's scrambled text layer, and why a guide cannot print NUTS digits.
resource: https://www.casact.org/sites/default/files/2021-02/08-Meyers.pdf
tags: [bayes, reserving, meyers, crc, csr, cay, mack, odp, validation, guides, reference, parity]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-17T18:00:00-07:00 }
sources:
  - id: monograph
    resource: https://www.casact.org/sites/default/files/2021-02/08-Meyers.pdf
    title: Glenn Meyers (2019), Stochastic Loss Reserving Using Bayesian MCMC Models, 2nd ed., CAS Monograph 8
  - id: guide
    resource: ../docs/guides/reserving-meyers.md
    title: The guide that runs the reproduction (its checked blocks run in CI)
  - id: test
    resource: ../python/tests/test_guides.py
    title: The harness that runs the guide and checks every printed line
  - id: code
    resource: meyers-monograph-8.md
    title: Meyers' Stan code and its quirks (prior sd 3.162, the 1e-5 variance floor, ceiling)
---

# What it is

* Meyers tests reserving models by their reputation: fit each of 200 CAS
  Loss Reserve Database triangles (50 each of CA, PA, WC, OL), read the
  percentile of the actual 120-month total in the predictive distribution,
  and test the 200 percentiles for uniformity with a p-p plot and the
  Kolmogorov-Smirnov statistic `D = max |p_i - 100 i / n|`, critical value
  `136 / sqrt(n)` (19.2 for 50, 9.6 for 200). Mack on incurred is too light
  in the tails, Mack and the ODP bootstrap on paid predict too high; CRC
  repeats both, CSR fixes paid, CAY improves incurred.[^monograph]
* The 200-triangle panel is not in the repository's data, so the guide
  reproduces the group 353 tables and the test's mechanics only.[^guide]

# Printed values and prospicio's

* Mack, a log-normal by moments for the percentile (Tables 3.1 and 4.1):
  incurred 38,914, sd 1,057, outcome 40,061 at 86.03 (prospicio 38,914,
  1,056.8, 86.06); paid 39,177, sd 1,442, outcome 40,000 at 72.02
  (prospicio 39,177, 1,442.5, 72.00). Deterministic; the last-digit gaps are
  rounding.[^guide]
* ODP bootstrap paid (Table 4.1): 39,193, sd 1,389, 73.91. prospicio's
  `OdpBootstrap` over seeds 0 to 19 (10,000 simulations): 39,197 (sd 16),
  1,393 (12), 73.78 (0.57); seed 0 gives 39,231, 1,410, 72.73.[^guide]
* Outcome percentiles of the Bayesian models (Meyers counts simulated
  totals at or below the outcome): CRC paid 51.88, CRC incurred 74.75, CSR
  paid 86.26, CAY incurred 73.24 (Tables 5.2, 5.3, 7.2, 8.2). prospicio at
  the defaults over seeds 0 to 19 (Windows): 51.39 (0.81), 75.74 (0.66),
  86.58 (0.68), 73.55 (0.68). Totals' means over the seeds are within 40 of
  the book, sds of the totals within 3%, elpd_loo within 0.5.[^guide]
* Table 5.1's text layer (pdftotext) scrambles the right-hand column;
  read from the rendered page, `sigma[1]` is 0.2965 (sd 0.1034) and
  `sigma[10]` 0.0202 (sd 0.0128), with `beta[6..9]` near zero.[^monograph]

# For a guide that shows NUTS results

* NUTS draws at a fixed seed differ between Windows and Linux (the last bit
  of `exp` and `ln` diverts the trajectories), so a CI-checked guide cannot
  print a Meyers or `BayesOdp` value to its digits. The guide prints the
  book's value, a tolerance (0.15 posterior sd on means, 15% on parameter
  sds, 1% on totals, 50% on the totals' heavy-tailed sds, 5 points on
  percentiles, 2 on elpd_loo) and whether prospicio's value falls inside,
  and shows prospicio's own values in a `python no-run` block recorded from
  a Windows run. Deterministic results (Mack) and the ODP bootstrap print
  their digits.[^guide]
* Timing (release build): the four default fits with 10,000 simulations
  and their elpd_loo take about 1 s each, the guide about 8 to 10 s; the
  default one-year view of CRC paid takes 62 s (2,000 simulations, 1,778
  moment-matched, 202 refitted, one warning from failed refit diagnostics),
  so it is a no-run block; `refit="reweight_only"` takes 1 s.[^guide]
