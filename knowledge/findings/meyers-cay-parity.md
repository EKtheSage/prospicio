---
type: Finding
title: Meyers' CAY against his Stan code and his printed results
description: MeyersModel::Cay matches Meyers' transcribed CAY.R Stan code on group 353 incurred at every one of 200 seeds, within tolerances of at least 4.1 times its own Monte Carlo error, and his printed results (total 39,193, rho 0.1709, elpd_loo 68.65) within their combined error; on these data sigma[10] is near 0.013, so the recursion in the prediction is invisible to the parity and a unit test checks it instead.
tags: [bayes, reserving, meyers, cay, stan, clrd, validation]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-13T12:00:00-07:00 }
sources:
  - id: test
    resource: ../validation/tests/reserving_meyers_cay.rs
    title: The parity tests and the seed sweep
  - id: meyers
    resource: ../crates/prospicio-reserving/src/bayes/meyers.rs
    title: prospicio_reserving::bayes::Meyers with MeyersModel::Cay
  - id: csv
    resource: ../validation/reference/reserving_meyers_cay_stan.csv
    title: Stan summaries and Meyers' printed numbers
  - id: script
    resource: ../validation/scripts/reserving_meyers_cay_stan.py
    title: The reference generator
  - id: monograph
    resource: https://www.casact.org/sites/default/files/2021-02/08-Meyers.pdf
    title: Meyers (2019), CAS Monograph 8, 2nd ed., sec. 8, Tables 8.1 to 8.3 (pp. 39 and 41)
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decisions 7 and 10, PR 5)
---

# Finding

* Group 353 incurred (total ultimate mean and sd; ours at the defaults,
  seed 0; Stan 4 x 25,000 draws of the transcribed `CAY.R`; Meyers'
  printed):[^test][^csv][^monograph]

  | | Ours | Stan | Meyers |
  |---|---|---|---|
  | Total ultimate | 39,178 (1,834) | 39,142 (1,862) | 39,193 (1,859) |
  | `rho` | 0.1686 (0.2075) | 0.1729 (0.2069) | 0.1709 (0.2071) |
  | `logelr` | -0.3947 (0.0144) | -0.3947 (0.0150) | -0.3945 (0.0150) |
  | elpd_loo | 69.05 | 68.47 | 68.65 |

  CAY's total sd is about 15% above CRC's on the same column (Stan 1,862
  against 1,616), and elpd_loo favours CRC on this triangle (Meyers 70.97
  against 68.65), as the monograph says.
* Monte Carlo error of the defaults (4 chains of 2,000 draws, 10,000
  simulations), the seed-to-seed sd over seeds 0 to 199 with `s` Stan's
  posterior sd: at most `0.025 s` on a mean, 3.3% on a parameter sd, 2.4%
  on an origin's reserve sd (1989 to 1996), 1.9% on the sd of the 1988 to
  1996 sum, `0.063 s` on a percentile, 0.019 on the correlation of
  neighbouring origins' reserves and 0.43 on elpd_loo. The total sds are
  heavy-tailed through the 1997 log-normal, as for CRC and CSR, and more
  so: +66% at seed 38 against a seed-to-seed sd of 5.9%. They get 75%;
  every other tolerance is at least 4.1 times its row's spread, and seeds
  0 to 199 pass all 136 rows (at most 0.88 of a tolerance).[^test]
* What the parity cannot see: CAY's recursion in the prediction (each
  origin's mean at age 10 takes `rho` times the previous origin's
  residual there). With `sigma[10]` near 0.013, setting `rho` to 0 in the
  prediction alone moves the neighbour correlations of reserves by at most
  0.03 (from 0.50 to 0.47 for 1990/1991) and the total sd by 0.1%, under
  2 Monte Carlo errors, and passed every parity row. Most of that
  correlation (0.2 to 0.5) comes from the shared parameters. A unit test
  checks the recursion at `sigma_m = 0.5`, `rho = 0.8` (lag-one correlation
  `rho / (1 + rho^2)`, none at lag two) and fails without it.[^meyers]
* The defaults (target acceptance 0.95, 2,000 kept draws, the pilot-run
  scale of CRC and CSR) passed every diagnostic at 199 of 200 seeds; seed
  73 had one divergent transition in 8,000 draws. The largest R-hat was
  1.0073. A default fit takes 0.5 s in a release build.[^test]
* The Stan transcription itself agrees with Meyers' printed run within its
  Monte Carlo error (total 0.13% apart, p_loo 15.72 against 15.64): max
  R-hat 1.0002, no divergence, Pareto k up to 0.92.[^script]
