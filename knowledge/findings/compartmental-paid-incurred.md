---
type: Finding
title: Compartmental fitted to paid and incurred matches its own Stan reference, and its reserve is 1.3% above paid and outstanding on company 337
description: Compartmental with Fitted::PaidAndIncurred (incurred replaces outstanding as the second response, mean RLR (OS + RRF PD) times premium) matches model 1 with that one change reconstructed in Stan on company 337 at all 3,040 reference-row comparisons of a 20-seed sweep; its total reserve is 175,318 (sd 9,469) against Stan's 175,570 (9,478), and 1.3% above the paid and outstanding fit's 173,341 (10,076), with a 6% smaller sd.
tags: [bayes, reserving, compartmental, stan, incurred, clrd, validation]
status: draft
generated: { by: claude-code/local-session, at: 2026-10-22T16:00:00-07:00 }
sources:
  - id: test
    resource: ../validation/tests/reserving_compartmental_incurred.rs
    title: The parity tests and the seed sweep
  - id: csv
    resource: ../validation/reference/reserving_compartmental_incurred_stan.csv
    title: Stan summaries of the paid and incurred model
  - id: stan
    resource: ../validation/scripts/stan/compartmental_gm1_incurred.stan
    title: The Stan program, compartmental_gm1.stan with the incurred mean
  - id: script
    resource: ../validation/scripts/reserving_compartmental_incurred_stan.py
    title: The reference generator
  - id: gm1
    resource: compartmental-gm1-parity.md
    title: Model 1 against Stan and the printed posterior (paid and outstanding)
  - id: calibration
    resource: meyers-calibration-compartmental.md
    title: The calibration panel, with 123 cells whose outstanding is at or below zero
  - id: design
    resource: ../docs/design/reserving-compartmental.md
    title: Design note, decision 12
---

# Finding

* `Fitted::PaidAndIncurred` (decision 12, the user's decision of
  2026-10-22) replaces outstanding by incurred as the second response. An
  incurred cell's mean is `RLR_i w_i (OS(t) + RRF_i PD(t))`, with the
  cumulative `PD`; the parameters are those of paid and outstanding, and the
  second sigma (`log_sigma_os`) is the incurred response's.[^design]
* The reference is the Stan program of model 1 with that one change
  (`compartmental_gm1_incurred.stan`), 4 chains of 25,000 draws on company
  337 as the paper trains it (calendar years up to 1996, 90 observations),
  max R-hat 1.0003, no divergences, elpd_loo -809.1. There is no printed
  posterior of this variant, so the Stan reference is the only check besides
  unit tests (the log density against a hand computation to 1e-12, the
  gradient against central differences and against the generic dual pass).[^csv][^stan]
* Ours at the defaults with the paper's preset (seed 0) against Stan, and
  Stan's paid and outstanding fit for comparison:[^test]

  | | Ours, paid and incurred | Stan, paid and incurred | Stan, paid and outstanding |
  |---|---|---|---|
  | `b_ker` | -5.32 | -5.33 (0.82) | -5.39 (0.83) |
  | `b_kp` | -8.57 | -8.57 (0.36) | -8.49 (0.36) |
  | `b_RLR` | 1.37 | 1.38 (0.27) | 1.44 (0.26) |
  | `b_RRF` | -1.21 | -1.20 (0.42) | -1.36 (0.42) |
  | `tau_RLR` | 0.66 | 0.66 (0.15) | 0.64 (0.15) |
  | `tau_RRF` | 0.70 | 0.70 (0.22) | 0.74 (0.22) |
  | `rho` | 0.54 | 0.54 (0.26) | 0.53 (0.26) |
  | `log_sigma_os` (second response) | 8.22 | 8.21 (0.17) | 8.25 (0.16) |
  | `log_sigma_pd` | 6.68 | 6.67 (0.20) | 6.67 (0.20) |
  | total reserve | 175,318 (9,469) | 175,570 (9,478) | 173,341 (10,076) |
  | 1997 reserve | | 31,391 | 31,321 |
  | 1996 reserve | | 52,949 | 51,807 |
  | elpd_loo | -808.9 | -809.1 | -811.0 (not comparable: other second response) |

* A 20-seed sweep (seeds 0 to 19, the ignored `seed_sweep`) compares every
  reference row at every seed: all 3,040 comparisons pass, at tolerances
  borrowed from model 1's (each at least 4.25 times model 1's seed-to-seed
  sd; not re-measured for this variant on 200 seeds). One of the 20 fits
  warns (max R-hat 1.013); the minimum bulk ESS is 601 and there are no
  divergences. A fit takes about 8 s on 16 threads.[^test]
* Against paid and outstanding on the same triangle the reserve is higher
  (1996 by 2.2%, the total by 1.3%) and its sd 6% smaller. Why is not
  investigated: each fit matches its own Stan reference, so the difference
  is between the two models (different second responses and sigmas).
* Incurred is positive where an outstanding can be zero or negative (123
  cells of the calibration panel have an outstanding at or below zero), so
  the log-normal likelihood of model 2 applies to the incurred of any triangle
  that has positive incurred.[^calibration]

[^design]: [Design note, compartmental reserving](../docs/design/reserving-compartmental.md), decision 12.
[^csv]: [Stan summaries](../validation/reference/reserving_compartmental_incurred_stan.csv).
[^stan]: [The Stan program](../validation/scripts/stan/compartmental_gm1_incurred.stan).
[^test]: [The parity tests](../validation/tests/reserving_compartmental_incurred.rs).
[^calibration]: [The calibration panel](meyers-calibration-compartmental.md).
