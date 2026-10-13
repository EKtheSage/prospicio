---
type: Finding
title: Meyers' CRC and CSR against his Stan code and his printed results
description: Meyers (prospicio's CRC and CSR) matches Meyers' own Stan programs on group 353 paid and incurred at every one of 200 seeds, within tolerances of at least 4 times its own Monte Carlo error, and his printed totals within 0.3%; the match needs his 1e-5 floor on each variance step, and sampling needs a pilot-run scale, since the joint mode sits deep in the funnel of the late sigmas.
tags: [bayes, reserving, meyers, crc, csr, stan, clrd, validation]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-12T17:30:00-07:00 }
sources:
  - id: test
    resource: ../validation/tests/reserving_meyers.rs
    title: The parity tests and the seed sweep
  - id: meyers
    resource: ../crates/prospicio-reserving/src/bayes/meyers.rs
    title: prospicio_reserving::bayes::Meyers
  - id: csv
    resource: ../validation/reference/reserving_meyers_stan.csv
    title: Stan summaries and Meyers' printed numbers
  - id: bindings
    resource: ../python/tests/test_reserving_meyers.py
    title: The Python binding tests (R's are R/prospicio/tests/test-reserving-meyers.R)
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decisions 7 and 10)
---

# Finding

* Totals on group 353 (total ultimate mean and sd; ours at the defaults,
  seed 0; Stan 4 x 25,000 draws of Meyers' transcribed code; Meyers'
  printed):[^test][^csv]

  | Case | Ours | Stan | Meyers |
  |---|---|---|---|
  | CRC paid | 40,119 (2,538) | 40,136 (2,497) | 40,121 (2,487) |
  | CRC incurred | 39,080 (1,541) | 39,132 (1,616) | 39,147 (1,642) |
  | CSR paid | 37,612 (2,487) | 37,537 (2,391) | 37,597 (2,401) |
  | CSR incurred | 38,168 (1,588) | 38,165 (1,567) | not printed |

  elpd_loo: ours 47.90, 71.42, 49.82, 71.12; Stan 48.04, 71.22, 49.56,
  71.26; Meyers 47.80, 70.97, 49.76. CSR paid `gamma`: ours 0.0447 (sd
  0.0286), Stan 0.0450 (0.0280), Meyers 0.0446 (0.0282).
* Monte Carlo error of the defaults (4 chains of 2,000 draws, 10,000
  simulations), the seed-to-seed sd over seeds 0 to 199 with `s` Stan's
  posterior sd: at most `0.026 s` on a mean (CSR incurred `sigma[8]`),
  3.6% on a parameter sd, 2.6% on the reserve sd of origins 1989 to 1996,
  4.3% on the total reserve's sd, `0.063 s` on a reserve percentile and
  0.43 on elpd_loo. The sd of 1997's reserve is heavy-tailed: that reserve
  rides on the one-cell `alpha[10]` (posterior sd 0.33) through a
  log-normal, so its seed-to-seed sd (up to 7.5%) understates the tail.
  At 20% it failed at 12 of 200 seeds (CRC incurred +63% at seed 170, CSR
  paid +23% at seed 38); the totals inherit a milder tail (CRC incurred
  +37% at seed 170). A 20-seed sweep had put the spread at 4.8% and missed
  this. So the parity drops 1997's reserve sd row (its 5%, 50% and 95%
  percentiles check its spread) and gives the total sds 50%. The
  tolerances are `0.15 s` on means, 15% on parameter sds, 20% on origin
  reserve sds, 50% on total sds, `0.35 s` on percentiles and 2.0 on
  elpd_loo; seeds 0 to 199 pass every row, using at most 0.86 of a
  tolerance, seed 0 at most 0.41.[^test][^design]
* Meyers' printed numbers sit within each run's Monte Carlo error of the
  Stan rerun (totals within 0.16%, sds within 1.6%), so his runs and the
  transcription agree. Their tolerances add his run's error to ours:
  totals 1%, total sds 50% (the same heavy tail), `logelr` and `gamma` means
  0.15 printed sd and sds 15%, elpd 2.[^csv]
* The floor matters: Meyers' `a_ig < 1e5` makes each variance step
  `U(1 - exp(-1e-5), 1)`, not `U(0, 1)`. Without the floor the late
  `sigma_d` on incurred data came out 10 to 20% below Stan's (outside 4
  Monte Carlo errors), elpd_loo 3.5 above, and CRC incurred gave 80
  divergent transitions in 4,000 draws; with it, none.[^meyers]
* The joint posterior mode is not a usable centre: on CRC paid, Newton
  creeps toward `sigma[10]` near 5e-4 (`u_10` near -15) against a posterior
  mean of 0.020, along the funnel where the late sigmas and the late
  residuals shrink together, and did not converge in 200 iterations. A
  pilot run (whitened at a least-squares start, then by the pilot draws'
  mean and covariance, as Stan's dense metric adapts) samples it.[^meyers]
* Defaults: at target acceptance 0.8 the paid fits diverged at nearly
  every seed (1 to 56 in 8,000 draws); at 0.95 none in 120 fits. With
  1,000 kept draws per chain the late sigmas' R-hat passed 1.01 (at most
  1.016) in 5 of 120 fits; with 2,000 every fit passes every
  threshold.[^test]
* Runtime: a default fit takes 0.3 s (CRC) to 0.5 s (CSR) in a release
  build and 6 to 8 s in a debug one; the validation test file about 34 s
  in a debug build.[^test]
* Through the Python and R bindings (release builds), the default CRC and
  CSR fits of group 353 paid at seed 0 with 2,000 simulations pass every
  diagnostic, land within 0.2% of Meyers' printed totals (CRC 40,043, CSR
  37,588 against 40,121 and 37,597) and within 0.1 of
  his elpd, and the seven Python tests take about 8 s in all.[^bindings]
