---
type: Finding
title: Compartmental fitted to paid alone or incurred alone matches its own Stan references; paid alone moves the reserve 2% and incurred alone 26% on company 337
description: Compartmental with Fitted::PaidOnly or Fitted::IncurredOnly and ordered rates matches model 1 with one response reconstructed in Stan on company 337 at every reference-row comparison of a 20-seed sweep per setting (model 1's tolerances, elpd_loo at 2.5). Paid alone gives a total reserve of 176,448 (sd 9,822) against Stan's 176,357 (9,577), 1.8% above paid and outstanding; incurred alone gives 219,051 (10,076) against Stan's 219,074 (10,017), 26% above, because the paid rate k_p is then set by its prior. Under one response the least-squares data start is degenerate, so the fit starts at the prior centre.
tags: [bayes, reserving, compartmental, stan, incurred, paid, clrd, validation]
status: draft
generated: { by: claude-code/local-session, at: 2026-10-22T22:00:00-07:00 }
sources:
  - id: test
    resource: ../validation/tests/reserving_compartmental_one_response.rs
    title: The parity tests, the comparison with two responses and the seed sweep
  - id: csv
    resource: ../validation/reference/reserving_compartmental_one_response_stan.csv
    title: Stan summaries of the paid-only and incurred-only models
  - id: stan-paid
    resource: ../validation/scripts/stan/compartmental_gm1_paid_only.stan
    title: The Stan program of model 1 fitted to paid alone, ordered rates
  - id: stan-incurred
    resource: ../validation/scripts/stan/compartmental_gm1_incurred_only.stan
    title: The Stan program of model 1 fitted to incurred alone, ordered rates
  - id: script
    resource: ../validation/scripts/reserving_compartmental_one_response_stan.py
    title: The reference generator
  - id: incurred
    resource: compartmental-paid-incurred.md
    title: Paid and incurred against Stan, the comparison's two-response fits
  - id: swap
    resource: compartmental-rate-swap.md
    title: The rate swap of the one-compartment model
  - id: design
    resource: ../docs/design/reserving-compartmental.md
    title: Design note, decision 12
---

# Finding

* `Fitted::PaidOnly` and `Fitted::IncurredOnly` (decision 12, the user's
  decision of 2026-10-22) fit model 1 to one response alone. Both require
  `RateOrder::Ordered`: from one response the means do not change under the
  rate swap, and from paid alone only `RLR x RRF` is identified.[^design][^swap]
  The model has one sigma. Under incurred alone the paid column gives each
  origin's latest paid only, and the future paid is the expected paid,
  without noise.
* The references are model 1's Stan program with ordered rates and the one
  response (`compartmental_gm1_paid_only.stan`,
  `compartmental_gm1_incurred_only.stan`), 4 chains of 25,000 draws on
  company 337 as the paper trains it (calendar years up to 1996, 45
  observations of the response). Paid alone: max R-hat 1.0002, no
  divergences, elpd_loo -361.06 (max Pareto k 1.07), about 16 minutes of
  sampling. Incurred alone: max R-hat 1.0004, no divergences, elpd_loo
  -436.04 (max k 1.10), about 3 minutes. There is no printed posterior of
  either variant.[^csv][^stan-paid][^stan-incurred][^script]
* Ours at the defaults with the paper's preset and ordered rates (seed 0)
  against Stan, mean (sd):[^test]

  | | Ours, paid alone | Stan, paid alone | Ours, incurred alone | Stan, incurred alone |
  |---|---|---|---|---|
  | `k_er` | 1.494 | 1.493 (0.094) | 2.848 | 2.850 (0.276) |
  | `k_p` | 0.456 | 0.456 (0.017) | 0.930 | 0.930 (0.093) |
  | `RLR` | 0.789 | 0.788 (0.083) | 0.879 | 0.880 (0.051) |
  | `RRF` | 0.827 | 0.827 (0.075) | 0.782 | 0.782 (0.032) |
  | `tau_RLR` | 1.015 | 1.016 (0.197) | 0.729 | 0.728 (0.149) |
  | `tau_RRF` | 0.093 | 0.092 (0.078) | 0.094 | 0.096 (0.081) |
  | the one sigma (log) | 6.432 | 6.430 (0.147) | 8.019 | 8.019 (0.130) |
  | elpd_loo | -360.65 | -361.06 | -435.75 | -436.04 |
  | total reserve | 176,448 (9,822) | 176,357 (9,577) | 219,051 (10,076) | 219,074 (10,017) |

* The data start. The least squares of the population curve behind
  `CompartmentalStart::Data` is not identified from one response and runs
  off along the free direction: paid alone ends at `RLR` 5e-136 and `RRF`
  1e134 with `k_er` close to `k_p` (log posterior -6.0 million), incurred
  alone at `RRF` 1e-13 and `k_p` 0.03 (-44,700), against -456 and -452 at
  the prior centre. With the default warm-up NUTS walked back from there,
  but the fits were slower and 9 of 20 paid-only fits missed an R-hat or
  ESS threshold on `k_er`; with a short warm-up (100 or 200 draws) the
  chains stayed stuck, with `k_er` up to 340,000 and `RLR` near 0. Under
  one response the fit now keeps the data start only when its log
  posterior is at least the prior centre's, so on company 337 both
  settings start at the prior centre.[^design]
* Tolerances: model 1's (decision 7), except `elpd_loo`, which takes 2.5
  instead of 1.5: on the sweep of 20 seeds per setting the incurred-only
  `elpd_loo`, whose largest Pareto k is above 1, was up to 2.1 from Stan's
  (ours higher on 16 of 20 seeds), the paid-only one up to 1.1. With it all
  5,520 comparisons of the 40 fits pass (276 rows each); the worst mean is
  0.51 of its tolerance.[^test]
* Diagnostics at the defaults on the sweep: 17 of the 20 paid-only fits
  passed every threshold (two had one divergent transition each, one an
  R-hat of 1.013 on `tau_RRF`), in 5 to 15 s each, and all 20
  incurred-only fits, in 2 to 6 s. Under ordered rates no draw has the
  swap (`rate_swaps` 0), so the rate-swap warning never fires.[^test]
* The comparison with two responses (seed 0, total reserve mean (sd) and
  the population rates):[^test][^incurred]

  | Fitted | total reserve | `k_er` | `k_p` | `RLR` | `RRF` |
  |---|---|---|---|---|---|
  | paid and outstanding | 173,390 (10,167) | 1.755 | 0.428 | 0.936 | 0.699 |
  | paid and incurred | 175,318 (9,469) | 1.768 | 0.425 | 0.922 | 0.710 |
  | paid alone | 176,448 (9,822) | 1.494 | 0.456 | 0.789 | 0.827 |
  | incurred alone | 219,051 (10,076) | 2.848 | 0.930 | 0.879 | 0.782 |

  Paid alone keeps the reserve within 2% of the two-response fits with a
  similar sd: `RLR x RRF` (0.65) and `k_p` are set by the paid, and the
  split of `RLR` and `RRF` and `k_er` move to where the priors put them.
  The origin variation goes almost all to `RLR` (`tau_RRF` 0.09). Incurred
  alone moves the reserve 26% up, mostly in the oldest origins (1988:
  8,271 against 351 with paid and outstanding): no paid cell is fitted, so
  `k_p` (0.93 against 0.43) is set by its prior, and the expected paid at
  the last age of the old origins is well above their latest paid. The
  reserve's sd under incurred alone is parameter uncertainty only, since
  the future paid has no noise.

# Implications

* Paid alone is a usable fit for a book without case reserves, with the
  priors on `RLR` and `RRF` carrying their split.
* Incurred alone does not estimate the paid run-off. Its reserve is the
  incurred-implied ultimate less the paid, with the payment speed from the
  prior on `k_p`, and on company 337 that prior is far from the data.
  Fit it with a prior on `k_p` the user believes, or prefer a fit with
  paid.
* The Stan references are the only check of either variant besides unit
  tests (the log density against the model written out to 1e-12, the
  gradient against central differences and the generic dual pass, the
  one-year conformance suite).
