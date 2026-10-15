---
type: Finding
title: Compartmental model 2 against the paper's code in Stan and its printed posterior
description: Compartmental::model_2 with the paper's preset matches Gesmann and Morris' model 2 reconstructed in Stan on company 337 at every seed of a 200-seed sweep and the paper's printed m2fit; k_p mixes slowly, so its parameter sds are heavy-tailed from seed to seed (dropped) and model 2 samples 4 chains of 2,000 warm-up and 8,000 kept draws at target acceptance 0.999, where model 1's sampler warned at most seeds.
tags: [bayes, reserving, compartmental, stan, brms, clrd, validation]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-14T06:00:00-07:00 }
sources:
  - id: test
    resource: ../validation/tests/reserving_compartmental_gm2.rs
    title: The parity tests and the seed sweep
  - id: model
    resource: ../crates/prospicio-reserving/src/bayes/compartmental.rs
    title: prospicio_reserving::bayes::Compartmental
  - id: csv
    resource: ../validation/reference/reserving_compartmental_gm2_stan.csv
    title: Stan summaries and the paper's printed numbers
  - id: script
    resource: ../validation/scripts/reserving_compartmental_gm2_stan.py
    title: The reference generator and its tolerances
  - id: design
    resource: ../docs/design/reserving-compartmental.md
    title: Design note, compartmental reserving (decisions 2, 5 and 7)
---

# Finding

* Company 337, the paper's training triangle (90 observations of
  outstanding and incremental paid loss ratios, 9 accident years and 9
  development years, 1997 with premium only); ours with
  `Compartmental::model_2(CompartmentalPriors::gesmann_morris_2())` at seed
  0, Stan 4 x 25,000 draws of the reconstruction at `adapt_delta` 0.999,
  the paper's printed `m2fit` (mean and sd):[^test][^csv]

  | | Ours | Stan | Printed |
  |---|---|---|---|
  | `b_ker` | -1.294 (1.064) | -1.271 (1.056) | -1.31 (1.08) |
  | `b_kp` | -5.112 (1.692) | -5.064 (1.722) | -5.07 (1.75) |
  | `b_RLR` | 1.539 (0.447) | 1.534 (0.455) | 1.54 (0.44) |
  | `b_RRF` | -1.456 (0.697) | -1.446 (0.696) | -1.45 (0.69) |
  | `tau_RLR` | 0.699 (0.306) | 0.695 (0.306) | 0.69 (0.31) |
  | `tau_RRF` | 0.731 (0.470) | 0.743 (0.472) | 0.73 (0.47) |
  | `tau_ker` | 0.263 (0.214) | 0.260 (0.213) | 0.26 (0.22) |
  | `tau_kp` | 2.175 (1.315) | 2.211 (1.337) | 2.23 (1.35) |
  | `rho` | 0.358 (0.474) | 0.357 (0.475) | 0.37 (0.47) |
  | `tau_dev_RLR` | 0.352 (0.264) | 0.361 (0.267) | 0.35 (0.26) |
  | `tau_dev_RRF` | 1.124 (0.498) | 1.119 (0.497) | 1.11 (0.49) |
  | `tau_dev_ker` | 0.274 (0.228) | 0.272 (0.228) | 0.27 (0.23) |
  | `tau_dev_kp` | 0.483 (0.511) | 0.481 (0.518) | 0.45 (0.43) |
  | `log_sigma_os` | -1.787 (0.165) | -1.787 (0.164) | -1.78 (0.16) |
  | `log_sigma_pd` | -1.887 (0.162) | -1.887 (0.162) | -1.89 (0.16) |
  | total reserve | 152,387 (13,909) | 152,242 (13,938) | |
  | 1997 reserve | 30,322 (6,876) | 30,296 (6,801) | |
  | elpd_loo | 203.75 | 204.29 | |

  `k_er` is about 2.6 and `k_p` about 0.60 at the population (prior
  medians 3 and 1): model 2 sits much nearer the preset than model 1
  (1.76 and 0.43), because the rates now vary by cell. The total reserve
  (paid to age 10) is 152,000 against model 1's 173,000.
* Model 2's sampler: 4 chains of 2,000 warm-up and 8,000 kept draws at
  target acceptance 0.999. `k_p` mixes slowly (its accident-year sd has a
  wide posterior, 2.2 (1.3), and `b_kp` is weakly identified against the
  mean of its effects; the paper's own fit has a bulk ESS of 499 on `okp`
  in 4,000 draws). Model 1's sampler (1,000 and 2,000 draws at 0.99) warned
  at 7 of 8 seeds, with divergent transitions at 4; 0.95 left up to 142
  divergences; 2,000 and 4,000 draws at 0.999 warned at 41 of 200 seeds
  (R-hat up to 1.018 on `k_p`'s parameters or a tail ESS below 400 on
  `k_p[o]`); 8,000 at 5 of 200 (4 with one divergent transition, 1 with a
  tail ESS of 398). The CmdStan reference at the paper's `adapt_delta` 0.99
  also left 5 divergent transitions in 100,000 draws.[^test][^design]
* A default model 2 fit takes about 61 s in release on the machine of
  `knowledge/environment/local-cmdstan.md` (4 chains in parallel), about 11
  times model 1's 5.5 s, with a gradient of 10 µs (decision 4 of the design
  note; the chunked dual pass cost 67 to 74 µs).[^design][^model]
* Monte Carlo error of the defaults, the seed-to-seed sd over seeds 0 to
  199 with `s` Stan's posterior sd: at most `0.036 s` on a parameter mean
  (`tau_dev_kp`; `0.017 s` off `k_p`), 1.4% on a parameter sd, `0.011 s` on
  an ultimate loss ratio, `0.017 s` on a reserve mean, 1.6% on a reserve
  sd, `0.037 s` on a reserve percentile, `0.014 s` on a test-diagonal mean
  and 0.45 on elpd_loo. The sds of `tau_dev_kp`, `dev_kp[a]` and `k_p[o]`
  are heavy-tailed from seed to seed (4% to 14% over seeds 0 to 39, 18%
  with 4,000 kept draws), so their 19 rows are dropped and their means
  kept. Every tolerance is at least 4.4 times its row's spread, and seeds 0
  to 199 pass all 282 rows (at most 0.93 of a tolerance).[^test][^script]
* No draw has population `k_er <= k_p`, and no draw and observed cell has
  `k_er <= k_p` (`cell_rate_swaps` 0), at any of the 200 seeds.[^test]
* elpd_loo is PSIS-LOO with Pareto k above 0.7 on a few cells (up to 1.04
  in arviz, 1.3 in ours); with 4,000 kept draws ours ran 0.39 above arviz's
  on average, with 8,000 0.01, so the row checks the computation rather
  than the fit's predictive accuracy.[^test][^csv]
* Exact replay: seeds 0 to 39 rerun in other processes reproduced every
  diagnostic and value bit for bit.[^test]
