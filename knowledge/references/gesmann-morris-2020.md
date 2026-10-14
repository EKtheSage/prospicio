---
type: Reference
title: Gesmann and Morris (2020), hierarchical compartmental reserving models
description: The CAS Research Paper behind prospicio's Compartmental; its brms code (not a Stan program) reconstructed as the Stan program brms generates reproduces its printed model 1 posterior to rounding on company 337; its text and code disagree on the origin hierarchy (both models) and on model 2's sigma prior, and the code wins for parity.
tags: [reserving, compartmental, bayes, brms, stan, reference]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-13T21:00:00-07:00 }
sources:
  - id: paper
    resource: Casualty Actuarial Society, CAS Research Papers (2020)
    title: Gesmann and Morris (2020), "Hierarchical Compartmental Reserving Models", CAS Research Paper
  - id: design
    resource: ../docs/design/reserving-compartmental.md
    title: Design note, compartmental reserving (References, decisions 2 and 7)
  - id: stan
    resource: ../validation/scripts/stan/compartmental_gm1.stan
    title: The reconstruction of model 1
  - id: script
    resource: ../validation/scripts/reserving_compartmental_stan.py
    title: The reference generator, with the printed m1fit
---

# Facts

* The paper prints R code for brms 2.10.0 on rstan 2.19.2 (sec. 7.2.2 for
  model 1, 7.2.3 for model 2) with the one-OS closed form as Stan function
  snippets, not Stan programs, and states no licence. The reference
  reconstructs the program brms generates from that code, without brms
  installed, and copies nothing.[^paper][^stan]
* brms conventions the reconstruction must reproduce: `nl = TRUE` gives
  each nonlinear parameter (`oRLR`, `oRRF`, `oker`, `okp`) its own
  intercept with a `normal(0, 1)` prior and no centring; the group sds are
  half Student-t (class `sd`, bounded at 0) on the `o` scale; the effects
  are non-centred, `r = (diag(sd) L z)'` with `z ~ std_normal()` and
  `L ~ lkj_corr_cholesky(1)`, correlated only within one `|ID|` term (model
  1: `oRLR` and `oRRF` by accident year); and `sigma ~ 0 + deltaf` with the
  log link puts the `student_t(1, 0, 1000)` priors (class `b`) on
  `log sigma`, not a half-Cauchy on sigma, with no Jacobian.[^design][^stan]
* Model 1's natural-scale map: `RLR = 0.7 e^(0.2 oRLR)`, `RRF = 0.8 e^(0.1
  oRRF)`, `k_er = 3 e^(0.1 oker)`, `k_p = e^(0.1 okp)`; the sd priors are
  `student_t(10, 0, 0.2)` on `sd(oRLR)` and `student_t(10, 0, 0.1)` on
  `sd(oRRF)`.[^paper]
* Text against code, the origin hierarchy (both models): the text (sec.
  5.2.1) writes `RLR[i] ~ mu_RLR + u_RLR` with the sd prior on an additive
  `u`; the code puts the effect on the standardised `o` scale,
  `RLR_i = 0.7 e^(0.2 (b + u_i))`, with the sd prior on `o`. The printed
  `sd(oRLR)` 0.64 and `sd(oRRF)` 0.74 are on the `o` scale. The code wins
  for parity.[^paper][^design]
* Text against code, model 2's sigma prior: the text gives a median of
  0.1, the code `log sigma ~ normal(log 0.2, 0.2)`. The code wins for
  parity (PR C).[^paper][^design]
* The paper's `m1fit` ran `brm(..., control = list(adapt_delta = 0.99,
  max_treedepth = 15), seed = 123, iter = 2000, chains = 4)`: 4 chains of
  1,000 kept draws, with bulk ESS 1,300 to 2,200 printed for the
  group-level rows, so its means carry a Monte Carlo error of about 0.03
  posterior sd besides the rounding to two decimals.[^paper]
* The printed model 1 posterior (`m1fit`, sec. 7.2.2, mean and sd):
  `oRLR` 1.44 (0.26), `oRRF` -1.36 (0.42), `oker` -5.40 (0.81), `okp` -8.49
  (0.36), `sd(oRLR)` 0.64 (0.15), `sd(oRRF)` 0.74 (0.22), their correlation
  0.54 (0.26), `log sigma_OS` 8.25 (0.16), `log sigma_PD` 6.66 (0.20). The
  reconstruction under CmdStan 2.39.0 (4 x 25,000 draws) gives 1.442
  (0.259), -1.358 (0.416), -5.392 (0.830), -8.488 (0.358), 0.642 (0.152),
  0.737 (0.224), 0.534 (0.260), 8.252 (0.161) and 6.671 (0.201): every mean
  within 0.04 printed sd and every sd within 2.5%, so no brms convention
  was missed.[^script]
* The posterior sits far from the preset: `oker` -5.4 and `okp` -8.5 are
  5.4 and 8.5 prior sds from 0 (`k_er` about 1.75 and `k_p` about 0.43
  against medians 3 and 1). A prior-data conflict, not a prior-identified
  fit.[^design]
* brms' `predict()` for a level with no data uses `sample_new_levels`,
  whose default reuses a randomly chosen existing level's effect; the
  reference instead draws a new effect from the population, as
  `Compartmental` does (decision 2), so the two match.[^stan][^design]
