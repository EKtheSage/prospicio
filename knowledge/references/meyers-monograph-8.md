---
type: Reference Implementation
title: Meyers' CRC and CSR code (CAS Monograph 8) and its Stan transcription
description: Meyers' R scripts CRC.R and CSR.R hold his Stan programs as strings; they carry no licence and no longer compile on CmdStan 2.39, so validation/scripts/stan/meyers_{crc,csr}.stan transcribe them. The code differs from the monograph's description in ways that matter for parity (prior sd 3.162, a floor of 1e-5 on each variance step, reversed sigma labels, pmax and ceiling, elpd on the log scale).
resource: ../validation/scripts/stan/meyers_crc.stan
tags: [bayes, reserving, meyers, crc, csr, stan, validation, reference]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-12T17:30:00-07:00 }
sources:
  - id: zip
    resource: https://www.casact.org/sites/default/files/2021-04/meyers-appendix.zip
    title: meyers-appendix.zip (sha256 480e99a63fd4041ee546058412cb398f40071aa27e52e51cf671d5aaafec0f9a), Appendix/CRC.R and CSR.R
  - id: monograph
    resource: https://www.casact.org/sites/default/files/2021-02/08-Meyers.pdf
    title: Meyers (2019), Stochastic Loss Reserving Using Bayesian MCMC Models, 2nd ed., CAS Monograph 8
  - id: errata
    resource: https://www.casact.org/sites/default/files/2026-06/Monographs-8-Errata.pdf
    title: CAS Monograph 8 errata
  - id: stan
    resource: ../validation/scripts/stan/meyers_csr.stan
    title: The CSR transcription (the CRC one is the resource above)
  - id: script
    resource: ../validation/scripts/reserving_meyers_stan.py
    title: The reference generator
  - id: csv
    resource: ../validation/reference/reserving_meyers_stan.csv
    title: The committed reference summaries and Meyers' printed numbers
---

# What it is

* The CAS publishes the monograph's code as `meyers-appendix.zip`. The
  Stan program of each model is a string in its R script:
  `CRCmodel_stan` in `Appendix/CRC.R` (lines 149-191) and `CSRmodel_stan`
  in `Appendix/CSR.R` (lines 157-204). Neither the zip, its Read Me nor the
  monograph's page states a licence, so the repository holds a
  transcription with attribution, not a copy.[^zip]
* The transcriptions change syntax and generality only: `real x[10]`
  becomes `array[10] real x` (the old array syntax was removed in Stan 2.33,
  so the verbatim code does not compile on CmdStan 2.39), `gamma_cdf(y, 1,
  1)` becomes `gamma_cdf(y | 1, 1)`, the size 10 becomes data `n`, and the
  generated quantities add `sigma` and the predictive ultimates and
  reserves that Meyers draws in R.[^stan]
* The generator runs both models on paid and incurred of group 353 with
  seed 20261012, 4 chains of 2,000 warm-up and 25,000 draws, `metric =
  "dense_e"`, `adapt_delta` 0.99, `max_treedepth` 12, and Meyers' own
  initial values (`init.CRC`, `init.CSR`) drawn per chain; it fails on any
  R-hat above 1.005 or any divergence (there were none; the largest R-hat
  was 1.0004). About 1 minute per case, half of it arviz's PSIS-LOO on
  100,000 x 55 log likelihoods.[^script]

# Quirks of the code (what parity has to copy)

* Priors are `normal(0, 3.162)` on `alpha` and `beta` and `normal(-.4,
  3.162)` on `logelr`, a standard deviation of `sqrt(10)` rounded (the
  monograph's text says "10"; the errata confirm `sqrt(10)`), and
  `normal(0, 0.05)` on `gamma`.[^zip][^errata]
* The variance steps: `a_ig[i] ~ inv_gamma(1, 1)` declared
  `<lower=0, upper=100000>`, and `s = gamma_cdf(1 / a_ig | 1, 1) =
  1 - exp(-1 / a_ig)`. Without the bound `s` would be exactly `U(0, 1)`;
  the bound makes it `U(1 - exp(-1e-5), 1)`, a floor of about `1e-5` on
  every step, so `sigma_d >= sqrt((11 - d) 1e-5)`. The floor matters: a
  model with `U(0, 1)` steps put the late `sigma_d` 10 to 20% lower on
  incurred data, its elpd 3.5 higher, and diverged.[^zip]
* `sig2[10] = s(a_ig[10])` and `sig2[10 - i] = sig2[11 - i] + s(a_ig[i])`:
  the step labels run in reverse of the monograph's `sigma_d^2 = sum over
  i >= d of a_i`. The steps are iid, so the posterior of `sig` is the same,
  but parity must compare `sigma_d`, never the steps.[^zip]
* CSR's `speedup[w] = speedup[w - 1] * (1 - gamma)` is `(1 - gamma)^(w -
  1)`.[^zip]
* Data handling: `pmax(loss, 1)` raises any value below 1 to 1 before the
  log (prospicio errors instead); training cells are `cal < 11`.[^zip]
* The prediction draws only age 10: `ceiling(rlnorm(n, log(Premium[w]) +
  alpha[w] + logelr, sig[10]))` for `w >= 2`, with the oldest origin's
  observed age-10 value as is. "Estimate" is the mean total ultimate, not a
  reserve. The `ceiling()` adds about +0.5 per origin, +4.5 on the total;
  neither the transcription nor prospicio applies it.[^zip]
* `elpd_loo` is R `loo` on `normal_lpdf(log C | mu, sig[d])`, the log
  scale, so it is comparable only between CRC and CSR on the same
  column.[^zip]
* His runs: `iter = 5000` (2,500 warm-up), 4 chains, `adapt_delta =
  .9999`, `max_treedepth = 50`, seed 12345, thinning doubled until R-hat
  is below 1.05.[^zip]

# Printed numbers on group 353 (checked against the PDF)

* CRC paid total ultimate 40,121, SE 2,487 (Table 5.2, p. 18); CRC
  incurred 39,147, SE 1,642 (Table 5.3, p. 18); CSR paid 37,597, SE 2,401
  (Table 7.2, p. 33).[^monograph]
* CRC paid `logelr` -0.3965, sd 0.0233 (Table 5.1, p. 17); CSR paid
  `logelr` -0.3956, sd 0.0246, `gamma` 0.0446, sd 0.0282 (Table 7.1,
  p. 32).[^monograph]
* elpd_loo: CRC paid 47.80, CSR paid 49.76 (Tables 6.4, p. 29, and 7.3,
  p. 34); CRC incurred 70.97 (Table 6.4, whose columns are misaligned in
  the PDF; its LOOIC -141.93 is -2 x 70.97).[^monograph]
* The CmdStan 2.39 run of the transcription agrees: totals 40,136 (sd
  2,497), 39,132 (1,616) and 37,537 (2,391); elpd_loo 48.04, 71.22 and
  49.56; `gamma` 0.0450 (sd 0.0280).[^csv]
