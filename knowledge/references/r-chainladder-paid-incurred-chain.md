---
type: Reference Implementation
title: R ChainLadder PaidIncurredChain (paid-incurred chain reference)
description: R ChainLadder 0.2.21's PaidIncurredChain, the reference for Merz and Wuthrich's (2010) paid-incurred chain; it zeroes the posterior precision between Phi_n (n >= 1) and Psi_0, leaves the oldest origin out of its ultimate, and returns only the total's standard error.
resource: https://cran.r-project.org/package=ChainLadder
tags: [reserving, paid-incurred, merz-wuthrich, bayesian, parity, r]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-03T06:00:00+00:00 }
verified: { by: process:ci, at: 2026-11-03T06:00:00+00:00 }
sources:
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 32, Merz and Wuthrich's paid-incurred chain
  - id: script
    resource: ../validation/scripts/reserving_paid_incurred_r.R
    title: Generator of validation/reference/reserving_paid_incurred_r.csv
  - id: tests
    resource: ../validation/tests/reserving_paid_incurred.rs
    title: matches_r_chainladder, the_models_posterior_is_close_to_rs
  - id: unit
    resource: ../crates/prospicio-reserving/src/paid_incurred.rs
    title: the_posterior_is_the_models_and_r_leaves_out_a_coupling
  - id: paper
    resource: https://doi.org/10.1016/j.insmatheco.2010.02.004
    title: Merz and Wuthrich (2010), Paid-incurred chain claims reserving method, Insurance Mathematics and Economics 46(3), 568-579
---

# Use

`PaidIncurredChain(USAApaid, USAAincurred)` returns `Ult.Loss.Origin`,
`Ult.Loss`, `Res.Origin`, `Res.Tot` and `s.e.` (the total's standard
error only). The generator evaluates the function body statement by
statement in an environment whose parent is `asNamespace("ChainLadder")`
to read the intermediate `sigma2.hat`, `tau2.hat`, `beta`, the posterior
means and the posterior covariance.[^script]

# Facts

* Its posterior precision has zero entries between `Phi_n` for `n >= 1`
  and `Psi_0`. The loops that fill the block test `n == 0 | m == 0`, where
  the model gives zero only for `n == 0`. Zeroing those entries in an
  otherwise exact assembly of the model's posterior reproduces every
  value R returns to 1e-9 on USAA and on Dahms' data; a direct Gaussian
  conditioning of the model equals the posterior with the entries
  kept.[^tests][^unit] Whether the paper's printed formula has the same
  omission was not checked (the paper was not available).
* The omission moves the total reserve by 0.50% and its standard error by
  0.49% on USAA (R 1,596,953.58 and 110,976.85; the model 1,604,986.13 and
  111,518.89), and by 1.14% and 1.68% on Dahms' data (R 11,074,832.95 and
  392,409.36).[^tests]
* `Ult.Loss` sums the ultimates of origins 1 to J only: the oldest
  origin, already settled, is left out. Its `Res.Tot` is unaffected.
* The variances are sample variances (divisor n - 1); `sigma_0^2` is the
  variance of `log P[i,0]`. The last `sigma^2` and the last `tau^2`, which
  have one observation, come from `lm(log(s2) ~ dev)` over every estimated
  value and are predicted at the missing age.
* The priors are non-informative; there is no argument for them or for
  given variances.

# Consequences

* `prospicio_reserving::PaidIncurredChain` defaults to the model's
  posterior (`PicConvention::Model`) and reproduces R with
  `PicConvention::RChainLadder`; the parity runs under the latter.[^design]
* The parity compares R's `Ult.Loss` with prospicio's total ultimate less
  origin 0, and checks R's `beta_j` against the incurred weight of the
  origin whose latest age is `j`.

[^design]: Design note, decision 32
[^script]: Generator of the reference
[^tests]: Validation tests
[^unit]: Unit test of the posterior
