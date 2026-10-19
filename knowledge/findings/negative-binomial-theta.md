---
type: Finding
title: The negative binomial's theta by alternating IRLS and Newton
description: MASS glm.nb estimates theta by alternating a negative binomial GLM fit at fixed theta with Newton's method on theta's score at fixed means (theta.ml, from the moment start n / sum w (y/mu - 1)^2); the standard error is 1/sqrt of theta's observed information at the final means, treating the coefficients as known, and the AIC counts theta as a parameter. prospicio-glm's NegativeBinomialGlm does the same and matches MASS 7.3-60 on the quine data.
tags: [prospicio-glm, negative-binomial, mass, glm, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T06:00:00Z }
verified: { by: process:ci, at: 2026-10-19T06:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-glm/src/negbin.rs
    title: NegativeBinomialGlm and theta_ml
  - id: test
    resource: ../validation/tests/models.rs
    title: negative_binomial_theta_matches_mass_glm_nb
  - id: script
    resource: ../validation/scripts/r_glm_nb.R
    title: The quine design and MASS's targets
---

# Facts

* `MASS::glm.nb` alternates two steps until theta and the log-likelihood
  settle: IRLS for the coefficients at fixed theta (each fit warm-started
  from the last), and `theta.ml`, Newton's method on the score of theta
  at the fitted means, started from the method of moments
  `n / sum(w (y / mu - 1)^2)`. The first means come from the Poisson GLM.
* `SE.theta` is `1 / sqrt(I(theta))`, theta's observed information at the
  final means. It ignores the coefficients' uncertainty, so it is a
  conditional standard error, and prospicio reports the same one.
* The AIC counts theta: `-2 loglik + 2 (p + 1)`.
* On `MASS::quine`, `Days ~ Eth + Sex + Age + Lrn`: theta 1.27489264581101,
  standard error 0.161035178973597, log-likelihood -546.575509144992 (MASS
  7.3-60); prospicio agrees to 1e-6 in theta and the coefficients and to
  1e-9 in the log-likelihood.
* `validation/data/quine.csv` holds the treatment-coded design (EthN, SexM,
  AgeF1 to AgeF3, LrnSL) written by `validation/scripts/r_glm_nb.R`, so the
  Rust test needs no R.
