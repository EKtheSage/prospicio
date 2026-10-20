---
type: Reference
title: glmmTMB and lme4 as references for random-intercept GLMMs
description: glmmTMB 1.1.15.2 and lme4 2.0.6 (R 4.3.3) as the parity references for prospicio_glm::glmm. glmmTMB's nlminb stops with gradients near 1e-3, about 3e-5 short on the coefficients, so the reference polishes each fit by Newton steps on TMB's objective; its gamma dispersion parameter is the log shape (-log phi) and sigma() the coefficient of variation; it multiplies the log-likelihood by prior weights; its condVar adds the estimates' uncertainty, lme4's does not; glmer's log-likelihood with nAGQ > 1 leaves out the saturated model's.
tags: [prospicio-glm, glmm, glmmtmb, lme4, r, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-20T12:00:00Z }
verified: { by: process:ci, at: 2026-10-20T12:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-glm/src/glmm.rs
    title: Glmm, GlmmFit and the Laplace and quadrature fits
  - id: test
    resource: ../validation/tests/models.rs
    title: glmm_matches_glmmtmb_and_glmer
  - id: script
    resource: ../validation/scripts/r_glmm.R
    title: The data and the reference values
---

# glmmTMB and lme4

`validation/scripts/r_glmm.R` fits random-intercept GLMMs with glmmTMB
1.1.15.2 (TMB 1.9.25) and lme4 2.0.6 on R 4.3.3, and
`glmm_matches_glmmtmb_and_glmer` checks prospicio's fits against them
(1,237 cases: coefficients to 2e-5, standard errors to 2e-4, each level's
mode to 1e-4, log-likelihoods to 1e-8). What a reader of either package's
output needs to know:

- **glmmTMB stops short.** With its default `nlminb` settings (and even
  with `rel.tol = 1e-14`, which ends in "singular convergence (7)") the
  Salamanders fit stops with a gradient of 1.3e-3: coefficients 3e-5 and
  a nested level's mode 1.4e-4 off the maximum, while the log-likelihood
  agrees to 1e-8 because the surface is flat. The reference polishes each
  fit by Newton steps on `fit$obj$fn` and `fit$obj$gr` (`optimHess`) to a
  gradient below 1e-9, then refits from the polished values with an
  optimizer that returns its start, so that `vcov`, the `sdreport` and
  `ranef` are all evaluated there. `glmer` with `bobyqa` at
  `rhoend = 1e-12` and prospicio agree without it.
- **The gamma's dispersion.** `sigma()` of a gamma fit is the coefficient
  of variation `1/sqrt(shape)`, so `phi = sigma^2`; the dispersion
  parameter `betadisp` is the log shape, `-log phi`, so its standard error
  is `log phi`'s. For the Tweedie `sigma()` is `phi` and `betadisp` is
  `log phi`. The Tweedie power is the parameter `psi = qlogis(p - 1)`; the
  reference fixes it with `start = list(psi = qlogis(0.5))` and
  `map = list(psi = factor(NA))`.
- **Weights.** glmmTMB multiplies each row's log-likelihood by its prior
  weight rather than dividing the dispersion by it (`Family::log_likelihood`
  and R's `glm` divide), so the reference fits use no weights.
- **Conditional variances.** lme4's `ranef(condVar = TRUE)` is the
  diagonal of `H^-1` at the estimates (prospicio's
  `conditional_variance`). glmmTMB's comes from `sdreport` and adds the
  estimates' uncertainty, `J Sigma J'` with `J = du/d(beta, theta)`
  (prospicio's `prediction_variance`); on Salamanders it is 1.12 to 4.07
  times lme4's, on grouseticks 1.005 to 2.4 times.
- **glmer's quadrature log-likelihood.** With `nAGQ > 1` `glmer` computes
  the deviance from deviance residuals and leaves out the saturated
  model's log-likelihood, `sum(dpois(y, y, log = TRUE))`: on Salamanders
  it reports -610.749 with nine nodes. Adding the saturated term back
  gives -972.385, prospicio's value, on the same scale as the Laplace
  fit's log-likelihood.
- **Installing on R 4.3.** glmmTMB needs `pbkrtest`, which needs `doBy`,
  which needs `Deriv`; CRAN's Deriv 4.3.5 needs R 4.5, so the archived
  Deriv 4.1.6 is installed first. lme4 2.0.6's `print(VarCorr(fit))` fails
  on R 4.3 (`%||%` is base R from 4.4); `attr(VarCorr(fit)[[g]],
  "stddev")` works.
