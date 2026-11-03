---
type: Finding
title: pscl's zeroinfl leaves the prior weights out of the negative binomial's theta score, and its BFGS stops short of the maximum
description: "pscl 1.5.9's zeroinfl(dist = 'negbin') multiplies the count and zero parts' scores by the prior weights but not theta's (gradNegBin's wres_theta), so with weights its BFGS solves an unweighted theta equation and its optim Hessian, the finite difference of that gradient, gives standard errors up to 28% off the weighted likelihood's on the seeded book. Without weights, its BFGS at reltol 1e-15 from the EM start stops short of the maximum (coefficients off by up to 2e-6 absolute, 8.5e-5 relative, on bioChemists and the seeded book), and its standard errors are within 4e-4 of the maximum's. prospicio's ZeroInflated and Hurdle maximise the weighted likelihood by Newton on analytic gradients; they agree with pscl's likelihood maximised to the end to 1e-9 and with its Hessian's inverse to 1e-7."
tags: [prospicio-glm, zero-inflated, hurdle, pscl, r-parity, weights]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-03T12:00:00Z }
verified: { by: process:ci, at: 2026-11-03T12:00:00Z }
sources:
  - id: pscl
    resource: https://cran.r-project.org/package=pscl
    title: R pscl 1.5.9 (Jackman, Zeileis), zeroinfl() and hurdle()
  - id: script
    resource: ../validation/scripts/r_zero_counts.R
    title: The references, pscl's likelihood maximised by Newton from its estimates
  - id: test
    resource: ../validation/tests/models.rs
    title: zero_inflated_and_hurdle_counts_match_pscl, 545 cases of validation/reference/zero_counts_pscl.csv
---

# Facts

* In `zeroinfl`'s `gradNegBin` the last column is `wres_theta`, the
  rows' scores for `log(theta)`, summed without `weights`; the count and
  zero columns are `wres_count * weights * X` and `wres_zero * weights *
  Z`. The log-likelihood `ziNegBin` weights every row. So with prior
  weights the gradient BFGS follows is not the likelihood's: its estimates
  stop where the unweighted theta score is zero (theta 1.0670562 against
  the weighted maximum's 1.0670535 on the seeded book), and the Hessian
  from `optim(hessian = TRUE)`, a finite difference of that gradient, is
  not the likelihood's. Its standard errors are then up to 28% above the
  weighted likelihood's (count intercept 0.0886 against 0.0728, zero
  intercept 0.450 against 0.353, the coefficient of `x` 6% and 12%).
* Without weights the gradient is right, but BFGS at `reltol = 1e-15`
  from the EM start stops short of the maximum, the coefficients up to
  2e-6 absolute (8.5e-5 relative) from it: on `bioChemists` the
  zero-inflated Poisson's coefficient of `phd` in the zero part is
  0.0012721561 against the maximum's 0.0012722641. The standard errors
  from `vcov` (optim's step-1e-3 Hessian) are within 4e-4 relative of the
  maximum's, so the test checks them at 2e-3.
* A Newton refinement of pscl's likelihood needs an analytic gradient:
  with numDeriv's numerical gradient (Richardson) the intercepts stop
  about 6e-9 from the maximum. The reference script writes the score out
  and checks it against numDeriv's at pscl's estimates.
* `zeroinfl(y ~ x + offset(log(e)))` without `|` applies the offset to
  the zero part too; prospicio's one-design fit and R `zeroinfl_fit()`
  apply it to the count part only (`docs/design/models.md`, "Zero-inflated
  and hurdle counts", item (b)).
* `pscl::vuong` counts `length(coef(m))`, the coefficients without the
  negative binomial's theta, in its AIC and BIC corrections.
