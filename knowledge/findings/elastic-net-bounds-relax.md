---
type: Finding
title: glmnet's relaxed lasso blends at gamma of at least 1e-5, and a hinge basis needs an active-set solve
description: "glmnet 5.1's relaxed fit refits each lambda's nonzero columns at lambda = 0 with the same bounds (update(fit, lambda = 0, exclude = ...)) and blends gamma * penalized + (1 - gamma) * refit, but blend.relaxed raises gamma to max(gamma, 1e-5), so coef(fit, gamma = 0) is 1e-5 of the way back to the lasso; its fit$relaxed$beta is the exact refit. cv.glmnet(relax = TRUE) chooses the lowest mean deviance over every (lambda, gamma), and for one standard error the largest lambda, then the largest gamma, under the bar. Coefficient bounds clamp the coordinate update on the standardized scale (bound times the column's sd). On nearly collinear hinge columns max(0, x - k) coordinate descent finds the active set in a few sweeps and then creeps; solving the problem restricted to that set by Cholesky and checking the KKT conditions took a 20-hinge cross-validated path from 43 s to 0.8 s."
tags: [prospicio-glm, elastic-net, glmnet, relaxed-lasso, coordinate-descent, r-parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-20T20:00:00Z }
verified: { by: process:ci, at: 2026-10-20T20:00:00Z }
sources:
  - id: glmnet
    resource: https://cran.r-project.org/package=glmnet
    title: R glmnet 5.1, relax.glmnet, blend.relaxed and cv.glmnet(relax = TRUE)
  - id: script
    resource: ../validation/scripts/r_glmnet_relax.R
    title: Bounds, relaxed fits and their cross-validation in glmnet
  - id: test
    resource: ../validation/tests/models.rs
    title: bounded_and_relaxed_elastic_nets_match_glmnet and relaxed_cross_validation_matches_cv_glmnet
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/356
    title: "Issue #356: bounds, relaxed lasso and speed on hinge bases (CAS Monograph 16, sections 4.5 and 5.7.2)"
---

# Facts

* glmnet's `relax.glmnet` refits each distinct active set once with
  `update(fit, lambda = 0, exclude = <the zero columns>)`, so the refit
  keeps `lower.limits`, `upper.limits`, the offset and the weights. Its
  `fit$relaxed$beta` matches `ElasticNetFit::relax(.., 0.0)` to every
  printed digit.
* `blend.relaxed(fit, gamma)` sets `gamma = max(gamma, 1e-5)` before
  blending, so `coef(fit, s, gamma = 0)` and the cross-validation at
  `gamma = 0` are at `gamma = 1e-5`: about `1e-5 (β − β₀)` from the refit,
  a difference proportional to `λ`. prospicio's `γ = 0` is the refit, and
  the reference files record glmnet's effective `γ`.
* `cv.glmnet(relax = TRUE)`'s choice (`getOptcv.relaxed`): over every
  `(λ, γ)` pair, the lowest mean; among pairs with mean at or below it
  (or at or below the first lowest pair's mean plus its standard error),
  the largest `λ`, ties broken by the largest `γ`. On the Gaussian test
  data it picks `γ = 0` at the second-largest `λ` for one standard error.
* glmnet 5.1 takes `thresh` and `maxit` either as arguments (deprecated,
  with a warning; glmnet 4 has no `control`) or as `control = list(...)`;
  the reference values are the same either way.
* Bounds on the design's scale become `limit × sⱼ` on the standardized
  scale the coordinate update works in; the update is clamped there.
* With bounds every coefficient has a status: zero, at its lower bound,
  at its upper bound, or free with a sign. When two consecutive sweeps
  agree on every status, the weighted least squares restricted to the
  free columns (the lasso term a constant `λ pf α sign`) is a linear
  system; solved by Cholesky and kept only when every free coefficient
  keeps its sign and bounds and every other one meets its optimality
  condition, it ends the creep. A rejected pattern is not tried again.
  On #356's case (441 Poisson cells, 25 λ, 7 folds) 20 hinges went from
  43 s to 0.8 s, 10 hinges from 24 s to 0.4 s, 20 steps from 0.7 s to
  0.12 s, with glmnet's coefficients unchanged.
