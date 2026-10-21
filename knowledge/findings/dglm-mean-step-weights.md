---
type: Finding
title: dglm 1.8.6's scoring steps are right only for a canonical link without offsets
description: "R dglm 1.8.6 fits the mean by least squares with working weights w V(mu) / phi and evaluates mu.eta at the linear predictor without the offset; scoring needs w mu.eta(eta + offset)^2 / (V(mu) phi). The two agree for a canonical link (mu.eta = V) without an offset, so dglm is right for the Gaussian with the identity and the Poisson with the log, and wrong for the gamma or Tweedie with the log link and for any offset: at a constant dispersion its gamma/log fit is not glm()'s. Its REML criterion also adds 0 for log det(X' W X), its default method is ML, and its stop on m2loglik alone pins the coefficients to about 1e-7. Its dispersion step takes the log link's derivative without the dispersion's offset, the same slip. prospicio's DoubleGlm is dglm's loop with these corrected."
tags: [prospicio-glm, double-glm, dglm, dispersion, reml, r-parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-21T12:00:00Z }
verified: { by: process:ci, at: 2026-10-20T17:00:00Z }
sources:
  - id: dglm
    resource: https://cran.r-project.org/package=dglm
    title: R dglm 1.8.6 (Dunn and Smyth), dglm() and dglm.constant()
  - id: script
    resource: ../validation/scripts/r_dglm.R
    title: The corrected loop, checked against dglm, glm() and nlminb
  - id: test
    resource: ../validation/tests/models.rs
    title: double_glms_match_dglm, 180 cases of validation/reference/dglm_r.csv
  - id: smyth
    resource: https://doi.org/10.1111/j.2517-6161.1989.tb01422.x
    title: Smyth (1989), Generalized linear models with varying dispersion, JRSS B 51; Smyth and Verbyla (1999), Adjusted likelihood methods for modelling dispersion in GLMs, Environmetrics 10
---

# Facts

* dglm's mean step is `lm.wfit(X, z, w V(μ) / φ)` with
  `z = η + (y − μ) / μ'(η)` and `μ'` evaluated at `η` without the offset.
  Fisher scoring needs the weights `w μ'(η + o)² / (V(μ) φ)` and the working
  response with `μ'(η + o)`. For a canonical link `μ' = V`, so the weights
  agree; with an offset `o` the log link's `μ'(η) = e^η` misses the factor
  `e^o`.
* So dglm is right for the Gaussian with the identity link and the
  Poisson with the log link, without an offset (the corrected loop matches
  it to its own precision, and `glm()` at a constant dispersion to 1e-12).
  For the gamma with the log link and `~ 1` dispersion it gives
  6.2795, 0.4705, 0.4189 on `validation/data/glmm_simulated.csv` against
  `glm()`'s 6.3098, 0.4256, 0.3768. The Tweedie with the log link and a
  Poisson with an exposure offset are wrong the same way.
* The corrected loop reaches the maximum of dglm's own `m2loglik` found
  directly by `nlminb` over both coefficient vectors (to 5e-6 in the
  parameters; nlminb never finds a lower value), so the correction is the
  ML estimate, not a different model.
* Under `method = "reml"` dglm means to add `log det(Xᵀ W X)` as
  `2 log |prod(diag(mfit$R))|`, but `lm.wfit` returns no `R`, so it adds 0
  and `m2loglik` is the plain `-2 log L` at the REML estimates. The REML
  dispersion step (each deviance's mean less `φ h`, the weights corrected
  by the leverage `h`) is applied correctly.
* `dglm.constant` switches on "Gaussian" and "Poisson", which are not R's
  family names, so every family but the gamma gets the default constant
  `Σ log(2π V(y))` with `V(0)` taken as `V(1/6)`; the gamma's is
  `2 Σ log y`. prospicio's `minus_two_log_likelihood` uses the same
  constants, so it is dglm's number.
* dglm's default `method` is `"ml"`. The user asked on #353 for dglm
  "with its default REML fit", so the request names REML but dglm's
  default is ML. By the user's decision of 2026-10-21 `DoubleGlm::new`
  uses ML, as dglm, and `.reml()` (Python and R `method = "reml"`) gives
  REML.
* The dispersion step has the same slip: `hdot <- 1/dfamily$mu.eta(deta)`
  with `deta` the linear predictor without the dispersion's offset
  (`offset()` in `dformula`), so with an offset its weights are off by
  `e^q`. On the simulated book, `count ~ x + urban` with
  `~ urban + offset(log(exposure))` (Poisson, ML) stops at an `m2loglik` of
  945.04, where the maximum is 942.56. The reference script checks a gamma
  with a dispersion offset against `nlminb` instead. dglm's start also
  adds the mean's offset to the dispersion's (`phi <- linkinv(deta +
  offset)`), which only moves the start.
* With prior weights and no offsets dglm is right (the Gaussian with the
  exposure as weights matches the corrected loop by ML and REML).
* dglm stops when the relative change in `m2loglik` falls below its
  `epsilon`, which pins the coefficients only to about the square root;
  the reference script also waits for the coefficients to stop moving.
