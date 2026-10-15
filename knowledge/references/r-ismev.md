---
type: Reference
title: R ismev's threshold diagnostics
description: ismev 1.43's gpd.fit, gpd.fitrange and mrl.plot, the reference for prospicio_prob::evt::threshold_diagnostics; Nelder-Mead and a numerical Hessian put its estimates within about 1e-3 of the exact maximum, its parameter order is (scale, shape), and mrl.plot's bands use the sample standard deviation.
resource: https://cran.r-project.org/package=ismev
tags: [probability, evt, gpd, threshold, parity, r]
status: stable
verified: { by: process:ci, at: 2026-10-15T12:00:00Z }
generated: { by: claude-code/cloud-session, at: 2026-10-15T12:00:00Z }
sources:
  - id: ismev
    resource: https://cran.r-project.org/package=ismev
    title: ismev 1.43 (GPL), Heffernan and Stephenson, the R code for Coles (2001); gpd.fit, gpd.fitrange and mrl.plot
  - id: script
    resource: ../validation/scripts/r_evt_thresholds.R
    title: r_evt_thresholds.R, the reference values for prospicio's threshold diagnostics
---

# What the functions compute

* `gpd.fit(x, u)` fits the GPD to `x[x > u] - u` by `optim`'s
  Nelder-Mead from a moment start (scale `sqrt(6 var) / pi`, shape 0.1),
  and returns `mle = c(scale, shape)`, `cov = solve(hessian)` from
  `optim`'s numerical Hessian and `se`. The order is (scale, shape), the
  reverse of `prospicio_prob::evt::Gpd`'s `(xi, beta)`.[^ismev]
* `gpd.fitrange(x, umin, umax, nint)` refits at `nint` evenly spaced
  thresholds and plots the modified scale `scale - shape * u` and the
  shape with `1.96` standard errors; the modified scale's comes by the
  delta method with `d = c(1, -u)`.[^ismev]
* `mrl.plot(x)` plots the mean excess with bands
  `qnorm((1 + conf) / 2) * sd / sqrt(n)`, `sd` the sample standard
  deviation (divisor `n - 1`) of the exceedances. It filters `data` in
  place threshold by threshold, which is harmless only because its
  thresholds increase.[^ismev]

# How close it is

* Nelder-Mead stops early: on 1,000-point Burr and Weibull samples the
  shape differs from the exact maximum (the root of the profile score)
  by up to about `1e-3` relative and the standard errors by up to
  `5e-3`, so prospicio's parity test compares with `gpd.fit` at `2e-3`
  and `2e-2`, and with an exact refit in the same script (`uniroot` on the
  score, the observed information from R's symbolic `deriv()`) at `1e-9`
  and `1e-8`.[^script]
* The mean excess and its standard error are plain arithmetic and match
  at `1e-12`.

[^ismev]: `R/gpd.R` and `R/mrl.R` in the ismev 1.43 sources.
[^script]: `validation/scripts/r_evt_thresholds.R` and `validation/tests/distributions.rs`.
