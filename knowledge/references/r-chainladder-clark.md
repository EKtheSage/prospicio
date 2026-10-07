---
type: Reference Implementation
title: R ChainLadder ClarkLDF and ClarkCapeCod
description: R ChainLadder 0.2.21's Clark growth-curve methods, the reference for prospicio_reserving's ClarkLdf and ClarkCapeCod; their age, scale and reporting definitions, a loose optimizer stop, a silent cap of 10 on the Cape Cod ELR, and a wrong Weibull second derivative.
resource: https://cran.r-project.org/package=ChainLadder
tags: [reserving, clark, growth-curve, maximum-likelihood, parity, r]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-05T18:00:00Z }
sources:
  - id: reserving-v02
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2 (decision 5)
  - id: generator
    resource: ../validation/scripts/reserving_clark_r.R
    title: Clark R reference generator
  - id: clark
    resource: ../crates/prospicio-reserving/src/clark.rs
    title: prospicio_reserving::clark
---

# Facts

Read from `print(ChainLadder:::ClarkLDF)`, `ClarkCapeCod`, `LL.ODP`,
`d2LL.ODPdt2`, `MU.LDF`, `MU.CapeCod`, `R.LDF`, `R.CapeCod` and the
`loglogistic` / `weibull` growth-function objects.[^generator]

* **Ages.** With `adol = TRUE` (the default) the origin width defaults to
  the mean difference of the column ages and the average date of loss to
  half of it. An age at or past the width becomes `age - width / 2`, an
  earlier one `age * (1 - 1/2)`; `maxage` becomes `maxage - width / 2`.
  Each column's from-age is the previous column's shifted age, 0 for the
  first. Column names are the ages, so a triangle in months gives `theta`
  in months.
* **Likelihood.** Over-dispersed Poisson on incremental losses,
  `sum(c log(mu) - mu)` with `mu` floored at machine epsilon,
  maximized jointly in all parameters by L-BFGS-B. ClarkLDF divides the
  triangle by the largest chain-ladder ultimate first (`magscale`) and
  scales results back.
* **Scale.** `SIGMA2 = sum((c - mu)^2 / mu) / (nobs - np)`, with
  `np = n_origins + 2` (LDF) or `3` (Cape Cod).
* **Covariance.** `FI` is the observed Hessian of the log-likelihood,
  `sum((c/mu - 1) d2mu - (c/mu^2) dmu dmu')`, and `vcov = -SIGMA2 *
  solve(FI)`. If `rcond(FI) < .Machine$double.eps` parameter risk is NA.
  A negative delta-method variance (per origin or total) is set to 0 with a
  warning.
* **Reported reserve.** ClarkLDF's `FutureValue` is the truncated LDF on
  the latest diagonal, `latest * (G(maxage.used) / G(age) - 1)`; at the
  maximum this equals the fitted `U * (G(maxage.used) - G(age))` because
  `U = latest / G(age)` on complete rows. ClarkCapeCod's is the fitted
  `ELR * P * (G(maxage.used) - G(age))`. `UltimateValue = latest +
  FutureValue`.
* **Process variance quirk.** ClarkLDF computes the process variance as
  `SIGMA2 * U * (G(maxage) - G(age))` with the *unshifted* `maxage`
  (`R.LDF(theta, G, CurrentAge.to, maxage, ...)`), while its reserve and
  parameter risk use `maxage.used`. With `maxage = Inf` the two agree.
  ClarkCapeCod uses `maxage.used` throughout. prospicio_reserving reproduces
  both for parity.[^clark][^reserving-v02]
* **Optimizer stop.** `optim(..., method = "L-BFGS-B", control =
  list(factr = .Machine$double.eps^-0.5))` stops at a relative
  log-likelihood change of 1.5e-8. On GenIns and RAA that leaves the
  parameters, reserves and standard errors up to 4e-3 (relative) short of
  the maximum; the same code with `factr = 1` agrees with prospicio_reserving's
  Nelder–Mead maximum to 2e-6.[^generator]
* **Weibull second derivative.** The Weibull `d2Gdt2` gives
  `d2G/domega2 = 2 v log(x/theta) (1 - u)`; the derivative of its own
  `dG/domega = v log(x/theta)` is `v log(x/theta)^2 (1 - u)`
  (`u = (x/theta)^omega`, `v = u exp(-u)`; checked against finite
  differences). The other entries and the whole log-logistic Hessian are
  right. The error enters only the Fisher information, so Weibull
  parameter and total standard errors are off: by up to 7.3% on the
  reference triangles. prospicio_reserving uses the correct derivative, and the
  Weibull standard-error references come from R with that one entry
  replaced.[^generator]
* **Bounds.** L-BFGS-B bounds the Weibull at `omega <= 2` and
  `theta <= 2 * max(age)`, both curves at `omega >= 0.01`, and the
  log-logistic at `theta >= min(0.5, ages)`, and warns when a curve
  parameter sits on a bound. `ClarkCapeCod` also bounds the ELR at
  `ELR <= 10`, and its warning checks only the curve parameters, so an ELR
  pinned at 10 passes silently. That happens whenever exposure is in
  smaller units than the losses (counts, premium in thousands): on RAA
  with `Premium = 1000` R stops at ELR 10, `omega = 1.7617`,
  `theta = 22.510` and a reserve of 27,419, where the maximum (R's own fit
  with `Premium = 4000`, scaled) is ELR 27.609, `omega = 1.3755`,
  `theta = 36.125` and a reserve of 115,105. prospicio_reserving's search on
  `(ln omega, ln theta)` with the ELR profiled out is unbounded and returns
  the maximum, a deliberate difference from R. No reference triangle
  reaches a bound.[^clark]

[^reserving-v02]: Design note, reserving v0.2 (decision 5)
[^generator]: Clark R reference generator
[^clark]: prospicio_reserving::clark
