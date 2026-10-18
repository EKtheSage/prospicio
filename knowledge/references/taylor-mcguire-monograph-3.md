---
type: Reference
title: Taylor and McGuire, Stochastic Loss Reserving Using GLMs (CAS Monograph 3), reproduced in prospicio
description: CAS Monograph 3's worked example (the New Jersey Manufacturers workers compensation paid triangle, 1988 to 1997) recomputed by the guide reserving-taylor-mcguire.md; the conventions behind its numbers (Pearson phi on 36 degrees of freedom, ln b1 = 0, delta-method MSEP, normal parameter draws) and the printed values that differ, with their causes.
resource: https://www.casact.org/monograph/cas-monograph-no-3
tags: [reserving, glm, odp, chain-ladder, prediction-error, bootstrap, guides, reference, parity]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-17T22:00:00-07:00 }
verified: { by: process:ci, at: 2026-10-17T22:00:00-07:00 }
sources:
  - id: monograph
    resource: https://www.casact.org/monograph/cas-monograph-no-3
    title: Greg Taylor and Grainne McGuire (2016), Stochastic Loss Reserving Using Generalized Linear Models, CAS Monograph 3
  - id: guide
    resource: ../docs/guides/reserving-taylor-mcguire.md
    title: The guide that recomputes the monograph's numbers
  - id: test
    resource: ../python/tests/test_guides.py
    title: The harness that runs the guide and checks every printed number in CI
---

# What it is

* The CAS monograph on the chain ladder as a GLM: the cross-classified and the
  "ODP Mack" GLMs that both give the chain ladder's forecasts, prediction error
  by the delta method and the parametric bootstrap, residual diagnostics and
  heat maps, and parameter reduction (quadratic accident-year effect, linear
  spline in development year, interactions) compared by AIC, BIC and GCV.
* One data set throughout: the incremental paid triangle of the New Jersey
  Manufacturers group, workers compensation, accident years 1988 to 1997,
  from the Meyers and Shi (2011) CAS database, $000 (Table 1-1). The guide
  carries it as a literal.
* Fitted in SAS GENMOD. The monograph is copyrighted; a local copy is not
  committed.

# Conventions behind the printed numbers

* Cross-classified ODP: `ln mu = ln a_k + ln b_j` with `ln b_1 = 0` and no
  intercept; phi is Pearson's chi-square over n - p = 55 - 19 = 36 (114.5).
  prospicio's `Glm("poisson", "log", dispersion="pearson")` reproduces Tables
  3-5 and 5-1 (estimates and standard errors) to every printed digit.
* Table 3-3 renormalises so the b_j sum to 1 (a_k becomes the ultimate);
  Table 3-5's renormalised a_k are one unit off Table 3-3 in four places
  (the monograph's rounding).
* ODP Mack (Table 3-4): response f_kj - 1, prior weight X_kj, log link, one
  factor per development year; exp(coefficients) are the volume-weighted
  factors less one.
* Delta method (Table 5-4): MSEP = phi * sum(mu) + g' Sigma g with
  g = sum(mu * x) over the origin's future cells; matches every RMSEP and CVP.
  The independent-years total (10,275) matches too.
* Parametric bootstrap (Tables 5-5, 7-2, 7-4, 7-6): coefficients drawn
  N(beta_hat, Sigma); the process distribution is not stated. Its means exceed
  the chain ladder by the lognormal bias sum(mu * exp(x' Sigma x / 2)):
  374,994 against the printed 374,992: the monograph attributes the gap to
  sampling error, but the bias accounts for it.
  `GlmFit.predict_distribution(..., "normal")` reproduces the tables within
  simulation noise (seed 1: total 374,883 and 14,432 against 374,992 and 14,286).
* Table 7-7's AIC and BIC use `sum(y ln mu - mu) / phi` with phi held at the
  interaction model's 53.93 for every model; GCV is n * RSS / (n - p)^2 with
  raw residuals. GCV matches to five figures; AIC and BIC are 0.7 to 1.4 above
  the printed values (a relative 2e-6 of the log-likelihood: phi or convergence
  in SAS), with the differences between models equal within rounding.
* Model 7-10's c is 0.1544995, printed 0.155 (a seventh-decimal difference
  in SAS's estimate flips the rounding).

# Gaps in prospicio found here

* Reserving (#333): `OdpGlm` is not in the bindings; no analytic MSEP; no
  custom design (reduced models, calendar terms, interactions); no residual
  diagnostics; ODP only.
* Models (#334): `GlmFit` lacks residuals, leverages, prediction standard
  errors for sums of rows, and GCV.
* Mack on this triangle gives a total standard error of 10,939 against the
  ODP's 14,076: same forecast, different variance assumptions.
