---
type: Reference
title: Bahnemann, Distributions for Actuaries (CAS Monograph 2), reproduced in prospicio
description: CAS Monograph 2's worked examples and problem answers recomputed by the two Bahnemann guides; its parameterisations against prospicio's (the Pareto is a Lomax, the gamma has a scale, the negative binomial is in mean and contagion), and the printed numbers that do not reproduce, with their causes (rounding carried into differences, a misprinted mu in Ex 6.6, spreadsheet solver results, and a shifted gamma that overstates aggregate-limit credits).
resource: https://www.casact.org/monograph/cas-monograph-no-2
tags: [severity, frequency, aggregate, ilf, deductibles, layers, guides, reference, parity]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-17T12:00:00-07:00 }
verified: { by: process:ci, at: 2026-10-17T12:00:00-07:00 }
sources:
  - id: monograph
    resource: https://www.casact.org/monograph/cas-monograph-no-2
    title: David R. Bahnemann (2015), Distributions for Actuaries, CAS Monograph 2
  - id: guide1
    resource: ../docs/guides/distributions-for-actuaries-claim-size.md
    title: The guide to chapters 2, 5 and 6 (claim size, excess claims, limits and deductibles)
  - id: guide2
    resource: ../docs/guides/distributions-for-actuaries-counts-aggregates.md
    title: The guide to chapters 3 and 4 and the aggregate limits of section 6.4
  - id: test
    resource: ../python/tests/test_guides.py
    title: The harness that runs both guides and checks every printed number in CI
---

# What it is

* The CAS's textbook on loss distributions for pricing (2015): claim size
  (limited moments, gamma, lognormal, Pareto, fitting grouped, censored and
  truncated data, transformations and inflation), claim counts (Poisson,
  mixed Poisson, negative binomial, contagion), aggregate claims (moment
  approximations, Panjer, Heckman-Meyers, simulation), excess claims and the
  leveraging effect, and limits and deductibles (ILFs with ALAE, risk loads,
  aggregate limits, straight, franchise and diminishing deductibles).
* The monograph is copyrighted; a local copy is not committed. The guides
  quote short formulas and the printed numbers as literals, not text or
  tables.
* All data are grouped tables; the only individual losses are the twelve of
  Problem 6.13. Fits are on a spreadsheet solver (a footnote warns results vary).

# Parameterisations against prospicio

* Gamma (alpha, beta) has beta a scale: `Gamma(alpha, beta)`; exponential beta
  is `Gamma(1, beta)`.
* The Pareto of every example is the shifted one (a Lomax),
  F = 1 - (beta/(x+beta))^alpha from 0: `GeneralizedPareto(1/alpha, beta/alpha)`
  or `Burr(alpha, 1, beta)`. prospicio's `Pareto(t, alpha)` is the monograph's
  classical Pareto (from gamma = t), used only in Problem 2.1.
* Weibull (beta, delta) is `Weibull(shape=delta, scale=beta)`; Burr
  (alpha, beta, delta) with F = 1 - (beta/(x^delta+beta))^alpha is
  `Burr(alpha, delta, beta^(1/delta))`.
* Negative binomial (alpha, nu), mean nu and variance nu + nu^2/alpha, is
  `NegativeBinomial(r=alpha, beta=nu/alpha)`; with contagion gamma = 1/alpha it
  is `claim_count(nu, 1 + gamma*nu)`.
* The midpoint discretization of section 4.3 is `Grid.rounding`; Ex 4.5's
  printed grid moments and recursion values reproduce exactly.
* The monograph's deductible credit C(d) = (E[X; d] + e F(d))/(E[X; b] + e) is
  measured against the basic limit b; `prospicio.pricing.loss_elimination_ratio`
  measures E[X; d]/E[X] against the unlimited mean (a factor of more than two
  apart for Ex 6.7's lognormal). In Problems 6.13 and 6.14 the "LER" counts
  the limit's elimination too.

# Printed numbers that do not reproduce

* Rounding carried into differences: Ex 5.8's excess mean 8,515 (8,518, as
  P5.2(e) itself prints) and layer CV 0.6452 (0.6385; the rounded inputs give
  0.6452), Ex 5.10's 2,701 (2,699) and 1.27% (1.22%), Ex 5.13's E[S] 5,775 and
  Var 24,178,800 (5,785 and 24,117,273), Ex 6.9's diminishing credits in the
  fourth decimal, and a fourth decimal of some ILFs.
* Slips: Ex 2.7's beta 397.931 (618,850/1,555 = 397.974; its tables use the
  printed value), P6.10's 3.4114 (4.3114), Table 6.7's franchise 0.0864 (0.0846),
  P6.16's 3.973 (3.794), Ex 4.5's grid mean 1,199.88 for the step of 20
  (1,199.98).
* Ex 6.6 states lognormal mu = 7.000; its E[X; 500K] = 21,743, E[X; 2M] = 28,338,
  sd 151,311, skewness 9.4728 and all of Table 6.4 need mu = 7.6.
* Ex 6.6, Table 6.4, P6.11 and P6.12 price aggregate limits with a shifted
  gamma; prospicio's shifted gamma reproduces them within a unit, but the exact
  aggregate by FFT removes 8, not 91, under a 3M aggregate limit on 2M claims,
  and P6.11's ratios are three to ten times the exact ones.
* Solver results: Ex 2.8's minimum chi-square 1.828 is 1.020 at its own
  parameters (whose fitted table does reproduce) and the true minimum is 0.614
  at (7.2857, 0.4570); Ex 5.3's 1.6610 is 1.806 at its parameters (minimum
  1.790 at (8.6869, 1.1684)); P5.23's parameters are not the minimum under any
  reading of its cells tried. Ex 2.10, 2.11, P2.16, P2.39 and P2.43 reproduce to
  four decimals by the same methods (SciPy Nelder-Mead on prospicio's cdf and lev).
* Chi-squares with rounded expected counts: Ex 2.7 (3.095 for 3.096), Ex 3.3
  (2.45 for 2.43). Ex 3.6's 0.7753 is about 0.65 here, source unclear; its alpha
  2.44285 comes from a variance rounded to 0.130074 (2.44302 unrounded).

# Gaps in prospicio met

* No mean-excess-regression fit. (Closed by #328: minimum chi-square and
  grouped maximum likelihood in `grouped_fit::fit_grouped`, the minimum
  limited-mean distance in `fit_lev_distance`, truncation from above in
  `LargeLosses::truncated_above`, and the Lomax in `fit_severity`; they
  reproduce Ex 2.7, 2.10, 2.11, P2.16, P2.39 and P2.43 to the printed
  digits.)
* No `layer_second_moment` on `Lognormal` itself (use `Truncated(ln)`), no third
  limited moment, no `lev` on `Sampled`.
* No stop-loss or limited mean on an `Approximation` (the shifted gamma's is
  `Gamma.stop_loss` of its unshifted part); no Heckman-Meyers inversion (the
  FFT replaces it). (The normal power and Wilson-Hilferty approximations
  came with #329.)
* No count thinning or contagion accessor, no count fitting beyond
  `NegativeBinomial.from_mean_variance`.
