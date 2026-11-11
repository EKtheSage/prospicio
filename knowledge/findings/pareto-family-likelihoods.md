---
type: Finding
title: The Pareto families score on the body fits' likelihood only when every reporting threshold is at least t, and Riegel's GPD from t is the Lomax conditioned on t
description: "fit_severity_with_threshold fits the single Pareto and Riegel's generalized Pareto from a threshold t and scores them on the likelihood the body families maximize, each loss w (ln f(y) - ln S(a)). The Pareto families' survival is 1 below t, so they condition on max(r, t) where a body fit conditions on r; the two AICs compare only when every r >= t. With r = t for every loss, the Lomax's excess over t is a GPD, so the Riegel fit and the Lomax fit reach the same log-likelihood (scipy parity to 1e-6 on 89 synthetic losses above 150,000). The exponential's truncated, censored MLE is sum w (y - r) / (uncensored weight). Nine families at parity with SciPy's own densities."
tags: [prospicio-prob, severity, pareto, maximum-likelihood, aic]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-11T12:00:00Z }
verified: { by: process:ci, at: 2026-11-11T12:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-prob/src/severity_fit.rs
    title: fit_severity_with_threshold, Family::ALL
  - id: test
    resource: ../validation/tests/severity_fit.rs
    title: every_family_matches_scipy and the_pareto_families_score_on_the_body_likelihood
  - id: script
    resource: ../validation/scripts/scipy_severity_fits.py
    title: the SciPy reference (Nelder-Mead then BFGS on logpdf and logsf)
---

# The Pareto families on the body fits' likelihood

Issue #800 asked for one ranking by AIC of seven to nine severity families
on large losses above a reporting threshold: the body families (lognormal,
gamma, Weibull, loglogistic, exponential, Burr, Lomax) and the two Pareto
families fitted from a threshold `t` (the single Pareto and Riegel's
generalized Pareto).

## One likelihood

Every family is scored on the likelihood `fit_severity` maximizes: loss
`i` with weight `w`, reporting threshold `r` and value `y` adds
`w (ln f(y) − ln S(r))`, or `w (ln S(y) − ln S(r))` when censored. The
Pareto families' survival is 1 below `t`, so `ln S(r) = 0` for `r < t`:
they condition each loss on `max(r, t)`, a body family on `r`. Their
log-likelihoods, and so their AICs, compare only when every reporting
threshold is at least `t`. `t` is pinned, so it is not counted in `k`
(pareto 1, gen_pareto 2, exponential 1).

## Two identities the tests check

- With every reporting threshold at `t`, the Lomax conditioned on `X > t`
  has excess `X − t` distributed as a GPD with `ξ = 1/α` and scale
  `(θ + t)/α`. Riegel's form at location `t` has `ξ = 1/α_tail` and
  `β = t/α_ini`, so the two fits reach the same maximum:
  `α_tail = α` and `α_ini = α t/(θ + t)`, the Lomax's local Pareto alpha at
  `t`. On the synthetic losses (and in issue #800's table, -1181.4314 for
  both) the log-likelihoods agree to 1e-6.
- The exponential is memoryless, so with reporting thresholds and
  censoring its maximum is in closed form, `θ = Σ w (y − r) / Σ_uncensored w`
  (`mean(y) − t` with one threshold and no censoring); the standard error
  is `θ / sqrt(n_uncensored)`.

## Parity

`validation/scripts/scipy_severity_fits.py` fits every family with
SciPy's `lognorm`, `gamma`, `weibull_min`, `fisk`, `expon`, `lomax`,
`pareto` and `genpareto` (Nelder-Mead then BFGS on `logpdf` and `logsf`)
on two synthetic sets: 89 lognormal losses above 150,000 (a stand-in for
issue #800's data, which lives in another repository) and 172 losses with
reporting thresholds 0, 50,000 and 150,000 in turn and a 400,000 cap.
Parameters agree to 1e-5 relative and log-likelihoods to 1e-6. On the
first set the gamma's shape runs to 0, as in the issue's table, and only
its log-likelihood is compared (to 1e-4). The issue's own test vectors
could not be run here.
