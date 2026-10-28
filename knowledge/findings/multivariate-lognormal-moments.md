---
type: Finding
title: A multivariate lognormal matches a mean and covariance exactly, but a positive definite covariance can have no lognormal
description: "log_cov_ij = ln(1 + cov_ij / (m_i m_j)) and mu_i = ln m_i - log_cov_ii / 2 reproduce the mean and covariance on the original scale to rounding; a covariance at or below -m_i m_j has no lognormal, and the log-scale covariance can fail to be positive definite when the original one is (correlation -0.9 at cv 1)."
tags: [prospicio-prob, multivariate, lognormal, normal, covariance, capital]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-27T18:00:00Z }
verified: { by: process:ci, at: 2026-10-27T18:00:00Z }
sources:
  - id: module
    resource: ../crates/prospicio-prob/src/multivariate.rs
    title: prospicio_prob::multivariate and its unit tests
  - id: python
    resource: ../python/tests/test_risk.py
    title: test_multivariate_normal_and_lognormal_joints, against scipy.stats by Kolmogorov-Smirnov
  - id: note
    resource: ../docs/design/predictive-distribution.md
    title: Design note, Decisions (#564)
---

# A multivariate lognormal from its moments

For `X = exp(Y)` with `Y` multivariate normal (mean `mu`, covariance `S`),
`E X_i = exp(mu_i + S_ii / 2)` and
`Cov(X_i, X_j) = E X_i E X_j (exp(S_ij) - 1)`. Inverting,

    S_ij = ln(1 + cov_ij / (m_i m_j)),   mu_i = ln m_i - S_ii / 2,

so a mean `m` and covariance `cov` are matched exactly, the conversion
behind Mack's lognormal reserve. On three components with standard
deviations 20, 10, 30 and correlations 0.6, -0.2, 0.3 the round trip is
exact to 1e-12 relative, and 400,000 draws reproduce the covariance
within 2% of its scale.

Two limits follow from the formula:

- `1 + cov_ij / (m_i m_j)` must be positive: a pair of lognormals cannot
  have a covariance at or below `-m_i m_j`, so strongly negative
  correlations between skewed components are impossible.
- `S` must be positive definite, and it can fail to be when `cov` is.
  With means 1, variances 1 (cv 1) and correlation -0.9, `cov` is
  positive definite but `S` has diagonal `ln 2 = 0.69` and off-diagonal
  `ln 0.1 = -2.30`, so its determinant is negative.

`MultivariateLognormal::from_moments` refuses both with a message naming
the cell or the log scale. A multivariate normal needs only `cov`
positive definite; a singular one (a zero variance or a perfect
correlation) is refused rather than drawn degenerate.
