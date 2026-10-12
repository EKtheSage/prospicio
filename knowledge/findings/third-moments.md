---
type: Finding
title: Third raw moments, closed forms against mpmath
description: Severity::third_moment and Counting::third_moment match mpmath at 30 digits to 1e-10 on 16 severities, 5 counts and 2 Tweedies; mpmath's nsum fails on a Poisson mean of 1,000, and the mixed Poisson's pmf carries about 2e-12 relative error into its moments.
tags: [distributions, moments, parity, mpmath]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-12T15:57:31Z }
sources:
  - id: script
    resource: ../validation/scripts/mpmath_third_moments.py
    title: mpmath_third_moments.py, the 30-digit reference
  - id: tests
    resource: ../validation/tests/distributions.rs
    title: third_moments_match_mpmath and third_moments_diverge_combine_and_forward
---

# What was checked

`E[X^3] = 3 ∫ x² S(x) dx` by `mpmath.quad` on each severity's survival
function, independently of the closed forms in `prospicio-prob`, agrees
with them to `1e-10` on 16 severities: lognormal, gamma, Weibull,
loglogistic, inverse gamma, Burr, scaled beta, inverse Gaussian, Pareto
(untruncated and truncated, with `α = 2` and `α = 3` where the truncated
integral has the `3 − α` pole), and GPD (heavy and bounded tails). The
Tweedie's moment from its cumulants agrees with the compound
Poisson–gamma's.[^script]

# Numerical points

* `mpmath.nsum` over a Poisson pmf with mean 1,000 returns about `1e-410`
  instead of `1,003,001,000`: its series acceleration fails on terms that
  rise before they fall. An exact sum over `0..3000` is right.[^script]
* The default count moment, a sum of `k³ P(N = k)`, is as good as the pmf
  it sums: on a gamma-mixed Poisson (`λ = 8`, cv 0.5) it is `2e-12` low
  against the equal negative binomial's closed form, the mixed Poisson's
  pmf error.[^tests]

[^script]: mpmath_third_moments.py
[^tests]: distributions.rs
