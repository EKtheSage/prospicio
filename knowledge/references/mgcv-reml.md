---
type: Reference
title: mgcv 1.9.1's REML score, the reference for prospicio's REML smoothing
description: mgcv's method = "REML" score (gcv.ubre) is the Laplace-approximate restricted likelihood with the observed (Newton) information and reml.scale as the dispersion, not sig2; its Newton search stops once the score settles to about 3e-7 relative, short of the minimum where the score is flat.
resource: https://cran.r-project.org/package=mgcv
tags: [gam, reml, smoothing, reference-implementation]
status: stable
verified: { by: process:ci, at: 2026-10-14T23:59:00Z }
generated: { by: claude-code/cloud-session, at: 2026-10-14T23:59:00Z }
sources:
  - id: wood
    resource: https://doi.org/10.1111/j.1467-9868.2010.00749.x
    title: Wood (2011), Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models
  - id: tests
    resource: ../validation/tests/models.rs
    title: gams_match_mgcv
---

mgcv's REML criterion (Wood 2011) reported as `gcv.ubre` is

`V = D_p/(2φ) − l_s(φ) + ½ ln|XᵀWX + S| − ½ ln|S|₊ − (M_p/2) ln(2πφ)`,

with `D_p` the deviance plus `β̂ᵀSβ̂`, `l_s` the saturated log-likelihood
of `fix.family.ls(family)$ls`, `|S|₊` the product of the penalty's
positive eigenvalues and `M_p` the dimension of its null space (the
parametric terms included).[^wood] Recomputed in R from a fit's pieces it
matches `gcv.ubre` to `1e-11` for the Gaussian and exactly for the
Poisson, but only under two conditions.

- **`W` is the observed (Newton) weight matrix.** For a gamma with a log
  link, the fit's `weights` (expected) give a score 2.04 too high; the
  observed weights `w y/μ` give it exactly. The two agree for canonical
  links.
- **`φ` is `reml.scale`.** For the gamma, `sig2` (0.4545, the Pearson-type
  estimate the fit reports) differs from `reml.scale` (0.5057, the
  dispersion REML estimates). For the Gaussian the two differ by about
  `1e-5`.

mgcv's Newton iteration stops once the score settles to about `3e-7`
relative. Where the score is flat in the smoothing parameters (a nearly
linear smooth, or two smooths searched jointly) prospicio's coordinate and
Nelder–Mead search reaches a lower score, by up to `2e-3` on a score of
4,700, with effective degrees of freedom up to 0.7% apart; where the smooth
is clearly non-linear the two agree to `1e-6`.[^tests]

[^wood]: Wood (2011)
[^tests]: gams_match_mgcv
