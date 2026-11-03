---
type: Finding
title: quantreg's rho leaves the weights out, expectreg centres a parametric term, and expectreg's own basis is reproduced from its design matrix
description: "R quantreg 5.97's rq(weights = w) minimises the weighted check loss but reports rho as the loss without the weights; rq(method = 'fn') stops at a duality gap of 1e-6, so its coefficients agree with method = 'br''s vertex to about 1e-5; summary.rq's default standard errors depend on the size (rank below 1,001 rows, nid above). expectreg 0.54 centres a parametric covariate, so its intercept is the expectile at the covariate's mean; rb(type = 'special', B =, P =) fails on R 4.3 (a scalar || on a matrix), so a P-spline check takes expectreg's own design ($design) and penalty ($bases$x$P) from a fit at a fixed lambda. prospicio's QuantileRegression and ExpectileRegression match br to 1e-10, the kernel standard errors to 1e-9 and expectreg to 1e-9 (linear) and 1e-8 (P-spline)."
tags: [prospicio-glm, quantile-regression, expectile, quantreg, expectreg, r-parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-03T12:00:00Z }
verified: { by: process:ci, at: 2026-11-03T12:00:00Z }
sources:
  - id: quantreg
    resource: https://cran.r-project.org/package=quantreg
    title: R quantreg 5.97 (Koenker), rq(), summary.rq(), bandwidth.rq()
  - id: expectreg
    resource: https://cran.r-project.org/package=expectreg
    title: R expectreg 0.54 (Sobotka, Schnabel, Schulze Waltrup and others), expectreg.ls()
  - id: script
    resource: ../validation/scripts/r_quantile_expectile.R
    title: The references on Engel's data and a seeded weighted loss book
  - id: test
    resource: ../validation/tests/models.rs
    title: quantile_and_expectile_regression_match_quantreg_and_expectreg, 135 cases of validation/reference/quantile_expectile_r.csv
---

# Facts

* `rq(y ~ x, weights = w)` scales the rows by `w` and minimises
  `sum(w * rho_tau(y - x b))`, but its `rho` is `sum(rho_tau(y - x b))`
  over the unscaled residuals: on the seeded book (weights 1 to 3) 981.9
  at `tau = 0.5` against the weighted 2,124.8. prospicio's `objective` is
  the weighted loss.
* `rq(method = "fn")` (Frisch-Newton) stops at a duality gap of 1e-6, so
  its coefficients differ from `method = "br"`'s exact vertex by about
  1e-5 relative; an interior point followed by the exact fit through the
  `p` closest rows gives the vertex to 1e-10.
* `summary.rq(se = "ker")` scales the rows and residuals by the weights,
  takes the bandwidth of `bandwidth.rq` (Hall and Sheather by default),
  halves it until `tau +- h` lies in (0, 1), and uses
  `min(sd, IQR / 1.34)` of the scaled residuals, the IQR from R's type 7
  quantiles. Without `se`, `summary.rq` uses `rank` below 1,001 rows and
  `nid` above, not `ker`.
* `expectreg.ls(y ~ rb(x, "parametric"))` centres `x`: its `intercepts`
  are the expectile at the mean of `x`, so the uncentred intercept is
  `intercept - slope * mean(x)`. With `center = FALSE` it drops the
  intercept.
* `rb(x, "special", B = B, P = P)` fails in expectreg 0.54 on R 4.3 with
  "'length = 2350' in coercion to 'logical(1)'" (`is.na(B) || is.na(P)`
  on matrices). A fit `rb(x, "pspline")` at `smooth = "fixed"` exposes its
  design (`$design`, intercept first) and penalty (`$bases$x$P`), and
  asymmetric least squares with `lambda P` on the spline columns
  reproduces its coefficients to 1e-8.
* expectreg reports no standard errors for `laws`; the sandwich
  `A^-1 B A^-1` of asymmetric least squares at `tau = 0.5` is `lm`'s HC0
  (`sandwich::vcovHC(type = "HC0")`).
