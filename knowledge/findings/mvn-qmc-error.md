---
type: Finding
title: Genz's multivariate normal by scrambled Sobol points is unbiased against mvtnorm, and three standard errors over ten scrambles is exceeded in about 1.5% of calls
description: "prospicio_math::mvn estimates P(a < X < b) by ten independently scrambled Sobol sequences over the same points; on 32 rectangles and orthants in 3 to 10 dimensions its value agrees with R mvtnorm 1.2.4's pmvnorm (GenzBretz, abseps 1e-8) without bias (the signs of the differences are mixed and the differences shrink with abs_tol), but |difference| / standard error exceeds 2 in 8 of 96 runs and 3 in about 1.5%, as a t with 9 degrees of freedom predicts, so a parity check takes twice the two errors' sum."
tags: [probability, copula, multivariate-normal, quasi-monte-carlo, sobol, mvtnorm, prospicio-math]
status: draft
verified: { by: process:ci, at: 2026-10-29T12:00:00Z }
generated: { by: claude-code/cloud-session, at: 2026-10-29T12:00:00Z }
sources:
  - id: design
    resource: ../docs/design/risk.md
    title: risk.md, "Sobol points and the multivariate normal in any dimension"
  - id: reference
    resource: ../validation/reference/mvn_mvtnorm.csv
    title: pmvnorm reference values
---

# The multivariate normal's error estimate

`mvn_probability` reports as its error three standard errors of the mean
over its scrambles. The standard error is itself estimated from ten
scrambles, so the ratio of the true error to it is a t with 9 degrees of
freedom: it exceeds 3 in about 1.5% of calls and 2 in about 7.6%.

Run three ways on the 32 rows of `validation/reference/mvn_mvtnorm.csv`
(abs_tol `1e-6` at seeds 0 and 1, and `1e-7` at seed 0), the difference
from mvtnorm 1.2.4 exceeded two of our standard errors in 8 of 96 runs
(one 8-dimensional run at `1e-7` is explained by mvtnorm's own error of
`2.1e-7`), with both signs, and shrank as `abs_tol` fell: the estimate is
unbiased and the error estimate is as honest as its degrees of freedom
allow. The parity test therefore compares within twice the sum of both
errors.
