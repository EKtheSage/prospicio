---
type: Reference
title: R earth's MARS forward pass, the reference for prospicio's MARS knots
description: earth 5.3.6 with degree = 1, the reference for prospicio_glm::mars; its span formulas, the gates on a knot's gain, a weighted pass that checks a pair's lower hinge against a basis with a constant intercept, and fixed-rank-deficient terms dropped afterwards.
resource: https://cran.r-project.org/package=earth
tags: [models, glm, mars, splines, parity, r]
status: stable
verified: { by: process:ci, at: 2026-10-21T12:00:00Z }
generated: { by: claude-code/cloud-session, at: 2026-10-21T12:00:00Z }
sources:
  - id: earth
    resource: https://cran.r-project.org/package=earth
    title: earth 5.3.6 (GPL-3), Milborrow, from Friedman's MARS and Hastie and Tibshirani's mda; src/earth.c
  - id: friedman
    resource: https://doi.org/10.1214/aos/1176347963
    title: Friedman (1991), Multivariate adaptive regression splines, Annals of Statistics 19(1)
  - id: script
    resource: ../validation/scripts/r_mars_earth.R
    title: r_mars_earth.R, the reference values for prospicio's MARS knots
---
# What it is

`earth(y ~ ., degree = 1)` adds hinge pairs `h(x - k)`, `h(k - x)` one step at a time, each step the variable and knot that most reduce the residual sum of squares, and then prunes by GCV. `prospicio_glm::mars::Mars` reproduces its knots, RSS and GCV exactly on the Chalk portfolio (`examples/data/chalk_portfolio.csv`, `freq ~ age + veh_age + exper + noise`), unweighted and weighted by exposure, with the defaults and with `nk = 15, thresh = 0`, with and without the backward pass (`validation/tests/models.rs::mars_matches_earth_on_the_chalk_portfolio`, Python and R tests too).

# The details that change the knots

- Spans (`GetSpanParams` in `earth.c`): the end span is `(int)(7.32193 + ln p / 0.69315)` and the minimum span `(int)((2.9702 + ln(p n)) / 1.7329)`, with `p` the predictors and `n` the rows with nonzero weight. Knots are scanned from the second largest x downwards, the counter moving only on rows with positive weight (unweighted, also only where the candidate column's centred sum of squares is positive).
- Unweighted, a knot is accepted when its gain beats the best so far and is below `min(1.01 RSS, 10 × the previous step's gain)`, starting from the linear term's gain; a new column is kept if its residual sum of squares after orthogonalizing exceeds 0.01 relative (`1e-5` after 15 columns). Weighted, the gain needs to beat the best by `1e-10 var(y)`.
- The pair's lower hinge enters as a new term only when the variable is not yet in the model (earth's `GetNewFormFlag`), and unweighted only when the residual sum of squares of x itself exceeds 0.01.
- The weighted pass checks the lower hinge with a basis whose intercept is the constant 1, not √w as in the weighted fit. A hinge can pass that check and be dependent in the real weighted basis; earth then prints "fixed rank deficient bx" and drops it. On the Chalk portfolio that is why weighted `exper 7` is a pair whose lower hinge vanishes while `age 79` enters as one term.
- A variable enters linearly when no knot beats the linear term (`Auto.linpreds = TRUE`).
- Stopping: no candidate, a step's R² gain below `thresh` (the step dropped), GRSq below −10, R² at least `1 − thresh`, or the terms reaching `nk`, by default `min(200, max(20, 2p)) + 1`.
- GCV: `RSS / (n (1 − (m + penalty (m − 1) / 2) / n)²)` with `m` the terms and `penalty = 2` for `degree = 1`.

# Where prospicio departs

- The GLM criterion (`Mars::glm`) is prospicio's own: each step scores candidates on the IRLS working response and weights, and the deviance takes the RSS's place. earth's `glm =` argument fits a GLM after a least-squares search, so there is no earth value to match.
- Speed: earth's weighted Chalk fit took about 28 s here, prospicio's about 0.2 s.
