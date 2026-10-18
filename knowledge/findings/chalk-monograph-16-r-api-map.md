---
type: Finding
title: The CAS Monograph 16 guides in R, what the R package covers
description: The calls of the two Python guides glm-to-comprehensive-pricing-chalk*.md mapped onto the R package; numpy's draws shipped as CSV files so R reads the same portfolio; every printed number equal to Python's, the LightGBM predictions bit for bit; no R gap beyond the four Models issues the Python guide filed.
tags: [models, glm, elastic-net, gam, boosting, lightgbm, r, python, api, bindings, guides]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-17T23:30:00-07:00 }
verified: { by: process:ci, at: 2026-10-17T23:30:00-07:00 }
sources:
  - id: guide
    resource: ../docs/guides/glm-to-comprehensive-pricing-chalk.qmd
    title: The R guide, part 1, every chunk's printed text checked by R/prospicio/tests/test-guides.R
  - id: guide2
    resource: ../docs/guides/glm-to-comprehensive-pricing-chalk-2.qmd
    title: The R guide, part 2
  - id: data
    resource: ../examples/data/make_chalk_portfolio.py
    title: The script that writes the Python guide's numpy draws to examples/data/chalk_portfolio.csv and chalk_toys.csv
  - id: boosting
    resource: ../R/prospicio/R/boosting.R
    title: The R boosting adapter on main, 2026-10-17
---

# Facts (2026-10-17)

* Python to R:[^guide][^guide2] `deviance(family, y, mu)` is
  `family_deviance(family, y, mu)`; `Glm("poisson").fit(Design(cols, names,
  offset=log_expo), y).predict(new)` is `predict(glm_fit(claims ~ ... +
  offset(log(exposure)), d), new)`, factors with the base level first giving
  the Python guide's dummies; `ElasticNet(family, alpha, lam).fit` is
  `elastic_net_fit(..., alpha =, lambda =)` (coefficients in
  `@coefficients[, 1]`), `ElasticNet.lambda_max` is the first value of
  `elastic_net_fit(..., lambda = NULL)@lambda`, `lambda_path(n, min_ratio)` is
  that path with `nlambda`, `lambda_min_ratio`, and `cross_validate(lams,
  splits)` is `elastic_net_cv(..., lambda = lams, foldid = fold + 1)`
  (`lambda_min`, `lambda_1se`, `fit@df`); `Gam(Glm("poisson"), [("age",
  12)])` is `gam_fit(..., smooths = "age", n_basis = 12)` and the monotone
  `{"bs": "mpd"}` is `bs = "mpd"`; `Booster(...).fit(d, y).predict(new)` is
  `predict(booster_fit(formula, d, n_rounds =, learning_rate =, params =,
  seed =), new)`. A design without an all-ones column is `y ~ 0 + level`.
* R cannot replay numpy's `default_rng`, so the portfolio (20,000 policies),
  the 300 true make effects, part 1's small examples and part 2's fold
  permutations are written by the Python guide's own code to CSV files
  (shortest round-trip decimals) and read by the R guides.[^data] With the
  same data every printed number is equal to Python's.
* The R and Python boosting adapters call LightGBM 4.7.0 with the same
  parameters (`seed`, `deterministic = TRUE`, the same base score and
  `init_score`).[^boosting] Given the same matrix (same columns in the same
  order, region and power as alphabetical level codes, as `np.unique` gives),
  fold 0's predictions were equal to the last bit on Windows.
* R gaps: none beyond those the Python guide filed for the Models lane (#355
  to #358); R has no `lambda_max()` or `lambda_path()` function, but the
  default path supplies both.
