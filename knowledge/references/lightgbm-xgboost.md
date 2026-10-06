---
type: Reference Implementation
title: LightGBM and XGBoost (boosting engines)
description: The engines behind actuarialrs.boosting and R's booster_fit(); how offsets, starting scores and precision behave, the R packages' prediction calls, and how to install them.
resource: https://lightgbm.readthedocs.io/
tags: [boosting, lightgbm, xgboost, python, r, offsets, quantile, dispersion]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-06T04:10:00Z }
verified: { by: process:ci, at: 2026-10-06T04:32:20Z }
sources:
  - id: adapter
    resource: ../python/actuarialrs/boosting.py
    title: actuarialrs.boosting
  - id: tests
    resource: ../python/tests/test_boosting.py
    title: Boosting adapter tests
  - id: radapter
    resource: ../R/actuarialrs/R/boosting.R
    title: R booster_fit() and its tests (R/actuarialrs/tests/test-boosting.R)
  - id: lgb
    resource: https://pypi.org/project/lightgbm/
    title: lightgbm 4.7.0 on PyPI
  - id: xgb
    resource: https://pypi.org/project/xgboost-cpu/
    title: xgboost-cpu 3.2.0 on PyPI
---

# Offsets and starting scores

* The offset (log exposure for a frequency) goes in as LightGBM's
  `init_score` and XGBoost's `base_margin`, on the link scale; predictions
  then need the offset added back (`raw_score=True` /
  `output_margin=True`) before the inverse link.[^adapter]
* With an offset given, the trees start from the offset alone: a log-link
  model of responses near 150 starts at a mean of 1 and 100 rounds at
  learning rate 0.05 leave it about 8% short. The adapter adds
  `log(Σ w y / Σ w e^offset)` (the identity link: the weighted mean of
  `y − offset`) to the offset and keeps it in the fit.[^tests]

# The R packages

Checked with R lightgbm 4.7.0 and xgboost 3.2.1.1 from CRAN.[^radapter]

* lightgbm: `lgb.Dataset(x, label, weight, init_score)`, then
  `predict(model, x, type = "raw")` gives the trees' raw score *without*
  the `init_score`, so the offset is added back by hand, as in Python.
* xgboost 3.x: `xgb.DMatrix(x, label, weight, base_margin)`, then
  `predict(model, xgb.DMatrix(x, base_margin = start), outputmargin =
  TRUE)` gives the margin *with* the new `base_margin` included.
* lightgbm's `predict` returns a named vector; the adapter strips the
  names.
* Both packages take minutes to compile from source (about 15 in the
  cloud container); CI's R job uses Posit's public binaries
  (`use-public-rspm: true`).

# Quantile objectives

* LightGBM `objective = "quantile"` takes the level as `alpha`; XGBoost
  `objective = "reg:quantileerror"` takes it as `quantile_alpha`, in both
  Python and R.[^adapter]
* Separate fits per level can cross; sorting each row's predictions
  across levels fixes that and never raises pinball loss. On 3,000 rows
  with a spread that grows with x, the [10%, 90%] band from 200 rounds
  covers 80% ± 4% of the training rows with either engine.[^tests]

# Dispersion model

* Pearson residuals from in-sample means of a flexible booster are
  biased low, because the trees partly fit the noise; 5-fold cross-fitted
  means fix that. With them, a gamma booster on the residuals recovers a
  dispersion of 0.25 against 1 (gamma shapes 4 and 1, 4,000 rows) to
  within 0.06 and 0.2, with either engine and in both languages.[^tests]
* The residuals can be 0 only when y equals the mean exactly; they are
  floored at `1e-12 × mean` because both engines' gamma objectives need a
  positive label.

# Precision

XGBoost computes margins in single precision: doubling every exposure
doubles its means only to about 3e-6 relative, where LightGBM's agree to
the 1e-9 the test asks.[^tests]

# Installing

* `xgboost-cpu` (5.6 MB manylinux x86-64 wheel) installs the same
  `xgboost` module as `xgboost`, whose Linux wheel bundles over 200 MB of
  GPU libraries; `python/pyproject.toml` uses the CPU wheel on Linux.[^xgb]
* `lightgbm` 4.7.0 is a 3.5 MB wheel.[^lgb]

[^adapter]: actuarialrs.boosting
[^tests]: Boosting adapter tests
[^radapter]: R booster_fit() and its tests
[^lgb]: lightgbm on PyPI
[^xgb]: xgboost-cpu on PyPI
