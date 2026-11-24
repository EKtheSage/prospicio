---
type: Reference
title: The apc packages for age-period-cohort reserving (R apc archived, Python apc 1.0.2)
description: Nielsen's R apc, the reference implementation of Kuang, Nielsen and Nielsen's canonical APC parameterisation and its I2, I1 and I0 forecast rules, was archived on CRAN on 2026-03-02; Harnau's Python apc 1.0.2 (2020) implements the same; their reserving datasets and the documented Taylor-Ashe forecast.
resource: https://cran.r-project.org/src/contrib/Archive/apc/
tags: [reserving, apc, glm, calendar-year, inflation, reference]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-24T12:00:00-08:00 }
sources:
  - id: cran-archive
    resource: https://cran.r-project.org/src/contrib/Archive/apc/
    title: R apc on the CRAN archive (1.0 to 3.0.0)
  - id: rdrr
    resource: https://rdrr.io/cran/apc/
    title: R apc 2.0.1 documentation
  - id: pypi
    resource: https://pypi.org/project/apc/
    title: Python apc 1.0.2 (Jonas Harnau)
  - id: knn
    resource: https://doi.org/10.1093/biomet/asn038
    title: Kuang, Nielsen and Nielsen (2008), Identification of the age-period-cohort model and the extended chain-ladder model, Biometrika 95(4)
---

# What it is

* R `apc` (Nielsen and others): `apc.data.list`, `apc.fit.model(data,
  model.family, model.design)` with families such as `poisson.response`,
  `od.poisson.response` and `log.normal.response` and designs `APC`, `AP`,
  `AC`, `PC` and smaller; `apc.forecast.apc(fit, extrapolation.type)`
  with `I2`, `I1` and `I0` (the default) for the period effect;
  `apc.forecast.ac` for the chain ladder. Fits report the canonical
  coefficients (three anchors and the second differences) and their
  covariance.
* Archived on CRAN on 2026-03-02 ("issues were not corrected despite
  reminders"); the last version is 3.0.0 (2025-07-07). A generator must
  install from the archive with a pinned version.
* Reserving datasets: `data.loss.TA` (Taylor-Ashe, which is
  `validation/data/genins.csv`), `data.loss.BZ` (Barnett and Zehnwirth,
  with counts), `data.loss.VNJ`, `data.loss.XL`.
* The package documentation's Poisson APC fit of Taylor-Ashe forecasts a
  total reserve of 19,061,428 under `I0` (from the documentation, not
  run here); the chain ladder gives 18,680,856.
* Python `apc` 1.0.2 (Jonas Harnau, PyPI, 2020-01-22) follows the same
  parameterisation: `apc.Model()`, `data_from_df`, `fit(family,
  predictor)`, `forecast()`, with `apc.loss_TA()`.
* The AP design is Taylor's separation model as a GLM (see
  [Taylor 1977](taylor-1977-separation.md)).

# Quirks found building the APC GLM (2026-11-24, #979)

* **Canonical design.** Python `apc` 1.0.2 and the Rust `ApcGlm` use the
  same columns: a level, a development slope (the 0-based development
  index) and an accident slope (the 0-based origin index), then hinge
  columns `(x - l + 1)+` for `l = 2..k - 1` on the development index, the
  origin index and the calendar index `i + j`: `3k - 3` parameters on a
  `k` by `k` triangle. `apc` calls the effects age, cohort and period and
  anchors them at the first cell; the sub-models keep the level and both
  slopes and drop a block of hinges (`AC` the period block, `AP` the
  cohort block, `PC` the age block). `AC` on Taylor-Ashe gives the chain
  ladder's 18,680,856.
* **Python `apc` 1.0.2 does not run on current libraries.** It imports
  `pkg_resources` (pin `setuptools<81`), indexes string-labelled columns
  by integer slices (pandas 2 raises; use pandas 1.5.3 with numpy 1.26.4)
  and its age and cohort second-difference loops end with one iteration
  that selects no rows, which pandas 1.5.3 still rejects as a shape
  mismatch; skipping that empty iteration gives the design of the other
  iterations. Its `forecast()` fails on a triangle (`matrices are not
  aligned`: the forecast design has the future period columns) and only
  continues the last calendar trend, which is `I2`; it has no `I1` or
  `I0`.
* **The extrapolation rules, from R `apc`'s source** (`apc.forecast.apc`,
  read on the CRAN mirror): with `DD` the `n = k - 2` fitted period second
  differences, `I2` sets the future ones to 0; `I1` sets the first to
  `mean(cumsum(DD)) - cumsum(DD)[n]` (mean over all `n`) and the rest to
  0; `I0` regresses all `n` points of `cumsum(cumsum(DD))` on an intercept
  and the trend `1..n` demeaned, takes the line at `n + 1` and `n + 2`,
  and sets the first two future `DD` to the second differences of the
  series that ends `y[n-1], y[n], f1, f2`, the rest 0. Only this form
  reproduces the documented Taylor-Ashe `I0` total of 19,061,428 (it
  gives 19,061,428.33 here; including a leading zero in the cumsums gives
  18.2 million or 16.5 million). The Poisson APC's totals on Taylor-Ashe
  are 27.63 million (`I2`), 19.65 million (`I1`) and 19.06 million
  (`I0`); the deviance is 1,395,518. `I1` is confirmed by the source (cran/apc `R/apc_forecast.R` line 673, `mean(cumsum(xi.per.dd)) - cumsum(xi.per.dd)[length(xi.per.dd)]`; lines 586-596 document I1 and I0); only its total has no independent number.
* **Sign and scale.** The second differences are on the log scale of the
  mean (log link), so a positive `DD` is an accelerating effect; the
  period effect is identified only up to a linear trend, and the rules are
  the ones that do not depend on that trend.
