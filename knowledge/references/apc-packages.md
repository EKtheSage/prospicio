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
