---
type: Finding
title: ALEPlot's grid depends on how R's seq rounds, ALEPlot is archived on CRAN, and shap's KernelExplainer is exact once it enumerates every coalition
description: "ALEPlot 1.1 cuts at quantile(x, seq(1/K, 1, length.out = K), type = 1); seq's rounding (0.30000000000000004 at K = 10) puts n p just above a whole number, so type 1 takes the next order statistic, and an ALE built on exact fractions differs by 0.2% in an effect; computing the probabilities as 1/K + k (1 - 1/K)/(K - 1) reproduces R. ALEPlot is archived on CRAN (install the 1.1 tarball from the archive with yaImpute). shap 0.54.1's KernelExplainer with nsamples = 2^M and l1_reg = False enumerates every coalition, and prospicio's exact shap_values match it to 1e-9; a log output must stay finite, since shap's LARS step fails on NaN."
tags: [explanations, ale, shap, aleplot, r-parity, python-parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-03T12:00:00Z }
verified: { by: process:ci, at: 2026-11-03T12:00:00Z }
sources:
  - id: aleplot
    resource: https://cran.r-project.org/src/contrib/Archive/ALEPlot/
    title: R ALEPlot 1.1 (Apley), archived on CRAN
  - id: shap
    resource: https://pypi.org/project/shap/0.54.1/
    title: shap 0.54.1, KernelExplainer
  - id: scripts
    resource: ../validation/scripts/r_ale.R
    title: r_ale.R and shap_kernel.py, the references on validation/data/explain_frame.csv
  - id: tests
    resource: ../python/tests/test_models.py
    title: test_ale_matches_aleplot and test_shap_values_match_kernel_shap, and their R twins in R/prospicio/tests/test-models.R
---

# Facts

* ALEPlot's numeric grid is `c(min(x), quantile(x, seq(1/K, 1, length.out
  = K), type = 1))`, made unique. R's `seq` computes `from + (0:(K-1)) *
  by` with `by = (1 - 1/K) / (K - 1)`, so at `K = 10` the third
  probability is 0.30000000000000004; with 300 rows `n p` is then
  90.00000000000001, above R's fuzz of four machine epsilons, and type 1
  takes the 91st order statistic, not the 90th. ALE on exact fractions
  moved two grid points and an effect by 0.2%; the same arithmetic in
  Python reproduces ALEPlot to 1e-10.
* ALEPlot's intervals are `cut(x, z, include.lowest = TRUE)`, right-closed
  with the first also closed below; each holds at least one row, since its
  upper end is a value of `x`. The effects are centred by the trapezoid
  rule weighted by the interval counts; a factor's on its level
  frequencies.
* ALEPlot orders a factor's levels by `cmdscale(D, k = 1)` of summed
  distances over the other columns (Kolmogorov-Smirnov on 100 type 7
  quantiles for a numeric column, half the L1 distance of the conditional
  frequencies for a categorical one); scaling fixes the order only up to
  direction, so prospicio reads it from the end whose level sorts first.
* ALEPlot is archived on CRAN; version 1.1 installs from the archive after
  `yaImpute`.
* shap 0.54.1's `KernelExplainer(f, background).shap_values(x, nsamples =
  2^M, l1_reg = False)` enumerates every coalition and returns the exact
  interventional Shapley values. On a function with negative values a log
  output gives NaN, and shap's LARS feature selection then fails with
  "cannot convert float NaN to integer".
