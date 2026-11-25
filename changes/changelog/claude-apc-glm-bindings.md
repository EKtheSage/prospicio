### Reserving

- The age-period-cohort reserving GLM in Python and R (#979, PR 7, #1086):
  `ApcGlm` (effects `"APC"`, `"AP"`, `"AC"` or `"PC"`; extrapolation `"I0"`,
  `"I1"` or `"I2"`; the Poisson, gamma or Tweedie family) with `fit` and
  `ApcGlmFit` (canonical level, slopes and second differences, fitted
  cells, reserves, `future` as `FutureCells`, `odp_fit`,
  `prediction_error()` and `predict_distribution()`) in
  `prospicio.reserving`; `apc_glm()` and `apc_glm_fit` in R. The
  arithmetic separation method as a Poisson GLM, with its prediction
  error: `Separation.glm()` in Python, `separation_glm()` in R.
