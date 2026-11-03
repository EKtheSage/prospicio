### Models

- Zero-inflated and hurdle Poisson and negative binomial models
  (`prospicio_glm::zero::ZeroInflated`, `Hurdle`, `TwoPartFit`) with the
  probability of each count, joint draws and Vuong's test (`vuong`,
  `count_log_likelihoods`), matching pscl 1.5.9's `zeroinfl` and `hurdle`
  maximised to the end; Python `ZeroInflated`, `Hurdle`, R
  `zeroinfl_fit(y ~ x | z)`, `hurdle_fit()`, `vuong_test()` (#548).
- Linear quantile regression (`prospicio_glm::quantile::QuantileRegression`)
  with Powell's kernel and paired bootstrap standard errors, several `tau`
  at once with rearrangement, and expectile regression
  (`ExpectileRegression`), penalised or not, with sandwich standard
  errors, matching quantreg 5.97's `rq(method = "br")` and `summary.rq(se
  = "ker")` and expectreg 0.54's `expectreg.ls`; Python
  `QuantileRegression`, `ExpectileRegression`, R `quantreg_fit()`,
  `expectreg_fit()` (#550).
