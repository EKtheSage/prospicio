### Models

- R's `save_model()` and `load_model()` also save the models from
  `glmm_fit()`, `bayes_glm_fit()`, `bayes_glmm_fit()`, `dglm_fit()`,
  `zeroinfl_fit()`, `hurdle_fit()` and `expectreg_fit()`, with every
  setting `predict()` needs (the grouping factors, the dispersion and
  zero-part formulas); the estimates round-trip exactly. Files saved by
  earlier versions still load (#978).
