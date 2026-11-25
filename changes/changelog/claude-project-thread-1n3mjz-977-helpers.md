### Models

- `prospicio.models.mean_deviance`, `weighted_quantile`, `base_score`,
  `pearson_terms`, `pearson_dispersion` and `monotone_violations` in Python,
  from the Rust core. Python's `Booster` and R's `booster_fit()` start from
  the same base score, and compute the same dispersion and monotonicity
  check, through these functions instead of their own copies (#977).
- R's `bayes_glmm_model@coefficients` takes the posterior means in Rust
  instead of copying every draw into R.
