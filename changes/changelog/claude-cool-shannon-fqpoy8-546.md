### Models

- Conformal prediction intervals around any model: `prospicio_models::conformal` with `split` (absolute, Pearson or scaled scores), `weighted_split`, `cqr` (conformalised quantile regression) and `cv_plus` (CV+ and the jackknife+), on predictions from any engine, and `split_fitted` and `cv_plus_fit` on Rust models; at parity with MAPIE 0.9.1 (`models.md`, "Conformal prediction"). Python `conformal_split`, `conformal_cqr`, `conformal_plus`, `conformal_intervals`; R the same names (#546).
