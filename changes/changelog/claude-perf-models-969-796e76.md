### Models
- Faster GLM fits: `X'WX` and `X'Wz` are summed in fixed row blocks in
  parallel, each block reducing contiguous column slices, and IRLS reuses
  its buffers; the Bayesian GLM's log density and gradient run in row
  blocks in parallel. A Poisson GLM on 678,013 rows and 60 columns
  fits in about 1 s instead of 20 to 90 s, and the Bayesian GLM samples
  about 5 times faster. New `Design::linear_predictor_into`
  writes `X beta + offset` into a buffer. Benchmarks:
  `cargo run --release -p prospicio-glm --example glm_bench` and
  `cargo run --release -p prospicio-bayes --example bayes_glm_bench`
  (#969, #1059).

### Breaking changes
- Seeded results move in the last bits, since the row sums are now added
  in blocks (eight running sums even on a small triangle): GLM, GAM and
  every other `prospicio-glm` fit that runs IRLS (among them the Laplace `Glmm`, the
  Tweedie profile, the double and zero-inflated GLMs: coefficients,
  standard errors, smoothing parameters), the Bayesian GLM's and
  `BayesGlmm`'s NUTS draws (`BayesGlmm` starts from the Laplace fit), and
  the Reserving lane's ODP GLM (`odp_glm`) and diagonal backtest fits,
  which call `Glm::fit`. Results no longer
  depend on the order of a row-by-row loop, and stay the same on any
  number of threads (#969, #1059).
