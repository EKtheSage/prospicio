### Models
- Faster GLM fits: `X'WX` and `X'Wz` are summed in fixed row blocks in
  parallel, each block reducing contiguous column slices, and IRLS reuses
  its buffers; the Bayesian GLM's log density and gradient run in row
  blocks in parallel. A Poisson GLM on 678,013 rows and 60 columns
  fits in about 1 s instead of 20 to 90 s, and the Bayesian GLM samples
  about 5 times faster. New `Design::linear_predictor_into`
  writes `X beta + offset` into a buffer. Benchmarks:
  `cargo bench -p prospicio-glm` and `cargo bench -p prospicio-bayes`
  (#969).

### Breaking changes
- Seeded results move in the last bits: GLM and GAM fits (coefficients,
  standard errors, smoothing parameters) and the Bayesian GLM's NUTS
  draws, since the row sums are now added in blocks. Results no longer
  depend on the order of a row-by-row loop, and stay the same on any
  number of threads (#969).
