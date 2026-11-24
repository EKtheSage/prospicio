### Aggregate and reinsurance

- `fft_bivariate`, `finite_horizon`, `finite_horizon_poisson`,
  `wiener_hopf` and `renewal_count` use `prospicio_math::fft`: the
  bivariate compound is 1.2 to 1.4 times faster and the finite-time ruin
  probabilities 1.7 to 2.4 times, by real-input transforms. Results move by
  round-off (#973, #1050).

### Distributions and probability

- `prospicio_math::fft::Plan` and `RealPlan`: the transforms of one length,
  planned once, for a caller that transforms that length several times
  (#973, #1050).
