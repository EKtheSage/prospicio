### Models

- The effective sample sizes of a Bayesian fit are computed 9 to 12 times
  faster: each chain's autocovariance takes real transforms through
  `prospicio_math::fft`, which keeps the plan between chains. Results move
  by round-off (#973, #1052).
