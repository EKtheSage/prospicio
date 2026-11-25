### Distributions and probability

- Every copula pickles in Python, and a loaded copula samples the same
  numbers; in Rust, `GaussianCopula::from_factor` and
  `StudentTCopula::from_factor` rebuild one from its Cholesky factor bit
  for bit (#978).
