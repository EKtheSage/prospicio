### Distributions and probability

- Rust: `risk::quantile_type1_sorted` and `risk::quantile_type7_sorted`
  give R's `quantile(type = 1)` and `quantile(type = 7)` bit for bit,
  R's four-ulp fuzz included; the explanation kernels (`ale_grid`,
  `ale_numeric_distance`) use them instead of their own copies (#977).
