### Breaking changes

- `Discrete` (Rust `Discrete::new`, Python `Discrete(values, probs)`, R
  `discrete_distribution()`) refuses probabilities that do not sum to 1
  within `1e-5`, naming the sum, as `aggregate` 2.1.0 refuses `dsev`;
  before, it rescaled any positive total. Rescale weights or counts on
  purpose with `Discrete::normalized`, Python `normalize=True` or R
  `normalize = TRUE`. Python `Count.fixed` takes `3.0` and refuses a
  fractional count such as `2.5` with a `ValueError`, as R
  `fixed_count()` already did (#774, #815).
