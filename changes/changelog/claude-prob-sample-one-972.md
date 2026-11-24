### Distributions and probability

- `Distribution::sample_one(&mut StreamRng) -> f64` draws one value without
  allocating a one-element vector. `k` calls give the same numbers, bit for
  bit, as `sample(rng, k)` from the same stream; `sample` now defaults to
  `n` calls of `sample_one`, and `Gamma` and `InverseGamma` override
  `sample_one`. Seeded draws are unchanged. A type outside the crate that
  overrides `sample` with a different sampler should override `sample_one`
  too (#972, #1055).
