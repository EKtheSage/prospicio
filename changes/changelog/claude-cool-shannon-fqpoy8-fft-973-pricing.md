### Pricing

- `natural::Portfolio::from_independent` is 2.2 to 3.7 times faster: its
  transforms are real ones on one plan, through `prospicio_math::fft`.
  Results move by round-off (#973, #1051).
