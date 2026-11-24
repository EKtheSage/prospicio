### Aggregate and reinsurance

- `fft` and `audit_grid`'s FFT cells are 2.8 to 4.6 times faster: the
  aliasing audit reads the short buffer off the long one instead of
  transforming twice, and the severity takes a real-input transform, with
  half the pgf evaluations. `aliasing_error` is measured as before; results
  move by round-off (`1e-15`) (#973, #1049).

### Distributions and probability

- `prospicio_math::fft`: `forward`, `inverse`, `real_forward` and
  `real_inverse`, with each thread's plans and scratch space kept between
  calls (#973, #1049).
