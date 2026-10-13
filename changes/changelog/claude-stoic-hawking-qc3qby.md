### Aggregate and reinsurance

- Two lines hit by the same events: their joint annual loss by a 2-D
  compound FFT from a joint per-event severity
  (`prospicio_aggregate::bivariate::{fft_bivariate, JointGrid}`, with each
  line's marginal, the total, mixed moments and the correlation), and
  `aggregate`'s `clash` model, given by the expected events hitting line
  A only, line B only and both (`Clash`), matching `aggregate` 1.0.1.
  Python `fft_bivariate`, `JointGrid`, `Clash`; R `fft_bivariate()`,
  `clash_model()`, `clash_severity()`, `clash_aggregate()`.
