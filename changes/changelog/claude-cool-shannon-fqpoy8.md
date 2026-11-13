### Distributions and probability

- A law's distribution on a grid from its characteristic function,
  `aggregate`'s `AggregateCHF` without tilting (`prospicio_prob::chf`:
  `invert` on a normal, gamma, exponential, inverse Gaussian, Tweedie with
  `1 < p < 2`, a scaled and shifted one of them, or your own
  characteristic function on a given grid; the window sized from the
  Chernoff edges and the bucket walked down past a truncation of 1e-6;
  Python `invert_chf` and `ChfDistribution`, R `invert_chf()` and
  `chf_to_grid()`), at parity with `aggregate` 2.1.0 (#779).
