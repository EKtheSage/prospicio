### Reserving

- A development pattern per posterior draw from Meyers' CRC, CSR and CAY, aligned
  with the reserve draws, for the cash-flow engine's paid patterns (#600, #678):
  `MeyersFit::patterns` and `MeyersFits::patterns` (`DevelopmentPatterns`, with
  `pattern_draws` and `monthly` as for the other fits; one fit or segments,
  carried through a rank correlation). A draw's pattern is the ratio of the
  expected cumulative values at each age to the one at the oldest age,
  `exp(beta_d + sigma_d^2 / 2)` over `exp(sigma_m^2 / 2)`; in CSR the pattern
  of the origin after the last, and in CAY CRC's formula on CAY's draw (the
  expected values of an origin whose previous origin has no residual, not
  CAY's own, which also depend on `rho`). The reserves
  and ultimates are unchanged. Python: `MeyersFit.patterns`,
  `MeyersFits.patterns`. R: `pattern_draws()` and `monthly_pattern_draws()`
  take `meyers_fit` and `meyers_fits`. **Breaking** in Rust: `MeyersFit` and
  `MeyersFits` gain the public `patterns` field, so a struct literal that
  leaves it out no longer builds.
