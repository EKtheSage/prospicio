### Reserving

- Selected development patterns: `SelectedPattern` on `Development`
  (`pattern`) gives benchmark factors that replace, or blend with, the
  factors estimated from the triangle, as age-to-age factors, factors to
  ultimate or percents developed (`PatternKind`, named by the user), for
  all ages or some (the rest stay estimated), with a weight per pattern or
  per age (`PatternWeight`). The chain ladder, `ExpectedLoss`,
  `BornhuetterFerguson`, `Benktander`, `CapeCod`, `Mack` (with Merz and
  Wüthrich's one-year view) and `MackBootstrap` take it through their
  `Development`, `OdpBootstrap` through its own `pattern`. A selected
  factor carries no parameter error and a blended one `(1 - w)` of the
  estimate's, so a pattern lowers the reported uncertainty; the process
  error stays the triangle's. A cumulative pattern's level is rescaled to
  the factors in use and the tail (`ChainLadderFit::pattern_rescale`; at
  the oldest age, percent developed times the tail factor is 100%). Fits
  report the estimate next to the selection (`DevelopmentFit::selected`,
  `weight`, `blended_ldf()`, `blended_std_err()`, and five new columns of
  `development_table()`). Parity with chainladder-python's
  `DevelopmentConstant` and with R's `MackChainLadder` on the triangle's
  own factors (`reserving-v02.md`, decision 12). **Breaking (Rust):**
  `Development` is no longer `Copy` or `Eq`, and `ChainLadder`, `Mack`
  and the four expected-loss settings are no longer `Copy`; `Development`,
  `OdpBootstrap`, `DevelopmentFit` and `ChainLadderFit` gain public
  fields, and `Error` gains `Pattern`.
