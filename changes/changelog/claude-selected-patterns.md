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
  error stays the triangle's. A cumulative pattern's level at its last age
  is rescaled to the factors in use and the tail
  (`ChainLadderFit::pattern_rescale`; at the oldest age, percent developed
  times the tail factor is 100%). In the one-year view a bootstrap's
  pattern must select the same factors and weights as the method's, in
  any order, kind or form of weight. Fits
  report the estimate next to the selection (`DevelopmentFit::selected`,
  `weight`, `blended_ldf()`, `blended_std_err()`, and five new columns of
  `development_table()`). Parity with chainladder-python's
  `DevelopmentConstant` and with R's `MackChainLadder` on the triangle's
  own factors (`reserving-v02.md`, decision 12). **Breaking (Rust):**
  `Development` is no longer `Copy` or `Eq`, and `ChainLadder`, `Mack`
  and the four expected-loss settings are no longer `Copy`; `Development`,
  `OdpBootstrap`, `DevelopmentFit` and `ChainLadderFit` gain public
  fields, and `Error` gains `Pattern` (#264).
- Selected development patterns in Python and R: `ChainLadder`, `Mack`,
  `ExpectedLoss`, `BornhuetterFerguson`, `Benktander`, `CapeCod`,
  `OdpBootstrap` and `MackBootstrap` (in R `chain_ladder()`, `mack()`,
  `expected_loss()`, `bornhuetter_ferguson()`, `benktander()`,
  `cape_cod()`, `odp_bootstrap()`, `mack_bootstrap()`, `odp_one_year()`
  and `mack_one_year()`) take `ldf` (a dict, or a named vector or
  data.frame in R, from age in months to value), `ldf_kind` (`"ldf"`,
  `"cdf"` or `"percent_developed"`, required with `ldf`) and `ldf_weight`
  (a number or a weight by age); the one-year views (R's
  `odp_one_year()` and `mack_one_year()`, Python's `OdpBootstrap.one_year`
  and `MackBootstrap.one_year`) also take `bootstrap_ldf`, default true:
  a bootstrap without a pattern of its own takes the method's. Chain-ladder and Mack fits gain `selected_ldf`,
  `ldf_weight`, `blended_ldf`, `estimated_std_err` and `pattern_rescale`,
  and their `std_err` is the standard error of the factor in use
  (unchanged without a pattern) (#264).
