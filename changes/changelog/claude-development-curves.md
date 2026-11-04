### Breaking changes

- Rust: `prospicio_reserving::CurveShape` has the variants `Hoerl`,
  `Sherman` and `Growth(GrowthCurve)`, and `prospicio_reserving::Error` the
  variant `DevelopmentCurve`, so an exhaustive `match` on either needs an
  arm for each (#556, #729).

### Reserving

- Development curves fitted over every age (#556, #729): `DevelopmentCurve`
  fits the exponential, inverse power, Hoerl, Sherman or a log-logistic or
  Weibull growth curve to a triangle's estimated factors, keeps the
  observed factors before `keep_observed_until`, and gives the fitted
  factors as a selected pattern (`pattern()`, Python `selected()`, R
  `development_curve_selected()`) with the curve's tail (`tail()`), the
  residuals, the sum of squares, R-squared and adjusted R-squared;
  `compare` fits several shapes to the same factors. Python
  `DevelopmentCurve(shape=...)`, R `development_curve()` and
  `development_curve_compare()`. `TailCurve` (R `tail_curve()`) takes
  every shape. Matches R's `lm` and `nls` on RAA and GenIns.
