### Aggregate and reinsurance

- `TowerModel::diagnostics` warnings print each position at six
  significant figures (`the piece from 1.28998`, not every digit of the
  float); Python `TowerModel.diagnostics()` and R `tower_diagnostics()`
  carry the same text.

### Distributions and probability

- Threshold diagnostics for peaks over threshold:
  `prospicio_prob::evt::threshold_diagnostics` gives, at each threshold,
  the mean excess with its standard error and the generalized Pareto
  fitted to the exceedances, with the shape, the modified scale
  `beta - xi * u` and their standard errors from the observed information,
  the numbers behind R `ismev`'s `mrl.plot` and `gpd.fitrange` (parity).
  `Gpd::fit_with_covariance` returns a fit with the covariance of
  `(xi, beta)`. Python `prospicio.risk.threshold_diagnostics` (a dict of
  columns) and R `threshold_diagnostics()` (a data frame).

### Pricing

- Log-log interpolation of tabulated exposure curves:
  `TabulatedCurve::with_interpolation(x, g, Interpolation::LogLog)` runs
  the curve as a power law between points (the first piece stays
  straight) and refuses a table it would make non-concave, naming the
  point where the elasticity rises. Linear stays the default. Python
  `TabulatedCurve(x, g, interpolation="log-log")`, R
  `tabulated_curve(x, g, interpolation = "log-log")`.
