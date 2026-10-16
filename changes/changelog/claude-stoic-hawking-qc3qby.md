### Aggregate and reinsurance

- `TowerModel::diagnostics` warnings print each position at six
  significant figures (`the piece from 1.28998`, not every digit of the
  float); Python `TowerModel.diagnostics()` and R `tower_diagnostics()`
  carry the same text (#276).

### Distributions and probability

- Threshold diagnostics for peaks over threshold:
  `prospicio_prob::evt::threshold_diagnostics` gives, at each threshold,
  the mean excess with its standard error and the generalized Pareto
  fitted to the exceedances, with the shape, the modified scale
  `beta - xi * u` and their standard errors from the observed information,
  the numbers behind R `ismev`'s `mrl.plot` and `gpd.fitrange` (parity).
  `Gpd::fit_with_covariance` returns a fit with the covariance of
  `(xi, beta)`. Python `prospicio.risk.threshold_diagnostics` (a dict of
  columns) and R `threshold_diagnostics()` (a data frame) (#282).

### Pricing

- Log-log interpolation of tabulated exposure curves:
  `TabulatedCurve::with_interpolation(x, g, Interpolation::LogLog)` runs
  the curve as a power law between points (the first piece stays
  straight) and refuses a table it would make non-concave, naming the
  point where the elasticity rises. Linear stays the default. Python
  `TabulatedCurve(x, g, interpolation="log-log")`, R
  `tabulated_curve(x, g, interpolation = "log-log")` (#282).

### Models

- Cyclic smooths in `prospicio_glm::gam`: `PSpline::cyclic(column)`,
  with `period(lo, hi)`, is mgcv's `s(x, bs = "cp")`: the fit and its
  slope meet at the ends of the period, and new data outside it wraps
  around (mgcv parity). Python `Gam(glm, [{"column": "month", "bs": "cp",
  "period": (0, 12)}])`, R `gam_fit(..., bs = "cp", period = list(month =
  c(0, 12)))`. A GAM artifact with a cyclic smooth is format version 2;
  one without stays version 1 (#282).
- Monotone smooths in `prospicio_glm::gam`: `PSpline::increasing` and
  `PSpline::decreasing` are scam's `bs = "mpi"` and `"mpd"`, smoothed by
  GCV, UBRE or fixed parameters (scam parity). Python `{"column": "x",
  "bs": "mpi"}`, R `gam_fit(..., bs = "mpi")`. This completes the GAM
  scope of v0.6 (#282).

### Breaking changes

- Rust: `PSpline` gains a public `kind` field (`SmoothKind`); code that
  builds a `PSpline` in full adds it or uses `PSpline::new` (#282).

### Aggregate and reinsurance

- Multi-year contract terms: `Tower::apply_term` applies a tower to
  consecutive years of the same simulated paths (one `EventSet` per year;
  `EventSet::term_years` splits one simulation into the years of a term),
  and `apply_aggregate_term` does the same on aggregate losses; the result
  has a `year` dimension. A profit commission can carry its deficits
  forward (`Layer::profit_commission_carry_forward`) and a sliding scale
  its deficits, credits or both (`Layer::sliding_scale_carry_forward`),
  to extinction or for a number of years, used up oldest first;
  `profit_commissions_over` and `ceding_commissions_over` give the
  commissions over consecutive years. Tower documents with a carry are
  format version 2. Python `Layer.with_profit_commission_carry_forward`,
  `with_sliding_scale_carry_forward`, `Tower.apply_term`,
  `apply_aggregate_term`, `EventSet.term_years`; R
  `with_profit_commission_carry_forward()`,
  `with_sliding_scale_carry_forward()`, `apply_tower_term()`,
  `term_years()` (#301).

### Breaking changes

- Rust: `Layer` gains the public fields `profit_commission_carry` and
  `sliding_scale_carry`; code that builds it in full adds them (`None`)
  or uses the builders (#301).
