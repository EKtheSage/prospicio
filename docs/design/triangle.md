# Design note: Triangle

Status: **Implemented** in `crates/prospicio-reserving` (v0.1), except the Arrow feature; reading files (decision 6) built, its proposed items accepted by the user on 2026-10-28 · Depends on: `distributions.md`, `predictive-distribution.md` (period keys) · Replaced: the provisional root `src/` sandbox

## Goal

A loss triangle whose semantics match what users know from
chainladder-python, stored as a dense, masked Rust array, with Arrow in and
Arrow out.

## Axes

Four axes, in chainladder-python's order:

| Axis | Meaning | Examples |
|---|---|---|
| index | segment | company, line of business, state |
| column | measure | paid, incurred, reported counts, premium |
| origin | origin period | accident year 2019, report quarter 2021Q3 |
| development | age or valuation | 12, 24, 36 months; or valuation dates |

## Storage

```rust
pub struct Triangle {
    values: Vec<f64>,        // dense, row-major over (index, column, origin, development)
    mask: Vec<bool>,         // true where a value is observed
    shape: [usize; 4],
    keys: Vec<String>,       // names of the key columns, e.g. ["lob", "state"]
    index: Vec<Label>,       // one label per index position, one part per key
    columns: Vec<String>,
    origins: Vec<Month>,     // start month of each origin period
    origin_grain: Grain,     // M, Q, S, Y
    development: Vec<Lag>,   // lag in months from origin start
    development_grain: Grain,
    valuation: Month,        // month of the latest diagonal
    cumulative: bool,
}
```

- **Mask, not NaN.** A missing cell can be missing for different reasons
  (not yet observed, excluded), and a zero is a valid observation, so
  observation is stored separately. The provisional ragged-row design
  could not represent a hole in the middle of a row; the mask can.
- **Dense.** Real triangles are small (tens of origins × tens of ages ×
  a few measures). A dense layout is simpler and SIMD-friendly. Large
  index axes (thousands of segments) stay dense along the three inner axes.
- **Own the array type or use `ndarray`?** An own minimal struct keeps
  WASM builds light and the API narrow. `ndarray` gives slicing for free.
  Recommendation: own struct, with `ndarray` views behind a feature flag
  if needed.

## Periods and grain

- `Period` is a start month plus a grain; `Lag` is months from origin
  start. Dates have month resolution (`Month`); a valuation month means
  its last day. Periods align to the calendar year.
- **Partial periods:** the latest origin may be only partly exposed on the
  valuation date. Its age is measured from its start, and `valuation` makes
  the diagonal exact.
- **Grain changes:** `grain(origin = Y, development = Y)` aggregates
  monthly or quarterly data. Development ages are recomputed from the new
  grain and valuation date, matching chainladder-python's `grain()`.

## Transformations

| Operation | Notes |
|---|---|
| `to_incremental()` / `to_cumulative()` | sets `cumulative`, respects the mask |
| `dev_to_val()` / `val_to_dev()` | development axis as ages ↔ as valuation dates (calendar view) |
| `latest_diagonal()` | per index × column × origin |
| `link_ratios()` | age-to-age ratios as a Triangle (masked where either side is missing) |
| `grain(origin, development)` | coarsen periods |
| `select(key = values)` / `select_columns(names)` | segments by key value (conditions ANDed), measures by name |
| `group_by(keys)` | sum segments over the other keys; `group_by([])` is the total |
| `max_age(age)` | the development ages up to `age` months (#667) |
| `from_long` / `to_long` | long ↔ wide: rows of (key columns by name, origin, development or valuation, value columns) |
| `view(keys, column)` / `summary()` | one segment × measure as a grid; one row per segment × measure |

**Views for reading (implemented, decision 5 step 4).** `view` takes one
value per key, chosen as `SegmentFits::segment` chooses (keys not named
may take any value, several matches are `AmbiguousSegment`), and a column
that may be left out when there is only one (`MultipleColumns`
otherwise). It returns a `TriangleView`: origins, ages and the cells,
`None` where unobserved. `summary()` gives a `SummaryRow` per segment ×
measure: `n_origins` (origins with an observed value), `first_origin`
and `last_origin` of those, `valuation` (the latest valuation with an
observed value), and `latest`, the sum over origins of the latest
cumulative value, so an incremental triangle sums its increments and
gives the same total as its cumulative form. `Display` (and
`to_text(max_rows, max_cols)`, `to_html`) prints the grid for one segment
and measure and the summary table otherwise, rounding numbers by the
size of a typical value (whole amounts, three decimals for link ratios)
and showing at most 20 origins and 12 ages, the first and last halves
around a `...` (`max_rows` and `max_cols`; 0 for no limit).
The printout lives in Rust so Python and R print the same text.

| | Python | R |
|---|---|---|
| one segment × measure | `tri.view(column=None, **keys)`: DataFrame, origins × ages (a dict of lists without pandas) | `as.matrix(tri, ..., column = NULL)`: matrix with origin and age dimnames |
| summary | `tri.summary()`: DataFrame (dict without pandas) | `summary(tri)`: data.frame |
| printout | `repr`, `_repr_html_`, `tri.to_string(max_rows, max_cols)` | `print`, `format(tri, max_rows, max_cols)` |

R uses base `as.matrix()` rather than a `view()` verb, which would mask
`tibble::view()` under the tidyverse; for the same reason `summary` and
`format` are base generics. Summary valuations are dates (the last day of
the month) in both bindings, as `valuation` is.

## Arrow boundary

- **In:** an Arrow table in long format (the natural shape of claims
  extracts), with the index, origin, development/valuation and measure
  columns named by the caller. Python passes pandas/Polars/pyarrow; R
  passes data.frame/arrow.
- **Out:** the same long format.
- The `arrow` crate is heavy, so the core Triangle builds from plain slices
  (`from_long(&[origin], &[development], &[value], …)`). Arrow conversion
  lives behind an `arrow` feature, enabled in the Python and R bindings and
  off for WASM. **Not built yet:** it arrives with the Python and R
  Triangle bindings. The core takes `Long` (borrowed slices: named key
  columns, ages or valuation months, named value columns) and returns
  `LongTable` with the same named key columns.
- **Key columns (implemented, decision 5 step 1).** `Long::keys` is
  `&[(name, &[value])]`, one string per row per key; `key_names()` gives
  the names and each index `Label` has one part per key, in key order.
  Without keys there is one segment with an empty `Label`, displayed as
  `Total`. Key names may not repeat or equal a value column's name. The
  bindings store key values as strings and reject missing ones; Python
  takes `keys={"lob": [...]}` in `from_long` and `keys=["lob"]` in
  `from_frame`, R takes `keys = c("lob")` in `triangle()`. Differences by
  idiom: values become strings with each language's own conversion
  (Python `str(7.0)` is `"7.0"`, R `as.character(7)` is `"7"`); without
  keys Python's `index` is `["Total"]` while R's `@index` is a data.frame
  with one row and no columns.
- **Select and group by name (implemented, decision 5 step 2).**
  `select(&[(key, &[values])])` keeps the segments whose value of each
  named key is one of its values (conditions ANDed, segments keep their
  sorted order, no conditions keep everything); `select_columns(&[names])`
  keeps measures in the order given. `group_by(&[keys])` sums the segments
  that share the named keys' values and drops the other keys: the result's
  keys are in the order given and its segments sorted; `group_by(&[])` is
  one total segment without keys. Cumulative values are summed cell by
  cell and a cell is observed if any member is; an incremental triangle is
  summed as cumulative values and returned incremental, so grouping then
  fitting equals fitting the triangle built from the summed rows. Unknown
  keys, values or columns, repeats, empty value lists and a selection that
  matches nothing are errors. Selection by position (`slice`) is gone.
  Bindings, same semantics:
  - Python: `tri.select(columns=None, **keys)` (per key one value, or any
    iterable of values that is not a string, such as a list, NumPy array
    or pandas Series; compared as `str()`), `tri.group_by(keys)`.
  - R: `subset(tri, key = values, columns = NULL)` and
    `aggregate(tri, keep = keys)` (default `character()`, the total). They
    are methods on base generics rather than `select()`, `filter()` or
    `group_by()`, which would mask dplyr's verbs of other meanings, and
    `aggregate(keep =)` is what the package already uses to sum a
    `predictive_distribution` over its keys. A key named `columns` (or
    `x` in R) cannot be selected by keyword.
- **Keep the ages up to an age (implemented, #667,
  `reserving-bayes.md` decision 18).** `max_age(age)` (a `Lag`, months)
  keeps every development age at or before `age`, in every segment,
  measure and origin, as chainladder-python's
  `tri[tri.development <= age]`; the later ages are left out, not folded
  into a tail. `age` need not be one of the triangle's ages: any value at
  or past the first age keeps the ages up to it, and one at or past the
  last age returns the triangle unchanged. An age before the first, which
  keeps nothing, is `Error::AgeBeforeFirst`, and a cut that leaves no
  observed cell is `Error::Empty`. The origins (an origin observed only at
  dropped ages keeps an empty row, which the fits trim when it is a
  segment's first or last origin and refuse between observed origins),
  the measures (a premium column among
  them, so premium recorded only at a dropped age loses those cells), the
  grains and the basis are kept, and the valuation becomes the latest
  valuation of a kept observed cell. On a cumulative triangle where every
  origin keeps an observed cell the result equals `from_long` on the rows
  of `to_long` at the kept ages (an origin left without cells, such as the
  oldest origins of a trapezoid, is not in those rows, so the rebuild has
  fewer origins and the same fits); on an
  incremental one that rebuild does not drop the ages, since `from_long`
  runs an incremental table's ages to the oldest origin's age at the
  valuation and fills the missing rows with zero increments, so
  `max_age` cuts the development axis itself. Its use is the remedy for a
  falling last age in the Bayesian chain ladder models
  (`reserving-bayes.md` decision 26). Bindings: Python `tri.max_age(age)`,
  R `max_age(tri, age)` (no base or dplyr generic has the name).

## Relationship to reserving methods

Methods take `&Triangle` and a column. Output per origin is keyed by the
Triangle's `Period`s, so a reserve `PredictiveDistribution` component
`{lob, origin}` joins back to the triangle without conversion.

**Every segment at once (implemented, decision 5 step 3).** `fit` still
needs a single-segment triangle. `fit_segments` on `ChainLadder`, `Mack`
and `OdpBootstrap` fits one column in every segment, each on its own (the
same numbers as selecting the segment and calling `fit`), and a failure
names its segment (`Error::InSegment`) when the triangle has keys.
Each segment fits on the origins and ages it observes, from its first to
its last: a line that starts later, or has fewer ages than the triangle,
fits on its own range (as does a column with empty trailing origins or
ages). An empty origin inside that range is still an error. Long tables
have one row per origin the segment covers.

- `ChainLadder` and `Mack` return `SegmentFits<T>`: `key_names`, `labels`
  and `fits` in index order, `get(&Label)`, and `segment(&[(key, value)])`
  for the one segment matching those values (keys not named may take any
  value; several matches are `AmbiguousSegment`). Its long tables
  (`FitTable`: key columns, then `origin` or `age`, then values) are
  `to_long()` (one row per segment × origin: `latest`, `ultimate`,
  `reserve`, and for Mack `process_risk`, `parameter_risk`,
  `standard_error`), `totals()` (one row per segment, the same for the
  segment total) and `development_table()` (one row per segment × age:
  `ldf`, `cdf`, `sigma`, `std_err`; NaN past the oldest link).
- `OdpBootstrap::fit_segments` returns `OdpBootstrapFits`: per segment the
  chain ladder, fitted values, residuals and scale (`SegmentFits<
  OdpBootstrapSegment>`), and one joint `PredictiveDistribution` whose
  dimensions are the key names and `origin`, with components segment-major
  like the rows of `to_long()`. Segments are independent; simulation `i`
  uses stream `i` for every segment in index order, so the draws are
  reproducible and independent of the thread count, but a segment's draws
  differ from bootstrapping it alone. Its tables add the `mean` and
  `std_dev` of the bootstrapped reserve per row, and `totals()` the
  `scale`; `segment()` keeps that segment's part of the joint draws with
  the same dimensions.

The bindings always fit every segment. Per-origin fields follow the long
rows (segment by segment), so a single-segment fit reads as before;
per-age fields (`ldf`, `cdf`, `sigma`, `std_err`), Mack's total standard
errors and CV, and the bootstrap's `scale`, `fitted` and `residuals` need
one segment and otherwise raise an error pointing to the frames or
`segment()`. `total_ultimate` and `total_reserve` sum over segments.

| | Python | R |
|---|---|---|
| Per-origin table | `fit.to_frame()` | `as.data.frame(fit)` |
| Per-segment totals | `fit.totals_frame()` | `totals_frame(fit)` |
| Development factors | `fit.development_frame()` | `development_frame(fit)` |
| One segment | `fit.segment(lob="Auto")` | `segment(fit, lob = "Auto")` |
| Keys, segment labels | `fit.keys`, `fit.index` | `fit@keys`, `fit@index` |

Differences by idiom: R's per-origin vectors are named by origin, or by
`"segment / origin"` with several segments; R turns NaN into `NA` in the
frames; the frames' `origin` is the period label in both. With keys, the
bootstrap's `reserves` has the key dimensions even for one segment (a
keyless triangle keeps the single `origin` dimension).

Development factors (`Development`) follow Mack's weighted regression, as
R ChainLadder and chainladder-python do:

- `Average::{Volume, Simple, Regression}` is Mack's `alpha` = 1, 0, 2.
- A sigma that cannot be estimated (one link ratio at that age) is filled
  by `SigmaInterpolation::LogLinear` (default in both references) or
  `SigmaInterpolation::Mack` (Mack 1993). Mack's rule fills gaps in order,
  as R does.
- Where the two references disagree, we follow R ChainLadder, except for the
  p-value fallback:
  - A sigma of exactly zero stays zero and is left out of the log-linear
    fit (R). chainladder-python keeps it in the fit as `1e-320`.
  - R falls back to Mack's rule when the log-linear slope's p-value is
    above 0.05; chainladder-python never does, and neither do we.
- An origin whose value is zero at an age informs that age's factor (its
  weight is `C^(alpha-1)`) but not its sigma, where its weight would be
  infinite. This is our choice, not a reference behaviour: R ChainLadder
  fails on such a triangle.
- A sigma that cannot be filled either is NaN: the chain ladder does not
  need it, and Mack rejects it.

`Mack` uses R's `mse.method = "Mack"` recursions: per-origin process and
parameter risk, and a total parameter risk that carries the covariance
between origins through the shared factors.

`OdpBootstrap` follows R ChainLadder's `BootChainLadder` (England and
Verrall 2002), not chainladder-python's `BootstrapODPSample`, which uses
hat-matrix residuals, drops zero residuals and re-centres the rest to mean
zero (its Gamma process error is added later, when a downstream estimator
projects the resampled triangles):

- Residuals are unscaled Pearson residuals on every observed incremental
  cell, adjusted by `sqrt(n / (n - p))` with `p = origins + ages - 1`, zero
  corner residuals included; the scale is `sum(r^2) / (n - p)`.
- Process error is Gamma (signed like the mean, variance `phi |m|`), or none
  for parameter error only. R's over-dispersed Poisson (a negative
  binomial draw) is not offered yet: our inverse-transform count sampler is
  too slow for incremental means in the hundreds of thousands.
- Unlike R, triangles need not be square, but every origin must be observed
  from its first age to its latest.
- The result is a `PredictiveDistribution` with dimension `origin`, keyed by
  the triangle's `Period`s. Simulation `i` uses stream `i`, so results are
  identical for any thread count.
- Parity: the scale and residuals match R exactly; reserve means, standard
  deviations and total quantiles match R within four Monte Carlo standard
  errors (`validation/scripts/reserving_bootstrap_r.R`).

### ODP GLM

`OdpGlm` is the same over-dispersed Poisson model fitted as a GLM
(`docs/design/models.md`, "Fitting a triangle with several models"):
`ln E[X_od] = c + a_o + b_d`, `Var X_od = φ E[X_od]`, by
`prospicio_glm::Glm::over_dispersed_poisson()` (Pearson's `φ`) on the observed
rows of a `TriangleFrame`.

- Coding: `Terms` with an intercept and `origin` and `development`
  factors, levels learned on every cell so the future rows code with the
  training columns. The references are the first origin and the first age,
  as in R's `factor()`; the other levels are in string order
  (`development[108]` before `development[24]`).
- `OdpGlmFit` holds the `GlmFit` (coefficients, standard errors, `φ`), the
  `Coding`, the future cells `(Period, age)` with their fitted means, and
  reserves by origin (0 for a fully developed origin).
- `predict_distribution(n_sims, seed)` re-keys the GLM's joint draws over
  the future cells to dimensions `["origin", "development"]`;
  `aggregate(["origin"])` gives reserves by origin. Each draw takes the
  coefficients from their normal approximation (parameter uncertainty) and
  each cell as `φ · Poisson(μ / φ)` (process uncertainty). The parameter
  draws are mean-preserving by default (`prospicio_glm::ParameterDraws`, #114):
  each cell's linear predictor is shifted by `-v / 2`, `v = xᵀ Σ x`, so the
  draws average the Chain Ladder reserve. `predict_distribution_with` takes
  `Normal` (unshifted: each cell's mean is `predictive_means()`, +7.2% on
  GenIns, +0.4% on ABC) or `Fixed` (process uncertainty only).
- The fit fails, naming the cell or level, on a hole (a past cell with no
  increment) and on an origin or age with no observed increment (a level
  only in future cells). Negative increments are fitted by the
  quasi-likelihood, which needs only `V(μ) = μ` and `μ > 0` (#114); R's
  `quasipoisson` refuses them. A level whose only increments are negative
  has no positive mean, and the GLM fails.
- Parity (`validation/tests/reserving.rs`): on RAA, GenIns and ABC the
  future cells summed by origin equal the Chain Ladder reserves and `φ`
  equals the bootstrap's scale (relative 1e-8; Renshaw and Verrall 1998);
  on GenIns and ABC the coefficients, standard errors, `φ` and reserves
  match R's `glm(inc ~ factor(origin) + factor(dev), family = quasipoisson)`
  to relative 1e-9 (`validation/scripts/reserving_glm_r.R`). RAA has no R
  reference: its negative increment (1982 at 84 months, -103) makes R
  refuse it.
- `prediction_error()` gives the reserves' analytic prediction error by the
  delta method (#333): for a set of future cells, the process variance
  `φ Σ μ` and the parameter variance `gᵀ Σ g` with `g = Σ μ x`
  (`GlmFit::sum_prediction_error`), by origin and in total. The total is
  not the root sum of the origins' squares: the origins share
  coefficients. On Taylor and McGuire's NJ Manufacturers triangle (CAS
  Monograph 3, Table 5-4) every forecast and RMSEP reproduces to the unit,
  the total 373,346 with RMSEP 14,076 (process 6,539, parameter 12,465),
  and 10,275 for the years combined as if independent
  (`validation/tests/reserving.rs`).
- Bindings (#333, first PR): Python `OdpGlm().fit(triangle, column)`
  returning `OdpGlmFit` (its `glm`, `future_design`, `prediction_error()`,
  `predict_distribution(n_sims, seed, parameters)`), and R
  `odp_glm(triangle)` returning an `odp_glm_fit` with
  `prediction_error` and a `predict_distribution()` method. Single segment
  only, like `OdpGlm`.
- `residuals()` gives each observed cell's actual, fitted, actual-to-fitted
  ratio, Pearson and deviance residuals, leverage and standardized
  residuals (#333, second PR), with its origin, age and calendar period. A
  cell with leverage 1 (the only observation of its origin or age) is
  fitted exactly and its standardized residual is NaN (`NA` in R).
  `OdpResiduals::summary` groups the standardized deviance residuals by
  origin, development or calendar period (count, mean, sd with divisor
  `n - 1`). On the NJ Manufacturers triangle every percentage of Taylor
  and McGuire's Figure 6-22 heat map reproduces, and the spread by
  development year and the mean by calendar year match their Chapter 6
  figures to two decimals (`validation/tests/reserving.rs`). Python
  `OdpGlmFit.residuals()` and `residual_summary(by)`, R `fit@residuals`
  and `residual_summary(fit, by)`.
- `fit_design(triangle, column, design)` fits another design (#333, third
  PR): a `Design` with one row per row of the triangle's `TriangleFrame`, in
  its order (every origin × development cell, origin-major, a cell after a
  hole included), its offset and weights used too. A precomputed design,
  not a new kind of term, because the R bindings build designs with R's
  own `model.matrix()`, and a design on the cells covers what `Terms`
  cannot say (a single age's indicator, indicators multiplied together):
  Rust users code one with `Terms` on `cells.frame()` or build the columns
  from the frame's indices. The hole check stays; the check that every
  origin and age has an observed increment is the cross-classified
  model's, so a design instead fails on a column that is zero on every
  observed cell, and on collinearity in the GLM. `OdpGlmFit::coding` is
  `None` for such a fit (breaking: it was a `Coding`); `glm.names()`
  names the columns. On the NJ Manufacturers triangle Taylor and McGuire's
  models 7-2, 7-10 and 7-14 reproduce Tables 7-1, 7-3 and 7-5 to the
  printed digits with `φ` 102.6, 107.3 and 53.9, model 7-10's heat map is
  Figure 7-5, and their delta-method forecasts and RMSEPs are the guide's
  (372,532 and 13,030; 373,006 and 13,215; 370,493 and 11,021;
  `validation/tests/reserving.rs`). Python `OdpGlm.cells(triangle,
  column)` (a dict of the frame's columns with `observed` and `future`)
  and `OdpGlm().fit(triangle, column, design=Design(...))`; R
  `odp_glm_cells()` and `odp_glm(triangle, formula = ~ ...)`, the formula
  run through `model.matrix()` on the cells.
- `OdpGlm::family` swaps the error (#333, fourth PR): `Family::Poisson`
  (the default, the ODP), `Family::Gamma` (`Var X_od = φ μ²`, every cell's
  relative error weighed alike, Taylor and McGuire section 6.5) or
  `Family::Tweedie { power }` with `1 < p < 2` (`φ μ^p`), each with the log
  link and Pearson's `φ`, on the cross-classified design or any
  `fit_design`. A field on `OdpGlm` rather than a free `Glm`, because the
  link must stay the log for the mean-preserving draws and the reserves'
  delta method, and because another family (Gaussian, binomial) is not a
  reserving error: it fails at fit time. The reserves are no longer the
  Chain Ladder's, nor monotone in the power (on the NJ Manufacturers
  triangle the ODP's total is 373,346, the gamma's 373,466 and the
  Tweedie's at `p = 1.5` 373,129). The gamma needs every observed increment
  positive and the Tweedie every one at least 0; another fails naming its
  cell. The prediction error's process variance, the residuals and the
  draws follow the family (`φ Σ V(μ)`, `(y − μ) / √V(μ)` and its unit
  deviance, gamma or Tweedie process draws). Parity
  (`validation/tests/reserving.rs`): on GenIns and ABC the coefficients,
  standard errors, `φ` and reserves match R's `glm(family = Gamma(link =
  "log"))` and statmod's `tweedie(var.power = 1.5, link.power = 0)` to
  relative 1e-9, both converged to 1e-14 (R's deviance test stops IRLS on
  the non-canonical log link about 1e-8 short of the coefficients, so the
  reference script refits from them until they settle); on the NJ
  Manufacturers triangle the gamma fit is the guide's section 9. Python
  `OdpGlm(family="gamma")` and `OdpGlm(family="tweedie", power=1.5)`, R
  `odp_glm(triangle, family = "gamma")` and `family = "tweedie", power =
  1.5`.

## Calendar-diagonal backtest

`diagonal_backtest` (`prospicio-reserving/src/backtest.rs`) scores any model of
the cells of a `TriangleFrame` the way reserving uses it: for each of the
latest `k` calendar diagonals, refit on the earlier diagonals and forecast
the held-out one (`docs/design/models.md`, "Fitting a triangle with several
models").

- A model takes part through the `TriangleModel` trait: `forecast(cells,
  train, test, n_sims, seed)` returns a `CellForecast` (means, a joint
  `PredictiveDistribution` over the test rows, and optional pointwise log
  predictive densities). `GlmCandidate { name, terms, glm }` wraps any
  `prospicio-glm` GLM; the ODP model is the quasi-Poisson GLM with intercept,
  origin and development factors.
- A held-out cell whose origin or development level has no training row
  (the newest origin, and the oldest origin at an age not seen before)
  cannot be forecast by a model with origin and development effects. It
  is left out for every model, so all models score the same cells, and
  `Backtest::excluded` counts it: two per diagonal on a full triangle.
  "Training row" means one every model fits on (`TriangleModel::fit_rows`,
  all of them by default): an age whose only training cell is a response
  the model does not accept (a negative increment under a Poisson with
  fixed dispersion) is unseen too.
- Scores per model and diagonal: mean cell CRPS, coverage of the central
  `interval` of each cell's draws, actual vs expected on the diagonal
  total (`Σy / Σμ`), the CRPS of the diagonal total from the joint draws
  summed per simulation, the mean held-out log density of the scored rows
  (`mean_log_density`) and the share of the simulated totals at or below
  the actual total (`total_percentile`). A forecast also carries the
  sampler diagnostics of its fit (`CellForecast::diagnostics`, `None` for
  the GLM); `Backtest::diagnostics(model)` and `Backtest::warnings()` give
  them per split, and a fit with warnings is still scored. Two models
  with the same name are an error before any fit; `NamedModel` renames
  one. One seed serves every model and diagonal,
  so the models share their random numbers, and results are deterministic.
- `GlmCandidate` fits the training cells its GLM accepts
  (`Glm::accepts`): the quasi-Poisson takes negative increments (#114),
  a Poisson with fixed dispersion does not. Its predictive draws are
  mean-preserving under a log or identity link, so each cell's simulated
  mean is its fitted mean.
- Its log densities are log predictive densities with parameter
  uncertainty (the response density averaged over coefficient draws);
  plug-in densities treat the young origins' factors as known and
  penalize the ODP model where it is least certain. The response density
  is `Family::log_density`, which scores the over-dispersed Poisson by a
  normalized density in `y` (#114).
- Parity: on a held-out diagonal the ODP candidate's means equal Chain
  Ladder's one-period forecasts from the triangle valued before it
  (GenIns, three diagonals, to `1e-7`).

### Feeding stacking

`Backtest::log_densities()` returns, per model, the held-out log density
of every scored cell, diagonals in order, aligned across models (`None`
for a model that gives none). The models that give one are the input of
`prospicio_models::stack::stacking_weights` (and `pseudo_bma_weights`), the
cross-validated counterpart of PSIS-LOO pointwise values (#95, #102); the
weights then blend the models' predictive distributions of the future
cells with `PredictiveDistribution::blend`. A held-out cell that a model
gives zero density (a negative increment under the Poisson) makes its
log density `-∞`, which stacking rejects: choose diagonals without one.

Findings on the reference triangles, ODP against development factors
only: on ABC the ODP model wins on cell and total CRPS and stacking
favours it (about 0.58 to 0.42, three diagonals), though its 90%
intervals cover only about half the cells. On RAA's latest two diagonals
development alone scores better (cell CRPS about 1000 against 1400): the
young origins' factors rest on one or two noisy cells.

## Migration from the sandbox (done)

`development.rs` and `chain_ladder.rs` (volume and simple averages,
cumulative factors, chain ladder, Mack) are ported onto this Triangle in
`prospicio-reserving`. `validation/tests/reserving.rs` checks every R ChainLadder
and chainladder-python reference value on RAA, GenIns and ABC (chain ladder,
simple average, and Mack with `alpha` 0, 1, 2 and both sigma rules), and the
root sandbox crate is deleted.

## Decisions

1. **Relationship to chainladder-python:** decided, standalone. This is
   its own Rust implementation of actuarial models, exposed to Python and R
   through our bindings; it will not be a chainladder-python backend.
   chainladder-python and R ChainLadder are parity references: where they
   agree we match them, and where they differ or are silent we choose and
   record the choice here. Familiar axis semantics are kept for users, not
   as a compatibility contract.
2. **Development axis storage:** always ages in months. `dev_to_val()`
   returns a borrowed `CalendarView` keyed by valuation month;
   `val_to_dev()` returns the triangle.
3. **Multi-part index labels:** superseded by decision 5. A `Label` is a
   tuple of strings, one per named key column.
4. **Exclusions** (chainladder's `drop`): a parameter of the development
   estimator, keeping the Triangle pure data. Not implemented yet.
5. **Several measures, lines and other identifiers** (decided 2026-10-05).
   The four-axis storage stays as the engine. What users see is a long
   table with named key columns, plus 2-D views on demand. Chainladder's
   cube is hard to read because of how it is presented: its keys have no
   names, you select by position over four axes, and it prints well only
   once you have narrowed it to one segment and one measure.
   - **Named key columns.** A triangle knows its keys by name, for example
     `["lob", "coverage", "company"]`. Each index position's `Label` holds
     one value per key. With no keys there is one segment and an empty
     label.
   - **Measures are columns.** Paid, incurred and counts are columns of one
     triangle, not separate triangles. They share keys, origins and ages,
     so methods that combine them (Munich chain ladder, paid–incurred) get
     aligned data.
   - **Long table in, long table out.** `from_long` takes named key columns,
     and `to_long` returns them. Method results are long tables with one
     row per key × origin and quantities (ultimate, reserve, standard
     error) as columns.
   - **Select and group by name.** `select(key = values)` filters segments
     and `group_by(keys)` sums the other keys away. Neither works by
     position.
   - **Every segment at once.** Methods fit each segment of a column and
     return one long table. Stochastic methods return one joint
     `PredictiveDistribution` with the key names and `origin` as
     dimensions, so `aggregate(["lob"])` keeps the dependence between
     segments.
   - **Views for reading.** The bindings give a plain origin × development
     table for one segment and measure (`view`). A triangle with many
     segments prints a summary: one row per segment and measure, with its
     origins, valuation and latest diagonal total.
   - **Build order** (one PR each):
     1. named keys in `from_long`, `to_long` and the bindings;
     2. `select` and `group_by`;
     3. fitting every segment, with long results;
     4. `view` and the summary printout.

     `TriangleFrame` gains the key columns as features, so one model can
     share information across segments.
6. **Triangles from files and the Schedule P loader** (#552, branch
   `claude/triangle-io`; items marked *proposed* were accepted by the
   user as built on 2026-10-28).
   - **Behind a feature.** `prospicio-reserving`'s `io` feature, off by
     default, turns on the `io` module and the optional dependencies
     `csv` 1.4 and `calamine` 0.35 (both pure Rust), so the default build
     reads no files and the WASM check (`-p prospicio-reserving -p
     prospicio`) compiles neither. The umbrella crate's `io` (and `full`)
     forwards it; the Python and R bindings and `prospicio-validation`
     turn it on, which with feature unification puts the module's unit
     tests and clippy under the workspace `cargo test` and `cargo clippy`
     with no change to `ci.yml`. `Triangle::from_csv(reader, &layout)`
     takes any `std::io::Read`, so a caller with its own bytes needs no
     file access; `read_csv(path, ...)` and `read_excel(path, sheet, ...)`
     open the file. calamine reads `.xlsx`, `.xlsm`, `.xlsb`, `.xls` and
     `.ods`. *Proposed:* calamine 0.35 rather than 0.37, because 0.36 and
     0.37 need rust-version 1.88 against the workspace's 1.85. It pins
     quick-xml 0.39 and zip 7, which carry RustSec advisories
     (RUSTSEC-2026-0194 and 0195, denial of service on a crafted
     workbook; RUSTSEC-2026-0367 to 0369, extraction to disk, which
     calamine does not call). By the user's decision of 2026-10-28 they
     are ignored in `deny.toml`, with those reasons, until the workspace
     moves to Rust 1.88 and calamine 0.37. `encoding_rs` (via calamine) is `(Apache-2.0 OR MIT) AND
     BSD-3-Clause` and has a licence exception in `deny.toml`.
   - **Layouts.** `FileLayout::long(origin, development, columns)` (one
     row per cell; no columns means every column that is not the origin,
     the development or a key) and `FileLayout::wide(origin, column)`
     (one row per origin, every other column an age of the one measure,
     named by its age in months unless `ages` gives them), with `keys`,
     `grains`, `cumulative`, `development_is_valuation` (long only) and a
     CSV `comment` character as builder methods. The name is `FileLayout`
     rather than the issue's `CsvLayout`, since Excel uses it too.
   - **How cells are written** (*proposed*). An origin is a year (`1988`,
     or `1988.0` as a spreadsheet writes it), a date (`2021-07-15`,
     `2021-07`, a time after the date is ignored, or an Excel date cell)
     or a period label as the triangle prints it (`2021Q3`, `2021H2`). An
     age is a whole number of months. A valuation is a year meaning its
     December (as R's `triangle()` and Schedule P's `DevelopmentYear`), a
     date, or a period label meaning its last month. A value is a finite
     number; an empty cell, `NA`, `NaN` or `nan` is not observed; anything
     else is an error. A key is text, a number written as it reads (`353`,
     not `353.0`). Fields are trimmed, rows whose cells are all empty
     (a line of only spaces too) are skipped, columns with no name and no
     values (a header ending in a delimiter) are dropped while a column
     with values and no name is an error, and a UTF-8 byte order mark is
     dropped. CSV lines end in `\n`, `\r\n` or a bare `\r`. A sheet's used range
     starts with its header row (a title above the table is not skipped).
   - **Duplicates** (*proposed*). A long file sums rows with the same
     (keys, origin, development), as `from_long` does for a claims extract;
     a wide file with two rows in one (keys, origin period) is an error naming
     both rows (`2021-01` and `2021-02` are one origin under annual grains),
     and so are two columns at one age (`12` and `12.0`, or repeated given
     ages), naming both columns.
   - **Errors** are `Error::File(message)`, naming the file line and
     column for CSV (`line 3, column "paid": "1O0" is not a number`) or the
     cell and sheet for Excel (`cell C3 (column "paid") of sheet "bad"`),
     and listing the columns or sheets when one is missing; an age off the
     development grid names its cell, and a file with no observed value
     says so (with the path, from `read_csv` and `read_excel`). **Breaking** in Rust: a new variant
     of the exhaustive `Error`.
   - **Python**: `Triangle.read_csv(path, origin, development=None,
     columns=None, keys=None, ages=None, ..., comment=None)` and
     `Triangle.read_excel(path, ..., sheet=None)`, wide when `development`
     is `None`, the other arguments as `from_frame`'s; pandas and openpyxl
     are not needed.
   - **Schedule P in Python** (*proposed*): `schedule_p(line, company=None,
     columns=None, *, accident_years=(1988, 1997), lower=False)` reads the
     `cas-schedule-p` package's `mart_reserving_model_training` with
     pyarrow, as `validation/scripts/calibration_panel.py` does, and raises
     `ImportError` with an install hint when either is missing. Measures:
     `paid` (`cum_paid_loss`), `incurred` (`incurred_loss`, bulk and IBNR
     included), `case_incurred` (`incurred_loss - bulk_loss`, Meyers'
     incurred), `premium` (`earned_prem_net`), `premium_direct`
     (`earned_prem_direct`), all five by default: the panel's columns
     rather than the issue's `column="paid"`. Ages are `development_age *
     12`. One segment per company, keyed `company` (the code as text).
     Lines by the panel's codes (`ca`, `pa`, `wc`, `ol`, and `pl`, `mm`),
     the CAS file names (`comauto`, ...) or the mart's names. `lower=True`
     keeps the cells valued after the last accident year (the held-out
     lower triangle), so a complete company is its 10 x 10 square. The
     package is not a dependency of the Python package or its dev group,
     so CI skips the test that reads it; it ran locally.
   - **Schedule P in R** (*proposed*): `schedule_p_triangle(path, company =
     NULL, columns = NULL, accident_years = c(1988, 1997), lower = FALSE)`
     reads a file of the CAS loss reserve database in its own layout
     (`comauto_pos.csv` and the other five, downloaded by the user from
     the CAS web site; nothing is downloaded), finding each loss and
     premium column by its prefix since the suffix differs by line (`_C`
     for commercial auto). R has no package that bundles the mart, and
     reading its Parquet would need `arrow`. The measures and options are
     Python's. `read_triangle(path, origin, development = NULL, ...,
     sheet = NULL, format = c("auto", "csv", "excel"), comment = NULL)`
     calls the Rust readers, so Excel needs no R package either.
   - **Tests.** The module's unit tests (both layouts round-trip RAA,
     a file with a byte order mark and a comment, keys, valuations,
     periods, every error); `validation/tests/reserving_io.rs` (RAA from
     `validation/data/raa.csv` equals the built-in RAA, and from both data
     sheets of `validation/data/raa.xlsx`, written by
     `validation/scripts/raa_xlsx.py`); `python/tests/test_reserving_io.py`
     and `R/prospicio/tests/test-reserving-io.R` (the same files, the
     errors, and Schedule P group 353 equal to
     `validation/data/clrd_comauto_353.csv`, in R from
     `validation/data/schedule_p_comauto_sample.csv`, groups 353 and 8672
     in the CAS layout written by `validation/scripts/schedule_p_sample.py`).
