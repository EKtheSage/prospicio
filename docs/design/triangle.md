# Design note: Triangle

Status: **Implemented** in `crates/act-reserving` (v0.1), except the Arrow feature · Depends on: `distributions.md`, `predictive-distribution.md` (period keys) · Replaced: the provisional root `src/` sandbox

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
| `from_long` / `to_long` | long ↔ wide: rows of (key columns by name, origin, development or valuation, value columns) |

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

## Relationship to reserving methods

Methods take `&Triangle` and a column. Output per origin is keyed by the
Triangle's `Period`s, so a reserve `PredictiveDistribution` component
`{lob, origin}` joins back to the triangle without conversion.

**Every segment at once (implemented, decision 5 step 3).** `fit` still
needs a single-segment triangle. `fit_segments` on `ChainLadder`, `Mack`
and `OdpBootstrap` fits one column in every segment, each on its own (the
same numbers as selecting the segment and calling `fit`), and a failure
names its segment (`Error::InSegment`) when the triangle has keys.

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
`act_glm::Glm::over_dispersed_poisson()` (Pearson's `φ`) on the observed
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
  draws are mean-preserving by default (`act_glm::ParameterDraws`, #114):
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

## Calendar-diagonal backtest

`diagonal_backtest` (`act-reserving/src/backtest.rs`) scores any model of
the cells of a `TriangleFrame` the way reserving uses it: for each of the
latest `k` calendar diagonals, refit on the earlier diagonals and forecast
the held-out one (`docs/design/models.md`, "Fitting a triangle with several
models").

- A model takes part through the `TriangleModel` trait: `forecast(cells,
  train, test, n_sims, seed)` returns a `CellForecast` (means, a joint
  `PredictiveDistribution` over the test rows, and optional pointwise log
  predictive densities). `GlmCandidate { name, terms, glm }` wraps any
  `act-glm` GLM; the ODP model is the quasi-Poisson GLM with intercept,
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
  total (`Σy / Σμ`), and the CRPS of the diagonal total from the joint
  draws summed per simulation. One seed serves every model and diagonal,
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
`act_models::stack::stacking_weights` (and `pseudo_bma_weights`), the
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
`act-reserving`. `validation/tests/reserving.rs` checks every R ChainLadder
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
