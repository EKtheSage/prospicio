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
    index: Vec<Label>,       // one label per index position (possibly multi-part)
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
| `slice(index=…, column=…)` | select segments and measures |
| `from_long` / `to_long` | long ↔ wide: rows of (index…, origin, development or valuation, value columns) |

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
  Triangle bindings. The core takes `Long` (borrowed slices, with an
  optional `Label` per row and ages or valuation months) and returns
  `LongTable`.

## Relationship to reserving methods

Methods take `&Triangle` and a column. Output per origin is keyed by the
Triangle's `Period`s, so a reserve `PredictiveDistribution` component
`{lob, origin}` joins back to the triangle without conversion.

v0.1 methods fit a triangle with a single index position (slice first);
fitting every segment at once, as chainladder-python broadcasts, comes later.

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

## Migration from the sandbox (done)

`development.rs` and `chain_ladder.rs` (volume and simple averages,
cumulative factors, chain ladder, Mack) are ported onto this Triangle in
`act-reserving`. `validation/tests/reserving.rs` checks every R ChainLadder
and chainladder-python reference value on RAA, GenIns and ABC (chain ladder,
simple average, and Mack with `alpha` 0, 1, 2 and both sigma rules), and the
root sandbox crate is deleted.

## Decisions

1. **Relationship to chainladder-python** (plan's open decision): still
   open. Implemented behaviour follows chainladder-python's semantics
   (axes, ages, `grain()` anchored on the valuation date), and the parity
   suite is the contract. Full backend parity (option b) would widen scope.
2. **Development axis storage:** always ages in months. `dev_to_val()`
   returns a borrowed `CalendarView` keyed by valuation month;
   `val_to_dev()` returns the triangle.
3. **Multi-part index labels:** `Label` is a tuple of strings, like a pandas
   `MultiIndex` row.
4. **Exclusions** (chainladder's `drop`): a parameter of the development
   estimator, keeping the Triangle pure data. Not implemented yet.
