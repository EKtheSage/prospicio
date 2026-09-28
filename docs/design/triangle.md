# Design note: Triangle

Status: **Draft for review** · Phase 0 · Depends on: `distributions.md`, `predictive-distribution.md` (period keys) · Replaces: provisional `src/triangle.rs`

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
    mask: BitVec,            // true where a value is observed
    shape: [usize; 4],
    index: Vec<Label>,       // one label per index position (possibly multi-part)
    columns: Vec<String>,
    origins: Vec<Period>,    // start of each origin period
    origin_grain: Grain,     // M, Q, S, Y
    development: Vec<Lag>,   // lag in months from origin start
    development_grain: Grain,
    valuation: Date,         // date of the latest diagonal
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

- `Period` is a start date plus a grain; `Lag` is months from origin start.
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
  off for WASM.

## Relationship to reserving methods

Methods take `&Triangle` and a column. Output per origin is keyed by the
Triangle's `Period`s, so a reserve `PredictiveDistribution` component
`{lob, origin}` joins back to the triangle without conversion.

## Migration from the sandbox

Port `development.rs`, `chain_ladder.rs` (volume and simple averages,
cumulative factors, chain ladder, Mack) onto this Triangle in
`act-reserving`. Then point `validation/tests/reserving.rs` at the new
crate and delete the root sandbox crate.

## Open questions

1. **Relationship to chainladder-python** (plan's open decision): if we
   become its optional backend (option b), axis semantics and grain
   behaviour must match it exactly, edge cases included. Parity rather than
   "similar" changes the scope of this note. **Needed before this design is
   finalized.**
2. **Development axis storage:** always ages, with valuation derived (the
   proposal), or store whichever the user supplied?
3. **Multi-part index labels:** a tuple of strings (like a pandas
   MultiIndex), or a separate small dimension table?
4. **Exclusions** (dropping link ratios by origin/age, as chainladder's
   `drop`): a second mask on the Triangle, or a parameter of the
   development estimator? Recommendation: estimator parameter, keeping the
   Triangle pure data.
