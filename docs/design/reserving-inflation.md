# Design note: calendar-year and superimposed inflation in reserving (indexation, Taylor's separation method, an age-period-cohort GLM)

Status: **Accepted** (2026-11-24; issue #979, milestone 0.3.0). The user
accepted all six recommendations of the open decisions on 2026-11-24,
recorded as decisions 11 to 16.

* **PR 1 is built** (branch `claude/inflation-index`, 2026-11-24):
  `CalendarIndex`, `Indexation` (`factor`, `restate`, `inflate`,
  `reinflate`), `FutureCells` and `Reinflated`, in Rust, with Taylor's
  re-inflation of Example 1 reproduced.
* **PR 2 is built** (branch `claude/inflation-index-bindings`,
  2026-11-24): `CalendarIndex`, `Indexation`, `FutureCells` and
  `Reinflated` in Python and R. Python builds `FutureCells` from explicit
  columns, `from_chain_ladder(fit, segment=None)` or `from_odp_glm(fit)`;
  R's `future_cells()` takes a chain-ladder fit, an ODP GLM fit or a data
  frame (`segment` chooses one segment of a fit over several).
  `FutureCells::from_odp_glm` returns a `Result` (PR 1 review).
* **PR 3 is built** (branch `claude/separation-arithmetic`, 2026-11-24):
  `Separation` with the arithmetic method, a normaliser,
  `FutureInflation` (`Rate`, `Index`, `Trend`), `fit_segments` and
  `SeparationFit`, in Rust, at parity with statsmodels' Poisson GLM
  (1e-9) and Taylor's Example 1.
* **PR 4 is built** (branch `claude/separation-geometric`, 2026-11-24):
  `SeparationKind::Geometric(Retransformation)`, least squares on the
  logs with proportions summing to 1, geometric means by default or
  lognormal means (decisions 12 and 13; at least 3 origins),
  `SeparationFit::log_variance`, in Rust, at parity with statsmodels'
  OLS (1e-9).

Lane: Reserving (`crates/prospicio-reserving/`; the
bindings in new module files of `prospicio-python` and `prospicio-r`) ·
Depends on: `triangle.md` (calendar periods, decision 2),
`reserving-v02.md` (the chain ladder, `OdpGlm` and its designs, the
calendar-year test `calendar_year_effect`)

## Goal

`prospicio-reserving` tests for calendar-year effects (Mack's
`calendar_year_effect`, `OdpResiduals::summary` by calendar period) and
Meyers' CAY carries a calendar trend inside a Bayesian model, but no
method separates a triangle's development from the inflation of the
calendar periods its payments fall in. Three pieces are missing (#979):

1. **Indexation.** Restate a triangle's increments at one price level
   with an index by calendar period (CPI, wages, a claims-inflation
   index), develop the restated triangle with any method, and re-inflate
   the projected future increments at a future index.
2. **Taylor's separation method** (Taylor 1977), arithmetic and
   geometric: development proportions `r_j` and calendar indices
   `lambda_h`, with `E[s_ij] = r_j lambda_{i+j}`, and a projection under
   chosen future indices. On a triangle already deflated by an external
   index the `lambda_h` measure superimposed inflation.
3. **An age-period-cohort (APC) reserving GLM**: accident, development
   and calendar effects together, identified as Kuang, Nielsen and
   Nielsen (2008) do, with the future calendar effects extrapolated by a
   rule whose forecasts do not depend on the identification.

## What exists

* `Triangle` (`triangle.rs`) stores dense cells with a mask, origins and
  ages in months, and gives every cell, future cells included, its
  valuation month (`Triangle::valuation_of(origin, dev)`, the origin's
  first month plus the age less one). `to_incremental` and
  `to_cumulative` work row by row and respect the mask; the crate-private
  `map_rows`, `cells` and `with_cells` build transformed copies.
* Calendar identity in the crate is the valuation `Month` (`CalendarView`,
  `CalendarYearTest`, `triangle.md` decision 2) or, in `TriangleFrame`, an
  integer index at the development grain counted from the earliest
  valuation (`frame.rs`).
* `ChainLadderFit` keeps `origins`, `latest_position`, `latest` and the
  factors in use (`ldf()`, at the grid's ages; the tail separately in
  `tail.factor`); the crate-private `projection(origin)` gives the
  cumulative values from the latest age to the oldest age.
* `OdpGlm::fit_design` (`odp_glm.rs`) fits any design over the
  `TriangleFrame` rows and predicts future cells by selecting rows of the
  same design, so a numeric `calendar_index` column extrapolates as a
  linear trend, while a `calendar` factor cannot be estimated (its future
  levels have no observed cell, and `prospicio-glm` does not alias a
  collinear design: it errors). `OdpGlmFit` exposes `future` (origin,
  age) and `future_means`.
* `BerquistSherman` (`berquist_sherman.rs`) restates a triangle and
  returns a cumulative copy for any method to develop: the shape item 1
  follows. Its trend is per origin, not per calendar period.
* Nothing indexes cells by calendar period today.

## The method

### 1. Indexation

An index `I(p)` is given per calendar period `p` at its own grain (a
year, a quarter, a month). The increment of the cell at origin `i` and
age `j` is paid in the period `p(i, j)` that contains the cell's
valuation month. Restated at the price level of a base period `b`:

    x*_ij = x_ij I(b) / I(p(i, j))

The restated triangle is cumulated again when the input was cumulative,
so any method (chain ladder, Mack, BF, the bootstraps, `OdpGlm`) develops
it unchanged. A future increment `y*_ij` projected at the base level is
re-inflated to the money of its payment period:

    y_ij = y*_ij I(p(i, j)) / I(b)

with `I` beyond the last given period extended at a constant annual rate
`f`, `I(p) = I(last) (1 + f)^(m / 12)`, `m` the months from the last
given period's start to `p`'s start. The tail past the oldest age has no
cell; its increment is placed one development period after the oldest
age, as Taylor (1977, equation 8) places `s_{i,k+}` at `lambda_{k+i+1}`.

Incurred triangles can be restated the same way, but an incurred
increment mixes payments with changes in case reserves set at the
valuation's price level; the note documents this and leaves the choice
to the user (as Friedland, chapter 13, does for Berquist-Sherman).

### 2. Taylor's separation method

Let `s_ij` be the increment of origin `i` at development period `j`
divided by a normaliser `n_i` (Taylor: claims incurred in the origin
period, fixed at development period 0 so that the columns stay
comparable; the normaliser is optional, 1 by default). For a `k + 1` by
`k + 1` triangle (`i, j = 0..k`):

    E[s_ij] = r_j lambda_{i+j},     sum_{j=0..k} r_j = 1

**Arithmetic.** With `v_j` the column sums and `d_h` the diagonal sums of
`s`, Taylor's equations (5) and (6) solve backward from `h = k`:

    lambda_h = d_h / (1 - sum_{j>h} r_j)
    r_h      = v_h / sum_{m>=h} lambda_m

These are the marginal equations of a Poisson likelihood with
development and calendar factors on `s_ij` (Verbeek 1972; Taylor 1977,
section 6), with no origin factor and no offset (the normaliser divides
the data before the fit). Checked on 2026-11-24: on Taylor's Example 1
the recursion's fitted values and those of `statsmodels` `glm(y ~ C(j) +
C(c), Poisson)` differ by at most 2.3e-13.

**Geometric.** `log s_ij = log r_j + log lambda_{i+j} + e_ij`, fitted by
least squares on the logs (a two-way additive model on a triangle: its
normal equations set the column and diagonal sums of the fitted logs to
those of the data). Its fitted values differ from the arithmetic ones
(on Example 1, 28.39 against 28.53 in the cell (0, 1)). Its `r_j` sum
to 1 as the arithmetic method's (decision 12), and the fit returns to
amounts by the geometric means or, on request, the lognormal means
`exp(sigma^2 / 2)` times larger, which need at least 3 origins for a
residual variance (decision 13).

**Projection.** Future `lambda_h` (`h > k`) are not fitted: Taylor
chooses them. The fit takes them as a constant annual rate from the last
fitted `lambda_k`, as an index (`CalendarIndex`, item 1, scaled to
`lambda_k` at its last observed period) or, as an option, the
exponential trend fitted to the last `m` fitted `lambda_h` (decision
14). The future increment of origin `i` at age `j` is `n_i r_j
lambda_{i+j}`.

### 3. An APC reserving GLM

`log mu_ij = delta + alpha_i + beta_j + gamma_{i+j}` with an
over-dispersed Poisson (or the other `OdpGlm` families) response on the
increments. Because the calendar index is the sum of the origin and
development indices, a linear trend can be moved between the three
effects: the design with all three factors is rank-deficient by one
beyond the usual references. Kuang, Nielsen and Nielsen (2008a) identify
the model by its canonical parameter: three anchor values (the levels at
`(0, 0)`, `(1, 0)` and `(0, 1)`) and the second differences of each
effect, `3k - 3` free parameters on a `k` by `k` triangle. Fitted values,
deviance and dispersion do not depend on the identification.

Forecasts need the future calendar second differences, `Delta^2
gamma_h` for `h > k`. Following Kuang, Nielsen and Nielsen (2008b) and
R `apc`'s `apc.forecast.apc`, three rules are invariant to the
identification:

* `I2`: future second differences are zero (the last linear calendar
  trend continues);
* `I1`: the next second difference makes the next first difference equal
  to the mean of the fitted first differences, then zero;
* `I0` (`apc`'s default): the calendar effect regressed on an intercept
  and a demeaned trend, extrapolated two steps, the second differences
  backed out, then zero.

The fit builds the canonical design by hand (it is full rank, so
`prospicio-glm` fits it unchanged) and the future design from the chosen
rule, and reuses `OdpGlm`'s private `finish` path: reserves, prediction
error by the delta method and `PredictiveDistribution` draws then cover
the extrapolated calendar effect as a deterministic function of the
fitted parameters. Sub-models (`AP`, the separation model as a GLM; `AC`,
the chain ladder; `PC`) are the same design with columns dropped.

## API

Rust, in a new module `crates/prospicio-reserving/src/inflation.rs` (PR
1), then `separation.rs` (PR 3) and `apc.rs` (PR 6), re-exported from
`lib.rs`, one line each.

```rust
/// An index by calendar period: CPI, wages, a claims-inflation index.
#[derive(Debug, Clone, PartialEq)]
pub struct CalendarIndex { /* first period, values, future rate */ }

impl CalendarIndex {
    /// `values[0]` is the index in `first`, then one value per period of
    /// `first`'s grain. Every value finite and positive.
    pub fn new(first: Period, values: Vec<f64>) -> Result<Self>;
    /// Extends the index past its last period at a constant annual rate
    /// (finite, above -1), compounded by months.
    pub fn with_future_rate(self, rate: f64) -> Result<Self>;
    /// The index of the period that contains `month`.
    pub fn value(&self, month: Month) -> Result<f64>;
    pub fn periods(&self) -> Vec<Period>;
    pub fn values(&self) -> &[f64];
    pub fn future_rate(&self) -> Option<f64>;
}

/// Restating at one price level and re-inflating a projection.
#[derive(Debug, Clone, PartialEq)]
pub struct Indexation {
    pub index: CalendarIndex,
    /// The base price level: the index's period that contains this month.
    pub base: Month,
}

impl Indexation {
    /// `I(period of month) / I(base)`.
    pub fn factor(&self, month: Month) -> Result<f64>;
    /// A copy of `triangle` with `column`'s increments restated at the
    /// base level, in the triangle's basis (cumulative or incremental),
    /// every segment. Other columns are unchanged.
    pub fn restate(&self, triangle: &Triangle, column: &str) -> Result<Triangle>;
    /// The inverse of `restate`: the base level back to the money of each
    /// payment period.
    pub fn inflate(&self, triangle: &Triangle, column: &str) -> Result<Triangle>;
    /// Future increments at the base level, re-inflated to the money of
    /// their payment periods, summed by origin.
    pub fn reinflate(&self, cells: &FutureCells) -> Result<Reinflated>;
}

/// Projected future increments, one per (origin, age), with the month
/// their period ends in (the valuation month of the cell).
#[derive(Debug, Clone, PartialEq)]
pub struct FutureCells {
    pub origins: Vec<Period>,
    /// (origin position, age in months, valuation month, increment)
    pub cells: Vec<FutureCell>,
}

impl FutureCells {
    /// A chain ladder's future increments inside the grid, and its tail's
    /// increment one development period past the oldest age (decision 5).
    pub fn from_chain_ladder(fit: &ChainLadderFit) -> Self;
    /// An `OdpGlm` fit's future means.
    pub fn from_odp_glm(fit: &OdpGlmFit) -> Result<Self>;
}

pub struct Reinflated {
    pub cells: FutureCells,         // re-inflated, with their origins
    pub reserves: Vec<f64>,         // re-inflated, by origin
    pub base_reserves: Vec<f64>,    // at the base level, by origin
}
// Reinflated::total_reserve(), Reinflated::total_base_reserve()

/// PR 3: Taylor's separation method.
pub struct Separation {
    pub kind: SeparationKind,            // Arithmetic (PR 3), Geometric (PR 4)
    pub normaliser: Option<Vec<f64>>,    // n_i per origin; None is 1
    pub future: FutureInflation,         // Rate(f64) | Index(CalendarIndex) | Trend { periods }
}
impl Separation {
    pub fn fit(&self, triangle: &Triangle, column: &str) -> Result<SeparationFit>;
    pub fn fit_segments(&self, triangle: &Triangle, column: &str)
        -> Result<SegmentFits<SeparationFit>>;
}
pub struct SeparationFit {
    pub origins: Vec<Period>,
    pub ages: Vec<Lag>,
    pub proportions: Vec<f64>,           // r_j, summing to 1
    pub calendar: Vec<Period>,           // the observed calendar periods
    pub indices: Vec<f64>,               // lambda_h, observed
    pub future_indices: Vec<(Period, f64)>,
    pub fitted: Vec<Vec<Option<f64>>>,  // n_i r_j lambda_{i+j}, observed cells
    pub future: FutureCells,             // n_i r_j lambda_{i+j}, future cells
    pub reserves: Vec<f64>,
}

/// PR 6: the APC GLM.
pub struct ApcGlm {
    pub family: Family,                  // as OdpGlm
    pub effects: ApcEffects,             // Apc | AccidentPeriod | AgePeriod | AccidentAge
    pub extrapolation: Extrapolation,    // I2 | I1 | I0
    pub tolerance: f64,
    pub max_iterations: usize,
}
impl ApcGlm { pub fn fit(&self, triangle: &Triangle, column: &str) -> Result<ApcGlmFit>; }
// ApcGlmFit: the canonical coefficients and their covariance, the
// accident, development and calendar second differences, the slopes,
// fitted and future means, reserves, prediction_error(),
// predict_distribution(n_sims, seed); FutureCells::from_apc_glm.
```

One new error variant, `Error::Inflation(String)` ("inflation: ..."),
for the data failures of the three methods (a missing index period, a
hole, a zero diagonal); bad settings use `InvalidSetting` and settings a
method cannot take `NotSupported` (`reserving-v02.md` decision 29).

Python (`prospicio.reserving`): `CalendarIndex(first, values,
future_rate=None)`, `Indexation(index, base)` with `restate`, `inflate`
and `reinflate(cells)`, `FutureCells.from_chain_ladder(fit)` and
`.from_odp_glm(fit)`; `Separation(kind="arithmetic", normaliser=None,
future_rate=None, future_index=None, trend_periods=None).fit(triangle,
column)`; `ApcGlm(...)`. R: `calendar_index()`, `indexation()`,
`restate()`, `inflate()`, `reinflate()`, `future_cells()`,
`separation()`, `apc_glm()`, S7 classes. All in new module files
(`crates/prospicio-python/src/reserving_inflation.rs`,
`crates/prospicio-r/src/reserving_inflation.rs`,
`R/prospicio/R/reserving_inflation.R`), so no PR of this note touches
`crates/prospicio-python/src/reserving.rs` (#963 splits it). No new
method goes on an existing fit class: `FutureCells::from_chain_ladder`
takes the fit instead.

## Decisions

1. **The calendar is a `Month`, the index has its own grain.** A cell's
   payment period is the index period that contains the cell's valuation
   month (`Triangle::valuation_of`), as `triangle.md` decision 2 keys the
   calendar view by valuation month: the index of the period that holds
   the end of the cell's development period. An index finer than the
   triangle (quarterly on annual cells) therefore takes its last
   sub-period; a user who wants the period's average gives the index at
   the triangle's grain. Quarterly or monthly development
   with annual origins, and an index at a coarser or finer grain than the
   triangle, then need nothing special. An index period before the first
   given period is an error; past the last one, the future rate is used,
   or the call fails without one.
2. **Indexation acts on increments.** `restate` and `inflate` take the
   triangle's increments (a cumulative triangle is differenced, scaled and
   cumulated back), respect the mask, act on every segment and keep the
   other columns. An increment after a hole covers several periods; it is
   scaled at the period of its own cell, which the doc states.
3. **The base is explicit.** No default base: the user names the month
   whose period sets the price level (usually the valuation's). Restating
   to another base is a constant factor per triangle, so ratios (link
   ratios, `r_j`) do not depend on it; reserves do, and `Reinflated`
   carries both levels.
4. **Future index by a constant annual rate.** Values given past the
   valuation are used as given; beyond the last value one constant rate,
   compounded monthly over the months from the last period's start. A
   path per simulation is decision 11 (PR 8).
5. **The tail's payment period.** A chain ladder's tail increment is
   placed one development period after the oldest age, Taylor's
   equation (8) convention. A user who wants a longer payout of the tail
   restates with a pattern beyond the triangle first (out of scope).
6. **Re-inflation takes future cells, not a fit.** `FutureCells` is the
   common shape; constructors from `ChainLadderFit`, `OdpGlmFit` and
   later `SeparationFit` and `ApcGlmFit`. The bootstraps and Bayesian
   fits produce draws by cell; re-inflating them is decision 1's
   question.
7. **Separation needs a full triangle.** Origin grain equal to
   development grain (so `h = i + j` exactly), one segment per fit
   (`fit_segments` for several), no holes, every origin at the latest
   diagonal, and at least two ages. Taylor's recursion needs the square
   triangle; anything else is `Error::Inflation` with the reason.
8. **The arithmetic method by Taylor's recursion.** It is exact and
   needs no iteration; its equality with the Poisson GLM on
   development and calendar factors is a test, not the implementation.
   Zero and negative increments are accepted as long as every diagonal
   sum and every `1 - sum_{j>h} r_j` is positive (else an error).
9. **The APC model is identified canonically.** Kuang, Nielsen and
   Nielsen's anchors and second differences, built by hand as a full-rank
   design; the future calendar effect by `I2`, `I1` or `I0`; the same
   grain and full-triangle conditions as decision 7. The reserve does not
   depend on the identification, which a test checks against a second
   identification (two calendar references) fitted through
   `OdpGlm::fit_design`.
10. **`Error::Inflation(String)`**, one variant for the three methods.
11. **Stochastic future inflation is PR 8** (the user, 2026-11-24, open
    decision 1 (a)): `Indexation::reinflate_draws` takes a
    `PredictiveDistribution` of future cells (the bootstraps', `OdpGlm`'s)
    and one index path per simulation, and gives the reserve draws in the
    money of the payment periods; after PR 2.
12. **Geometric normalisation `sum r_j = 1`** (the user, 2026-11-24, open
    decision 2 (a)), as the arithmetic method, so the two methods'
    `lambda` compare. Fitted values do not depend on it.
13. **Geometric retransformation: both** (the user, 2026-11-24, open
    decision 3 (c)): the geometric means `exp(eta)` (Taylor's) by default,
    the lognormal means `exp(eta + sigma^2 / 2)` as an option.
14. **Future `lambda` by a fitted trend, as an option** (the user,
    2026-11-24, open decision 4 (b)): besides a given rate or index, the
    exponential trend of the last `m` fitted `lambda`, with no default
    `m`.
15. **The APC reference is Python `apc` 1.0.2** (the user, 2026-11-24,
    open decision 5 (a)) under `uv` with pins, falling back to
    `statsmodels` fitted values and deviance with the invariance tests
    (option (c)). It installs under `uv` on 2026-11-24 but imports
    `pkg_resources`, so the generator pins `setuptools` below 81.
16. **Standard errors for separation** (the user, 2026-11-24, open
    decision 6): none in PR 3 (Taylor gives none); the Poisson GLM's
    delta-method prediction error with PR 6's machinery.

## Validation targets

1. **Indexation (PR 1).** No independent package restates a triangle by
   a calendar index in the same terms, so the tests are exact
   identities, each to 1e-12 relative: a unit index changes nothing;
   `inflate(restate(t)) = t`; a change of base scales every cell by one
   constant; a constant-rate index on a triangle built as `r_j (1 +
   f)^(i + j)` restates to columns proportional to `r_j`, its chain
   ladder at the base level is exact, and `reinflate` gives back the
   triangle's own future cells `r_j (1 + f)^(i + j)`; a quarterly
   index on an annual triangle picks the quarter of each cell's
   valuation month.
2. **Arithmetic separation (PR 3).** Taylor (1977) Example 1 (4 by 4,
   a motor account): `r = 0.5835, 0.2878, 0.0866, 0.0421`, `lambda =
   86.4, 98.9, 102.0, 113.9`, the fitted triangle, its 3 by 3
   sub-triangle (`0.6101, 0.2980, 0.0921`; `82.6, 94.9, 97.7`) and the
   projection at 10% a year, to the printed rounding (5e-4 on `r`, 0.06
   on `lambda` and the cells). A reference file from `statsmodels` (a
   PEP 723 generator, `reserving_separation_statsmodels.py`): the
   Poisson GLM with development and calendar factors on GenIns
   (Taylor-Ashe) and ABC, its fitted values and future cells under a
   given rate, to 1e-9 relative; the `lambda` and `r` follow after the
   normalisation `sum r = 1`.
3. **Geometric separation (PR 4).** `statsmodels` OLS on the logs, the
   same datasets, to 1e-9.
4. **APC GLM (PR 6).** Python `apc` 1.0.2 (Harnau; the reference
   implementation of Kuang, Nielsen and Nielsen, PyPI 2020) if it still
   installs under `uv` with pins, on Taylor-Ashe (`apc.loss_TA()`, our
   `genins.csv`): fitted values, deviance, the canonical second
   differences and the `I2`, `I1` and `I0` forecasts; R `apc`'s
   documented Poisson APC `I0` total for Taylor-Ashe, 19,061,428, as a
   cross-check. R `apc` was archived on CRAN on 2026-03-02, so an R
   generator would install from the archive; R is not on the local
   machine, so the Python package is the first choice (decision 15).
   The invariance of fitted values and forecasts under a second
   identification, exactly.

## Pull requests

Each a small PR, stacked on the one before, linking #979.

1. **`[inflation-index]` Indexation, Rust.** `CalendarIndex`,
   `Indexation` (`factor`, `restate`, `inflate`, `reinflate`),
   `FutureCells` (`from_chain_ladder`, `from_odp_glm`), `Reinflated`,
   `Error::Inflation`; unit tests and
   `validation/tests/reserving_inflation.rs` (target 1); allow-list
   entries for the new items, bound in PR 2.
2. **`[inflation-index-bindings]` Python and R** for PR 1, in new module
   files, with tests (`python/tests/test_reserving_inflation.py`,
   `R/prospicio/tests/test-reserving-inflation.R`) and the regenerated
   stub, `man/`, `NAMESPACE` and wrappers.
3. **`[separation-arithmetic]` Arithmetic separation, Rust**, with the
   normaliser, the future rate or index, `fit_segments`,
   `FutureCells::from_separation` (target 2).
4. **`[separation-geometric]` Geometric separation, Rust** (target 3),
   under decisions 12 and 13.
5. **`[separation-bindings]` Python and R** for PRs 3 and 4.
6. **`[apc-glm]` The APC GLM, Rust** (target 4, decision 15), with the
   separation method's delta-method prediction error (decision 16).
7. **`[apc-glm-bindings]` Python and R** for PR 6.
8. **`[reinflate-draws]` Stochastic re-inflation, Rust, Python and R**
   (decision 11), after PR 2.

A guide (deflate, develop, re-inflate; separation on a deflated triangle
for superimposed inflation) belongs to the Guides lane: an issue for it
when PR 5 merges.

## Not in this note

* Calendar effects inside the Bayesian models (Meyers' CAY has one).
* Claims-inflation estimation from external series (the index is input).
* The Barnett and Zehnwirth probabilistic trend family (piecewise
  trends in three directions); the APC GLM's design covers its
  unrestricted case.

## References

* G. C. Taylor (1977), "Separation of inflation and other effects from
  the distribution of non-life insurance claim delays", ASTIN Bulletin
  9, 219-230.
* J. Verbeek (1972), "An approach to the analysis of claims experience
  in motor liability excess of loss reinsurance", ASTIN Bulletin 6,
  195-202.
* D. Kuang, B. Nielsen and J. P. Nielsen (2008a), "Identification of the
  age-period-cohort model and the extended chain-ladder model",
  Biometrika 95(4), 979-986; (2008b), "Forecasting with the
  age-period-cohort model and the extended chain-ladder model",
  Biometrika 95(4), 987-991.
* S. Bjorkwall, O. Hossjer and E. Ohlsson (2010), "Bootstrapping the
  separation method in claims reserving", ASTIN Bulletin 40(2).
* G. Barnett and B. Zehnwirth (2000), "Best estimates for reserves",
  PCAS 87, 245-321.
* R `apc` (Nielsen et al., archived on CRAN 2026-03-02); Python `apc`
  1.0.2 (J. Harnau, 2020).
