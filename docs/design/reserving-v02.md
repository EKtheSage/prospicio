# Design note: Reserving v0.2 (expected-loss methods, tails, Clark, one-year view)

Status: **Implemented** (PRs #131–#137, #150, #156, #157, #159, #160, #163; decision 9 on branch `claude/bootstrap-tail-error`) · Lane: Reserving · Depends on: `triangle.md`, `docs/architecture.md` (v0.2 row of the roadmap)

## Goal

Broaden `prospicio-reserving` past the chain ladder: the expected-loss family
(expected loss ratio, Bornhuetter–Ferguson, Benktander, Cape Cod) driven by
an exposure column, tail factors, Clark's growth-curve methods, and the
one-year view of Merz and Wüthrich. Every method lands in Rust, Python and R
in the same PR, with parity tests against chainladder-python and R
ChainLadder.

## Pull requests

Each PR holds one family, in this order, so that each can merge on its own:

| PR | Branch | Holds | Base |
|---|---|---|---|
| 0 | `claude/v02-design` | this note | `main` |
| 1 | `claude/prospicio-math-nelder-mead` | `prospicio_math::optimize::nelder_mead` (Probability lane's crate, with the user's go-ahead) | `main` |
| 2 | `claude/v02-expected-loss` | `ExpectedLoss`, `BornhuetterFerguson`, `Benktander`, `CapeCod` | PR 0 |
| 3 | `claude/v02-tails` | `Tail` and its estimators, Mack with a tail | PR 0 |
| 4 | `claude/v02-one-year` | Merz–Wüthrich claims development result | PR 0 |
| 5 | `claude/v02-clark` | `ClarkLdf`, `ClarkCapeCod` | PRs 1 and 2 |

PRs 2–4 touch the same binding files (`crates/prospicio-python/src/reserving.rs`,
`crates/prospicio-r/src/reserving.rs`, `R/prospicio/R/reserving.R`) and the
generated stub, `NAMESPACE` and `man/`. Whichever merges second merges
`main` and regenerates the generated files; it does not merge them by hand.

## References

| Method | Primary reference | Secondary |
|---|---|---|
| Expected loss, BF, Benktander, Cape Cod | chainladder-python 0.10.1 (`ExpectedLoss`, `BornhuetterFerguson`, `Benktander`, `CapeCod`) | hand-computed cases |
| Tails | chainladder-python (`TailConstant`, `TailCurve`, `TailBondy`); R ChainLadder 0.2.21 `MackChainLadder(tail = TRUE / number)` for Mack's tail sigma and standard error | — |
| Clark | R ChainLadder `ClarkLDF`, `ClarkCapeCod` | chainladder-python `ClarkLDF` |
| One-year view | R ChainLadder `CDR(MackChainLadder(...))`, `dev = "all"` for the full run-off | Merz and Wüthrich (2008), published example |
| Simulated one-year view | ODP: R ChainLadder `CDR(MackChainLadder(...))` times the measured ODP-to-Mack ratio, `BootChainLadder` for the origin with one cell left. Mack's process: R ChainLadder `CDR(MackChainLadder(...))` itself | England, Verrall and Wüthrich (2019), Tables 2 and 4 |
| Mack's bootstrap, lifetime view | R ChainLadder `MackChainLadder(...)` process and parameter risks | England, Verrall and Wüthrich (2019), Table 4 |
| Tails in the bootstraps | R ChainLadder `MackChainLadder(tail = ...)` process and parameter risks with the tail (`reserving_tails_r.csv`) | Mack (1999); the delta method on R's tail rule |

Each family has its own generator script under `validation/scripts/` and
reference CSV under `validation/reference/` in the format of
`reserving_chainladder_r.csv` (`dataset,method,quantity,arg,expected,abs_tol,rel_tol,source`),
and its own parity test file `validation/tests/reserving_<family>.rs` that
evaluates every row. R references are generated with R ChainLadder 0.2.21;
chainladder-python references with
`uv run --no-project --with chainladder==0.10.1`.

Data: RAA, GenIns and ABC as in v0.1. The exposure methods need premium, so
two inputs are added to `validation/data/`:

* `clrd_wkcomp.csv`: the CAS loss reserve database, workers' compensation
  summed over companies (chainladder-python
  `load_sample('clrd').groupby('LOB').sum().loc['wkcomp']`), columns
  `CumPaidLoss` and `EarnedPremNet` as `paid` and `premium`;
* `genins_premium.csv`: the premium of R ChainLadder's Clark example,
  `10000000 + 400000 * (0:9)` on GenIns origins.

## Decisions

### 1. Exposure is a measure column

Methods that need exposure take the name of a measure column of the same
triangle: `fit(&triangle, "paid", "premium")`. Each origin's exposure is
that column's latest observed value for the origin in the segment being
fitted, matching chainladder-python's use of `sample_weight.latest_diagonal`.
An origin with no observed, finite, positive exposure is an error that names
the origin. `fit_segments` reads each segment's own exposure.

As implemented: the value read is the latest *cumulative* value, since
segments are read cumulatively (an incremental triangle's exposure is
cumulated like its losses, so a premium repeated on every age of an
incremental triangle counts once per age; chainladder-python's
`premium.latest_diagonal` reads the last increment). Origins are matched by position in the
triangle, so the exposure column may observe more origins or ages than the
losses. The error is `Error::InvalidExposure { column, origin }`; an
unknown exposure column is `Error::UnknownColumn`.

### 2. The expected-loss family

```rust
pub struct ExpectedLoss { pub apriori: f64, pub chain_ladder: ChainLadder }
pub struct BornhuetterFerguson { pub apriori: f64, pub chain_ladder: ChainLadder }
pub struct Benktander { pub apriori: f64, pub n_iters: usize, pub chain_ladder: ChainLadder }
pub struct CapeCod { pub trend: f64, pub decay: f64, pub chain_ladder: ChainLadder }
```

With `q = 1 / cdf` the share developed at the origin's latest age and
`U0 = apriori * exposure`:

* expected loss: `ultimate = U0`;
* BF: `ultimate = latest + (1 - q) * U0`;
* Benktander: iterate `U(k) = latest + (1 - q) * U(k-1)` `n_iters` times
  from `U0`, so `n_iters = 1` is BF and a large `n_iters` approaches the
  chain ladder;
* Cape Cod: the apriori of each origin is estimated from the used-up
  exposure `exposure * q`, trended by `trend` and weighted by `decay` across
  origins exactly as chainladder-python's `CapeCod` does, and the ultimate
  is BF with the detrended apriori.

All four return one fit type:

```rust
pub struct ExpectedLossFit {
    pub chain_ladder: ChainLadderFit, // the development pattern and latest values
    pub exposure: Vec<f64>,           // per origin
    pub apriori: Vec<f64>,            // expected loss ratio applied per origin
    pub ultimate: Vec<f64>,           // this method's ultimate, not the chain ladder's
}
```

`CapeCod::fit` returns a `CapeCodFit` that holds an `ExpectedLossFit` and the
trended apriori before detrending (chainladder-python's `apriori_`).

`ReserveFit` gains `fn ultimate(&self) -> &[f64]`, defaulting to the chain
ladder's, and the long tables use it, so `SegmentFits` of any method report
that method's ultimate and reserve.

As implemented (`crates/prospicio-reserving/src/expected_loss.rs`, parity in
`validation/tests/reserving_expected_loss.rs`, every row of
`reserving_expected_loss_python.csv` to 1e-9 relative):

* Defaults follow chainladder-python: `apriori = 1`, `n_iters = 1`,
  `trend = 0`, `decay = 1`. `apriori` must be finite and positive, `trend`
  finite and above -1, `decay` in `[0, 1]`; otherwise
  `Error::InvalidSetting { name, value, expected }`. chainladder-python
  checks none of these.
* Cape Cod's trend factor is `(1 + trend)^(m / 12)`, `m` the months from the
  last month of the origin period to the triangle's valuation (at least 0),
  as chainladder-python's `Triangle.trend(axis="origin")`; in `fit_segments`
  every segment trends to the whole triangle's valuation. The ultimates do
  not depend on that choice, since each origin is detrended by its own
  factor. The decay weight uses the distance between origin positions.
* `CapeCodFit` is `{ expected_loss: ExpectedLossFit, trended_apriori:
  Vec<f64> }`; `ExpectedLossFit::apriori` holds the detrended apriori
  (chainladder-python's `detrended_apriori_`). chainladder-python's
  `CapeCod(n_iters)` is not offered; ours is its default, `n_iters = 1`.
* Benktander uses chainladder-python's closed form
  `sum(p^k, k < n) latest + p^n U0`, `p = 1 - q`, with `p^n` and the sum
  built by repeated squaring, so a huge `n_iters` is cheap. Stepping one at
  a time is not: with negative development (`cdf < 1`, so `p < 0`) the
  floating-point steps can end in a two-cycle and never stop. Where
  `cdf < 1/2`, `|p| > 1` and the ultimate diverges as `n_iters` grows.
* An origin without a value on the valuation diagonal uses its latest
  observed value and the cdf at that age, as the chain ladder does, and
  Cape Cod pools it with the other origins. chainladder-python gives such
  an origin a NaN ultimate and leaves it out of the Cape Cod pool, so there
  one hole changes the apriori and ultimates of every other origin.
* Long tables add `exposure` and `apriori` per origin (Cape Cod also
  `trended_apriori`) and the total `exposure` per segment.
* Python: `ExpectedLoss`, `BornhuetterFerguson`, `Benktander` and `CapeCod`
  take the settings above plus `average`, `sigma_interpolation` and `tail`
  for the chain ladder, and `fit(triangle, column, exposure)` fits every
  segment, returning `ExpectedLossFit` or `CapeCodFit`.
* R: `expected_loss()`, `bornhuetter_ferguson()`, `benktander()` and
  `cape_cod()` take `(triangle, column, exposure, ...)` with the same
  settings and return an `expected_loss_fit` or `cape_cod_fit` with the
  properties of the Python fits; `as.data.frame()`, `totals_frame()`,
  `development_frame()` and `segment()` work on both.

### 3. Tails

The chain ladder's `tail: f64` becomes `tail: Tail`:

```rust
pub enum Tail {
    Constant(TailConstant), // a given factor, with chainladder-python's decay
    Curve(TailCurve),       // exponential or inverse-power fit of f - 1, extrapolated
    Bondy(TailBondy),
    LogLinear,              // R ChainLadder's tail = TRUE rule (tailfactor)
}
```

`Tail::default()` is a constant 1, and `From<f64>` builds a constant, so
`ChainLadder { tail: 1.05.into(), .. }` reads as before. Fitting a tail
against a `DevelopmentFit` gives a `TailFit` with the factor and, for Mack,
the tail's sigma and standard error. Mack accepts a tail and adds its
process and parameter risk the way R `MackChainLadder` does, extrapolating
`tail.sigma` and `tail.se` log-linearly when they are not given.

As built (the references forced these details):

```rust
pub struct TailConstant { pub factor: f64, pub decay: f64, pub attachment_age: Option<Lag> }
pub struct TailCurve {
    pub curve: CurveShape,                         // Exponential | InversePower
    pub fit_period: (Option<Lag>, Option<Lag>),    // ages [from, to), chainladder-python's convention
    pub extrap_periods: usize,                     // 100
    pub attachment_age: Option<Lag>,
}
pub struct TailBondy { pub earliest_age: Option<Lag>, pub attachment_age: Option<Lag> }

pub struct TailFit {
    pub attachment: usize, // first development position the tail replaced
    pub ldf: Vec<f64>,     // selected factors: estimated, then the tail's, then past the oldest age
    pub factor: f64,       // oldest age to ultimate
    pub sigma: f64,
    pub std_err: f64,
}

pub struct Mack {
    pub development: Development,
    pub tail: Tail,
    pub tail_sigma: Option<f64>,   // R tail.sigma
    pub tail_std_err: Option<f64>, // R tail.se
}
```

* `ChainLadderFit.tail` is the `TailFit` (it was the factor), and
  `ChainLadderFit::ldf()` the selected factors within the triangle, which
  the projection and Mack's recursions use. A tail attached before the
  oldest age replaces estimated factors, as chainladder-python does; Mack
  keeps the estimated sigmas and standard errors there, as it does.
* `TailFit.ldf` runs past the oldest age as chainladder-python's `ldf_`
  (`projection_period` 12): one factor per development period of the next
  year, then one to ultimate. `LogLinear` gives one factor, as R appends
  one. Only their product, `factor`, enters ultimates and Mack.
* `TailBondy` keeps the factor from its attachment age to the next and
  replaces those after, while `TailConstant` and `TailCurve` replace the
  factor from their attachment age; both follow chainladder-python. The
  Bondy exponent is the exact least-squares optimum; chainladder-python's
  `least_squares` stops early (relative cost change 1e-8), so generalized
  Bondy rows are checked to 1e-4 (`knowledge/findings/bondy-least-squares-stop.md`).
* `LogLinear` is R's `tailfactor` exactly, including its quirks: it tests
  the third- and second-last factors (`f[n-2] * f[n-1] > 1.0001`, not the
  last two) and resets a tail above 2 to 1.
* The tail's sigma and standard error follow R's `tail_SE`, which
  chainladder-python's `_get_tail_stats` matches: the tail's position on
  the line through `ln(f - 1)` (factors above 1) is where it reaches
  `ln(factor - 1)`, read off lines through `ln(sigma)` and `ln(std_err)`.
  A factor of exactly 1 is no tail and carries no risk. A factor below 1
  follows chainladder-python, a deviation from R: it scales the
  ultimates, as the chain ladder's cdf does, and its sigma and standard
  error are read where a factor of 1.001 would be
  (`_get_tail_weighted_time_period`); given values apply to any factor
  other than 1. R's `MackChainLadder` ignores a tail below 1 altogether,
  so following it would make Mack's ultimates differ from the chain
  ladder's for the same tail. Every origin, the oldest included, carries
  the tail's risk.
* `TailCurve.fit_period` and `TailBondy.earliest_age` take the last age at
  or before the given one, which is chainladder-python's positional
  `int(age / grain - 1)` on ages that are multiples of the grain; the
  attachment ages are read by value, as Python does. A `TailConstant`
  attached at or before the youngest age replaces every estimated factor;
  chainladder-python ignores that attachment (`if attach_idx:` with index
  0), a deviation kept on purpose and noted in
  `knowledge/references/chainladder-tails.md`.
* A tail that cannot be fitted is `Error::Tail(reason)`; a non-positive
  constant stays `Error::InvalidTail`.

Bindings: Python `ChainLadder(tail=...)` and `Mack(tail=...)` accept a float
or a `TailConstant`, `TailCurve`, `TailBondy` or `TailLogLinear`; R accepts a
number or the matching constructor. The expected-loss methods
(`ExpectedLoss`, `BornhuetterFerguson`, `Benktander`, `CapeCod`; R
`expected_loss()` and the rest) take the same `tail` for their chain
ladder. In Python the default is `tail=None` (no tail); `Mack` also takes `tail_sigma` and `tail_std_err`, and the fits
report `tail`, `tail_ldf`, `tail_sigma`, `tail_std_err`,
`tail_attachment_age` and `estimated_ldf` (the factors before the tail
replaced any). With several segments, `totals_frame()` has each segment's
`tail`, `tail_sigma` and `tail_std_err` (`SegmentFits::totals`). Parity:
`validation/tests/reserving_tails.rs` against
`reserving_tails_r.csv` and `reserving_tails_python.csv`.

### 4. The one-year view

```rust
impl MackFit {
    pub fn claims_development_result(&self) -> Result<ClaimsDevelopmentResult>;
}

pub struct ClaimsDevelopmentResult {
    pub origins: Vec<Period>,
    pub one_year_standard_error: Vec<f64>,       // per origin, R "CDR(1)S.E."
    pub total_one_year_standard_error: f64,
    pub by_calendar_year: Vec<Vec<f64>>,         // [year][origin], R dev = "all"
    pub total_by_calendar_year: Vec<f64>,
}

impl ClaimsDevelopmentResult {
    pub fn run_off_standard_error(&self) -> Vec<f64>;   // sqrt of summed yearly MSEPs
    pub fn total_run_off_standard_error(&self) -> f64;
}
```

Merz and Wüthrich's formulas assume volume-weighted factors and no tail, so
any other `alpha` or a tail is an error (R only warns for `alpha != 1`).
With the `TailFit` of decision 3, no tail means a factor of exactly 1 that
replaced no estimated factor (`attachment` at the number of factors); the
CDR then uses `ChainLadderFit::ldf()`, the factors the projection uses. They also assume a full trapezoid, the latest values on one
calendar diagonal with one new origin per period, as R reads it
positionally; any other shape is an error. The formulas need the volume
`S_k` behind each factor, so `DevelopmentFit` gains `volume: Vec<f64>`
(`sum(C[k]^alpha)` over the link pairs).

The check is on the latest values only, so an origin with an interior hole
(a missing value before its latest) is accepted. Its `S_k` is the pair
volume, which leaves the hole out now and next year, and the run-off adds
up to Mack's. This is a deliberate deviation from R: R's `CDR` takes the
volumes from the full triangle, imputed cell included, and on RAA without
1982 at 48 its run-off (24,837) falls short of its own Mack (24,848);
prospicio-reserving's matches Mack.

`by_calendar_year` has one year per age-to-age factor. R reports one per
age, so its last year, `CDR(n)S.E.`, is past the run-off and always zero;
the parity test reads it as zero. R's `Mack.S.E.` column is the square
root of the summed yearly MSEPs, which equals Mack's ultimate standard
error; the parity test checks it against both `MackFit::standard_error`
and `run_off_standard_error`.

Data: `validation/data/mw2008.csv` and `mw2014.csv` are R ChainLadder's
`MW2008` and `MW2014`, origins relabelled from 2001. The paper's Table 4
totals (reserves 2,237,826, one-year 81,080, Mack 108,401) are unit tests;
its two oldest open origins differ from R in the fourth digit
(`knowledge/references/r-chainladder-cdr.md`).

The closed form is first order: Merz and Wüthrich (2008), Appendix A,
(A.1), replace each product `prod(1 + a_j) - 1` of the conditional MSEP
by `sum(a_j)`, a lower bound. The exact one-year SD under their
conditional resampling is at most 0.09% above it on RAA, GenIns and ABC
(RAA total 25,185.83 against 25,166.30), which
`validation/tests/reserving_one_year_mack.rs` pins; the closed form stays.

Bindings: Python `MackFit.claims_development_result()` returns a
`ClaimsDevelopmentResult`; R `claims_development_result(fit)` takes a
`mack_fit` and returns the S7 class of that name, with `by_calendar_year`
as an origin x calendar-year matrix. Both need a single-segment fit.

### 5. Clark's growth curves

```rust
pub enum GrowthCurve { LogLogistic, Weibull }
pub struct ClarkLdf { pub curve: GrowthCurve, pub max_age: Option<f64> }
pub struct ClarkCapeCod { pub curve: GrowthCurve, pub max_age: Option<f64> }
```

`ClarkLdf::fit(&triangle, column)` and `ClarkCapeCod::fit(&triangle, column,
exposure)` fit incremental losses by maximum likelihood with an
over-dispersed Poisson scale, at the average date of loss as R's
`adol = TRUE` does. `max_age` truncates the curve as R's `maxage`. Given
`(omega, theta)`, the ultimates (LDF) or the ELR (Cape Cod) have closed
forms, so the likelihood is profiled onto two parameters and minimized with
`prospicio_math::optimize::nelder_mead` on `(ln omega, ln theta)`. The parameter
covariance is the inverse Fisher information times the scale, from analytic
derivatives of the curve, as in Clark (2003). The fit reports `omega`,
`theta`, the scale, ultimates, reserves, process, parameter and total
standard errors per origin and in total, and implements `ReserveFit`.

As implemented (`crates/prospicio-reserving/src/clark.rs`, parity in
`validation/tests/reserving_clark.rs`; R's definitions and quirks are in
`knowledge/references/r-chainladder-clark.md`):

* Both methods return one `ClarkFit`: the volume-weighted `ChainLadderFit`
  (origins, latest values; R also fits it, for its starting values),
  `curve`, `omega`, `theta`, `elr` and `exposure` (Cape Cod only),
  `scale`, `max_age`, `origin_width`, `expected_ultimate` (`U_i`, to
  infinity), `ultimate`, `process_risk`, `parameter_risk`,
  `standard_error` and their totals, the parameter `covariance` (`U_i` or
  `ELR`, then `omega`, `theta`) and `n_observations`; `growth(age)` gives
  the fitted share developed by a development age. Long tables add
  `exposure` (Cape Cod), `expected_ultimate` and the standard errors per
  origin, and `omega`, `theta`, `scale` (and `elr`) per segment.
* Ages follow R's `adol = TRUE` with its default `adol.age`, half the
  origin width. The width is the origin period's length (12 months for
  annual origins), where R defaults to the mean step between ages; the two
  agree whenever origin and development grains match, as in every parity
  case. Incremental values are differences between an origin's observed
  cumulative values, from the previous observed age (0 for the first), so a
  missing cell does not drop its neighbour as R's `cum2incr` does.
* Reported values are R's: ClarkLDF's reserve is `latest * (G(m) / G(age)
  - 1)` and its process variance `scale * U * (G(max_age) - G(age))` with
  `max_age` unshifted, as R computes it (`m` is the shifted `max_age`);
  Cape Cod's reserve is the fitted `ELR * exposure * (G(m) - G(age))`.
  The scale divides by observations less parameters; negative parameter
  variances are set to 0; a Fisher information whose reciprocal condition
  number is below machine epsilon gives NaN parameter risk, as R.
* Deviation from R: the Weibull Fisher information uses the correct
  `d2G/domega2 = v ln(x/theta)^2 (1 - u)`, where R has
  `2 v ln(x/theta) (1 - u)`; Weibull parameter standard errors differ
  from R as shipped by up to 7.3% on the reference triangles, and their
  reference rows come from R with that entry corrected.
* The search is Nelder–Mead to `tolerance = 1e-10` from R's starting
  curve parameters; it is unbounded where R bounds the Weibull at
  `omega <= 2`, `theta <= 2 * max(age)`, both curves at `omega >= 0.01`,
  the log-logistic at `theta >= min(0.5, ages)`, and the Cape Cod ELR at
  10 (without a warning). Deviation from R: where R's ELR is pinned at 10,
  as with exposure in smaller units than the losses, prospicio_reserving returns
  the unbounded maximum (RAA with premium 1000: ELR 27.61, R 10). Losses are divided by the largest
  chain-ladder ultimate while fitting (R's `magscale`). Errors: fewer than
  four ages is `TooFewAges`; a `max_age` before the last age is
  `InvalidSetting`; an origin (LDF) or all origins (Cape Cod) without a
  positive latest value, too few observations, or no convergence is
  `Error::Clark`.
* Parity: every row of `reserving_clark_r.csv` (R as shipped to 1e-2,
  since its L-BFGS-B stops up to 4e-3 short; R with `factr = 1` to 1e-5)
  and of `reserving_clark_python.csv` (chainladder-python's `ClarkLDF`
  where comparable, to 1e-3).
* Python: `ClarkLdf(curve, max_age).fit(triangle, column)` and
  `ClarkCapeCod(curve, max_age).fit(triangle, column, exposure)` return a
  `ClarkFit`.
* R: `clark_ldf(triangle, column = NULL, curve = c("loglogistic",
  "weibull"), max_age = Inf)` and `clark_cape_cod(triangle, column,
  exposure, curve, max_age = Inf)` return a `clark_fit`; `max_age = Inf`
  or `NULL` means no truncation, where Python uses `None`.
  `growth(fit, age)` is vectorised over `age`.

### 6. `nelder_mead`

```rust
pub struct NelderMead { pub initial_step: f64, pub tolerance: f64, pub max_iterations: usize }
pub struct Minimum { pub x: Vec<f64>, pub value: f64, pub iterations: usize, pub converged: bool }
pub fn nelder_mead(f: impl FnMut(&[f64]) -> f64, x0: &[f64], options: &NelderMead) -> Minimum;
```

Standard simplex moves (reflection 1, expansion 2, contraction 1/2,
shrink 1/2), converged when the spread of the simplex's values and its
diameter are both within `tolerance`, and one restart from the best point
to avoid a collapsed simplex. A domain crate never writes its own
optimizer.

### 7. Bindings

Each Rust method gets a Python class and an R function in the style of
`ChainLadder`/`Mack` and `chain_ladder_fit()`/`mack_fit()`: same argument
names, `fit(triangle, column, exposure)` where exposure is needed, results
by origin and in total, `fit_segments` long tables. Docs are regenerated
with `cargo xtask docs`.

### 8. The simulated one-year view

Merz–Wüthrich (decision 4) is exact only for volume-weighted factors with
no tail. Any other method, weighting or tail gets its one-year view by
re-reserving on the ODP bootstrap ("actuary in the box": Ohlsson and
Lauzeningks 2009; England, Verrall and Wüthrich 2019):

```rust
pub enum OneYearMethod {
    ChainLadder(ChainLadder),
    ExpectedLoss(ExpectedLoss, String),        // the String names the exposure column
    BornhuetterFerguson(BornhuetterFerguson, String),
    Benktander(Benktander, String),
    CapeCod(CapeCod, String),
}

impl OdpBootstrap {
    pub fn one_year(&self, triangle: &Triangle, column: &str, method: &OneYearMethod)
        -> Result<OneYearFit>;
    pub fn one_year_segments(&self, triangle: &Triangle, column: &str, method: &OneYearMethod)
        -> Result<OneYearFits>;
}

pub struct OneYearFit {
    pub bootstrap: OdpBootstrapSegment, // fitted values, residuals and scale
    pub opening_ultimate: Vec<f64>,     // the method on the observed triangle
    pub opening_reserve: Vec<f64>,
    pub cdr: PredictiveDistribution,    // claims development result, dimension origin
}

pub struct OneYearSegment {           // OneYearFit without the CDR, per segment
    pub bootstrap: OdpBootstrapSegment,
    pub opening_ultimate: Vec<f64>,
    pub opening_reserve: Vec<f64>,
}

pub struct OneYearFits {
    pub segments: SegmentFits<OneYearSegment>,
    pub cdr: PredictiveDistribution,    // dimensions: the keys, then origin
}
```

`OneYearFits::to_long` and `totals` are `SegmentFits`' tables with
`ultimate` and `reserve` renamed `opening_ultimate` and `opening_reserve`,
plus `cdr_mean` and `cdr_std_dev` (and the bootstrap's `scale` in
`totals`); `segment` picks one segment and its part of the joint CDR.

Per simulation, on its own `StreamRng` stream:

1. Resample the adjusted residuals into a pseudo triangle and re-estimate
   the volume-weighted factors, as the ODP bootstrap does (parameter
   error).
2. Simulate every cell of the coming year (below), each origin's in
   development order from its latest cell: the increment into age `j + 1`
   has mean `C*_j (f*_j - 1)`, with `C*_j` the pseudo triangle's latest
   value `C*_latest` carried forward on the pseudo factors
   (`C*_j+1 = C*_j f*_j`), and its own process error with scale `phi`.
   That is how the lifetime ODP bootstrap projects (England 2002), and the
   pseudo latest value carries the estimation error of the origin's level,
   part of the ODP's parameter error: projecting from the observed latest
   value would leave it out (RAA's total standard deviation 30% lower). So
   an origin whose remaining cells all fall in the year has a one-year view
   distributed as its lifetime bootstrap reserve. An origin already at the
   triangle's last age gets no new cell.
3. Add the increments to the origin's observed latest value, append the
   year's cells to the observed triangle (exposure columns carry each
   origin's latest value forward) and refit `method` on it. Cape Cod trends
   to the valuation twelve months later.
4. `CDR_i = U0_i - U1_i`, the opening ultimate less the re-estimated one,
   which equals the opening reserve less the year's simulated payment and
   the closing reserve.

#### The coming year

The coming year is the twelve months after the segment's valuation `V`,
the valuation month of its latest cell, which is the triangle's valuation
(`Triangle::valuation`) unless the segment stops earlier: a cell is in the
year when its valuation month `v` (`Triangle::valuation_of`, the last
month the cell covers) has `V < v <= V + 12` months. So the horizon is
twelve months whatever the grain: an annual development grain gives one
cell per origin, the next diagonal; a quarterly grain four, a monthly one
twelve, fewer for an origin that reaches the triangle's last age within
the year (with quarterly origins, the youngest quarters reach a 12-month
last age after one, two or three cells). Segments may end on different
diagonals, and each simulates the twelve months after its own, as before;
Cape Cod's refit trends to the triangle's valuation plus twelve months.

An origin whose latest cell lags `V` (it stops short of its segment's
latest diagonal; the bootstrap still needs it observed from the first age
to its latest) develops from its own latest cell. The cells between that
cell and the year, valued at or before `V`, are drawn as steps of the chain
but not appended: they are in the past and were not observed at `V`, and
the closing triangle gets only the cells the year reveals, the origin's
value at its first cell after `V` and on. So a lagging origin's CDR covers
its development from its latest cell, not only the year's. An origin with
no cell left in the year gets none.

Across the several cells of a year, the parameter error is drawn once per
simulation, for the whole year, and the process error once per cell:

* ODP: one pseudo triangle and one set of pseudo factors per simulation;
  every cell's increment is projected from the pseudo latest value on
  those factors, as the lifetime bootstrap projects, with an independent
  Gamma process error. The increments' means do not depend on the drawn
  values of the cells before, as in the ODP model, whose increments are
  independent given the parameters.
* Mack: one set of pseudo factors per simulation; each cell is drawn from
  the one before (the observed latest value for the first), mean `f*_k C`
  and variance `sigma_k^2 |C|^(2 - alpha)`, because Mack's model is a
  Markov chain conditional on the latest value. The absolute value only
  matters for a drawn value below zero (a normal process, or a negative
  pseudo factor), which the observed values never are.

On an annual triangle every origin has one cell, the next, so the draws
are bit for bit those of the annual-only implementation: checked by
hashing every draw of both models, every Mack process, centred and not,
four methods (two chain ladders, Bornhuetter–Ferguson and Cape Cod), both
entry points, on RAA, GenIns and ABC, with and without a segment on an
earlier diagonal (297 hashes, before and after; not a CI test, since the
Gamma's draws need not agree across platforms).

Checks of the grain and the lag (unit tests in `one_year_bootstrap.rs`,
`validation/tests/reserving_one_year_bootstrap.rs`):

* The cells of the year: one per origin on RAA; four per origin for
  quarterly development of annual origins; one, two and three for
  quarterly origins a quarter, half and three quarters short of a
  12-month last age; for a lagging origin, the steps and the appended
  cells, annual and quarterly.
* An exact pattern split into quarters has scale zero and a zero CDR in
  every simulation, under the ODP and Mack's model (all sigmas zero), and
  the annual opening reserve.
* A lagging origin on an exact pattern moves by hand under
  Bornhuetter–Ferguson, with the growth over both its steps; on RAA, 1985
  cut back a year has a wider CDR than on the diagonal, under both models.
* RAA, GenIns and ABC split into quarters, each year's increment in four
  equal parts: the opening reserve is the annual one (a year's quarterly
  volume-weighted factors telescope to its annual factor) and the one-year
  standard deviation is about half the annual one under both models
  (measured at 5,000 simulations: ODP 0.43 to 0.47 per origin, 0.44 to
  0.47 in total; Mack 0.48 to 0.72 and 0.53 to 0.55). The two models get
  there differently. The ODP's variance is linear in the mean, so the
  halving is its scale's: the split leaves the Pearson chi-square
  unchanged (each quarter's residual is half the annual cell's), so the
  quarterly scale is the annual one times the ratio of degrees of freedom,
  exactly (36/171 on RAA and GenIns, 45/210 on ABC), and the process
  standard deviation falls by its square root, about 0.46. Mack's model
  takes successive link ratios as independent, and the split's quarterly
  ones deviate by about a quarter of the annual ones, so each quarter
  carries about a sixteenth of the year's variance, the four together a
  quarter. A split triangle is smoother than real quarterly data, so this
  checks the mechanics, not a quarterly calibration.
* Its first year's quarterly link ratios are equal across origins, so
  those sigmas are zero; such a factor gives Mack's bootstrap no residuals
  (they would be `0 / 0`; as zeros they cut RAA's pool mean square to
  0.854) and its pseudo factor is its factor. A test pins the quarterly
  RAA pool's mean square at 1. Quarterly RAA under Mack's Gamma process
  with seed 3 once drew a cell near zero and the next draw's shape
  underflowed: a Gamma or lognormal out of floating point range now draws
  its limit, zero (a regression test).
* A lagging origin appends only the year's cells: on RAA with 1985 cut
  back a year, both models' cells of the year hold its 84-month cell and
  not the 72-month step valued at the valuation.

The CDR is joint across origins (and segments), so its quantiles, VaR and
TVaR come from `PredictiveDistribution`. A new origin period written in the
coming year is not simulated, as in Merz–Wüthrich, and the tail beyond the
triangle's last age develops only through the refitted tail factor.

A refit can fail inside a simulation (a zero value under a simple
average, a tail curve that cannot be fitted). The simulation still runs to
the end, and the call returns `Error::OneYear { failed, n_sims, source }`
with the number that failed and, as `source`, the failure whose message
sorts first, so the error does not depend on the threads.

Checks. No published one-year standard deviation of the ODP bootstrap
was found, so the independent checks are these:

* The origin with one cell left has a one-year view equal to its run-off,
  so its standard deviation matches R `BootChainLadder`'s for that origin
  within the Monte Carlo tolerance (RAA, GenIns, ABC), and every other
  origin's is below its lifetime one.
* The re-reserving (append the diagonal, refit, `U0 - U1`) is checked with
  Mack's process instead of the ODP's: England, Verrall and Wüthrich's
  (2019) bootstrap of Mack's model (their Appendix 1), fed through the
  same re-reserving in a unit test, reproduces Merz–Wüthrich on GenIns per
  origin and in total within five Monte Carlo standard errors (measured
  within 0.6%). That harness is now `MackBootstrap` (below), which the
  test shows draws the same values bit for bit. Their simulated one-year
  view (Table 4) bootstraps Mack's model, not the ODP, so it is a
  reference for `MackBootstrap`, not for this method; their analytic
  Table 2 (Mack and Merz–Wüthrich on Taylor–Ashe, Mack's rule for the last
  sigma) is checked.
* A triangle that lies exactly on its chain-ladder pattern has scale zero
  and a CDR of zero in every simulation.

On the volume-weighted chain ladder the ODP's standard deviation is not
Merz–Wüthrich's: the ODP's variance is `phi` times the mean, Mack's
`sigma_k^2` times the cumulative value. The ratio, per origin 0.50 to 5.96
and in total RAA 0.62, GenIns 1.37, ABC 1.14 (measured with the
Marsaglia–Tsang Gamma sampler; by inverse transform, before 2026-10-08,
0.61, 1.36 and 1.13), is recorded in
`knowledge/findings/one-year-bootstrap-vs-merz-wuthrich.md`, and a
seed-pinned regression test holds each standard deviation to R's value
times that ratio within five Monte Carlo standard errors. It pins this
implementation's output; it is not a check against Merz–Wüthrich.

Bindings: Python `OdpBootstrap.one_year(triangle, column, method,
exposure=None)`, `method` a `ChainLadder`, `ExpectedLoss`,
`BornhuetterFerguson`, `Benktander` or `CapeCod` (exposure required for the
last four), returns a `OneYearFit` over every segment, with `cdr` the
`PredictiveDistribution`, the bootstrap's `development`, `fitted`,
`residuals` and `scale`, and `to_frame()`, `totals_frame()`,
`development_frame()` and `segment()` as the other fits. R has no method objects (its fitting functions
fit at once), so the method is named by a string, next to
`odp_bootstrap()`: `odp_one_year(triangle, column = NULL, method =
c("chain_ladder", "expected_loss", "bornhuetter_ferguson", "benktander",
"cape_cod"), exposure = NULL, apriori = 1, n_iters = 1, trend = 0,
decay = 1, average = "volume", sigma_interpolation = "log-linear",
tail = 1, n_sims = 10000, seed = 0, process = c("gamma", "none"))`, with
the settings of `chain_ladder()`, `expected_loss()`,
`bornhuetter_ferguson()`, `benktander()` and `cape_cod()`. Giving a
setting the method does not read (`apriori` to the chain ladder, say) is
an error rather than ignored. It returns a `one_year_fit` with `cdr` a
`predictive_distribution`, and `as.data.frame()`, `totals_frame()`,
`development_frame()` and `segment()` as the other fits.

#### Mack's process

So that the simulated view can be reconciled with Merz–Wüthrich, Mack's
process is a second bootstrap model beside the ODP, England, Verrall and
Wüthrich's (2019) bootstrap of Mack's model (their Appendix 1), fed through
the same re-reserving:

```rust
pub struct MackBootstrap {
    pub n_sims: usize,
    pub seed: u64,
    pub process: MackProcess,       // Gamma (default), Lognormal, Residuals, Normal, None
    pub development: Development,   // Mack's alpha and how a lone sigma is filled
    pub centre_residuals: bool,     // subtract the pool's mean first (default); false: EVW as written
}

impl MackBootstrap {
    pub fn one_year(&self, triangle: &Triangle, column: &str, method: &OneYearMethod)
        -> Result<OneYearFit<MackBootstrapSegment>>;
    pub fn one_year_segments(&self, triangle: &Triangle, column: &str, method: &OneYearMethod)
        -> Result<OneYearFits<MackBootstrapSegment>>;
}

pub struct MackBootstrapSegment {
    pub mack: MackFit,         // the model on the observed triangle (no tail here; decision 9)
    pub residuals: Vec<f64>,   // link-ratio residuals, origin x development
}

pub struct OneYearFit<B = OdpBootstrapSegment> { pub bootstrap: B, /* as above */ }
// likewise OneYearSegment<B> and OneYearFits<B>
```

A sibling type, not an option on `OdpBootstrap`: the two models share
nothing but the re-reserving. The ODP resamples Pearson residuals of the
incremental values into a pseudo triangle, has one scale `phi` and projects
from the pseudo latest value; Mack's resamples residuals of the link ratios,
has one `sigma_k` per factor, its own averaging (`alpha`), and is
conditional on the observed latest value. Their settings differ (Mack's
process has five shapes, the ODP's two) and so do their fits (`scale` and
`fitted` against `MackFit`). A field on `OdpBootstrap` would also break
every struct literal of it, and a model named for the ODP that runs Mack's
would mislead. The re-reserving is generic over the model's fit (a private
`NextYear` trait that draws the cells of the coming year), and `OneYearFit`,
`OneYearSegment` and `OneYearFits` take the fit as a type parameter that
defaults to `OdpBootstrapSegment`, so existing code compiles unchanged. The
ODP path's draws, input hashes and tables are bit-identical to before the
change (checked by hashing every draw on RAA, GenIns and ABC, both
processes, two methods, both entry points).

Per simulation, after EVW's Appendix 1, generalized to Mack's `alpha` (the
conditional variance of `C_k+1` is `sigma_k^2 C_k^(2 - alpha)`):

1. Resample the scaled bias-adjusted residuals of the link ratios,
   `r = sqrt(n_k / (n_k - 1)) C_k^(alpha / 2) (F - f_k) / sigma_k`, pooled
   over the factors with two or more link ratios, into a pseudo ratio
   `F* = f_k + r* sigma_k / C_k^(alpha / 2)` for every observed link, and
   take the pseudo factor `f*_k = sum(C_k^alpha F*) / sum(C_k^alpha)` with
   the observed weights. A link from a zero has no variance in Mack's model:
   it keeps its observed later value and gives no residual. Nor does a
   factor whose sigma is zero (its link ratios all equal, as in a split
   triangle's first year): its residuals would be `0 / 0`, zeros in their
   place would shrink the pool, and its pseudo factor is its factor. With
   every sigma zero nothing is resampled.
2. Draw every cumulative value of the coming year (above), each from the
   one before `C`, the observed latest value for the first, mean
   `f*_k C`, variance `sigma_k^2 |C|^(2 - alpha)`, with the same pseudo
   factors all year and the observed triangle's sigmas (those of a single
   link ratio interpolated as `Mack` does). On an annual triangle that is
   one draw per origin, from its observed latest value, EVW's step 7.
   `MackProcess` gives the shape: Gamma or lognormal (EVW's
   parametric choices; a negative mean, a pseudo factor below zero, gets
   the distribution of its absolute value negated, as the ODP's Gamma
   does), `Residuals` (the mean plus a resampled residual times the
   standard deviation, EVW's non-parametric choice), `Normal` (Mack's model
   is distribution-free; not in EVW) or `None`. Gamma, lognormal and normal
   have exactly that mean and variance; `Residuals` has the pool's moments,
   mean `f*_k C + m sd` and variance `(1 - m^2) sd^2` with `m` the pool's
   mean (below).
3. Append, refit and record the CDR as in steps 3 and 4 above.

The residuals of each factor have a zero `C_k^(alpha / 2)`-weighted sum,
not a zero mean, so the pool's mean `m` is not zero: 0.14 on RAA, 0.01 on
GenIns, -0.06 on ABC (its mean square is 1). Resampled as they are, they
bias every pseudo factor, `E[f*_k] = f_k + m sigma_k sum(C_k^(alpha / 2))
/ sum(C_k^alpha)`, and so the CDR, whose expectation under Mack's model is
zero: at 20,000 simulations its mean is -0.214 (RAA), -0.038 (GenIns) and
+0.176 (ABC) times its standard deviation, which shifts every quantile,
the 99.5% value at risk included; the lifetime view's mean reserve is
about +17% (RAA), +0.7% (GenIns) and -0.8% (ABC) off the chain ladder's;
and RAA's one-year standard deviations are up to 1.3% above Merz and
Wüthrich's (below). `centre_residuals: true`, the default, subtracts `m`
from the pool before resampling, for the pseudo factors and the
`Residuals` process. Centred, the mean CDR is -0.011, -0.003 and +0.007
times its standard deviation at the same seed, within Monte Carlo error
of zero, and the standard deviation reconciles with Merz and Wüthrich's
on every origin and total. EVW's Appendix 1 does not mention centring,
but their Table 4 lifetime expected reserves on Taylor–Ashe (GenIns, pool
mean 0.01) agree with the centred bootstrap and not the uncentred one
(the lifetime view, below), and their one-year standard deviations agree
with either (GenIns's pool mean is small). So the default centres, and
`centre_residuals: false` keeps EVW's Appendix 1 as written, for
comparison.

The model has no tail. Mack's tail is one more step from the oldest age to
ultimate with its own sigma and standard error; it has no calendar year,
so no coming year holds it (R's `CDR` rejects a tail for the same reason).
As with the ODP, development past the oldest age moves only through the
refitted method's tail, and an origin at the last age gets no new cell.
Mack's model needs no negative cumulative value and, like the ODP, every
origin observed from the first age to its latest.

Checks, all independent of the simulation:

* With the volume-weighted chain ladder and no tail, each origin's and the
  total standard deviation of the CDR on RAA, GenIns and ABC, with either
  rule for the last sigma, is R ChainLadder's `CDR(1)S.E.` within five
  Monte Carlo standard errors of the simulated standard deviation,
  `sd sqrt((kurtosis - 1) / (4 n))`, at 20,000 simulations
  (`validation/tests/reserving_one_year_mack.rs`), with the default,
  centred residuals. This is the reconciliation, of the standard
  deviation; the same test pins the centred mean CDR above as a
  seed-pinned regression, which a second test checks is within three
  Monte Carlo standard errors of the exact zero.
* England, Verrall and Wüthrich's Table 4 (500,000 simulations of the
  same bootstrap on Taylor–Ashe, Mack's rule for the last sigma): every
  origin and the total within five standard errors of the two simulations
  combined, centred (and, before centring became the default, uncentred).
* Unit tests: each factor's squared residuals sum to `n_k` for every
  `alpha`; RAA's 1982, one cell left behind a factor resting on one link
  ratio, has the variance worked out by hand for every `alpha`, which is
  Mack's own when the residuals' variance is 1; every process shape gives
  Merz–Wüthrich's GenIns total; on RAA the uncentred mean CDR is far below
  zero, and centred, the Gamma's and the `Residuals` process's means are
  within four Monte Carlo standard errors of zero; the uncentred bootstrap
  draws bit for bit as a hand-written harness of EVW's Appendix 1.

Uncentred and measured with 200,000 simulations, GenIns and ABC are within
0.4% of R per origin, but RAA's three youngest origins and its total come
out 0.4% to 1.3% above it, beyond Monte Carlo error; the
20,000-simulation test, whose five standard errors are 2.5% to 5%, does
not resolve it. The cause is the uncentred residuals, not the closed form,
and it is one of the reasons the default centres. On an annual triangle with
the volume-weighted chain ladder and no tail, each origin's closing
ultimate is a product of independent factors, each linear in one origin's
next value, so the CDR's covariance under the bootstrap's own model has a
closed form (`exact_covariance` in the validation test): it needs only
each pseudo factor's mean and variance and the process variance, not the
process shape. Linearised, with Merz and Wüthrich's factor moments
(`f_k`, `sigma_k^2 / S_k`), it is R's `CDR(1)S.E.` to rounding; exact, it
is at most 0.09% above (their Appendix A replaces products by sums), so
the approximation is not the gap. With the bootstrap's moments, mean
`f_k + m sigma_k sum(sqrt(C)) / S_k` and variance
`(1 - m^2) sigma_k^2 / S_k`, RAA's 1988 to 1990 and total are 0.48%,
0.93%, 1.20% and 1.29% above R (the first pseudo factor is 14% high) and
the 200,000 simulations are within 1.4 Monte Carlo standard errors of
that (Gamma on two seeds, normal on one); centred, every origin and total
of the three triangles is between 0.46% below and 0.04% above R (the
centred pseudo factors keep the variance `(1 - m^2) sigma_k^2 / S_k`:
centring shifts the pool without rescaling it). Over
every run and triangle the simulations are within 2.7 standard errors of
the exact values; only two runs per triangle (Gamma, two seeds) are
independent, the others share their random numbers. A fast test pins the
exact standard deviations, centred and uncentred, and the exact mean CDR
(-0.204, -0.034 and +0.168 times the SD uncentred, zero centred); an
ignored one checks the 200,000 simulations against them within four
standard errors.

Bindings: Python `MackBootstrap(n_sims=10000, seed=0, process="gamma",
average="volume", sigma_interpolation="log-linear",
centre_residuals=True).one_year(triangle, column, method, exposure=None)`
returns the same `OneYearFit` with `model == "mack"`, `mack` (the
`MackFit` of every segment), Mack's `residuals`, and no `fitted` or
`scale` (an error). R `mack_one_year(...)`, beside `odp_one_year()` with
the same arguments, `process = c("gamma", "lognormal", "residuals",
"normal", "none")`, `mack_average` and `mack_sigma_interpolation` for
Mack's model (the method's `average` and `sigma_interpolation` are
taken), and `centre_residuals = TRUE`, returns a `one_year_fit` with
`model == "mack"` and `mack` a `mack_fit`.

#### Mack's lifetime view

Beside the one-year view, `MackBootstrap` gives the lifetime view, as
`OdpBootstrap::fit` does for the ODP: EVW's Appendix 1, steps 7(a) to (g),
which is the bootstrap their Table 4 reports.

```rust
impl MackBootstrap {
    pub fn fit(&self, triangle: &Triangle, column: &str) -> Result<MackBootstrapFit>;
    pub fn fit_segments(&self, triangle: &Triangle, column: &str) -> Result<MackBootstrapFits>;
}

pub struct MackBootstrapFit {
    pub mack: MackFit,                  // the model on the observed triangle, with the tail (decision 9)
    pub residuals: Vec<f64>,            // link-ratio residuals, origin x development
    pub reserves: PredictiveDistribution, // dimension origin
}

pub struct MackBootstrapFits {
    pub segments: SegmentFits<MackBootstrapSegment>,
    pub reserves: PredictiveDistribution, // dimensions: the keys, then origin
}
```

The names follow the ODP's: `fit` and `fit_segments` for the lifetime
view, `one_year` and `one_year_segments` for the one-year view, on the
same settings. `MackBootstrapFits` has the ODP's `to_long`, `totals`,
`development_table` and `segment`: the chain ladder's `latest`, `ultimate`
and `reserve` (Mack's averaging) and the `mean` and `std_dev` of the
simulated reserve, without a `scale`.

Per simulation, on its own stream: the pseudo factors of step 1 above,
then every future cumulative value of every origin to the triangle's last
age, each from the one before (the observed latest value for the first),
mean `f*_k C`, variance `sigma_k^2 |C|^(2 - alpha)`, with the same pseudo
factors throughout; the reserve is the last value less the latest. The
draws are the one-year view's machinery (`MackDraw`): the same pseudo
factors from the same stream, then the cells in origin order, so an origin
with one cell left draws the same value in both views (a unit test checks
that its lifetime reserve plus its one-year CDR is its opening reserve in
every simulation). Any development grain works, and an origin short of the
latest diagonal develops from its own latest cell.

No tail, as first built: a tail factor is not an average of link
ratios, so no residual gives its parameter error, and EVW give no
distribution for it (Mack's tail standard error is an extrapolation, not
an estimate), so the reserves ran to the triangle's oldest age. Decision 9
adds one: an estimated tail refitted on each simulation's pseudo factors,
a constant one drawn from the lognormal with Mack's standard error, and
the step to ultimate with Mack's tail sigma.

The mean. Centred (the default), the pseudo factors are unbiased and
independent of each other and of the cell they multiply, so the mean
reserve is the chain ladder's. Uncentred (EVW's Appendix 1 as written,
`centre_residuals: false`), the pool's
mean biases every pseudo factor and the bias compounds over an origin's
remaining factors: the total mean reserve is about 1.17 (RAA), 1.007
(GenIns) and 0.992 (ABC) times the chain ladder's, and RAA's standard
deviation 1.09 times Mack's, because it grows with the mean (50,000
simulations). EVW's Table 4 expected reserves on Taylor–Ashe (GenIns,
500,000 simulations) are within Monte Carlo error of the chain ladder's
(+0.02% in total): they agree with the centred bootstrap, not the
uncentred one, which comes out eleven combined standard errors above them.
That is an inference about their implementation, not something they
state; EVW's Appendix 1 does not mention centring. It is the main reason
the default centres: with it the bootstrap matches EVW's Table 4 and the
chain ladder in the mean and Mack and Merz–Wüthrich in the standard
deviation, and `centre_residuals: false` remains for the Appendix as
written.

The standard deviation. Mack's standard error is the first-order (linear)
approximation of the bootstrap's. One difference is exact: the resampled
residuals have the pool's variance `v = 1 - m^2`, not 1, so every pseudo
factor's variance, and with it every parameter error, is `v` times Mack's
(RAA 0.981, GenIns 0.9998, ABC 0.996). Checks
(`validation/tests/reserving_mack_bootstrap.rs`, 20,000 simulations):

* Centred (the default), every origin's and the total standard deviation is
  `sqrt(process^2 + v parameter^2)` from R ChainLadder's
  `MackChainLadder` process and parameter risks within five Monte Carlo
  standard errors of the simulated standard deviation, and the mean within
  five standard errors of the mean of the chain ladder's reserve: every
  dataset under both rules for the last sigma, Gamma process. At 50,000
  simulations (Gamma by inverse transform, before 2026-10-08, centred,
  either rule) the standard deviation is 0.991 to 1.007 times Mack's plain
  standard error per origin and 0.994 to 1.002 in total, every origin
  within three Monte Carlo standard errors.
* Parameter error alone (`MackProcess::None`) is `sqrt(v)` times R's
  parameter risk, every origin and the total, both rules, all three
  datasets. Before the `v` adjustment RAA's came out 0.8% to 1.2% below
  R's at every origin but 1990 (2.7 to 3.8 standard errors at 50,000);
  after it, RAA 1990 is above, the linear approximation's neglected terms
  on its volatile young factors. Centred, the pseudo factors are
  independent with mean `f_k` and variance `v sigma_k^2 / S_k`, so the
  exact parameter variance is `C^2 (prod(f_k^2 + v sigma_k^2 / S_k) -
  prod f_k^2)` against Mack's linear `C^2 prod f_k^2 sum(v sigma_k^2 /
  (f_k^2 S_k))`: 1.0066 times `sqrt(v)` times Mack's for RAA 1990, 1.0022
  for 1989 and at most 1.0009 earlier (log-linear sigma; the linear form
  reproduces R's 7,275 for 1990). The 1.0% to 1.5% measured at 50,000
  simulations is within two Monte Carlo standard errors of that.
* EVW's Table 4 (Mack's rule, GenIns): every origin's and the total
  expected reserve and standard deviation within five standard errors of
  the two simulations combined, centred; uncentred, the total expected
  reserve is more than five above theirs.
* Unit tests: an exact pattern has a zero-variance run-off equal to the
  chain ladder's reserve; an origin with one cell left draws as in the
  one-year view; on RAA the uncentred mean is far above the chain ladder's
  and the centred one within four standard errors of it; parameter error
  alone and the one-year view are narrower than the lifetime view; draws
  do not depend on the seed's thread count, and `fit` equals
  `fit_segments` on a single segment.

The Gamma process inverted its cdf until 2026-10-08, which was slow at
the large shapes of late cells (ABC's run to the thousands): ABC's
lifetime view at 2,000 simulations took 30 s in a debug build, and the
validation test simulated ABC with the lognormal of the same mean and
variance. `Gamma::sample` is now Marsaglia and Tsang's sampler (see
`rng.md`, stability log): ABC's lifetime view at 20,000 simulations takes
0.02 s in a release build (39 s before), as the lognormal's does, and the
validation test uses the Gamma on all three triangles under both rules
for the last sigma (every origin within 2.4 Monte Carlo standard errors
of Mack's). The lognormal stays unsuitable for RAA: its young origins'
shapes are below one, where the lognormal's heavy tail makes the standard
deviation's own standard error unreliable (at 20,000 simulations RAA 1990
came out 6% low, 3.3 estimated standard errors; at 200,000, 0.8%). With
the weightings `alpha = 0` and `2` the centred
bootstrap also agrees with R's `MackChainLadder(alpha = ...)` on GenIns
and ABC (within 3.5 standard errors at 50,000 simulations), but RAA's youngest origins come out up to 7% above (`alpha = 0`,
whose variance is proportional to `C^2`, so the linear approximation
fails sooner) and two older ones 1.5% to 2% below (`alpha = 2`); measured
once, not in CI.

Bindings: Python `MackBootstrap(...).fit(triangle, column)` returns a
`MackBootstrapFit` over every segment, with `chain_ladder`, `mack` (a
`MackFit`), `keys`, `index`, `origins`, `development`, `residuals`,
`reserves`, `to_frame()`, `totals_frame()`, `development_frame()` and
`segment()`, as `OdpBootstrapFit`. R `mack_bootstrap(triangle, column =
NULL, n_sims = 10000, seed = 0, process = c("gamma", "lognormal",
"residuals", "normal", "none"), average = "volume", sigma_interpolation =
"log-linear", centre_residuals = TRUE)` returns a `mack_bootstrap_fit`
with `chain_ladder`, `mack`, `origins`, `development`, `residuals` and
`reserves`, and `as.data.frame()`, `totals_frame()`,
`development_frame()` and `segment()` as the other fits. The model's
settings are named as in `mack()`, since there is no method to tell them
from (`mack_one_year()`'s `mack_average` keeps them apart from the
method's).

### 9. Tails in the bootstraps' lifetime view

Until this decision `MackBootstrap::fit` had no tail (decision 8) and
`OdpBootstrap::fit` none either, as R's `BootChainLadder`: the simulated
reserves ran to the oldest age, so a reserve with a tail had no simulated
distribution, and neither the tail factor's parameter error nor the
process error of development past the oldest age was simulated. Both
lifetime views now take a tail:

```rust
pub struct OdpBootstrap {
    /* n_sims, seed, process, as before */
    pub tail: Tail,                 // default none
    pub tail_std_err: Option<f64>,  // a constant tail's standard error; None keeps it fixed
}

pub struct MackBootstrap {
    /* n_sims, seed, process, development, centre_residuals, as before */
    pub tail: Tail,                 // default none
    pub tail_sigma: Option<f64>,    // R's tail.sigma; None extrapolates, as Mack
    pub tail_std_err: Option<f64>,  // R's tail.se; None extrapolates, as Mack
}

pub enum Error { /* ... */ TailRefit { failed: usize, n_sims: usize, source: Box<Error> } }
```

The fields mirror `Mack`'s and `ChainLadder`'s `tail`, so the settings
read the same in every method. `Tail` holds floats, so neither bootstrap
derives `Eq` any more, and a struct literal that names every field needs
`..Default::default()`. Without a tail (a constant 1 at the oldest age,
whatever its decay and whatever sigma or standard error is given, which
are unused as in `Mack`) every draw, input hash and provenance is as
before: checked by hashing every draw of `fit`, `fit_segments` and
`one_year` under both ODP processes and all five of Mack's, centred and
not, on RAA, GenIns and ABC (108 hashes, identical before and after; not a
CI test, as in decision 8), and by unit tests that a constant 1 with
another decay draws as the default.

Per simulation, after the pseudo factors (the pseudo triangle's
volume-weighted factors for the ODP, the re-averaged pseudo link ratios
for Mack) and before any process draw:

1. **The tail's parameter error.**
   * An estimated tail (`Tail::Curve`, `Tail::Bondy`, `Tail::LogLinear`)
     is refitted on the simulation's pseudo factors, with the same
     settings and ages. That gives the factor to ultimate and the selected
     factors within the triangle: a tail attached before the oldest age
     replaces the pseudo factors from its attachment on, as it replaces
     the estimated ones on the observed triangle. No random number is
     drawn. The tail is a function of the estimated factors, so refitting
     it carries their parameter error through the tail estimator itself,
     its nonlinearity included, where Mack's (1999) `tail.se` is a
     log-linear extrapolation of the factors' standard errors, not the
     estimator's sampling error. A `tail_std_err` with an estimated tail
     would count that error twice and is an error.
   * A constant tail is drawn, when it has a standard error, from the
     lognormal with the factor as mean and the standard error as standard
     deviation, after the pseudo factors on the same stream and
     independent of them. For Mack the standard error is Mack's: the given
     `tail_std_err` or, as in `Mack`, extrapolated (R's `tail.se`), so the
     drawn factor has exactly the variance Mack's formula gives the tail.
     The ODP has no estimate of its own, so its constant tail is fixed
     unless `tail_std_err` is given. The lognormal is positive, as the
     factor is; it has exactly the mean and variance, so to first order
     the bootstrap reproduces Mack's parameter error; at the coefficient of
     variation a tail factor's standard error gives it (R's extrapolated
     `tail.se` on RAA: 1.8% at a tail of 1.05, 0.4% under `tail = TRUE`)
     it is close to the normal, as the pseudo factors of the triangle are;
     and it works for a factor below 1, which `Tail` allows. Rejected: the
     normal, which can go below zero at a large standard error; a
     lognormal or Gamma of `f - 1`, which keeps development past the oldest
     age positive but fails for a factor below 1 and is very skewed (RAA's
     `f - 1` has a coefficient of variation of 0.38 at 1.05 and 0.46 under
     `tail = TRUE`), a skew Mack's estimate gives no ground for.
   * A constant tail attached before the oldest age (`TailConstant` with an
     `attachment_age`) replaces the factors from its attachment on with
     its own, which do not depend on the pseudo factors, but `Mack` still
     charges those positions the estimated factors' standard errors. So
     each of those factors moves by its pseudo factor's deviation from the
     estimate, `f_tail,k + (f*_k - f_k)`: the ages keep their parameter
     error, with the variance Mack charges and the mean the selected
     factor's. Kept fixed, they gave the middle origins of RAA, GenIns and
     ABC standard deviations 17-30% below `Mack`'s with the same tail
     (Gamma process, 1.05 attached at the fourth-oldest age, 20,000
     simulations), the means agreeing. Rejected: rejecting an early
     attachment in the bootstraps (a `ChainLadder` and `Mack` setting they
     should take as well), and zeroing those standard errors in `Mack`
     (a selected factor is still uncertain; `Mack`'s behaviour, not this
     decision's). An estimated tail attached early needs neither: its
     factors are refitted on the pseudo factors.
2. **The process error of development past the oldest age.**
   * Mack: every origin, the oldest included, takes one more step after
     the oldest age, its ultimate drawn by `MackProcess` with mean
     `T* C` and variance `tail_sigma^2 |C|^(2 - alpha)`, `T*` the
     simulation's tail factor and `C` its value at the oldest age: Mack's
     (1999) process variance of the tail step, the term his recursion
     adds. `tail_sigma` is the observed fit's (given, or extrapolated as
     R's `tail.sigma`), as every sigma of the bootstrap is the observed
     triangle's.
   * ODP: the step to ultimate is one more future increment, with mean
     `C* (T* - 1)`, `C*` the pseudo value at the oldest age (the pseudo
     latest value carried forward on the pseudo factors, as every
     projected increment is), and the process error of every other
     increment, Gamma with variance `phi |mean|`. The ODP's process
     variance is the scale times the mean of a future increment, and
     development past the oldest age is a future increment like the
     others; with no observed cell beyond the oldest age there is nothing
     to estimate a scale of its own from, and no process error at all
     would leave out the least known part of the run-off.
3. **Failures.** A refit can fail on a simulation's pseudo factors (a
   curve left with fewer than two factors above 1.00001 in its fit period,
   a Bondy exponent outside (0, 1), a log-linear tail without two factors
   above 1). The simulation's row is zeroed, every simulation still runs,
   and the call returns `Error::TailRefit` with the number that failed and
   the failure whose message sorts first, as the one-year view's
   `Error::OneYear` does, so the error does not depend on the threads.
4. **The one-year view is unchanged.** `one_year` and `one_year_segments`
   of both bootstraps reject a bootstrap tail (`Error::Bootstrap`): as
   decision 8 says, Mack's tail is a step without a calendar year, so no
   coming year holds it (R's `CDR` rejects a tail too), and development
   past the oldest age moves only through the refitted method's own tail.
5. **What the fits report.** `OdpBootstrapFit::chain_ladder` and
   `MackBootstrapFit::mack` are fitted with the bootstrap's tail (ultimate,
   reserve, and Mack's standard errors with the tail), so `to_long` and
   `totals` compare the simulated reserves with the tailed chain ladder.
   The ODP's fitted values and residuals do not change: they come from the
   estimated factors, which a tail does not touch. The provenance records
   the tail's settings when there is a tail and nothing new without one.

Checks (`validation/tests/reserving_bootstrap_tail.rs`, 20,000
simulations, against `reserving_tails_r.csv`, R ChainLadder 0.2.21's
`MackChainLadder(tri, tail = ...)` on RAA, GenIns and ABC):

* **A constant tail reconciles with Mack.** With the Gamma process, each
  origin's and the total standard deviation is R's `Mack.S.E` with the
  tail within five Monte Carlo standard errors, and the mean R's reserve
  within five standard errors of the mean: `tail = 1.05` with the tail's
  sigma and standard error extrapolated (either rule for the last sigma)
  or given (`tail.se` half the last factor's, `tail.sigma` twice its
  sigma), and `tail = TRUE` too when the bootstrap's tail is the constant
  R's rule fits, since R's `tail.se` for `tail = TRUE` is the
  extrapolation at that factor, as a constant tail's is (the test checks
  the tail's sigma and standard error are R's to `1e-8`). As without a
  tail, the reference parameter variance of the factors within the
  triangle is scaled by the resampled residuals' variance `v`; the tail
  factor's draw has Mack's variance exactly, so the reference is
  `process^2 + v parameter^2 + (1 - v) (C se)^2`, `C` the projected value
  at the oldest age. Parameter error alone (`MackProcess::None`) is
  checked the same way. Measured at seed 20,261,008: every origin and
  total within 3.2 standard errors, the largest RAA 1990 and RAA's total
  (1.04 times R under the Gamma, parameter error alone 1.004), the
  young-origin nonlinearity the run-off without a tail shows too.
* **A constant tail attached early reconciles with `Mack`.** A 1.05
  attached at the fourth-oldest age, against prospicio's `Mack` with the
  same tail (R's `MackChainLadder` has no attachment age; `Mack`'s tail
  terms are R's by the check above), the same reference formula, Gamma and
  parameter error alone: every origin and total within 3.1 standard
  errors in standard deviation and 3.2 in mean, the largest again RAA 1990
  and RAA's total under the Gamma.
* **A refitted tail has Mack's process error.** Run with the Gamma and
  with no process on the same seed, the two share every pseudo factor and
  refitted tail, since the process draws come after them, so the
  per-simulation difference of their reserves is the process error alone:
  its standard deviation is R's `Mack.ProcessRisk` with `tail = TRUE`
  within five standard errors for every origin and total, both rules for
  the last sigma (measured within 2.5).
* **A refitted tail's parameter error is the refit's.** On the oldest
  origin, whose reserve is the tail step alone, parameter error alone is
  `C (T* - 1)`; its standard deviation is within five standard errors of
  the first-order (delta-method) error of R's rule, `C sqrt(sum (dT/df_k)^2
  v se_k^2)` with the gradient by central differences, on ABC (0.998
  times), and above it on RAA (1.56) and GenIns (1.12), where the
  extrapolation runs far past the data and the refitted tail is convex in
  the factors: its mean is above the plug-in tail too (RAA's oldest
  reserve 200 against 178, +13%; GenIns -2%, ABC -0.1%). The first-order
  standard error of the refitted tail factor is of the order of R's
  extrapolated `tail.se` (RAA 0.0051 against 0.0044, GenIns 0.0099
  against 0.0083, ABC 0.00068 against 0.00089; the test checks they are
  within a factor of 2). In total the refitted bootstrap's standard
  deviation is 1.003 (RAA), 1.072 (GenIns) and 1.011 (ABC) times R's
  `Mack.S.E` with `tail = TRUE` (measured once, not in CI). So a user who
  wants R's `tail = TRUE` standard error gives the constant R's rule fits;
  the estimated tail gives the tail estimator's own error.
* **Unit tests** (`bootstrap.rs`, `mack_bootstrap.rs`): with a constant
  tail and no process error, each simulation's reserve of the oldest and
  the next origin is rebuilt bit for bit from the pseudo triangle and the
  lognormal's draw on the same stream; an estimated tail's refit is
  rebuilt bit for bit from the pseudo factors of the same stream, under
  both models; RAA's oldest origin with `tail = 1.05`, `tail_sigma = 1.5`
  and `tail_std_err = 0.003` has Mack's exact variance
  `C^2 0.003^2 + 1.5^2 C` and mean `0.05 C` within five Monte Carlo
  standard errors; a constant tail attached at 84 months keeps RAA's
  pseudo factors before 84 and moves its own by the pseudo factors'
  deviations, exactly; on a keyed triangle of two segments each segment's
  oldest origin takes the tail (`fit_segments`, both models), and a
  refit that fails in one segment is `TailRefit` wrapping `InSegment`
  with its label; the ODP's oldest origin with a constant tail has the
  variance `(T - 1)^2 v sum|m| + phi (T - 1) E[P]` worked out from its
  fitted increments `m` and pseudo latest value `P`; a curve fitted to two
  factors barely above 1 fails in some simulations and is reported with
  the count, under both models; a standard error with an estimated tail,
  a negative or NaN one, and a tail in the one-year view are errors.

The ODP has no reference with a tail (R's `BootChainLadder` has none);
its checks are the unit tests above. Measured once (20,000 simulations,
seed 20,261,008, not in CI): with a constant 1.05 the total mean is the
tailed chain ladder's times about the same ratio as without a tail (RAA
1.029 against 1.032, GenIns 1.010 against 1.011, ABC 0.999 both: the ODP
bootstrap's own bias, which R's `BootChainLadder` shares) and the
coefficient of variation falls (RAA 0.350 to 0.310), the tail adding a
nearly fixed share; a standard error of 0.01 widens ABC's total standard
deviation by 31% and RAA's by 0.4%. A refitted log-linear tail moves much
more on the ODP's pseudo factors than on Mack's, whose late link ratios
are steadier: RAA's oldest origin has a mean reserve of 316 against the
plug-in 178 (+78%) and a parameter standard deviation of 249, and the
total mean is 1.064 times the tailed chain ladder's (1.033 without a
tail); GenIns 1.017 (1.011), ABC 0.999 (0.999). On RAA that is R's rule
refitted where its guards, written for one estimate, decide: 1,136 of the
20,000 simulations (5.7%) have the product of the third- and second-last
pseudo factors at most 1.0001 and so exactly no tail (the oldest origin's
reserve exactly 0), and 9,790 of the others leave a pseudo factor at or
below 1 out of the line, with a mean `T* - 1` of 0.0211 against 0.0153
for the rest and 0.0094 plug-in. The point mass pulls the mean down (the
oldest origin's mean is 336 without it); the selection and the
extrapolation's convexity push it up. GenIns and ABC have no simulation
without a tail, and Mack's pseudo factors on RAA none either. The refit is
kept, as R's rule is what `Tail::LogLinear` means, and documented on
`OdpBootstrap::tail`: a constant tail with a `tail_std_err` is the one to
use on the ODP for a tail of a stable form.

Bindings: Python `OdpBootstrap(n_sims=10000, seed=0, process="gamma",
tail=None, tail_std_err=None)` and `MackBootstrap(..., centre_residuals=True,
tail=None, tail_sigma=None, tail_std_err=None)`, `tail` a float or a
`TailConstant`, `TailCurve`, `TailBondy` or `TailLogLinear` as
`ChainLadder`'s and `Mack`'s, with getters of the same names; their
`repr` shows the tail (`tail=1.0` without one, as `Mack`'s). R
`odp_bootstrap(..., process = c("gamma", "none"), tail = 1, tail_std_err =
NULL)` and `mack_bootstrap(..., centre_residuals = TRUE, tail = 1,
tail_sigma = NULL, tail_std_err = NULL)`, `tail` a number or a tail
constructor as `mack()`'s. The one-year functions are unchanged: their
`tail` is the refitted method's. `tail_std_err` left out means different
things in the two, and each binding's parameter documentation says so:
the ODP keeps a constant tail fixed, while Mack's bootstrap extrapolates
Mack's standard error, so changing only the bootstrap changes whether a
constant tail has parameter error. The ODP's chain ladder computes the
same extrapolation, but the ODP's model has no tail standard error of its
own, so the default stays fixed. The mean is the chain ladder's (Mack,
centred residuals) only without a tail or with a constant one; the
bindings' documentation qualifies it the same way.
