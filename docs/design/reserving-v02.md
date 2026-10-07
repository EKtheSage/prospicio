# Design note: Reserving v0.2 (expected-loss methods, tails, Clark, one-year view)

Status: **Proposed** · Lane: Reserving · Depends on: `triangle.md`, `docs/architecture.md` (v0.2 row of the roadmap)

## Goal

Broaden `act-reserving` past the chain ladder: the expected-loss family
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
| 1 | `claude/act-math-nelder-mead` | `act_math::optimize::nelder_mead` (Probability lane's crate, with the user's go-ahead) | `main` |
| 2 | `claude/v02-expected-loss` | `ExpectedLoss`, `BornhuetterFerguson`, `Benktander`, `CapeCod` | PR 0 |
| 3 | `claude/v02-tails` | `Tail` and its estimators, Mack with a tail | PR 0 |
| 4 | `claude/v02-one-year` | Merz–Wüthrich claims development result | PR 0 |
| 5 | `claude/v02-clark` | `ClarkLdf`, `ClarkCapeCod` | PRs 1 and 2 |

PRs 2–4 touch the same binding files (`crates/act-python/src/reserving.rs`,
`crates/act-r/src/reserving.rs`, `R/prospicio/R/reserving.R`) and the
generated stub, `NAMESPACE` and `man/`. Whichever merges second merges
`main` and regenerates the generated files; it does not merge them by hand.

## References

| Method | Primary reference | Secondary |
|---|---|---|
| Expected loss, BF, Benktander, Cape Cod | chainladder-python 0.10.1 (`ExpectedLoss`, `BornhuetterFerguson`, `Benktander`, `CapeCod`) | hand-computed cases |
| Tails | chainladder-python (`TailConstant`, `TailCurve`, `TailBondy`); R ChainLadder 0.2.21 `MackChainLadder(tail = TRUE / number)` for Mack's tail sigma and standard error | — |
| Clark | R ChainLadder `ClarkLDF`, `ClarkCapeCod` | chainladder-python `ClarkLDF` |
| One-year view | R ChainLadder `CDR(MackChainLadder(...))`, `dev = "all"` for the full run-off | Merz and Wüthrich (2008), published example |
| Simulated one-year view | R ChainLadder `CDR(MackChainLadder(...))` times the measured ODP-to-Mack ratio; `BootChainLadder` for the origin with one cell left | England, Verrall and Wüthrich (2019), Table 2 |

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

As implemented (`crates/act-reserving/src/expected_loss.rs`, parity in
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
act-reserving's matches Mack.

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
`act_math::optimize::nelder_mead` on `(ln omega, ln theta)`. The parameter
covariance is the inverse Fisher information times the scale, from analytic
derivatives of the curve, as in Clark (2003). The fit reports `omega`,
`theta`, the scale, ultimates, reserves, process, parameter and total
standard errors per origin and in total, and implements `ReserveFit`.

As implemented (`crates/act-reserving/src/clark.rs`, parity in
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
  as with exposure in smaller units than the losses, act_reserving returns
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
2. Simulate the next calendar diagonal: each origin's next incremental has
   mean `C*_latest * (f*_k - 1)`, from the pseudo triangle's latest value
   `C*_latest`, and the bootstrap's process error with scale `phi`. That
   is how the lifetime ODP bootstrap projects (England 2002), and the
   pseudo latest value carries the estimation error of the origin's level,
   part of the ODP's parameter error: projecting from the observed latest
   value would leave it out (RAA's total standard deviation 30% lower). So
   an origin with one cell left has a one-year view distributed as its
   lifetime bootstrap reserve. An origin already at the triangle's last
   age gets no new cell.
3. Add each increment to the origin's observed latest value, append that
   diagonal to the observed triangle (exposure columns carry each origin's
   latest value forward) and refit `method` on it. Cape Cod trends to the
   valuation a year later.
4. `CDR_i = U0_i - U1_i`, the opening ultimate less the re-estimated one,
   which equals the opening reserve less the year's simulated payment and
   the closing reserve.

The development grain must be a year, so that one development period is
the coming year (a quarterly triangle would otherwise give a one-quarter
CDR), and every origin short of the last age must have its latest value on
its segment's latest diagonal (segments may end on different diagonals),
so that its next cell is in the coming year; anything else is
`Error::Bootstrap`. Simulating several cells per origin within the year,
or catching up a lagging origin, is left for later.

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
  within 0.6%). Their simulated one-year view (Table 4) bootstraps Mack's
  model, not the ODP, so it is not a reference for this method; their
  analytic Table 2 (Mack and Merz–Wüthrich on Taylor–Ashe, Mack's rule for
  the last sigma) is checked.
* A triangle that lies exactly on its chain-ladder pattern has scale zero
  and a CDR of zero in every simulation.

On the volume-weighted chain ladder the ODP's standard deviation is not
Merz–Wüthrich's: the ODP's variance is `phi` times the mean, Mack's
`sigma_k^2` times the cumulative value. The ratio, per origin 0.50 to 5.98
and in total RAA 0.61, GenIns 1.36, ABC 1.13, is recorded in
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
