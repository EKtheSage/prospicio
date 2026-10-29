# Design note: Reserving v0.2 (expected-loss methods, tails, Clark, one-year view, dependence between lines)

Status: **Implemented** (PRs #131–#137, #150, #156, #157, #159, #160, #163, #171 for decision 10, #172 for decision 9; decision 11 on branch `one-year-tail`; decision 12 built on branch `claude/selected-patterns`, in Rust, Python and R; decision 13 built on branch `claude/development-averages`, in Rust, Python and R; decision 14 built on branch `claude/bootstrap-exclusions`, in Rust, Python and R; decision 15 built on branch `claude/diagonal-pr1`, in Rust; decision 16 built on branch `claude/diagonal-pr2`, in Rust; decision 17 built on branch `claude/diagonal-pr3`, in Rust; decision 18 built on branch `claude/diagonal-basis`, in Rust; decisions 15 to 18 in Python and R and decision 19 built on branch `claude/diagonal-bindings`, in Rust, Python and R; decision 20 built on branch `claude/one-value-pattern`, in Rust, Python and R; decision 21 built on branch `claude/uy-pattern`, in Rust, Python and R; decision 22 built on branch `claude/apriori-by-origin`, in Rust, Python and R; decision 23 built on branch `claude/benchmark-curve`, in Rust, Python and R; decision 24 built on branch `claude/ulr-column`, in Rust, Python and R, its item 4 follow-up `ulr_mean` on branch `claude/ulr-mean`, in Rust, Python and R; decisions 20 to 24 accepted by the user on 2026-10-27; decision 25 accepted by the user on 2026-10-27 (replace, the Weibull the default) and built on branch `claude/ay-monthly-reading`, in Rust, Python and R; decision 26 built on branch `claude/loss-ratio-draws`, in Rust, Python and R, its item 5 accepted by the user on 2026-10-28 as proposed; decision 27 built on branch `claude/mack-assumption-tests`, in Rust, Python and R, its item 6 accepted by the user on 2026-10-28 as built) · Lane: Reserving · Depends on: `triangle.md`, `docs/architecture.md` (v0.2 row of the roadmap)

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
| Dependence between segments | Kirschner, Kerley and Isaacs (2008), Two approaches to calculating correlated reserve indications across multiple lines of business, Variance 2(1); Taylor and McGuire (2007), A synchronous bootstrap to account for dependencies between lines of business, NAAJ 11(3) | the residuals' own correlation; Iman and Conover (1982) through `prospicio_prob` |

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
  `10000000 + 400000 * (0:9)` on GenIns origins;
* `clrd_lines.csv`, for the dependence between lines (decision 10): the
  paid losses of all six lines of the CAS loss reserve database summed
  over companies (chainladder-python
  `load_sample('clrd').groupby('LOB').sum()`, `CumPaidLoss`), 1988-1997 at
  12 to 120 months, in a `lob` column; its wkcomp rows are
  `clrd_wkcomp.csv`'s `paid`.

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

The `apriori` of the first three is one loss ratio or one per origin since
decision 22 (`Apriori`). With `q = 1 / cdf` the share developed at the
origin's latest age and `U0 = apriori * exposure`:

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
    pub mack: MackFit,         // the model on the observed triangle, with the bootstrap's tail
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

*Superseded by decision 11 when the bootstrap has a tail.* Without one,
the model has no tail. Mack's tail is one more step from the oldest age to
ultimate with its own sigma and standard error; R's `CDR` rejects a tail.
As with the ODP, development past the oldest age then moves only through
the refitted method's tail, and an origin at the last age gets no new cell.
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
4. **The one-year view** (*superseded by decision 11*). As first built,
   `one_year` and `one_year_segments` of both bootstraps rejected a
   bootstrap tail (`Error::Bootstrap`), the tail being a step without a
   calendar year. Decision 11 gives it one and takes it in the coming
   year.

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
  the count, under both models; a standard error with an estimated tail
  and a negative or NaN one are errors, and so, in the one-year view, is
  a bootstrap tail that is not the method's (decision 11).

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

#### With dependence between segments

A tail combines with every `SegmentDependence` of decision 10 in
`fit_segments`; the one-year views still take no bootstrap tail.

* **Independent.** Each segment refits or draws its own tail, as a single
  segment does.
* **Synchronized.** Only the residuals' positions are shared, as
  Kirschner, Kerley and Isaacs share them. An estimated tail is refitted
  on each segment's own pseudo factors, which come from the shared
  positions, so its parameter error follows the shared residuals: two
  identical lines without process error keep equal reserves, tail
  included. A constant tail's lognormal draw is each segment's own, like
  its process error, after that segment's pseudo factors on the same
  stream. A shared draw (one normal quantile for every segment) would
  assert that the lines' tail judgments err together, which the residuals
  say nothing about; a rank correlation of the segments' totals is the
  way to ask for more dependence. On two copies of CLRD's wkcomp,
  10,000 simulations without process error, a constant 1.05 with
  `tail_std_err` 0.02 brings the copies' correlation from 1 to 0.19 (ODP)
  and 0.16 (Mack): the tail's draw dominates the reserve's parameter error
  there.
* **Rank correlation.** Each segment is simulated independently, tail
  included, and whole rows are reordered afterwards, so every component
  keeps the independent fit's draws and the segments' totals, tail
  included, take the target Spearman matrix (0.501 against 0.5 on comauto
  and wkcomp).

`validation/tests/reserving_dependence.rs` (`tails_with_dependence`) and
each binding's tests check the synchronized and rank-correlated cases.

### 10. Dependence between segments

A triangle with several segments (lines of business) is bootstrapped into
one joint distribution, components by segment and origin, so a total over
the lines, its quantiles and a capital allocation come from one set of
simulations. Until now the segments were independent: each resampled its
own residuals on the shared stream of simulation `i`. For capital the
lines' dependence matters, so it is now a choice,
`prospicio_reserving::SegmentDependence`, a field `dependence` on
`OdpBootstrap` and `MackBootstrap` next to `process`, used by
`fit_segments` and `one_year_segments` (lifetime and one-year views of
both bootstraps; the single-segment `fit` and `one_year` ignore it):

| Variant | What it does | Dependence from |
|---|---|---|
| `Independent` (default) | each segment resamples its own residuals | none |
| `Synchronized` | every segment resamples the residuals of the same positions | the data: the lines' paired residuals |
| `RankCorrelation { spearman }` | each segment bootstrapped independently, then whole simulations of each segment reordered by Iman–Conover on the segment totals | the user: a Spearman matrix |

A field, not a method argument, because it is a setting of the bootstrap
like `process`, the bindings take it as a constructor argument, and it
keeps one `fit_segments` per bootstrap. The rank-correlation matrix is a
`Vec<f64>`, so `OdpBootstrap` and `MackBootstrap` are no longer `Copy` or
`Eq` (still `Clone` and `PartialEq`); the bindings borrow them.

**Independent** draws are bit-identical to those before the field: a
checksum of the draws of `fit_segments` and `one_year_segments` of both
bootstraps (the Mack one with the Gamma and residual processes, centred and
not; three CLRD lines, 2,000 simulations) matched `main`'s on all ten
configurations, and no seed-pinned test moved.

**Synchronized** is the synchronous bootstrap of Taylor and McGuire (2007)
and the correlated bootstrap of Kirschner, Kerley and Isaacs (2008, section
4.5, Table 9: "the choice of which variability parameter to pick is the
same across lines"). Precisely: each simulation first draws, from its
stream, one position index for every residual a segment draws, in the
order the segment draws them (ODP: every observed cell, row-major; Mack:
every observed link ratio, factor by factor, then by origin), uniformly
from the positions where *every* segment has a residual, with the formula
the independent bootstrap uses. Each segment then takes its own residual
at each position and goes on drawing its process error from the same
stream, so process error is independent between segments, as the
residuals carry the dependence (Kirschner et al. resample only the
residuals in common). Segments must have the same origins, ages and
observed cells (`Error::Bootstrap` otherwise), so that a position is the
same origin and age, i.e. the same calendar period, in every line. When
every segment has a residual wherever any has (the usual case), each
segment's own distribution is exactly its independent one. With one
segment the draws are identical for the ODP, and for Mack's bootstrap
when no link has a zero cumulative value or a zero sigma (unit test, on
RAA); otherwise the synchronized Mack path still draws a position for
such a link, which the independent one skips, so the stream shifts and
the draws are equal in distribution only. A position that only some
segments have a residual at (a zero fitted value in one line, a zero
link) is left out of all of them, and Mack's centring is then over the
positions kept; the `Residuals` process resamples from the same shared
pool.

To first order a line's reserve is `sum_c a_c r(p_c)`, so two synchronized
lines' parameter error has correlation `rho a1.a2 / (|a1| |a2|)`, `rho` the
correlation of their paired residuals: at most `|rho|`, and close to it
when the lines develop alike. On the six CAS loss reserve database lines
(`validation/data/clrd_lines.csv`; 15 pairs, `rho` from -0.15 to 0.60,
10,000 simulations) the ODP's parameter-error correlation is within 0.055
of `rho`, Mack's within 0.145 (ppauto, which develops much faster, against
wkcomp: 0.18 against 0.32). Process error dilutes it (ODP comauto with
wkcomp: 0.53 parameter only, 0.40 with the Gamma process; Mack's process
error is larger, 0.49 and 0.17).

**RankCorrelation** is Kirschner et al.'s first approach, a rank
correlation of the simulated reserves by line to a user's matrix, with
`prospicio_prob::PredictiveDistribution::reorder_groups` (Iman and Conover
1982 on the group totals, moving each group's simulations as whole rows,
so each line keeps its distribution and its joint structure across
origins). `reorder_groups` groups by one dimension, so the reordering runs
on a copy whose components are keyed by segment number, which also covers
triangles with several keys, and the result takes back the fit's
dimensions and components. Iman–Conover sets the correlation of normal
scores, whose Spearman rho is `(6/pi) asin(r/2)`, so the Spearman matrix
the user gives is converted to `r = 2 sin(pi rho / 6)` (the diagonal kept
at exactly 1); a matrix that is not positive definite after conversion is
refused (`Error::Core`), and a matrix of the wrong size, without a unit
diagonal, asymmetric or out of `[-1, 1]` is `Error::InvalidSetting`.
Iman–Conover keeps the first group's column of target ranks in order (only
the others are shuffled before the rotation, which is lower triangular),
so its result has the first segment's simulations sorted by their total:
row `i` would be the `i`-th smallest, a prefix of the rows a biased sample,
and two such results joined with `Pairing::Independent` (accepted, as their
seeds differ) strongly dependent. The paired rows are therefore put in a
random order (Fisher–Yates), which keeps the joint law and the Spearman
structure; row `i` is then no longer simulation `i`, but the rows are
exchangeable. Both the reordering's seed (the first number) and the
shuffle (the rest) come from stream `n_sims` of the bootstrap's seed: no
simulation uses that stream, so they share no random numbers with the
draws, and the result is reproducible from the bootstrap's settings alone.
A unit test checks that the row index has no Spearman correlation with
either segment's total. The provenance records `dependence` and
`rank_correlation`. On comauto, ppauto and wkcomp with target rhos 0.5,
0.25 and -0.3 the totals' Spearman rhos are 0.501, 0.253 and -0.304 at
10,000 simulations.

**Capital.** The joint reserves aggregate to one component per line
(`reserves.aggregate(&["lob"])`) and go to
`PredictiveDistribution::capital` with a distortion (TVaR) and an
allocation method. On comauto and wkcomp (ODP, Gamma process, 10,000
simulations) the total's TVaR at 99% and the TVaR diversification benefit
are 4,949,627 and 136,529 independent, 5,011,717 and 78,069 synchronized,
5,075,655 and 10,501 at a Spearman rho of 0.9, and in every case the
standalone VaRs at 99.5% add to more than the total's.

Validation (`validation/tests/reserving_dependence.rs`, unit tests in
`dependence.rs`):

* two copies of wkcomp, synchronized without process error, have equal
  reserves and CDRs in every simulation, for both bootstraps and both
  views; with the Gamma process their correlation is between 0.3 and 0.95,
  independent within four standard errors of zero;
* on the 15 CLRD pairs: the synchronized parameter-error correlation is
  within 0.08 (ODP) or 0.16 (Mack) of the residuals' and not above it in
  absolute value beyond four standard errors; with process error it keeps
  the sign and shrinks for pairs with `|rho| > 0.25`; independent pairs
  are uncorrelated within four standard errors; the ODP lines' standard
  deviations are their independent ones within 5%;
* the rank correlation's Spearman rhos are within four standard errors of
  the target, every component keeps its draws, a line's origins stay
  together, and Mack's one-year view takes the same rank correlation;
* the capital path: the TVaR is the total's, the Euler allocation adds up
  to it, the diversification benefit is positive and the standalone VaRs
  add to more than the total's, and both the TVaR and the benefit order
  independent, synchronized, rank 0.9;
* unit tests: one synchronized segment draws as the independent bootstrap
  (ODP; Mack with the Gamma and residual processes; both views); a line
  and its double are proportional draw by draw; segments of other origins
  or observed cells are refused when synchronized and accepted
  otherwise; the matrix checks; the rank correlation reorders whole rows
  (both bootstraps, both views), puts them in a random order and is
  reproducible.

The segments of a simulation read separate, consecutive parts of its
stream, so no two lines share a uniform. On the six CLRD lines at 4,000
simulations, seeds 1 to 3, the mean of the 15 pairwise correlations of the
independent totals was within 0.005 of zero for both bootstraps, as was
that of the synchronized Mack and ODP process error alone (the Gamma run
minus the run without process error, whose position picks come first in
the stream and so match).

Bindings: Python `OdpBootstrap(..., dependence="independent",
spearman=None)` and `MackBootstrap(..., dependence=..., spearman=...)`,
with `dependence` one of `"independent"`, `"synchronized"`,
`"rank_correlation"` and `spearman` a square list of lists, required with
`"rank_correlation"` and refused otherwise; both are read back as
properties and shown in the repr when not independent. R
`odp_bootstrap()`, `mack_bootstrap()`, `odp_one_year()` and
`mack_one_year()` take `dependence = c("independent", "synchronized",
"rank_correlation")` and `spearman = NULL`, a square numeric matrix, and
`capital_allocation()` takes the aggregated reserves.

Not done: other copulas than Iman–Conover's normal scores (a t copula's
tail dependence would need `reorder_groups` to take other scores), a
correlation estimated from the data other than through the synchronized
residuals, dependent process error, and synchronizing segments of
different shapes (it would need positions matched by calendar period
rather than by origin and age).

### 11. The one-year view with a bootstrap tail

Decisions 8 and 9 left the bootstraps' tail out of the one-year view: Mack
(1999) treats the tail as a factor with its own sigma and standard error
but gives it no calendar time, and R ChainLadder 0.2.21's
`CDR(MackChainLadder(tri, tail = 1.05))` stops with "tail factors are
currently not supported" (re-checked 2026-10-11). chainladder-python has no
CDR. So there is no reference implementation, and these are modelling
choices.

1. **The tail is one more development period.** The step from the oldest
   age to ultimate is valued one development period (the difference of the
   last two ages) after the oldest age, and falls in the coming year when
   that valuation is at most `V + 12` months. With an annual grain only
   the origins already at the oldest age take it; with a quarterly grain
   also those that reach the oldest age within nine months. This keeps
   decision 8's property that an origin whose remaining development falls
   in the year has its lifetime run-off as its one-year view. Rejected:
   keeping the tail without a calendar year (the bootstrap tail would add
   nothing to the one-year view), and spreading it over several years (no
   data calibrates a spread). The choice is conservative for an origin at
   the oldest age if the true tail runs off over several years: its whole
   tail uncertainty is in the coming year. A lagging origin whose tail
   valuation is before `V` (it stopped at the oldest age long ago) takes
   the step too: none of it is observed, and it is realised by the end of
   the year. The step is the whole tail, whatever its shape: a curve
   (`TailCurve`) or a decayed constant (`decay < 1`) selects one factor
   per period for a year past the oldest age and a remainder beyond, and
   the step takes their product, all of it within one development period
   (three months on a quarterly grain). So such a tail is collapsed into
   one step, and the one-year standard deviation of an origin that takes
   it is its lifetime tail risk, an upper bound on its one-year risk when
   the tail's own factors put part of the development in later years.
   Not built: taking only the factors valued in the year and leaving the
   origin open at the next age, re-reserved with the method's tail.
2. **A closed origin.** An origin that takes the tail step is closed at the
   end of the year: its closing ultimate is its simulated ultimate, not the
   refitted method's, and its CDR is the opening ultimate less it. The step
   is drawn as in the lifetime view (decision 9): Mack's from the value at
   the oldest age `C` (observed for an origin already there), mean `T* C`,
   variance `tail_sigma^2 |C|^(2 - alpha)`; the ODP's as one more increment,
   mean the pseudo value at the oldest age times `T* - 1`, with the
   bootstrap's process error. For the expected loss ratio method, whose
   ultimate ignores the losses, a closed origin's CDR is then its opening
   ultimate less the simulated ultimate.
3. **The tail's parameter error.** Each simulation draws the tail exactly
   as the lifetime view does (`TailDraw::draw`), after the pseudo factors
   and before any process draw: an estimated tail is refitted on the pseudo
   factors, a constant one is fixed or drawn from its lognormal, and an
   early attachment moves its factors within the triangle by the pseudo
   factors' deviations. The selected factors drive the year's cells. An
   origin that stays open keeps the method's own tail at the end of the
   year (a constant one fixed, an estimated one refitted on the closing
   triangle); nothing updates it from the closed origins' tail steps. So a
   constant tail's parameter error enters the one-year view only through
   the closed origins, as in Merz and Wüthrich only estimates revised in
   the year contribute.
4. **The method's tail must be the bootstrap's.** With a bootstrap tail
   that is not none, `method.tail()` must equal it (`Tail: PartialEq`, so
   the attachment age and decay count), or the call returns
   `Error::Bootstrap("the bootstrap's tail must be the re-reserving
   method's tail")`. The bootstrap's tail models the true development and
   the method's is the actuary's estimate of it: a mismatch would move the
   closed origins' mean CDR for a reason the user did not choose, and
   overriding the method would silently change what the user specified. A
   bootstrap without a tail and a method with one is the previous
   behaviour, unchanged.
5. **Random stream and failures.** Each simulation draws the pseudo
   factors, then the tail, then the year's cells origin by origin, then the
   tail steps in origin order, after all the cells. Without a bootstrap
   tail every draw is as before (the seed-pinned one-year tests are
   unchanged), and with a fixed constant tail attached at the oldest age
   the open origins' cells are the untailed ones, so their CDR is the tail
   factor times the untailed CDR, simulation by simulation. Drawn with a
   standard error, the lognormal draw shifts the random stream and the
   relation holds in distribution only; attached earlier, the tail's
   factors replace estimated ones inside the triangle and move with the
   pseudo factors, and it does not hold. A tail that cannot be refitted on a
   simulation's pseudo factors fails that simulation like a failed refit:
   `Error::OneYear { failed, n_sims, source: Error::Tail(..) }`, the source
   inside `Error::InSegment` with segments.
6. **Dependence.** Synchronized segments share residual positions only:
   the refit follows each segment's own pseudo factors (two identical lines
   without process error keep equal CDRs, tail steps included), and a
   constant tail's lognormal draw is each segment's own. Rank correlation
   reorders whole CDR rows, as before.

What the fits report: the bootstrap model in `OneYearSegment::bootstrap`
is now fitted with the bootstrap's tail (its chain ladder's ultimate and,
for Mack, its standard errors include it); the provenance records the
tail's settings only when there is a tail.

Checks (`validation/tests/reserving_one_year_tail.rs`, 20,000 simulations
per run, both rules for the last sigma, RAA, GenIns and ABC, five Monte
Carlo standard errors of the simulated standard deviation):

* **A constant tail `T = 1.05`** with Mack's process, its sigma and
  standard error extrapolated as R does: every open origin's standard
  deviation is `T` times R's `CDR(1)S.E.`; the oldest origin's is
  `sqrt(C^2 se^2 + tail_sigma^2 C)`, which is R's `Mack.S.E` of that origin
  under `MackChainLadder(tri, tail = 1.05)`, with mean zero; and the
  total's is `sqrt(T^2 total^2 + var_oldest)`, the oldest origin's CDR
  being independent of the others. All within 2.2 standard errors.
* **The refitted log-linear tail:** the Gamma run less the run without
  process error, on one seed, isolates the oldest origin's tail-step
  process error, whose standard deviation is R's `tail_sigma sqrt(C)`; and
  the oldest origin's one-year standard deviation is its lifetime one on
  another seed. The mean is not asserted: the refit is convex in the
  factors (decision 9).
* Unit tests: the tail step's timing (annual, quarterly development of
  annual origins, quarterly origins); an exact pattern with a bootstrap
  tail has a zero CDR; the per-simulation scaling of the open origins for
  both models; the ODP oldest origin's closing value equals the observed
  value plus its pseudo value at the oldest age times the tail less 1,
  rebuilt from the stream; the oldest origin's one-year and lifetime
  standard deviations agree; the mismatch error, attachment age included;
  a failed refit counted, and named by segment.
* `validation/tests/reserving_dependence.rs`: synchronized log-linear
  twins give equal CDRs, a constant tail's draws differ by line, and rank
  correlation keeps each component's draws and reaches the target
  Spearman.

Bindings: Python needs no new parameter (`OdpBootstrap(tail=...)` and
`MackBootstrap(tail=...)` with `one_year(..., ChainLadder(tail=...))`);
the docstrings state the rule. R's `odp_one_year()` and `mack_one_year()`
use `tail` for the method's tail, so they gain `bootstrap_tail = FALSE`
(when `TRUE` the bootstrap takes the same tail, which satisfies item 4 by
construction), `tail_std_err = NULL`, and for Mack `tail_sigma = NULL`;
either of these given without `bootstrap_tail = TRUE` is an error, since
the bootstrap would have no tail to apply it to.

### 12. Selected development patterns

Status: **Built** (2026-10-14), branch `claude/selected-patterns`, in Rust,
Python and R; revised on 2026-10-15 by the user's decisions (a cumulative
pattern's level as the tail with no tail, interpolated gaps, the Rust
bootstraps taking the method's pattern in the one-year view, and the ODP's
level error kept). What changed from the proposal while building it is
listed under "As built" at the end of this decision, and the revisions
under "Revisions of 2026-10-15"; the items, the Bindings and the Checks
below are the built and revised versions.

Every factor-based method estimates its age-to-age factors from the
triangle alone. The only outside input is `TailConstant { factor,
attachment_age }`, one benchmark point that replaces the factors from an
age on with a decay shape it makes up. A full benchmark pattern, a
credibility blend of a benchmark with the estimate, or a benchmark for the
immature ages only cannot be expressed. This decision adds a selected
pattern to the development settings, so that the chain ladder, the
expected-loss family (decision 2), Mack (with Merz and Wüthrich's one-year
view) and both bootstraps take it.

Neither reference does all of it. R ChainLadder 0.2.21's `MackChainLadder`
takes `Triangle, weights, alpha, est.sigma, tail, tail.se, tail.sigma,
mse.method` and no factors (checked 2026-10-14); `weights` only drops
link ratios. chainladder-python 0.10.1 has `DevelopmentConstant(patterns,
style = "ldf" | "cdf", callable_axis)`: a full pattern by age, per index
level through a callable, no blend; `MackChainladder` rejects its output
("Triangle not compatible with MackChainladder"), and a partial pattern
gets factors of 1 at the ages it does not give (checked 2026-10-14). So
the uncertainty rules below are modelling choices, checked by closed forms
(see Checks).

#### API

```rust
/// How the values of a selected pattern are expressed. The user names it:
/// age-to-age and age-to-ultimate factors are both at least 1 in the usual
/// case and cannot be told apart.
pub enum PatternKind {
    AgeToAge,         // f(a): age a to the next age of the triangle
    AgeToUltimate,    // CDF(a): age a to ultimate
    PercentDeveloped, // p(a) = 1 / CDF(a)
}

/// Weight on the selected factor in `w * selected + (1 - w) * estimated`.
pub enum PatternWeight {
    All(f64),              // every selected age; the default is All(1.0)
    ByAge(Vec<(Lag, f64)>), // listed ages; a selected age not listed gets 1
}

pub struct SelectedPattern {
    pub kind: PatternKind,
    pub values: Vec<(Lag, f64)>, // (age in months, value), any order
    pub weight: PatternWeight,
}

pub struct Development {
    pub average: Average,
    pub sigma_interpolation: SigmaInterpolation,
    pub pattern: Option<SelectedPattern>, // None: every factor estimated, as now
}
```

* **It lives on `Development`**, the setting that says how the age-to-age
  factors are obtained. `ChainLadder`, `Mack` and `MackBootstrap` hold a
  `Development`, and `ExpectedLoss`, `BornhuetterFerguson`, `Benktander`
  and `CapeCod` hold a `ChainLadder`, so all of them take the pattern with
  no field of their own. `OdpBootstrap` has no `Development` (its factors
  are always volume-weighted, as R's `BootChainLadder`), so it gains
  `pattern: Option<SelectedPattern>`. `ClarkLdf`, `ClarkCapeCod`,
  `OdpGlm`, `BayesOdp` and the Meyers models have their own development
  model and take no pattern.
* **Breaking (Rust):** `Development` drops `Copy` and `Eq` (it now owns a
  vector and `f64`s), and so do `ChainLadder`, `Mack`, `MackBootstrap`
  and the four expected-loss settings that contain it; code that copies
  them clones instead. `DevelopmentFit` and `ChainLadderFit` gain public
  fields, which breaks struct literals of them.
* `DevelopmentFit`'s existing fields keep their meaning, the triangle's own
  estimate (`ldf`, `sigma`, `std_err`, `volume`), so the ODP residuals and
  the tail statistics, which need the estimate, do not change. It gains,
  one element per link:

  ```rust
  pub selected: Vec<Option<f64>>, // the pattern's age-to-age factor, after conversion
  pub weight: Vec<f64>,           // w on it; 0 where `selected` is None
  ```

  and `blended_ldf()` (`w * selected + (1 - w) * ldf`, the factors before
  the tail) and `blended_std_err()` (`(1 - w) * std_err`). Without a
  pattern both equal the estimate, so every existing result is unchanged.
* `ChainLadderFit` gains `pattern_rescale: Option<f64>` and
  `pattern_tail: Option<f64>` (item 5). `ChainLadderFit::ldf()` stays "the factors the projection uses": the
  blended factors, with the tail's from its attachment.
* A new `Error::Pattern { age: Option<Lag>, reason: &'static str }`
  reports every check of item 4, inside `Error::InSegment` with segments.

#### Rules

1. **Conversion to age-to-age factors.** The ages are the triangle's
   development ages `a_0 < ... < a_n` (months); link `k` runs from `a_k`
   to `a_{k+1}`.
   * `AgeToAge`: the value at `a_k` is link `k`'s factor. An age-to-age
     value at the oldest age `a_n` is an error: development past the
     oldest age is the tail's.
   * `AgeToUltimate` and `PercentDeveloped` (with `CDF = 1 / p`): the
     values sit at two or more ages of the triangle, from `a_m` to
     `a_{m+r}`, `r >= 1`; the ages they skip are interpolated (item 2), so
     every age `a_m, ..., a_{m+r}` has a CDF, and the links between them
     are fixed: `f_k = CDF(a_k) / CDF(a_{k+1})` for `m <= k < m + r`. The
     level of the pattern, its value at its last age `a_{m+r}`, fixes no
     link; item 5 makes it the tail or rescales it.
   * A pattern with an age that is not a development age of the triangle
     is read on its own ages instead (decision 18), except a single
     age-to-age factor, which keeps the error "not a development age of
     the triangle".
   * Links the pattern does not fix are estimated from the triangle, as
     now: those before its first age and those from its last age on. So a
     cumulative pattern for the immature ages only (CDFs at 12, 24 and 36)
     selects the links 12-24 and 24-36, and the links from 36 on are
     estimated.

   Why cumulative patterns fix ratios and not their level: a CDF at age
   `a` is a statement about every link from `a` to ultimate, including the
   ones the user leaves to the estimate and the tail. Taking it as exact
   would force some of those to absorb the difference (rejected: an
   arbitrary single factor, or a rescaling of estimated factors the user
   asked to keep). The user's tail rule (item 5) already gives the level
   to what follows the pattern, so the same rule holds at every last age,
   and the rule is one sentence: *a cumulative pattern fixes the factors
   between the ages it gives; what develops after its last age comes from
   the estimate and the tail.* At the oldest age with no tail, nothing
   follows the pattern but the tail, so its level is the tail (item 5).

2. **Gaps** (revised 2026-10-15; the proposal rejected them). A cumulative
   pattern with triangle ages missing between two of its given ages `b <
   c` is interpolated: `ln CDF` is linear in the development age between
   them (percents developed converted to CDFs first),

   ```text
   ln CDF(a) = ln CDF(b) + (a - b) / (c - b) * (ln CDF(c) - ln CDF(b)),   b <= a <= c
   f_k = (CDF(b) / CDF(c))^((a_{k+1} - a_k) / (c - b))                      b <= a_k < c
   ```

   so the given ratio `CDF(b) / CDF(c)` is split over the links between
   `b` and `c` in proportion to their length in months (evenly on an
   evenly spaced triangle), geometrically. The interpolated links are
   selected links like the others (a `ByAge` weight may name an
   interpolated age), and the interpolated pattern selects exactly what
   the same CDFs given at every age do. Ages before the first and after
   the last given age are not interpolated: they stay estimated. A single
   cumulative value is still an error: it fixes no link (use a tail to
   set development past an age). Age-to-age patterns may have any ages
   (a missing age-to-age factor is estimated, not interpolated).

3. **Blending** is arithmetic, link by link: `f_k = w_k s_k + (1 - w_k)
   e_k`, with `s_k` the selected factor and `e_k` the estimate. Chosen
   over the geometric blend `s^w e^(1-w)` because (i) the variance of the
   blend is exactly `(1 - w)^2 Var(e_k)`, which items 6 to 8 rely on; the
   geometric blend's is only a delta-method approximation; (ii) credibility
   blends of factors are usually stated this way; (iii) the two differ by
   about `w (1 - w) (s - e)^2 / (2 e)`, below 0.001 for factors within 0.06
   of each other. For a cumulative pattern the weight applies to the
   age-to-age factors it converts to, so a blended CDF is the product of
   blended factors, not the blend of the two CDFs. `PatternWeight::ByAge`
   keys a weight by the age that starts the link.

4. **Validation**, each an `Error::Pattern` naming the age:
   * no values; an age given twice (values or weights);
   * except as decision 18 reads a pattern off the triangle's ages: an
     age that is not a development age of the triangle (of the segment
     being fitted, with segments);
   * `AgeToAge`: a value that is not finite and positive (below 1 is
     allowed: negative development); a value at the oldest age;
   * `AgeToUltimate`: a value that is not finite or is below 1; values that
     increase with age;
   * `PercentDeveloped`: a value outside `(0, 1]`; values that decrease
     with age. The two cumulative kinds carry the same information and get
     the same checks;
   * a cumulative pattern with fewer than two ages (a gap is interpolated,
     item 2); superseded by decision 20, which takes one value as the tail
     from its age;
   * a weight that is not finite or is outside `[0, 1]`; a `ByAge` weight
     at an age with no selected link (for a cumulative pattern, its last
     age);
   * a selected link at or after `TailFit::attachment`, the first link
     the tail replaces (for `TailBondy`, the one after the factor it
     keeps): the pattern and the tail would both claim it. A cumulative pattern
     may end at the attachment age, since its last age fixes no link.

   The estimate is still computed at every link (reported side by side,
   and Mack needs its sigma), so a link the triangle cannot estimate stays
   an error even when the pattern selects it fully.

5. **Tails.** The tail is fitted on the blended factors: an estimated tail
   (`TailCurve`, `TailBondy`, `LogLinear`) extrapolates the pattern in use,
   as chainladder-python's `TailCurve` after `DevelopmentConstant` does.
   Its sigma and standard error for Mack are read off the triangle's own
   estimate (the lines through `ln(e - 1)`, `ln(sigma)` and
   `ln(std_err)`), since they describe the triangle's variability and a
   fully selected link has no standard error to put on a log line.

   **No tail: the level is the tail** (revised 2026-10-15). With no tail
   (`Tail::is_none()`: a constant factor of 1 at the oldest age, the
   default) and a cumulative pattern whose last age is the oldest age
   `a_n`, the pattern's level there is the tail, as chainladder-python's
   `DevelopmentConstant(style = "cdf")` keeps it:

   ```text
   pattern_tail = CDF_given(a_n) = 1 / p_given(a_n)
   ```

   used as `Tail::Constant(TailConstant { factor: pattern_tail, .. })` at
   the oldest age (chainladder-python's decay over the periods past it),
   and reported as `ChainLadderFit::pattern_tail` and a `pattern_tail`
   column of `totals()`. `pattern_rescale` is then 1. A level of 1 (100%
   developed) implies a tail of 1, no tail. `pattern_tail` is `None` with
   a tail, for an age-to-age pattern, and for a cumulative pattern that
   stops before the oldest age: its level is not at the oldest age, so it
   implies no tail and is rescaled to the estimated factors after it, by
   the rule below. The implied tail is a constant with no parameter error
   from the user, so it is treated as a selected factor: its standard
   error is 0 (Mack's parameter risk has no tail term unless the user
   gives `tail_std_err`, which applies as for any constant tail); its
   sigma, which gives the process error of the step past the oldest age,
   follows a `TailConstant` without `tail_sigma` today: extrapolated
   log-linearly from the triangle's sigmas at the tail factor (R's
   `tail.sigma` rule), and an error asking for `tail_sigma` when it cannot
   be extrapolated. Merz and Wüthrich's closed form rejects it as any tail
   other than 1 ("needs no tail factor"). In the bootstraps it is a
   constant, the same in every simulation (unless a `tail_std_err` is
   given, which draws it as a constant tail's), with the step to ultimate
   one more future increment as for a bootstrap tail (decision 9); in the
   one-year view the bootstrap develops past the oldest age with it and
   closes the origins whose tail step falls in the coming year (decision
   11), and the method must use the same tail: its implied tail (the same
   level at the oldest age) or an explicit constant tail of the same
   factor at the oldest age, or the call fails with `Error::Bootstrap("the
   tail the bootstrap's pattern implies must be the re-reserving method's
   tail")`. The check compares the tails as each model uses them, both
   ways: a bootstrap's explicit constant tail is equally the tail a
   method's pattern implies when the factors agree (to a relative
   `1e-12`). A bootstrap that takes the method's pattern
   (`OneYearPattern::Method`, no pattern and no tail of its own) reads it
   as the method does: when the method has a tail of its own, to which it
   rescales the pattern's level, the pattern implies no tail for the
   bootstrap either (it is used as the age-to-age factors it selects), so
   the default never fails on the method's tail; only a bootstrap's own
   pattern can imply a tail the method does not use. Since an implied
   tail's `tail_std_err` (and Mack's `tail_sigma`) change the draws, a
   bootstrap with a pattern records the tail settings in its provenance
   even without a tail of its own.

   **With a tail, the tail is kept** and a cumulative pattern's level is
   rescaled to it; the same rule rescales a pattern that stops before the
   oldest age. With `a_L` the pattern's last age and `CDF_used(a_L)` the product of
   the factors used from `a_L` on times the tail factor,

   ```text
   pattern_rescale = CDF_given(a_L) / CDF_used(a_L) = 1 / (p_given(a_L) * CDF_used(a_L))
   ```

   the factor that multiplies the supplied percents developed (and divides
   the supplied CDFs) to reconcile the pattern's level at its last age with
   what is used there. It reconciles that level only: at an earlier age
   `a`, the rescaled percent developed is
   `CDF_given(a_L) / (CDF_given(a) * CDF_used(a_L))`, which is the
   projection's `1 / CDF_used(a)` only when every link from `a` to `a_L` is
   fully selected. A blended link uses `w * s + (1 - w) * e`, not the
   given factor, so with a blend the rescaled percents at earlier ages are
   not the projection's (with `p = (0.5, 1.0)` at 12 and 24 months, an
   estimate of 2.4 for 12-24, `w = 0.5` and no tail, the factor used is
   2.2 and the rescale 1, but the projection's percent developed at 12
   months is `1 / 2.2 = 0.4545`, not 0.5). `ChainLadderFit::cdf` is the
   projection's at every age. When the pattern ends at the oldest age with
   a tail, `CDF_used = T`, so the rescaled percent developed at the oldest
   age times the tail factor is 100%, the user's rule; ending at a
   `TailConstant`'s attachment age, `CDF_used` is that tail's factor and
   the same holds there. The rescale changes no factor (it multiplies
   every given value alike), so it needs no iteration with the tail fit.
   It is `None` for an age-to-age pattern, which stops at the last link
   and lets the tail multiply on top, as now. With a `TailConstant` after
   it, chainladder-python replaces the level with the tail, as here; with
   no tail it keeps the level as the tail, as the rule above now does
   (both checked on RAA, GenIns and the expected-loss methods; Checks).
   Before the revision of 2026-10-15 a pattern to the oldest age with no
   tail was rescaled to 100% there, which deviated from chainladder-python
   for a level above 1.

6. **Mack's lifetime view.** The pattern is taken as known: a fully
   selected factor has no estimation error, and the blend's error is the
   estimate's share, `Var(f_k) = (1 - w_k)^2 sigma_k^2 / S_k`. The process
   variance is the triangle's: `sigma_k` stays as estimated around `e_k`
   (and interpolated as now), and only the factors it is combined with
   change. In Mack's (1993) closed form, for origin `i` at position `I_i`,

   ```text
   mse(R_i) = U_i^2 sum_{k >= I_i} (sigma_k^2 / f_k^2) (1 / C_ik + (1 - w_k)^2 / S_k)
   ```

   with `f_k` the used factors and `C_ik`, `U_i` projected with them; the
   total's covariance terms `2 U_i U_j sum_k (1 - w_k)^2 sigma_k^2 /
   (f_k^2 S_k)` scale the same way, since in the lifetime view every
   parameter term is the variance of one factor's estimate. In the
   recursion `Mack::fit` uses, this is `std_err_k` replaced by
   `(1 - w_k) std_err_k` (`blended_std_err`) in the per-origin and total
   parameter recursions; the process recursion is unchanged. The tail's
   own parameter error is not part of the pattern: it stays as now
   (`tail_std_err = 0` removes it). A fully selected pattern with no tail
   standard error therefore has zero parameter risk, and the reported
   standard error is the process risk alone. A tail the pattern implies
   (item 5) has a standard error of 0 unless the user gives one, and its
   sigma extrapolated as a `TailConstant`'s without `tail_sigma`, so a
   fully selected cumulative pattern with its level as the tail is process
   risk alone too, the tail step included.

7. **Merz and Wüthrich's one-year view.** Allowed with a pattern (it still
   needs volume-weighted estimates and no tail). The model is Mack's
   lifetime view of item 6 carried forward: under Merz and Wüthrich's
   conditional resampling the true factor `phi_k` varies around the used
   factor `f_k` with the blend's parameter variance, `phi_k - f_k ~ (0,
   g_k^2 tau_k)` with `g = 1 - w` and `tau_k = sigma_k^2 / S_k`; each
   future link ratio is `phi_k` plus Mack's process (`sigma_k^2 / C`); and
   the factor used at every later valuation is `w s + g e'`, `e'` that
   valuation's volume-weighted re-estimate. Linearising each origin's CDR
   in calendar year `y` (its link that year `j`) gives three terms. Write
   `T_l = sigma_l^2 / S'_l`, the estimate's variance at the year's start
   (`S'` the volume then; in the arrangement `ClaimsDevelopmentResult`
   uses, R's `CL_MSEPs`, `T_l / f_l^2` is the existing parameter term
   `t_l = kept(l, y) sigma_l^2 / (f_l^2 S_l)`), `k_l = S_l / S'_l` (R's
   `kept(l, y)`, 1 in the coming year), `a_l = 1 - k_l`, and `s_l` the
   share of the year's new cell in the re-estimate (`share[l - y]`).
   * **Own link.** `(F - f^(y)_j) / f_j` with `F = phi_j + process`. The
     used factor at the year's start is `w s + g ((1 - a) e + a (phi +
     noise))`, so `phi - f^(y) = (1 - g a)(phi - mean) - g a noise`, of
     variance `g^2 tau (1 - g a)^2 + g^2 a^2 sigma^2 / dS =
     g^2 T ((1 - g a)^2 / k + a)`. Process unchanged; parameter term
     `g_j^2 ((1 - g_j a_j)^2 / k_j + a_j) t_j`.
   * **A later link `l`, re-estimated:** `g_l s_l (phi_l - e^(y)_l +
     process) / f_l`. `Var(phi - e^(y)) = sigma^2 (g^2 S + dS) / S'^2 =
     T (1 - (1 - g^2) k)`, and with the new cell's process `sigma^2 / C =
     T S' / C` the term is `g_l^2 (1 - s_l k_l (1 - g_l^2)) s_l t_l`.
   * **The covariance of an origin's own link with a younger origin's
     re-estimate of it** (the new cell is the older origin's own):
     `g s (Cov(phi - f^(y), phi - e^(y)) + sigma^2 / C) / f^2 =
     g_j (1 - s_j (1 - g_j^2 (1 - g_j a_j) - g_j a_j)) t_j`.

   ```text
   msep_o = process_o + u_o^2 ( own_j + sum_{l > j} later_l )
   total  = sum_o [ msep_o + 2 u_o Y_o ( cross_j + sum_{l > j} later_l ) ]
   ```

   with `u_o` the origin's ultimate and `Y_o` the younger origins'
   ultimates. In the coming year (`k = 1`, `a = 0`) the scales are `g^2`,
   `g^2 (1 - s (1 - g^2))` and `g (1 - s (1 - g^2))`. At `w = 0` every scale
   is 1 (today's formula, R's `CDR`); at `w = 1` on every link each
   origin's one-year MSEP is its process term and the total is the sum of
   the origins' (no shared estimate, no covariance), and the years add up
   to Mack's process risk. Strictly between, the used factor follows the
   data less than the estimate does, so the years' CDRs are correlated:
   their variances need not add up to Mack's, and with `s != e` the CDR's
   mean is not zero. These are standard deviations.

   The proposal had the new cell's parameter variance at the estimate's
   `tau` instead of the blend's `g^2 tau` (a later link at `g^2 s t`, the
   cross term at `g (1 - (1 - g) s) t`) and left later years to the same
   scales.
   Both were wrong: the coming year is checked against the exact
   first-order covariance of this model to `1e-9`, and every year against a
   simulation of it; the coming year's scales carried to later years are
   3% to 6% below the simulation from the second year on (Checks).

8. **Bootstraps** (`OdpBootstrap`, `MackBootstrap`, lifetime and one-year
   views). The residuals, the ODP's fitted values and scale `phi`, and
   Mack's sigmas come from the triangle's estimate, as now: the selection
   is not data, and fitting residuals around it would put its bias into
   every pseudo triangle. Each simulation then uses `w s + (1 - w) e*`,
   with `e*` the simulation's pseudo factor: a fully selected link is the
   same in every simulation (no parameter error) and a blended one moves
   by `(1 - w)` times the pseudo factor's deviation. The process error of
   the future cells is unchanged (Gamma with `phi`, or Mack's process
   around the used factors). An estimated tail is refitted on each
   simulation's blended factors; a constant tail is drawn as now. This is
   unlike a constant tail attached early, whose in-triangle factors move
   with the pseudo factors (decision 9): those are the tail's shape, not a
   user's selection. In the one-year view the re-reserving method's
   pattern must equal the bootstrap's when the bootstrap has one, by the
   argument of decision 11 item 4 (`Error::Bootstrap("the bootstrap's
   pattern must be the re-reserving method's pattern")`). Equal means
   the same selection: the two are resolved on each segment's ages and
   compared by their selected factors (to a relative `1e-12`) and weights,
   so the order of the values, the kind they are given in (an age-to-age
   pattern and the factors to ultimate it implies are the same) and the
   form of the weight (`All(1.0)` or `ByAge` with weights of 1) do not
   matter; the level at a cumulative pattern's last age is not compared
   as a selection, but when it implies a tail (item 5) that tail must be
   the method's. A bootstrap without a pattern of its own takes the
   method's in the one-year view by default (revised 2026-10-15:
   `OdpBootstrap::one_year_pattern` and `MackBootstrap::one_year_pattern`,
   `OneYearPattern::Method`), as the bindings' `bootstrap_ldf = True`
   does, so the same analysis gives the same CDR distribution in Rust,
   Python and R. `OneYearPattern::Estimated` keeps the explicit way to
   simulate from the triangle's own factors (the bindings' `False`): the
   pattern then enters only when the method re-reserves, and the pseudo
   factors are simulated unblended; it is an error with a bootstrap that
   has a pattern of its own (`Error::Bootstrap("a bootstrap with a pattern
   of its own cannot simulate from the estimated factors: ...")`). The
   error for a bootstrap pattern that differs from the method's stays.
   The rescale factor is reported for the deterministic fit only.

   **The ODP keeps the origins' level error** (kept by the user's decision
   of 2026-10-15). England and Verrall's bootstrap (decision 8) projects
   each origin from its pseudo latest value, so with every link selected
   and no process error the reserve still varies: on RAA its standard
   deviation is about 73% of the unblended one (seed 5, 2,000
   simulations). This is the ODP's parameter error in the origin effects,
   which a development pattern does not fix, and it stays: replacing the
   pseudo latest values by the observed ones would drop a real part of
   the ODP's parameter error for every user of a pattern. Mack's bootstrap
   projects from the observed latest value, so there the reserve is the
   deterministic one in every simulation and only the process error is
   left. The Rust docs (`OdpBootstrap::pattern`, `MackBootstrap::development`,
   the `pattern` module), Python's and R's bootstrap docs say so.

9. **Segments.** One pattern applies to every segment, as one tail does
   now, and is checked against each segment's own ages (an error names the
   segment). A pattern per segment (chainladder-python's `callable_axis`)
   is not in this decision: it would be a map from segment keys to
   patterns on `fit_segments` and the bootstraps' `SegmentDependence`
   runs, and can follow in its own PR without changing this API.

10. **What the fits report.** Side by side, one value per link:
    `estimated_ldf` (the triangle's, as today's `estimated_ldf` in the
    bindings), `selected_ldf` (the pattern's after conversion; missing
    where none), `ldf_weight` (0 where none), `blended_ldf`, and `ldf` (the
    factors used, after the tail), with `std_err` (used, `(1 - w)` times
    the estimate's) and `estimated_std_err`; and per segment
    `pattern_rescale` (missing for an age-to-age pattern or none) and
    `pattern_tail` (missing where the pattern implies no tail). These go
    into `development_table()` / `development_frame()` and `totals()`.
    The docs of every method that takes a pattern say that a supplied
    pattern lowers the reported uncertainty: its ages carry no parameter
    error, so a wrong benchmark is not reflected in the standard errors.

#### Bindings

* Python: `ChainLadder`, `Mack`, `ExpectedLoss`, `BornhuetterFerguson`,
  `Benktander`, `CapeCod`, `OdpBootstrap` and `MackBootstrap` take
  `ldf=None` (a `dict`, or any mapping, from age in months to value),
  `ldf_kind=None` (`"ldf"`, `"cdf"` or `"percent_developed"`; required
  when `ldf` is given, since the kind cannot be guessed) and
  `ldf_weight=None` (a float or a `dict` from age to weight; not given, 1),
  and read them back as properties of the same names (`None` without a
  pattern) and in their `repr`. `ChainLadderFit` and `MackFit` gain
  `selected_ldf` (`None` where none), `ldf_weight`, `blended_ldf`,
  `estimated_std_err`, `pattern_rescale` (`None` for an age-to-age
  pattern or none) and `pattern_tail` (`None` where the pattern implies
  no tail); `std_err` is the factor in use's. The other fits
  reach them through `chain_ladder` and their frames. The one-year view
  takes the method's pattern from the method object, and the bootstrap's
  from the bootstrap object. `OdpBootstrap.one_year` and
  `MackBootstrap.one_year` take `bootstrap_ldf=True`, as R's one-year
  functions: a bootstrap without `ldf` of its own takes the method's, so
  its simulations are blended; `False` simulates from the triangle's own
  factors, and is an error with a bootstrap that has its own `ldf`. Since
  2026-10-15 they set `one_year_pattern` (`Method` or `Estimated`) on the
  Rust bootstrap instead of copying the pattern themselves.
* R: `chain_ladder()`, `mack()`, `expected_loss()`,
  `bornhuetter_ferguson()`, `benktander()`, `cape_cod()`,
  `odp_bootstrap()`, `mack_bootstrap()`, `odp_one_year()` and
  `mack_one_year()` take `ldf = NULL` (a named numeric vector whose names
  are ages, or a `data.frame` with columns `age` and `value`),
  `ldf_kind = NULL` (the same three strings; required with `ldf`) and
  `ldf_weight = 1` (a number or a named vector). The one-year functions
  also take `bootstrap_ldf = TRUE`: the bootstrap takes the method's
  pattern too, as item 8 requires of a bootstrap with one; `FALSE`
  simulates from the triangle's own factors (`OneYearPattern::Estimated`).
  A `chain_ladder_fit` (and so a `mack_fit`) gains the properties
  `selected_ldf` (`NA` where none), `ldf_weight`, `blended_ldf`,
  `estimated_std_err`, `pattern_rescale` (`NULL` for an age-to-age
  pattern or none) and `pattern_tail` (`NULL` where the pattern implies no
  tail). With `bootstrap_tail = FALSE`, a `tail` of the method's own and a
  pattern whose level implies a tail in the bootstrap, the one-year
  functions fail with item 5's tail error: give `bootstrap_tail = TRUE` or
  `bootstrap_ldf = FALSE`.
* No new dependency: `prospicio-reserving`'s default path stays
  `wasm32-unknown-unknown` clean.

#### Pull requests

1. `claude/selected-patterns` (this note, then): `SelectedPattern`, the
   conversion, validation and rescale, the chain ladder and expected-loss
   family, Mack and the one-year closed form, and the bootstraps (item 8,
   lifetime and one-year), with the bindings. The proposal left the
   bootstraps to a second PR; they were small enough to build here (a
   blend of the pseudo factors in the three places that draw them), so
   there is one PR.

#### Checks

`validation/tests/reserving_patterns.rs`, with references in
`validation/reference/reserving_patterns_python.csv` (generator
`validation/scripts/reserving_patterns_python.py`) and the existing R rows
of `validation/reference/reserving_chainladder_r.csv`:

* **chainladder-python parity** (`DevelopmentConstant` then `Chainladder`
  on RAA and GenIns; then `ExpectedLoss`, `BornhuetterFerguson`,
  `Benktander` and `CapeCod` on `clrd_wkcomp.csv` with premium): a full
  benchmark pattern (`s = 1 + 1.1 (e - 1)`) as age-to-age factors, as
  factors to ultimate with 1 at the oldest age (where the level
  conventions agree), as factors to ultimate with the level 1.03 and no
  tail (both keep the level as the tail, since the revision of
  2026-10-15; `pattern_tail = 1.03`), as factors to ultimate with the
  level 1.03 followed by `TailConstant(1.05)` (where they agree too), and
  a blend
  (`0.6 s + 0.4 e`, blended by the script since chainladder-python has
  none, so these rows check how blended factors reach each method, not the
  blend); factors in use, factors to ultimate and ultimates to 1e-9
  relative, and the given pattern read back (age-to-age from the selected
  factors, cumulative as the factor to ultimate in use times
  `pattern_rescale`). The cumulative rows are checked again with the
  pattern given as percents developed.
* **Hand cases** (unit tests in `pattern.rs`): each kind converts to the
  same factors (`p = (0.5, 0.8, 0.95)` gives links 1.6 and 1.1875; with a
  tail of 1.02, `pattern_rescale = 1 / (0.95 * 1.02)` and the rescaled
  0.95 times the tail is 100%); round trips between the kinds; partial
  patterns at the young and the old end; the blend's limits (`w = 0` is the
  estimate, `w = 1` the selection) in the conversion and through the chain
  ladder; every error of item 4; a selected link at a tail's attachment
  rejected, a cumulative pattern ending there accepted; an estimated tail
  fitted on the blended factors; Bornhuetter-Ferguson by hand; segments
  (an error names the segment, the tables' new columns, unchanged tables
  without a pattern). Since 2026-10-15: interpolation (CDF 4 at 12 and 1
  at 48 gives three links of `4^(1/3)`; percents developed converted
  first; uneven ages split by months; an interpolated age takes a weight;
  a gapped pattern selects what the same CDFs given in full do); the
  implied tail (95% at the oldest age with no tail: `pattern_tail = 1 /
  0.95`, rescale 1, standard error 0, the same ultimates as the constant
  tail `1 / 0.95`; none with a tail, a partial or an age-to-age pattern;
  100% implies a tail of 1); Mack with the implied tail equal to the
  constant tail with `tail_std_err = 0`, no parameter risk, a given
  `tail_std_err` applied, and Merz and Wüthrich rejecting it.
* **Mack against R `MackChainLadder`** (the existing rows of
  `reserving_chainladder_r.csv`, since R takes no factors): selecting the
  triangle's own estimated factors with `w = 1` reproduces R's ultimates
  and `Mack.ProcessRisk` per origin and in total, with zero parameter risk;
  with `w = 0.4` the parameter risk is 0.6 times R's `Mack.ParameterRisk`,
  per origin and in total, the process risk R's; on RAA, GenIns and ABC, to
  1e-9. A partial, non-uniform pattern on GenIns is checked against the
  closed form of item 6 computed apart from the recursion (unit test).
* **Merz and Wüthrich:** `w = 0` is R's `CDR` (unit test, and every
  existing R row unchanged); `w = 1` on every link gives the process-only
  form, a total with no covariance, and a run-off that adds up to Mack's
  process risk (unit test). With a uniform (`w = 0.5`) and an age-by-age
  blend of the triangle's own factors, the coming year equals the exact
  first-order covariance of item 7's model, computed directly from the
  triangle's cells, to 1e-9 on RAA, GenIns and ABC; every calendar year of
  the run-off is within five Monte Carlo standard errors (about 1.9%) of
  40,000 simulated run-offs of the model on GenIns (largest gaps 0.9%,
  1.1% and 1.1% without a pattern, uniform and by age; mean 0.3%); and
  Mack's bootstrap re-reserving with `w = 0.5` reconciles with the closed
  form's coming year within five Monte Carlo standard errors (20,000
  simulations on GenIns).
* **Bootstraps** (unit tests): Mack's bootstrap with every link selected
  and no process error gives the deterministic reserve in every simulation
  and a CDR of 0 in the one-year view; with Mack's process its SD matches
  Mack's process risk within five Monte Carlo standard errors; a uniform
  blend of the triangle's own factors with `w = 0.6` scales the
  parameter-only SD to 0.4 of the unblended one (within 0.02); the ODP
  bootstrap on a triangle with no residuals gives the blended chain ladder's
  reserve in every simulation, and on RAA with every link selected and no
  process error keeps only the levels' error of "As built" (below 80% of
  the unblended SD, centred on the chain ladder); the pattern-mismatch
  error of item 8 in both bootstraps, and a method with a pattern accepted
  by a bootstrap without one. Since 2026-10-15: a bootstrap without a
  pattern gives the same one-year draws as one given the method's
  (`OneYearPattern::Method`), different ones with `Estimated`, and
  `Estimated` with a pattern of its own is an error (both bootstraps);
  with the implied tail the ODP bootstrap on a triangle with no residuals
  gives the chain ladder's reserve, tail included, in every simulation,
  Mack's bootstrap with every link selected and no process error the
  deterministic reserve with the tail, the one-year view a CDR of 0 on
  that triangle, the same draws with the method's explicit constant tail
  of the same factor, and the implied-tail error with another tail.

#### As built (2026-10-14)

Changes from the proposal, made while building it:

* **Item 7's terms.** The proposal's later-link and cross terms used the
  estimate's parameter variance for the new cell instead of the blend's,
  and it carried the coming year's scales to later years. Both are
  corrected above, with the exact and simulated checks.
* **The ODP bootstrap keeps the origins' level error** (item 8). England
  and Verrall's bootstrap, as built in decision 8, projects each origin
  from its pseudo latest value, so with every link selected and no process
  error the reserve still varies: on RAA its SD is about 73% of the
  unblended one (seed 5). This is the ODP's parameter error in the origin
  effects, which a development pattern does not fix. Mack's bootstrap
  projects from the observed latest value, so there the reserve is the
  deterministic one in every simulation, as the proposal's Checks said of
  both.
* **One PR**, bootstraps included (Pull requests).
* **Tables.** `development_table()` gains its five pattern columns, and
  `totals()` its `pattern_rescale`, only when a pattern selects a factor
  in some segment, so a fit without a pattern has the same tables as
  before. `std_err` is always the factor in use's, `(1 - w)` times the
  estimate's (unchanged without a pattern).
* **Provenance.** `Development`'s `Debug` leaves out `pattern` when there is
  none, and `OdpBootstrap` records `pattern` only when it has one, so the
  provenance of results without a pattern is as before.
* **Bindings.** Python's `ldf_weight` defaults to `None` (meaning 1), not
  `1.0`: PyO3 cannot give a float default to an argument that also takes
  a dict, and `None` lets an `ldf_weight` given without `ldf` be an error.
  R's one-year functions gained `bootstrap_ldf` (default `TRUE`), since
  they build the bootstrap and the method from one set of arguments;
  Python's `OdpBootstrap.one_year` and `MackBootstrap.one_year` take the
  same argument with the same default, so that one analysis gives the
  same CDR distribution in both languages (a review finding: without it,
  Python simulated the pseudo factors unblended where R blended them).
  The one-year check compares the two patterns by their resolved
  selection on each segment's ages, not as given (item 8; a review
  finding: Python dicts in another key order were rejected).
  Python's `std_err` (and R's) on a fit is now the factor in use's,
  `(1 - w)` times the estimate's, as the tables already were; without a
  pattern it is unchanged. Tests: `python/tests/test_reserving_patterns.py`
  and `R/prospicio/tests/test-reserving-patterns.R` (each kind, partial
  patterns, blends uniform and by age, the tail rescale, the five methods,
  Mack's risks, both bootstraps and their one-year views, segments, and
  every error, a missing or wrong kind label included).
* **Breaking (Rust)**, as proposed: `Development` loses `Copy` and `Eq`, and
  `ChainLadder`, `Mack`, `MackBootstrap`'s contents and the four
  expected-loss settings lose `Copy`; `Development`, `OdpBootstrap`,
  `MackBootstrap` (`one_year_pattern`, 2026-10-15), `DevelopmentFit` and
  `ChainLadderFit` gain public fields; `Error` gains `Pattern`.

#### Revisions of 2026-10-15

Decided by the user on 2026-10-15, after review of the built PR:

1. **A cumulative pattern's level is the tail when no tail is given**
   (item 5). It replaces the earlier rule, which rescaled such a pattern
   to 100% at the oldest age. The implied tail factor is the pattern's
   CDF at the oldest age (1 over its percent developed there), a
   `TailConstant` at the oldest age, reported as `pattern_tail`. With a
   tail the earlier rule stands (the level is rescaled so that percent
   developed at the oldest age times the tail is 100%). Decided here: a
   partial cumulative pattern that stops before the oldest age implies no
   tail (its level is not at the oldest age) and keeps the rescale to the
   estimated factors after it; in Mack the implied tail has no parameter
   error unless the user gives `tail_std_err`, and its process error past
   the oldest age follows a `TailConstant` without `tail_sigma` (sigma
   extrapolated log-linearly, an error when it cannot be); in the
   bootstraps and the one-year view it is a constant, the same in every
   simulation, and the one-year method must use the same tail. The
   chainladder-python rows with the level 1.03 and no tail are now in the
   parity reference and match at 1e-9.
2. **Rust bootstraps take the method's pattern by default** in the
   one-year view (item 8): `OneYearPattern::Method` on `OdpBootstrap` and
   `MackBootstrap`, as the bindings' `bootstrap_ldf = True`;
   `OneYearPattern::Estimated` simulates from the triangle's own factors
   and is an error with a pattern of the bootstrap's own; the mismatch
   error stays.
3. **Gaps are interpolated** (item 2): `ln CDF` linear in development age
   between the given ages; a single given age is still an error (until
   decision 20); ages
   before the first and after the last given age stay estimated.
4. **The ODP bootstrap keeps the origins' level error** with every factor
   selected (item 8): documented in the Rust, Python and R docs.

### 13. Averages over the latest periods, excluding high and low, and geometric

Status: **Built** (2026-10-18), branch `claude/development-averages`,
issues #336 and #339, in Rust, Python and R; the bootstraps and the
one-year views following the exclusions are a second PR (decision 14). What changed
from the proposal is under "As built" at the end of this decision.

Reserving practice selects age-to-age factors from averages over the
**latest n** accident years, from medial averages (the latest n without
the highest and lowest link ratios) and from geometric averages (Friedland,
*Estimating Unpaid Claims Using Basic Techniques*, chapter 7, Exhibits I
and II; Werner and Modlin, *Basic Ratemaking*, Appendix A). `Development`
averages every origin. The guides work around it today: a triangle cut to
its latest `n + 1` diagonals for the volume-weighted latest n, and averages
computed from `Triangle.link_ratios()` and passed in as a selected
pattern for the rest (`docs/guides/reserving-friedland.md`,
`docs/guides/ratemaking-werner-modlin.md`).

The reference is chainladder-python 0.10.1's `Development(n_periods,
drop_high, drop_low, preserve)` (read 2026-10-18). R ChainLadder's
`MackChainLadder(weights = W)` takes the same exclusions as a 0/1 matrix,
which is the reference for Mack's standard errors under them. Neither has
a geometric average.

1. **Settings.** `Development` gets three fields, each with a default
   that keeps the current fit:

   ```rust
   pub struct Development {
       pub average: Average,
       pub sigma_interpolation: SigmaInterpolation,
       pub pattern: Option<SelectedPattern>,
       /// Average only the link ratios that end in the latest `n` origin
       /// periods of calendar time; `None` uses every one.
       pub n_periods: Option<usize>,
       /// Leave out the `drop_high` highest and `drop_low` lowest of the
       /// link ratios kept by `n_periods`, at each age.
       pub drop_high: usize,
       pub drop_low: usize,
   }
   ```

   Its hand-written `Debug` leaves the new fields out at their defaults,
   as it does `pattern`, so that the settings recorded in a bootstrap's
   provenance stay as they were.

2. **`n_periods` counts calendar time, not origins.** A link ratio from
   age `k` to `k + 1` of an origin is kept when the valuation of its
   starting cell is at least the segment's latest valuation less `n`
   origin periods, in months. That is chainladder-python's rule (its
   `val_offset`): with the origin and development grains equal it keeps
   the link ratios ending on the latest `n` diagonals, and on a complete
   triangle those are the latest `n` origins at each age. On a sparse
   triangle, or one with a development grain finer than its origin grain,
   the two differ and the diagonals are the reference's choice. An age
   with fewer than `n` link ratios uses all of them ("latest 5" with
   three years is the three, as in Friedland). `n_periods = Some(0)` is an
   error; `Some(n)` at or past the number of diagonals is every link.

3. **`drop_high` and `drop_low` rank within the kept link ratios.** At
   each age the link ratios kept by item 2 are ranked by value, ties
   broken by the starting cell's value (chainladder-python's `lexsort`),
   and the top `drop_high` and bottom `drop_low` are left out. When that
   would leave no link ratio, the age keeps every link ratio item 2 kept,
   as chainladder-python with its default `preserve = 1`: Friedland's
   medial average of two values is their simple average, of one value the
   value itself. `preserve` is not a setting; a fixed 1 is what both books
   do. The fit reports what was used (item 5), so a skipped drop is
   visible rather than a warning.

4. **The exclusions apply to the factor and to sigma alike.** Every
   average (volume, simple, regression) is the same weighted regression
   as now, on the kept link ratios only: its sigma is estimated from them
   with `kept - 1` degrees of freedom, and its volume `sum(C[k]^alpha)`
   runs over them, as chainladder-python's weights `w_` and R's
   `weights`. An age left with a single link ratio has its sigma filled
   in as now (`sigma_interpolation`). `n_periods = Some(1)` leaves every
   sigma unestimable, so `ChainLadder` and the expected-loss methods fit
   and `Mack` gives the existing error that a sigma can neither be
   estimated nor interpolated.

5. **The fit says what it used.** `DevelopmentFit` gets `n_links:
   Vec<usize>`, the number of link ratios behind each factor. Python and
   R expose it as `n_links`. Selected factors (decision 12) apply after
   the average, as now.

6. **Geometric.** `Average::Geometric` is `exp(mean(ln r))` over the kept
   link ratios. It is not a weighted regression and Mack's model has no
   variance for it, so its sigma and standard errors are NaN, its
   `DevelopmentFit::alpha` is NaN, and `Average::alpha` returns
   `Option<f64>` (`None` for it). That is a breaking change of a public
   enum and of `alpha`'s signature, marked as such in the changelog.
   `ChainLadder` and the four expected-loss methods take it. `Mack`, both
   bootstraps and the one-year views reject it before fitting, with an
   error that names the average. A link ratio of 0 or below is an error
   for it, as a zero value is for the simple average.

7. **Which methods take the exclusions in this PR.** `ChainLadder`, the
   four expected-loss methods and `Mack` (lifetime). Merz and Wüthrich's
   closed form (`Mack::one_year`, `claims_development_result`), the ODP
   and Mack bootstraps and their one-year views reject `n_periods`,
   `drop_high` and `drop_low` before fitting: the closed form's volumes
   and the bootstraps' residuals and re-reserving assume every link ratio,
   and making them follow the exclusions (chainladder-python's
   `BootstrapODPSample` takes `n_periods` and the drops) is a second PR
   with its own checks. Decision 14 is that PR: it lifts the rejection for
   both bootstraps and their one-year views, and keeps it for the closed
   form. `ClarkLdf`, `ClarkCapeCod`, the ODP GLM and the
   Bayesian models do not use `Development`'s averages and are
   unchanged.

8. **Factors to ultimate below 1** (issue #336, item 3). A selected
   pattern with `PatternKind::AgeToUltimate` today requires every value to
   be at least 1 and not to increase with age, and `PercentDeveloped`
   every value in `(0, 1]` and not to decrease. Friedland's XYZ Insurer
   reported pattern (0.992 at 96 and 108 months, then 0.999 and 1.000) is
   real negative development and fails both. Those checks cannot tell an
   age-to-age pattern passed as cumulative from a cumulative one anyway
   (both decrease toward 1 in the usual case), so they are relaxed to
   finite and positive for both kinds, in any order. The age-to-age
   factors the conversion gives may then be below 1, which
   `PatternKind::AgeToAge` already allows. A cumulative pattern's last
   value is still its tail level (decision 12, revision 1), and a level
   below 1 is a tail below 1, allowed wherever a `TailConstant` below 1
   is.

**Checks.**

* chainladder-python 0.10.1 on RAA and GenIns: `ldf_`, `sigma_` and
  `std_err_` under `n_periods = 3`, under `drop_high = drop_low = 1`,
  under both, for each of volume, simple and regression; `n_periods = 1`
  (factors only); and the weights `w_` against `n_links`. Stored as a
  reference CSV with the script that wrote it.
* R ChainLadder's `MackChainLadder(RAA, weights = W)` with the 0/1 matrix
  of `n_periods = 3` and of the drops: Mack's standard errors by origin
  and in total.
* The books' printed numbers: Friedland's U.S. Industry Auto averages
  (simple, medial, volume over the latest 5 and 3, geometric over the
  latest 4, to the three decimals printed) and XYZ Insurer's volume over
  the latest 2 and its reported CDFs below 1 as `ldf_kind = "cdf"`;
  Werner and Modlin's Appendix A straight average excluding high and low
  (1.0665, 1.0279, 1.0208, 0.9799 on the book's rounded link ratios).
* Errors: `n_periods = Some(0)`; the geometric average in Mack and the
  bootstraps; the exclusions in the bootstraps and the closed-form
  one-year view; a link ratio of 0 under the geometric average.
* The default `Development` gives bit-identical fits and the same
  provenance `Debug` text as before.

**Bindings.** Python: `n_periods=None`, `drop_high=0`, `drop_low=0` and
`average="geometric"` on `ChainLadder`, `Mack`, `ExpectedLoss`,
`BornhuetterFerguson`, `Benktander` and `CapeCod`, and on the bootstraps
(where a value other than the default raised, naming the follow-up;
decision 14 lifts it); the fit's `n_links`. R: the same arguments on `chain_ladder()`, `mack()` and
the expected-loss functions. The guides' workarounds can then go; that
is the Guides lane's change, filed as an issue for it.

**As built** (2026-10-18). What changed from the proposal while building
it:

1. `DevelopmentFit` also has `excluded: bool`, true when the exclusions
   removed a link ratio at some age; `n_links` alone cannot tell an
   exclusion from a hole in the triangle. It is exposed in Python and R
   beside `n_links`, and both raise on a fit with several segments, as
   `n_links` does.
2. Rejections go through a new `Error::NotSupported { setting, method,
   reason }`, which names the setting and the method. Merz and
   Wüthrich's closed form keeps `Error::ClaimsDevelopment` and rejects any
   exclusion setting, even one that removes nothing at the valuation,
   since the window slides with the coming diagonal.
   `OdpBootstrap` has no `Development`, so only its one-year view, through
   the re-reserving method, can reject anything.
3. Under the geometric average `volume` is NaN as well as `alpha`, `sigma`
   and `std_err`.
4. An age that `n_periods` leaves with no link ratio (a sparse triangle
   whose only pairs at that age are old) is `Error::Factor`, "no link
   ratio is left after n_periods".
5. Link ratios that are not finite (a zero starting value) are never
   dropped and do not count toward the ranks, so the result does not
   depend on how a platform orders NaN. A huge `n_periods` or drop count
   saturates instead of overflowing: `n_periods` past the number of
   diagonals keeps every link ratio, a drop count past the number of
   link ratios keeps them all.
6. The references: chainladder-python's `w_` counts the starting cell on
   the latest diagonal under `n_periods`, so `n_links` is compared with
   the link ratios that have a weight above 0 and a finite value; its
   standard error at an interpolated sigma divides by the first origin's
   earlier value, not by the volume, so standard errors are compared only
   where sigma is estimated (prospicio divides by the volume, as Mack's
   formulas and R do). R falls back to Mack's sigma rule in a few
   settings; those are left out of the R reference and listed in its
   script.
   chainladder-python also returns every link ratio once `n_periods` is
   at least the number of origins less 1, which on a triangle with a
   development grain finer than its origin grain still has older link
   ratios to cut; prospicio keeps to the calendar rule there, and its
   test of that case (annual origins, quarterly ages) stays below the
   threshold, where the two agree.
7. Bindings: in Python the new arguments come after `ldf_weight`, and in
   R after the last existing argument, so positional calls keep working.
   R's `mack_one_year()` takes `mack_n_periods`, `mack_drop_high` and
   `mack_drop_low` beside `mack_average`. Python's `OdpBootstrap` takes
   none of them (it has no average).

### 14. The bootstraps and the one-year views under the exclusions

Status: **Built** (2026-10-19), issue #401, branch
`claude/bootstrap-exclusions`. This is decision 13's second PR. The
user chose the three settings that the issue left open (items 2, 4
and 6). The other items follow from the evidence below. "As built" at the
end lists what changed from the proposal while building it.

Decision 13 gave `Development` the settings `n_periods`, `drop_high` and
`drop_low` for the chain ladder, the expected-loss methods and Mack's
lifetime view. These methods still reject the three settings:

- Mack's bootstrap.
- The ODP bootstrap, which has no such settings.
- The one-year views of both bootstraps.
- Merz and Wüthrich's closed form.

This decision lets both bootstraps and their one-year views follow the
settings. The closed form keeps its rejection, for the reason in item 8.

The reference is chainladder-python 0.10.1's
`BootstrapODPSample(n_periods, drop_high, drop_low)` (read 2026-10-19
from the 0.10.1 wheel, `adjustments/bootstrap.py`). It has no Mack
bootstrap and no one-year view. R ChainLadder's `BootChainLadder` takes
no weights. So outside the ODP bootstrap's deterministic steps, the
checks are internal (item 9).

1. **One mask of kept link ratios.** `Development` gets a crate-internal
   `kept_mask(&self, segment) -> Result<Vec<bool>>`, indexed by
   `origin * (n_dev - 1) + link`. It is true where the fit uses the link
   ratio. The fit uses the same code: the `n_periods` cutoff becomes a
   method, and `kept_links` keeps the origin index of each pair. This
   way the mask and the factors cannot disagree. `DevelopmentFit` gets
   no new public field, because a public field would break the struct
   literals of external users. With no exclusions, every observed link
   ratio is kept.

2. **The drops stay fixed on the observed triangle** (the user's
   decision). A simulated triangle keeps the link ratios that the
   observed triangle keeps. The bootstrap does not rank the pseudo link
   ratios again. Under this rule, Mack's bootstrap describes the model
   that Mack's lifetime view already computes with 0/1 weights (R's
   `MackChainLadder(weights = W)`, decision 13). Its standard deviation
   can therefore be checked against Mack's standard error under the
   same settings.

   Ranking the pseudo link ratios again in each simulation would
   bootstrap the dropping rule as well. That would capture the
   selection effect, but no result exists to check it against. It can
   come later as an option. `n_periods` keeps the same positions under
   either rule.

3. **Mack's bootstrap.** Its own `development` decides which link
   ratios are kept:

   - **Residuals.** Only kept link ratios have a residual. The
     small-sample factor is `sqrt(n / (n - 1))`, where `n` is the number
     of informative kept link ratios at the age. This is the same count
     that the fit's `sigma` uses (`kept - 1` degrees of freedom).
     Excluded positions have a NaN residual and are not in the pool.
     The pool is centred over the kept residuals, as before.
   - **Pseudo factors.** The pseudo factors are re-averaged over the
     kept links only. An excluded link draws no residual and adds
     nothing to the numerator or the denominator, including a link that
     starts at zero.
   - **Draw positions.** `draw_positions` still lists every observed
     link. With synchronized segments, a position that any segment
     excludes therefore drops out of the shared pool (`synchronize`
     keeps only positions where every segment has a residual). Segments
     with different exclusions still have the same shape.
   - **Unchanged parts.** The process, the tail step and the run-off use
     `f`, `sigma` and the pseudo factors. These already follow the mask,
     so these parts do not change.
   - **Rejections.** The geometric average stays rejected: Mack's model
     has no variance for it.

4. **The ODP bootstrap gets `n_periods`, `drop_high` and `drop_low`.**
   They are new fields with decision 13's defaults, and they apply to
   the volume-weighted chain ladder only. The ODP bootstrap has no
   average setting. They change four steps:

   - **Fitted values.** The fitted values are back-cast from each
     origin's latest value with the factors under the exclusions. This
     is the same as chainladder-python's `full_expectation_` of a
     `Chainladder` on `Development(n_periods, drop_high, drop_low)`.
   - **Which cells have a residual.** An incremental cell `(o, k + 1)`
     has a residual when the link ratio `(o, k)` is kept, because its
     increment is the numerator of that link. The first age's cells
     always have one.
     - chainladder-python gives the first age's cell the weight of link
       0, so dropping link `(o, 0)` drops two residuals. We do not copy
       that quirk. The parity check (item 9) compares the other cells
       and lists this one.
     - Cells without a residual are NaN and are not in the pool. Zero
       residuals and the corner cells stay in the pool, uncentred, as
       R's `BootChainLadder` does now.
   - **Scale** (the user's decision). The scale is
     `phi = sum of the kept r^2 / (N - p)`, where `N` is the number of
     observed cells and `p = n_origins + n_ages - 1`. This is
     chainladder-python's rule, and with no exclusions it gives
     today's `phi`.
     - The rule is biased low under exclusions: on RAA with
       `n_periods = 3`, `phi` is about 70% of the plain one (694.0
       against 983.6; "As built", item 2). The design note and the
       docstring say so.
     - The residuals keep today's factor `sqrt(N / (N - p))`.
   - **Simulation.**
     - Every observed cell still gets a resampled residual, so the
       random stream does not change and a later kept link has its
       starting value.
     - The pseudo factors are the volume-weighted average over the kept
       links only, with R's rule of 1 for an empty denominator.
     - The process is unchanged.

   `prepare` keeps its Bayesian callers (`BayesOdp` and `BayesBf`) on
   the defaults. Their results stay bit-identical.

5. **The geometric average** stays rejected in both bootstraps and in
   their one-year views. Neither bootstrap has a variance for it.
   Re-reserving with it on a volume-weighted simulation would put a mean
   in the claims development result that no setting asked for.

6. **One-year views: the bootstrap inherits the method's exclusions**
   (the user's decision, the rule of `OneYearPattern::Method`).
   - When the bootstrap sets no exclusions, it takes the re-reserving
     method's `n_periods`, `drop_high` and `drop_low` for the coming
     year's simulation.
   - When the bootstrap sets its own exclusions and they differ from
     the method's, the view is an error that names both, as a
     pattern mismatch is.
   - This removes the rejection in `Sims::check`.
   - Re-reserving fits the method again on the triangle with the coming
     diagonal added. The `n_periods` window therefore moves by one
     diagonal, and the drops are ranked again with the new link ratio,
     as the actuary will do next year.

7. **The claims development result has a known mean under the
   exclusions.**
   - **`n_periods`.** When the window moves, a known link ratio leaves
     the average. At age `k` the expected revision of the factor is, to
     first order, `C_old * (f_k - F_old) / S'_k`, where `S'_k` is the
     closing volume. So the mean of the claims development result is
     not zero.
   - **Drops.** Ranking again also moves the mean, by an amount that has
     no closed form.
   - Both are what next year's re-reserving does, so they are reported
     and not removed. The docstrings say that the mean of the
     claims development result is not zero under the exclusions.

8. **Merz and Wüthrich's closed form keeps its rejection.**
   - **With fixed 0/1 weights** (this year's kept set, the new diagonal
     kept), the formulas go through with the kept volumes. But that is
     not the estimator next year:
     - the window moves, so a link ratio leaves the average and the
       mean is not zero (item 7);
     - the drops make the estimator nonlinear in the new diagonal.
   - **R's `CDR` with `weights`** cannot serve as a reference. It moves
     the weights by one column in its `alpha`. At age 1 it ignores them.
     At later ages it agrees with a moving window only because, under a
     calendar window, the link coming in is the link kept next year. It
     has no term for the mean. (A total `CDR(1)S.E.` of 21,510.79 on RAA
     with `n_periods = 3`, quoted when this decision was written, was not
     reproduced and is in no test; `knowledge/references/r-chainladder-cdr.md`.)
   - The error message points to the one-year view of Mack's bootstrap.

9. **Checks.**
   - **Invariance.** Exclusions that leave out no link ratio give
     bit-identical draws, seed for seed, in both bootstraps and in both
     one-year views. Examples are `n_periods` at or past the number of
     diagonals, or drops of 0. Provenance is byte-identical at the
     defaults, and a new setting appears in it only when it is set.
   - **Mack's bootstrap, lifetime.** Under `n_periods = 5` and under
     `drop_high = drop_low = 1` on RAA and GenIns, check two things:
     - the mean is the chain ladder's under the same settings;
     - the standard deviation by origin and in total is Mack's standard
       error under the same settings, within the tolerance of
       `knowledge/findings/mack-bootstrap-lifetime-vs-mack.md`. Mack's
       standard error is itself at parity with R's `MackChainLadder(weights = W)`.
   - **Construction.** The residuals are NaN exactly where the mask
     excludes a link ratio. The pool's size is the sum of the
     informative kept link ratios.
   - **The ODP bootstrap against chainladder-python 0.10.1** on RAA and
     GenIns, under `n_periods = 3`, under `drop_high = drop_low = 1`,
     and under both. A script writes a reference CSV with:
     - the fitted incremental values;
     - which cells have a residual (except the first age's quirk);
     - the unscaled residuals;
     - `phi`.
     These are all compared exactly. The simulated mean of the reserves
     is compared within Monte Carlo error, with chainladder-python's
     samples refitted by the same `Development`. The standard deviation
     is not compared, because the conventions differ (centring, dropped
     zeros, hat matrix).
   - **One-year views.**
     - The invariance above.
     - An origin with one cell left has the same one-year result as its
       lifetime result, under the exclusions.
     - Under `n_periods`, the mean of the chain ladder's claims
       development result by origin matches item 7's first-order
       revision within Monte Carlo error.
     - Under the drops, a pinned seed regression.
   - **Errors.** The geometric average in both bootstraps; a bootstrap
     whose own exclusions differ from the method's; the closed form
     under any exclusion; `n_periods = Some(0)` on `OdpBootstrap`.

10. **Bindings.**
    - **Python.** `OdpBootstrap(n_periods=None, drop_high=0,
      drop_low=0)`, placed after the existing arguments. `MackBootstrap`
      already takes the three settings and now uses them. The one-year
      views take the method's settings.
    - **R.** `odp_bootstrap()` gets `n_periods`, `drop_high` and
      `drop_low` after its last argument. `mack_bootstrap()` and the
      one-year functions stop raising.
    - **Tests.** The binding tests that expected the rejection now check
      the fits.

**As built** (2026-10-19). What changed from the proposal while building
it:

1. `Error::ExclusionsMismatch { bootstrap, method }` is a new variant of
   the public `Error` enum. The mismatch error has to write out both
   settings, and `Error::Bootstrap` holds a static string. It is marked
   breaking in the changelog, as are the three new public fields of
   `OdpBootstrap`.
2. `phi` under `n_periods = 3` on RAA is 694.0, not the 322.4 that item 4
   first quoted. 322.4 is chainladder-python's `scale_`, which also drops the
   first-age residuals of the origins whose link 0 is dropped (its weight
   quirk); prospicio keeps them, as item 4 says. The plain `phi` is 983.6.
3. The reference means in the ODP check come from chainladder-python's
   samples with its pool replaced by prospicio's (first age kept, no
   centring, zeros kept, the `sqrt(N / (N - p))` factor), because with its
   native pool the means differ by up to 16% on RAA. Its refits use the
   observed exclusions as fixed drop positions. RAA under the window with
   both drops is heavy tailed (outliers of 1e8 on a median of 9e4), so its
   medians are compared, not its means.
4. The first-order revision of item 7 is checked on Mack's bootstrap only
   (RAA with `n_periods = 3`, GenIns with `n_periods = 5`, within 4
   standard errors plus 2%). The ODP bootstrap's uncentred residual pool
   biases its claims development result far from the first-order revision
   under `n_periods` (RAA 1990: -7,300 simulated against +335 predicted,
   and -700 without exclusions), so no assertion is made on it.
5. `Development::reject_exclusions` had no caller left and was removed.
6. Mack's bootstrap against Mack's standard error under the same settings
   (item 9) holds on RAA and GenIns under `n_periods = 5` and under one
   high and one low dropped, every origin and the total within 2 Monte
   Carlo standard errors at 20,000 simulations
   (`knowledge/findings/mack-bootstrap-under-exclusions-vs-mack.md`).
7. Bindings: Python's `OdpBootstrap` takes `n_periods`, `drop_high` and
   `drop_low` after `ldf_weight`, with getters and the settings in its
   `repr` when set; R's `odp_bootstrap()` takes them after `ldf_weight`.
   In R, `odp_one_year()` has no bootstrap settings of its own and takes the
   method's, and `mack_one_year()`'s `mack_n_periods`, `mack_drop_high` and
   `mack_drop_low` are the bootstrap's own: they must equal the method's
   or stay at their defaults. A bootstrap in Python that sets its own
   different settings gets a `ValueError` naming both.
8. The geometric average stays rejected in the bootstraps and their
   one-year views, and the closed form's rejection now points to
   `MackBootstrap::one_year`.

### 15. A latest diagonal with a selected pattern

Status: **Built** (2026-10-23), issue #544, branch `claude/diagonal-pr1`,
in Rust. This is the issue's first PR; the rest (a dedicated diagonal
type, development fitted on one triangle and applied to another, the
bindings) is planned in the issue and listed in the lane's Next.

A cedant's data, summarised by origin, often gives one latest value per
origin with its age, not a triangle, and the pattern comes from elsewhere
(a prior study, an industry triangle, a benchmark). `Triangle::from_long` already builds a triangle from such
rows: one observed cell per origin. The chain ladder and the expected-loss
methods then failed even with a fully selected pattern, with "factor from
development index 0: no origin is observed at both ages", because
`Development::fit_segment` asked `kept_links` for the age's link ratios
before it checked whether the age is fully selected. The excess-layer
leniency (`reserving-excess.md`, decision 8) covered only the estimate's
errors, which come later.

1. **A fully selected age with no link ratio is missing, not an error.**
   At an age whose selected factor has weight 1 and where no origin is
   observed at both ages, the fit leaves the triangle's own factor
   missing: `ldf` and `volume` NaN, no sigma, `n_links` 0, as the
   excess-layer leniency does. The check comes before `kept_links`. An
   age that is not fully selected keeps the old error with the same text,
   and so does a fully selected age whose link ratios are all removed by
   `n_periods` ("no link ratio is left after n_periods"): that is a
   setting at odds with the data, not a diagonal.
2. **The kept mask skips an age with no link ratio.** The bootstraps'
   `Development::kept_mask` no longer errors at such an age; it keeps
   none there, and the fit says whether the age is an error.
3. **What works on a latest diagonal.** The chain ladder, the expected
   loss method, Bornhuetter-Ferguson, Benktander and Cape Cod (with its
   `trend` and `decay`), with the pattern as age-to-age factors, factors
   to ultimate or percents developed, and any tail. An estimated tail
   (`Tail::Curve`, `Tail::Bondy`, `Tail::LogLinear`) is fitted to the
   factors in use, here the selected ones, so a diagonal takes the tail
   its full triangle would. The results are the same fit types, so
   `totals()`, `reserves()` and the long tables work unchanged. The
   pattern's ages follow decision 12 unchanged: each must be on the
   triangle's grid, which `from_long` builds from the youngest observed
   age to the oldest (so a pattern from 12 months does not fit a diagonal
   whose youngest origin is at 24 months), a cumulative pattern fills the
   ages between two of its own, and an age-to-age pattern must give every
   age the diagonal needs. Ages off the grid or before the youngest origin
   are for the dedicated diagonal type of the issue's next PR.
4. **What a latest diagonal is rejected by, by name.** Mack's model (and
   with it Merz and Wüthrich's closed form), Mack's bootstrap and the ODP
   bootstrap need link ratios for sigma, the scale and the residuals. A
   segment where every origin is observed at one age at most
   (`Segment::is_latest_diagonal_only`) gets the new
   `Error::LatestDiagonalOnly { method }`, which says the triangle holds a
   latest diagonal only and that the chain ladder and the expected-loss
   methods project it with a fully selected pattern. Before, all three
   failed with the chain ladder's link-ratio error ("factor from
   development index 0: no origin is observed at both ages"), or Mack's
   bootstrap with "every origin must be observed from the first age to
   its latest". A triangle with holes and no link ratio, or with link
   ratios at some ages and a fully selected age with none elsewhere, is
   not a latest diagonal and gets the errors it had: Mack fills the
   missing sigma where its interpolation can and otherwise says it
   cannot, and the bootstraps reject the holes or the missing factor.
5. **Checks** (`validation/tests/reserving_latest_diagonal.rs`): a
   six-origin diagonal by hand for the chain ladder (also on percents
   developed; the youngest origin's 150 at a factor to ultimate of 12/7
   gives 257.14), Bornhuetter-Ferguson (208.33 at an ELR of 0.7 on a
   premium of 200), Benktander, Cape Cod and the expected loss method;
   the latest diagonal cut from GenIns and RAA, with the full triangle's
   own factors selected as age-to-age factors and as factors to ultimate,
   gives every origin the full triangle's chain ladder ultimate; the
   diagonal cut from the CAS workers' compensation triangle gives the full
   triangle's Bornhuetter-Ferguson, Benktander and Cape Cod ultimates
   under the same selected pattern; the three estimated tails on GenIns's
   diagonal equal the full triangle's; a partial pattern keeps the
   link-ratio error at its first unselected age, and a fully selected age
   whose link ratios `n_periods` removes keeps its error; Mack, Mack's
   bootstrap and the ODP bootstrap return `LatestDiagonalOnly`, and a
   triangle with holes and no link ratio does not. chainladder-python's
   `DevelopmentConstant` on a latest diagonal is left for the dedicated
   type's PR, with the fit-and-apply path it compares against.

### 16. The latest diagonal as a type of its own

Status: **Built** (2026-10-23), issue #544, branch `claude/diagonal-pr2`,
in Rust. This is the issue's second PR. Items 2 to 6 answer the issue's
open questions; they are proposals for the user to accept, and the PR
lists them.

Decision 15 lets a triangle that holds only a latest diagonal reach the
chain ladder and the expected-loss methods. A latest diagonal on its own
is not a triangle, though: its ages need not lie on one grid, the
pattern's ages need not be the diagonal's, and a triangle trims its development axis to the ages it
observes, so a pattern that starts before the youngest origin's age does
not fit it. `LatestDiagonal` holds the latest values as they are, and
`fit_diagonal` on `ChainLadder`, `ExpectedLoss`, `BornhuetterFerguson`,
`Benktander` and `CapeCod` projects it, returning the same fit types.

1. **One code path.** The type places the diagonal on a grid of its own
   (the crate's `Segment`, built directly, since a triangle would trim
   it) and calls the same `fit_segment` as the triangle methods, through
   decision 15's fully selected path. On a diagonal whose ages are the
   pattern's, the fits are exactly the triangle path's (checked to the
   bit). The fit's development axis is the grid, which spans the
   pattern's ages as well as the diagonal's.
2. **The grid.** Its ages are the diagonal's and the pattern's together,
   from the earliest to the latest, at the coarsest development grain
   (year, half year, quarter, month) that divides the origin grain and
   every gap between them. An age between two of the pattern's ages (18
   months against an annual pattern) is read by decision 12's rule, `ln
   CDF` linear in the age.
3. **Age-to-age factors.** A factor runs from its age to the next age
   the pattern gives, and the last one by the spacing before it (by the
   origin grain for a single factor), so a pattern of factors at 12 to 108
   months spans 12 to 120. On a grid finer than that spacing the factors
   are turned into factors to ultimate (their products, 1 at the end of
   the span), which the same rule spreads over the ages between. This
   also reads chainladder-python's convention of a factor of 1 at the
   oldest age as one more period of no development. The factors and any
   weights by age are checked before they are turned, so an error names
   the factor given, and the fit reports no pattern tail or rescaling,
   since factors carry no tail of their own. An estimated tail (curve,
   Bondy, log-linear) is fitted to the factors on the pattern's own ages,
   as a triangle at those ages would fit it, and applied on the finer
   grid as a constant; fitted to the interpolated factors of the shorter
   periods it would be another tail.
4. **Ages outside the pattern are an error** (open question 2), unless
   `PastPatternEnd` says otherwise (decision 18): an origin
   younger than the pattern's first age or older than its last (for
   age-to-age factors, the end of their span) names the origin, its age
   and the pattern's range. Development past the pattern is the tail's,
   applied from the grid's last age: an origin at the end of the span
   takes the tail alone.
5. **Ages from a valuation** (open question 1): `LatestDiagonal::at_valuation`
   gives each origin the months from its start to the valuation month
   inclusive, as a triangle's valuation dates do, so an accident year
   2024 valued at June 2024 is 6 months and an accident quarter in
   progress is its months so far.
6. **The rest of the open questions.** Every origin needs an exposure for
   the expected-loss methods, Cape Cod included, finite and positive (an
   error, not dropped; open question 4). The origins span every period
   from the earliest to the latest, and a skipped one is an error
   (`EmptyOrigin`); an origin with nothing reported is given as 0. An
   origin given twice is an error, not summed. The pattern must be fully
   selected (weight 1), since there is no estimate to blend with, and
   `n_periods` and the drops are an error, since they choose link ratios.
   There is no standard error (open question 3): no Mack, bootstrap or
   one-year view on a diagonal, and no user-given pattern uncertainty until
   asked for. Errors that only a diagonal has are the new
   `Error::Diagonal`. The average and sigma settings have no effect,
   since there are no link ratios. The fits list the origins oldest first
   over every period from the earliest to the latest, not in the order
   given. A grid of more than 2,400 development periods is an error.
7. **Checks** (`validation/tests/reserving_latest_diagonal.rs`): the type
   gives exactly the triangle path's ultimates on the diagonals of GenIns
   and RAA (age-to-age factors, factors to ultimate and percents
   developed) and of the CAS workers' compensation triangle (the expected
   loss method, Bornhuetter-Ferguson, Benktander and Cape Cod, with its
   trended apriori); chainladder-python 0.10.1's `DevelopmentConstant` on
   the latest diagonal, every origin and the total, at 1e-9 relative
   (`reference/reserving_latest_diagonal_python.csv`, from
   `scripts/reserving_latest_diagonal_python.py`); by hand, an origin at
   18 months on a half-year grid (`sqrt(2 × 1.25)`), accident quarters
   from 3 months and accident years at 13 and 25 months on a monthly
   grid, age-to-age factors on a finer grid equal to the same factors to
   ultimate, with and without a tail, an origin at the end of the factors'
   span taking the tail alone, an estimated tail on a half-year grid equal
   to the one at the pattern's annual ages; ages from a valuation equal a
   triangle's; and the errors above (`n_periods` and the drops, a blended
   pattern, no pattern, ages outside it, no exposure or a non-finite one, a
   repeated or skipped origin, an invalid factor or weight by age, a grid
   too large). chainladder-python has no ragged
   diagonal and gives an age its pattern does not list a factor of 1
   (`knowledge/references/chainladder-python-development-constant.md`), so
   the off-grid readings have no reference beyond the hand checks.

### 17. A fitted development applied to another book

Status: **Built** (2026-10-23), issue #544, branch `claude/diagonal-pr3`,
in Rust. This is the issue's third PR. Items 2 to 9 answer the issue's
open questions, and were accepted by the user on 2026-10-23 as built.
Item 11 stays open: `the_plain_path_on_an_older_book_mixes_in_its_own_factors`
still pins it, and decision 18 leaves the plain path's on-grid reading
unchanged. Decision 18 revises items 4 and 6 where they meet ages off the
source's or past its end. Decision 19 builds what item 9 left for later:
latest diagonals over segments, a source fit applied segment by segment,
and the Python and R arguments.

A reserving actuary often fits a development on one book (a mature line,
an industry triangle, last year's study) and applies it to another: a
younger book, a cedant's latest values by origin, the same book cut
differently. chainladder-python does it with `Chainladder().fit(source).predict(target)`.
Decisions 15 and 16 already project a latest diagonal with a fully
selected pattern, so a fit only has to become such a pattern, and another
triangle only has to give its latest diagonal.
`ChainLadderFit::to_chain_ladder` and `LatestDiagonal::from_triangle` do
the two.

1. **One code path.** A fit becomes a fully selected pattern and a
   constant tail, and is applied through decision 16's `fit_diagonal`
   and the shared `fit_segment`. There is no second projection kernel. The
   source can be any `ChainLadderFit`: a triangle's, one segment of a
   `SegmentFits`, the `chain_ladder` of an expected-loss fit or a fit made
   by `fit_diagonal`.
2. **The conversion.** `ChainLadderFit::to_chain_ladder()` returns a
   `ChainLadder` whose pattern is the factors in use (`ldf()`) as
   age-to-age factors at the fit's ages, with weight 1, and whose tail is
   `Tail::from(tail.factor)`, a constant at the oldest age. Every source
   setting is already in those two: the average and the exclusions, a
   blend frozen at its blended value, a cumulative pattern's
   interpolation and level, and an estimated tail, one attached inside
   the triangle included. The other `Development` settings are the
   defaults (no `n_periods`, no drops), as `fit_diagonal` requires. It
   does not carry sigma, the standard errors or the tail's form. On the
   fit's own ages the factors are the fit's to the bit; the tail is spread
   again by the constant tail's decay, so the tail factor, the factors to
   ultimate and the ultimates agree to about 1e-15 relative, and to the
   bit with no tail.
3. **Age-to-age factors, not factors to ultimate.** They are exact on the
   links. On the diagonal route the two kinds are the same, since decision
   16 turns factors into factors to ultimate on a finer grid. On the plain
   triangle path an age-to-age factor runs to the target's next age, so a
   target with a finer development grain is misread (item 4). Factors to
   ultimate (from `cdf`) would read such a target correctly on that path
   too, at about 1 ulp a link, and would set `pattern_tail` and
   `pattern_rescale`. Neither kind avoids the tail floor of item 5.
4. **Another triangle means its latest diagonal.**
   `LatestDiagonal::from_triangle(triangle, column, exposure)` takes each
   origin of a single-segment triangle at the age of its last observed
   cell, with an optional exposure column read as the expected-loss
   methods read it on a triangle. This is chainladder-python's predict:
   each origin's latest value times the source's factor to ultimate at its
   age. Apply a fit to any book other than its own this way. The plain
   triangle path (`to_chain_ladder().fit(&other, column)`) is right only
   on a target whose development ages are the source's, and has two
   hazards that give no error: a target with a finer development grain is
   misread, since an age-to-age factor runs to the target's next age (this
   came with decision 12); and on a target with ages past the source's
   oldest, the target's own factors are estimated there and the source's
   tail attaches at the target's oldest age, which mixes the target's
   development into the source's and is not what predict gives. A target
   without one of the source's ages failed with `Error::Pattern` (not a
   development age of the triangle); since decision 18 such a target has
   the source's factors read as decision 18 item 4 reads them, on the
   source's own ages, and the factors past the target's oldest age fold
   into a constant tail there (decision 18 item 5), which projects as
   `fit_diagonal` does. That covers a target with fewer ages and a target
   whose grain is coarser than the source's (an annual book under a
   quarterly fit). This PR adds no guard (item 11).
5. **The tail is carried as a constant, not refitted.** It attaches at
   the source's oldest age and is spread at decay 0.5; chainladder-python's
   predict also carries the source's tail factor. A tail factor below
   exp(-0.75), about 0.4724, cannot be spread at that decay (the seed's
   quadratic `a x^2 + b x - ln f = 0`, with `a = 4/3` and `b = 2` at
   decay 0.5, has no real root when `b^2 + 4 a ln f < 0`), so applying it
   is an error (`Error::Tail`); a factor just above it is spread.
6. **No extrapolation.** The grid is decision 16's: the source's ages, the
   end of the factors' span and the target's ages, at the coarsest grain
   they share. An origin younger than the source's first age or older than
   its oldest is `Error::Diagonal`, naming the origin, its age and the
   range (an older one can take a factor of 1 or the tail with
   `PastPatternEnd`, decision 18). A target with later origins, and so fewer ages than the source
   (GenIns from 2006), works, since the grid keeps the source's ages. A
   fit's ages are evenly spaced (a triangle's development axis, or
   decision 16's grid), so the factors' span ends at the source's oldest
   age.
7. **Expected-loss methods.** They take the frozen chain ladder in their
   `chain_ladder` field. Bornhuetter-Ferguson and Benktander apply their
   own apriori to the target's exposure. Cape Cod estimates its apriori on
   the target's exposure and latest values with the source's pattern, and
   trends it to the diagonal's valuation, as chainladder-python's
   `CapeCod.predict` does. From a triangle, `from_triangle`'s diagonal
   holds the fitted column alone, so the trend runs to that column's own
   latest diagonal, where `CapeCod::fit` trends to the triangle's
   valuation, the latest month of any column: when another column (a
   premium) runs later, the trended aprioris differ by the trend over the
   gap, and the detrended aprioris and the ultimates agree as long as
   every origin ends by the fitted column's valuation. To carry the
   source's apriori instead, use `BornhuetterFerguson` with it; that is
   exact at a trend of 0 only.
8. **Where it departs from chainladder-python.** An origin whose cell on
   the latest diagonal is missing is projected from its last observed cell,
   where chainladder-python gives NaN. A target valued earlier than the
   source is projected at its own ages, where chainladder-python values it
   at the source's valuation and gives NaN. Cape Cod on a diagonal reads
   each origin's own age, where chainladder-python's `CapeCod.predict` on
   a `latest_diagonal` reads every origin's apriori at the first factor to
   ultimate unless `val_to_dev()` is called first
   (`knowledge/references/chainladder-python-predict-traps.md`).
9. **A single segment and no standard error.** `from_triangle` takes one
   segment (`Error::MultipleSegments` otherwise). A map from source
   segments to target segments, `LatestDiagonal` over segments and the
   Python and R arguments come later; the bindings' PR unpacks
   `to_chain_ladder`'s pattern and tail into their existing `ldf`,
   `ldf_kind` and `tail` arguments. An applied fit reports the source's
   selection, factors and tail and the target's ultimates; its estimated
   factors, sigma and standard errors are the target's, NaN on a diagonal,
   and there is no Mack, bootstrap or one-year view on it (decision 16,
   item 6).
10. **Checks** (`validation/tests/reserving_fit_apply.rs`): GenIns and RAA
    fitted with each average, `n_periods`, the drops, a blended, a
    cumulative and a percent-developed pattern and each tail (an early
    attachment too), applied to their own latest diagonal and their own
    triangle, give the source's factors to the bit and its tail factor,
    factors to ultimate and ultimates within 1e-14 relative (to the bit
    with no tail), and so do both segments of a `ChainLadder::fit_segments`
    (GenIns and a perturbed GenIns, with a curve tail); the expected-loss
    methods on the CAS workers' compensation triangle give the source's
    development and exposure to the bit and its ultimates within 1e-12,
    and Cape Cod's aprioris too, since that triangle's premium runs no
    later than its paid (item 7). chainladder-python 0.10.1 at 1e-9
    relative (`reference/reserving_fit_apply_python.csv`, from
    `scripts/reserving_fit_apply_python.py`) in two arms, our own source
    fit through `to_chain_ladder` and the reference's own `cdf_` as a
    pattern of factors to ultimate. On GenIns, `fit(source).predict(target)`
    for the sources with no tail, `TailConstant(1.05)` and both curves
    applied to the diagonal and a perturbed book, whose own curve tails,
    refitted in the first arm, differ from the carried ones by more than
    the tolerance; the source with no tail also applied to a doubled and a
    shifted book and the origins from 2006 and to 2004; and for a book
    valued before 2008, `DevelopmentConstant` with the source's `cdf_`,
    not predict (item 8). On the CAS workers' compensation triangle, predict
    for Bornhuetter-Ferguson, Benktander and two Cape Cods applied to the
    diagonal, a perturbed book with 10% more premium and a shifted book
    (there is no doubled book), and for the Cape Cods also the origins from
    1993 and the apriori on the raw latest diagonal, which predict reads
    with every origin at the first age and the test reproduces so (item
    8). The GenIns targets at the source's ages (the diagonal, the
    perturbed, doubled and shifted books and the origins to 2004) also go
    through the plain triangle path against the same rows. By hand: ages
    between the source's (18 and 30 months, accident quarters), a target
    with fewer ages and its plain triangle error, a target with more ages
    through the plain path (its own factors past the source's oldest age
    and the tail at its oldest, pinned; item 11), a missing latest cell, an
    earlier valuation, the applied selection, Cape Cod on a diagonal
    trended to the paid's valuation when the premium runs a year later
    (item 7), and the errors above, the tail floor from both sides. The
    interpolated, quarterly and truncated cases and the missing cell and
    earlier valuation of item 8 have no chainladder-python reference; the
    generator asserts that predict gives NaN on those two and misreads the
    raw diagonal, and raises if chainladder-python stops doing so. The
    targets are GenIns and the workers' compensation triangle scaled,
    sliced or shifted, not a second real book.
11. **Open: a guard on the plain triangle path.** A proposal for the user:
    `to_chain_ladder().fit(&other, column)` could refuse a target whose
    development ages are not the source's (a finer grain, or ages past the
    source's oldest), with an error that points to `from_triangle` and
    `fit_diagonal`. The frozen chain ladder would then have to carry its
    source's ages, which a `ChainLadder` with a selected pattern does not,
    so the guard touches decision 12's selected patterns too. Until it is
    decided the hazards stand as item 4 states them, and
    `the_plain_path_on_an_older_book_mixes_in_its_own_factors` pins
    today's behaviour so that a guard changes it on purpose.

### 18. A pattern on another origin basis, and ages past its end

Status: **Built** (2026-10-23), issue #544, branch `claude/diagonal-basis`,
in Rust. This is the issue's fourth PR; decision 19 brings it to Python
and R. It answers the user's decisions of 2026-10-23: (b) a monthly (or any
finer) pattern applies to yearly or quarterly origins on both the
diagonal route and the plain triangle path, read at the origin's age or
averaged over the origin period's months; (c) an origin older than the
pattern's end takes a factor of 1, an a-priori tail or a tail estimated on
the pattern's own factors, and with none chosen stays an error naming the
three; (d) an origin at age 0 stays an error (`LatestDiagonal::new`
rejects it with `NonPositiveAge`). Items 1 to 10 were proposals, accepted by the user on 2026-10-24
as built; item 11 was open, decided by the user on 2026-10-24 as built. It revises decision 12's rule 1 and item 4 (the "not a
development age" check), decision 16's item 4 and decision 17's items 4
and 6, and adds no breaking change: `SelectedPattern` and `ChainLadderFit`
keep their fields, and `LatestDiagonal`'s fields are private.

1. **The origin basis is a conversion, not a field.**
   `SelectedPattern::averaged_over_origins(pattern_origin, origin)` turns
   a pattern measured on origin periods of `pattern_origin` (an
   accident-month pattern is `Grain::Month`) into the pattern of origin
   periods of `origin`, a whole multiple of it, with the exposure even
   through the period. An origin of `n = origin / pattern_origin`
   sub-periods of `s` months is at age `a` when its `q`-th sub-period
   from the start (`q = 0` the earliest) is at age `a - q s`, so

   ```text
   P(a) = (1/n) * sum_{q=0}^{n-1} p(a - q s)
   ```

   on percents developed, never on CDFs or `ln CDF`. Ages follow
   `at_valuation`: a sub-period at the end of its own months is at age
   `s`. The result is a `PercentDeveloped` pattern at every month from the
   input's first age to its end `E` plus `(n - 1) s`, `E'`, and goes into
   `Development::pattern` like any other, so it reaches both routes, Mack,
   both bootstraps and the one-year views. The default basis, read at the
   origin's age, is the pattern as given, with no call. A field on
   `SelectedPattern` or `ChainLadderFit` would break their struct
   literals, which the crate itself builds. An origin still in progress
   (an accident year at June, 6 months) projects to the whole period's
   ultimate. `All(w)` carries over; `ByAge` weights do not (weight the
   result); `n = 1` returns the pattern unchanged; a single age-to-age
   factor has no span to average; a result of more than 2,400 months is an
   error.
2. **Below the first age the share is linear from 0.** `p` is 0 at an age
   of 0 or less (a sub-period that has not begun), linear in `p` from
   `(0, 0)` to `(first, p(first))` between, decision 12's rule between the
   given ages (an age-to-age pattern first turned into factors to ultimate,
   1 at the end of its span), and held at its value at the end past it.
   `ln CDF` interpolation cannot start from `p = 0`, and an error would
   forbid accident-month patterns that start at 3 or 12 months; the rule is
   exact for a pattern linear from 0.
3. **Exact to the pattern's end, deferred after it.** For origins at ages
   up to `E` the ultimates are exact under even exposure, whatever the
   tail, since every sub-period reaches ultimate the same way and the
   average is linear. Between `E` and `E'`, `P` holds `p(E)` for the
   sub-periods past the end, so the input's own tail is taken to develop
   nothing until the period's last age: development is deferred and the
   ultimate slightly overstated. With a tail `T` on the result, decision
   12's rescale reports `pattern_rescale = P(E') / T`, which is `1 / T` for
   an age-to-age input. An estimated tail on an averaged pattern is fitted
   on its monthly factors (decision 16 item 3), so `TailCurve`'s default
   100 extrapolated periods are 100 months; a constant tail taken from the
   accident-month pattern's own tail is the safer choice, and the docs say
   so. Uneven exposure through the period projects each origin to the sum
   of its months' reported values over `P`, not to the sum of their
   ultimates; the test computes both by hand and pins the miss, `sum_j
   U_j p(a - j) / P(a) - sum_j U_j`.
4. **The plain path has one gate.** In `SelectedPattern::resolve(ages)`,
   when every age the pattern gives is a development age of the triangle,
   decision 12 applies unchanged, to the bit: sparse age-to-age patterns,
   the error for an age-to-age value at the oldest age, and decision 17
   item 11's misreading of a finer-grained or older target included.
   Otherwise the pattern is read on its own ages as decision 16 reads it,
   where every such pattern was an error before: the value checks first,
   with their texts (an age given twice, a value that is not finite and
   positive, one cumulative value); a single age-to-age factor off the
   triangle's ages keeps "not a development age of the triangle"; a
   cumulative pattern ends at its last age, an age-to-age factor runs to
   the next age given and the last one by the spacing before it, and `C`
   is `ln`-linear between the pattern's own ages (its given value at one).
   A link `a_k` to `a_{k+1}` is selected when `f0 <= a_k` and `a_{k+1} <=
   E`: the product of the age-to-age factors over it when both ends are
   the pattern's own (a link of one factor is that factor to the bit), and
   `C(a_k) / C(a_{k+1})` otherwise. Links that start before `f0`, or end
   past `E` while `E <= a_n`, stay estimated. A cumulative pattern's level
   is `C` at the last triangle age in `[f0, min(E, a_n)]`, the implied tail
   when that is `a_n`; an age-to-age pattern has none. A pattern that
   selects no link and does not fold is "the pattern selects no factor of
   the triangle". `ByAge` weights stay keyed by the triangle's link ages.
   So a monthly cumulative pattern on an annual triangle is its annual
   values to the bit, and a pattern at 6, 18 and 30 months on 12, 24 and
   36 is read between its ages (12 to 24 is `sqrt(C(6) / C(30))`).
   Adding one off-grid age to a sparse age-to-age pattern changes how its
   other factors are read, from decision 12's (each to the triangle's next
   age) to decision 16's (each to the pattern's next): `{12: 1.5, 18: 1.1,
   36: 1.2}` on RAA selects 12 to 24 at 1.5484, 24 to 36 at 1.0656 and 36
   to 48 at `1.2^(2/3)`, where `{12: 1.5, 36: 1.2}` selects 1.5 and 1.2.
   Nothing warns of this: `ChainLadderFit` has no warnings, and an error
   would also refuse a monthly pattern, whose factor at 12 months runs to
   13 rather than 24 just as 1.5 does to 18. The rustdoc says so, and the
   Python and R help of PR 4 must say it next to `ldf`. The two routes
   already disagreed before this PR on sparse age-to-age
   patterns on the grid: `{36, 96}` on RAA runs 36 to 48 and 96 to 108 on
   the plain path and 36 to 96 and 96 to 156 on the diagonal.
5. **A pattern past the oldest age folds into the tail.** When the
   off-grid reading's end `E` is past the triangle's oldest age `a_n` (and
   `f0 <= a_n`), what the pattern gives from `a_n` to `E`, and the tail
   after `E`, become one constant tail at `a_n`: with no tail, `C(a_n)`,
   the level included (decision 12 item 5's "the level is the tail" carried
   to `a_n`, the same as the pattern cut at `a_n`); with a `TailConstant`
   `T` (its attachment age unset), `C(a_n) / C(E) * T`; with an estimated
   tail, `T` fitted on the pattern's own ages as decision 16 fits it; a
   `TailConstant` with an attachment age is "a pattern that runs past the
   oldest age takes the tail at its end: leave the tail's attachment age
   unset". `pattern_tail` is the folded factor, set with or without a
   tail, `tail.std_err` is 0 and sigma is extrapolated as a constant
   tail's; for a cumulative pattern `pattern_rescale` is `C(E) / T` (1 with
   no tail). The folded tail is a `TailConstant` at `a_n` that keeps the
   given constant's decay, which spreads it over the periods past `a_n`
   (the bootstraps' tail step and the one-year view's closing of origins,
   decisions 9 and 11); a fold of an estimated tail takes decay 0.5, as the
   diagonal route carries one. The fold takes the pattern at full weight,
   so a pattern that folds must be fully selected: `All(w)` with `w < 1`,
   or any `ByAge` weight below 1, is "a pattern that runs past the oldest
   age folds into the tail at full weight: give it weight 1, or cut it at
   the oldest age to blend it" (`Error::Pattern` at `a_n`). A blend of the
   fold with the tail the triangle would give (`ln T = w ln(fold) + (1 -
   w) ln T_triangle`) would need a weight for a span that no link of the
   triangle starts, and a parameter error for the blended tail; neither is
   designed, so it is an error until the user asks for it. On the diagonal
   route every pattern is fully selected anyway (decision 16). The
   bootstraps keep the fold constant in every simulation, never refitted.
   A `tail_std_err` is read against the tail given: with an estimated tail
   it is the error it always was, folded or not, and with a constant `T`
   the folded factor is drawn with `T`'s coefficient of variation,
   `tail_std_err / T`, since the pattern's development from `a_n` to `E`
   is fixed. Mack's bootstrap draws no tail error on a fold, whose
   `tail.std_err` is 0. So the plain path now projects a target with fewer ages
   (GenIns from 2006) as `fit_diagonal` does, which changes decision 17
   item 4's error.
6. **`PastPatternEnd` on `LatestDiagonal`.**
   `LatestDiagonal::with_past_pattern_end(PastPatternEnd)`, with
   `Reject` (the default), `NoDevelopment` and `Tail`, and the getter
   `past_pattern_end()`. On the data rather than the method, because it is
   the only home that breaks nothing; the bindings must expose it on the
   diagonal. `Reject` keeps decision 16's message and adds a suffix that
   names the two other settings and the tails `Tail` takes (an a-priori
   `TailConstant`, or `TailCurve`, `TailBondy` or `Tail::LogLinear`
   estimated on the pattern's own factors); an origin younger than the
   first age keeps its message unchanged. `NoDevelopment` with a tail,
   `Tail` with none, and `Tail` with a `TailConstant` whose attachment age
   is set are `Error::Diagonal`. With no origin past `E`, `Tail` is
   today's fit to the bit, and `NoDevelopment` is a `TailConstant` of 1
   attached at `E`, the grid's oldest age: today's fit for age-to-age
   factors or a cumulative pattern at 100%, and another level rescaled to
   100% at `E`. So with a level other than 100% `NoDevelopment` moves
   every origin's ultimate, the young ones too, also when no origin is
   past `E`: 95% at 120 months on accident years at 120, 108 and 12
   months (40% at 12) gives the youngest 237.5 under `NoDevelopment`,
   where `Reject` keeps the level as the implied tail and gives 250. It
   is decision 12 item 5's rescale under a tail of 1, and the docs of
   `NoDevelopment` say so; an error for a level other than 100% under
   `NoDevelopment` is the alternative. On the plain triangle path no setting applies: links past
   a shorter pattern are estimated (decision 12), and a triangle that
   holds only a latest diagonal goes through `LatestDiagonal::from_triangle`
   to use the three choices.
7. **The spread steps by the origin grain.** With some origin past `E`,
   the tail's factor `T` (1 for `NoDevelopment`; the constant's factor and
   decay; an estimated tail's factor from the pattern's own ages at decay
   0.5) is spread from `E` as a `TailConstant` attached there over `ext =
   [E, E + g, ..., E + K g]`, `g` the origin grain in months and `K` the
   least integer with `E + K g` at or past the oldest origin's age;
   `R(k)`, the product of the spread's factors from step `k`, is what is
   left of `T` after `k` steps (`R(0) = T`). The pattern on the grid is
   one pattern of factors to ultimate: `C(v) / C(E) * T` at the pattern's
   own ages and `R(k)` at `E + k g`, with `Tail::from(R(K))` at the grid's
   oldest age; an origin between two of those ages reads `ln`-linearly
   (half a year past the end, `T / sqrt(1 + x)`). The fit reports
   `tail.factor = R(K)`, `pattern_rescale = C(E) / T` for a cumulative
   pattern (`None` for age-to-age factors) and no `pattern_tail`. At decay
   0.5, `T = 1.05` and `x = 0.02401073837969081` (the root of `(4/3) x^2 +
   2 x - ln 1.05 = 0`), an annual book one year past the end gets
   `1.0253798721500054`, chainladder-python's 1.02538 at 132 months on
   RAA (`tail.rs`), and two years `1.0132158418990078`; accident quarters
   one quarter past the end get the same `1.0253798721500054`. Stepping by
   the pattern's grain instead would make the spread depend on it: a
   monthly step gives `1.0011602413037541` one year past the end. An
   origin between two steps reads `ln`-linearly, so accident years valued
   at March, the oldest at 123 months, take `1.05 (1 + x)^(-1/4) =
   1.0437900950394003` there, where an accident quarter at 123 months, one
   step past the end, takes `1.0253798721500054`: the decay is per origin
   period, and the remaining development at a given age depends on how the
   origins are grouped (item 11). An estimated tail's own extrapolated factors are
   not used past `E`; it is carried as a constant, as decisions 16 and 17
   carry it. The floor `exp(-0.75) = 0.4723665527410147` at decay 0.5 is
   `Error::Tail`, from the spread's seed.
8. **The level-free one-year view refuses a fold.** A bootstrap with no
   tail that takes the method's pattern while the method has a tail reads
   the pattern without its level (`inherits_level_free`, decision 12).
   That cannot carry the method's tail into a pattern past the oldest age,
   so it is `Error::Bootstrap("a pattern that runs past the oldest age
   carries the re-reserving method's tail into it: give the bootstrap the
   method's tail")`, raised where the level-free pattern is built on the
   segment's ages, in both bootstraps (the bootstraps' `for_method` has no
   segment). With no tail on either side, or the same tail on both, each
   folds the same constant and the tail check passes. A follow-up could
   hand the bootstrap the method's tail instead. Merz and Wüthrich's
   closed form rejects a folded tail other than 1 ("needs no tail
   factor") with no new code, and `same_selection` compares resolved
   forms, so a monthly cumulative bootstrap pattern equals an annual method
   pattern of the same values.
9. **Estimated tails differ between the routes in one case.** On the
   triangle path a pattern that ends at or before `a_n` has its estimated
   tail fitted on the blended factors at the triangle's ages (decision 12,
   unchanged), and one that runs past `a_n` on the pattern's own ages
   (item 5); on the diagonal route always on the pattern's own ages
   (decision 16). So a finer pattern that ends exactly at `a_n` gets a tail
   fitted on annual products on the triangle path and on monthly factors
   on the diagonal. Constant tails agree.
10. **Checks** (`validation/tests/reserving_pattern_basis.rs`, unit tests in
    `pattern.rs`, and the changed tests of `reserving_fit_apply.rs` and
    `reserving_latest_diagonal.rs`): averaging `x/24`, `t/12`, `t/60` and
    `1 - 0.9^t` against their closed forms to 1e-15 (1e-14 for the
    exponential), the rule below the first age, the age-to-age input equal
    to its factors-to-ultimate twin, the errors; a synthetic book of
    accident months reported along `1 - 0.9^t`, summed to accident years
    and quarters, whose chain ladder with the averaged pattern gives each
    origin the sum of its months' ultimates to 1e-12 on the plain path and
    through its diagonal, an accident year at June included, and with
    uneven months the miss pinned against its value computed by hand; a
    monthly cumulative pattern on RAA and
    GenIns equal to its annual values to the bit for the chain ladder,
    Mack (fully selected and blended), both bootstraps' draws and both
    one-year views; monthly age-to-age factors multiplying to the annual
    ones, GenIns's factors spread monthly giving its chain ladder; the
    misaligned and the sparse age-to-age readings above; RAA's own pattern
    times 1.1 past 120 months folded with no tail (1.1), a constant
    (1.155), a constant's decay kept, a curve (equal to the diagonal route
    to 1e-14), the attachment error and the weight error (`All(0)`,
    `All(0.5)`, a `ByAge` weight); a monthly pattern ending at 120 months
    with a curve tail, whose tail differs between the routes (item 9); on
    the folded pattern, Mack and both bootstraps' lifetime and one-year
    draws equal to the pattern cut at 120 months, whose implied tail is
    the same 1.1, a curve tail with a `tail_std_err` the error again, and
    a constant's `tail_std_err` drawn at its own coefficient of variation;
    on the diagonal the spread's numbers above on annual, half-year and
    quarterly grids, the March book's `1.0437900950394003`, a monthly
    pattern equal to the annual one past the end, `NoDevelopment` (a
    level of 95% rescaled by 1.0526315789473684, the youngest 237.5
    against 250 under `Reject` with no origin past the end), estimated
    tails equal to the constant spread of their factor, `Tail` and
    `NoDevelopment` for age-to-age factors changing nothing to the bit with
    no origin past the end, a single age-to-age factor with a curve tail
    past the end, the errors and the floor, Bornhuetter-Ferguson,
    Benktander and Cape Cod past the end, and a development fitted on
    GenIns from 2006 applied to the whole GenIns diagonal; the one-year
    views with the folded tail and the level-free error, and Merz and
    Wüthrich's. No
    reference averages a pattern over origin bases: chainladder-python
    0.10.1's `DevelopmentConstant(patterns, style, callable_axis, groupby)`
    has no origin-basis argument and keeps the triangle's grain (confirmed
    with uv on 2026-10-23), so these checks are by hand or on synthetic
    books.
11. **The spread's step** (decided by the user on 2026-10-24: per
    origin period, as built). Item 7 steps the tail past `E` by the
    origin grain, with the decay applied per step, so the run-off of the
    tail in calendar time depends on how the origins are grouped: at `T =
    1.05` and decay 0.5 an accident year one year past the end and an
    accident quarter one quarter past it both take `1.0253798721500054`,
    so a quarterly book releases in 3 months what an annual book releases
    in 12. chainladder-python ties its decay to the development period,
    which is the precedent for the choice as built. The user's decision
    (c) asks for "the tail's remaining development from its own age",
    which can also be read as a function of age alone. The alternative is
    a fixed calendar step of 12 months, with origins between the steps
    read `ln`-linearly: it depends on neither grain, and gives an accident
    quarter at 123 months `1.0437900950394003`, the March book's value of
    item 7, where the build gives `1.0253798721500054`. Annual books are
    the same under both. The user chose the build: the decay is per
    origin period (`PastPatternEnd::Tail`), as chainladder-python ties it
    to the development period.

### 19. Latest diagonals over segments, and the bindings

Status: **Built** (2026-10-24), issue #544, branch `claude/diagonal-bindings`,
in Rust, Python and R. This is the issue's fifth PR. It carries decisions
15 to 18 into the bindings and adds what decision 17 item 9 left for later:
latest diagonals over segments and a source fit applied segment by segment.
Of items 1 to 12 below, the user decided four on 2026-10-24: items 3, 5
and 11 accepted as built, and item 8 with other names (R's
`latest_valuation()` and `as_latest_valuation()`, not `bordereau()` and
`as_bordereau()`). The other eight, items 1, 2, 4, 6, 7, 9, 10 and 12,
were accepted by the user on 2026-10-25 as built. The Excel functions
are xll.md decision 39.

**Rust** (`crates/prospicio-reserving/src/latest_diagonal.rs`, re-exported
from `lib.rs`):

- `LatestDiagonals { key_names, labels, diagonals }`: one `LatestDiagonal`
  per segment, the labels sorted as `Triangle` sorts them. `new(key_names,
  segments)` checks the labels (one part per key, none given twice) and one
  origin grain; `From<LatestDiagonal>` is one segment without keys.
- `from_long(&LongDiagonal)`: key columns, origins, `DiagonalAge::Ages`,
  `Valuation` (one month for every row) or `Valuations` (one per row), the
  latest values, an optional exposure and the origin grain. The ages,
  values and exposure are checked over the whole input first, so a row
  number in `NonPositiveAge` or `NonFinite` is the input's row; then the
  rows are grouped by key, and a repeated origin within a segment is an
  error that names the segment.
- `from_triangle(&Triangle, column, exposure)`: the latest diagonal of
  every segment, over the trimmed `triangle.segments(column)`, through the
  same private `LatestDiagonal::from_segment` as `LatestDiagonal::from_triangle`
  (which still refuses several segments).
- `with_exposure` (values in row order: the segments in index order, the
  origins in the order given), `with_past_pattern_end` (every segment), and
  `key_names`, `labels`, `diagonals`, `len`, `is_empty`, `get`, `iter`,
  `position`, `segment`, `origin_grain` and `valuation` (the latest month
  any origin of any segment is valued at). `LatestDiagonal::valuation` is
  new and public too.
- `developments(&SegmentFits<ChainLadderFit>, &SegmentMap)`: one frozen
  `to_chain_ladder()` per segment of the diagonals, keyed by their labels.
  `SegmentMap::ByLabel` (the default) takes the source segment of the same
  label, or broadcasts a source of one segment to every segment;
  `SegmentMap::Pairs(Vec<(target, source)>)` names each.
- On `ChainLadder`, `ExpectedLoss`, `BornhuetterFerguson`, `Benktander` and
  `CapeCod`: `fit_diagonal_segments(&LatestDiagonals)` and
  `fit_diagonal_segments_from(&LatestDiagonals, &SegmentFits<ChainLadder>)`,
  returning `SegmentFits` of each method's fit, the second with each
  segment's chain ladder taken from the development.

**Semantics.** Each segment runs the existing `fit_diagonal`, so one
segment without keys gives exactly `LatestDiagonal`'s fit, to the bit. An
error in a segment is wrapped in `Error::InSegment` when there are keys, as
`fit_segments` wraps it (`segments::collect_labelled`, which `collect_fits`
now calls). Cape Cod trends every segment to the one common valuation, as
`CapeCod::fit_segments` trends every segment to the triangle's valuation
(a private `fit_diagonal_at`). With a development, the method's own chain
ladder must have no pattern, no tail (a constant of 1) and no link ratio
exclusions; the apriori, `n_iters`, `trend` and `decay` stay the method's.
The development's labels must be the diagonals'. The plain triangle path
is unchanged (decision 17 item 11 stays open).

**Errors**, each naming the label: a segment of the diagonals with no
source segment of its label ("segment b of the diagonals has no segment of
the same label in the source"), source and diagonals with different key
names under `ByLabel` ("map the segments with SegmentMap::Pairs"), a map's
target that is not a segment, a target listed twice, a source that is not
a segment of the source, a segment the map gives no source; a method that
sets its own development ("the development comes from the source fit: give
the method no pattern, no tail and no link ratio exclusions"); and
developments whose segments are not the diagonals'.

**Python** (`crates/prospicio-python/src/reserving.rs`,
`python/prospicio/reserving.py`):

- `LatestDiagonal(origin, age, latest, exposure=None, keys=None,
  origin_grain="Y", past_pattern_end="reject")`, `LatestDiagonal.at_valuation`
  (the valuation a date: an integer is a `TypeError`, since a year would
  read as January), `LatestDiagonal.from_frame` (exactly one of `age`, a
  column, and `valuation`, a date or a column of dates) and
  `LatestDiagonal.from_triangle` (any number of segments); `with_exposure`,
  `with_past_pattern_end` (`"reject"`, `"no_development"` or `"tail"`),
  `segment(**keys)`, `to_frame()`, and the properties `keys`, `index`,
  `origin_grain`, `origins`, `ages`, `latest`, `exposure` and
  `past_pattern_end`. It wraps `LatestDiagonals`.
- `fit_diagonal(diagonal, development=None, segment_map=None)` on
  `ChainLadder`, `ExpectedLoss`, `BornhuetterFerguson`, `Benktander` and
  `CapeCod`. `development` is a `ChainLadderFit` (another method's fit
  passes its `.chain_ladder`); `segment_map` is a dict from a segment of the
  diagonal to one of the development, labelled as `index` labels them.
  The source's factors are read at each origin's own age, with no
  `ldf_origin`: a source fitted on another origin grain goes through
  `to_chain_ladder()` with `ldf_origin` added, one segment at a time, and
  the help of `development` says so in Python and R.
- `ChainLadderFit.to_chain_ladder()`, for a single-segment fit: a
  `ChainLadder` whose `ldf` (a dict), `ldf_kind == "ldf"` and `tail` (a
  float) unpack into any method's arguments with the same projection, to
  the bit (decision 17 item 9).
- `ldf_origin` (`"M"`, `"Q"`, `"S"` or `"Y"`) on every method that takes
  `ldf`: `ChainLadder`, `Mack`, `ExpectedLoss`, `BornhuetterFerguson`,
  `Benktander`, `CapeCod`, `OdpBootstrap` and `MackBootstrap`. It is stored
  on the method and applied at fit time with the triangle's or diagonal's
  origin grain (`SelectedPattern::averaged_over_origins`), in `fit`,
  `fit_diagonal`, the bootstraps and the one-year views. The `ldf` getter
  returns the pattern as given.
- The `ldf` help no longer says each age must be a development age of the
  triangle: ages off the grid are read on the pattern's own ages and what
  runs past the oldest age folds into a constant tail (decision 18). It
  warns that one off-grid age changes how a sparse age-to-age pattern's
  other factors are read (on RAA, `{12: 1.5, 18: 1.1, 36: 1.2}` selects
  1.5484 from 12 to 24 months where `{12: 1.5, 36: 1.2}` selects 1.5).

**R** (`crates/prospicio-r/src/reserving.rs`, `R/prospicio/R/reserving.R`):

- An S7 class `latest_valuation` over `LatestDiagonals`, each origin's
  latest valuation: `latest_valuation(data, origin, latest, age = NULL,
  valuation = NULL, exposure = NULL, keys = NULL, origin_grain = "Y",
  past_pattern_end = "reject")`, `as_latest_valuation(triangle, column =
  NULL, exposure = NULL, past_pattern_end = "reject")`,
  `with_exposure()`, `with_past_pattern_end()`, `segment()`,
  `as.data.frame()` and `print()`, with the same properties as Python's.
- `chain_ladder()`, `expected_loss()`, `bornhuetter_ferguson()`,
  `benktander()` and `cape_cod()` take a latest valuation as their first
  argument (with no `column` or `exposure`), and `development` (a
  `chain_ladder_fit` or a fit with `@chain_ladder`) and `segment_map` (a
  named character vector, target to source, each label its key values
  joined by `" / "`). `development` with a triangle is an error that
  points to `as_latest_valuation()`. `column` and `exposure` of the
  expected-loss methods now default to `NULL`, so that a latest valuation
  can leave them out; with a triangle they are still required.
- `to_chain_ladder(fit)` returns `list(ldf, ldf_kind = "ldf", tail)`, `ldf`
  named by the age that starts each link, for `do.call()`.
- `ldf_origin = NULL` on every function that takes `ldf`.
- `mack()`, the bootstraps and the one-year views reject a latest
  valuation with a message naming the five functions that take one.

**Checks.** `validation/tests/reserving_diagonal_segments.rs` (Rust): each
segment takes the source segment of its label, to the bit for the factors
and within 1e-14 for the ultimates; a map of pairs and a broadcast source
are bit-equal to the per-segment application; the expected-loss methods
over segments equal each segment fitted alone; Cape Cod with segment b
valued a year before a trends b by `(1 + trend)^1`; `from_long` with keys
and valuations equals `from_triangle`; and every error. The Python and R
tests (`python/tests/test_reserving_latest_diagonal.py`,
`R/prospicio/tests/test-reserving-latest-diagonal.R`) reproduce the Rust
validation's numbers through each binding: the hand diagonals, half-year
origins, the rule past the pattern's end, `ldf_origin`, RAA's off-grid
reading, every row of `reserving_latest_diagonal_python.csv` and of
`reserving_fit_apply_python.csv` (both arms, with the `development=` route
equal to the unpacked route to the bit), the segments and the
constructors.

**Decision items** (all decided: items 3, 5, 8 and 11 by the user on
2026-10-24, the others accepted by the user on 2026-10-25 as built):

1. **A separate `LatestDiagonals` type** (accepted by the user on 2026-10-25 as built), rather than
   keys on `LatestDiagonal`, whose per-origin accessors would change
   meaning. Python's `LatestDiagonal` and R's `latest_valuation` wrap it,
   so one segment without keys is the plain case.
2. **Segment order** (accepted by the user on 2026-10-25 as built): sorted by label, as `Triangle`
   sorts it, not the input's order.
3. **Cape Cod across segments trends to one common valuation** (the latest
   of all segments), as `CapeCod::fit_segments` does (decided by the user
   on 2026-10-24: one common valuation, as built). The alternative is
   each segment's own valuation.
4. **The map** (accepted by the user on 2026-10-25 as built): by label, with a source of one segment
   broadcast; unused source segments ignored and an unmatched target an
   error; explicit `Pairs` from target to source, which in the bindings
   must name every target. An unmatched target does not fall back to a
   broadcast, and a partial map is not filled by label.
5. **A development source and the method's own settings** (decided by
   the user on 2026-10-24: as built): the method's pattern, tail and link
   ratio exclusions must be unset, an error rather than silently
   overridden; the apriori, `n_iters`, `trend` and `decay`
   stay the method's. The exclusions are refused too, beyond the design's
   pattern and tail, since they would otherwise be ignored silently.
6. **Python `to_chain_ladder()` needs a single segment** (accepted by the user on 2026-10-25 as built).
   Per-segment application goes through `fit_diagonal(..., development=)`.
   A dict of chain ladders by segment can be added later without breaking
   this.
7. **`averaged_over_origins` as an `ldf_origin` argument** (accepted by the user on 2026-10-25 as built),
   stored on every method with `ldf` and applied at fit time with the
   target's grain. The
   alternative is a module function that returns arguments to unpack,
   which touches no method.
8. **R naming** (decided by the user on 2026-10-24: `latest_valuation()`
   and `as_latest_valuation()`, the S7 class `latest_valuation`). The
   proposal was `bordereau()` and `as_bordereau()`, because
   `latest_diagonal()` is taken. The user declined it: a bordereau (bdx)
   is the data extract a cedant sends, not a latest diagonal, so the name
   should say each origin's latest valuation. `latest_diagonal()` keeps
   its name and meaning (a triangle's latest values). The other
   alternatives were `diagonal_data()`, or extending `latest_diagonal()`.
9. **R dispatch** (accepted by the user on 2026-10-25 as built): inside the existing five functions,
   rather than `*_diagonal()` twin functions.
10. **A valuation must be a date** (accepted by the user on 2026-10-25 as built) in Python and R;
    valuations per row are supported (`DiagonalAge::Valuations`).
11. **Error wording** (decided by the user on 2026-10-24, as built):
    Rust messages pass through unchanged, so a `"reject"` error names
    `LatestDiagonal::with_past_pattern_end`, the
    errors of a tail under `"no_development"` or none under `"tail"` name
    `PastPatternEnd::NoDevelopment` and `PastPatternEnd::Tail`
    (`past_pattern_end` in Python and R), and a map by label between
    different key names ends "map the segments with SegmentMap::Pairs"
    (`segment_map` in Python and R). The alternative is
    to reword the Rust text in binding-neutral terms. The binding tests
    match only the neutral part of these messages, so either choice keeps
    them. One message is reworded here already, since its advice could not
    be followed from the bindings: a weight by age under `ldf_origin`
    (`averaged_over_origins`) now ends "give one weight for every factor"
    instead of naming `with_weight`.
12. **`development=` only on diagonals** (accepted by the user on 2026-10-25 as built). To apply
    another fit's pattern to a triangle, take the triangle's latest
    diagonal (`LatestDiagonal.from_triangle`, R `as_latest_valuation()`)
    and fit that with `development=`. The plain triangle path keeps
    decision 17 item 11 open.

### 20. A cumulative pattern with one value

Status: **Built** (2026-10-26), issue #632, branch
`claude/one-value-pattern`, in Rust, Python and R. The user decided on
2026-10-26 that a cumulative pattern with one value is valid and selects
the factor from its age to ultimate, as the tail from that age. Items 2
to 6, and item 1's reading of a longer pattern that ends at the oldest
age, are how this PR reads that decision against the other settings, and
are proposed for the user's acceptance.

A downstream pipeline fits small books: a diagonal-only triangle, and a
single origin observed at one age. Decision 12 item 4 refused a factor
to ultimate (or a percent developed) at one age, "one value selects no
factor", so a book observed at one age could take a pattern only as two
or more ages (which decision 18 reads past the oldest age since #614).

1. **The rule.** `AgeToUltimate` or `PercentDeveloped` with one value
   `CDF(a)` (`1 / p(a)`) at a development age `a` of the triangle is the
   factor to ultimate from `a`. It fixes no link, so by item 5 of
   decision 12 it is the tail when `a` is the oldest age and no tail is
   given: `pattern_tail = CDF(a)`, a constant with no parameter error,
   `pattern_rescale = 1`. On a triangle observed at one age (one origin,
   or every origin at one age, a latest diagonal included) the oldest age
   is the only age, so the value is the whole projection: the issue's
   `{12: 3.0}` on an origin at 12 months with 200 paid gives 600. On a
   full triangle it is the same fit as a constant tail of `CDF(a)`. A
   longer cumulative pattern that ends at the oldest age and selects no
   link of the triangle is read the same way (proposed), as its level there: on a
   triangle observed at one age the pattern's ages before it fix no link,
   so the issue's full pattern from 12 to 60 months on an origin at 60
   months gives the latest value (a factor of 1), and the pattern from 12
   to 48 months on an origin at 48 months with 800 paid gives 896, as
   `{48: 1.12}` does. Decision 18 had refused these as "the pattern
   selects no factor of the triangle", while a pattern that runs past the
   only age already folded into the tail. The same reading applies on a
   longer triangle when the pattern's earlier ages all fall inside its
   last link (`{30: x, 36: y}` on ages 12, 24, 36): the link from 24 to
   36 is not fully covered, so it stays estimated as decision 18 leaves
   any such link, and `y` is the tail.
2. **At another age it is an error** (proposed): "a cumulative pattern
   with one value is the tail from its age: give it at the oldest age, or
   give two or more ages". Read by decision 12's rules it would select
   nothing and only report `pattern_rescale`, which contradicts the
   user's decision. The literal alternative, a constant tail attached at
   `a` that replaces the estimated factors from `a` on, is not built: it
   needs a tail attached before the oldest age carried through
   `pattern_tail`, Mack and the bootstraps, and spreads the factor over
   the older ages by a decay the user did not give. Off the triangle's
   ages (`{18: 1.5}` on an annual triangle) it is decision 18's existing
   error, "the pattern selects no factor of the triangle".
3. **With a tail it is an error** (proposed): "a cumulative pattern with
   one value is the tail, and a tail is given: give no tail, or two or
   more ages". Decision 12 rescales a longer pattern's level to a given
   tail, which with one value would leave the pattern inert. On a latest
   diagonal, `PastPatternEnd::Tail` (which needs a tail) and
   `PastPatternEnd::NoDevelopment` (a factor of 1 past the value's age)
   are such tails, and give the same error whether or not an origin is
   older than the value's age: decision 18 would rescale the one level to
   the tail spread past the pattern's end, so nothing of the value would
   be left (`{12: 3.0}` and `{12: 9.0}` on origins at 24 and 12 months
   would give the same ultimates).
4. **A weight below 1 is an error** (proposed): "a cumulative pattern
   with one value is the tail, taken at full weight: give it weight 1".
   No estimate past the oldest age is there to blend with, as for a
   pattern that runs past the oldest age (decision 18). A weight by age
   at the value's age keeps decision 12's error, "a weight is given at an
   age with no selected factor".
5. **Bootstraps.** A bootstrap with no pattern of its own reads the
   method's pattern without its level when the method has a tail
   (`OneYearPattern::Method`). A one-value pattern there is only its
   level, so this happens only when the method's pattern and tail already
   conflict by item 3, and the bootstrap's error names that conflict:
   "the re-reserving method's pattern is only its level, its tail, and
   the method has a tail of its own: give the method no tail, or two or
   more ages". Where the method has no tail the one value is the
   method's implied tail and the bootstrap's, as any implied tail is.
   Comparing a bootstrap's pattern with the method's
   (`same_selection`) ignores levels, as before; the tails are compared
   on their own, so two one-value patterns with different values still
   differ there.
6. **`averaged_over_origins`.** One value is averaged by the rules as
   stated: linear from 0 at age 0 to the value at its age, held at the
   value past it, so an accident-month `{3: 0.5}` gives the accident year
   `1/12` at 3 months and `0.5` from 14 months. The result has many ages
   and is read as any pattern is.
7. **Checks.** `validation/tests/reserving_latest_diagonal.rs`: both
   cases of the issue through the plain triangle path at weight 1, by
   hand for the chain ladder, Bornhuetter-Ferguson, the expected loss
   method, Benktander's two iterations and Cape Cod (a diagonal-only
   triangle with the full pattern, one origin at 12 months with the full
   pattern, with `{12: 3.0}` and with `{12: 1/3}` percent developed, and
   one origin at 48 months), a pattern that ends at the only age (the
   full pattern on an origin at 60 months, the pattern to 48 months on an
   origin at 48 months, `{6: 4.0, 12: 3.0}` on an origin at 12 months),
   the one value through `LatestDiagonal`, and item 3's error on a
   diagonal with an origin older than the value, under
   `PastPatternEnd::Tail` and `NoDevelopment`.
   `validation/tests/reserving_pattern_basis.rs`: the errors of items 2
   to 4 on RAA, the one value at 120 months equal to a constant tail of
   1.05, item 5 (the ODP bootstrap with the one value draws as with a
   constant tail of 1.05, and both bootstraps' one-year views take it as
   the method's implied tail, and the bootstrap's error when the method
   has a tail too) and item 6. The same cases in `python/tests/test_reserving_patterns.py`
   and `R/prospicio/tests/test-reserving-patterns.R`.

### 21. An accident-period pattern on an underwriting basis

**Accepted** by the user on 2026-10-27: items 1 to 8 as built (writing through the month, coarser patterns converted only in the exact cases). Item 3's miss of an annual accident-year pattern in the first two years is #646, fixed by decision 25 (accepted 2026-10-27, built on branch `claude/ay-monthly-reading`): the reading between annual ages is now the Weibull through the averages, and item 3's rule is `AnnualReading::LnLinear`.

Status: **Built** (2026-10-26), issue #633, branch `claude/uy-pattern`, in
Rust, Python and R. It answers the user's decision of 2026-10-26: an
underwriting year's pattern is the accident pattern convolved with the
year's exposure over accident months,

```text
P_UY(a) = sum_t e(t) p_AY(a - t)
```

on percents developed, with `e` the writing pattern (monthly shares over
the underwriting year) convolved with even earning over the policy term,
and by default even writing over 12 months and 12-month policies (the
parallelogram, a triangular density over 24 months). The open points the
issue listed are items 1 to 8 below; each is a proposal for the user's
acceptance, built as written. Decision 18's `averaged_over_origins` is
unchanged to the bit (its checks moved to a shared helper, with the same
texts), and nothing breaks: `UnderwritingBasis` and the method are new,
and the bindings add one argument.

1. **Writing through the month** (proposed). A policy written in month `j`
   is written at an even rate through the month and earns evenly over `m`
   months, so its earning by accident month is `1/(2m)` in month `j`, `1/m`
   in each of the next `m - 1` and `1/(2m)` in month `j + m`;
   `UnderwritingBasis::exposure` is the writing convolved with that kernel,
   `n + m` months for `n` months of writing. The default gives `e(t) = (2t
   + 1)/288` for `t` up to 11 and `(47 - 2t)/288` from 12 to 23: the
   triangular density of the issue, integrated month by month, with its
   mean 11.5 months after the start (12 months in continuous time, less
   the half month an accident month's own age already counts). The
   alternative is writing at the start of each month, as
   `prospicio_pricing::cashflow::MonthlyPatterns::from_annual` writes it:
   a density over 23 months with mean 11 and `(a - 11)/T` where this
   build gives `(a - 11.5)/T` on a linear pattern. Under that convention
   an underwriting year written evenly through December would be counted
   as written at December's start.
2. **The input basis: by month directly, by a longer period without
   deconvolution** (proposed). `pattern_origin` names the accident
   periods the pattern was measured on, as in decision 18. An
   accident-month pattern (`Grain::Month`) is `p` itself, read at any age
   by decision 18 item 2's rule, under any writing and term. A pattern of
   accident periods of `s > 1` months is taken as decision 18's
   even-exposure average of `s` accident months, `P_s = (1/s) sum_q p(a -
   q)`, and converted as `P_UY(a) = sum_j v(j) P_s(a - j)`, where `v` is
   the weights that give `e` when each is spread evenly over `s` months.
   `v` exists in two cases, both exact: a policy term of `r` whole periods
   of `s` months (`v` the writing spread half and half over two months,
   then over `r` steps of `s` months, each `1/r`), or writing even within
   each period of `s` months from the start (`v` the periods' shares,
   convolved with a policy's earning). Otherwise the conversion is an
   error that asks for an accident-month pattern. Recovering monthly
   shares from an accident-year pattern (a deconvolution,
   `p(a) - p(a - 12) = 12 (P_Y(a) - P_Y(a - 1))`) is the alternative; it is
   ill-posed, needs the accident-year pattern at every month, and its
   negative weights can produce shares that are not increasing, so it is
   not built. The default (even writing, 12-month policies) and the common
   cases (seasonal writing of annual policies, even writing of six-month
   policies) are covered.
3. **An annual accident-year pattern is approximate in the first two
   years** (a finding, recorded with item 2; superseded by decision 25,
   which keeps this rule as `AnnualReading::LnLinear` and reads between
   annual ages by default with the Weibull through the averages, 0.9989,
   1.0004 and 0.9999 at 12, 24 and 36 months on the book below). As built
   here, `P_s` between its given ages was read by decision 18's rule:
   linear from 0 below the first age and `ln CDF` linear between ages. For an accident-year pattern given at
   every month (the chain ladder of a triangle with monthly development)
   the conversion is exact; given at 12, 24, ... months it is not, since
   an accident year at six months has much less than half of its twelve
   months' share. On a simulated book written evenly every month with
   annual policies, reported along `1 - 0.9^t`, the converted share over
   the book's own underwriting-year share is 1.3116 at 12 months, 0.9155 at
   24, 0.9866 at 36, 0.9967 at 48, 0.9991 at 60 and 0.9996 at 72, and
   exact from 84. The docs say so next to `ldf_underwriting` and advise an
   accident-month pattern, or one given at every month, where the first
   years matter. A better rule below the first age would change decision
   18's rule and is not proposed here; decision 25 (issue #646) studies
   one, and the user accepted it on 2026-10-27 as the default inside
   `to_underwriting_basis` only, so these misses now hold under
   `AnnualReading::LnLinear`.
4. **Ages from the underwriting period's start** (proposed). The age `a`
   is measured as `at_valuation` measures it: an underwriting year at the
   end of its twelfth month is at age 12, and an accident month at the end
   of its own month at age 1, so the accident month `t` months after the
   start is at age `a - t`. The result is a `PercentDeveloped` pattern at
   every month from the input's first age, as decision 18's average
   starts, so it goes into `Development::pattern` like any other and
   reaches both routes, Mack, both bootstraps and the one-year views.
5. **The longer pattern and its tail** (proposed, as decision 18 item 3).
   The result ends where every accident month of the period is past the
   input's end `E`: at `E + n + m - 1` for an accident-month pattern and
   `n` months of writing (`E + 23` under the default) and at `E + n + m -
   s` for a pattern of `s` months (`E + 12` for an accident-year pattern
   under the default). There it is the input's level exactly (1 for
   age-to-age factors), not a sum that rounds near it. Between `E` and the
   end, the accident months past `E` hold the level, so development a tail
   gives there is deferred to the result's end and the ultimate is
   slightly overstated, as in decision 18 item 3. A tail on the method
   applies after the result's end, decision 18 item 5 folds a result that
   runs past the triangle's oldest age with no new code, and the
   rescale reports `P(end) / T`. An estimated tail on the result is fitted
   on its monthly factors, so a constant tail taken from the input's own
   tail is the safer choice, and the docs say so.
6. **Weights and blends** (proposed, as decision 18 item 1). `All(w)`
   carries over to the result, so `ldf_weight` as a number blends the
   converted factors with the triangle's; a `ByAge` weight is an error
   with decision 19 item 11's text ("give one weight for every factor"),
   since the input's ages are not the result's. Kinds are converted as in
   decision 18: factors to ultimate and percents developed as given, an
   age-to-age pattern first to factors to ultimate, 1 at the end of its
   span; a single age-to-age factor has no span and is an error.
7. **The writing shares** (proposed). `UnderwritingBasis { writing,
   policy_months }` takes shares by month from the period's start, finite
   and not negative with a positive sum, and scales them to sum to 1;
   trailing zeros are dropped and leading zeros kept (a period that starts
   writing late). The period is the target's origin period, which
   `to_underwriting_basis` does not see, so the conversion itself takes
   any number of months, and `UnderwritingBasis::check_within(grain)`
   checks the writing, its trailing zeros dropped, against a period of the
   target's grain. The Python and R bindings know the target's grain at
   fit time and call it, so there writing longer than the origin period is
   an error (a period's policies are written within it). The default stays
   a fixed year of writing rather than writing even over the target's
   period, since the basis is built before the triangle is seen: on
   underwriting quarters the default is an error that names the fix
   (Python `UnderwritingBasis([1, 1, 1])`, R
   `underwriting_basis(c(1, 1, 1))`) instead of a wrong pattern, and the
   docs say that the default is a year's. A policy term of 0 and a result
   longer than 2,400 months are errors.
8. **API placement and names** (proposed). Rust:
   `SelectedPattern::to_underwriting_basis(pattern_origin, &UnderwritingBasis)`
   next to `averaged_over_origins`, and `UnderwritingBasis::{new, even,
   exposure}` with `Default` the parallelogram, re-exported from
   `lib.rs`. Python: a class `UnderwritingBasis(writing=None,
   policy_months=12)` with `writing`, `policy_months` and `exposure()`,
   and `ldf_underwriting=` beside `ldf_origin` on the eight methods that
   take `ldf` (`ChainLadder`, `Mack`, `ExpectedLoss`,
   `BornhuetterFerguson`, `Benktander`, `CapeCod`, `OdpBootstrap`,
   `MackBootstrap`), stored on the method with a getter and in its
   `repr`, and applied at fit time as `ldf_origin` is (decision 19 item 7).
   R: `underwriting_basis(writing = NULL, policy_months = 12)`, a list of
   class `underwriting_basis` with its `exposure`, and `ldf_underwriting =
   NULL` beside `ldf_origin` on every function that takes `ldf`. With
   `ldf_underwriting`, `ldf_origin` names the accident periods of `ldf`,
   and without it they are the target's origin grain, so an accident-year
   pattern on underwriting-year origins needs no `ldf_origin`; the
   conversion takes the place of `averaged_over_origins`, never both. The
   alternatives were three arguments (`ldf_basis`, `ldf_writing`,
   `ldf_policy_months`) on every method, or a module function that
   returns the converted pattern to pass as `ldf`. The pricing crate's
   `MonthlyPatterns::from_annual` and the aggregate crate's
   `timeline::Exposure::uniform` are other lanes' and are not reused: the
   convolution is a few lines, and the validation crate checks the two
   against each other.

**Checks** (`validation/tests/reserving_uy_pattern.rs`, the doctests, and
the Python and R tests `python/tests/test_reserving_uy_pattern.py` and
`R/prospicio/tests/test-reserving-uy-pattern.R`):

- the parallelogram by hand: the exposure `(2t + 1)/288` and `(47 -
  2t)/288` to 1e-15; a linear accident-month pattern `t/60` converted to
  `a(a + 1)(2a + 1)/(1728 * 60)` up to 12 months and `(a - 11.5)/60` from
  23 to 60, ending at 83 months at exactly 1;
- the accident-year route equal to the accident-month route to 1e-14
  when the accident-year pattern is the monthly one averaged at every
  month, under the default, seasonal writing of 24-month policies, even
  writing of 6-month policies and 24 months of even writing of 7-month
  policies, and an accident-quarter pattern under quarterly writing even
  within each quarter; the error for seasonal writing of 6-month policies
  on an accident-year pattern and for writing uneven within a quarter;
- kinds, the level kept, `All(w)` carried, the `ByAge`, single-factor,
  basis and 2,400-month errors, and shares scaled with trailing zeros
  dropped;
- `MonthlyPatterns::from_annual` (pricing crate, `cashflow.rs`): with the
  accident-month pattern `p(k) = paid_by((k - 1/2)/12)` and the written
  increments spread half and half over two months (item 1's writing
  through the month on the engine's grid of month starts), the converted
  share at age `a` is the engine's paid share at the end of month `a - 1`
  to 1e-13, under the default, seasonal writing of 12- and 7-month
  policies and three months of 1-month policies;
- a simulated book: three policies a month, written through each month
  from January 2000 to December 2019, each earning evenly over its term,
  losses at a rate of 1 per unit of exposure, each accident month
  reported along `1 - 0.9^t` to 60 months, summed to an accident-year and
  an underwriting-year triangle (origins 2010 to 2019, valued monthly to
  December 2019). With even writing and 12- and 6-month policies, the
  conversion of the accident-year triangle's chain ladder pattern is the
  underwriting-year triangle's chain ladder pattern to 1e-12 at every
  month, and projects each underwriting year to its ultimate to 1e-12;
  with seasonal writing and 12-, 6- and 18-month policies, the conversion
  of the accident-month pattern is the underwriting-year triangle's; and
  on the triangles valued annually the misses of item 3, pinned.

No reference implementation converts a reserving pattern from accident to
underwriting periods (chainladder-python 0.10.1 has no such argument, as
decision 18 item 10 found), so these checks are by hand, against the
pricing crate and on the simulated book.

### 22. An a-priori loss ratio per origin

Status: **Built** (2026-10-26), issue #634, branch `claude/apriori-by-origin`,
in Rust, Python and R. Items 1 to 3 restate the issue's proposal; items 4
to 8 are choices made in building it, proposed for the user's acceptance.
The Excel functions keep one number (`.into()` at their six call sites,
behaviour unchanged); a per-origin a priori on a sheet is #635 for the
Excel lane. The guides that still give the one-number workaround
(`apriori=1.0` with the prior in the exposure column) are #642 for the
Guides lane.

The expected loss, Bornhuetter-Ferguson and Benktander methods took one
`apriori: f64` for every origin, so a book whose plan loss ratio differs by
year (rate changes, a plan by underwriting year) needed one fit per origin
or a rescaled exposure. `apriori` is now an `Apriori`:

```rust
pub enum Apriori {
    All(f64),                       // one for every origin
    InOrder(Vec<f64>),              // one per origin, oldest first
    ByOrigin(Vec<(Period, f64)>),   // one per origin period
}
impl From<f64> for Apriori { ... }  // `apriori: 0.7.into()`
Apriori::by_origin(pairs)           // from (Period, f64)
Apriori::from_labels(pairs)?        // from ("2021", 0.7), ("2021Q3", 0.7), ...
```

Each origin's expected ultimate is its own `apriori[o] * exposure[o]`; the
methods' arithmetic is otherwise unchanged, so each origin of a fit with one
loss ratio per origin is that origin of the fit at its loss ratio for every
origin, to the bit.

1. **Keyed by origin, not by position.** A fit's origins are not the rows
   given: `fit_diagonal` takes a diagonal's rows in any order and fits its
   origins oldest first (`latest_diagonal.rs`; a diagonal that skips an
   origin is `Error::EmptyOrigin` in any case), and a triangle's segment is
   trimmed to the origins its column observes. A vector by position would
   not line up with the user's rows, so a key is an origin period and the
   values in order are read as one.
2. **`From<f64>`.** Existing calls write `apriori: 0.7.into()`. Struct
   literals that set `apriori` to a number no longer compile: a breaking
   change for Rust callers, marked so in the changelog. `Apriori::default()`
   is 1, as before.
3. **Mismatches are errors** (`Error::Apriori`): an origin fitted without a
   value, a key that names no origin, an origin given twice, an empty map,
   and a number of values in order other than the number of origins. A
   value that is not finite and positive is `Error::InvalidSetting`, as
   before.
4. **What "the origins" are** (proposed). Keys and values in order are
   matched once, at the entry point, against the triangle's origins
   (`Triangle::origins`, every origin of every segment), a latest
   diagonal's origins, or the origins of every segment of a
   `LatestDiagonals` together, sorted and without repeats. Each segment
   then takes the values of its own origins and ignores the others, so one
   map serves lines that start in different years, while a typo (an origin
   of no segment) is still an error. A key for an origin of the triangle
   that the loss column does not observe passes this check and is ignored.
   The alternative, an exact match per segment, would make a per-origin a
   priori unusable on a triangle with keys. On latest diagonals, values in
   order (`Apriori::InOrder`, a Python sequence, an unnamed R vector) are
   accepted only when the rows, segment after segment, are the origins
   oldest first, each once; any other order is `Error::Apriori`, telling
   the user to key by origin. A diagonal's exposure, latest values and ages
   are read in the order of its rows, so a column of a priori values passed
   next to them from a data frame would otherwise go to the wrong origins
   without an error whenever the rows are not oldest first. Reading values
   in order by row on diagonals was the alternative: it would give
   `InOrder` one meaning on a triangle and another on a diagonal, and over
   segments it would need one value per row, so one origin could take two
   values.
5. **Keys match by period, grain included** (proposed): `Period::year(2021)`
   names no quarterly origin, and the labels are the ones fits report
   (`"2021"`, `"2021H1"`, `"2021Q3"`, `"2021-07"`, exactly as `Period`
   displays them). Matching a month to the origin that contains it was the
   alternative; it would let `2021` silently name `2021Q1`.
6. **Through every path.** `fit` and `fit_segments`, `fit_diagonal`,
   `fit_diagonal_segments` and `fit_diagonal_segments_from`, and the
   one-year re-reserving method (`OneYearMethod`) of both bootstraps, which
   keys the a priori against the triangle's origins once before the
   simulations and re-reserves each simulated year with each origin's own
   value. `ExpectedLoss` and `BornhuetterFerguson` go through `Benktander`
   on every path, so the three share one implementation.
7. **Bindings** (proposed). Python's `apriori` takes a float, a sequence in
   origin order (a list, tuple or numpy array) or a dict keyed by origin
   label or integer year; dates as keys are not read, since the origin
   grain is not known when the method is built. The `apriori` property and
   the repr give it back as given (a float, a list, or a dict by label).
   R's `apriori` takes a number, an unnamed vector in origin order or a
   vector named by origin label; a vector with only some names is an error.
   The Python default is now `None`, read as 1, since a sequence cannot be
   a signature default.
   *Amended 2026-10-27* (issue #644, branch `claude/apriori-series`; the
   user accepted this item on 2026-10-27 with this follow-up): a pandas
   Series is read by its index, as a dict keyed by origin label or integer
   year (the same keys a dict takes), so `df.set_index("origin")["elr"]`
   lines up in any row order; it was read in order, so a Series of the
   right length in the wrong order put its values on the wrong origins
   without an error. A Series with a default `RangeIndex` (0 to n - 1;
   one over `range(1988, 1998)` is years) is an error that asks for origin
   keys or `.tolist()`, since its order cannot be checked;
   an origin twice in the index is an error, as in a dict (the pairs are
   read by `items()`, not `to_dict()`, which would keep the last). pandas
   is recognised by its type's module and name, not imported. Polars Series
   and numpy arrays have no index and stay sequences in origin order. R is
   unchanged: a named vector already keys by name.
8. **CapeCod is left out** (the issue's proposal): it estimates its loss
   ratio from the triangle, so a per-origin input does not apply to it.

**Checks.** Rust unit tests (`expected_loss.rs`, `latest_diagonal.rs`,
`one_year_bootstrap.rs`): each origin of a keyed fit equals that origin
fitted alone at its own loss ratio for `n_iters` 0, 1, 2 and 5 on RAA;
keyed, in order and by label in reverse order agree exactly; segments that
start in different years take their own origins; a diagonal given newest
first lines up by origin through `fit_diagonal`, `fit_diagonal_segments`
and `fit_diagonal_segments_from`, where values in order are an error, and
values in order on rows oldest first equal the keyed fit; the one-year view's claims development
result matches the closed form `(f - 1) (a[o] E / cdf - latest)` on an
exact triangle, and keyed and in-order views draw the same; every error and
label form. Python (`test_reserving.py`, `test_reserving_latest_diagonal.py`)
and R (`test-reserving.R`, `test-reserving-latest-diagonal.R`) repeat the
identities on `clrd_wkcomp` and the shuffled diagonal through each binding
(a dict or named vector on the shuffled rows, a sequence or unnamed vector
on the rows oldest first, and the error for a sequence on the shuffled
rows), with the one-year view and the errors.

### 23. A growth curve fitted to a benchmark pattern

Issue #637, branch `claude/benchmark-curve`. A benchmark pattern comes at a
few ages (12, 24, ..., 120 months, say); a pipeline needs its values at
other ages, a smooth pattern and a tail. Clark's methods (decision 5) fit a
growth curve to a triangle only. This adds the fit of the same curves to a
given pattern.

```rust
pub struct BenchmarkCurve { pub curve: GrowthCurve, pub max_age: Option<f64>, pub origin_width: Option<f64> }
impl BenchmarkCurve {
    pub fn fit(&self, kind: PatternKind, values: &[(f64, f64)], weights: Option<&[f64]>) -> Result<BenchmarkCurveFit>;
}
impl BenchmarkCurveFit {
    pub fn percent_developed(&self, age: f64) -> f64;
    pub fn cdf(&self, age: f64) -> f64;
    pub fn curve_age(&self, age: f64) -> f64;
    pub fn pattern(&self, ages: &[Lag]) -> Result<SelectedPattern>;
}
```

As built (`crates/prospicio-reserving/src/benchmark_curve.rs`, parity in
`validation/tests/reserving_benchmark_curve.rs`; the finding is
`knowledge/findings/benchmark-curve-fit.md`). Python `BenchmarkCurve(curve,
max_age, origin_width).fit(pattern, kind, weights)` returns a
`BenchmarkCurveFit` with `selected(ages)`, the `ldf`, `ldf_kind` and
`ldf_weight` arguments; R `benchmark_curve(pattern, kind, curve, weights,
max_age, origin_width)` returns a `benchmark_curve_fit`, with
`benchmark_percent_developed()`, `benchmark_cdf()` and
`benchmark_selected()`.

Each item is proposed, for the user's acceptance:

1. **Least squares on the percents developed** (as the issue asks), not on
   their logits, the factors to ultimate or the age-to-age factors: the
   objective is `sum w_i (p_i - p(a_i))^2`, minimized with `nelder_mead` on
   `(ln omega, ln theta)` to a tolerance of 1e-12. The percent scale weights
   the early ages, where most of the pattern moves, as much as the late
   ones; a weight by age shifts that. The alternatives were least squares on
   `logit p` (which the start uses, and which overweights the ages near 0%
   and 100%) or on the factors to ultimate (which overweights the early
   ages).
2. **The fitted percent developed is Clark's**: `p(a) = G(c(a)) /
   G(c(m))`, `m` the `max_age` at which development stops (`G(c(m)) = 1`
   without one) and 1 at `m` and after. So `latest / p(a)` is Clark's LDF
   ultimate, `latest * G(c(m)) / G(c(a))`, and a curve fitted to Clark's
   own percents gives his ultimates back in the chain ladder (checked to
   1e-7). A benchmark that reaches 100% at a finite age needs `max_age`,
   since an untruncated curve is below 1 at every finite age.
3. **The age convention**: benchmark ages are development ages in months
   from the start of the origin period, as a triangle's (any positive
   number, not only whole months). With `origin_width` set the curve is
   evaluated at Clark's ages from the average date of loss (decision 5:
   `a - w / 2` at or past the width `w`, `a / 2` before it), so a pattern
   read off a Clark fit is fitted on Clark's ages. The default is no shift,
   since a benchmark's ages are usually its own; the alternative was the
   shift by default, as Clark's methods have it.
4. **The tail is the pattern's level**, not a `Tail` of its own:
   `pattern(ages)` gives the fitted percents developed at the ages asked for
   as a fully selected `PatternKind::PercentDeveloped` pattern. Passed to the
   chain ladder or an expected-loss method with no tail and ending at the
   triangle's oldest age, its level there is the tail (decision 12),
   `cdf(oldest) = 1 / p(oldest)`, the curve's development past the oldest
   age, a constant with no parameter error, as Clark's own tail is part of
   his curve. Ages past the oldest fold into the same tail (decision 18).
   The pattern is fully selected; to blend it, set the weight on the
   `SelectedPattern` (or `ldf_weight` in the bindings) on ages that end at
   the oldest.
5. **Input checks**: percents developed or factors to ultimate
   (`PatternKind::AgeToAge` is an error: age-to-age factors give no level
   to fit), each finite and positive, ages finite, positive and each once,
   weights finite and not negative, one per value, at least two ages with
   positive weight, `max_age` at least the last age. Percents above 1 are
   accepted (the curve then fits them as best it can below 1), as a
   `SelectedPattern` accepts them. The fit is checked too: a fit that runs
   off to the edge of the parameters, omega below 1e-6 or above 1e6 or
   `|ln theta|` above 700 (below the overflow of `exp` at about 709.8), is
   an error. A flat or decreasing benchmark does this: the search drives
   omega to 0 or theta past `f64::MAX`, and the result is a flat line, not
   a growth curve. The check is on the fit, not on monotone input, so a
   noisy benchmark with a small dip still fits.
6. **The start** is the weighted straight line of `logit p` (log-logistic)
   or `ln(-ln(1 - p))` (Weibull) on the log curve age, over the percents in
   (0, 1); it is exact for an exact untruncated curve. Clark's (R's) start
   is the fallback when that line has fewer than two points or does not
   increase.
7. **No standard errors**: the fit reports the residuals and the weighted
   sum of squares only. A benchmark is not data with a sampling model, so a
   least-squares covariance would describe the curve's misfit, not the
   benchmark's uncertainty.
8. **Later work**: the exponential and inverse power decays on `f - 1`
   (`TailCurve`'s shapes) fitted to a benchmark's percents developed. They
   do not fall out of the growth-curve fit: their percents are infinite
   products of the age-to-age factors, and `TailCurve` already fits them to
   a pattern's own ages for a tail (decision 18). An R `nls` or scipy
   reference beyond the one scipy run is not planned.

### 24. A `ulr` column in the long tables of the methods with an exposure

Issue #636 asked for each origin's ultimate over its exposure in the
expected-loss fits' frames, so that a pipeline need not divide two columns
itself. Built on branch `claude/ulr-column`.

1. **One rule in the table, not a column per method.** `ReserveFit` has a
   default `exposure()` (`None`); a fit that has an exposure returns it,
   and `SegmentFits::to_long` and `totals` then end the method's own
   columns with `ulr`: the same row's `ultimate` over the exposure, and in
   the totals the total ultimate over the total exposure. So `ulr` always
   equals `ultimate / exposure` of its own row, and the Python `to_frame`,
   `totals_frame` and R `as.data.frame()`, `totals_frame()` carry it with
   no binding code.
2. **Which fits.** The expected-loss methods (`ExpectedLossFit`, and
   `CapeCodFit` delegating to it), Clark's Cape Cod form (the LDF form has
   no exposure and no column), `BayesBf` and `BayesBfNormalised` (their
   exposure) and Meyers' CRC, CSR and CAY (their premium), alone and over
   segments. Compartmental keeps its own `ultimate_loss_ratio` and is not
   changed.
3. **A loss ratio only on premium.** The column is named `ulr` as the issue
   asks, and the docs in Rust, Python and R say that it is a loss ratio
   only when the exposure is premium (with exposure in car years it is a
   pure premium). A NaN or zero exposure gives NaN or an infinite `ulr`,
   with no special case.
4. **The Bayesian tables' `ulr` is the chain ladder's** (accepted by the
   user on 2026-10-27). Their `ultimate` column is the plug-in chain
   ladder's, so their `ulr` is too; the posterior mean ultimate over the
   exposure is `(latest + mean) / exposure` (Meyers: `ultimate_mean /
   premium`). A posterior `ulr_mean` was left out to keep the PR small; it
   would be one line for Meyers, which already has `ultimate_mean`, and
   needs an ultimate column first for the two Bayesian BFs.

   *2026-10-27: accepted by the user with this follow-up (#645), built on
   branch `claude/ulr-mean` in Rust, Python and R.* `ulr` stays the row's
   `ultimate` over its exposure (item 1), so in these tables the chain
   ladder's. After the posterior moments the tables now carry `ulr_mean`,
   the posterior mean ultimate over the exposure, per origin and in total
   (the total `ultimate_mean` over the total exposure): for Meyers' CRC,
   CSR and CAY, alone and over segments, `ultimate_mean / premium`, after
   `ultimate_std_dev`; for `BayesBf` and `BayesBfNormalised` (fitted alone
   only; neither has a segment fit) a new `ultimate_mean`, `latest + mean`,
   then `ulr_mean = ultimate_mean / exposure`. Both are read off the
   table's own columns by one helper next to `push_moments`, so each
   equals the ratio of its own row exactly, as `ulr` does, with no special
   case for a NaN or zero exposure. The two Bayesian BFs get no
   `ultimate_std_dev`: the latest value is fixed, so it would repeat the
   reserve's `std_dev`. The new columns come last, so no existing column
   moves. The one-year views are unchanged: they keep `opening_ulr` (item
   5) and carry neither `ultimate_mean` nor `ulr_mean`. Unit tests in
   `bayes/bf.rs`, `bayes/bf_normalised.rs` and `bayes/meyers.rs` (alone and
   over segments) and the Python and R tests that list exact columns check
   the values.
5. **The one-year views name it `opening_ulr`** (accepted by the user on
   2026-10-27). A Bayesian one-year view's table has the opening ultimate,
   which `opening_names` renames from `ultimate`; the ratio follows it as
   `opening_ulr`, the opening ultimate over the exposure. The bootstrap
   one-year views of the expected-loss methods keep no exposure in their
   tables, so they have no column.
6. **Name** (accepted by the user on 2026-10-27): `ulr` as in the issue,
   next to the Clark Cape Cod total's fitted `elr` (the expected loss
   ratio of the prior, not of the ultimate) and Compartmental's
   `ultimate_loss_ratio`. The alternative is to call every one
   `ultimate_loss_ratio`.
7. **Checks.** Unit tests in `expected_loss.rs`, `clark.rs`, `bayes/bf.rs`
   (lifetime and one-year view) and `bayes/meyers.rs` assert the column
   equal to the ultimate over the exposure, per origin and in total; the
   Python and R tests that list exact column names take the new column and
   check its values.

### 25. Reading an annual accident-year pattern at monthly ages before the underwriting-year conversion

**Accepted** by the user on 2026-10-27: replace, as item 7 proposes, with
the Weibull the default (`wbl+exact`, item 6) and decision 18's rule kept
by name. Status: **Built** (2026-10-27), issue #646, branch
`claude/ay-monthly-reading`, in Rust, Python and R ("As built" below);
the design study is `validation/scripts/ay_monthly_reading.py` and the
finding `knowledge/findings/ay-pattern-monthly-reading.md`. It answers decision 21
item 3: an annual accident-year pattern given at 12, 24, ... months
converts to an underwriting basis with the converted share 1.3116 times
the book's own at 12 months and 0.9155 at 24, because decision 18 item 2
reads the pattern between its ages as if an accident year developed
`ln`-linearly through the year and linearly from 0 before its first age.

1. **The relationship any reading must satisfy.** An accident period of
   `s` months reports at age `a` the even-exposure average of its months,
   decision 18's `P_s(a) = (1/s) sum_{q<s} p(a - q)` with `p` the
   accident-month pattern and `p(t) = 0` at or below 0. Two consequences:
   (i) for `a < s` only `a` months have begun, so `P_s(a) <= (a/s) p(a)`,
   and reading `P_12(6)` as half of `P_12(12)` (decision 18's line from 0)
   takes an accident year at June to be half as developed as at December,
   when its six months are both fewer and younger: that is the 31%; (ii)
   `p(a) - p(a - s) = s (P_s(a) - P_s(a - 1))` pins `p` only when `P_s` is
   given at every month (decision 21's exact case), and at annual ages
   each year puts one constraint on twelve unknowns, so a reading is a
   choice. The least any reading must satisfy: `p` not decreasing where
   the given pattern is not, `0 <= p <= level`, the given values
   reproduced exactly as the averages of `p`, and the expanded `P_s` at
   every month between the given ages not decreasing. The conversion then
   takes the expanded `P_s` on decision 21's exact route, so every
   candidate is a drop-in: the only code that changes is the reading of
   `P_s` inside `to_underwriting_basis` for `s > 1` on ages coarser than
   a month. `averaged_over_origins`, the triangle and diagonal routes'
   `ln`-linear reading (decisions 12 and 16) and decision 18 item 2
   elsewhere are untouched, and a pattern given at every month reduces to
   the exact route under every candidate (to 3e-16).
2. **Candidates** (the script's names). `baseline`: decision 18 item 2 on
   `P_s`. `pchip`: a shape-preserving cubic through `(0, 0)` and the given
   points, on `P_s` itself. `linknot` and `lnknot`: the issue's option
   (b), `p` linear or `ln`-linear between the given ages with its knots
   solved so that the averages are the given values (a forward recursion,
   `P(12k) = (6.5 p_k + 5.5 p_{k-1}) / 12` for the linear one). `monoqp`:
   the exact monotone `p` on monthly ages with the least sum of squared
   second differences (a quadratic programme). `llog`, `wbl`: Clark's
   loglogistic and Weibull, `p(t) = L G(t) / G(E)`, with `(omega, theta)`
   fitted so that the twelve-month averages of `G` match the given values
   in least squares on the percents (decision 23's scale). `llog+exact`,
   `wbl+exact`: the issue's option (a), the curve's monthly increments
   moved as little as possible (least squares) to a `p` whose averages are
   the given values exactly, not decreasing except within a span whose
   given value is below the one before it (downward development, as an
   incurred pattern with factors below 1 has it), capped at the level.
   `wbl+scale`: a closed form, the Weibull's increments rescaled span by
   span so that each span's average is the given value. `llog-mid`:
   decision 23's `origin_width` shift, the curve fitted to the given
   values at the mid-year age `a - 6`.
3. **On synthetic books with a known accident-month pattern** (the ratio
   of the converted underwriting-year share to the book's own, default
   parallelogram, the given pattern at 12 to 120 months):

   | truth | candidate | 12 | 24 | 36 | 48 | 60 |
   |---|---|---|---|---|---|---|
   | book `1 - 0.9^t` (P(12) = 0.46) | baseline | 1.3116 | 0.9155 | 0.9866 | 0.9967 | 0.9991 |
   | | pchip | 1.3487 | 0.9732 | 0.9989 | 0.9997 | 1.0000 |
   | | monoqp | 1.0816 | 0.9731 | 1.0066 | 0.9975 | 1.0008 |
   | | llog+exact | 0.9308 | 1.0230 | 0.9946 | 1.0014 | 0.9997 |
   | | wbl+exact | 0.9989 | 1.0004 | 0.9999 | 1.0000 | 1.0000 |
   | | wbl+scale | 0.9986 | 1.0006 | 0.9997 | 1.0002 | 0.9999 |
   | slow loglogistic 1.5, 36 (P(12) = 0.09) | baseline | 1.6192 | 0.8928 | 0.9705 | 0.9881 | 0.9942 |
   | | monoqp | 1.0051 | 0.9975 | 1.0012 | 0.9997 | 1.0000 |
   | | llog+exact | 1.0011 | 0.9995 | 1.0002 | 1.0000 | 1.0000 |
   | | wbl+exact | 1.0153 | 0.9974 | 1.0002 | 1.0000 | 1.0000 |
   | | wbl+scale | 1.0636 | 0.9804 | 1.0091 | 0.9952 | 1.0031 |
   | medium Weibull 1.5, 18 (P(12) = 0.20) | baseline | 1.5928 | 0.8868 | 0.9691 | 0.9901 | 0.9972 |
   | | monoqp | 1.0048 | 0.9971 | 1.0017 | 0.9994 | 1.0003 |
   | | llog+exact | 0.9561 | 1.0130 | 0.9955 | 1.0009 | 0.9998 |
   | | wbl+exact | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
   | fast Weibull 1.2, 6 (P(12) = 0.60) | baseline | 1.3055 | 0.9075 | 0.9939 | 0.9997 | 1.0000 |
   | | monoqp | 1.1001 | 0.9693 | 1.0033 | 0.9997 | 1.0000 |
   | | llog+exact | 0.9303 | 1.0213 | 0.9977 | 1.0002 | 1.0000 |
   | | wbl+exact | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
   | very fast exponential 2 (P(12) = 0.87) | baseline | 1.1083 | 0.9551 | 0.9999 | 1.0000 | 1.0000 |
   | | monoqp | 1.0480 | 0.9807 | 1.0000 | 1.0000 | 1.0000 |
   | | llog+exact | 0.9791 | 1.0084 | 1.0000 | 1.0000 | 1.0000 |
   | | wbl+exact | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

   `linknot` and `lnknot` reproduce the given values but are not monotone
   on any truth (`lnknot` 1.18, 0.92, 1.06, 0.95, 1.04 on the book:
   exact at the given ages and oscillating between them), so option (b)
   is not viable without a monotonicity constraint, at which point it is
   the quadratic programme. `pchip` is worse than the baseline at 12
   months, since no interpolant of `P_s` alone knows that `P_s(a)` for
   `a < 12` averages a partial year. `llog-mid` is 0.59 to 0.88 at 12
   months: decision 23's mid-year shift is not a substitute for the
   averaging relation. The baseline's error depends on the basis too:
   1.40 at 12 months under seasonal writing (three times the rate in the
   last quarter) and 1.17 under six-month policies, where `wbl+exact` is
   0.9986 and 0.9994.
4. **On industry patterns** there is no truth; the band between the exact
   monotone candidates is what the given values cannot pin down. The
   converted underwriting-year share at 12 months (`monoqp`, `llog+exact`,
   `wbl+exact`, then the baseline): CLRD private passenger auto paid
   0.164, 0.146, 0.153 against 0.195; workers compensation paid 0.094,
   0.087, 0.091 against 0.122; medical malpractice paid 0.0111, 0.0105,
   0.0102 against 0.0207; other liability paid 0.0300, 0.0305, 0.0301
   against 0.0459; Friedland's XYZ paid selection 0.0157, 0.0160, 0.0155
   against 0.0230. The exact candidates agree within 2% to 12% of each
   other at 12 months and within 1% from 36; the baseline is 19% to 86%
   above the band's top at 12 months, 5% to 13% below its bottom at 24,
   1% to 4% below at 36, 0.3% to 2% below at 48 and about 1% below at
   60. Every
   exact candidate keeps the expanded pattern monotone and within `[0,
   level]` on every monotone input. On Friedland's XYZ reported selection
   (factors of 0.993 and 0.999 at 108 and 120 months) `wbl+exact` and
   `llog+exact` reproduce the given values exactly with the development
   downward in those spans, as the pattern says; the uncorrected curves
   smooth the dip away (`wbl` misses the given values by 0.03).
5. **Robustness.** A pattern given at every month comes back unchanged
   under every candidate (ratios 1 to 3e-16), so decision 21's exact
   cases are kept to the bit. An accident-year pattern at quarterly ages
   and an accident-quarter pattern (`s = 3`) at quarterly ages, where the
   baseline is already within 1.3%, are exact under `wbl+exact` and
   within 0.1% under `monoqp` and `llog+exact` (SLSQP stops short on the
   accident-year pattern at quarterly ages, so those figures are not an
   exact optimum; see "As built"). With three given ages
   only (12, 24, 36) `wbl+exact` was 0.978, 1.008 and 0.999 against the
   baseline's 1.312, 0.916 and 0.982 with the cap at the level, and is
   0.976, 1.010 and 0.994 as built without it. A level below 1 (every share 0.95
   of the book's) changes nothing in the ratios. `wbl+scale`, the closed
   form, is exact where the cap does not bind and 6% high at 12 months on
   the slow loglogistic. Cost in the prototype (scipy, SLSQP): 0.1 to 0.3
   s for the curve fit and the projection together on 120 monthly ages, 5
   s with 40 given ages; the baseline is under a millisecond. A least
   roughness on the log increments (zero on the book's geometric
   increments) was tried and left out: SLSQP on 120 log increments did
   not converge in 15 of 17 runs, so it gave no evidence.
6. **Recommendation: `wbl+exact`** (proposed). The Weibull fitted through
   the averaging relation, corrected to the nearest monotone
   accident-month pattern whose yearly averages are the given values.
   It is exact at the given ages by construction, monotone, within 0.2%
   of the truth on the book at every age, exact on the Weibull family, at
   worst 1.5% off at 12 months on the slow loglogistic, and it keeps
   downward development where a pattern has it. `llog+exact` is the
   better prior only on the slow loglogistic (0.1%) and is 7% low at 12
   months on the book and the fast Weibull: a loglogistic's heavy tail
   fits a pattern that reaches 100% at a finite age badly, and the
   correction cannot fix the first year's shape. `monoqp` needs no curve
   but is 5% to 11% high at 12 months on fast patterns, since least
   roughness flattens the first year. A curve as the prior is the issue's
   option (a); the Weibull is the default and the loglogistic a choice.
7. **Replace or option** (the question for the user). The proposal is to
   **replace** decision 18 item 2 as the reading inside
   `to_underwriting_basis` for a pattern of `s > 1` months given at ages
   coarser than a month, with the new reading the default and decision
   18's rule kept by name for bit-for-bit reproduction, as a setting on
   `UnderwritingBasis`: `UnderwritingBasis::with_reading(AnnualReading)`
   with `AnnualReading::Curve(GrowthCurve)` (the default,
   `GrowthCurve::Weibull`) and `AnnualReading::LnLinear` (decision 18 item
   2, today's result); Python `UnderwritingBasis(..., reading="weibull" |
   "loglogistic" | "ln-linear")`, R `underwriting_basis(..., reading =
   "weibull")`. The reason for a new default: the current one is 31% off
   in the most common use, an annual industry benchmark on an
   underwriting-year book, and nobody would pick it knowingly; a pattern
   given at every month, an accident-month pattern and the exact cases
   are the same under both. The alternative, the new reading as an opt-in
   with decision 18's rule the default, keeps every pinned number and
   leaves the 31% in place for anyone who does not read the docs.
   `UnderwritingBasis` has two `pub` fields, so a third would break
   `UnderwritingBasis { writing, policy_months }` literals; the reading
   goes on a private field behind `with_reading` and a getter, and
   `new`, `even` and `Default` keep their signatures (the type is a day
   old, and Python and R go through `new`). A new default changes the
   pinned misses in `an_annual_accident_year_pattern_converts_approximately`
   (they move under `AnnualReading::LnLinear` and get new values under the
   default), the Python and R underwriting-year tests, the
   `ldf_underwriting` docs in both languages, and decision 21 item 3's
   text.
8. **Implementation sketch** for the follow-up PR (Rust, Python and R in
   one PR, as decision 21 was built). A module
   `crates/prospicio-reserving/src/monthly_reading.rs`: (a) the curve fit
   through the averages, `nelder_mead` on `(ln omega, ln theta)` with
   decision 23's start and checks, the objective the sum of squared
   differences between the given values and the `s`-month averages of `L
   G(t) / G(E)` at the given ages (about 80 lines, sharing
   `benchmark_curve.rs`'s curve and start); (b) the projection, `min
   ||d - d_curve||^2` over monthly increments `d` on `1..E` subject to `A
   d = b` (one row per given age, the window's cumulative weights over
   `s`), `d >= 0` except within a span whose given value is below the one
   before it, and `sum d <= level`: an active-set loop on the bounds
   around the equality-constrained least-squares solve by
   `prospicio_math::linalg::solve` on the KKT system (about 150 to 250
   lines; the prototype's SLSQP is not needed); (c) `AnnualReading` and
   `with_reading` on `UnderwritingBasis`, the expansion called from
   `to_underwriting_basis` before `weights_over` when `s > 1` and the
   pattern's ages are not every month, the errors (a fit that runs to the
   edge of the parameters, decision 23 item 5; a curve on a pattern with
   one given age); (d) unit tests: the given values reproduced to 1e-12,
   monotone, the monthly identity, downward development kept, the book's
   ratios; (e) the validation test's pinned misses under both readings
   and the seasonal and six-month cases; (f) Python `reading=` with its
   getter and `repr`, R `reading =`, their tests and help. About 700 to
   900 lines in all, no new dependency.

**As built** (2026-10-27, `crates/prospicio-reserving/src/monthly_reading.rs`,
checked in its unit tests and `validation/tests/reserving_uy_pattern.rs`;
Python `UnderwritingBasis(..., reading=)` and R `underwriting_basis(...,
reading = )` with their tests in `python/tests/test_reserving_uy_pattern.py`
and `R/prospicio/tests/test-reserving-uy-pattern.R`):

- **API.** `AnnualReading { Curve(GrowthCurve), LnLinear }`, `Default`
  `Curve(GrowthCurve::Weibull)`, re-exported from `lib.rs`;
  `UnderwritingBasis::with_reading` and `reading()` on a private field, so
  `UnderwritingBasis { writing, policy_months }` literals no longer
  compile (breaking; `new`, `even` and `Default` keep their signatures).
  Python takes `reading="weibull" | "loglogistic" | "ln-linear"`, with a
  getter and the reading in the `repr`; R `reading = c("weibull",
  "loglogistic", "ln-linear")`, kept in the list (a list without it reads
  as the Weibull).
- **The gate.** The reading applies when `pattern_origin` is longer than a
  month and the given ages are not every month from 1 to the end (a
  pattern given at every month from a later age is read too, since below
  its first age decision 18's line from 0 is the same misreading). The
  result then starts at 1 month, not at the input's first age; the
  2,400-month check counts from there and runs before the reading.
  Accident-month patterns and patterns at every month from 1 convert to
  the bit under every reading (unit test).
- **The curve fit.** `nelder_mead` (tolerance 1e-12) on `(ln omega, ln
  theta)` from decision 23's linearized start, each given value read at the
  middle of its window (`a - (s - 1) / 2`), and the prototype's grid of
  nine starts (omega 0.8, 1.5, 3 by theta 6, 18, 36 months); the best
  value is kept even when a search stops on its iteration limit, since the
  curve is only the projection's prior (two given ages can ask for a curve
  that steepens toward a step). Decision 23's edge check is kept as an
  error. A flat pattern does not reach the edge: the curve reports
  everything in the first month and the projection reproduces it.
- **No cap at the level** (a departure from items 1, 2 and 8, made after
  review on 2026-10-27, confirmed by the user on 2026-10-27). The study capped the
  increments at the level, `sum d <= L`. The given value at the last age
  `E` is the average of `p` over the last `s` months, so with `p` not
  decreasing the cap forces `p` to equal the level over the whole last
  window. At annual ages that flattens the last year of any pattern still
  developing at `E`; at ages finer than `s` (quarterly ages of accident
  years) the overlapping windows then fix each quarter's block from the
  next one back, and a pattern still developing at its last age, even by
  3e-6 (`1 - 0.9^t` read quarterly to 120 months), has no solution at
  all, so the default refused common inputs with an error that blamed
  decreasing percents. Without the cap `p(E)` may exceed the level (the
  development past `E` that a tail carries), and the expanded `P_s`, a
  moving average of a monotone `p` whose last window averages to the
  level, still stays within `[0, level]` on a monotone input, so item 1's
  conditions on the expanded pattern hold. Where the given values already
  hold `p` flat over the last window (two equal values in a row, as the
  book's from 72 months), the cap was redundant and nothing moves: the
  book's pinned misses are unchanged. CLRD private passenger auto paid
  moves by at most 1.3e-6 at 12 to 72 months (the prototype run again
  without the cap, below), and the three-ages case of item 5 from 0.978,
  1.008, 0.999 to 0.976, 1.010, 0.994. `validation/scripts/ay_monthly_reading.py`
  drops the cap for `llog+exact` and `wbl+exact` (`monoqp` keeps it, as
  studied); every other table of items 3 and 4 is unchanged to its
  printed digits.
- **The projection.** Goldfarb and Idnani's dual active-set method with the
  identity as its Hessian: the equality rows (one per given age,
  `min(s, a - u + 1) / s` on the increment of month `u`) first, then the
  most violated bound added one at a time, with bounds dropped when a
  multiplier would turn negative. Held months are taken out of the solve,
  so each step solves the Gram system of the equality rows on the free
  months by a Cholesky factorization that skips a row whose pivot is at
  most 1e-10 of its diagonal (a row dependent, to rounding, on the
  others), and after each bound is added the point is solved afresh with
  one pass of refinement. One tolerance, `1e-12` of the largest given
  value (or of 1), governs rounding: a given value below the one before it
  by no more than it is not downward development; two given values in a
  row equal within it hold the increments of months `a - s + 2` to `a'` at
  0 from the start (each is not negative and they sum to `s (P_s(a') -
  P_s(a))`), so held spans leave no dependent rows; a bound violated by no
  more than it whose normal depends on the active set is met as it stands.
  The result is checked on exit (averages and bounds to the same
  tolerance), and a failure is an error, never a near solution. The
  expanded pattern is the given value itself at each given age.
- **Errors** (`Error::Pattern`): a single given value ("gives no shape to
  read between ages", naming the ln-linear reading, under which it still
  converts), the curve's edge, no monotone accident-month pattern with the
  given averages, and a projection that does not converge. Each one that
  points to the ln-linear reading names it as `AnnualReading::LnLinear`
  and as `reading="ln-linear"` for Python and R, since the bindings pass
  the message through.
- **Numbers.** On the book of decision 21's checks, valued annually, the
  default's misses on the underwriting year's factors to ultimate are
  -0.00107, 0.00037, -0.00010, 0.00002, 0.00002 and -0.00002 at 12 to 72
  months (ratios 0.9989, 1.0004, 0.9999, 1.0000, 1.0000, the study's) and
  exact from 84; decision 18's are pinned unchanged under `LnLinear`.
  Seasonal writing and six-month policies give 0.9986 and 0.9994 at 12
  months (decision 18's 1.40 and 1.17). The converted shares at 12 to 72
  months agree with the scipy prototype's `wbl+exact` and `llog+exact`
  (without the cap) within 1.1e-9 on the book and CLRD private passenger
  auto paid (both searches stop near their tolerance, at slightly
  different points). SLSQP still stops short on the accident-year pattern
  at quarterly ages (an iteration limit or "incompatible constraints"), so
  the prototype's four-digit figures there are not an exact optimum; the
  Rust projection solves those cases (unit tests below). Factors to
  ultimate and age-to-age factors convert as their percents to 1e-10
  under the curve (the inputs' last bits move the search) and to 1e-14
  under `LnLinear`.
- **Unit tests** (twelve, about 0.3 s together in an optimized debug
  build): besides the ones above, accident years at quarterly and
  semiannual ages, flat over their last months but not over the whole
  last year of ages, and still developing at the last age (`1 - 0.9^t`
  and `1 - 0.97^t` to 120 months, to 240 months held from 120); a flat
  tail lowered by one ulp, read as monotone; and a seeded sweep of 60
  monotone accident-month patterns built from sparse increments, read at
  ages spaced below their accident period (`s = 12` at 3 and 6 months,
  `s = 6` at 3), each under both curves. The sweep at 3,000 patterns
  (6,000 readings, about 70 s in release) found no failure; before these
  fixes about 1,100 of 2,940 such readings failed (review of PR #650).

### 26. Ultimates and loss ratios per simulation for towers, deal books and cash flows

Issue #602 asked for a documented bridge from a reserving fit's draws to
the vectors per simulation that `Tower::apply_annual`, `DealRun` and
`DealBook` and `CashflowTerms::run` take, with the guarantee that index
`i` is the same simulation on every side. Built on branch
`claude/loss-ratio-draws`, in Rust, Python and R.

1. **One trait in this crate, not copies on each fit.**
   `prospicio_reserving::UltimateDraws` (`draws.rs`) asks a fit for two
   things, its joint `reserves` and each component's latest value in the
   same order, and provides the rest: `ultimates()` (the reserves with
   the latest values added, the same dimensions, components and rows, so
   `join`, `select`, `aggregate` and `reorder_groups` work on it),
   `ultimate_draws(origin)`, `loss_ratio_draws(origin, premium)`,
   `total_ultimate_draws()` and `total_loss_ratio_draws(premium)`, the
   totals summed index by index. A helper on `PredictiveDistribution`
   was the other choice the issue named; it would have to be told the
   latest values from outside, and `prospicio-prob` is the Probability
   lane's crate, so the trait sits with the fits that know them.
2. **Which fits.** Every fit with reserves per simulation:
   `OdpBootstrapFit`, `OdpBootstrapFits`, `MackBootstrapFit`,
   `MackBootstrapFits`, `BayesOdpFit`, `BayesOdpFits`, `BayesBfFit`,
   `BayesBfNormalisedFit` (so `LayerSharePrior::fit` too), `MeyersFit`,
   `MeyersFits`, `CompartmentalFit` and `CompartmentalFits`. The one-year
   views (claims development results) are not ultimates and do not
   implement it; their opening `fit` does.
3. **The basis.** The ultimate is the latest value of the fitted column
   plus the reserve draw: paid for a paid triangle, incurred for an
   incurred one. Meyers' models simulate the value at the oldest age
   itself and keep it as `ultimates`; the trait returns those draws (its
   `ultimates` is overridden), which the reserves plus the latest values
   match to rounding (1e-12 relative in the test). Compartmental models
   reserve on the paid basis (the paid at the square's last age less the
   latest paid), so their ultimate is the latest paid plus the reserve,
   equal to the last future paid cell of each origin in the same
   simulation; there is no tail past the square.
4. **Naming an origin.** By its component key in `reserves`, as
   `PredictiveDistribution::component_index` reads it: `[origin]` for one
   segment, `[key values..., origin]` over segments, a period by its label
   (`"2021"`, `2021`, `"2021Q3"`). A fit over segments needs the segment
   in the key (an origin alone is an error, not the sum over segments);
   the sum over segments is `ultimates()` with `aggregate` or `select`.
   Python takes a tuple or a single label, R a list or a vector; `None`
   or `NULL` gives the total.
5. **The premium is an argument** (accepted by the user on 2026-10-28). `loss_ratio_draws` takes the
   origin's premium as the issue states, finite and positive, even for
   Meyers' and the compartmental fits, which carry a premium of their own
   (`premium`), and the Bayesian BFs, which carry an exposure. The
   proposal keeps one signature for every fit and leaves the program's
   subject premium (which need not be the triangle's premium column) to
   the caller; a later convenience could default to the fit's own premium
   where it has one.
6. **The order guarantee, and where it is stated.** Row `i` of a
   `PredictiveDistribution` is one simulation for every component, a
   simulated one draws simulation `i` from `StreamRng::new(seed, i)`, so
   the same inputs and seed give the same rows whatever the threads, and
   NUTS fits with the same seed repeat their draws on one machine. The
   operations that keep rows whole keep the pairing; `reorder_groups`
   keeps each group's rows whole but pairs row `i` of one group with
   another simulation of another (`reorder_groups_indexed` says which).
   The trait's module docs state it, and doc comments only (a cross-lane
   edit, no code) state it on `PredictiveDistribution`
   (`prospicio-prob`), `Tower::apply_annual` (`prospicio-aggregate`),
   `DealBook` and `CashflowTerms::run` (`prospicio-pricing`).
7. **Bindings.** Python: an `ultimates` property on every fit above
   (Meyers' already had it, with the same draws), `ultimate_draws(origin
   = None)` and `loss_ratio_draws(origin, premium)`. R:
   `ultimate_draws(fit, origin = NULL)`, `loss_ratio_draws(fit, origin =
   NULL, premium)` and `ultimate_distribution(fit)`, each calling the
   Rust trait through the fit's pointer.
8. **Checks.** Unit tests in `draws.rs` (ODP and Mack bootstraps on RAA
   and on RAA twice under a `lob` key): each origin's ultimate is its
   latest plus its reserve in the same simulation, found by key; labels
   and integers name the same origin; the totals and loss ratios by
   index; the same seed gives the same draws and another seed different
   ones; `join` keeps row `i` of each part, and after `reorder_groups_indexed`
   every column of a model's row is the latest plus the reserve of the
   one source simulation the index names. The mean of each origin's
   ultimates is the chain ladder's within four standard errors of the
   mean; the total is the bootstrap's own expected total (a run of ten
   times the simulations on another seed) to that error, but not the
   chain ladder's: on RAA the ODP bootstrap's mean total ultimate is
   about 0.77% (1,650, 3.2% of the reserve) above the chain ladder's,
   a bias that adds up over the origins (measured by hand; the test
   bounds the gap below 2%). The issue's `expected_ultimate()`
   exists only on `BayesBfNormalisedFit`, where it is the posterior mean
   of the prior's `U_w` without the process error and the cells already
   paid, so it is not the target; the Bayesian checks compare with the
   long tables instead. `validation/tests/reserving_loss_ratio_draws.rs`
   puts the draws through towers: a stop loss on `total_ultimate_draws`
   by `apply_annual` cedes, simulation by simulation, what it cedes on
   `ultimates()` by `apply_aggregate` and on `reserves` with the
   attachment lowered by the latest values; the same for a contract on
   one origin's loss ratio and for each line of a fit over segments.
   `validation/tests/reserving_bayes_loss_ratio_draws.rs` (slow-bayes)
   checks the layout, the long tables' means (Meyers' `ultimate_mean` and
   `ulr_mean`) and the same-seed draws for `BayesOdp` (alone and over
   segments), both Bayesian BFs, Meyers' CRC (alone and over segments)
   and compartmental model 1 (alone and over segments, its ultimate the
   last future paid cell; over segments the second segment is the first
   with every value doubled, so a swap of the segments' latest values
   shows). Python and R tests repeat the bootstrap checks
   and the tower.
9. **Later.** The how-to in `docs/guides/` that runs the chain end to end
   (fit, loss ratios, tower, deal book, cash flows) belongs to the Guides
   lane and is ready to write.

### 27. Mack's tests of the chain ladder's assumptions

Status: **Built** (2026-10-28), issue #551, branch
`claude/mack-assumption-tests`, in Rust, Python and R; item 6 was
accepted by the user on 2026-10-28 as built. Mack (1994, "Measuring the
variability of chain ladder reserve estimates", CAS Forum Spring 1994)
gives two cheap closed-form tests of the chain ladder's assumptions that
reviewers ask for beside a Mack exhibit: a test for calendar-year effects
(appendix H) and one for correlation between adjacent development factors
(appendix G). R ChainLadder 0.2.21 has them as `cyEffTest` and `dfCorTest`.

1. **What is built.** `prospicio_reserving::assumptions`:
   `calendar_year_effect(&Triangle, column, ci) -> CalendarYearTest` and
   `development_correlation(&Triangle, column, ci) ->
   DevelopmentCorrelationTest` on every observed link ratio of a
   single-segment triangle, and `MackFit::calendar_year_effect(ci)` and
   `MackFit::development_correlation(ci)` on the link ratios the fit
   averaged (item 6). `CalendarYearTest` holds each diagonal's `S_j`,
   `L_j`, `Z_j`, `n`, `m`, `E[Z_j]` and `Var[Z_j]` (R's `test_table`), the
   totals, the interval and `passes()`; `DevelopmentCorrelationTest` each
   pair's `T_k`, origins in common and weight, `T`, `Var[T]`, the interval
   and `passes()`. `passes()` is true when the statistic lies within the
   interval, ends included (R prints the opposite, "Calendar Year Effect:
   FALSE"). Python: `prospicio.reserving.calendar_year_effect(tri,
   column, ci=0.95)`, `development_correlation(tri, column, ci=0.5)`, the
   same two methods on `MackFit`, and `CalendarYearTest` and
   `DevelopmentCorrelationTest` with `diagonals`, `pairs` and `to_frame()`.
   R: `calendar_year_effect(x, column = NULL, ci = 0.95)` and
   `development_correlation(x, column = NULL, ci = 0.5)` on a `triangle`
   or a `mack_fit`, returning `calendar_year_test` and
   `development_correlation_test` with `print` and `summary` methods
   shaped as R ChainLadder's (`Table`, `Totals`, `Range`; `Results`,
   `Range`, and the pairs as `Table`).
2. **Mack's example is RAA, not GenIns.** The issue says Mack prints both
   tests on the Taylor-Ashe triangle; his 1994 paper's example (chapter 6)
   is the RAA triangle. His printed values are reproduced on RAA in the
   unit tests: `T_k = 4/21, -9/28, 3/7, -1/5, 2/5, -1/2, 1` with weights 7
   to 1, `T = .070`, `Var(T) = 1/28` (p. 161); the S/L table of p. 167,
   `Z = 14`, `E(Z) = 12.875`, `Var(Z) = 3.9785`. His intervals use 2
   standard deviations for 95% and .67 for 50%; prospicio, like R, uses
   `qnorm((1 + ci) / 2)`, and his `(8.886, 16.864)` comes back with
   `ci = 2 Phi(2) - 1`.
3. **Default levels: 95% and 50%.** The issue's sketch had
   `CalendarTestOptions { ci: 0.5 }`; Mack (p. 166) and R's `cyEffTest`
   test calendar-year effects at 95%, and only the correlation test at
   50% (Mack: the test is approximate and meant to catch correlation in a
   substantial part of the triangle). The defaults are Mack's and R's,
   exported as `CALENDAR_YEAR_CI` and `DEVELOPMENT_CORRELATION_CI`; the
   level is a plain `f64` argument, strictly between 0 and 1.
4. **Calendar diagonals by valuation month.** At each age the link ratios
   are split at their median (R's `median`, the mean of the two middle
   values for an even count); one equal to the median is in neither set.
   A link ratio belongs to the diagonal of its later cell, keyed by that
   cell's valuation month, which is R's anti-diagonal on a square triangle
   and stays defined on a non-square one or with quarterly ages. Every
   diagonal with at least two link ratios is reported, which on a square
   triangle is R's rows `j = 2, ..., I - 1`; a diagonal with `n <= 1` has
   `Z_j = E[Z_j] = Var[Z_j] = 0` (Mack leaves it out; it changes no
   total). `choose(n - 1, m) / 2^n` is a product up to `n = 1000` and log
   gamma beyond.
5. **Correlation pairs over the origins they share.** Each pair of
   adjacent link columns is ranked within the origins observed at both,
   which is Mack's "leave out the last element and rank again" and R's
   `use = "pairwise.complete.obs"`; Spearman's coefficient is Pearson's on
   average ranks, which equals Mack's (G4) without ties and R's with them.
   The weight is `n - 1` and `Var(T) = 1 / sum(weights)`: on a full
   triangle exactly Mack's (G5) and (G6), and still 1 over the variance of
   each `T_k` when exclusions or a non-square shape change the counts (R
   hard-codes the weights `(I - 3):1` by position). A pair with fewer than
   two origins is left out; one whose ranks are all tied in a column has
   `T_k = NaN` and weight 0; no usable pair is an error.
6. **Exclusions** (accepted by the user on 2026-10-28 as built). The issue asks for the
   `MackFit` methods to use "the same column and exclusions as the fit".
   Built so: the fit stores the link ratios `Development::kept_mask`
   keeps, and its two methods test those alone, so they need no triangle
   argument and cannot be run against another one; the free functions on
   a `Triangle` test every observed link ratio (R and chainladder-python
   parity). Consequences, documented in the bindings: `n_periods` keeps
   only the diagonals after its cutoff (on RAA with `n_periods = 4`, the
   diagonals of 1987 to 1990), so the test asks whether the window the
   factors use is free of calendar effects; `drop_high` and `drop_low`
   remove, at each age, exactly the extreme factors the calendar test
   counts, which weakens it; a selected pattern (`ldf`) changes no link
   ratio and so not the tests. The alternative is to test every observed
   link ratio always, the fit's methods then being the free functions on
   its triangle. Recommendation: as built, since the tests are meant to
   justify the factors the fit averaged, and the free functions give the
   whole-triangle view in one call.
7. **One segment.** Both tests need a single-segment triangle or fit (the
   bindings name `segment(...)`), as `claims_development_result()` does. A
   link ratio with a zero earlier value is not observed, as in
   `Triangle::link_ratios`.
8. **chainladder-python as a second target.** chainladder-python 0.10.1's
   `ValuationCorrelation(total=True)` gives the same `z`, `z_expectation`
   and `z_variance`, its `total=False` the same per-diagonal `z`, `E[Z_j]`
   and `Var[Z_j]` (with one more, always-zero, row for the first
   diagonal), and `DevelopmentCorrelation` the same `T_k`, weights, `T`,
   variance and interval on RAA, GenIns and ABC. Its `total=True` range and
   `z_critical` are not targets: the range is centred on the sum of the
   variances instead of `E[Z]`, and `z_critical` compares that sum with
   it, so it is false on every triangle (also on ABC, where R's test
   rejects). Its `total=False` test is Mack's 1997 per-diagonal binomial
   test, not built here. See
   `knowledge/references/chainladder-python-correlation.md` and
   `chainladder-python-correlation-traps.md`.
9. **Checks.** `validation/tests/reserving_assumptions.rs`: every row of
   `reserving_assumptions_r.csv` (R ChainLadder 0.2.21's `cyEffTest` at
   0.95 and 0.9 and `dfCorTest` at 0.5 and 0.75, with its `T_k`, on RAA,
   GenIns and ABC; `validation/scripts/reserving_assumptions_r.R`) and of
   `reserving_assumptions_python.csv` (chainladder-python's statistics of
   item 8; `reserving_assumptions_python.py`) to 1e-12; the Mack fit's
   tests equal the free functions without exclusions; and a 15-origin
   synthetic triangle passes until two calendar years' payments are
   raised by 50%, after which it fails at 95% with no small factor on the
   shocked diagonals. The same shock on RAA and GenIns still passes: on a
   10 x 10 triangle the test has little power. On the real triangles R
   and prospicio agree that RAA passes both tests, GenIns passes the
   calendar test and fails the correlation test at 50% (`T = -0.164`
   against `+-0.127`), and ABC fails both (`Z = 8` outside
   `(12.22, 20.38)`, `T = 0.544`). Unit tests reproduce Mack's printed
   values (item 2), his `n = 5` moments (`E = 50/32`, `Var = 95/256`), the
   exclusions of item 6 and the errors.

### 28. A development pattern per simulation, for the cash-flow engine

Status: **Built** (2026-10-28), issue #600 (its Reserving part), branch
`claude/reserving-pattern-draws`, in Rust, Python and R; items 2, 4, 5 and
6 were proposed and built as recommended, and the user accepted them as
built on 2026-10-28. The cash-flow engine takes a
paid pattern per simulation (`PatternDraws`, `time-value.md` decision 33);
this decision gives those patterns from a reserving fit, row `i` the same
simulation as row `i` of the fit's `reserves` and of every vector
decision 26 gives.

1. **What a simulation's pattern is.** The cumulative share of the
   ultimate that an origin with nothing yet observed would have reached at
   each development age, under the development the simulation drew: the
   bootstraps' pseudo factors (selected, blended and with the tail drawn
   as the reserves use them), `BayesOdp`'s column effects `b_d`,
   `BayesBf`'s `beta_d` and `BayesBfNormalised`'s shares `p_d` of the
   posterior draw the simulation took. It is the parameter uncertainty in
   the pattern, the same in every origin of the segment, and not an
   origin's simulated cells divided by its ultimate: those carry process
   error that differs by origin, and an old origin has only the end of its
   curve left. A deal on a new year pays by the first.
2. **The tail** (accepted 2026-10-28). A fit with a tail reaches 1 one development
   period after the oldest age (12 months on an annual triangle), the age
   at which the one-year view with a bootstrap tail (decision 11) closes
   an origin; the share at the oldest age is `1 / tail`. The Bayesian
   models have no tail, so their pattern reaches 1 at the oldest age.
3. **Alignment.** The pattern is computed inside the same simulation as
   the reserves, from what that simulation already drew, so no random
   number is drawn for it and the reserves are bit for bit what they were
   (a test pins them). Over segments, each segment's patterns move with its
   reserves under a rank correlation: the reorder is
   `reorder_groups_indexed` (equal to `reorder_groups` on the reserves),
   and its index carries the patterns.
4. **Monthly from inception** (accepted 2026-10-28). `DevelopmentPatterns::monthly`
   gives, per simulation, `months` values for the engine's `paid`: the
   share at the end of month `t + 1`. On the origin's own basis
   (`MonthlyBasis::Origin`, the default), a pattern of accident periods
   longer than a month is read at every month as decision 25 reads it
   before converting (the Weibull curve by default, `ln`-linear on
   request); on an underwriting basis (`MonthlyBasis::Underwriting`), it
   is converted as `to_underwriting_basis` converts it (decision 21), at
   every month from 1 under either reading (`to_underwriting_basis`
   itself starts at the pattern's first age under the `ln`-linear
   reading, and the months before it would pay nothing). The reading costs
   about 9 ms a simulation with the Weibull (30 microseconds `ln`-linear),
   so 10,000 simulations take about a minute and a half on one thread. The
   terms' `written` and `earned` must describe the same basis; the engine
   cannot check it.
5. **A falling pattern** (accepted 2026-10-28). A pseudo factor below 1 makes a
   simulation's share fall at that age, which the engine refuses.
   `Monotone::Refuse` (the default) refuses it, naming the number of such
   simulations and the first; `Monotone::RunningMax` holds each share at
   the largest before it, capped at 1, so the money is paid when first
   reached and none is returned. Mack's bootstrap draws its factors from
   normal distributions, and on RAA about 0.15% of its simulations draw a
   negative first factor, so the share at the first age is negative: the
   default refuses that too, and `RunningMax` gives such a share the next
   positive share, which that simulation then pays from its first age
   (its factors mean nothing there). On RAA a share falls in 54%
   of the ODP bootstrap's simulations and 22% of Mack's
   (`knowledge/findings/pattern-draws-falling.md`). The default leaves the
   choice to the caller rather than altering a fit's draws without a
   word.
6. **The horizon** (accepted 2026-10-28). The patterns end at exactly 1, so the
   terms' `paid` must end at 1 for the engine to take them; `months`
   shorter than the age at which the converted pattern reaches 1 is an
   error, not a truncation, and longer is padded with 1.
7. **Which fits.** `OdpBootstrapFit(s)`, `MackBootstrapFit(s)`,
   `BayesOdpFit(s)`, `BayesBfFit` and `BayesBfNormalisedFit` (so
   `LayerSharePrior::fit` too), each with a `patterns` field. Meyers'
   models simulate only the value at the oldest age, log-normal cell by
   cell, and Compartmental's pattern is the ODE's for the drawn rates;
   both are later items. No fit knows whether its column is paid: a
   pattern from an incurred triangle is an incurred pattern, and the
   engine would read it as paid. **Meyers' models are built** (2026-10-28,
   #678, `MeyersFit` and `MeyersFits`, proposed and built as recommended,
   awaiting the user's acceptance): `reserving-bayes.md` decision 27 has the
   pattern (the ratio of the expected cumulative values to the one at the
   oldest age, per posterior draw) and its alternatives. `Compartmental`
   is still a later item (#679).
8. **Bindings.** Python: `fit.patterns` (`DevelopmentPatterns`) with
   `n_sims`, `segments`, `ages(segment=None)`, `mean(segment=None)`,
   `pattern_draws(segment=None)` (a list per simulation) and
   `monthly(months, segment=None, basis=None, monotone="refuse")`, whose
   result goes to `CashflowTerms.run(..., paid_patterns=)`; `basis` is
   `"origin"`, `"origin_ln_linear"` or an `UnderwritingBasis`. A fit of
   one segment, including one picked from several, needs no label. R:
   `pattern_draws(fit, segment = NULL)` and `monthly_pattern_draws(fit,
   months, segment = NULL, basis = NULL, monotone = "refuse")`, a matrix
   with one row per simulation for `cashflow_run(..., paid_patterns = )`.
