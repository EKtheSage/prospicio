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
`crates/act-r/src/reserving.rs`, `R/actuarialrs/R/reserving.R`) and the
generated stub, `NAMESPACE` and `man/`. Whichever merges second merges
`main` and regenerates the generated files; it does not merge them by hand.

## References

| Method | Primary reference | Secondary |
|---|---|---|
| Expected loss, BF, Benktander, Cape Cod | chainladder-python 0.10.1 (`ExpectedLoss`, `BornhuetterFerguson`, `Benktander`, `CapeCod`) | hand-computed cases |
| Tails | chainladder-python (`TailConstant`, `TailCurve`, `TailBondy`); R ChainLadder 0.2.21 `MackChainLadder(tail = TRUE / number)` for Mack's tail sigma and standard error | — |
| Clark | R ChainLadder `ClarkLDF`, `ClarkCapeCod` | chainladder-python `ClarkLDF` |
| One-year view | R ChainLadder `CDR(MackChainLadder(...))`, `dev = "all"` for the full run-off | Merz and Wüthrich (2008), published example |

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
cumulated like its losses). Origins are matched by position in the
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
* Benktander iterates as written above rather than chainladder-python's
  closed form `sum(p^k, k < n) latest + p^n U0` (equal up to rounding) and
  stops once an ultimate no longer changes, so a huge `n_iters` is cheap.
* Long tables add `exposure` and `apriori` per origin (Cape Cod also
  `trended_apriori`) and the total `exposure` per segment.
* Python: `ExpectedLoss`, `BornhuetterFerguson`, `Benktander` and `CapeCod`
  take the settings above plus `average`, `sigma_interpolation` and `tail`
  for the chain ladder, and `fit(triangle, column, exposure)` fits every
  segment, returning `ExpectedLossFit` or `CapeCodFit`. R bindings follow
  in a later PR.

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

Bindings: Python `ChainLadder(tail=...)` and `Mack(tail=...)` accept a float
or a `TailConstant`, `TailCurve`, `TailBondy` or `TailLogLinear`; R accepts a
number or the matching constructor.

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
```

Merz and Wüthrich's formulas assume volume-weighted factors and no tail, so
any other `alpha` or a tail other than 1 is an error. R reports Mack's
ultimate standard error beside the full run-off; the parity test checks
both.

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
