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

impl ClaimsDevelopmentResult {
    pub fn run_off_standard_error(&self) -> Vec<f64>;   // sqrt of summed yearly MSEPs
    pub fn total_run_off_standard_error(&self) -> f64;
}
```

Merz and Wüthrich's formulas assume volume-weighted factors and no tail, so
any other `alpha` or a tail other than 1 is an error (R only warns for
`alpha != 1`). They also assume a full trapezoid, the latest values on one
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
