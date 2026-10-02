# Design note: distribution representations

Status: **Decided; partly implemented** (parametric, sampled, `Severity`, `Grid`, `Counting`) · Depends on: nothing · Next: `Dist` enum with the second severity family

## Goal

One trait family for every univariate loss distribution, with three concrete
representations (parametric, discretized, sampled), each exact under
different operations. An operation that is not exact for a representation
is not offered on it; the caller must convert explicitly, and every
conversion reports the error it introduces.

## What exists

- `act_prob::Distribution`: `mean`, `variance`, `std_dev`, `cdf`,
  `quantile -> Result<f64>`, and `sample(&mut StreamRng, n)` defaulting to
  inverse transform.
- `act_prob::Lognormal`, parameterized as SciPy / R (`meanlog`, `sdlog`),
  plus `from_mean_cv`. Parity with SciPy is checked by
  `validation/tests/distributions.rs` against
  `validation/reference/distributions_scipy.csv`.
- `act_prob::Sampled` and the `Empirical` trait (`draws`, `sorted`,
  `mean_of`, `var`, `tvar`), and `act_prob::risk::{var_sorted, tvar_sorted}`,
  the shared risk measures every domain calls.
- `act_prob::Severity` (`lev`, `stop_loss`, `layer`), implemented for
  `Lognormal`. Parity: `validation/reference/severity_mpmath.csv`, from
  30-digit integration of the survival function
  (`validation/scripts/mpmath_severity.py`), at limits out to the
  `1 - 1e-9` quantile.

- `act_prob::Grid`, the discretized representation, with
  `Grid::local_moment` (needs `Severity`), `Grid::rounding` and
  `Grid::lower`, each returning a `DiscretizationReport`. A `Grid` is a
  `Distribution` and a `Severity` (exact LEV, stop-loss and layers on the
  grid). Parity: `validation/reference/grid_mpmath.csv`, masses from the
  textbook definitions at 30 digits (`validation/scripts/mpmath_grid.py`).

- `act_prob::Counting` (`pmf`, `cdf`, `mean`, `variance`, `panjer_ab`, `pgf`,
  `quantile`, `sample`), with `Poisson` and `NegativeBinomial` (Klugman's
  `r`, `beta`; SciPy `nbinom(n=r, p=1/(1+beta))`). Parity: claim-count rows
  in `validation/reference/distributions_scipy.csv`.

### Decisions for `Counting`

- Probabilities come from log-gamma closed forms, not by running the
  `(a, b, 0)` recursion, so they carry no accumulated rounding; a test
  checks they satisfy the recursion `panjer_ab` reports.
- `quantile(1)` is `u64::MAX`; sampling is inverse transform from 0, so its
  cost grows with the mean (fine for annual claim counts).

### Decisions for `Grid`

- **Probabilities always sum to 1.** Discretization lumps the source's mass
  above the last point onto that point, and the report records it as
  `tail_mass = S((n - 1)h)`, with `source_mean`, `grid_mean` and
  `mean_error()`. Choosing `n` so `tail_mass` is negligible is the caller's
  job; aggregation will check it.
- **Local moment matching keeps the limited mean:** the grid's mean is
  exactly `LEV((n - 1)h)`, so `mean_error() = -stop_loss((n - 1)h)`.
- **No upper-bound method.** Lumping the tail onto the last point moves mass
  down, which breaks the upper bound; it returns if a use needs it, with
  the tail handled separately.
- Grids start at 0 (open question 2).

### Decisions for `Severity`

- `stop_loss` is computed directly, not as `mean() - lev(d)`: in the tail
  that difference cancels (Lognormal(0, 1) at d = 1135 keeps ~6 digits
  against ~14 for the direct form). `layer(l, a)` defaults to
  `stop_loss(a) - stop_loss(a + l)` for the same reason.
- `lev(d)` for `d <= 0` is `d`, and `lev(inf)` is the mean: severities are
  non-negative.

### Decisions for `Sampled`

- **The draws are the distribution.** Every `Distribution` method describes
  the empirical distribution: `variance` divides by `n`, `quantile` inverts
  the empirical distribution function (R `type = 1`, NumPy
  `inverted_cdf`), and `sample` resamples with replacement. This keeps
  `quantile` consistent with the trait's "smallest `x` with
  `cdf(x) >= p`" contract. Interpolated quantiles, if a parity target
  needs them, are a separate function, not a different `quantile`.
- **TVaR** is `(1 / (1 - p)) * integral from p to 1 of VaR(u) du`, with the
  atom at the VaR split fractionally. It is coherent and continuous in
  `p`. When `p * n` is a whole number it is the mean of the largest
  `n * (1 - p)` draws (the draw at the VaR gets no weight).
- **Simulation order is kept.** `draws()` returns draws in simulation order
  so marginals from one `PredictiveDistribution` can be paired row by row;
  a sorted copy is stored alongside for quantiles (2 × 8 bytes per draw).
- **No weights** (open question 3): a `WeightedSampled` joins when
  importance sampling does (v1.x).

## Proposed trait layout

```text
Distribution                 every representation: moments, cdf, quantile, sample
├── Severity                 parametric + discretized: lev(limit), layer(limit, attach),
│                            stop_loss(d), excess_mean(d)
├── Discretize               parametric: to_grid(step, truncation, method) -> (Grid, DiscretizationReport)
└── Empirical                sampled: draws(), weights(), tvar/var on draws
```

Capabilities are separate traits so that "not exact here" is a compile-time
absence, not a runtime error. Example: `Sampled` does not implement
`Severity::lev` exactly, so it does not implement `Severity` at all; a caller
who wants an approximate LEV from draws calls `Empirical::mean_of(|x| x.min(l))`
and owns that choice.

## The three representations

| | Parametric | Discretized (`Grid`) | Sampled |
|---|---|---|---|
| Storage | parameters | step `h`, probabilities `p[0..n]` on `0, h, 2h, …`, tail mass beyond `n·h` | draws `x[0..n]`, optional weights |
| Exact | cdf, quantile, moments, LEV, limit/excess, scaling | convolution, layer, stop-loss, quantile on the grid | any statistic of the draws (as an estimate) |
| Converts to | Grid (`Discretize`), Sampled (`sample`) | Sampled | Grid (histogram) |
| Metadata | – | `step`, `truncation`, `method`, `tail_mass`, `mean_error` | `seed`, stream ids, `n` |

Discretization methods: rounding (mass dispersal), mean-preserving (local
moment matching), lower and upper bounds. The `DiscretizationReport` carries
the truncated tail mass and the mean difference against the source, so
aggregate results can state their discretization error (a v0.3 plan
requirement).

## Static vs dynamic dispatch

Recommendation: concrete structs plus a **closed enum** at the boundary.

```rust
pub enum Dist {
    Lognormal(Lognormal),
    Pareto(Pareto),
    // ... every native parametric family
    Grid(Grid),
    Sampled(Sampled),
    Custom(CustomDist), // Python/R callback: single-threaded, flagged slow
}
```

- Hot loops (FFT, Monte Carlo, towers) are generic over `D: Distribution`
  and monomorphize: no virtual calls per draw.
- Bindings, serialization and model outputs use `Dist`, so objects cross
  the Python/R boundary, pickle and serialize without trait objects.
- `Custom` is the single "slow path" door the plan describes; code that
  sees it runs single-threaded and records that in diagnostics.

## Sampling

Inverse transform by default: draws are a pure, monotone function of the
uniforms, so common random numbers and replaying a single simulation both
work. Families with a slow quantile may override `sample` with a faster
method, but must keep output a pure function of `(seed, stream)` and
document the change in the RNG stability log (see `rng.md`).

## Parameterization

Each family's native parameters follow SciPy and R (actuar) naming, with
the mapping stated in the doc comment. Actuarial conveniences
(`from_mean_cv`, `from_mean_sd`) are constructors, never alternate storage.

## Validation

Every family ships with SciPy (and, where it differs, actuar) parity rows
in `validation/reference/`, generated by a checked-in script, with
tolerances recorded per row.

## Open questions

All decided on 2026-10-01, following the recommendations:

1. ~~Enum vs trait objects~~: concrete structs plus a closed `Dist` enum at
   the boundary (bindings, serialization, model outputs); hot loops stay
   generic over `D: Distribution`. The enum is introduced with the second
   family.
2. ~~Grid origin~~: severity grids start at 0 for v0.3. Signed grids (net
   cash flow, P&L) wait until capital needs them.
3. ~~Weights on `Sampled`~~: decided, see "Decisions for `Sampled`".
4. ~~Frequency distributions~~: a separate `Counting` trait over integer
   support, exposing the (a, b, 0) / (a, b, 1) form Panjer needs.
