# Design note: an economic scenario generator and multi-year DFA

Status: **Accepted** (2026-11-27; issue #983, in the 0.4.0 milestone at the
user's request of 2026-11-27). The user decided open decisions 1 to 10 on
2026-11-27 as recommended (6: the AR(2) cycle; 10: option (b) for the
curve fits and Hull-White's `θ`, the rest as inputs first). PR 1 (the
crate `prospicio-esg`) is built (branch
`claude/project-thread-n5mttf-esg-crate`), and PR 2 (inflation, branch
`claude/project-thread-n5mttf-esg-inflation`), and PR 3 (the projection,
branch `claude/project-thread-n5mttf-esg-dfa`); PRs 4 to 6 are to come. Lane: Aggregate (a new
crate, decision 1; the lane table's Owns column changes in a small PR of
its own) · Depends on: `aggregate.md` (`EventSet`, `Tower::apply_term`,
`EventSet::correlated_years`, `ruin`), `reserving-inflation.md` (Reserving
lane, read only: `IndexPaths`, `Indexation::reinflate_draws`),
`reserving-ifrs17.md` (`DiscountCurve`), `risk.md` (`PredictiveDistribution`,
capital allocation), `rng.md` (`StreamRng`)

## Goal

The library answers one-year questions well: reserve risk (the one-year
views), premium risk (aggregates and towers), capital and its allocation,
eventual ruin (Pollaczeck-Khinchine, Wiener-Hopf) and finite-horizon ruin
of the classical surplus process (`ruin::finite_horizon`, #562). Dynamic
financial analysis (DFA) and an ORSA project the balance sheet over
several years under correlated economic scenarios: what the surplus looks
like in five years, and how likely it is to fall below the required
capital on the way. Three pieces are missing:

1. **An economic scenario generator (ESG)**: simulated paths of the short
   rate and the yield curve, of economic inflation, and of claims
   (superimposed) inflation, correlated, seeded and reproducible.
2. **A multi-year projection**: premiums under an underwriting cycle,
   losses from the existing simulations scaled by the inflation paths,
   the reserve run-off, investment income on the assets from the rate
   paths, expenses, tax, and dividends or capital injections by a rule,
   rolled forward to the surplus at the end of each year of each path.
3. **Finite-time ruin and capital breach** on those paths: the chance that
   the surplus falls below zero, or below a required capital, within `T`
   years, with the expected shortfall at the first breach.

## What exists

* `StreamRng` (`prospicio-core`, `rng.md`): seeded streams, one per
  simulation, so a parallel split keeps the seeded numbers.
* `prospicio_aggregate::term` (`Tower::apply_term`,
  `EventSet::term_years`, `correlated_years`, `calibrate_year_correlation`):
  a tower over consecutive years of one simulated path, with years whose
  totals are rank-correlated. This is the losses' side of a multi-year
  path.
* `prospicio_aggregate::ruin`: `pollaczeck_khinchine`, Wiener-Hopf under
  renewal counts, `finite_horizon` (discrete periods, a count and a
  severity on a grid) and `finite_horizon_poisson`. These are analytic, for
  a surplus with a constant premium and no economics; they are the parity
  targets of the simulated ruin (validation 5).
* `prospicio_reserving::Indexation::reinflate_draws` with `IndexPaths`
  (`reserving-inflation.md`, PR 8): future cells re-inflated along one
  index path per simulation. Its module says "the paths are an input: this
  module does not generate them". The ESG generates them.
* `prospicio_reserving::discount::DiscountCurve`: spot rates by maturity,
  annually compounded. An ESG curve at a future date can be handed to it.
* `prospicio_prob`: copulas (Gaussian, t, Archimedean, R-vines),
  `PredictiveDistribution` (simulation-major draws with components) and
  `capital` (distortion measures and allocation).
* No interest-rate model, yield-curve fit, inflation model or balance-sheet
  type exists anywhere in the workspace.

## The method

### 1. The ESG

All models run on a time step `Δ` (monthly by default, decision 4) over a
horizon of `T` years, one `StreamRng` stream per simulation, with
correlated standard normal shocks `ε_t = L z_t` (`L` the Cholesky factor
of the shocks' correlation matrix).

**Short-rate models** (Brigo and Mercurio, chapter 3):

* **Vasicek**: `dr = a (b - r) dt + σ dW`. Exact transition: `r_{t+Δ}` is
  normal with mean `b + (r_t - b) e^{-aΔ}` and variance
  `σ² (1 - e^{-2aΔ}) / (2a)`. Zero-coupon prices in closed form,
  `P(t, T) = A(t, T) e^{-B(t, T) r_t}`, so each path carries its whole
  curve at every step.
* **CIR**: `dr = a (b - r) dt + σ √r dW`. Exact transition by the
  noncentral chi-square (degrees `4ab/σ²`), so no discretisation bias and
  no negative rates; the Feller condition `2ab ≥ σ²` is reported, not
  required. Closed-form bond prices as for Vasicek.
* **Hull-White** (one factor): `dr = (θ(t) - a r) dt + σ dW`, with `θ(t)`
  fitted so that the model reproduces today's curve exactly. The initial
  curve is a `DiscountCurve` or a Nelson-Siegel-Svensson fit.

**The curve shape**: **Nelson-Siegel-Svensson**,
`y(m) = β0 + β1 f1(m/τ1) + β2 f2(m/τ1) + β3 f2(m/τ2)` with
`f1(x) = (1 - e^{-x})/x` and `f2(x) = f1(x) - e^{-x}`, fitted by least
squares to observed yields (with `τ1`, `τ2` on a grid, then the betas
linear, the usual robust fit), and used to build an initial curve for
Hull-White or a `DiscountCurve`. A dynamic Nelson-Siegel (the betas as
AR(1) factors, Diebold and Li) is decision 3.

**Inflation**: economic inflation as an Ornstein-Uhlenbeck process on the
continuously compounded rate, `di = κ (μ - i) dt + σ_i dW_i`, correlated
with the short rate through `L`; the price index is
`I_t = I_0 exp(∫ i)`. Claims inflation is economic inflation plus a
superimposed component, an AR(1) on the annual excess with its own mean,
persistence and shock (decision 5). Each simulation's index path is an
`IndexPaths` row, so reserve cells re-inflate through
`reinflate_draws` without new code there.

**Output**: `Scenarios`, simulation-major like `PredictiveDistribution`:
for each simulation and each step the short rate, the curve at a set of
maturities (from the closed forms, or the model's own), the inflation
rates and the indices. A curve at any step converts to a `DiscountCurve`,
and an index series to an `IndexPaths`.

### 2. The multi-year projection

`Projection` rolls one company (or one segment) forward a year at a time
along each scenario. The inputs are the opening balance sheet (assets by
a duration bucket, reserves, surplus), and per year:

* **Premium**: exposure times a rate level. The underwriting cycle is an
  AR(1) on the log rate adequacy (the ratio of the charged rate to the
  technical rate), `x_{t+1} = μ + φ (x_t - μ) + σ ε`, so the loss ratio
  falls in a hard market and rises in a soft one (decision 6).
* **Losses of the new year**: the existing annual simulations (an
  `EventSet`'s years, a tower's net losses through `apply_term`, or any
  `PredictiveDistribution` of annual losses), in the money of the
  projection's base year and scaled by each path's claims index. The
  rank dependence between the losses and the economy (a high-inflation
  year with large losses) is a copula on the ranks (decision 7).
* **The reserve run-off**: the opening reserves' payments by year (a
  payment pattern, or a reserving fit's future cells re-inflated along the
  path), and the one-year change in the reserve (the CDR) when the user
  brings it.
* **Investment income**: the assets earn the path's curve; a bond bucket of
  duration `D` is revalued by its rate change (`-D Δy` to first order,
  decision 8), cash earns the short rate.
* **Expenses**, **tax** (a flat rate on the year's profit, losses carried
  forward or not) and **capital management**: a dividend of the surplus
  above a target ratio, a capital injection up to a floor, or none.

Each step yields, by simulation: the year's premium, losses, expenses,
investment return, tax, dividends and injections, and the closing
assets, reserves and surplus. The result is a `ProjectionPaths`: a
`PredictiveDistribution` per line with one component per year, so the
existing quantiles, TVaR and capital allocation apply to any line of any
year, and the paths keep their joint ranks.

### 3. Finite-time ruin and capital breach

On the surplus paths `U_{s,t}`:

```
ruin(T)          = #{s : min_{t ≤ T} U_{s,t} < 0} / n
breach(T, K)     = #{s : min_{t ≤ T} (U_{s,t} - K_{s,t}) < 0} / n
shortfall(T, K)  = mean of -(U - K) at the first breach, over the paths that breach
```

with `K` a required capital per year (a fixed amount, a ratio of premium
or reserves, or the one-year VaR of the next year's result, decision 9),
and each estimate's binomial standard error. Ruin can be checked at year
ends only (the projection's grid) or at each ESG step; the year-end check
understates ruin within a year, and the note says so.

## API (sketch, for the decisions)

Rust (`crates/prospicio-esg/`, a new crate, decision 1):

```rust
pub enum ShortRate {
    Vasicek { a: f64, b: f64, sigma: f64, r0: f64 },
    Cir { a: f64, b: f64, sigma: f64, r0: f64 },
    HullWhite { a: f64, sigma: f64, initial: DiscountCurve },
}
pub struct Svensson { pub beta: [f64; 4], pub tau: [f64; 2] }
impl Svensson {
    pub fn fit(maturities: &[f64], yields: &[f64]) -> Result<Self>;
    pub fn spot(&self, m: f64) -> f64;
}
pub struct Inflation { pub kappa: f64, pub mean: f64, pub sigma: f64, pub i0: f64 }
pub struct Superimposed { pub mean: f64, pub persistence: f64, pub sigma: f64 }

pub struct Esg {
    pub rate: ShortRate,
    pub inflation: Option<Inflation>,
    pub superimposed: Option<Superimposed>,
    pub correlation: Vec<Vec<f64>>,   // between the shocks, in that order
    pub step: Step,                   // Monthly, Quarterly, Annual
    pub maturities: Vec<f64>,         // the curve reported at each step
}
impl Esg {
    pub fn simulate(&self, years: usize, sims: usize, seed: u64) -> Result<Scenarios>;
}
impl Scenarios {
    pub fn short_rate(&self) -> PredictiveDistribution;        // components = steps
    pub fn curve(&self, sim: usize, step: usize) -> Result<DiscountCurve>;
    pub fn index_paths(&self, which: Index) -> Result<IndexPaths>;
    pub fn deflators(&self) -> PredictiveDistribution;         // exp(-∫ r)
}
```

and in `prospicio-aggregate` (`dfa.rs`, decision 2):

```rust
pub struct Projection { /* opening balance sheet, cycle, losses, run-off, rules */ }
impl Projection {
    pub fn run(&self, scenarios: &Scenarios) -> Result<ProjectionPaths>;
}
impl ProjectionPaths {
    pub fn line(&self, line: Line) -> &PredictiveDistribution;   // components = years
    pub fn ruin(&self, horizon: usize) -> Estimate;
    pub fn breach(&self, horizon: usize, capital: &RequiredCapital) -> Breach;
}
```

Python and R bind both, with scenarios and paths returned as arrays (a
matrix of simulations by steps) and the projection's inputs as columns.

## Open decisions

1. **The crate.** (a) A new crate `prospicio-esg` (depends on
   `prospicio-core`, `-math`, `-prob`; in the WASM scope, no threads on its
   default path), owned by the Aggregate lane. (b) A module of
   `prospicio-prob`. (c) A module of `prospicio-aggregate`. Recommended:
   (a): rate and inflation models are a domain of their own, the reserving
   and pricing crates can use them without depending on the aggregate
   crate, and the umbrella crate re-exports it.
2. **Where the projection lives.** (a) `prospicio-aggregate::dfa`, beside
   `term` and `ruin`, depending on `prospicio-esg`; the reserve run-off and
   the CDR come in as `PredictiveDistribution`s that the user takes from a
   reserving fit (the bindings chain the crates). (b) A crate
   `prospicio-dfa` that depends on the aggregate, reserving and ESG crates
   and reads fits directly. Recommended: (a), as `time-value.md` decision
   37(a) kept reserving and pricing apart; (b) if the projection grows into
   an ORSA model.
3. **Which curve dynamics.** (a) Vasicek, CIR and Hull-White one factor,
   with Svensson for today's curve. (b) Also a two-factor model (G2++) or
   a dynamic Nelson-Siegel. (c) Also the Smith-Wilson extrapolation of
   EIOPA's risk-free curve. Recommended: (a) first; (b) and (c) each an
   issue if a user asks. Real-world (P) dynamics by default, with the
   market price of risk a parameter; risk-neutral (Q) valuation is market
   consistent valuation, out of scope for DFA.
4. **The time step.** (a) Monthly by default, quarterly and annual as
   settings, with exact transitions so a coarse step is unbiased. (b)
   Annual only. Recommended: (a): exact transitions make the step a
   reporting choice, and monthly steps catch ruin within a year.
5. **Inflation.** (a) Ornstein-Uhlenbeck economic inflation correlated
   with the short rate, and an AR(1) superimposed component added to it
   for claims. (b) A Wilkie-style cascade (inflation drives rates and
   returns). Recommended: (a), the simplest that gives correlated paths
   and closed-form moments to validate against; (b) later.
6. **The underwriting cycle.** (a) AR(1) on the log rate adequacy, with
   the loss ratio following. (b) AR(2), which gives the cycle's period
   (the literature's six to ten years for US P&C). Recommended: (b) as the
   model, with (a) its special case `φ2 = 0`: the AR(2) is no harder and
   the cycle's period is what a user calibrates.
7. **Dependence between the losses and the economy.** (a) A copula on the
   ranks of the year's losses and the year's claims inflation (any
   `prospicio-prob` copula). (b) Independence, the losses scaled by the
   index only. Recommended: (a), default independence beyond the scaling.
8. **Assets.** (a) Cash and bond buckets by duration, revalued by the
   curve; no equities. (b) Also an equity index with a regime-switching
   lognormal return. Recommended: (a) first; (b) as a follow-up.
9. **The required capital in the breach test.** (a) A fixed amount or a
   ratio of premium or reserves per year, given by the user. (b) Also the
   one-year VaR of the next year's result, simulated inside each path
   (nested simulation). Recommended: (a); (b) is expensive and a separate
   issue.
10. **Calibration.** (a) Parameters are inputs; the guide shows typical
    values. (b) Maximum likelihood on a historical series from the exact
    transition densities (Vasicek and CIR have them), and Hull-White's
    `θ(t)` to the curve. Recommended: (b) for the curve fits and the
    Hull-White `θ`, which are mechanical; (a) for the rest in the first
    build, (b) of Vasicek and CIR as a later PR.

## Validation targets

1. **Vasicek and CIR**: the Monte Carlo mean of the stochastic discount
   factor `exp(-∫ r)` over many paths against the closed-form bond price
   `P(0, T)` within simulation error, and the exact transition's mean and
   variance against the formulas.
2. **Hull-White**: the model's bond prices reproduce the initial curve to
   1e-10, and the simulated discount factors reproduce it within
   simulation error.
3. **Svensson**: the Federal Reserve's published parameters of the US
   Treasury curve (Gürkaynak, Sack and Wright, updated daily as a public
   dataset) give the published zero yields; a fit to those yields
   recovers the parameters' curve to a basis point. The dataset is to be
   transcribed into `validation/reference/` with its generator.
4. **Inflation**: the OU process's stationary mean and variance and the
   correlation of the shocks, within simulation error; an `IndexPaths`
   from the ESG through `reinflate_draws` against re-inflation by hand on
   one path.
5. **Ruin**: a projection with no economics (zero rates, no inflation,
   no cycle, a constant premium, independent years of compound losses)
   against `ruin::finite_horizon` on the same count and severity, within
   simulation error.
6. **The projection's accounts**: on every path, opening surplus plus the
   year's result less dividends plus injections equals closing surplus, to
   round-off.

## Pull requests

1. `prospicio-esg`: Vasicek, CIR, Hull-White, Svensson and `Scenarios`
   (validation 1 to 3); a new crate, so it also adds its entry in
   `[workspace.dependencies]`, the umbrella re-export, the WASM check and
   the lane table's small PR.
   **Done** (branch `claude/project-thread-n5mttf-esg-crate`;
   `validation/tests/aggregate_esg.rs`). What it settled:
   * the crate depends on `prospicio-core`, `-math` and `-prob` only, as
     decision 1 has it, so a simulated curve is a `Curve` of annually
     compounded spot rates by maturity, for
     `prospicio_reserving::DiscountCurve::spot`, rather than a
     `DiscountCurve`; Hull-White's initial curve is an `InitialCurve` (a
     spot curve on `DiscountCurve`'s convention, or a `Svensson` fit),
     and the test checks that both conventions price alike to 1e-10;
   * Vasicek and Hull-White simulate the Gaussian factor and its integral
     over a step jointly and exactly, so their deflators are exact and
     Hull-White's reproduce the initial curve in expectation; CIR's rate is
     exact (the noncentral chi-square, as `(Z + √λ)² + χ²(d - 1)` when its
     degrees exceed 1 and as a Poisson mixture otherwise) and its integral
     the trapezoid over the step, which moves a monthly deflator by far
     less than a basis point of price;
   * `Svensson::fit` searches the decay times on a log grid of 0.25 to 30
     years with the betas linear, then refines them by Nelder-Mead; it
     recovers a curve from its yields to a basis point. The Federal
     Reserve's published parameters (validation 3) could not be fetched
     from a cloud session (`knowledge/environment/cloud-network.md`): that
     parity waits for a local session, as a Next item;
   * paths are drawn one after another, each from its own stream
     (`StreamRng::new(seed, path)`), with no threads, so the crate is in
     the WASM scope and a later parallel split keeps the numbers;
   * the correlation of the shocks arrives with inflation in PR 2, the
     first second shock.
2. Inflation and superimposed inflation, `IndexPaths` from the scenarios
   (validation 4). **Done** (`validation/tests/aggregate_esg_inflation.rs`).
   As built:
   * `Inflation` (OU on the continuously compounded rate, the price index
     `exp(∫ i)`) and `Superimposed` (AR(1) on the annual excess, from its
     mean) are options of `Esg`, with `correlation` over the shocks
     simulated, in the order rate, inflation, superimposed (empty for
     independence);
   * the Gaussian factors (Vasicek's or Hull-White's rate, inflation) and
     their integrals over a step are one exact Gaussian draw: two noises'
     covariance is `ρ ∫_0^Δ f g` of their kernels (`σ e^(-av)`,
     `σ B_a(v)`, or 1 for the Brownian increment), factored once a
     simulation; a rate-only generator draws the same numbers as PR 1;
   * year `y`'s superimposed shock is drawn at the year's end given the
     year's Brownian increments of the rate and inflation (a regression on
     them, the rest independent), so its correlation with each increment is
     the given one; the year's excess is spread evenly over the year, so the
     claims index inside a year looks ahead to the year's end (fine for a
     scenario set, which is drawn whole);
   * a CIR rate is not Gaussian, so its shock cannot be correlated (an
     error); inflation runs beside it independently;
   * `Scenarios::index_levels(Index, grain)` gives each path's levels at
     the end of each period of the grain, 1 today; the crate does not
     depend on `prospicio-reserving`, so the caller scales them by the
     `CalendarIndex`'s last level and builds the `IndexPaths`.
3. `prospicio-aggregate::dfa`: `Projection` and `ProjectionPaths`, the
   accounts and the ruin and breach estimates (validation 5 and 6).
   **Done** (branch `claude/project-thread-n5mttf-esg-dfa`,
   `validation/tests/aggregate_dfa.rs`). As built:
   * `prospicio-aggregate` depends on `prospicio-esg`; `Projection::run`
     takes a `Scenarios` set path by path, and `run_without_economy` has no
     investment return and an index of 1 (validation 5's setting);
   * the new losses are a simulation-major matrix of paths by years in base
     money (or a `PredictiveDistribution` with one component a year),
     scaled by the path's claims index (or price index); the dependence of
     the losses on the economy beyond that scaling (decision 7's copula)
     comes with PR 4;
   * premium and expenses come at the start of the year and payments at
     its end; the reserves are the payments due at the year end's index,
     with no allowance for later inflation, so the reserve development
     line is the prior reserves' revaluation;
   * cash earns `exp(∫ r) - 1` from the deflators, a bond bucket
     `e^y - 1 - D Δy` on the yield at its duration, which must be one of
     the scenarios' maturities, rebalanced each year;
   * ruin and breach are checked at year ends; `RequiredCapital` is a fixed
     amount, one by year, or a ratio of premium or reserves (decision 9(a));
   * no economy and compound Poisson losses paid in their year give
     `ruin::finite_horizon`'s probability within four standard errors at
     40,000 paths, and the accounts balance to 1e-9 on every path.
4. The underwriting cycle, tax and capital rules.
5. Python and R bindings.
6. A guide: a five-year projection of the guides' treaty book (Guides
   lane, with the user's go-ahead or as an issue).

Each PR appends its Done bullet to `docs/lanes/done/aggregate.md`, adds a
knowledge fragment, and a changelog fragment under "Aggregate".

## Sources

The texts below are cited from the author's knowledge and were not read
for this note; each is read before the PR that relies on it.

- Brigo, D. and Mercurio, F., *Interest Rate Models: Theory and Practice*,
  2nd ed., Springer, 2006: Vasicek, CIR and Hull-White, their bond prices
  and exact simulation.
- Vasicek, O., An equilibrium characterization of the term structure,
  *Journal of Financial Economics* 5, 1977.
- Cox, J., Ingersoll, J. and Ross, S., A theory of the term structure of
  interest rates, *Econometrica* 53, 1985.
- Hull, J. and White, A., Pricing interest-rate-derivative securities,
  *Review of Financial Studies* 3, 1990.
- Nelson, C. and Siegel, A., Parsimonious modeling of yield curves,
  *Journal of Business* 60, 1987; Svensson, L., Estimating and
  interpreting forward interest rates, NBER working paper 4871, 1994.
- Gürkaynak, R., Sack, B. and Wright, J., The U.S. Treasury yield curve:
  1961 to the present, *Journal of Monetary Economics* 54, 2007, and the
  Federal Reserve's dataset of its parameters.
- Kaufmann, R., Gadmer, A. and Klett, R., Introduction to dynamic
  financial analysis, *ASTIN Bulletin* 31(1), 2001: a DFA model of the
  kind section 2 describes, with an AR underwriting cycle.
- Wilkie, A. D., A stochastic investment model for actuarial use,
  *Transactions of the Faculty of Actuaries* 39, 1986 (decision 5(b)).
