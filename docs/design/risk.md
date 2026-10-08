# Design note: risk measures, dependence and allocation

Status: **In progress** · v0.4 · Depends on: `distributions.md`, `predictive-distribution.md` · Lane: Probability

## Goal

Measure the risk of any result (reserve, aggregate loss, tower net),
combine results under a dependence structure, and split a portfolio's risk
back to its parts. All of it works on the representations `prospicio-prob`
already has: sampled draws, grids, and the joint `PredictiveDistribution`.

| Piece | Module | Works on |
|---|---|---|
| VaR, TVaR | `risk` | sorted draws (done) |
| Distortion risk measures | `distortion` | sorted draws, discrete distributions, grids |
| Allocation (co-measures) | `PredictiveDistribution::allocate` | `PredictiveDistribution` |
| Copulas | `copula` | uniforms per simulation, then marginals |
| Iman-Conover | `copula` | reorders existing draws |
| EVT tails | `evt` | sorted draws (peaks over threshold) |

## What exists

- `prospicio_prob::risk::{var_sorted, tvar_sorted}`.
- `prospicio_prob::Distortion`: `Tvar(p)`, `Wang(λ)`, `ProportionalHazard(ρ)`,
  `DualPower(β)`, `Exponential(k)`, and the families of Mildenhall and
  Major (*Pricing Insurance Risk*, 2022) and CAS Monograph 15: `Ccoc(r)`
  (constant cost of capital), `BiTvar`, `WeightedTvar`, `CappedLinear`,
  `CappedLogLinear`, `Lep`, `LinearYield`, `Beta`, `Mixture`, `Minimum`
  and `Convex` (the concave hull of `(s, g)` points, for distortions read
  off cat bond or credit spreads). Methods `g`, `g_inv`, `g_dual` (the
  bid), `mass` (`g(0+)`), `weights(n)`, `apply_sorted`, `apply_discrete`,
  `apply_discrete_dual`; `Empirical::distortion` and `Grid::distortion`.
- `distortion::calibrate(Family, values, probs, premium)`: the member of a
  one-parameter family (`Family::STANDARD` is CCoC, PH, Wang, dual and
  TVaR) whose price is the premium. Python `prospicio.risk.calibrate`, R
  `calibrate_distortion()`; both cap the loss at `assets` when given.
- `PredictiveDistribution::allocate(&Distortion)`: co-measure allocation
  of the total's risk measure to the components (CoTVaR for `Tvar`).
- `PredictiveDistribution::capital(&Distortion, AllocationMethod)` in
  `prospicio_prob::capital`: the total's measure, each component's stand-alone
  measure and an allocation by `Euler`, `Covariance`, `Proportional`,
  `Marginal` (Merton–Perold) or `Shapley`, with the diversification
  benefit overall and per component.
- `prospicio_prob::copula`: the `Copula` trait, `GaussianCopula`,
  `StudentTCopula`, and `copula::simulate` to join marginals into a
  `PredictiveDistribution`. `prospicio_math` gained `linalg::cholesky`,
  `special::beta_inc` and `special::student_t_cdf` for them.
- `copula::iman_conover(&pd, correlation, seed)`: reorders each
  component's draws to a target correlation.
- `ArchimedeanCopula` (Clayton, Gumbel, Frank, Joe), exchangeable in any
  dimension.
- `prospicio_prob::evt`: `Gpd` (generalized Pareto, with a maximum likelihood
  `Gpd::fit`, an optional location, and closed-form layer moments as a
  `Severity`; see `pareto.md`) and `PotTail`, a peaks-over-threshold tail fitted to the
  draws above an empirical quantile, with VaR and TVaR beyond the draws.
- Python `prospicio.risk` (`Distortion`, `allocate`, the three copula
  classes, `simulate`, `iman_conover`) and R (`distortion`,
  `risk_measure`, `allocate`, `gaussian_copula`, `t_copula`,
  `archimedean_copula`, `copula_sample`, `copula_simulate`,
  `iman_conover`), and for EVT, Python `Gpd` and `PotTail` and R
  `gpd_fit` and `pot_tail` (with `VaR` and `TVaR` methods).

- Spectral measures with exponential risk aversion,
  `Distortion::Exponential(k)`: `g(s) = (1 - e^(-ks)) / (1 - e^(-k))`.
- Exponential-utility measures, which are not distortions:
  `risk::entropic` (`(1/θ) log E e^(θX)`, convex but not positively
  homogeneous) and `risk::esscher` (the Esscher premium).
- Systemic measures on a `PredictiveDistribution`:
  `marginal_expected_shortfall(p)` (the CoTVaRs, `allocate` with `Tvar`),
  `covar(component, p, q)` (the total's VaR at `q` given the component at
  or above its VaR at `p`, as Girardi and Ergün) and
  `esscher_allocation(h)` (adds up to the total's Esscher premium).
  Python `Distortion.exponential`, `entropic`, `esscher`,
  `marginal_expected_shortfall`, `covar`, `esscher_allocation`; R
  `distortion("exponential", k)`, `entropic_risk`, `esscher_premium`,
  `marginal_expected_shortfall`, `covar`, `esscher_allocation`.

- EVT diagnostics: `evt::mean_excess` (the empirical mean-excess function
  with counts above each threshold, linear above a threshold where a GPD
  fits) and `evt::hill` (Hill estimates of the tail index for a range of
  `k`), the usual aids to choosing a threshold before `PotTail::fit`.

- `prospicio_pricing::natural`: the pricing and natural allocation of
  Mildenhall and Major (*Pricing Insurance Risk*, 2022) and CAS Monograph
  15. A `Portfolio` holds the total's distinct values with their
  probabilities and each unit's `κᵢ(x) = E[Xᵢ | X = x]`, from scenarios
  (`from_rows`, `from_predictive`) or from independent units' grids by FFT
  (`from_independent`). `price(g, a, Allocation)` prices `X ∧ a` and
  allocates loss (equal priority in default), margin, premium, capital
  and assets to units, linear or lifted; `bodoff(a)` is Bodoff's
  percentile layer of capital; `epd(a)` and `assets_for_epd`; `assets(p)`
  the capital standard; `calibrate(Family, a, Target)` to a premium,
  return or loss ratio. `Pentagon` holds `L, M, P, Q, a` and
  `Pentagon::solve` fills it from any three determining amounts or ratios.
  Python `prospicio.pricing.Portfolio`, `NaturalPrice`, `Pentagon`; R
  `capital_portfolio()`, `natural_price()`, `calibrate_portfolio()`,
  `bodoff_allocation()`, `epd_ratio()`, `assets_for_epd()`, `pentagon()`.

## Decisions

- **A distortion is a closed enum** of concave distortions of the
  survival function, so every measure offered is coherent; it is `Clone`,
  not `Copy`, because mixtures and weighted TVaRs hold vectors. Custom
  distortions wait for a use. `Beta` needs `a <= 1 <= b` and
  `CappedLogLinear` `b <= 1` to be concave. aggregate's Wang-t and power
  distortions are left out: Wang-t is not concave for every parameter,
  and the power distortion is defined through a severity.
- **A distortion can put mass on the largest outcome.** `Ccoc`, and
  `CappedLinear`, `Lep` and `LinearYield` with `r0 > 0`, jump at `s = 0`;
  `apply_discrete` handles the jump without a special case, because the
  top value's weight is `g(P(X = max))`, which includes the mass. That is
  how `Ccoc(r)` prices `ν E[X] + δ max X`.
- **Calibration solves on the discrete distribution, by bisection.** Every
  family's price is monotone in its parameter, so bisection (on a log
  scale for an open-ended parameter, after doubling past the target)
  reaches full precision; `Ccoc` is closed form,
  `r = (P - E[X]) / (max X - P)`. The caller caps the loss at the assets
  (`min(X, a)`) to price a limited-liability portfolio, as Mildenhall and
  Major do; the premium must lie strictly between the capped mean and
  maximum. LEP's price is bounded below the maximum (its `g` tends to
  `min(1, s + √(s (1 - s)))`), so a high premium can have no LEP.
- **The measure is a weighted sum by rank.** For `n` equally likely draws
  sorted ascending, draw `i` (0-based) gets
  `g((n - i) / n) - g((n - i - 1) / n)`. On a discrete distribution the
  value `x_k` gets `g(P(X >= x_k)) - g(P(X > x_k))`, with survival summed
  from the top so small tail probabilities keep their precision. Ties
  need no special case: tied draws share the weight of one atom.
- **TVaR is one of the distortions**, `g(s) = min(s / (1 - p), 1)`, and
  agrees with `tvar_sorted` (including the fractional weight at the VaR),
  so allocation of TVaR and of any other distortion is one code path.
- **Allocation is by co-measure (Euler).** For a joint distribution with
  total `S = Σ X_j`, sort the simulations by `S`, take the distortion's
  weights by rank, and give component `j` the weighted sum of its own
  values in that order. The contributions sum to `ρ(S)` exactly (up to
  rounding), and for `Tvar(p)` they are the CoTVaRs,
  `E[X_j | S in its top 1 - p]`. Simulations tied on `S` share their
  weights equally, so the result does not depend on how ties are sorted.
- **Several allocation methods, one report.** Euler is the default
  because it is the only method consistent with marginal changes to the
  portfolio. The others answer different questions: `Covariance` looks at
  the whole distribution, not the tail. `Proportional` ignores dependence.
  `Marginal` (`ρ(S) − ρ(S − X_j)`) does not add up, and the shortfall is
  capital no single component causes. `Shapley` averages marginal
  contributions over every joining order. Every method returns the
  stand-alone measures too, so the diversification benefit
  `Σ ρ(X_j) − ρ(S)` and its split come with the allocation.
- **Shapley is exact and capped at 12 components.** It evaluates `ρ` on
  all `2^m` sub-portfolios, each a sort of `n` draws. Beyond 12 components
  that cost grows too fast, and a sampled Shapley value would add noise
  the other methods do not have.
- **Copulas generate uniforms; marginals stay where they are.** A copula
  draws one vector of uniforms per simulation from
  `StreamRng::new(seed, sim)` (the `chacha20/sim-index/v1` scheme), and
  marginals are applied by inverse transform, so any simulation replays
  alone and results do not depend on thread count.
- **Normals by inverse transform, chi-square by Marsaglia–Tsang.** A
  copula draw takes its `d` normals first, as `Φ⁻¹(U)`, then (t copula)
  one chi-square from the same stream. Rejection sampling keeps each draw
  a pure function of `(seed, sim)`.
- **Archimedean copulas by frailty** (Marshall–Olkin): one frailty `V`
  per draw (gamma, positive stable by Kanter's representation,
  logarithmic by Kemp's LK, Sibuya by inverting its distribution
  function), then `d` unit exponentials, `u_j = ψ(E_j / V)`. This is exact
  in every dimension, uses a fixed number of uniforms except for the gamma
  rejection step, and gives exchangeable dependence only; nested and
  vine structures come later.
- **Uniforms stay inside `(0, 1)`.** Far in a tail, `Φ` or a generator
  rounds to 0 or 1, where marginal quantiles are infinite, so every copula
  clamps to `[f64::MIN_POSITIVE, 1 - 2^-53]`.
- **EVT tails are peaks over threshold.** `PotTail::fit(draws, level)`
  takes the threshold `u` at the empirical `level` quantile and fits a GPD
  to the exceedances; `P(X > x) = p_u S_GPD(x - u)`, so VaR and TVaR at
  levels above `level` have closed forms and extend past the largest
  draw. Choosing the threshold stays the caller's job.
- **The GPD fit maximizes the profile likelihood in `θ = ξ / β`**
  (Grimshaw): a log-spaced scan over `θ` in `(-1/max, ∞)` finds the
  maximum, then bisection on the analytic score refines it, since a search
  on the likelihood value pins a maximum only to about `sqrt(eps)`. The
  estimate is restricted to `ξ > -1`, where the maximum likelihood
  estimator exists.
- **Iman-Conover reorders draws**: it imposes a target correlation on
  existing marginals (for example, reserve and premium-risk results
  simulated separately) by permuting them, so every marginal keeps its
  exact draws. The target is the correlation of normal scores (van der
  Waerden), which is what the method controls; Spearman's rho is then
  close to `(6 / π) asin(ρ / 2)`, as for a Gaussian copula with
  correlation `ρ`. Score columns are shuffled by `StreamRng::new(seed, j)`.

- **Natural allocation follows `aggregate` 1.0.1.** The premium of unit
  `i` is `Σ_{x_k ≤ a} κᵢ(x_k) Δg_k` plus `a g(S(a))` times its share of
  the totals above `a`: the expected share `αᵢ(a) = E[Xᵢ / X | X > a]`
  (linear) or the distorted share `βᵢ(a)` (lifted). Capital goes layer by
  layer: unit `i`'s margin over a layer, `mᵢ(top) - mᵢ(bottom)` with
  `mᵢ` its premium less its expected loss at that asset level, times the
  layer's capital per unit of margin, `(1 - g) / (g - S)`. In layers the
  loss always reaches (`S = g = 1`) that ratio is the limit
  `g'(1) / (1 - g'(1))` (`Distortion::slope_at_one`). Under the linear
  allocation `mᵢ` jumps at each total, and the jump belongs to the layer
  below it, as on `aggregate`'s unit grid.
- **The portfolio works on the distinct totals, not a grid.** Layers run
  between consecutive totals, so any asset level works and no bucket is
  chosen. `S` is summed from the top and is exactly 0 at the largest
  total, which a distortion with a mass needs: a rounding residue there
  would add `mass × max` to the price.
- **Tied totals take the probability-weighted mean of the units.**
  `aggregate` takes the unweighted mean of tied scenarios, which differs
  when scenarios have unequal probabilities.

## Validation

`validation/tests/distributions.rs` checks distortions against
`validation/reference/distortion_mpmath.csv`
(`validation/scripts/mpmath_distortion.py`, 30 digits):

- an exact discrete distribution with a `1e-10` tail atom, at `1e-14`;
- the continuous integral `∫ g(S(x)) dx` for two lognormals, evaluated on
  a local-moment grid. The tolerance is `2e-4` for discretization plus the
  integral of `g(S)` above the last grid point, computed by the script, so
  each row states where its error comes from. The Wang rows also equal the
  closed form `exp(μ + λσ + σ²/2)` to 16 digits.

`validation/reference/distortion_aggregate.csv`
(`validation/scripts/aggregate_distortions.py`, Mildenhall's `aggregate`
1.0.1) checks `g` of every family the two share at twelve levels, at
`1e-12`, and the Monograph 15 InsCo example: ten equally likely totals,
assets 100, priced at a 15% cost of capital, `P = 53.565`. The price at
aggregate's calibrated CCoC, PH, Wang, dual and TVaR parameters matches at
`1e-12`, and `calibrate` recovers each parameter within aggregate's own
tolerance: aggregate stops at a premium error (up to `7.6e-6` for TVaR),
so the script turns that error into a parameter tolerance through the
price's slope.

`validation/reference/natural_aggregate.csv`
(`validation/scripts/aggregate_natural.py`) checks the natural allocation
against `aggregate` 1.0.1 at `1e-10`, 1,060 values: InsCo and the
*Pricing Insurance Risk* Discrete case (two independent units, built by
FFT), each priced under the CCoC, PH, Wang, dual and TVaR distortions
calibrated to it, linear and lifted, at three asset levels including two
with default; the loss, margin, premium, capital and assets of each unit
and the total, and Bodoff's allocation. Unit tests check that every
allocation adds up, `from_independent` against brute-force enumeration,
EPD, and that 46 of the 56 triples of pentagon quantities solve.

Unit tests check that `Tvar(p)` matches `tvar_sorted` for 101 levels, that
weights are a non-decreasing probability vector, and coherence: translation
and scale equivariance, bounds between the mean and the maximum, and
monotonicity in each parameter. Every family is checked to be
non-decreasing, above the diagonal and concave on a 1,000-point grid,
with its mass the limit at 0; `g_inv` against `g`; the bid below the mean
below the ask; `Ccoc` against the equivalent `BiTvar` and its closed-form
price; the convex hull dropping interior points and keeping a jump at 0;
and calibration of every family to three premiums.

`validation/reference/special_scipy.csv` checks `beta_inc` and
`student_t_cdf` against SciPy at `1e-12`. Copula tests check Kendall's
tau against `(2 / π) asin(ρ)` for the Gaussian and t copulas (including
`nu < 1`), uniform margins by a Kolmogorov–Smirnov test, the t copula's
joint extremes against its tail-dependence coefficient, the gamma
sampler's moments, and that any simulation replays alone. Archimedean
copulas are checked against their Kendall's tau (closed forms for Clayton
and Gumbel; Frank's Debye integral and Joe's series computed in the
test), uniform margins, and tail dependence: Clayton's lower tail
`2^(-1/θ)` and Gumbel's upper tail `2 - 2^(1/θ)`. Each frailty sampler has
a moment or probability check. Iman-Conover
is checked on three lognormal lines: marginals unchanged draw for draw,
normal-score correlations within 0.01 of the target, and Spearman's rho
within 0.01 of `(6 / π) asin(ρ / 2)`.

`validation/reference/gpd_mpmath.csv` checks GPD fits against the exact
maximum likelihood estimate at 30 digits (`validation/scripts/mpmath_gpd.py`,
the score root by the Illinois method) on four data sets with
`ξ` from -0.2 to 1.1, at `1e-8`. Unit tests check that fits recover
known parameters from 50,000 draws, the GPD's closed forms, and that a
`PotTail` built from known parts gives `P(X > VaR(p)) = 1 - p`.

Allocation tests check, on simulated dependent lines, that contributions
sum to the measure of the total for every distortion, that the mean
allocates to component means, and that CoTVaR equals the conditional tail
mean computed directly. Two exact cases: comonotonic lines (`X_2 = 2 X_1`)
each receive their own risk measure, and two simulations tied on the
total but split differently give the same allocation in either order.

`validation/reference/allocation_numpy.csv`
(`validation/scripts/numpy_allocation.py`) checks every method against
numpy, using four integer-valued components over 400 simulations with
many tied totals. The cases are TVaR at 75% and 90%, proportional hazard
and dual power, at `1e-12`. The script writes each method from its
definition: the measure over distinct values, Euler with tied weights
shared, and Shapley over all 24 joining orders rather than the subset
formula. Unit tests check that full allocations add up, that marginal
allocations fall short, the two-component Shapley closed form, and that
comonotonic lines have no diversification benefit.

## Next

1. Nested Archimedean and vine copulas; threshold diagnostics (mean
   excess plots) for EVT.
