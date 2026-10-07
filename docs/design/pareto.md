# Design note: Pareto-family severities for reinsurance pricing

Status: **Accepted** · v0.4 · Depends on: `distributions.md` (`Severity`, `Counting`, `Grid`), `aggregate.md` (towers), `risk.md` (`evt::Gpd`) · Lanes: Probability (`prospicio-prob`), Aggregate (`prospicio-aggregate`, `prospicio-pricing`)

## Goal

Price excess-of-loss treaties the way reinsurance pricing actuaries do:
describe large losses by Pareto alphas, rate layers from each other, and
fit one frequency–severity model to everything known about a programme
(experience-rated lower layers, exposure-rated upper layers, excess
frequencies, PML points). That model then feeds the machinery that
already exists: exact layer costs, Panjer/FFT aggregates, simulated
towers with annual terms, risk measures and allocation.

The distributions and methods follow Ulrich Riegel's R packages
[Pareto](https://github.com/ulrichriegel/Pareto) (CRAN) and
[LocalPareto](https://github.com/ulrichriegel/LocalPareto), and the papers
behind them (Riegel 2008, 2018, 2025). Both packages are GPL; this
project plans MIT/Apache-2.0 (`docs/architecture.md`), so **we implement
from the published mathematics and use the packages only to generate
parity references**, the same way SciPy and mpmath are used. No R code is
translated.

## Why Pareto-type distributions for treaty pricing

### The local Pareto alpha is the language of the market

For a severity with survival function `S`, the **local Pareto alpha** is
the elasticity of the survival function,

```text
α(x) = −x S'(x) / S(x),      so      S(x) = S(t) · exp(−∫_t^x α(u)/u du).
```

It says how fast the excess frequency falls as the threshold rises: if
the alpha at 5m is 2, the frequency of losses above 5m falls by about 2%
for each 1% increase in the threshold. Reinsurance pricing quotes tails
in this unit ("property per-risk alpha around 1.5, motor liability
around 2.5"). Every distribution in this note is described by its local
alpha, and that one function tells most of the pricing story:

| Distribution | Local alpha `α(x)` | Tail |
|---|---|---|
| Pareto (single parameter) | constant `α` | heavy, the same at every threshold |
| Piecewise Pareto | step function: `α_k` on `[t_k, t_{k+1})` | any shape, piece by piece |
| Generalized Pareto (Riegel) | moves from `α_ini` at `t` to `α_tail` as `x → ∞` | heavy, with a different start and end |
| Log-affine local Pareto | `α₀ (1 + γ ln(x/t))`, rising linearly in `ln x` | lighter further up, lognormal-like |
| Lognormal (for comparison) | `≈ (ln x − μ)/σ²` far in the tail | rising linearly in `ln x` |
| Weibull (for comparison) | `k (x/λ)^k` | rising as a power, light |

The lognormal row is why the log-affine local Pareto matters: for large
`x` a lognormal's local alpha is asymptotically affine in `ln x` (for
`σ = 1` it is 1.53, 2.37, 5.19 at `z = 1, 2, 5`), so the log-affine
local Pareto has a lognormal-type tail, written in the units pricing
actuaries use and anchored at the threshold where the data starts.

### Scale invariance and threshold invariance

For `X ~ Pareto(t, α)`:

- `cX ~ Pareto(ct, α)`: the alpha does not change with currency or
  inflation, so an alpha fitted on historical losses still applies after
  indexation, and alphas from different markets are comparable.
- `X | X > s ~ Pareto(s, α)` for `s > t`: the alpha does not depend on
  the reporting threshold. Large-loss data always arrives above some
  threshold (often several, one per data source), and the fitted alpha
  does not move when the threshold does.

Piecewise Pareto, the generalized Pareto and the local Pareto keep this
property in the form that matters: the distribution above a higher
threshold is again a member of the same family, with the local alpha
unchanged. That is what lets data above different reporting thresholds
be pooled, and what makes per-loss reporting thresholds in the maximum
likelihood fits natural.

### Layers can be priced from each other

Above the threshold, the ratio between expected losses of two layers
depends only on the alpha:

```text
E[min(c, (X − a)⁺)] = ∫_a^{a+c} S(x) dx = t^α ∫_a^{a+c} x^{−α} dx      (a ≥ t),
```

so the factor `t^α` cancels in any ratio. That gives the standard rating
moves, all in closed form:

- **Extrapolation:** with alpha 2 and an expected loss of 500 in
  4000 xs 1000, the expected loss of 5000 xs 5000 is 62.5. Burning-cost
  results on working layers price the unexposed layers above them.
- **Implied alpha:** two layers, a frequency and a layer, or two
  frequencies each determine the alpha between them (for example, 500 in
  4000 xs 1000 and 62.5 in 5000 xs 5000 imply alpha 2). This is how a
  pricing actuary checks whether an experience rate and an exposure
  rate are mutually consistent.
- **Frequency extrapolation:** `f(s) = f(t) (t/s)^α`.

### One model for a whole tower

A programme is usually priced in pieces: lower layers from experience
(burning cost), upper layers from exposure curves or market benchmarks.
Pricing each layer on its own can leave the tower inconsistent: no single
loss distribution produces all the layer costs at once, so aggregate
features that span layers (inuring, annual aggregate deductibles and
limits, reinstatements, stop-loss) cannot be priced coherently.

Riegel (2018) shows how to fit **one collective model** (a Panjer claim
count with a piecewise Pareto severity) that reproduces every layer's
expected loss exactly:

- The condition is on the **risk rate on line** (RRoL), the expected
  loss divided by the cover. It is the average excess frequency across
  the layer, so it can only fall as the attachment rises. The paper uses
  this term because "rate on line" usually means premium over cover. If
  the RRoLs of a contiguous tower are strictly decreasing, a matching
  model exists. If two layers have the same RRoL, an exact match needs a
  point mass, and merging those layers is the practical fix. If the
  RRoLs ever rise, no severity matches, and the inputs need review before
  any model is fitted.
- One Pareto piece per layer, chaining alphas up the tower (the paper's
  Matching Algorithm 1), always works for up to three layers but can fail
  beyond that. The paper's four-layer example (500 xs 1000, 1500, 2000,
  2500 with expected losses 100, 90, 50, 40, RRoLs 0.20, 0.18, 0.10, 0.08)
  cannot be matched this way for any starting frequency.
- Two pieces per layer, with a free intermediate threshold (Matching
  Algorithm 2), always work when the RRoLs are strictly decreasing
  (Theorem 1), and can also reproduce given excess frequencies at the
  attachment points. The R package's fit of that example (with an
  unlimited top layer at 3000 costing 100) reproduces all five expected
  losses; the algorithm is set out below.

The same machinery fits partial reference information (some layers,
some excess frequencies) and PML curves (return period and amount pairs
become excess frequencies).

Piecewise Pareto is a good model for this because it is dense in the
positive distributions (in the Lévy metric), so it imposes no shape, yet
every quantity a pricing actuary needs (layer mean, layer variance,
excess frequency, quantile) stays in closed form.

### Smooth alternatives to piecewise Pareto

Piecewise Pareto has kinks: its local alpha jumps at each threshold.
When the data or a benchmark says the alpha should move smoothly with
size, two parametric families describe that with two parameters:

- **Generalized Pareto (Riegel 2008):** `α_ini` at the threshold, moving
  to `α_tail` far out. It suits lines whose working layers look lighter
  or heavier than their tail. It is the extreme value GPD
  (`evt::Gpd`) reparameterized and shifted to start at `t`, with
  `ξ = 1/α_tail` and `β = t/α_ini`. One distribution, two
  parameterizations: EVT's for tail estimation, Riegel's for pricing.
- **Log-affine local Pareto (Riegel 2025):** `α(x) = α₀(1 + γ ln(x/t))`,
  equivalently `δ = α₀ γ ln 2`, the rise in alpha each time the amount
  doubles. It describes a tail that thins out with size (lognormal-like)
  through two directly interpretable numbers: the alpha at the threshold
  and its increase per doubling. Layer moments are in closed form.

A **general local Pareto** takes any local alpha function `α(x)` and is
converted to a piecewise Pareto with a stated maximum relative error in
the survival function, so every layer formula applies and the
approximation error is reported, as `Grid` discretization reports its
error.

### Fitting real large-loss data

Large-loss data is rarely complete. The fits handle the two usual
defects per loss:

- **Reporting thresholds:** losses from one source are known only above
  1m, from another only above 3m. Ignoring this biases the alpha down.
- **Censoring at policy limits:** a loss at a 5m policy limit is "at
  least 5m". Ignoring this biases the alpha up.

The single-parameter Pareto has a closed-form estimator with both;
piecewise Pareto, generalized Pareto and local Pareto are fitted
numerically.

### Where it plugs into this project

```text
large losses ──fit (thresholds, censoring)──► severity (Pareto family)
layer costs, frequencies, PML ──tower matching──► collective model
                                                      │
          ┌───────────────────────────┬───────────────┴────────────┐
   exact layer mean / variance   Grid → Panjer / FFT        simulate_events
   excess frequency              aggregate distribution     → Tower (inuring,
                                                              AAD/AAL, reinstatements)
                                                            → PredictiveDistribution
                                                            → Distortion / allocate
                                                            → risk_load: premium,
                                                              margin, capital
```

A collective model is any `Counting` with any `Severity`, so everything
downstream already exists; this work adds the severities, the claim
count chosen by dispersion, and the fitting and matching on top.

## Tower matching algorithm

Riegel (2018), Matching Algorithm 2 and the proof of Theorem 1
(Proposition 2), restated here as the specification we implement.

**Input:** contiguous layers `c_i xs a_i`, `i = 1..k`, with
`a_{i+1} = a_i + c_i` and the top layer unlimited (`c_k = ∞`), their
expected losses `e_i > 0`, and strictly decreasing risk rates on line
`e_1/c_1 > … > e_{k−1}/c_{k−1} > 0`. Optionally, expected excess
frequencies `f_i` at the attachment points with `f_1 > e_1/c_1` and
`e_{i−1}/c_{i−1} > f_i > e_i/c_i`, which must lie between the RRoLs of
the layers around each attachment point.

**Output:** a piecewise Pareto severity with `2k − 1` pieces,
thresholds `t_{2i−1} = a_i` and one free threshold `t_{2i}` inside each
limited layer, and `E[N] = f_1`. The model reproduces every `e_i` and
every `f_i`.

Writing `I_{t,α}(a, b)` for the expected loss of `b − a xs a` per loss
of a `Pareto(t, α)` (closed form; `t ln(b/a)` at `α = 1`):

1. **Frequencies, if not given.** Let `α*_i` be the Pareto alpha between
   layers `i` and `i + 1` (the unique alpha whose layer-loss ratio is
   `e_i/e_{i+1}`). Set `f_1 = e_1 / I_{a_1, α*_1}(a_1, b_1)` and
   `f_i = e_i / I_{a_i, α*_{i−1}}(a_i, b_i)` for `i ≥ 2`. These satisfy the
   bounds above.
2. **Normalize:** `s_i = f_i/f_1` (excess probabilities, `s_1 = 1`) and
   `l_i = e_i/f_1` (expected layer loss per loss).
3. **Top piece:** `α_{2k−1} = s_k a_k / l_k + 1`, from the unlimited
   layer's mean excess.
4. **Two pieces per limited layer.** For a candidate middle threshold
   `τ ∈ (a_i, a_{i+1})` and lower alpha `α`, the upper alpha that hits
   `s_{i+1}` at `a_{i+1}` is

   ```text
   σ(τ, α) = (ln(s_{i+1}/s_i) − α ln(a_i/τ)) / ln(τ/a_{i+1}),
   ```

   and `λ(τ, α) = s_i · I_{(a_i, τ),(α, σ(τ, α))}(a_i, a_{i+1})` is the
   resulting layer loss per loss. `λ` is decreasing in `α` and both
   bounding curves are increasing in `τ`, so bisection finds the feasible
   range `(τ_l, τ_u)`:
   `τ_l = inf{τ : λ(τ, 0) > l_i}` and
   `τ_u = sup{τ : λ(τ, ln(s_{i+1}/s_i)/ln(a_i/τ)) < l_i}`, the second
   being the `α` at which `σ = 0`. The paper proves `τ_l < τ_u`.
5. **Pick `t_{2i}` in `(τ_l, τ_u)`.** For that `t_{2i}`, solve
   `λ(t_{2i}, α) = l_i` for `α_{2i−1}` by bisection, then set
   `α_{2i} = σ(t_{2i}, α_{2i−1})`. Two selection rules:
   - **midpoint**, `t_{2i} = (τ_l + τ_u)/2`: deterministic and cheap;
   - **minimize** `max(α_{2i−1}/α_{2i}, α_{2i}/α_{2i−1})` (the default in
     the paper's example and in the R package), so the two pieces of a
     layer have similar alphas: a one-dimensional search over `t_{2i}`.

Every step is a closed form or a bisection on a monotone function, so the
algorithm cannot fail on a consistent tower, and an inconsistent one is
caught before any search: a non-decreasing RRoL, or given frequencies
outside their bounds. (The top piece's alpha, `s_k a_k / l_k + 1`, is
always above 1, so the unlimited layer never fails.)

**What we add around the paper** (the R package's options, re-derived):
input as unlimited-layer losses `u_i = Σ_{ν≥i} e_ν` as well as layer
losses; merging consecutive layers with equal RRoL; merging consecutive
pieces whose alphas agree; and an upper bound on the alphas. Point masses
for total-loss frequencies and truncation of the top piece come later.

**Check against the paper:** Example 4 (Table 3 tower, `f_1 = 0.25`,
minimize rule). The current R package (2.4.5) reproduces the first six
pieces of the paper's Table 4 to every printed digit
(`t = 1000, 1097, 1500, 1932, 2000, 2148`;
`α = 2.374, 0.199, 0.175, 9.685, 3.539, 0.817`; excess frequencies
`0.250, 0.201, 0.189, 0.180, 0.129, 0.100`). Above 2500 the table has
15 thresholds where the algorithm produces `2k − 1 = 13`, so it was
apparently made with an earlier version or slightly different upper
layers. We test against the six printed pieces and against R 2.4.5 for
the whole tower.

### Log-affine local Pareto without the preprint

The 2025 preprint is not available, so the log-affine formulas are
derived here from the definition. With `L = ln(x/t)` and
`α(x) = α₀(1 + γL)`,

```text
S(x) = exp(−α₀ L − ½ α₀ γ L²)      (x ≥ t),
```

a Gaussian in `L`. A layer's expected loss is
`∫ S(x) dx = t ∫ exp((1 − α₀)L − ½ α₀ γ L²) dL`, and its second moment
uses `exp((2 − α₀)L − …)`. Both complete the square and reduce to normal
distribution functions, so every layer quantity is in closed form via
`norm_cdf`, which `prospicio-math` already has. They are checked against
mpmath integration and against the LocalPareto package. The general
local Pareto (any `α(x)`) is converted to a piecewise Pareto by our own
adaptive scheme with a stated bound on the relative error of `S`.

## Plan

### `prospicio-prob` (Probability lane)

Done: `Severity::layer_second_moment` (with a default `layer_variance`),
`Pareto`, and `PiecewisePareto` with `Truncation::{LastPiece,
WholeDistribution}`. Survival at the thresholds is stored as a logarithm
so steep pieces do not underflow, and truncated survival is computed as
`S(x) (1 − S(T)/S(x))` so it keeps full precision just below `T`.
Riegel's generalized Pareto is `evt::Gpd::riegel(t, alpha_ini,
alpha_tail)`: `Gpd` gained a location (`Gpd::shifted`) and implements
`Severity`, with layer means and second moments in closed form for every
`ξ` (series for small `c / β`, where the closed forms cancel). Truncation
of the generalized Pareto, which the R package also offers, waits for a
use.
`LargeLosses` holds large-loss data with per-loss reporting thresholds,
censoring flags and weights; `Pareto::fit` (closed form, or a score
bisection when truncated) and `PiecewisePareto::fit` (closed form per
piece; a truncated last piece is a truncated Pareto fit) estimate alphas
from it. Truncated estimates are clamped to `[1e-3, 1e3]`, since data
rising towards the truncation point can put the maximum at or below 0.
`Gpd::fit_riegel` fits Riegel's generalized Pareto to the same data:
for fixed `k = α_ini/α_tail` the tail alpha is a closed form, so it is a
one-dimensional profile likelihood in `k`, solved by bisection on its
analytic score. R's optimizer stops about `2e-5` short of the maximum
(our estimate has the higher likelihood at 40 digits), so those parity
rows use `1e-4`. `PiecewisePareto::fit` with
`Truncation::WholeDistribution` maximizes the coupled likelihood by
coordinate ascent from the untruncated estimates (each alpha by the
Illinois method on its analytic partial derivative, clamped to
`[1e-3, 1e3]`), with `S(a) − S(T)` computed without cancellation. Our
estimates have the higher likelihood than R's in every case; R stops up to
`2e-3` away (compared at `5e-3`), and on a weighted data set whose
likelihood is nearly flat it stops 70% away, so that case is left out of
the parity rows. Fits of the local Pareto come later.
`LogAffinePareto` (`new(t, α₀, γ)` or `from_delta(t, α₀, δ)`) has its
layer moments in closed form as above, through the Mills ratio
(`erfc` below `z = 26`, a continued fraction beyond), so layers far in
the tail keep full relative precision: `1e-12` against 30-digit mpmath
integration. The R package LocalPareto computes those moments with
`stats::integrate` (relative tolerance about `1e-4`), so it is only a
coarse cross-check (`1e-6`) there, and an exact one for the distribution
function and quantile. `local_pareto_to_piecewise(t, α, options)` converts the
general local Pareto (any `α(x)`) to a piecewise Pareto. In `L = ln(x/t)`,
`ln S = −A(L)` with `A` the integral of the local alpha; a piecewise
Pareto is a chord interpolant of `A` that matches `S` exactly at its
thresholds, so pieces are grown greedily while the gap between `A` and
its chord keeps the relative error of `S` within the tolerance at 16
interior points (`A` by Gauss–Legendre). It stops at an amount or a
survival level and reports the largest error found and where the
approximated range ends. Tested against the closed-form log-affine
survival and a wavy alpha with an exact integral; the R package's
`LocalPareto_2_PiecewisePareto` places its breakpoints differently, so the
two are not compared piece by piece.
Bindings: Python `prospicio.distributions` (`Pareto`,
`PiecewisePareto`, `LogAffinePareto`, `GeneralizedPareto` with `riegel`,
`Binomial`, `claim_count`, and `fit` static methods) and R (`pareto()`,
`piecewise_pareto()`, `log_affine_pareto()`, `generalized_pareto()`,
`generalized_pareto_riegel()`, `pareto_fit()`, `piecewise_pareto_fit()`,
`binomial_count()`, `claim_count()`, with `survival()`,
`layer_variance()` and `local_alpha()`). Every severity also works with
grid discretization, compound distributions and event simulation.
The generalized Pareto fit (`GeneralizedPareto.fit_riegel`,
`generalized_pareto_fit()`), whole-distribution truncated piecewise fits
(`truncation_type="wd"`) and `local_pareto_to_piecewise` (taking a Python
callable or an R function as the local alpha) are bound too.
`Binomial` and `PanjerClass::from_mean_dispersion` complete the Panjer
class. A binomial needs a whole number of trials, so below dispersion 1
the trials are `mean / (1 − dispersion)` rounded up: the mean is kept and
the dispersion moves up to the nearest attainable value, which
`PanjerClass::dispersion` reports.

| Item | Notes |
|---|---|
| `Severity::layer_second_moment` | Layer variance for every severity; closed form for the Pareto family and the lognormal. |
| `Pareto { t, alpha, truncation }` | `Distribution` + `Severity`, closed-form layer moments; `α = 1` and `α = 2` handled as their logarithmic limits. |
| `PiecewisePareto { t, alpha, truncation }` | Two truncation modes as in the R package: truncate the last piece, or the whole distribution. |
| Riegel generalized Pareto | A constructor on `evt::Gpd` (with a shift to `t`), not a new type. |
| `Binomial` claim count; `Counting` by mean and dispersion | Dispersion below 1 is binomial, 1 Poisson, above 1 negative binomial (the Panjer class). |
| Fits with reporting thresholds and censoring | Pareto (closed form), piecewise Pareto, generalized Pareto, local Pareto. |
| `LocalPareto` | Log-affine in closed form; general `α(x)` via conversion to piecewise Pareto with a reported error bound. |
| `local_pareto_alpha` | For any distribution with a density; needs a density method on `Distribution` (only the Pareto family and the lognormal at first). |

### `prospicio-aggregate` (Aggregate lane)

`prospicio-aggregate` stays about aggregate distributions. The collective
model is the frequency–severity input to an aggregate, so it lives here.

Done: `CollectiveModel<N, X>` (with `Box<dyn Severity>` and
`Box<dyn Counting>` usable through blanket impls in `prospicio-prob`), and
`Distribution::survival` so excess frequencies keep their precision far
in the tail.

| Item | Notes |
|---|---|
| `CollectiveModel<N, X>` | Expected layer loss, layer variance (`E[N] Var[Y] + Var[N] E[Y]²`), excess frequency; simulation and Panjer/FFT through existing code. |

### `prospicio-pricing` (Aggregate lane, new crate)

The `pricing` namespace planned in `docs/architecture.md`. Most of it
serves primary and reinsurance pricing alike; only `tower` is
reinsurance-specific.

| Module | Item | Used by |
|---|---|---|
| `layer` | Increased limit factors (`LEV(limit)/LEV(basic)`), deductible credits (`1 − LEV(d)/E[X]`), Pareto extrapolation, implied alpha from two layers, a frequency and a layer, or two frequencies. | Primary (ILF tables, deductibles, large-loss loads) and reinsurance (rating upper layers) |
| `tower` | Tower matching (Riegel 2018, above, both selection rules); reference fits; PML-curve fits. | Reinsurance |
| `exposure` | Exposure curves for property per-risk rating: MBBEFD (Bernegger 1997) with the Swiss Re curves, tabulated curves, the curve of any severity capped at an MPL, layer shares, destruction-rate sampling. | Primary and reinsurance (property per-risk) |
| `risk_load` | Risk-loaded prices from simulated losses: a pricing distortion or a constant cost of capital on distortion-measured assets, for one cover or allocated across a portfolio's components. | Primary and reinsurance (technical price of a simulated cover or programme) |

Done in `layer`: `ilf`, `loss_elimination_ratio`, `XsLayer`,
`pareto_extrapolation`, and `alpha_between_layers`,
`alpha_between_frequency_and_layer` and `alpha_between_frequencies` (all
with optional truncation). The solvers bisect on `(0, 100]`, where each
ratio is monotone in alpha, and say which way an impossible input fails.
As in the R package, a frequency threshold may not lie strictly inside
the layer, whose loss would then include losses the frequency does not
count.

Done in `tower`: `match_tower(attachments, layer_losses, frequencies,
rule)` (Matching Algorithm 2 with `SelectionRule::MinimizeAlphaRatio` or
`Midpoint`) and `layer_losses_from_unlimited`. A limited layer that one
Pareto piece already matches (as the lowest layer always is when its
frequency is derived) gets one piece, not two. Findings from checking
against R 2.4.5:

- With the minimize rule, R agrees to about `1e-8` on Example 4 with
  `f_1` given, derived, or every frequency given.
- R's rule without minimization is not the paper's midpoint, so it is no
  reference for `Midpoint`.
- On the tower 5m xs 5m, 15m xs 10m, ∞ xs 25m (losses 2.4m, 1.5m, 1.2m,
  `f_1 = 1`), R's middle layer is not the minimum of the stated
  objective (spread `0.0703` where `0.0645` is attainable), and at that
  scale its model misses the layer's loss by `1.5e-5`. Our unit tests check
  that tower for exact reproduction, scale invariance and local
  optimality instead.

Overlapping reference layers need a small linear program. It uses the
pure-Rust simplex crate `microlp` (Apache-2.0), as a dependency of
`prospicio-pricing` only; no C or C++ build, so WASM, CRAN and Windows builds
are unaffected.

Done in `tower`: `fit_pml_curve` (closed form: the alpha between
consecutive PML points, a tail alpha above, optionally a truncated last
piece; matches R at `1e-13`) and `fit_references` for any mix of layer
losses (overlapping or with gaps) and excess frequencies. The reference
fit is our own construction, not the package's:

1. The unknowns are the excess-loss function `u` and the frequency `g` at
   every reference point. References are linear equalities, and the
   conditions `match_tower` needs (rates on line strictly between the
   frequencies around them) are linear inequalities. `microlp` finds a
   point with a positive relative margin, by bisection on the margin.
2. A vertex of that program puts free frequencies at the ends of their
   ranges, which gives alphas near 100 in the gaps between references
   and distorts the prices of layers there. So the point is projected
   exactly onto the equalities, then moved to the analytic center of the
   inequalities by Newton's method; gaps then get moderate alphas. A
   lowest frequency that no reference gives is unbounded above, and is
   derived as in step 1 of the algorithm.
3. The completed tower goes to `match_tower`. Every reference is
   reproduced to rounding (`1e-11` in tests). With references for every
   layer and frequency, the result is the tower match itself.

Where the references leave freedom, this completion differs from the R
package's (which fills gaps with a default alpha), so the two are not
compared.

Bindings: Python `prospicio.pricing` (`CollectiveModel`, `ilf`,
`loss_elimination_ratio`, `pareto_extrapolation`, the three implied-alpha
functions, `match_tower`, `fit_pml_curve`, `fit_references`,
`TowerModel`) and R (`collective_model()`, `collective_simulate()`,
`excess_frequency()`, `ilf()`, `loss_elimination_ratio()`,
`pareto_extrapolation()`, `alpha_between_*()`, `match_tower()`,
`fit_pml_curve()`, `fit_references()`, `tower_model`).

Done in `exposure`: `ExposureCurve` (`g(x)`, `layer_share(limit,
attachment, mpl)` = `G(min((a+l)/M, 1)) − G(min(a/M, 1))`), `Mbbefd::new(b,
g)` with its four closed-form cases (`g = 1` or `b = 0`, `b = 1`, `bg = 1`,
general; the special forms are used within 1e-10 of `b = 1` and `bg = 1`,
where the general one cancels), the destruction-rate `cdf`, `mean` and
`total_loss_probability` (`1/g`), `Mbbefd::swiss_re(c)` (`b = exp(3.1 −
0.15(1 + c)c)`, `g = exp((0.78 + 0.12c)c)`), and `SeverityCurve`
(`LEV(xM)/LEV(M)` for any `Severity`). Parity
(`validation/scripts/mpmath_mbbefd.py`): `G` and the mean by 30-digit
quadrature of the survival function for c = 1.5–5 and one curve per case,
56 values at 1e-12. Python `Mbbefd`, `severity_exposure_curve`; R
`mbbefd()`, `swiss_re_curve()`, `exposure_curve()`,
`exposure_layer_share()`, `severity_exposure_curve()`.

Done next in `exposure`: every curve is a destruction-rate distribution
(survival `G'(x)/G'(0)`): `ExposureCurve::rate_quantile(u)` (closed forms
for the four MBBEFD cases; `min(q(u), M)/M` for a severity curve) and
`mean_rate()` (`1/G'(0)`), checked by sampling: the draws' `E[min(D, x)]`
over the mean rate is `G(x)`. `TabulatedCurve` takes a published table
(Salzmann, Ludwig, ISO PSOLD, a reinsurer's curves) from `(0, 0)` to
`(1, 1)`, interpolated linearly; it must be concave, which makes its
destruction rate discrete, and its mean rate is the first chord's
(`x₁/G(x₁)`), so tables need fine first points. Riebesell's scale is a
Pareto's `SeverityCurve`. Python `TabulatedCurve`, `rate_quantile` on
both; R `tabulated_curve()`, `rate_quantile()`. These feed the risk-profile
simulator (`aggregate.md`), one curve per sum-insured band.

Done in `risk_load`: `price(losses, rule, assets)` on any `Empirical`
and `price_portfolio(pd, rule, assets)` on a `PredictiveDistribution`,
with `PremiumRule::{Distortion, CostOfCapital}`. This is where simulated
results meet pricing: a ceded result from `Tower::apply` or
`apply_aggregate`, a reserve bootstrap, or a blended model's draws.
Decisions:

- **Assets are a distortion measure** `a = ρ(X)` (say `TVaR_0.99`); the
  capital is `a − P`, so the premium funds part of the assets.
- **Cost of capital** `r` sets the margin to `r (a − P)`, so
  `P = (E[X] + r a) / (1 + r)`.
- **Natural allocation** (Mildenhall and Major, *Pricing Insurance Risk*,
  2022): premium and assets are each allocated by co-measure
  (`PredictiveDistribution::allocate`), so component prices add up to the
  portfolio's and every component earns `r` on its allocated capital. A
  hedge gets a negative margin. Standalone prices are reported beside the
  allocated ones, and their difference is the diversification credit.
- **Funding check**: a distortion premium above the assets is an error
  for a cover or portfolio, not for a component's standalone price.
- `Sampled` still does not implement `Severity` (`distributions.md`):
  ILFs and layer costs from draws stay `Empirical::mean_of`, an explicit
  estimate, while the pricing here needs only measures that are exact on
  the draws.

Bindings: Python `price`, `price_portfolio`, `Price`, `PortfolioPrice`;
R `risk_loaded_price()` and `price_portfolio()`.

`Layer`, `Tower` and `TowerGrids` (contract terms and their grid
results) are in the user-facing `reinsurance` namespace that
`docs/architecture.md` gives them: Python `prospicio.reinsurance`, and
R's `reinsurance.R` with its own reference section. In Rust they stay in
`prospicio_aggregate::reinsurance` and `prospicio_aggregate::grid_reinsurance`, next
to the compound and simulation code they are applied with; a separate
crate would gain nothing while nothing else depends on them.

Then Python and R bindings. Each row is one small PR, in roughly this
order.

## Validation

- **Closed forms** (cdf, quantile, layer moments) against 30-digit mpmath
  integration of the survival function, as for `Lognormal`.
- **R parity:** reference CSVs generated by a checked-in script that runs
  the Pareto and LocalPareto packages (installed from source, version
  recorded per row), for layer moments, implied alphas, extrapolation
  and maximum likelihood estimates.
- **Tower matching** is tested by what it promises: the fitted model
  reproduces every input layer loss and frequency to `1e-10`, and is
  rejected with a clear reason when the RRoLs do not strictly decrease.
  The pieces themselves are compared with the paper's Example 4 (the six
  printed pieces it shares with the current package) and with R 2.4.5
  under the same selection rule, since a tower has many matching models.
- **Fits** recover known parameters from simulated data, with and
  without reporting thresholds and censoring.

## Open questions

1. ~~Riegel (2018)~~: received; the algorithm above is taken from it.
   The 2025 local Pareto preprint is not available, so the log-affine
   formulas are derived from the definition (above).
2. ~~Namespace~~: decided, a new `prospicio-pricing` crate (above);
   `prospicio-aggregate` keeps aggregates and the collective model.
3. ~~Linear programming~~: decided, `microlp` rather than our own solver.

## References

- Fackler, M. (2025) Reinventing Pareto: fits for both small and large losses. British Actuarial Journal 30: e33.
- Philbrick, S. W. (1985) A practical guide to the single parameter Pareto distribution. PCAS LXXII: 44–84.
- Riegel, U. (2008) Generalizations of common ILF models. Blätter der DGVFM 29: 45–71.
- Riegel, U. (2018) Matching tower information with piecewise Pareto. European Actuarial Journal 8(2): 437–460.
- Riegel, U. (2025) The local Pareto distribution. Preprint.
- Schmutz, M., and Doerr, R. R. (1998) Das Pareto-Modell in der Sach-Rückversicherung. Swiss Re.
