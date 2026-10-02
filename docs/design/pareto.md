# Design note: Pareto-family severities for reinsurance pricing

Status: **Accepted** · v0.4 · Depends on: `distributions.md` (`Severity`, `Counting`, `Grid`), `aggregate.md` (towers), `risk.md` (`evt::Gpd`) · Lanes: Probability (`act-prob`), Aggregate (`act-aggregate`, `act-pricing`)

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
`norm_cdf`, which `act-math` already has. They are checked against
mpmath integration and against the LocalPareto package. The general
local Pareto (any `α(x)`) is converted to a piecewise Pareto by our own
adaptive scheme with a stated bound on the relative error of `S`.

## Plan

### `act-prob` (Probability lane)

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

### `act-aggregate` (Aggregate lane)

`act-aggregate` stays about aggregate distributions. The collective
model is the frequency–severity input to an aggregate, so it lives here.

Done: `CollectiveModel<N, X>` (with `Box<dyn Severity>` and
`Box<dyn Counting>` usable through blanket impls in `act-prob`), and
`Distribution::survival` so excess frequencies keep their precision far
in the tail.

| Item | Notes |
|---|---|
| `CollectiveModel<N, X>` | Expected layer loss, layer variance (`E[N] Var[Y] + Var[N] E[Y]²`), excess frequency; simulation and Panjer/FFT through existing code. |

### `act-pricing` (Aggregate lane, new crate)

The `pricing` namespace planned in `docs/architecture.md`. Most of it
serves primary and reinsurance pricing alike; only `tower` is
reinsurance-specific.

| Module | Item | Used by |
|---|---|---|
| `layer` | Increased limit factors (`LEV(limit)/LEV(basic)`), deductible credits (`1 − LEV(d)/E[X]`), Pareto extrapolation, implied alpha from two layers, a frequency and a layer, or two frequencies. Later: MBBEFD exposure curves. | Primary (ILF tables, deductibles, large-loss loads) and reinsurance (rating upper layers) |
| `tower` | Tower matching (Riegel 2018, above, both selection rules); reference fits; PML-curve fits. | Reinsurance |

Overlapping reference layers need a small linear program. It uses the
pure-Rust simplex crate `microlp` (Apache-2.0), as a dependency of
`act-pricing` only; no C or C++ build, so WASM, CRAN and Windows builds
are unaffected.

`Layer` and `Tower` (contract terms) still live in `act-aggregate`;
`docs/architecture.md` gives them their own `reinsurance` namespace, and
moving them is a separate, later PR.

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
2. ~~Namespace~~: decided, a new `act-pricing` crate (above);
   `act-aggregate` keeps aggregates and the collective model.
3. ~~Linear programming~~: decided, `microlp` rather than our own solver.

## References

- Fackler, M. (2025) Reinventing Pareto: fits for both small and large losses. British Actuarial Journal 30: e33.
- Philbrick, S. W. (1985) A practical guide to the single parameter Pareto distribution. PCAS LXXII: 44–84.
- Riegel, U. (2008) Generalizations of common ILF models. Blätter der DGVFM 29: 45–71.
- Riegel, U. (2018) Matching tower information with piecewise Pareto. European Actuarial Journal 8(2): 437–460.
- Riegel, U. (2025) The local Pareto distribution. Preprint.
- Schmutz, M., and Doerr, R. R. (1998) Das Pareto-Modell in der Sach-Rückversicherung. Swiss Re.
