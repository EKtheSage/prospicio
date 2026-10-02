# Design note: Pareto-family severities for reinsurance pricing

Status: **Proposed** · v0.4 · Depends on: `distributions.md` (`Severity`, `Counting`, `Grid`), `aggregate.md` (towers), `risk.md` (`evt::Gpd`) · Lanes: Probability (distributions), Aggregate (pricing)

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

- A necessary condition is that the rate on line (expected loss per unit
  of cover, which is the average excess frequency across the layer)
  falls as the attachment rises. If it does not, no severity can match
  the tower, and the inputs need reviewing before any model is fitted.
- One Pareto piece per layer, chaining alphas up the tower, is the
  obvious approach, but it can fail. The paper's four-layer example
  (500 xs 1000, 1500, 2000, 2500 with expected losses 100, 90, 50, 40,
  rates on line 0.20, 0.18, 0.10, 0.08) has no solution with one piece
  per layer.
- Two pieces per layer, with a free intermediate threshold, always work
  when the tower is consistent. The R package's fit of that example (with
  an unlimited top layer at 3000 costing 100) uses eight pieces and
  reproduces all five expected losses.

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

## Plan

### `act-prob` (Probability lane)

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

| Item | Notes |
|---|---|
| `CollectiveModel<N, X>` | Expected layer loss, layer variance (`E[N] Var[Y] + Var[N] E[Y]²`), excess frequency; simulation and Panjer/FFT through existing code. |
| Rating helpers | Extrapolation; implied alpha from two layers, a frequency and a layer, or two frequencies. |
| Tower matching | Riegel (2018), two pieces per layer; reports why a tower is inconsistent when it is. |
| Reference and PML fits | Partial references and PML curves; overlapping reference layers need a linear program and come later. |

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
  rejected with a clear reason when the rates on line do not fall.
  Exact pieces are compared with the R package only where our selection
  rule is the same as the paper's, since a tower has many matching
  models.
- **Fits** recover known parameters from simulated data, with and
  without reporting thresholds and censoring.

## Open questions

1. Copies of Riegel (2018) and the 2025 local Pareto preprint, to
   implement the matching rules and the log-affine formulas from the
   source rather than from package behaviour.
2. Namespace: the pricing helpers and tower matching live in
   `act-aggregate` for now; `docs/architecture.md` plans a `pricing`
   namespace (ILF, exposure curves, MBBEFD) that they may move to.
3. Overlapping reference layers need a small linear-programming solver;
   build one in `act-math` or leave the case out.

## References

- Fackler, M. (2025) Reinventing Pareto: fits for both small and large losses. British Actuarial Journal 30: e33.
- Philbrick, S. W. (1985) A practical guide to the single parameter Pareto distribution. PCAS LXXII: 44–84.
- Riegel, U. (2008) Generalizations of common ILF models. Blätter der DGVFM 29: 45–71.
- Riegel, U. (2018) Matching tower information with piecewise Pareto. European Actuarial Journal 8(2): 437–460.
- Riegel, U. (2025) The local Pareto distribution. Preprint.
- Schmutz, M., and Doerr, R. R. (1998) Das Pareto-Modell in der Sach-Rückversicherung. Swiss Re.
