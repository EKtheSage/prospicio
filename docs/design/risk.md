# Design note: risk measures, dependence and allocation

Status: **In progress** · v0.4 · Depends on: `distributions.md`, `predictive-distribution.md` · Lane: Probability

## Goal

Measure the risk of any result (reserve, aggregate loss, tower net),
combine results under a dependence structure, and split a portfolio's risk
back to its parts. All of it works on the representations `act-prob`
already has: sampled draws, grids, and the joint `PredictiveDistribution`.

| Piece | Module | Works on |
|---|---|---|
| VaR, TVaR | `risk` | sorted draws (done) |
| Distortion risk measures | `distortion` | sorted draws, discrete distributions, grids |
| Allocation (co-measures) | `PredictiveDistribution::allocate` | `PredictiveDistribution` |
| Copulas | `copula` | uniforms per simulation, then marginals |
| Iman-Conover | `copula` | reorders existing draws |
| EVT tails | later | — |

## What exists

- `act_prob::risk::{var_sorted, tvar_sorted}`.
- `act_prob::Distortion`: `Tvar(p)`, `Wang(λ)`, `ProportionalHazard(ρ)`,
  `DualPower(β)`, with `g`, `weights(n)`, `apply_sorted`,
  `apply_discrete`; `Empirical::distortion` and `Grid::distortion`.
- `PredictiveDistribution::allocate(&Distortion)`: co-measure allocation
  of the total's risk measure to the components (CoTVaR for `Tvar`).

## Decisions

- **A distortion is a closed enum** of concave distortions of the
  survival function, so every measure offered is coherent. Each variant
  has a parameter value that gives the mean. Custom distortions wait for a
  use.
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
- **Copulas generate uniforms; marginals stay where they are.** A copula
  draws one vector of uniforms per simulation from
  `StreamRng::new(seed, sim)` (the `chacha20/sim-index/v1` scheme), and
  marginals are applied by inverse transform, so any simulation replays
  alone and results do not depend on thread count.
- **Iman-Conover reorders draws**: it imposes a target rank correlation
  on existing marginals (for example, reserve and premium-risk results
  simulated separately) by permuting them, so every marginal keeps its
  exact draws.

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

Unit tests check that `Tvar(p)` matches `tvar_sorted` for 101 levels, that
weights are a non-decreasing probability vector, and coherence: translation
and scale equivariance, bounds between the mean and the maximum, and
monotonicity in each parameter.

Allocation tests check, on simulated dependent lines, that contributions
sum to the measure of the total for every distortion, that the mean
allocates to component means, and that CoTVaR equals the conditional tail
mean computed directly. Two exact cases: comonotonic lines (`X_2 = 2 X_1`)
each receive their own risk measure, and two simulations tied on the
total but split differently give the same allocation in either order.

## Next

1. Gaussian and t copulas; Iman-Conover.
2. Archimedean copulas (Clayton, Gumbel, Frank, Joe).
3. Python and R bindings.
4. EVT tails (GPD over a threshold) for extrapolating past the draws.
