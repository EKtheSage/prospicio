# Design note: aggregate loss and reinsurance

Status: **In progress** · v0.3 · Depends on: `distributions.md` (`Grid`, `Counting`, `Severity`), `predictive-distribution.md` · Lane: Aggregate

## Goal

The distribution of a period's total loss `S = X_1 + … + X_N` from a claim
count `N` and a severity `X`, three ways, and the split of losses between
an insurer and its reinsurers. All numerics that are not aggregation itself
(distributions, grids, risk measures) come from `act-prob`.

| Method | Input | Output | Exact for | Use when |
|---|---|---|---|---|
| Panjer | `Counting` in the `(a, b, 0)` class, severity `Grid` | aggregate `Grid` | the discretized problem | moderate claim counts, `O(n · m)` |
| FFT | any count with a pgf, severity `Grid` | aggregate `Grid` | the discretized problem, up to aliasing | large grids or claim counts, `O(n log n)` |
| Monte Carlo | any `Counting`, any severity | `PredictiveDistribution` | nothing (sampling error) | per-event losses, reinsurance terms, dependence |

## What exists

- `act_aggregate::panjer(&frequency, &severity_grid, points)` and
  `act_aggregate::fft(&frequency, &severity_grid, points)` return the
  aggregate `Grid` and a `CompoundReport`.
- `act_aggregate::simulate_events(&frequency, &severity, n_sims, seed)`
  returns an `EventSet`: each simulated year's individual losses, with
  `totals()` as a `PredictiveDistribution`.
- `act_aggregate::{Layer, Tower}`: per-occurrence excess-of-loss layers
  with share, annual aggregate deductible and limit, and reinstatements;
  `Tower::apply(&events)` returns gross, ceded per layer and net as one
  joint `PredictiveDistribution`.
- Python (`actuarialrs.aggregate`) and R (`compound_distribution`,
  `simulate_events`, `xol_layer`, `reinsurance_tower`) bindings for all of
  the above.

## Decisions

- **Error is reported, never hidden.** A compound grid lumps its mass above
  the last point onto that point (as `Grid` discretization does) and
  `CompoundReport` records `tail_mass = P(S > (n - 1)h)`, the untruncated
  mean `E[N] E[X]` on the severity grid, the grid mean and `mean_error()`.
  The severity grid's own `DiscretizationReport` stays with the caller.
- **Panjer underflow is an error.** For a Poisson mean above about 700 with
  no mass at zero severity, `P(S = 0)` underflows to 0 and the recursion
  would return all zeros. `panjer` fails and points to FFT instead of
  scaling tricks.
- **The aggregate grid uses the severity grid's step.**
- **FFT aliasing is measured, not estimated.** A finite FFT is circular, so
  mass beyond the buffer wraps onto the start. `fft` runs at a power-of-two
  buffer `L ≥ 2 · max(points, severity points)` and at `2L`, returns the `2L`
  result, and reports the largest difference over the returned points as
  `aliasing_error`. When it is not negligible, wrapped mass sits inside the
  grid and `tail_mass` understates the truth (a test pins this), so callers
  must check `aliasing_error`, not `tail_mass`, first.
- **FFT applies the claim-count pgf at complex points** via
  `Counting::pgf_complex`, written with `(re, im)` pairs so `act-prob` needs
  no complex-number dependency; `rustfft` (pure Rust) does the transforms.

- **Monte Carlo keeps events.** Reinsurance terms apply per loss, so the
  simulation keeps every year's individual losses (stored compactly, one
  offset per year) rather than only totals.
- **Monte Carlo stream order** (scheme `chacha20/sim-index/v1`): year `i`
  draws only from `StreamRng::new(seed, i)`, first the claim count, then
  each severity in order, all by inverse transform. Results are identical
  for any thread count, and any year replays alone (both tested).

- **Reinsurance terms are plain data** (`Layer` has public fields and
  builder methods), so towers can be stored and replayed. For one year,
  `ceded = share × min(max(Σ min(max(x - a, 0), l) - AAD, 0), AAL)`;
  `reinstatements(n)` sets `AAL = l × (n + 1)`.
- **A tower's result is one joint distribution** with dimensions
  `["kind", "layer"]`: `(gross, ground_up)`, `(ceded, <name>)` per layer
  and `(net, retained)`. `aggregate(&["kind"])` gives gross, total ceded and
  net per year, and `net = gross − Σ ceded` holds in every year (tested).
- **Not yet:** inuring order (every layer sees gross losses), reinstatement
  premiums, quota share and surplus, and stop-loss on the annual total.

## Validation

`validation/tests/aggregate.rs` checks Panjer and FFT against a brute-force
compound sum, `Σ_n P(N = n) f^{*n}`, with numpy convolutions and SciPy
pmfs (`validation/scripts/compound_convolution.py`): 80 points for a
Poisson and a negative binomial at `1e-14`. FFT is checked against the
same file and passes at the same tolerance. FFT also agrees with Panjer to
`1e-13` and handles a Poisson mean of 2,000 that makes Panjer underflow. Unit
tests check `E[S] = E[N] E[X]`, the compound variance
`E[N] Var[X] + Var[N] E[X]^2`, and `S = N` for a unit severity. Monte
Carlo totals are tested against the exact FFT result for a discrete
severity by a Kolmogorov–Smirnov statistic (200,000 years, below the 0.1%
critical value) under a fixed seed. Reinsurance is checked on hand-worked
layers and annual terms, gross = ceded + net in every simulated year, and
the simulated mean ceded loss against the exact `E[N] · Severity::layer`
(within four standard errors).

## Next

1. Inuring order, reinstatement premiums, quota share and surplus,
   aggregate stop-loss.
