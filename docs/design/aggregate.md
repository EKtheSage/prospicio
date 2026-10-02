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
  quota shares (`Layer::quota_share`) and aggregate stop-losses
  (`Layer::stop_loss`); `Tower::inuring` stages layers so later ones see
  losses net of earlier ones. `Tower::apply(&events)` returns gross, ceded
  per layer and net as one joint `PredictiveDistribution`.
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
- **Quota share and stop-loss are layers.** A quota share with cession
  `c` is unlimited cover from 0 with `share = c`; a stop-loss `l` xs `r`
  is unlimited cover from 0 with `AAD = r` and `AAL = l`. One formula
  covers all three contract types, and a stop-loss in a later stage covers
  the annual total net of earlier stages.
- **Inuring order is by stage.** `Tower::inuring(stages)` applies stages
  in order; layers within a stage see the same losses, and each later
  stage sees every event net of all earlier stages. `Tower::new(layers)`
  is one stage.
- **Annual terms are used up in event order.** Passing a net loss per event
  to the next stage needs each event's share of a layer with annual terms.
  `Layer::ceded_by_event` takes events as chronological: the AAD absorbs
  the first recoveries and the AAL stops the last ones, and event `k`
  cedes the increase in annual ceded loss it causes. The split sums to the
  annual ceded loss, which does not depend on order; only later stages do.
  Simulated events are in simulation order, which stands in for time
  until events carry dates.
- **Not yet:** reinstatement premiums; surplus treaties, which need sums
  insured per risk that events do not carry.

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
(within four standard errors). Inuring is checked on a hand-worked
two-stage tower and against the identity that a cession `c` inuring to
`l` xs `a` equals `(1 - c)` of `l / (1 - c)` xs `a / (1 - c)` on gross, in
every simulated year.

## Next

1. Reinstatement premiums.
2. Python and R bindings for quota share, stop-loss and inuring towers.
