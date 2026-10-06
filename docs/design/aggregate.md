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
- `Tower::apply_aggregate(&pd)`: the tower on any
  `PredictiveDistribution`, each simulation's total as one aggregate loss
  (an adverse development cover or loss portfolio transfer on a reserve
  bootstrap, a stop-loss or quota share on modelled premium risk). Same
  components as `apply`.
- `act_aggregate::CollectiveModel<N, X>`: a claim count and a severity
  with closed-form layer mean, layer variance
  (`E[N] Var[Y] + Var[N] E[Y]^2`) and excess frequency, the treaty
  pricing model of `pareto.md`; `simulate` reuses `simulate_events`.
  Python `actuarialrs.pricing.CollectiveModel`, R `collective_model()`.
- Python (`actuarialrs.aggregate`) and R (`compound_distribution`,
  `simulate_events`, `xol_layer`, `quota_share`, `aggregate_stop_loss`,
  `reinsurance_tower`, `inuring_tower`) bindings for all of the above.

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
  Simulated events are in simulation order, which is their time order;
  events may carry times too (below).
- **Reinstatement premiums are pro rata as to amount.** With layer loss
  `L` at 100% after annual terms, `paid_reinstatements(premium, rates)`
  charges `premium × Σ_k rates[k] × min(max(L - k·l, 0), l) / l`, where
  `premium` is the upfront premium for the placed share (so the share
  does not scale it again). The tower reports them as `(reinstatement_premium, <name>)` components, and
  `net` stays a loss: premiums are not netted against it.
- **Pro rata as to time.** `Layer::pro_rata_as_to_time()` (after
  `paid_reinstatements`) charges the limit each event uses up at `1 − t`,
  where `t` is its time as the fraction of the year elapsed: with `A_e`
  the layer loss after annual terms up to event `e`, the premium is
  `premium / l × Σ_e (1 − t_e) Σ_k rates[k] |[A_{e−1}, A_e] ∩ [k l, (k+1) l]|`.
  Events carry times through `EventSet::with_times` (one per loss, in
  `[0, 1]`, non-decreasing within a year: a catastrophe model's dated
  events) or `EventSet::with_uniform_times` (sorted uniform draws from
  stream `2^63 + i` of the set's seed, in the losses' drawn order; exact
  for a year's i.i.d. losses, and the losses are unchanged). Times are a
  fraction of the year rather than dates, so leap years and the
  contract's inception are the caller's. `Tower::apply` refuses such a
  layer on events without times; `apply_aggregate` and `on_grid` refuse
  it outright. Tested: the closed form on hand-worked years (across two
  limits, with an annual deductible), and one exhausting loss a year at a
  uniform time averages half the amount-only premium within four
  standard errors. Python `Layer(..., pro_rata_time=True)`,
  `EventSet.with_uniform_times()`, `EventSet.from_years(..., times=)`; R
  `xol_layer(pro_rata_time = TRUE)`, `with_uniform_times()`,
  `events_from_years(times =)`, `event_times()`.
- **Towers are data.** `Tower::to_json` writes a programme as a versioned
  document (`"format": "risk_rs.tower"`, version 1): its stages in inuring
  order, each a list of layers with every term (basis, and for a surplus
  its retention and lines; limit, attachment, share, annual deductible and
  limit, premium, reinstatement rates, pro rata as to time). Non-finite
  numbers are written `"inf"`, as in distribution documents.
  `Tower::from_json` rebuilds each layer through the same builders
  (`xol`, `surplus`, `share`, `aggregate_deductible`,
  `paid_reinstatements`, `aggregate_limit`, `pro_rata_as_to_time`), so
  an impossible term is refused rather than loaded. Tested: a three-stage
  programme with every kind of term round-trips to an equal tower and the
  same document, and cedes the same on simulated events with sums insured
  and times. Python `Tower.to_json` / `Tower.from_json` (and pickle), R
  `tower_to_json()` / `tower_from_json()`.
- **Towers also run exactly on the grid.** `Tower::on_grid(frequency,
  severity, points)` returns `TowerGrids`: gross, each layer's ceded loss
  and, where defined, net, as grids by FFT with no sampling error. A
  per-occurrence layer maps the severity grid through its recovery
  function (`Grid::map`). Compounding that grid with the same claim count
  gives the annual recovery, and annual terms map that. A share `c`
  rescales the step to `c h`, which is exact.
- **Off-point boundaries keep the mean.** `Grid::map` splits a value that
  falls between two points between them, so its mean is kept, the same
  rule as local moment matching. `TowerGrids::on_points` reports whether
  any split happened. With boundaries on multiples of the step, the grids
  are exact for the discretized problem.
- **Grids are marginal; net only when it is one compound total.** Net is
  returned when no layer has annual terms, where net is a function of each
  loss, or when the last stage is a single aggregate cover (attachment 0,
  unlimited per occurrence), where net is a function of the annual total
  net of earlier stages. Otherwise net depends jointly on several totals
  and is `None`. Joint results across layers come from `Tower::apply` on
  simulated events.
- **Annual terms may not inure on the grid.** A layer with annual terms
  in an earlier stage takes a share of each event that depends on event
  order, which a compound distribution does not have. `on_grid` rejects
  such a tower and points to Monte Carlo.
- **Sums insured ride on the events; a risk profile fills them (decided
  2026-10-06).** A surplus treaty cedes a share of each loss set by the
  sum insured (SI) of the risk it hit, `min(max(SI − R, 0), kR) / SI` for
  retention line `R` and `k` lines, so each event carries its SI. Options
  weighed: (A) simulate from a risk profile, (B) an optional SI per
  event that users fill, (C) expected values per band only. The user
  chose A built on B: B first (`EventSet::with_sums_insured`,
  `EventSet::from_years` for one's own years, Python
  `EventSet.from_years(years, sums_insured)`, R `events_from_years()`),
  then a profile simulator that fills it. C cannot sit ahead of a
  per-risk XL in a tower, which is the usual programme; it comes from the
  same profile as a check.
- **Surplus is a layer basis.** `Layer::surplus(name, R, k)` is a layer
  whose per-event amount is the ceded share of the loss (`Basis::Surplus`);
  an event limit, share and annual terms apply to it as to any layer, so it
  inures to a per-risk XL in a later stage with no other change. Every
  stage sees each risk's original SI. `Tower::apply` refuses a tower with a
  surplus on events without SIs; `apply_aggregate` and `on_grid` refuse
  one outright (no risks). Python `Layer.surplus`, R `surplus_treaty()`.
- **Risk profiles.** `act_pricing::profile::RiskProfile`: bands of sum
  insured, each with a number of risks, an expected loss (given, or
  premium × loss ratio, the user's choice per profile) and its own
  exposure curve (MBBEFD or tabulated, `BandCurve`). A band expects
  `EL / (SI × mean rate)` losses a year; a simulated year draws a Poisson
  count with the total mean, each loss in a band with probability
  proportional to its expected count, as the band's SI times a destruction
  rate from its curve (`ExposureCurve::rate_quantile`), and carries that
  SI. The exposure-rated expectations are option C, kept as the check:
  `expected_surplus_loss(R, k) = Σ cession_b EL_b` and
  `expected_layer_loss(l, a, surplus)`, where a per-risk XL net of a
  surplus sees each risk at SI `(1 − c) SI` on the same curve. Tested:
  50,000–100,000 simulated years of a three-band profile (two MBBEFD, one
  tabulated) give the surplus and the per-risk XL it inures to within four
  standard errors of these. It lives in act-pricing (exposure curves),
  which now depends on act-aggregate (`EventSet`). Python `RiskProfile`,
  R `risk_profile()`, `profile_simulate()`, `profile_layer_loss()`,
  `profile_surplus_loss()`.
- **Sums insured spread within a band.** Bounds are optional per band
  (`Band::with_bounds(lower, upper)`; Python `lower=`/`upper=` with `None`,
  R `lower`/`upper` with `NA`). A band with bounds has its risks' sums
  insured uniform by count between them, with the same loss frequency per
  risk, so each simulated loss draws its SI uniformly and the band's mean
  SI is `(L + U) / 2`; that replaces the band's SI in the expected count.
  The exposure-rated expectations average over the band weighted by sum
  insured, `∫ s f(s) ds / ∫ s ds` (a risk's expected loss is proportional
  to its SI), by Gauss–Legendre on 256 pieces. With bounds, the band's
  given SI is not used: a profile whose total SI over its risks differs
  from `(L + U) / 2` would need a tilted spread to match both, which is a
  later option. Tested: a band from 1m to 5m against a 2m surplus cedes
  3/8 (closed form), where its 3m mean risk cedes 1/3, and simulation
  agrees within four standard errors.

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

Grid towers are checked against 200,000 simulated years of the same
discrete severity by a Kolmogorov–Smirnov statistic at the 0.1% level.
The check covers gross, every ceded grid and net, for two per-occurrence
layers (Poisson and negative binomial counts), a layer with a deductible
and a reinstatement, and an excess-of-loss layer inuring to a stop-loss.
Expected reinstatement premiums are within four standard errors of the
simulated mean. Layer means with boundaries between points are exact, and
gross = Σ ceded + net in mean.

`CollectiveModel` is checked against the R package Pareto's `PPP_Model`
and `PGP_Model` (`validation/scripts/r_collective.R`): layer means at
`1e-12`, layer variances at `1e-8` (the package's second moments cancel on
high layers), and excess frequencies, for binomial, Poisson and negative
binomial counts. A unit test checks the layer mean and variance against
200,000 simulated years.

## Next

1. Seasonality: event times from a density over the year rather than
   uniform (a hurricane season).
2. Loss corridors (a retained band of the layer's annual loss), and other
   contract features in `architecture.md`'s reinsurance scope.
3. A spread within a band that matches both its bounds and its total sum
   insured (a tilted, not uniform, density), if profiles call for it.
