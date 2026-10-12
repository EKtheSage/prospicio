# Design note: aggregate loss and reinsurance

Status: **In progress** · v0.3 · Depends on: `distributions.md` (`Grid`, `Counting`, `Severity`), `predictive-distribution.md` · Lane: Aggregate

## Goal

The distribution of a period's total loss `S = X_1 + … + X_N` from a claim
count `N` and a severity `X`, three ways, and the split of losses between
an insurer and its reinsurers. All numerics that are not aggregation itself
(distributions, grids, risk measures) come from `prospicio-prob`.

| Method | Input | Output | Exact for | Use when |
|---|---|---|---|---|
| Panjer | `Counting` in the `(a, b, 0)` class, severity `Grid` | aggregate `Grid` | the discretized problem | moderate claim counts, `O(n · m)` |
| FFT | any count with a pgf, severity `Grid` | aggregate `Grid` | the discretized problem, up to aliasing | large grids or claim counts, `O(n log n)` |
| Monte Carlo | any `Counting`, any severity | `PredictiveDistribution` | nothing (sampling error) | per-event losses, reinsurance terms, dependence |

## What exists

- `prospicio_aggregate::panjer(&frequency, &severity_grid, points)` and
  `prospicio_aggregate::fft(&frequency, &severity_grid, points)` return the
  aggregate `Grid` and a `CompoundReport`.
- `prospicio_aggregate::simulate_events(&frequency, &severity, n_sims, seed)`
  returns an `EventSet`: each simulated year's individual losses, with
  `totals()` as a `PredictiveDistribution`.
- `prospicio_aggregate::{Layer, Tower}`: per-occurrence excess-of-loss layers
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
- `prospicio_aggregate::CollectiveModel<N, X>`: a claim count and a severity
  with closed-form layer mean, layer variance
  (`E[N] Var[Y] + Var[N] E[Y]^2`) and excess frequency, the treaty
  pricing model of `pareto.md`; `simulate` reuses `simulate_events`.
  Python `prospicio.pricing.CollectiveModel`, R `collective_model()`.
- `prospicio_aggregate::{recommend_grid, fft_auto, round_bucket}`: the FFT
  grid (bucket and points) chosen from the frequency and severity, as
  `aggregate` 1.0.1 does when no bucket is given ("Grid sizing" below).
  Python `prospicio.aggregate.{recommend_grid, fft_auto, round_bucket}`,
  R `recommend_grid()`, `compound_auto()`, `round_bucket()`.
- `prospicio_aggregate::pml::{occurrence_exceedance, occurrence_pml}`:
  the occurrence exceedance (OEP) curve `1 − G_N(F(x))` for any count and
  severity, and its inverse, the occurrence PML at an annual probability
  with the severity's exceedance and the occurrence and annual return
  periods ("Occurrence PML" below). `EventSet::maxima()` is each simulated
  year's largest loss, the simulated OEP curve. Python
  `prospicio.aggregate.{occurrence_exceedance, occurrence_pml}` and
  `EventSet.maxima()`, R `occurrence_exceedance()`, `occurrence_pml()`,
  `annual_maxima()`.
- `prospicio_aggregate::moments::{compound_moments, CompoundMoments,
  Approximation}`: the compound mean, variance and third cumulant (cv,
  skewness) from the count's cumulants and the severity's first three
  moments, and the normal, gamma, lognormal, shifted gamma and shifted
  lognormal fitted to them, with `cdf`, `quantile` and `mean` ("Compound
  moments" below). Python `prospicio.aggregate.compound_moments`, R
  `compound_moments()`, `moment_approximation()`, `approximation_cdf()`,
  `approximation_quantile()`.
- `prospicio_aggregate::layer_statistics::{occurrence_layer_statistics,
  LayerStatistics}`: a per-occurrence layer's chances of attaching,
  detaching and a loss in the year, its loss on line, and the moments of
  the excess count, the layer severity given a loss and the annual ceded
  loss; `TowerGrids::price` prices the gross, each layer's cession and the
  net on their own under a distortion (`TowerPrice`, `ViewPrice`) ("Layer
  statistics and prices by view" below). Python
  `prospicio.reinsurance.occurrence_layer_statistics` and
  `TowerGrids.price`, R `occurrence_layer_statistics()`, `tower_price()`.
- Python (`prospicio.aggregate`) and R (`compound_distribution`,
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
  `Counting::pgf_complex`, written with `(re, im)` pairs so `prospicio-prob` needs
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
- **Tilted spreads within a band (decided 2026-10-08).**
  `Band::with_tilted_bounds(lower, upper)` spreads a band's sums insured
  on the bounds with density `∝ exp(θ s)` and keeps the band's given SI
  as the mean, so the spread matches both the bounds and the band's total
  sum insured (`risks × SI`). Among densities on `[L, U]` with a given
  mean, this one has the most entropy, so it is the most even spread that
  matches, and `θ = 0` (an SI at the midpoint) is the uniform spread.
  `θ` solves `1/(1 − e^{−t}) − 1/t = (SI − L)/(U − L)` with
  `t = θ (U − L)` by bisection; the SI must lie strictly between the
  bounds. Simulated losses draw their SI by inverse transform, written
  with `expm1`/`ln_1p` from the heavier end so no exponential overflows.
  The exposure-rated expectations average `E[S f(S)] / E[S]` over the
  spread as integrals over probability, `∫₀¹ g(Q(p)) dp`, by
  Gauss–Legendre on 256 pieces, which stays accurate for a steep tilt
  (and is the same integral as before for a uniform spread). The claim
  count uses the spread's mean SI, the given one. Tested: the solved mean
  to `1e-12`; the midpoint tilt equals the uniform spread; a 1m–5m band
  with mean 2m against Simpson's rule in SI (to `1e-7`: the cession's
  kink at the retention falls inside a piece over probability); simulated
  SIs by mean and Kolmogorov–Smirnov at the 0.1% level; a surplus and
  the per-risk XL it inures to within four standard errors of exposure
  rating; a nearly degenerate tilt against the band's single risk.
  Python `RiskProfile(..., spread="tilted")`, R
  `risk_profile(spread = "tilted")`.
- **Seasonal event times (decided 2026-10-08).** `EventSet::with_seasonal_times(weights)`
  dates losses by a piecewise-constant density over the year: `m =
  weights.len()` equal periods (12 for months, 52 for weeks) starting at
  the contract's inception, period `k` with probability `w_k / Σ w`, and
  uniform within it; a zero weight is a period with no losses (outside a
  hurricane season). Year `i` takes the same sorted uniform draws as
  `with_uniform_times` and maps each `u` through the season's quantile,
  `t = (k + (u − C_k) / (C_{k+1} − C_k)) / m` with `C` the cumulative
  weights. The quantile is increasing, so the times stay sorted, the
  i.i.d. argument for uniform times carries over, and equal weights give
  the uniform times to rounding (tested to `1e-15`). Periods are equal
  fractions of the contract year, so a 1 July contract lists July first,
  and months of unequal length are the caller's (weight by days if it
  matters). A general density on `[0, 1]` is a later option; nothing in
  `prospicio-prob` lives there yet. Tested: pooled times of 50,000 years
  against the season's cdf by Kolmogorov–Smirnov at the 0.1% level, no
  time in a zero-weight period, and one exhausting loss a year dated in
  the second half only averages a quarter of the amount-only premium
  (`2 E[1 − t] = 0.5`), with variance `4 · 0.5² / 12`, within four
  standard errors. Python `EventSet.with_seasonal_times(weights)`, R
  `with_seasonal_times()`.
- **Loss corridors (decided 2026-10-08).** `Layer::loss_corridor(lower,
  upper, retained)`: of the annual layer loss at 100% after the annual
  deductible, the cedant keeps `retained` of the part between `lower` and
  `upper`, and the annual limit caps what is left:
  `ceded = share × min(D − retained × min(max(D − lower, 0), upper − lower), AAL)`
  with `D = max(R − AAD, 0)`. The corridor sits between the deductible
  and the limit because both keep their meaning: the AAD is what the
  cedant keeps first and the AAL the most the reinsurer pays, so a
  corridor does not shrink the reinsurer's maximum. Reinstatement
  premiums follow the loss after the corridor: limit kept in the corridor
  is not reinstated or charged for. Any layer takes one, including a quota
  share (the usual case) and a stop-loss. Bounds are amounts at 100% of
  the layer, like the AAD and AAL; a corridor quoted as loss ratios `lr`
  on the reinsurer's premium `P` for a share `s` is `lr × P / s`. The
  annual terms stay non-decreasing in the year's recovery, so the
  corridor is used up in event order like the deductible
  (`ceded_by_event`), on the grid it maps the annual recovery as the
  other annual terms do (a stop-loss with a corridor still has a net
  grid), and a corridor counts as an annual term that may not inure on
  the grid. Tower documents write it as `"loss_corridor": {"lower",
  "upper", "retained"}` only when there is one, and a document without
  the key loads with none, so the format stays version 1. Tested:
  hand-worked years through deductible, corridor and limit (whole and
  per event), the quota share's ceded loss against the closed form and
  gross = ceded + net in every simulated year, a corridor layer and a
  stop-loss with a corridor on the grid against 200,000 simulated years
  by Kolmogorov–Smirnov, and JSON round trips. Python
  `Layer.with_loss_corridor(lower, upper, retained=1.0)` and
  `Layer.loss_corridor`, R `with_loss_corridor()` and `@loss_corridor`.
- **Contract terms (decided 2026-10-09).** Premiums and commissions that
  depend on the year's ceded loss are terms of a `Layer`, with
  `aggregate` 1.0.1's definitions (`aggregate.contract_terms`):
  - *Premium quotes* set `premium`, the placed share's premium: a deposit
    for 100% of the layer (`deposit_premium`, `share × amount`), a rate on
    line (`rate_on_line`, `share × rol × limit`) or a rate on the subject
    premium (`premium_rate`, `share × rate × SPI`; a quota share's ceded
    premium is a rate of 1). Each reads the share when called, so the
    share comes first. `premium` keeps its old meaning (the base of paid
    reinstatements) and is now also the base of the commissions.
  - *Ceding commission*: flat (`ceding_commission(rate)`) or a sliding
    scale (`sliding_scale` from `(commission, loss_ratio)` anchors, linear
    between them and flat beyond the ends; the commission may not rise
    with the loss ratio), times the year's premium; the loss ratio is
    ceded loss over premium.
  - *Profit commission*: `share × max(premium (1 − allowance) − ceded,
    0)`, `aggregate`'s `pc <share> after <allowance>`. As there, it does
    not deduct the ceding commission; the allowance carries it.
  - *Swing rating*: `swing_rated(LossSensitivePremium)` makes the year's
    premium `clip(s·basic + lcm·ceded, s·minimum, s·maximum)` for the
    placed share `s`, terms quoted at 100%, as `aggregate`'s underwriter
    scales them. The minimum defaults to the basic premium, the maximum to
    no cap. `LossSensitivePremium::premium` alone is the retrospectively
    rated premium of an account's loss (`aggregate`'s `RetroTerms`); it
    is not a layer term, since it rates the cedant's own policy.
  - *What combines*: as in `aggregate`, a sliding scale and a profit
    commission need a positive premium first and refuse swing rating and
    paid reinstatements (a reinstatement premium would change the loss
    ratio's base); swing rating refuses paid reinstatements. Unlike
    `aggregate`, a layer may carry several terms at once (a sliding
    scale and a profit commission; a flat commission on a swing premium
    or beside paid reinstatements), since each is figured on its own.
  - *Results*: the terms never change ceded or net loss. `Tower::apply`
    adds a component per layer and term, `(swing_premium, name)`,
    `(ceding_commission, name)` and `(profit_commission, name)`, after
    the reinstatement premiums; a fixed premium is the layer's `premium`
    and gets none. `on_grid` reports each layer's `expected_premium`,
    `expected_ceding_commission` and `expected_profit_commission` from its
    ceded grid. Tower documents write each term only when present
    (`ceding_commission`, `sliding_scale`, `profit_commission`, `swing`),
    so older documents load unchanged (format version 1).
  - *Not done*: a deficit or credit carried forward between years in a
    profit commission or sliding scale (simulated years are independent;
    it needs multi-year simulation), and premium quoted as a rate of the
    gross premium the programme itself prices (`aggregate` refuses that
    loop too).
  Tested: hand-worked years and `aggregate`'s own examples for each term,
  the refused combinations, every simulated year's components against
  the per-year functions with gross = ceded + net, the grid's
  expectations within four standard errors of 200,000 simulated years, a
  JSON round trip with every term, and parity with `aggregate` 1.0.1's
  `SwingTerms`, `RetroTerms`, `SlideTerms` and `ProfitCommissionTerms` at
  475 losses through every kink to `1e-12`
  (`validation/scripts/aggregate_contract_terms.py`). Python
  `Layer.with_deposit_premium`, `with_rate_on_line`, `with_premium_rate`,
  `with_ceding_commission`, `with_sliding_scale`,
  `with_profit_commission`, `with_swing_rating`, `premium_for`,
  `ceding_commission_for`, `profit_commission_for` and
  `LossSensitivePremium`; R the same `with_*()` and `*_for()` functions
  and `retro_premium()`.
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
- **Risk profiles.** `prospicio_pricing::profile::RiskProfile`: bands of sum
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
  standard errors of these. It lives in prospicio-pricing (exposure curves),
  which now depends on prospicio-aggregate (`EventSet`). Python `RiskProfile`,
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
  from `(L + U) / 2` needs the tilted spread below to match both. Tested: a band from 1m to 5m against a 2m surplus cedes
  3/8 (closed form), where its 3m mean risk cedes 1/3, and simulation
  agrees within four standard errors.

### Grid sizing

`recommend_grid` sizes a 0-based grid with at most `2^log2` points
(default 16), with `aggregate`'s defaults (`p = 1 - 1e-5`,
`p* = 1 - 1e-12`, severity tail floor `1e-14`):

1. The bulk extent is the larger of the moment extent (the `p` quantile
   of a lognormal or gamma with the aggregate's mean and variance) and
   one big claim on a typical bulk at the same `p`,
   `E[S] - E[X] + q_X(1 - (1 - p)/E[N])`, so one claim leaves about
   `1 - p` beyond the grid.
2. `step = round_bucket(extent / 2^log2)` on `aggregate`'s ladder
   (`{1, 2, 4, 5, 8} × 10^k`, powers of two below 1), with the fewest
   points that cover the extent; when `2^log2` do not,
   `round_bucket(extent / (2^log2 - 1))`.
3. The single big jump at `p*` extends the grid when it fits at that step
   within `2^log2` points; a far tail never coarsens the step.
4. With an infinite variance the single big jump at `p` alone sizes the
   bulk. An infinite mean is refused: choose the step by hand.

`fft_auto` discretizes the severity by rounding on that grid (lumping the
mass beyond its top onto the last point) and runs `fft`.

Departures from `aggregate`, on purpose: it fits shifted lognormal and
gamma to three moments, where `Severity` stops at two, so step 1 adds the
bulk single big jump to reach as far on skewed books; its single big
jump uses a quantile fitted to the severity's moments, prospicio the
severity's own quantile; it refuses infinite-variance books without a
bucket. Signed and offset windows (a grid not starting at 0, for P&L)
wait on signed grids (`roadmap.md`, `prospicio-prob`).

### Occurrence PML

The year's largest occurrence is at most `x` with probability
`E[F(x)^N] = G_N(F(x))`, so the occurrence exceedance is
`OEP(x) = 1 − G_N(F(x))` for any count independent of the severities. For a
Poisson count that is `1 − exp(−λ S(x))` and the PML at `p` is the
severity quantile at `1 + ln(1 − p) / λ`, `aggregate`'s `oep`, which takes
Poisson counts only. prospicio takes any count through its pgf: a negative
binomial's clustering puts more of the years' occurrences in fewer years,
so the OEP is lower at the same mean.

`occurrence_pml` solves `1 − G_N(1 − s) = p` for the severity's exceedance
`s` by bisection on a log scale, then takes the severity quantile at
`1 − s`. `p` must lie in `(0, 1 − P(N = 0))`, the chance of at least one
occurrence. The occurrence return period is `1 / (E[N] s)`, the average
time between occurrences above the loss (for a Poisson count, the thinned
process's rate); it can be shorter than a year. The annual return period is
`1 / p`. Far in the tail (`s` below about `1e-12`) the quantile at `1 − s`
loses relative precision.

### Compound moments

By cumulants, so nothing cancels at large counts:
`E[S] = E[N] E[X]`, `Var[S] = E[N] Var[X] + Var[N] E[X]²` and
`κ₃[S] = E[N] κ₃[X] + 3 Var[N] E[X] Var[X] + κ₃[N] E[X]³`, with the count's
third cumulant from `Counting::third_cumulant` and the severity's from its
raw third moment (`distributions.md`, "Third moments"). A severity without
a closed-form third moment, or with an infinite one, is an error.

The approximations are `aggregate`'s method-of-moments fits
(`_fits.py`): the gamma `α = cv⁻²`, `θ = m/α`; the lognormal
`σ² = ln(1 + cv²)`, `μ = ln m − σ²/2`; the shifted gamma `α = 4/skew²`,
`θ = σ skew/2`, `shift = m − αθ`; the shifted lognormal with
`η = sqrt(e^σ² − 1)` solving `η³ + 3η = skew`, `shift = m − σ/η`,
`μ = ln(m − shift) − σ²/2`. Departures: `aggregate` falls back to a normal
for a symmetric book and reflects a left-skewed one; prospicio returns an
error for a shifted fit without positive skewness, and for a gamma or
lognormal fit without a positive mean. `Aggregate.approximate` fits the
moments estimated from its grid; prospicio fits the exact ones.

### Layer statistics and prices by view

`aggregate`'s `reins_stats_df` and `reins_price_df`, built from the pieces
above. For `share × min(max(X − a, 0), l)` on the severity grid:

- `pr_attach = P(X > a)` (exclusive) and `pr_detach = P(X ≥ a + l)`
  (inclusive), as `aggregate`, compared within `1e-9` of a step so a
  boundary on a point is not lost to rounding; `pr_loss` is the occurrence
  exceedance at `a`, `1 − G_N(F(a))` ("Occurrence PML");
- the excess count is the count thinned with `p = pr_attach`, by its
  cumulants: `κ₁ = p E[N]`, `κ₂ = p² Var[N] + p(1 − p) E[N]`,
  `κ₃ = p³ κ₃[N] + 3p²(1 − p) Var[N] + p(1 − p)(1 − 2p) E[N]`;
- the layer severity given a loss is the full layer's raw moments on the
  grid over `p` (`aggregate`'s conditional basis, so the excess count times
  it gives the annual mean); the annual ceded loss is `compound_moments`
  of the count and the full layer's grid; the loss on line is its mean
  over `share × l` (0 for an unlimited layer).

The share scales the `k`-th cumulant by `share^k` exactly. `aggregate`
instead maps the shared losses onto the grid, which splits a ceded 0.5
between 0 and 1: its `pr_loss` for a 50% layer is 0.4% low and its shape
smeared. The annual moments are exact for the grid; `aggregate` reports
the moments of its FFT density.

`TowerGrids::price` prices each view on its own: `L` is the grid's mean,
`P` its distorted expectation (`Grid::distortion`), `M = P − L`,
`LR = L / P`. A concave distortion's prices are subadditive, so then the
ceded and net premiums together are at least the gross. A stage with several layers has a price
per layer; `aggregate` also prices their total cession, which needs a
total-ceded grid (a small item below). `reins_price_df`'s capital columns
(`Q`, `a`, `ROE`) come with an asset level; `pricing.rs`'s pentagon gives
them for any grid.

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

Grid sizing is checked against `aggregate` 1.0.1
(`validation/scripts/aggregate_sizing.py`): `round_bucket` exactly on its
own docstring cases and their reciprocals, and on five books (Poisson and
gamma-mixed counts; gamma, lognormal, Weibull and Lomax severities) the
exact mean at `1e-3` and `aggregate`'s quantiles at 0.5 to 0.9999 within
two of its buckets, with aliasing below `1e-12`, the one-claim tail
estimate below `1e-5`, and a step never coarser than `aggregate`'s
(the same on two books, one to three rungs finer on three).

The occurrence PML is checked against `aggregate` 1.0.1's `oep`
(`validation/scripts/aggregate_oep.py`) on three Poisson books (lognormal,
gamma and Weibull severities) at eight annual probabilities from 0.3 to
`1e-5`: the loss at `1e-10`, the severity's exceedance and the occurrence
return period at `1e-9`. The simulated OEP, each year's largest event in
200,000 years, matches the exact curve for a negative binomial count (which
`oep` does not cover) within four standard errors at five losses.

Compound moments are checked on five books (Poisson with gamma, lognormal
and Weibull severities, a gamma-mixed Poisson, and a Poisson mean of 1,000)
against 30-digit mpmath closed forms at `1e-12`, and the five fits against
`aggregate`'s own `approximate_from_mcvsk` on those moments at `1e-10`
(`validation/scripts/aggregate_moments.py`, 85 values). The exact moments
are cross-checked against `aggregate`'s analytic statistics to `1e-8`:
`aggregate` is 6.5e-10 off the Weibull's mean and 2e-9 off its skewness
(numerical severity moments) and 2.5e-10 off the skewness at a Poisson mean
of 1,000 (raw moments converted).

Layer statistics and prices by view are checked against `aggregate`
1.0.1's `reins_stats_df`, `reins_price_df` (proportional hazard, ρ = 0.7)
and `reins_density_df` (`validation/scripts/aggregate_reinsurance.py`,
131 values) on three books on a step of 1 with `2^14` points: a Poisson
with one occurrence layer, the same with an aggregate stop-loss inuring
(an inuring tower), and a gamma-mixed Poisson with a gamma severity, a 50%
layer and a second layer. The grid towers match `aggregate`'s densities
point by point at `1e-13` absolute; statistics read from the severity grid
match at `1e-12`; the annual moments, which `aggregate` takes from its FFT
density, at `1e-9` (mean), `1e-8` (cv) and `5e-7` (skewness, observed
1.3e-7 at a skew of 18); premiums at `2e-8`, because the distortion raises
FFT round-off of `1e-16` in the far tail to the power 0.7 (`aggregate`'s
own two ways of forming the survival differ by as much). The 50% layer is
compared on its means and grid probabilities only (above).

## Next

1. The rest of `aggregate` parity tier 2, in this order (the gap list of
   2026-10-12 against `aggregate` 1.0.1's source; the OEP was its first
   item, now done):
   1. (Done: compound moments to third order and the moment
      approximations.)
   2. (Done: per-occurrence layer statistics and pricing each view under
      a distortion; the grid towers checked against `aggregate` itself.)
   3. A senior and junior priority ladder for capital (`priority_epd_df`,
      `priority_kappa`); prospicio has equal priority only.
   4. Eventual ruin by Pollaczeck–Khinchine (`pollaczeck_khinchine`,
      `eventual_ruin`).
   5. Layer loss picks (`Aggregate.picks`).
   6. `Bounds` and `PricingBounds`, the rest of the 2022 pricing bounds.
   Small items to fold into these: a total-ceded grid for a stage with
   several layers (its price and statistics, `reins_stats_df`'s `Ceded`
   column), statistics of aggregate covers, `tvar_threshold`, a stand-alone price on
   a grid portfolio, comonotonic allocation, named fixed, Bernoulli and
   geometric counts, and a reinstatement premium check against `aggregate`.
   Parked as large: the P&L ledger (`pnl`, which needs signed grids),
   bivariate FFT aggregates and `clash`, renewal counts with Wiener–Hopf
   ruin.
2. Deficit and credit carry-forward in profit commissions and sliding
   scales, once multi-year simulation exists.
