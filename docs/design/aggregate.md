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

Panjer and the FFT assume independence throughout: the claim count
independent of the severities, the severities independent and identically
distributed, and units combined on a grid (a portfolio's total as the
product of its units' transforms, `recommend_portfolio_grid`) independent of
each other. The dependent cases have their own tools: `fft_bivariate` and
the clash model (`Clash`) for two lines hit by the same events or sharing a
count ("Two lines and clash"), one mixed count on a severity `Mixture` for
components sharing a mixing variable ("A shared mixing variable"), and
`PredictiveDistribution::reorder_groups` or `reorder_groups_by_copula` for a
rank correlation or a copula between units' simulated totals. Monte Carlo on
an `EventSet` keeps whatever dependence its events carry (#491).

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
- `natural::Portfolio::{price_stand_alone, comonotonic_kappa}`: each unit
  priced alone at its own VaR against the book priced together, and
  Denuit et al.'s comonotonic allocation ("Stand-alone price and the
  comonotonic allocation" below). Python `Portfolio.price_stand_alone`
  (`StandAlone`) and `Portfolio.comonotonic_kappa`, R
  `price_stand_alone()` and `comonotonic_allocation()`.
- `prospicio_aggregate::bivariate::{fft_bivariate, JointGrid, Clash}`:
  the joint annual loss of two lines hit by the same events, by a 2-D
  compound FFT, and `aggregate`'s clash model ("Two lines and clash"
  below). Python `fft_bivariate`, `JointGrid`, `Clash`; R
  `fft_bivariate()`, `clash_model()`, `clash_severity()`,
  `clash_aggregate()`.
- `prospicio_aggregate::renewal::{renewal_count, Wait, WaitGrid}`: the
  number of claims by a horizon when the waits between claims follow any
  law, `aggregate`'s renewal frequency ("Renewal counts" below). Python
  `Count.renewal`, R `renewal_count()`.
- `prospicio_aggregate::ruin::{wiener_hopf, RenewalRuin}`: eventual ruin
  for renewal claims, `aggregate`'s `wiener_hopf` ("Eventual ruin for
  renewal claims" below). Python `wiener_hopf`, R `wiener_hopf()`.
- `prospicio_aggregate::pnl::{Ledger, Group, Leg, Source, Layout}`: the
  P&L ledger, `aggregate`'s `PnL`: signed legs in books that sell or buy
  cover, every row's exact distribution and the scenario columns, legs
  rebucketed on a step, the ratio exhibit (`Ledger::ratios`) and the stack
  of several ledgers' results (`pnl::stack`), and the builders of a book
  and its annual covers (`pnl::Book`: consolidated, walk and peel) and of
  its occurrence covers over the joint gross and ceded loss
  (`pnl::occurrence_source`, `Book::occurrence_ledger`), with variable
  terms, retro and reinstatement premiums as legs, the stitched peel of
  several occurrence layers (`Book::occurrence_peel`), and
  `pnl::pnl_program`'s premium ladder
  (`docs/design/pnl.md`). Python `Ledger`, `Group`, `Leg`, `stack`,
  `Book`, `Expense`; R `pnl_ledger()`, `ledger_ratios()`, `ledger_stack()`,
  `pnl_book()`, `book_ledger()`.
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
- **Labels split a tower's result** (#475). An `EventSet` may carry
  labels, a value per loss in each label dimension (a peril, a class),
  stored as an index into the sorted distinct combinations. With labels,
  `Tower::apply` gives dimensions `["kind", "layer", <label dims>]`:
  gross, each layer's ceded loss (the sum of its split by event, under the
  layer's `LimitAllocation`) and net once per combination, and the annual
  components (premiums and commissions) with `""` in every label
  dimension, so `aggregate(&["kind", "layer"])` gives back the unlabelled
  result and an unlabelled set's result is unchanged. `merge` keeps either
  set's labels (this set's dimensions, then the other's new ones, `""` for
  a missing value); `grouped` drops them, as it drops sums insured, since
  an occurrence may span labels; a term's years (`apply_term`) ignore them.
- **A reorder hands back its permutation** (#483, first part). A class's
  annual total reordered with others by `reorder_groups` or
  `reorder_groups_by_copula` must keep its events; the `_indexed` variants
  return, per group, the input simulation each result simulation came
  from, and `EventSet::permute(index)` takes the class's years in that
  order (losses, sums insured, times and labels together). The caller
  owns the alignment, as a reference model's simulation number does
  between its modules; no matching of values, which ties defeat. The
  index belongs to the group, so reorder on the class total (attritional
  plus large) and permute the large events with that group's index.
- **An aggregate part beside the events** (#483, second part). Attritional
  losses are an amount a year with no event and no time; wrapping them as
  one event lets a per-occurrence layer attach to them (on #483's three
  years, a 30 xs 25 cedes 0, 40, 35 instead of 0, 25, 30, 36% high on the
  mean). `EventSet::with_aggregate_part(pd)` keeps them apart: one column
  per component of `pd`, whose dimensions are label dimensions, so the
  part can be given by peril or class and joins `label_combinations`
  (the events get `""` in a dimension only the part has). Who sees it is
  structural: `Layer::sees_aggregate_part` is true for a layer on the loss
  with attachment 0 and no limit, a quota share or a stop loss, whatever
  its annual terms; any layer with a per-occurrence attachment or limit
  sees the events alone. Gross, net and `totals` include the part;
  `maxima` and `counts`, which count occurrences, do not. Inuring stages
  see it net of earlier stages that ceded from it.
  - *Where it sits in the year.* The part has no time, so it cannot go
    between dated events; it goes before them. Under the default
    `LimitAllocation::EventOrder` a binding annual deductible is used up
    on the attritional first and a binding annual limit stops the last
    large events. This moves only the split of the annual ceded loss by
    label and what a later inuring stage sees, never the annual ceded
    total; `LimitAllocation::ProRata` gives the split that does not depend
    on order. Which split a reference model uses is not settled by our
    sources (#483's open question), so this is a convention to revisit
    against one that states it.
  - `permute`, `slice`, `term_years` and `apply_term` carry the part with
    its years; `merge` keeps either set's and adds columns with the same
    labels; `grouped` keeps it summed over its labels, since it drops the
    events' labels; `with_labels` after the part is an error, so labels
    are never rebuilt without it.
- **Quota share and stop-loss are layers.** A quota share with cession
  `c` is unlimited cover from 0 with `share = c`; a stop-loss `l` xs `r`
  is unlimited cover from 0 with `AAD = r` and `AAL = l`. One formula
  covers all three contract types, and a stop-loss in a later stage covers
  the annual total net of earlier stages.
- **Plain annual losses** (#477). `Tower::apply_annual` takes one annual
  loss per simulation, each as one aggregate loss as `apply_aggregate`
  takes a predictive distribution's totals. A contract stated on the loss
  ratio is a layer on money amounts: a cap `c` on subject premium `P` is a
  stop loss of `c × P` xs 0 with `premium_rate(1, P)`, so the ceded loss
  ratio is the capped one and a sliding scale keyed on it reads it; scales
  on one grid add pointwise. Brokerage on a loss-ratio scale is not a leg
  yet (open, the user's decision).
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
  `Layer::allocation(LimitAllocation::ProRata)` splits the annual terms
  pro rata instead (#476): with per-event recoveries `u_e` summing to `U`,
  event `e` cedes `share × u_e × after_terms(U) / U` (factor 1 when
  `U = 0`). The annual total is the same; the nets a later stage sees are
  not. It is refused with pro rata as to time, whose charge depends on
  when the limit is used up.
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
- **Exposure-rating helpers and the spread's warning (decided
  2026-10-14, #212).** `TabulatedCurve::from_severity(severity, mpl,
  points)` tabulates a severity's curve for a band: half the points
  geometric from `1e-6` (the first chord sets the mean rate), half where
  `G` takes equal steps, so the curve's bend is sampled where the loss
  is. `Mbbefd::swiss_re_for_mean_rate` and `swiss_re_for_layer_share`
  calibrate the Swiss Re `c` on `[0, 10]` by bisection (both targets are
  monotone in `c`) and refuse a target outside that range. `layer_cv` is
  the CV of a layer's loss given a loss above the attachment, for
  `LimitedFluctuation`. A band may start at 0 (`with_bounds` takes
  `lower ≥ 0`, `upper > 0`). `Band::with_spread_bounds` tilts when the SI
  lies strictly between the bounds and spreads uniformly otherwise. The
  bindings keep the uniform spread as the default, so no result changes,
  but warn when it moves a band's mean; `spread="auto"` uses
  `with_spread_bounds`. The default stays uniform because the guides and
  notebooks print numbers from it. Tested: from_severity within `1e-3`
  of the exact curve, `c = 3` recovered from both targets to `1e-9`, the
  exponential's layer CV 1, a band from 0 rating a layer, and the warning.
- **Log-log interpolation of tabulated curves (decided 2026-10-15).**
  `TabulatedCurve::with_interpolation(x, g, Interpolation::LogLog)` runs
  the curve as a power law between points, `G(x) = Gₖ (x / xₖ)^pₖ` with
  the elasticity `pₖ = ln(Gₖ₊₁ / Gₖ) / ln(xₖ₊₁ / xₖ)`, straight on log-log
  axes, the way published scales are read off log-scaled charts and the
  way the CAS reading on exposure rating describes fitted alternatives.
  The first piece, from `(0, 0)`, stays straight (`ln 0` is undefined,
  and a power below 1 there would make `G'(0)` infinite and the mean rate
  0), so the mean rate is the first chord's under both interpolations.
  The interpolated curve is concave exactly when the elasticities never
  rise and the second piece's is at most 1; the constructor refuses
  anything else and names the point where the elasticity rises. A curve's
  elasticity `x G'(x) / G(x)` need not fall: a total-loss mass lifts it
  near `x = 1`, so the Swiss Re curves with `c` up to 3, read at a few
  points, fail and keep linear interpolation, while severity curves (a
  lognormal or Pareto capped at an MPL) and `c` = 4 and 5 pass. Rejected:
  a fallback to the chord on a piece whose elasticity rises, since the
  chord can still leave a convex kink at either end, and only a concave
  interpolant keeps the destruction-rate distribution valid; a general
  concave interpolant is a larger item, not planned. The destruction rate
  is continuous within each power piece, with atoms where the elasticity
  drops and a total loss with the last elasticity over `G'(0)`; its
  quantile inverts `G'` in closed form on each piece. Linear stays the
  default, so no result changes. Tested: the points reproduced, sampling
  against `G` and the mean rate, the atoms and a piece boundary, a flat
  top, and on ten-point tables of the Swiss Re `c` = 4 and 5 curves and
  of a lognormal and a Pareto severity curve the largest error past the
  first point at 0.36 to 0.67 of linear's. Python
  `TabulatedCurve(x, g, interpolation="log-log")` (kept through pickling),
  R `tabulated_curve(x, g, interpolation = "log-log")`.
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
  - *Not done*: premium quoted as a rate of the gross premium the
    programme itself prices (`aggregate` refuses that loop too). The
    deficit and credit carry-forward is done (below, "Contract terms over
    several years").
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
- **Contract terms over several years (decided 2026-10-16).** A profit
  commission or sliding scale may carry what a year could not absorb into
  the next years of a term, which needs years in sequence on the same
  simulated path. `aggregate` has no such term, so the definitions are
  ours, written to match how treaty wordings state them:
  - *Multi-year simulation*: `Tower::apply_term(&[EventSet])` takes one
    event set per year, all with the same number of simulations, and
    simulation `j` of every year is path `j`. The years are whatever the
    caller gives: independent years (`EventSet::term_years(n)` splits one
    simulation into `n` consecutive blocks, so path `j`'s year `t` is
    simulation `t·n_paths + j` and replays alone), or years that share a
    trend or a parameter drawn once per path. `apply_aggregate_term`
    does the same on aggregate losses. The result has dimensions `["kind",
    "layer", "year"]`, `apply`'s components once per year with the year
    `1..=T`, so `select("year", ...)` takes one year and `aggregate(&["kind",
    "layer"])` totals the term. Every year's ceded and net loss,
    reinstatement and swing premium is what `apply` gives for that year
    alone; only the carried terms read earlier years.
  - *Profit commission deficits*
    (`Layer::profit_commission_carry_forward(years)`): with `M = premium
    (1 − allowance)` and `D` the deficit carried in, the year pays `share ×
    max(M − ceded − D, 0)` and carries `max(ceded + D − M, 0)`. Carried to
    extinction, the commissions telescope: `Σ PC_t = share × (Σ (M −
    ceded_t) + D_T)`. Profit commissions carry no credits: a profit pays
    out in its year.
  - *Sliding scale deficits and credits*
    (`Layer::sliding_scale_carry_forward(deficit, credit, years)`): the
    year reads the scale at `(ceded + carried) / premium`; a deficit is
    the part above the last anchor's loss ratio (where the commission is
    at its minimum), a credit the part below the first anchor's (at its
    maximum), carried as a negative amount. Either or both may be carried.
  - *Oldest first, and a limit in years*: carried amounts are kept by the
    year they arose in and used up oldest first, so what a year carries
    out is its newest part. With `years = Some(n)` an amount from year `t`
    can be used in years `t + 1` to `t + n`, then is dropped (a common
    wording: "deficits carried forward for three years"); `None` carries
    it to extinction. `years = Some(0)` is refused.
  - *Elsewhere*: a single year (`apply`, `on_grid`, the P&L ledger) starts
    with nothing carried, so the term changes nothing there. A flat
    `ceding_commission` replaces a sliding scale and drops its carry.
    `Layer::profit_commissions_over` and `ceding_commissions_over` give
    the commissions over given ceded losses in consecutive years. Tower
    documents write `"profit_commission_carry": {"years"}` and
    `"sliding_scale_carry": {"deficit", "credit", "years"}` (`years` null
    without limit) only when present, and a document with one is stamped
    format version 2, the oldest that reads it, since a version 1 reader
    would drop the carry; others stay version 1.
  Tested (`prospicio-aggregate`'s `term` module): hand-worked paths
  through each rule (to extinction, a one- and two-year limit, deficits
  used oldest first, a sliding scale with deficits, credits and both),
  every year of a term against `apply` on that year alone without a
  carry, the telescoping identity on all 2,000 simulated paths of a
  four-year term, the refused builders, and JSON round trips. Python
  `Layer.with_profit_commission_carry_forward`,
  `with_sliding_scale_carry_forward`, `profit_commissions_over`,
  `ceding_commissions_over`, `Tower.apply_term`, `apply_aggregate_term`
  and `EventSet.term_years`; R `with_profit_commission_carry_forward()`,
  `with_sliding_scale_carry_forward()`, `profit_commissions_over()`,
  `ceding_commissions_over()`, `apply_tower_term()` and `term_years()`.
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
- **An annual loss on a grid takes any tower (#206).**
  `Tower::on_annual_grid(annual)` treats each year's total as one loss, as
  `apply_aggregate` does for draws: a quota share or a stop-loss on a
  loss-ratio or annual-loss distribution. Every layer's terms are then a
  function of the annual total, so inuring stages and annual terms are
  all allowed and the net always exists (`AnnualGrids`). The expected
  premium, reinstatement premium and commissions are taken at each annual
  loss, so exact for the grid; the ceded and net grids are mapped as in
  `on_grid`. A test checks it against `apply_aggregate` on the same years.
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
bucket. Signed and offset windows (a grid not starting at 0) are not
carried; the P&L ledger needs none (`pnl.md`: its rows are signed
functions of a non-negative source, on an irregular support).

`recommend_portfolio_grid` sizes one grid for a sum of independent units,
as `aggregate` sizes a `Portfolio`: the moment extent at the total's mean
and variance, and the single big jump on whichever unit needs the most
room (one large claim on that unit, the rest of the book at its mean).

### Grid audit

`recommend_grid` chooses a grid before any FFT runs; `audit_grid`
(#317) checks it afterwards, as `aggregate` 1.2.0's `sharpen` does.
Each cell discretizes the severity by rounding, runs `fft`, and scores
six moment errors, the severity's and the aggregate's mean, cv and
skewness against their exact values, each divided by its tolerance
(`eps`, `10 eps`, `100 eps`, `eps = 1e-4`) and combined by a power mean
(`p = 2`). A score of 1 is `aggregate`'s validation boundary. The
exact moments need the severity's third moment: its closed form, or the
quadrature of `E[X³]`, finite for any limited severity.

1. A center at or below `good_enough` (0.5) that holds all its mass is
   kept and nothing else is built.
2. Otherwise the step is doubled until the score stops improving, then
   halved likewise, at most four times each way, on the center's
   `log2` and on one less (above `2^8` points).
3. The choice is the best score among the cells that lose no mass off
   the grid (mass beyond `1e-12`), scores within a factor 1.25 counting as
   tied, ties going to fewer points, then to a smaller move. Only when
   that misses the target is `log2 + 1` searched (up to `2^20`), and taken
   if a sound cell there reaches it; failing that, the best cell anywhere,
   from which a second audit goes on.

`GridAudit` keeps every cell built (its moves, score, six terms, mass off
the grid, aliasing) with a note on why each walk stopped and the reason
for the choice. `audit_portfolio_grid` does the same for a sum of units:
each unit's compound transform on the grid, multiplied; the severity terms
score the claim-weighted mixture of the units' severities, the aggregate
terms the total. The bindings audit `recommend_grid`'s grid when no step
is given.

Departures from `aggregate`: the audit returns its verdict and leaves the
caller to rebuild on the chosen grid (prospicio's grids are values, not an
object updated in place); a discrete severity's pinned bucket is not
special-cased; the cells that lose mass are scored on prospicio's
rounding, which lumps a severity's mass beyond the top on the last point,
so their scores differ from `aggregate`'s.

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

Two more, not in `aggregate` but in Bahnemann (CAS Monograph 2, section
4.4), are defined by their quantiles: the normal power
`x_p = m + σ (z + skew (z² − 1)/6)`, whose distribution function inverts
the quadratic on its increasing branch, and Wilson–Hilferty
`x_p = m + σ (2/skew) ((1 + skew z/6 − skew²/36)³ − 1)`, the cube-root
normal approximation of the shifted gamma with the same moments (within
0.3% of its quantiles to the 99.9th at skewness 0.5). Both need a positive
skewness (#329).

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
ceded and net premiums together are at least the gross. A stage with
several layers also has its total cession, `TowerGrids::stage_ceded`,
which `aggregate` prices and reports as `Ceded`: without annual terms in
the stage it is a function of each loss, `Σ share × recovery`, compounded
once; with annual terms in a stage of several layers the layers' annual
totals are dependent and their sum is not one compound, so it is `None`.
`TowerPrice::stages` prices it.

`occurrence_cession_statistics` gives the `Ceded` column: the chance a
claim reaches the lowest layer and of a loss in the year, the loss on line
over `Σ share × limit`, and the moments of the ceded loss per ground-up
claim (every claim counted, as `aggregate`) and in the year, taken at the
grid's points, so exact with a shared layer too (`aggregate` smears it).
`aggregate_layer_statistics` gives an aggregate cover's (a stop loss on
any annual grid, gross or a tower's net): `P(S > a)`, which is also the
chance of a loss to it, `P(S ≥ a + l)`, the loss on line and the ceded
moments. `reins_price_df`'s capital columns
(`Q`, `a`, `ROE`) come with an asset level; `pricing.rs`'s pentagon gives
them for any grid.

### Priority ladder

`aggregate`'s `priority_epd_df` and `priority_kappa`, on
`natural::Portfolio`: each unit's recovery from assets `a` when it ranks
ahead of the other units pooled (senior, `E[Xᵢ ∧ a]`), pro rata (equal,
`E[Xᵢ min(1, a / X)]`, the existing `limited_expected`) or behind them
(junior, `E[min(Xᵢ, (a − X₋ᵢ)⁺)]`), and where in the total the shortfall
falls when unit `i` is junior.

- Neither the senior nor the junior leg is a function of `κ`, so
  `Portfolio` now keeps the joint law it was built from: the scenarios
  (rows and their probabilities) or the independent grids with each
  unit's leave-one-out density `p₋ᵢ` (one more inverse FFT per unit in
  `from_independent`).
- From scenarios every leg is a sum over scenarios, exact under any
  dependence; `aggregate` refuses anything but the independent combine of
  its units. From grids the junior leg is `Σ_y p₋ᵢ(y) E[Xᵢ ∧ (a − y)]`,
  one pass at one `a`; `E[Xᵢ ∧ t]` is linear between grid points, so `a`
  need not be one (`aggregate` snaps it to the grid). The conditional
  ladder is two FFT convolutions: with `y` the others' total,
  `min(x − a, x − y)` is `x − a` when `y ≤ a` and `x − y` otherwise.
- Both are two-tier readings, unit `i` against the rest pooled, as in
  `aggregate`: on three or more units the senior (or junior) recoveries
  are different hypotheticals and do not sum to `E[X ∧ a]`. The ordered
  k-tier schedule `min(Xⱼ, (a − Σ_{i<j} Xᵢ)⁺)` is not provided.
- The total row is `E[X ∧ a]` under every rule: at fixed assets
  subordination only redistributes. EPD is monotone in rank, senior ≤
  equal ≤ junior, for every unit; on two units
  `senior₁ + junior₂ = E[X ∧ a]`.

### Eventual ruin

`aggregate`'s `pollaczeck_khinchine` and `eventual_ruin`, in
`prospicio_aggregate::ruin`. Poisson claims with severity `X` and a premium
of `(1 + ρ)` times the expected loss: the maximal aggregate loss `L` is a
compound geometric, `P(N = n) = (1 − q) qⁿ` with `q = 1 / (1 + ρ)`, of the
equilibrium severity `S(x) / E[X]`, and `ψ(u) = P(L > u)`. The Poisson
rate cancels.

- On the severity's grid the equilibrium severity is `h S(jh) / E[X]` at
  `jh` (with the grid's mean, so it sums to 1), as `aggregate`. The
  geometric is the negative binomial with `r = 1`, `β = 1/ρ`, so the
  compound is the existing `fft`
  with its aliasing check; `aggregate` uses one FFT of twice the grid,
  wrapped. `ψ` at the last point is the mass beyond the grid.
- The grid puts ladder heights below `h` at 0, so `ψ(0)` is below the
  continuous `1 / (1 + ρ)`; `ψ` is `O(h / E[X])` off the continuous
  model (1% at `h = E[X] / 100` for exponential claims).
- `SeverityPart` carries `aggregate`'s `cap` (`X | X ≤ c`), `excess`
  (`X | X > c`, not shifted) and `stop_loss` (the mass above moved onto
  the first point at or above `s`).
- The capital for a ruin probability `p` is the smallest grid point with
  `ψ(u) < p` (`find_u`, `kind='index'`) or linear between grid points
  (`kind='interp'`). The Lundberg exponent solves
  `M_X(R) = 1 + (1 + ρ) E[X] R` on the grid's severity below the overflow
  guard `700 / max(x)`, else none.
- `eventual_ruin`'s simulated check of the same discretized model is not
  carried: the exact `ψ` is the deliverable. Ruin for renewal claims is
  "Eventual ruin for renewal claims" below.

### Renewal counts

`aggregate`'s `years ... wait ...` frequency (`_renewal.py`), in
`prospicio_aggregate::renewal`: the number of claims `N(T)` by a horizon
`T` when the waits between claims are independent draws from any law (a
Sparre Andersen process; exponential waits give the Poisson).

- `renewal_count(&Wait, horizon, grid)` returns a `RenewalCount`: the
  count as an `EmpiricalCount` (so `fft`, `panjer` and the rest take it),
  the wait grid, `P(W ≤ 0)`, the defect and the series length. A `Wait`
  is a `Law(&dyn Distribution)` (a `Mixture` covers `aggregate`'s wait
  mixtures) or `Atoms { values, probs }` (`dwait`), whose probabilities
  may sum to less than 1: a defective wait, which ends the process.
- `{N ≥ k} = {S_k ≤ T}`, so `P(N = k) = F^{*k}(T) − F^{*(k+1)}(T)`, each
  read off by Plancherel as the inner product of the wait's transform to
  the `k`-th power with the transform of the indicator of `[0, T]`, both
  tilted (`θL = 20`) to damp the wrap. The two transforms are taken
  separately: packed into one complex FFT, the indicator (up to `e^15`
  times the wait) puts its round-off on the wait and the count goes wrong
  (`renewal_count_pmf`).
- The series runs to `k_max = ⌈T/μ + 10√(Tσ²/μ³) + 10⌉` from the renewal
  central limit theorem on the grid's moments (capped at `ln 10⁻¹⁷/ln q`
  for a wait of total mass `q < 1`), and is cut at `⌊T/w⌋` for a smallest
  wait `w > 0`, where the tilted sums' round-off would put specks on
  impossible counts.
- Waits of zero are taken out exactly (`zero_wait_batches`):
  `N = G_1 + … + G_{M+1} − 1` with `M` the count of the positive waits and
  `G` geometric on `1, 2, …`; given `M = m`, `N − m` is negative binomial
  with `m + 1` successes, by the stable recurrence of its terms.
- The grid is `aggregate`'s `wait_grid` (`WaitGrid::sized`): the step
  resolves the shape (`μ/64`, and `σ/64` unless `σ < μ/10`) and the error
  budget (`k_max h² ≤ 1e-9`), with `T` at three quarters of `2^16` to
  `2^24` points and the reading at `T` on half the last cell; atoms on a
  lattice with `T` (their exact common step, each value first its closest
  fraction with a denominator up to `10⁹`, as `Fraction.limit_denominator`)
  take the lattice's step and the whole last cell. A `WaitGrid` can be
  given instead. The wait is put on the grid by rounding from survival
  differences, with the mass past `T` dropped.
- Not carried: `aggregate`'s unconditional splice window on a continuous
  wait (`splice [a b] !`, a defect by truncation), the layered wait
  (`wait y xs a`) and its snap of the grid to the cap. A `Truncated`
  wait covers the conditional splice.
- Python `Count.renewal(wait, horizon, *, step, points, closed)`, with
  `wait` a distribution or `(values, probs)`; R `renewal_count()`.

### Eventual ruin for renewal claims

`aggregate`'s `wiener_hopf`, in `prospicio_aggregate::ruin`:
`wiener_hopf(&severity, &wait, rho, points)` returns a `RenewalRuin`
(`ψ`, the all-time maximum's probabilities, the capital lookups of `Ruin`,
the premium rate and the grid's mean step). The waits between claims
follow any law; the premium comes in at `c = (1 + ρ) E[X] / E[W]`, so `ρ`
keeps its Pollaczeck–Khinchine meaning, and exponential waits give
`pollaczeck_khinchine` up to the grid.

- Ruin happens only at a claim, so `ψ(u) = P(M > u)` for the all-time
  maximum `M` of the walk with steps `Y = X − cW` (finite when `E[Y] < 0`,
  checked on the grid). Outside the Poisson case `M`'s ladder heights have
  no closed form; they come from the Wiener–Hopf factorization
  `1 − φ_Y = (1 − χ₊)(1 − χ₋)` by its cepstrum: the coefficients of
  `log(1 − χ₊)` sit on the strictly positive lags and those of
  `log(1 − χ₋)` on the rest, so the split is a cut (Spitzer's identity).
  The zero of `1 − φ_Y` at `z = 1` is divided out as `1 − z⁻¹` (a minus
  factor) with the value `−E[Y]` there, which leaves the phase with no
  winding, unwrapped as `numpy.unwrap`. Four FFTs on a circle of
  `2 · points`.
- `cW` is put on the severity's grid by rounding (survival differences on
  steps of `h / c`), its mass past the grid dropped; `E[Y]` is the grid's
  own, which absorbs that drop. The severity rides its grid as given (in
  `aggregate`, its rounding pmf rescaled to 1).
- `ψ` at the last point is the diagnostic (`aggregate` warns above
  `1e-6`): heavy tails need more `points`, which must be at least the
  severity's.
- Not carried: the defective wait (`aggregate` refuses it too) and the
  simulated paths of `ruin_example`.
- Python `wiener_hopf(severity, wait, rho, *, loss_ratio, points)`
  returning `RenewalRuin`; R `wiener_hopf()`.

### Layer loss picks

`aggregate`'s `Aggregate.picks`, as `prospicio_aggregate::picks::layer_loss_picks`:
reshape a severity grid so that its expected loss to each layer
`(a_{i−1}, aᵢ]` is a pick, keeping the tail above the top attachment.

- In each layer the survival function is rescaled about its value at the
  top, `S*(x) = ωᵢ + wᵢ (S(x) − S(aᵢ))`, worked from the top down with
  `ω_k = S(a_k)`, `wᵢ = (tᵢ − ωᵢ yᵢ) / ∫ (S − S(aᵢ))` and
  `ω_{i−1} = ωᵢ + wᵢ (S(a_{i−1}) − S(aᵢ))`; the layer integrals are the
  grid's, `h Σ S_j` over the points in `[a_{i−1}, aᵢ)`, as `aggregate`'s.
  The picks are per year for `claims` expected ground-up claims
  (`aggregate` divides by its `n`).
- Attachments must be grid points: a pick constrains an integral between
  two boundaries (`aggregate`'s ruling of 2026-08-24).
- A negative weight (a pick below what the layers above force through its
  layer) is an error here; `aggregate` warns and returns negative
  probabilities. So is a rescaled survival above 1 anywhere but the bottom
  layer.
- A bottom-layer pick whose rescaled survival passes 1 takes `aggregate`'s
  fallback: `S* = 1` up to a point, then a straight line to `ω₁` at `a₁`.
  `aggregate` rounds the continuous length `2(a₁ − t₁) / (1 − ω₁)` to the
  grid and misses (704.90 for a pick of 700 on a step of 1); prospicio
  solves the length on the grid, so the pick is met. It takes the line
  whenever the cap binds; `aggregate` only when the capped layer misses
  its pick by more than `1e-3`.

### Pricing bounds for other risks

`aggregate`'s `PricingBounds` and `Bounds` (Mildenhall, IME 2022), in
`prospicio_pricing::bounds`; the allocation bounds (`AllocationBounds`)
were already `Portfolio::premium_bounds` (`docs/design/risk.md`).

- The distortions pricing `X` at `P` have BiTVaRs as extreme points and
  price any `Y` at `∫ TVaR_p(Y) μ(dp)`, so `Y`'s price range is the
  vertical slice at `P` of the convex hull of `p ↦ (TVaR_p(X),
  TVaR_p(Y))`. For discrete risks both coordinates are affine in
  `1 / (1 − p)` between consecutive points of the union of the two sets of
  cumulative probabilities, so the curve is a polyline and the hull
  (monotone chain) exact. Each vertex's `TVaR` comes from tail masses and
  tail means summed from the top, located by bisection: `O(n log n)`.
- The curve ends at `p → 1`, `(max X, max Y)`: `TVaR_1` prices `X` at its
  maximum. `aggregate` stops at the last breakpoint below 1 and keeps the
  first vertex where `TVaR_p(X)` saturates, so where `X`'s top atom is
  larger than `Y`'s its upper bound is too low (52 where 64 is attained,
  for InsCo's X3 with both capped at 65 and `P = 65`). Where `TVaR_p(X)`
  ties, the lower hull takes the lowest `Y` and the upper the highest.
- `distortion_envelope`: `g(s)` of a BiTVaR is the price of a Bernoulli
  risk with `P(1) = s`, since its `TVaR_p` is `min(1, s / (1 − p))`. So the
  envelope of the distortions pricing `X` at `P` is that Bernoulli's price
  range, exact at any `s`. `aggregate`'s `Bounds` searches 256 levels and
  interpolates in `s`; it returns its lower envelope as a fitted concave
  distortion, prospicio returns the values.
- Not carried: the uniform reference (the Gini lens, `TVaR_p(U) = (1 +
  p) / 2`), which is affine in `p`, not `1 / (1 − p)`, so the hull on the
  breakpoints is not exact there; and `Bounds`' cloud and plots.

### Stand-alone price and the comonotonic allocation

`aggregate`'s `Portfolio.price_stand_alone` and
`make_comonotonic_allocations`, on `natural::Portfolio`.

- A unit alone holds assets at its own lower `p` quantile and is priced
  as `Aggregate.price` prices it: `L = E[Xᵢ ∧ aᵢ]`, `P = ρ_g(Xᵢ ∧ aᵢ)`.
  prospicio prices each unit's marginal as a one-unit portfolio, so the
  convention is `Portfolio::price`'s. The book is priced at the VaR of its
  total; `sum` adds the stand-alone loss, premium and assets, and the
  ratios follow from those sums, as `aggregate`'s `complete_pentagon`. A
  unit whose VaR is 0 has a zero pentagon.
- The comonotonic allocation is Denuit, Dhaene, Ghossoub and Robert
  (*IME* 120, 2025, Theorem 3.1), ported step for step from `aggregate`:
  the result depends on the sweep order, the violators' order and how the
  flattened mass is shared out (in proportion to the other units' rise),
  so a "cleaner" rewrite would not reproduce it. It runs on the totals
  with positive probability; `aggregate` runs on its whole grid, where
  the zero-probability points (with `exeqa` 0) leave the values at the
  others unchanged (to 3e-14), so they are not carried.

### Two lines and clash

`aggregate`'s `BivariateAggregate` with independent coupling and its
`clash` statement, in `prospicio_aggregate::bivariate`. Each of `N` shared
events gives a pair of losses `(X, Y)` from a joint per-event severity
`S` (a `JointGrid`: a step on each axis, rows for line A); mass at 0 on an
axis is an event that misses that line. The pgf acts elementwise on a
transform, so `(A, B)` is `iFFT2(P_N(FFT2 S))`, and each marginal is the
1-D compound of `N` with that axis's per-event marginal.

- `Clash::new(only_a, only_b, both)` takes `aggregate`'s three expected
  counts; independence within an event fixes `neither = only_a only_b /
  both`, so `n = only_a + only_b + both + neither`,
  `p_A = (only_a + both) / n`, `p_B = (only_b + both) / n`. The per-event
  severity is the outer product of `(1 − p_A) δ_0 + p_A f_A` and the same
  for B. `Clash::aggregate` checks that the count's mean is `n`. With a
  Poisson count, `Cov(A, B) = both · E[X] E[Y]` (since
  `(only_a + both)(only_b + both) = both · n`).
- The 2-D transform is rustfft's 1-D transform on the rows, a transpose
  and the columns: no threads, so the WASM build is unchanged. The buffer
  on each axis is the power of two at least twice the points (as `fft`);
  `aggregate` doubles its own axis length (`padding=1`).
- Like `fft`, the mass past the last row or column is carried on it, so
  the result sums to 1 and its marginals equal `fft`'s; `aggregate`
  truncates and reports a `deficit`. What the probabilities lack of 1
  (round-off, a severity summing to 1 within `1e-9`) stays missing: on the
  far corner it moved the second moments by `4e-9`.
- The FFT's round-off is kept signed until the output is folded, then
  negatives are set to 0: clamping each buffer cell first biased each
  marginal by up to `1e-11`. `aggregate` zeroes every cell below `1e-15`,
  which drops real tail mass (about `1e-11` on the parity grids, `3e-9`
  on the second moments).
- Aliasing: wrapping on one axis does not move the other, so the wrapped
  mass on each axis is that of its marginal, measured by the 1-D compound
  on a buffer twice as long (`aliasing_mass`); their sum bounds the total
  misplaced probability. A 2-D aliasing check at twice the buffer would
  cost four times the memory.
- The total `A + B` is exact on the anti-diagonals when the steps are
  equal, else each cell is split between the two points of the larger
  step around its total, keeping the mean (`aggregate`'s
  `JointBandsMixin.total`).
- Copula coupling (`JointGrid::from_copula(a, b, &copula)`, with
  `bivariate::triggered(g, p)` for a line an event hits with probability
  `p`): the per-event severity is the copula's rectangle probabilities over
  the two per-event marginals' cdfs, `S[i, j] = C(F_A(i), F_B(j)) −
  C(F_A(i − 1), F_B(j)) − C(F_A(i), F_B(j − 1)) + C(F_A(i − 1), F_B(j − 1))`,
  as `aggregate`'s copula `bivariate`. Any two-dimensional `Copula` with a
  distribution function (`Copula::cdf`: Gaussian, Clayton, Gumbel, Frank,
  Joe). `C(u, 1) = u` keeps the marginals exact; the last point of each
  axis is read at exactly 1. A difference of nearly equal cdf values can
  come out a little below 0 (round-off, to about `1e-15`): it is kept, as
  `aggregate` keeps it, since setting it to 0 added mass to every row and
  column and moved the second moments by 2e-9 to 7e-9 relative; a cell
  below `−1e-12` is an error. Python `JointGrid.from_copula`, R
  `copula_severity()`.
- Not carried: signed and `pnl` axes,
  the `netceded` joint of one aggregate's ceded and net, the disk-backed
  (massive) path, and sizing the axes automatically from a total budget
  (points are given per axis).

### A shared mixing variable

`aggregate` 1.2.0 shares a mixing variable `Θ` in two places only (#316,
read from its source). First, the severity components of one
`Aggregate` (`agg [10 20] claims sev [gamma lognorm] ... mixed gamma 0.3`)
share it: `update_work` builds one severity, the components weighted by
their expected counts, under one mixed count. In prospicio that is `fft`
of a `MixedPoisson` on a `Mixture`. Second, the two units of a
`bivariate` share its outer frequency, mixed or not (`BivariateCatPair`,
the library's common shock). In prospicio that is `fft_bivariate`, or
`Clash::aggregate`, with a mixed count. A `Portfolio`'s units share
nothing: its total is the product of the units' transforms, each unit
keeping its own mixing. Sharing is always full. The mixing families are
gamma, Delaporte, inverse Gaussian, shifted inverse Gaussian, beta and
Sichel, all in `Mixing` (next section). Since nothing
in `aggregate` shares `Θ` across the units of a portfolio, prospicio
builds no quadrature over a shared `Θ` (the user's decision of
2026-10-19: no extension beyond `aggregate`).

### Beta and Sichel mixings

`Mixing::Beta { cv, max }` is `aggregate`'s `mixed beta cv max`: the
Poisson mean `λ Θ` is `max` times a `Beta(a, b)` variable with
`a = (max - λ (1 + cv²)) / (cv² max)` and `b = a (max - λ) / λ`, which
needs `max > λ (1 + cv²)`. `P(N = k)` is
`max^k / k! · B(a + k, b) / B(a, b) · 1F1(a + k; a + b + k; -max)`; by
Kummer's transformation the series has positive terms, summed on a log
scale, so it holds where `aggregate`'s `hyp1f1` at complex points loses
digits.

`Mixing::Sichel { cv, shape }` is `mixed sichel cv shape`: `Θ` generalized
inverse Gaussian, density proportional to
`x^(λ - 1) exp(-ω (x / η + η / x) / 2)` with shape `λ`, its `ω` solved by
bisection on a log scale for the coefficient of variation and `η` set by
the mean 1 (`aggregate`'s `mu = η`, `beta = η / ω`). The variance is finite
for `λ` in `(-2 - 1 / cv², 1 / cv²)`. `Mixing::SichelSkewness { cv,
skewness }` solves the shape too, by bisection on that interval, where the
skewness falls from the inverse gamma's to the gamma's;
`Mixing::sichel_gamma(cv, f)` and `Mixing::sichel_ig(cv, f)` target the
Delaporte's `2 cv / (1 - f)` and the shifted inverse Gaussian's
`3 cv / (1 - f)`, `aggregate`'s `sichel.gamma` and `sichel.ig`.
`P(N = k) = (λ η)^k / k! · s^(-(k + λ)) K_{k+λ}(ω s) / K_λ(ω)` with
`s = √(1 + 2 λ η / ω)`, the Bessel ratios by the forward recurrence in the
order, which is stable for `K`. `ln K_ν(x)` for real order is new in
`prospicio_math::special::ln_bessel_k`: the trapezoidal rule on
`∫_0^∞ exp(-x cosh t) cosh(ν t) dt`, geometric in the step, to `2e-15`
against mpmath. The moments come from differences of `ln K`, since the
recurrence cancels at small `ω`. Neither mixing has a fixed part, as in
`aggregate`; the pgf is the probability table summed by Horner's rule.

Parity (`validation/scripts/aggregate_mixing.py`): the Sichel and
`sichel.ig` probabilities and solved parameters against `aggregate` 1.2.0
at `1e-9` and `1e-8`, the limit of its Broyden solve. Two cases are
checked elsewhere because `aggregate` is wrong there: the beta, whose
probabilities by quadrature agree at `1e-14` while `aggregate`'s are off
by up to `8e-9`, and `sichel.gamma 0.4 0.3`, where `aggregate`'s solve
stops, with no warning, at a mixing of mean 1.476 and cv 0.071 (the count's
mean is 14.76, not 10); SciPy's nested `brentq` solve gives the shape
0.4306 that meets all three targets, and prospicio agrees with it at
`1e-12`.

### Deal deductions and ratios

Issues #485 and #477, with the user's reading of a reference simulation
pricing model (`knowledge/findings/deal-ratios-and-scale-keys.md`).

1. **Brokerage** is a `Commission` in its own slot on `Layer`
   (`Layer::brokerage` flat, `Layer::brokerage_scale` on the loss ratio).
   The scale type and function are the sliding scale's (linear between
   anchors, held at the ends), but the rate need not fall. It sets the
   brokerage in the results and ratios. A provisional brokerage for cash
   flows belongs to #479.
2. **Other deductions** are `Deduction`s: a label, a basis
   (`ExpenseBasis`: a rate of the year's premium by default, of the ceded
   loss, or an amount) and whether NMBF counts it (yes by default; a
   manager's fee is kept outside). Each is its own `(label, layer)` kind in
   a tower's result, after the profit commission and brokerage. Labels
   cannot be a result's kinds or a ratio's names.
3. **The scales' key.** By default both scales read the ceded loss ratio,
   as `aggregate` does. `Layer::scales_before_annual_terms` keys them on the
   placed share of the year's recoveries before the annual deductible,
   corridor and annual limit, over the premium. That is the reference
   model's `LR`, the loss ratio before the cap and corridor, when the cap is
   stated as an `aggregate_limit`. The result then carries
   `(loss_before_terms, layer)`, and a sliding-scale carry over a term
   accumulates that key. The grid methods and the P&L ledger see only the
   ceded loss, so they refuse such a layer with annual terms and a sliding
   scale.
4. **The clamp.** `Layer::brokerage_within_commission_range` holds the
   brokerage key inside the commission scale's loss-ratio range. It is the
   reference model's convention but off by default.
5. **Sections.** A deal of several sections is one layer per section, each
   scale on its own section's loss ratio and premium, and the amounts add.
   Summing scales point by point on one grid is right only for one section.
   A deal-level scale is a `DealScale` added with `Tower::deal_scale`
   (`[deal-scale]`): a sliding-scale commission and a brokerage (flat or a
   scale, optionally held in the deal commission's range) read on the sum
   of the sections' keys over the sum of their premiums, paid on that
   summed premium, and added to the sections' own. Each section's key is
   the one its own scales read (the ceded loss, or the loss before annual
   terms), so a deal has no key flag of its own. A deal has at least two
   sections, each with a positive fixed premium (no swing rating or paid
   reinstatements) and in no other deal. Its commission carries deficits
   and credits over a term on the deal's key, apart from the sections'
   carries. The results gain `(ceding_commission, <deal>)`,
   `(brokerage, <deal>)` and, when a section reads its loss before terms,
   `(loss_before_terms, <deal>)`. `deal_ratios` takes a deal's name and
   sums the sections' amounts with the deal's over their premium; a
   section's own ratios leave the deal out, since no allocation of the
   deal's amounts to sections was decided. The grid methods and the P&L
   ledger refuse a tower with a deal: they see one layer's ceded loss at
   a time. JSON format version 4 carries `"deals"` beside the stages.
6. **Ratios.** `Tower::deal_ratios(result, layer)` reads a tower's result,
   annual or over a term (with a `year` dimension). Every amount is over
   the year's premium for the placed share (fixed or swing rated;
   reinstatement premiums are not in it). It gives `loss_ratio`,
   `loss_ratio_before_terms` where the scales read it, CC, PC and BK where
   present, each deduction, `total_deductions = CC + PC + BK + sum(d)`,
   `combined_ratio`, `margin = 1 - combined_ratio` and `nmbf` (the margin
   plus the deductions kept outside). The commissions are read from the
   result, so carried amounts count as they were paid.
7. **JSON.** A tower with brokerage, deductions or either flag is format
   version 3; others keep their version.
8. **The ledger and the grids.** The P&L builders book a tower only as
   covers the book buys (buy groups: the premium paid, the commissions
   received). Brokerage and the other deductions come out of the premium
   the reinsurer receives, so the cedant's ledger correctly leaves them
   out. The reinsurer's own ledger is `Tower::deal_ledger(gross, layer)`
   (`[deal-ledger]`): one sell group for a layer or a deal over the annual
   gross losses, the tower applied to each atom as one annual loss, with
   legs `Premium`, `Loss`, `Ceding commission`, `Profit commission`,
   `Brokerage` and each deduction by its label. Its amounts are those
   `deal_ratios` divides by the premium, and a deal books its sections'
   amounts with its own. NMBF is not a row. It refuses a joint source and
   a booked layer with paid reinstatements, which belong to occurrence
   covers. `TowerGrids` and `AnnualGrids` report the
   expected commissions but not brokerage or deductions. Those come from
   simulated years and `deal_ratios`.

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

The shared mixing variable is checked against `aggregate` 1.2.0
(`validation/scripts/aggregate_common_shock.py`): its two-component
`Aggregate` under one gamma mixing with cv 0.3, at a step of 0.5 on `2^14`
points, has the mean, cv and skewness of `fft(MixedPoisson, Mixture)` to
`1e-7` and the same 0.5 to 0.999 quantiles; each component mixed on its
own gives `aggregate`'s `independent` cv, 0.360832, to its
discretization.

The grid audit is checked against `aggregate` 1.2.0's `sharpen`
(`validation/scripts/aggregate_sharpen.py`) on two books: a lognormal
capped at 1,000 from a step of 1 on `2^13` points, where both move to a
step of 4 with a score of 0.035326 and the same six terms, and a gamma
from 0.5 on `2^10`, where nothing sound reaches the target and both grow
to a step of 1 on `2^11` (0.722254). Every sound cell's score agrees to
the six printed digits, and so does the mass off the grid of a cell
`aggregate` disqualifies.

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
198 values) on four books on a step of 1 with `2^14` points: a Poisson
with one occurrence layer, the same with an aggregate stop-loss inuring
(an inuring tower), a gamma-mixed Poisson with a gamma severity, a 50%
layer and a second layer, and a Poisson with two occurrence layers and an
aggregate cover on their net (the total cession's statistics and price,
and the cover's statistics). The grid towers match `aggregate`'s densities
point by point at `1e-13` absolute; statistics read from the severity grid
match at `1e-12`; the annual moments, which `aggregate` takes from its FFT
density, at `1e-9` (mean), `1e-8` (cv) and `5e-7` (skewness, observed
1.3e-7 at a skew of 18); premiums at `2e-8`, because the distortion raises
FFT round-off of `1e-16` in the far tail to the power 0.7 (`aggregate`'s
own two ways of forming the survival differ by as much). The 50% layer is
compared on its means and grid probabilities only (above), and so is the
total cession with it (its price at `1e-7`). The total cession's per-claim
cv and skewness, which `aggregate` reads off its ceded severity density,
are 1.5e-11 and 2.6e-9 off the points' moments (`1e-8`); an aggregate
cover's chances, read off the annual FFT density, carry its round-off
(`1e-12` absolute) and a rare cover's shape more (cv `1e-7`, skewness
`1e-5`, observed 1.4e-6 at a skew of 32).

The priority ladder is checked against `aggregate` 1.0.1's
`priority_epd_df` and `priority_kappa`
(`validation/scripts/aggregate_natural.py`, 212 values) on the Discrete
case of *Pricing Insurance Risk* and on a third independent unit added to
it, at four asset levels each, at `1e-10` (recoveries) and `1e-9` (the
conditional ladder). A test checks the identities above, and that the
ladder from the 36 joint scenarios of the three-unit case equals the one
from its grids at `1e-12`, at asset levels on and off the grid.

Eventual ruin is checked against `aggregate` 1.0.1's
`pollaczeck_khinchine` and its Lundberg exponent
(`validation/scripts/aggregate_ruin.py`, 110 values): a gamma severity
(mean 50, cv 1.5) on `2^16` points at ρ = 0.1 and 0.25, and capped,
in excess and stopped at ρ = 0.1, and claims of 1, 2 or 3. `ψ` matches at
`1e-12` absolute, the capital on the grid exactly, interpolated at `1e-10`
relative (observed 1.5e-11: round-off over a slope of about 1e-6), `R` at
`1e-10` relative. Unit tests check `ψ` against the exponential closed form
`e^{−ρu / ((1 + ρ)μ)} / (1 + ρ)` and its `R`, and the FFT against Panjer.

Renewal counts are checked against `aggregate` 1.0.1's renewal frequency
(`validation/scripts/aggregate_renewal.py`, 267 values) on the renewal
cases of its own test suite: exponential waits over 10 years, gamma waits
(shape 2) over 5, a fixed wait, waits of 0 or 1 (clusters), a defective
wait and a mixture of exponentials. The wait grid (step, points, reading)
and `P(W ≤ 0)`, the defect and `k_max` match exactly; the counts match at
`1e-8`. At `k = 0` and 1 on the continuous waits `aggregate` is up to
6e-9 from the exact values (the round-off of its tilted sums) and
prospicio within 5e-11; 42 more rows check prospicio against the exact
values at `1e-9`: the Poisson for exponential waits, and `P(N = 0) =
P(W > T)` for the others.

Eventual ruin for renewal claims is checked against `aggregate` 1.0.1's
`wiener_hopf` (`validation/scripts/aggregate_wiener_hopf.py`, 96 values):
a gamma severity (mean 10, cv 0.3) with exponential, gamma (shape 2) and
mixed exponential waits, and a lognormal (mean 10, cv 1) with gamma waits
of shape 0.5, on a step of 1/32 over `2^16` points, at ρ = 0.1 and 0.3.
The severities are put on the grid as `aggregate` does (the tail past the
grid dropped and the rest rescaled, not piled on the last point). `ψ` and
the maximum's probability at 0 match at `1e-10` (observed 3e-12), the
capital on the grid exactly, the severity's mean at `1e-12` relative. A
unit test checks exponential waits against `pollaczeck_khinchine`.

Layer loss picks are checked against `aggregate` 1.0.1's `Aggregate.picks`
(`validation/scripts/aggregate_picks.py`, 406 values) on a gamma severity
(mean 100, cv 1.5) on `2^14` points with layers to 100, 500 and 1,000:
picks that rescale every layer, and picks whose bottom layer takes the
straight line (compared above the bottom layer, where both agree). The
adjusted probabilities match at `1e-12`, the layer losses at `1e-9`
relative; the bottom layer's pick is met to `1e-9`.

Pricing bounds are checked against `aggregate` 1.0.1
(`validation/scripts/aggregate_bounds.py`, 80 values) with InsCo's total
as `X`: `PricingBounds` for InsCo's X3 and a four-point risk, uncapped at
six premiums and capped at 65 at five, at `1e-12` relative. Where
`aggregate` drops the `p → 1` vertex (six rows, capped) the reference is a
brute force over the BiTVaRs on the curve's vertices with `p = 1`, using
`aggregate`'s TVaR. The envelope is checked against `PricingBounds` on
Bernoulli risks at `1e-12`, and against `Bounds`' grid envelope as
inequalities (outside it, within 0.05; it pokes out by up to 2.4e-4, from
its root tolerance and interpolation). Unit tests reprice every bound's
BiTVaR: it prices `X` at `P` and `Y` at the bound, to `1e-10`.

The stand-alone price and the comonotonic allocation are checked against
`aggregate` 1.0.1 (`validation/scripts/aggregate_natural.py`, 276 values)
on the Discrete case and its three-unit extension: each unit's, the
book's and the sum's `L`, `P` and `a` under the proportional hazard at 0.7
and Wang at 0.3 at VaR levels 0.8, 0.9 and 0.95, and each unit's
comonotonic allocation at every total, at `1e-10`. A doctest checks that
the allocations add up to the total and do not fall.

The expected reinstatement premium of a grid tower is checked against
`aggregate`'s `ReinstatementTerms` (`contract_terms.py`) applied to its
own unlimited annual occurrence recovery
(`validation/scripts/aggregate_reinsurance.py`, 4 values): 500 xs 100 at
a 20% rate on line on the lognormal book at 2 and 10 claims, with one
reinstatement at 100% and with two at 100% and 50%, at `1e-10` relative.
`aggregate` prices reinstatements only inside its P&L builders, pro rata
as to amount; prospicio's pro rata as to time has no counterpart there.

Clash is checked against `aggregate` 1.0.1's `clash`
(`validation/scripts/aggregate_clash.py`, 152 values): 2 events a year
hitting A only, 3 B only and 1 both, gamma and lognormal losses, a
Poisson and a gamma-mixed (negative binomial) count, on steps 1 and 0.5
and on step 1 for both. The joint density, each marginal and the total
at `1e-12`, means, variances, `E[AB]` and the correlation at `1e-9`
relative (observed `4e-11`), with `aggregate`'s `1e-15` clip turned off.
Unit tests check each marginal against the 1-D `fft` within the reported
aliasing mass, and the covariance against `Var(N) E[X] E[Y]` at `1e-10`.

Copula coupling is checked against `aggregate` 1.0.1's copula
`bivariate` (`validation/scripts/aggregate_bivariate_copula.py`, 216
values): line A hit with probability 1/4 and losing the clash case's
gamma, B with 1/3 and its lognormal, coupled by a Gumbel and a Clayton
copula with Kendall's tau 0.4 and a Gaussian with rho 0.5, each under a
Poisson and a gamma-mixed count with mean 12, on steps 1 and 0.5. The
same quantities and tolerances as clash pass. `aggregate`'s Gaussian
evaluates scipy's bivariate normal cdf, which is exact in two dimensions:
replacing it with a `1e-13` one moves the density by `2e-17`. Unit tests
check that every copula keeps both marginals to `1e-15`, that a Gaussian
at rho 0 is the independent product, the sign of the covariance under
positive and negative dependence, and the errors (a three-dimensional
copula, the t copula, a trigger outside `[0, 1]`).

## Next

1. The rest of `aggregate` parity tier 2, in this order (the gap list of
   2026-10-12 against `aggregate` 1.0.1's source; the OEP was its first
   item, now done):
   1. (Done: compound moments to third order and the moment
      approximations.)
   2. (Done: per-occurrence layer statistics and pricing each view under
      a distortion; the grid towers checked against `aggregate` itself.)
   3. (Done: the senior and junior priority ladder, `priority_epd_df`
      and `priority_kappa`, on `natural::Portfolio`.)
   4. (Done: eventual ruin by Pollaczeck–Khinchine, `pollaczeck_khinchine`
      and `eventual_ruin`, in `prospicio_aggregate::ruin`.)
   5. (Done: layer loss picks, `Aggregate.picks`, in
      `prospicio_aggregate::picks`.)
   6. (Done: `Bounds` and `PricingBounds`, the rest of the 2022 pricing
      bounds, in `prospicio_pricing::bounds`.)
   (Done, tier 2 complete: the small items folded into these, the total
   cession of a stage with several layers, its price and statistics,
   `reins_stats_df`'s `Ceded` column, statistics of aggregate covers,
   `tvar_threshold` and the named fixed, Bernoulli and geometric counts
   (`distributions.md`), the stand-alone price on a grid portfolio, the
   comonotonic allocation, and a reinstatement premium check against
   `aggregate`.)
   Parked as large, and the candidates for what comes next: (done:
   bivariate FFT aggregates, `clash`, and copula coupling of the two lines;
   see "Two lines and clash"; renewal counts, see "Renewal counts");
   eventual ruin for renewal claims, see "Eventual ruin for renewal
   claims"); then the P&L ledger (`pnl` and `xpnl`), planned in
   `pnl.md`.
2. (Done: deficit and credit carry-forward in profit commissions and
   sliding scales, over multi-year terms; see "Contract terms over
   several years".)
