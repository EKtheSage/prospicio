# Changelog

What each release of prospicio contains. Every crate, the Python package
`prospicio` and the R package share one version (`docs/release.md`). A
release PR adds its section here; work merged since the last release goes
under "Unreleased" until then.

## Unreleased

## 0.0.5 (2026-10-23)

### Breaking changes

- Rust: `Layer` gains the public fields `brokerage`, `deductions`,
  `scales_before_annual_terms` and `brokerage_within_commission_range`;
  code that builds a `Layer` in full adds them (`None`, an empty `Vec`,
  `false`, `false`) or uses the builders (#511).
- Rust: `Tower` gains the public fields `deals`, `cover_basis` and
  `interlocking`; code that builds a `Tower` in full adds them (an empty
  `Vec`, `None`, `false`) or uses `Tower::new` and the builders (#512,
  #527, #532).
- Rust: `BayesOneYear` gains the public fields `light_refit` and
  `no_light_refit`; code that builds it in full adds them or starts from
  `BayesOneYear::default()` (#499).
- Rust: `prospicio_reserving::Error` has the variant `Excess`, so an
  exhaustive `match` on it needs a new arm (#507).
- Default change: Compartmental model 2's one-year view refits its
  flagged simulations with a light refit by default, so it is slower and
  more accurate than in 0.0.4; see the Reserving item below for how to
  turn it off (#499).
- Towers saved with brokerage or deductions, a deal scale, a cover basis
  or interlocking are JSON format versions 3 to 6, which 0.0.4 cannot
  read; towers without them are saved as before (#511, #512, #527, #532).

### Reserving

- The Bayesian Bornhuetter-Ferguson with a normalised development
  pattern, `BayesBfNormalised` (behind the `bayes` feature): the
  expected ultimate of each origin has the gamma prior (mean `apriori`
  times the exposure, coefficient of variation `cv`) and the pattern is a
  set of shares summing to 1, so `apriori` is the expected ultimate loss
  ratio and moves the reserve, where `BayesBf`'s (EVW's Model 3.1,
  unchanged) barely does. A flat prior is the Bayesian chain ladder; a
  tight prior at the chain ladder's own ultimates is the classical
  Bornhuetter-Ferguson reserve (at other loss ratios the pattern is
  re-estimated given the ultimates). It has the one-year view
  (`BayesBfNormalised::one_year`) through the generic engine. In Python
  `BayesBfNormalised` and `BayesBfNormalisedFit` (`pattern`,
  `expected_ultimate`) with `.one_year()`, and in R `bayes_bf_normalised()`,
  `bayes_bf_normalised_fit` and `bayes_bf_normalised_one_year()`; the fit
  of a one-year view is the new fit class (#410, #498).
- Excess-layer development, PR 1 (Rust only; issue #192, `docs/design/reserving-excess.md`):
  `ExcessPattern::from_ground_up` derives a layer's age-to-age factors from a
  ground-up incurred triangle, an optional claim count column, the ground-up
  tail and the ultimate severity by a scale shift of the severity, reproducing
  Bloom and Patrik's worked example exactly; `ExcessPattern::selected` turns it
  into a `SelectedPattern`. `layer_share` and `layer_exposure` give the layer
  share of a severity and a layer exposure column for a BF or Cape Cod prior.
  A paid basis is an error. Python, R and the credibility blending follow (#507).
- A fully selected age whose own factor cannot be estimated (every earlier value
  zero, as in the early ages of a sparse layer) is no longer an error in
  `Development`: its own factor is missing (NaN) and the selected factor is used.
  Every result that was an estimate before is unchanged; Merz and Wuthrich's closed
  form and the bootstraps reject such a fit (#507).
- Excess-layer development, PR 2 (Rust only; issue #192, `docs/design/reserving-excess.md`):
  `ExcessPattern::credibility_weights` gives the weight on the derived layer
  pattern for a blend with the layer triangle's own factors, `1 - n/(n + k)`
  with `n` the layer claims (or origins with a positive value) and `k`
  required, capped so that removing one origin moves the blended factor by at
  most `max_influence` of it, and exactly 1 where the own factor cannot be
  estimated. `SeverityByMaturity::ParetoShape` with `ShapeBase` derives layer
  factors from Pinto and Gogol's Pareto shape drift and a base development
  (the counts above the threshold or a base layer's losses), reproducing their
  and Bear's examples. A 500-book simulation study checks the derived pattern
  and the blend against the layer's own chain ladder (#510).
- Excess-layer development, PR 3 (Python and R; issue #192, `docs/design/reserving-excess.md`):
  `prospicio.reserving` has `ExcessLayer`, `ScaleShift`, `ParetoShape`,
  `ExcessPattern.from_ground_up` (with `selected`, `tail`, `ldf` and
  `credibility_weights`), `layer_share`, `layer_exposure` and `layer_at_shift`;
  R has `excess_layer()`, `scale_shift()`, `pareto_shape()`, `excess_pattern()`,
  `excess_selected()`, `excess_credibility_weights()`, `layer_share()`,
  `layer_exposure()` and `layer_at_shift()`. A layer's development pattern
  derived from the ground-up triangle goes into `ChainLadder`, `Mack` and the
  expected-loss methods through their `ldf`, `ldf_kind`, `ldf_weight` and `tail`
  arguments. Bloom and Patrik's, Pinto and Gogol's and Bear's examples are
  reproduced in both languages (#513).
- Excess-layer development, PR 4 (#192): `LayerSharePrior` (Rust, Python and R:
  `layer_share_prior()`, `bayes_bf_layer()`, `bayes_bf_layer_one_year()`) turns
  a layer share and its uncertainty into the gamma prior of the normalised
  Bayesian Bornhuetter-Ferguson, centred on the ground-up loss ratio times the
  share, with the coefficient of variation of a product of independent
  variables. A wider share cv widens the layer's predictive interval (#516).
- The excess-layer BF prior (`LayerSharePrior`,
  Python `LayerSharePrior`, R `layer_share_prior()`) treats the share's
  error as one error common to all origins (#518, `reserving-excess.md`
  decision 23(a), PR #522). `BayesBfNormalised` gets an optional multiplier `s` (Rust
  `common_cv`, Python `common_cv=`, R `common_cv =`, a draw named `log_share`)
  with mean 1 and the share's cv, sampled jointly with the other parameters so
  the data update the share for every origin. The total's prior interval is
  about three times as wide as with independent priors, but its posterior
  interval is narrower on the test layer (the origins pool what they say about
  the share), and the fitted total interval narrows on well-developed triangles (where
  the data cannot identify the share it stays wider). Pass
  `share_dependence="per_origin"` (R `share_dependence = "per_origin"`, Rust
  `ShareDependence::PerOrigin`) for independent priors by origin. A common
  share error needs `apriori_cv > 0`; `LayerSharePrior::model` returns a
  `Result` in Rust. New: `prior_total` (Python `prior_total()`, R
  `layer_prior_total()`), the closed-form mean and sd of the layer's total
  prior ultimate. Models without `common_cv` are unchanged bit for bit.
  (`LayerSharePrior` is new in this release, so no released default changes.)
- **Behaviour change:** compartmental model 2's one-year view (`Compartmental::one_year`, `one_year_segments`; Python `Compartmental.one_year`, `one_year_segments`; R `compartmental_one_year()`, `compartmental_model_2_one_year()`, `compartmental_one_year_segments()`) now refits its flagged simulations with a light refit by default, so it is slower and accurate by default. `BayesOneYear::light_refit` (`LightRefit`; Python `light_refit=` and R `light_refit =`) is a shorter NUTS run scaled by the opening posterior: model 2's totals come within about 0.03 sd of full refits on company 337 at about a thirtieth of their cost, but 2,000 simulations take about 31 minutes on 16 threads, against 85 s when reweighted only, which was the default before (about 0.4 sd off a refit). Turn it off with `BayesOneYear::no_light_refit`, Python `light_refit=False` or R `light_refit = FALSE` (reweight only, with the approximate warning), or set `n_sims` or `max_refits` to bound the time. Model 1 and the other Bayesian models are unchanged (#394, #499).
- `Compartmental` fits paid and incurred with `fitted = Fitted::PaidAndIncurred` (Python `fitted="paid_and_incurred"`, R `fitted = "paid_and_incurred"`; `reserving-compartmental.md` decision 12): incurred replaces outstanding as the second response. An incurred cell's mean is `RLR_i (OS(t) + RRF_i PD(t))` times premium (or as a loss ratio), with the cumulative paid `PD` even under incremental paid, and the second sigma (`log_sigma_os`) is the incurred response's. Pass the incurred column (`incurred=`) or an outstanding column (`outstanding=`, the incurred is then outstanding plus paid). The fit's `outstanding` then holds the future incurred cells. It has the one-year view, `fit_segments` and the diagonal backtest like paid and outstanding, and its own Stan reference on company 337. `PaidOnly` and `IncurredOnly` still return a not-yet-built error (#534).

### Distributions and probability

- Bootstrap standard errors of a statistics grid
  (`StatisticsGrid::bootstrap_se`): each replicate resamples the
  simulations once for every column, so the items' errors share their
  resamples, and is read off counts on each column sorted once;
  `bootstrap_se_holding` keeps a fixed part of each simulation and
  resamples only the rest. Python `bootstrap_se` (with `hold_fixed`), R
  `bootstrap_se()` (with `fixed`) (#488, #503).
- Where each simulation came from in a reorder:
  `PredictiveDistribution::reorder_groups_indexed` and
  `reorder_groups_by_copula_indexed` return each group's source
  simulations beside the result (Python `return_index=True`, R
  `index = TRUE`), so anything simulated with a group follows it without
  matching values (#483, #504).
- `Grid::normalized` rescales probabilities that do not sum to 1, which
  `Grid::new` rejects (Python `Grid(..., normalize=True)`, R
  `grid_distribution(..., normalize = TRUE)`), and
  `Distortion::parameter` gives a one-parameter family's parameter (Python
  `Distortion.parameter`, R `@parameter`) (#214, #519).

### Aggregate and reinsurance

- `EventSet::permute` takes a set's years in another order, losses, sums
  insured, times and labels together: with a group's index from a reorder
  it keeps a class's large events with its reordered totals (Python
  `EventSet.permute`, R `permute_events()`) (#483, #504).
- An aggregate part on an `EventSet` (`EventSet::with_aggregate_part`):
  amounts a year with no event or time, attritional losses by peril or
  class, which quota shares and stop losses cede from
  (`Layer::sees_aggregate_part`) and per-occurrence layers skip; it comes
  before the events in the year under annual terms, and gross, net and
  `totals` include it (Python `EventSet.with_aggregate_part`,
  `aggregate_part`, `Layer.sees_aggregate_part`; R `with_aggregate_part()`,
  `aggregate_part()`) (#483, #505).
- A deal's deductions on a layer: brokerage, flat (`Layer::brokerage`) or
  on a loss-ratio scale (`brokerage_scale`), and `Deduction`s (a TPA fee,
  tax, reinsurance cost, fronting, a manager's fee) on the premium, the
  ceded loss or an amount, inside or outside NMBF, each its own component
  of a tower's results. `Tower::deal_ratios` gives each simulation's loss
  ratio, each deduction, the total, the combined ratio, NMBF and the
  margin, by year over a term. `Layer::scales_before_annual_terms` keys the
  commission and brokerage scales on the loss ratio before the annual terms
  and adds `loss_before_terms` to the results.
  `brokerage_within_commission_range` holds the brokerage key in the
  commission scale's range. Towers with them are JSON format version 3
  (Python `with_brokerage`, `with_brokerage_scale`, `with_deduction`,
  `with_scales_before_annual_terms`, `Tower.deal_ratios`; R
  `with_brokerage()`, `with_deduction()`, `deal_ratios()`) (#485, #477, #511).
- A deal-level scale on several layers together (`Tower::deal_scale` with a
  `DealScale`): a sliding-scale commission and a brokerage read on the sum
  of the sections' keys over the sum of their premiums, paid on that
  premium and added to each section's own, with its own carry over a term.
  The results gain `(ceding_commission, <deal>)` and `(brokerage,
  <deal>)`, and `Tower::deal_ratios` takes a deal's name. Towers with a deal
  are JSON format version 4; the grids and the P&L ledger refuse them
  (Python `Tower.with_deal_scale`, `Tower.deals`; R `with_deal_scale()`)
  (#477, #512).
- The reinsurer's P&L ledger of a layer or a deal (`Tower::deal_ledger`):
  one sell group over the annual gross losses with the premium, the ceded
  loss, the commissions, the brokerage and each deduction as legs, the
  amounts `deal_ratios` divides by the premium (Python `Tower.deal_ledger`;
  R `deal_ledger()`) (#477, #514).
- A monthly cash-flow engine for collateralised deals
  (`prospicio_pricing::cashflow`): a trust account and an investor's
  operating account per simulation, from its ultimate loss ratio and
  commission, with every rule a setting (the collateral requirement as
  loads on named bases with a floor and a switch to the limit, a
  commutation requirement, premium collection and adjustments, funding
  dates, rule and source, release timing, interest, opening capital,
  expenses, a buffer and a tiered performance fee); `CashflowTerms::run`
  gives the investor's monthly flows for `npv_rows` and `irr_rows`, and
  `trace` every account line of one simulation (#479, #515, #517); in Python (`prospicio.pricing.CashflowTerms`, `CollateralRequirement`,
  `PremiumAdjustment`, `Cashflows`) and R (`cashflow_terms()`,
  `cashflow_requirement()`, `cashflow_adjustment()`, `cashflow_run()`,
  `cashflow_trace()`).
- `PaymentPattern::unpaid_from(age)`: the pattern of what is unpaid at an
  age, times from the age and shares over the outstanding (Python
  `PaymentPattern.unpaid_from`, R `payment_pattern_unpaid()`), and
  Pentagon's loss under Price's name, `expected_loss`, in Python (#214, #519).
- A book of deals on one shared simulation (`prospicio_pricing::book`):
  `DealRun` holds a deal's premium, loss and deductions in each
  simulation (from `Tower::deal_columns`, the amounts behind
  `deal_ratios`, or from arrays), its inception in months and its
  investor's flows; `DealBook` selects and shifts deals and gives every
  amount by deal and in total with ratios as sums over sums, the
  distribution of an item with each deal's contribution and marginal
  effect on either tail, the investors' flows on one calendar with the
  book's IRR, NPV and MOIC on the summed flows, and premium, paid loss
  and flows by calendar year (#490, #521, #523); in Python (`DealRun`, `DealBook`,
  `BookCashflows`) and R (`deal_run()`, `deal_book()` and the `book_*()`
  functions).
- Underwriting years with a rank correlation for multi-year terms:
  `EventSet::correlated_years` reorders whole simulated years so their
  totals have a Spearman or normal-score correlation, every year keeping the
  one-year distribution, and `calibrate_year_correlation` finds the
  correlation for a target ratio of the term's spread to one year's (#489, #524);
  in Python (`EventSet.correlated_years`, `.calibrate_year_correlation`,
  `year_correlation_for_ratio`) and R (`correlated_years()`,
  `calibrate_year_correlation()`, `year_correlation_for_ratio()`).
- One accident-period timeline shared by the underwriting years of a term
  (`prospicio_aggregate::timeline`): `Timeline` holds each event once over
  the term's accident months, `Exposure` gives each underwriting year's
  share of each month, and `Timeline::years` splits it into contract years
  for `Tower::apply_term` on a losses-occurring or risks-attaching basis
  (#489, #525; Rust).
- A tower's cover basis (`Tower::cover_basis`, losses occurring or risks
  attaching) and `Tower::apply_timeline`, which applies the tower over the
  contract years that basis gives on a `Timeline`; a tower with a basis is
  JSON format version 5 (#489, #527; Rust).
- The shared timeline and a tower's cover basis in Python (`Timeline`,
  `Exposure`, `Tower.with_cover_basis`, `Tower.apply_timeline`) and R
  (`timeline()`, `timeline_from_years()`, `timeline_years()`,
  `uniform_exposure()`, `with_cover_basis()`, `apply_tower_timeline()`)
  (#489, #528).
- Interlocking contracts under risks attaching (`Tower::interlocking`,
  off by default): an event that several underwriting years write cuts
  each year's contract limits by that year's share of the event; a tower
  with interlocking is JSON format version 6 (Rust, Python
  `Tower.with_interlocking`, R `with_interlocking()`) (#532).

### Excel

- `PRO.BAYES.BF`: the Bayesian Bornhuetter-Ferguson by MCMC on a
  triangle and a premium column, with a gamma prior on each origin's
  ultimate (mean `apriori` times exposure, CV `cv`, or `"flat"`) and an
  optional multiplier common to all origins (`common_cv`). Views:
  reserves, prior and posterior ultimates, the posterior and the
  sampler's diagnostics, as `PRO.BAYES.ODP`'s (#410, #529).
- `PRO.EXPOSURE.CURVE` and `PRO.EXPOSURE.QUANTILE` tabulate several
  curves at once: a column of x against a row of Swiss Re c values (or
  MBBEFD b or g) gives one column per curve (#509).
- The test workbook (`harness/make_workbook.py`) calls every one of the
  96 functions with a check beside it; the ODP sheet's P95 checks read
  the P95 column (#509).

### Examples

- The notebook tests check the numbers and claims the notebooks' prose states
  (`reinsurance_pricing`, the "Reading the result" bullets of `one_year_view`)
  and run the chart cells under matplotlib's Agg backend; stale skewness,
  quantile-tie and density-deficit figures in `reinsurance_pricing` and its R
  twin are corrected. A notebook
  opened outside the repository now raises a `FileNotFoundError` that says
  where to run it, not a bare `StopIteration` (#269, #531).
- `one_year_view` (notebook and R twin) models reserve risk as the one-year
  claims development result (`OdpBootstrap.one_year`, R `odp_one_year()`), not
  the run-off reserve; places the per-risk XL at the premium it prices (TVaR
  99% plus a 10% cost of capital, deposit plus expected reinstatements); and
  charges the net view what the reinsurance costs (ceded premiums less the
  surplus's ceding commission, plus reinstatement premiums), reporting the TVaR
  saving net of cost and the cost against it (#266, #267, #268, #535).

## 0.0.4 (2026-10-21)

### Breaking changes

- Rust: `PSpline` gains a public `kind` field (`SmoothKind`); code that
  builds a `PSpline` in full adds it or uses `PSpline::new` (#282).
- Rust: `Layer` gains the public fields `profit_commission_carry` and
  `sliding_scale_carry`; code that builds it in full adds them (`None`)
  or uses the builders (#301).
- `prospicio_models::Term` has new variants (`Interaction`, `Poly`,
  `Hinges`, `Steps`, `Bins`, `Log`, `Ordinal`, `OrthogonalPoly`,
  `NaturalSpline`), so an exhaustive `match`
  on it needs new arms (#350).
- `prospicio_aggregate::ApproximationKind` and `Approximation` have the
  variants `NormalPower` and `WilsonHilferty`, so an exhaustive `match` on
  them needs new arms (#329).
- `prospicio_prob::Dist` has the variant `Affine`, so an exhaustive `match`
  on it needs a new arm (#318).
- `prospicio_prob::severity_fit::Family` has the variant `Lomax`, so an
  exhaustive `match` on it needs a new arm (#328).
- Rust: `OdpGlmFit::coding` is an `Option<Coding>`, `None` for a fit of
  `OdpGlm::fit_design`; `glm.names()` names the columns of either (#333).
- Rust: `OdpGlm` gains a public `family` field; code that builds it in
  full adds `family: Family::Poisson` or uses `..OdpGlm::default()` (#333, #445).
- Rust: `ElasticNet` gains the public fields `lower_limits` and
  `upper_limits`; code that builds it in full adds them (`None`) or uses
  `ElasticNet::new` (#356).
- `Mixing` has three new variants (`Beta`, `Sichel`, `SichelSkewness`);
  Rust code that matches it exhaustively adds arms for them or uses
  `Mixing::cv`.

### Reserving

- **Breaking:** `CellForecast` has a new field `diagnostics:
  Option<Diagnostics>`, the sampler diagnostics of the fit behind the
  forecast (`None` for a model with no sampler); code that builds a
  `CellForecast` in full must set it. `Diagnostics` moves to
  `prospicio_reserving::Diagnostics` (outside the `bayes` feature), and
  `prospicio_reserving::bayes::Diagnostics` stays as a re-export.
- **Breaking:** `METRICS` grows from four to six names, `mean_log_density`
  and `total_percentile` after the old four, so code that sizes arrays by
  its length changes. `diagonal_backtest` scores both.
- `Backtest::diagnostics(model)`, `Backtest::warnings()` and
  `Backtest::diagonals()`: every split's diagnostics, the warnings as
  `(model, split, message)` and each split's held-out calendar period. A
  Bayesian fit with warnings is still scored.
- **Breaking:** `diagonal_backtest` returns an error naming two models
  with the same name, before any fit (it used to accept them);
  `NamedModel::new(name, model)` renames one. (#420)
- The diagonal backtest in Python and R (#342, #420). Python:
  `prospicio.reserving.diagonal_backtest(triangle, column, models,
  n_diagonals, premium=...)` with `BayesOdp`, `Meyers` or a new
  `GlmCandidate` (`GlmCandidate.odp()` for the ODP model), returning a
  `Backtest` with `scores()`, `mean()`, `diagnostics()`, `warnings`,
  `diagonals` and `log_densities`. R: `diagonal_backtest()` with the new
  settings objects `odp_glm_spec()`, `bayes_odp_spec()` and
  `meyers_spec()`, and the same tables as data frames. Both report each
  split's sampler diagnostics and the two new metrics.
- Gesmann and Morris' compartmental model over segments and in the diagonal
  backtest (behind the `bayes` feature). `Compartmental::fit_segments`
  fits each segment of a keyed triangle on its own seeds (so `fit`'s draws
  are unchanged) into joint distributions of the reserves and of the
  future paid and outstanding cells, the segments independent or
  rank-correlated by a Spearman matrix; the rank correlation reorders each
  segment's reserves, paid cells and outstanding cells together, and
  `Synchronized` is an error. `Compartmental` gains a `dependence`
  setting. `CompartmentalCandidate` puts the model in the diagonal
  backtest next to the ODP GLM, `BayesOdp` and `Meyers`, and
  `outstanding_backtest` scores the held-out outstanding diagonals with the
  same metrics (Rust only). In Python `Compartmental(..., dependence=,
  spearman=)` and `Compartmental.fit_segments(triangle, paid, premium, *,
  outstanding=None, incurred=None)`, returning a `CompartmentalFits`; in R
  `compartmental_segments()`, returning a `compartmental_fits`.
- The one-year view of Gesmann and Morris' compartmental models 1 and 2,
  `Compartmental::one_year` (behind the `bayes` feature): the claims
  development result over the coming year by Pareto smoothed importance
  sampling of the posterior. An origin with no observed cell (and, in
  model 2, an age with none) has its effects in each posterior draw, so
  the year's cells update them. On company 337 about four in five of
  model 1's simulations are flagged and all of model 2's; for model 1
  moment matching rescues about 95% of them within a twentieth of an sd
  of their refits, for model 2 none, and each refit takes as long as a
  fit (for model 2 about 31 s of wall time with 8 running side by side).
  So, by the user's decision of 2026-10-16, model 1's view moment-matches
  its flagged simulations and refits with NUTS those it cannot rescue,
  while model 2's (any model with development effects,
  `Compartmental::one_year_refit`) reweights them only and says in the
  view's `approximate` that it is approximate (about 0.4 sd of the total
  off a refit per simulation); setting `refit` still refits. Both run at
  most `Compartmental::ONE_YEAR_SIMS` (2,000) one-year simulations by
  default; the lifetime view keeps `n_sims`. In Python
  `Compartmental.one_year(triangle, paid, premium, *, outstanding=None,
  incurred=None, k_threshold=0.7, refit=None, max_refits=None,
  n_sims=None)`, returning a `CompartmentalOneYearFit` (with
  `approximate`, and a warning when it is set); in R
  `compartmental_one_year()` and `compartmental_model_2_one_year()` (with
  `refit = NULL` and `one_year_sims = NULL`), returning a
  `compartmental_one_year_fit` (#303).
- A refit budget for every Bayesian one-year view:
  `BayesOneYear::max_refits` refits at most that many simulations (the
  first due for a refit, in order) and reweights the others, listing them
  in the view's new `unrefitted`; `None`, the default, refits every one as
  before, bit for bit. Over segments the budget applies to each segment. Python's `one_year` and `one_year_segments` of
  `BayesOdp`, `BayesBf` and `Meyers` take `max_refits=None` and their
  views have `unrefitted`; R's `bayes_odp_one_year()`,
  `bayes_odp_one_year_segments()`, `bayes_bf_one_year()`,
  `meyers_one_year()` and `meyers_one_year_segments()` take
  `max_refits = NULL` and their views have `@unrefitted`. A warning names
  the simulations left unrefitted. **Breaking (Rust):** `BayesOneYear`,
  `BayesOneYearFit` and `BayesOneYearSegment` have a new public field, so
  a struct literal of them without `..` must set it (#303).
- **Breaking (Rust):** `BayesOneYear::refit` is an `Option<OneYearRefit>`,
  `None` by default for the model's own one-year default (moment matching
  for every model but `Compartmental` with development effects, so
  `BayesOdp`, `BayesBf` and `Meyers` behave as before); write
  `refit: Some(OneYearRefit::RefitFlagged)` where `refit:
  OneYearRefit::RefitFlagged` was. `BayesOneYearFit` has a new public
  field, `approximate` (#303).
- The one-year view of the Bayesian Bornhuetter-Ferguson,
  `BayesBf::one_year` (behind the `bayes` feature, settings in
  `BayesOneYear`): the claims development result over the coming year by
  Pareto smoothed importance sampling of the posterior, the flagged
  simulations refitted with NUTS, as for `BayesOdp`. Under a flat prior
  its standard deviation agrees with `BayesOdp`'s one-year view; under a
  tight one with the closed form of the strong-prior limit. In Python
  `BayesBf.one_year(triangle, column, exposure, k_threshold=0.7,
  refit="refit_flagged")` and in R `bayes_bf_one_year()`, returning the
  same `BayesOneYearFit` / `bayes_one_year_fit` as `BayesOdp`, whose `fit`
  is now the model's own lifetime fit (a `BayesBfFit` here); its frames
  add `exposure` and `prior_ultimate` (#274).
- The Bayesian one-year view is written once for every Bayesian model,
  on `prospicio_bayes::elpd::psis`, and checked by a conformance suite
  run on each model. `BayesOdp`'s one-year view is unchanged, bit for
  bit. A simulation whose year has a log likelihood that is not a number
  is now an error, as a failed refit is (#274).
- Moment matching in the Bayesian one-year view, opt-in:
  `OneYearRefit::MomentMatchThenRefit` moves the posterior draws of a
  flagged simulation by affine maps to the moments its importance weights
  estimate (Paananen et al. 2021, as R's `loo::loo_moment_match`, with
  its split estimate) and refits with NUTS only a simulation whose Pareto
  k stays above the threshold; `BayesOneYearFit::moment_matched` (and
  `BayesOneYearSegment::moment_matched`) lists the simulations it
  rescued. `RefitFlagged` stays the default. In Python
  `refit="moment_match_then_refit"` and `.moment_matched`, in R the same
  `refit` value and `@moment_matched` (1-based). **Breaking (Rust)** for
  code that matches `OneYearRefit` exhaustively or builds
  `BayesOneYearFit` or `BayesOneYearSegment` in full (#274).
- **Behaviour change:** moment matching is now the default fallback of the
  Bayesian one-year view. `OneYearRefit::default()` and
  `BayesOneYear::default()` are `MomentMatchThenRefit` (was
  `RefitFlagged`), and the `refit` argument of Python's `one_year` and
  `one_year_segments` and of R's `bayes_odp_one_year()`,
  `bayes_odp_one_year_segments()` and `bayes_bf_one_year()` defaults to
  `"moment_match_then_refit"`. The simulated years and their Pareto k are
  unchanged; a flagged simulation that matching rescues is no longer
  refitted, so a default run of `BayesOdp` or `BayesBf` takes seconds
  instead of minutes and its rows differ from before within the refits'
  accuracy. Pass `RefitFlagged` (`refit="refit_flagged"`) for the old
  behaviour (#280).
- The one-year view of Meyers' CRC, CSR and CAY models,
  `Meyers::one_year` and `Meyers::one_year_segments` (behind the `bayes`
  feature): the claims development result over the coming year by Pareto
  smoothed importance sampling of the posterior, reweighted on the
  unconstrained scale NUTS samples, with moment matching and then a NUTS
  refit for the flagged simulations. On group 353 nearly every simulation
  is flagged; matching rescues about nine in ten within a tenth of an sd
  of their refits. In Python `Meyers.one_year(triangle, column, premium,
  k_threshold=0.7, refit="moment_match_then_refit")` and
  `Meyers.one_year_segments`, in R `meyers_one_year()` and
  `meyers_one_year_segments()`, returning the same `BayesOneYearFit` /
  `bayes_one_year_fit` (its `fit` a `MeyersFit`) and `BayesOneYearFits` /
  `bayes_one_year_fits`, which now has `premium` for Meyers' views
  (`scale` is the ODP's only) (#280).
- **Behaviour change:** `Meyers::one_year` and `one_year_segments` run at
  most 2,000 one-year simulations by default (`Meyers::ONE_YEAR_SIMS`),
  not the model's `n_sims` (10,000 by default), since nearly every
  simulation of these models costs a moment match or a NUTS refit; the
  lifetime view keeps `n_sims`. A new `BayesOneYear::n_sims`
  (`Option<usize>`, `None` by default) sets the count for any Bayesian
  one-year view; `None` keeps `BayesOdp`'s and `BayesBf`'s own `n_sims`.
  In Python `n_sims=None` on `one_year` and `one_year_segments`, in R
  `one_year_sims = NULL` on `bayes_odp_one_year()`,
  `bayes_odp_one_year_segments()`, `bayes_bf_one_year()`,
  `meyers_one_year()` and `meyers_one_year_segments()`. Adding the field
  breaks a `BayesOneYear { .. }` literal without `..Default::default()`
  (#280).
- The ODP bootstrap, Mack's bootstrap and their one-year views follow
  `n_periods`, `drop_high` and `drop_low`. `OdpBootstrap` gets the three
  settings (volume-weighted chain ladder only): the fitted values are
  back-cast with the factors under the exclusions, an incremental cell has a
  residual only if the link ratio it closes is kept, and the pseudo factors
  average the kept links only. The scale follows chainladder-python's rule
  (the kept squared residuals over `N - p`), so it is biased low under
  exclusions (RAA, `n_periods = 3`: 694 against 984). Mack's bootstrap uses
  the settings it already took: its standard deviation under them is Mack's
  standard error under the same settings. The drops stay fixed on the
  observed triangle in the simulations. A one-year view takes the
  re-reserving method's exclusions; a bootstrap that sets different ones is
  an error naming both, and the mean of the claims development result is
  not zero under the exclusions. Merz and Wuthrich's closed form still
  rejects them. The geometric average stays rejected. In Python
  (`OdpBootstrap(n_periods, drop_high, drop_low)`) and R (`odp_bootstrap()`;
  `odp_one_year()`, `mack_bootstrap()` and `mack_one_year()` no longer
  raise) (#401, #434). **Breaking (Rust):** `OdpBootstrap` has three new public
  fields, so struct literals need `..Default::default()`; `Error` has a new
  variant, `ExclusionsMismatch`.
- `prospicio-reserving` has a `slow-tests` feature for its own test suite:
  the unit tests that sample with NUTS are ignored without it, and CI runs
  them in release (#279).
- `Compartmental` (models 1 and 2) in the Python and R diagonal backtest:
  `Compartmental.candidate(...)` and `compartmental_candidate()` go in the
  `models` of `diagonal_backtest`, and `outstanding_backtest` and
  `compartmental_outstanding_backtest()` score the held-out outstanding on
  the same fits. `CompartmentalCandidate` now owns its triangle and columns
  (no lifetime parameter, **breaking** for Rust code that named
  `CompartmentalCandidate<'_>`) and a clone shares its cache of fits.
- The one-year view of `Compartmental` over segments,
  `Compartmental::one_year_segments` (behind the `bayes` feature): each
  segment of a triangle is viewed as `Compartmental::one_year` views a
  single-segment triangle (its own premium column and outstanding cells,
  model 1 moment-matching by default and model 2 reweighting only with the
  warning that it is approximate, at most 2,000 simulations a segment,
  `max_refits` a budget of each segment) on the seeds of `fit_segments`, so
  a one-segment triangle's view is `one_year` on segment 0's streams. The
  segments' claims development results are joined by row into one joint
  result as `Compartmental::dependence` says: `Independent` as drawn,
  `RankCorrelation` by Iman-Conover on their totals (as `BayesOdp` and
  `Meyers` do); `Synchronized` is an error. The result is a
  `CompartmentalOneYearFits` (new): per-segment views, `to_long`, `totals`
  and `segment`. In Python `Compartmental.one_year_segments(triangle, paid,
  premium, *, outstanding=None, incurred=None, k_threshold=0.7, refit=None,
  max_refits=None, n_sims=None)`, returning a `CompartmentalOneYearFits`;
  in R `compartmental_one_year_segments()` with
  `compartmental_one_year()`'s arguments and `dependence` and `spearman`,
  returning a `compartmental_one_year_fits`.
- Development factors averaged over the latest periods and without the
  highest and lowest link ratios: `n_periods`, `drop_high` and `drop_low`
  on `Development` (chainladder-python's names), for the chain ladder, the
  four expected-loss methods and Mack. `n_periods` counts calendar
  diagonals, and an age with fewer link ratios uses them all. The drops
  rank within the kept link ratios and are skipped at an age where they
  would leave none, as Friedland's medial average does. The exclusions
  enter sigma and Mack's standard errors too, at parity with
  chainladder-python 0.10.1 and R ChainLadder's `MackChainLadder(weights
  = W)`. `DevelopmentFit::n_links` gives the number of link ratios behind
  each factor. The bootstraps and the one-year views reject them for now.
  In Python and R as `n_periods`, `drop_high` and `drop_low` (#336, #339, #403).
- A geometric average, `average = "geometric"`, for the chain ladder and
  the expected-loss methods; Mack and the bootstraps reject it, since it
  has no variance. (#403) **Breaking (Rust):** `Average` has a new variant and
  `Average::alpha` returns `Option<f64>`, `None` for it; `Error` has a new
  variant, `NotSupported`; `Development` and `DevelopmentFit` gain public
  fields (`n_periods`, `drop_high`, `drop_low`; `n_links`, `excluded`), so
  struct literals need `..Default::default()` (or the new fields).
- A selected pattern of factors to ultimate (or percents developed) may
  now go below 1 and need not be monotone, for negative development such
  as Friedland's XYZ Insurer reported pattern (#336, #403).
- `MackFit.cdr` is the property form of `claims_development_result()`,
  named as `OneYearFit.cdr`; the method stays as its alias (#214).
- `CapeCod`'s `trended_apriori` is documented at the valuation date's price
  level, so a treaty year needs more trend beyond it (#214).
- `Synchronized` dependence on a Bayesian model says that a posterior has
  no residuals to resample and points to `RankCorrelation` (#393).
- The ODP GLM's analytic prediction error: `OdpGlmFit::prediction_error`
  gives each origin's and the total reserve's process and parameter
  variance by the delta method, matching Taylor and McGuire's Table 5-4
  (CAS Monograph 3) to the unit. `OdpGlm` and `OdpGlmFit` in Python
  (`prospicio.reserving`) and R (`odp_glm()`, with a
  `predict_distribution()` method), single segment (#333, #436).
- The ODP GLM's residual diagnostics: `OdpGlmFit::residuals` gives each
  observed cell's actual, fitted, actual-to-fitted ratio, Pearson and
  deviance residuals, leverage and standardized residuals with its origin,
  age and calendar period, and `OdpResiduals::summary` groups them by
  origin, development or calendar period; Taylor and McGuire's Figure 6-22
  heat map (CAS Monograph 3) reproduces exactly. Python
  `OdpGlmFit.residuals()` and `residual_summary(by)`, R `fit@residuals`
  and `residual_summary(fit, by)` (#333, #438).
- The ODP GLM on a custom design: `OdpGlm::fit_design(triangle, column,
  design)` fits any design over the triangle's cells (one row per row of its
  `TriangleFrame`): an origin effect as a polynomial, development effects as
  a line with a kink, calendar trends, interactions. Taylor and McGuire's
  models 7-2, 7-10 and 7-14 (CAS Monograph 3, Tables 7-1, 7-3 and 7-5)
  reproduce to the printed digits, with their reserves, prediction errors,
  residuals and draws as for the cross-classified model. Python
  `OdpGlm.cells(triangle, column)` and `OdpGlm().fit(triangle, column,
  design=...)`; R `odp_glm_cells()` and `odp_glm(triangle, formula = ~ ...)`
  (#333).
- The ODP GLM with a gamma or Tweedie error: `OdpGlm::family` is
  `Family::Poisson` (the default, the ODP), `Family::Gamma` (variance
  `phi mu^2`, Taylor and McGuire's section 6.5) or `Family::Tweedie {
  power }` with `1 < p < 2`, each with the log link and Pearson's
  dispersion, on the cross-classified model or a custom design. The
  prediction error, residuals and draws follow the family. R
  `glm(family = Gamma(link = "log"))` and statmod's `tweedie(var.power =
  1.5, link.power = 0)` parity on GenIns and ABC. Python
  `OdpGlm(family="gamma")`, `OdpGlm(family="tweedie", power=1.5)`; R
  `odp_glm(triangle, family = "gamma")`, `family = "tweedie", power = 1.5`
  (#333, #445).

### Models

- `prospicio_bayes::elpd::psis(log_ratios, r_eff)`: Pareto smoothed
  importance sampling of any log importance ratios, public. It returns the
  smoothed log weights (unnormalized, the largest raw ratio shifted to 0)
  and the Pareto k, by the smoothing `loo` applies, with the tail length
  `min(ceil(0.2 S), ceil(3 sqrt(S / r_eff)))`; ratios that are all -inf
  (every weight zero) are an error. `loo` and `waic` are
  unchanged (#273).
- Cyclic smooths in `prospicio_glm::gam`: `PSpline::cyclic(column)`,
  with `period(lo, hi)`, is mgcv's `s(x, bs = "cp")`: the fit and its
  slope meet at the ends of the period, and new data outside it wraps
  around (mgcv parity). Python `Gam(glm, [{"column": "month", "bs": "cp",
  "period": (0, 12)}])`, R `gam_fit(..., bs = "cp", period = list(month =
  c(0, 12)))`. A GAM artifact with a cyclic smooth is format version 2;
  one without stays version 1 (#282).
- Monotone smooths in `prospicio_glm::gam`: `PSpline::increasing` and
  `PSpline::decreasing` are scam's `bs = "mpi"` and `"mpd"`, smoothed by
  GCV, UBRE or fixed parameters (scam parity). Python `{"column": "x",
  "bs": "mpi"}`, R `gam_fit(..., bs = "mpi")`. This completes the GAM
  scope of v0.6 (#282).
- Bayesian hierarchical credibility: `prospicio_bayes::credibility::
  HierarchicalCredibility` fits Bühlmann–Straub's model, or Jewell's
  hierarchy of it (groups in cohorts in regions...), as a normal
  random-effects model sampled with NUTS, the means integrated out in
  closed form so only the standard deviations are sampled. The premiums
  are the credibility formula averaged over the structure parameters'
  posterior; `predictive` draws risk premiums or next-period ratios.
  `credibility_premiums` gives the formula at given structure parameters:
  Bühlmann–Straub's and Bühlmann–Gisler's hierarchical premiums, matching
  `actuar::cm` at `1e-10`. Python `prospicio.credibility.hierarchical_credibility`,
  `credibility_premiums`; R `hierarchical_credibility()`,
  `credibility_premiums()`, `credibility_predictive()` (#304).
- Posterior predictive checks in `prospicio_bayes::ppc`: `check` (and
  `check_grouped`, `check_with`) compares test statistics of the data
  (mean, sd, extremes, quantiles, the share of zeros, or any function)
  with their distribution over replicated datasets, with posterior
  predictive p-values; `loo_pit` gives each observation's leave-one-out
  PIT under its Pareto-smoothed importance weights, randomized over ties,
  matching R `loo`'s `E_loo` at `1e-9`. `BayesGlmFit::replicate`,
  `posterior_predictive_check` and `loo_pit`. Python
  `BayesGlmFit.replicate`, `.posterior_predictive_check`, `.loo_pit`,
  `models.ppc_check`, `models.loo_pit`; R `bayes_replicate()`,
  `bayes_ppc()`, `bayes_loo_pit()`, `ppc_check()`, `loo_pit()` (#305).
- ONNX import in `prospicio_nn::onnx` (Rust, behind the default `onnx`
  feature): `OnnxModel::load` parses a network trained elsewhere
  (PyTorch, Keras) with Burn's ONNX parser `onnx-ir` and runs it in
  `f64` for the operators a tabular network exports (`Gemm`, `Linear`,
  `MatMul`, broadcasting arithmetic, the common activations, `Softmax`,
  `Concat`, `Gather` embeddings, reshapes and reductions), refusing
  others at load. `OnnxPredictor` binds graph inputs to design columns,
  reads the output as a mean or on a link's scale with the offset, and
  implements `Fitted` with process draws from a family. Matches
  onnxruntime within float32 rounding (#306).
- Python: the neural networks of `prospicio-nn` in `prospicio.models`:
  `Cann` and `AttentionCann` (fit on a `Design`, `predict`,
  `predict_distribution`, `correction`, early stopping; the attention
  CANN's `attention` gives a `TokenAttention` of per-row, per-head
  weights), and `OnnxModel` and `OnnxPredictor` for networks trained
  elsewhere (onnxruntime parity). The wheel links Burn through the
  bindings' default `nn` feature, which needs Rust 1.92 (#308).
- R: the same networks: `cann_fit()` and `attention_cann_fit()` on a
  formula (`predict()`, `predict_distribution()`, `network_correction()`,
  `attention_weights()`), and `onnx_model()`, `onnx_run()` and
  `onnx_predictor()` for networks trained elsewhere. The package builds
  Burn through the default `nn` feature and now needs Rust 1.92
  (`SystemRequirements`) (#309).
- Fitted networks save and load: `CannFit::to_json` / `from_json` and
  `AttentionCannFit::to_json` / `from_json` write versioned JSON
  artifacts (`risk_rs.cann_fit`, `risk_rs.attention_cann_fit`) with
  every weight by its path in the network, and read them back bit for
  bit; `input_hash` on both. Python `CannFit` and `AttentionCannFit`
  gain `to_json`, `from_json`, `input_hash` and pickling; R
  `save_model()` and `load_model()` take `cann_model` and
  `attention_cann_model`. The shared envelope helpers are now public in
  `prospicio_models::artifact` (#391).
- A GAM's log-likelihood and AIC: `GamFit::log_likelihood` and `aic`
  (`-2 ℓ + 2 edf`, the GLM's conventions), saved in its artifact; Python
  `GamFit.log_likelihood` and `.aic`, R `gam_model@log_likelihood` and
  `@aic` (#360). R gains `deviance_score()` for `cross_validate()` (#360),
  and `pit_from_draws()` and `pit_histogram()` as Python's (#382).
- The negative binomial GLM with its `theta` estimated:
  `prospicio_glm::negbin::NegativeBinomialGlm` alternates the GLM's IRLS
  with Newton's method on `theta` as `MASS::glm.nb` does, with `MASS`'s
  standard error of `theta`, the log-likelihood and an AIC that counts
  `theta` (parity on `quine`). Python `NegativeBinomialGlm` and the
  Tweedie GLM with its power estimated (`TweedieGlm`, `tweedie_profile`);
  R `glm_nb_fit()`, `tweedie_glm_fit()` and `tweedie_profile()` (#352).
- Generalized linear mixed models with random intercepts and slopes:
  `prospicio_glm::glmm::Glmm` fits the Poisson, gamma and Tweedie (fixed
  power) with the log link and one, nested (`Grouping::nested`) or crossed
  grouping factors, each with a random intercept and any random slopes
  (`Grouping::slope`, `without_intercept`) under an unstructured
  covariance, by the Laplace approximation or, for one factor with one
  term, by adaptive Gauss-Hermite quadrature (`Glmm::quadrature`).
  `GlmmFit` gives the fixed effects with their covariance, each term's
  standard deviation, each factor's correlations, every parameter with its
  covariance (glmmTMB's `theta`), the dispersion, the log-likelihood, AIC
  and BIC, and each level's modes with their conditional and prediction
  variances; `predict` gives a level the fit has not seen the population
  mean. At parity with glmmTMB and lme4's `glmer` (#358).
- The GLMMs in Python and R: `prospicio.models.Glmm` and `GlmmFit`, with
  grouping factors as a dict of labels or of `labels`, `slopes`,
  `intercept` and `correlated`, and R `glmm_fit()` with lme4's `(1 | g)`,
  `(1 + x | g)`, `(0 + x | g)`, `(1 + x || g)` and `(1 | a/b)` terms in
  the formula, returning a `glmm_model` with `sd`, `correlations`,
  `random_effects` and `predict()` (#358).
- Bayesian GLMMs by NUTS: `prospicio_bayes::glmm::BayesGlmm` samples the
  same models jointly with their random effects (non-centred, started at
  the Laplace fit), with normal priors on the coefficients, half-normal
  ones on the standard deviations and the dispersion and LKJ priors on the
  correlations. `BayesGlmmFit` gives the parameters' and the effects'
  draws, the summary with R-hat and ESS, each effect's posterior mean and
  sd, the Laplace fit and the posterior mean of new rows' means. In Python
  `prospicio.models.BayesGlmm` and `BayesGlmmFit`, in R
  `bayes_glmm_fit()` with the random terms in the formula (#358).
- Double GLMs: `prospicio_glm::dglm::DoubleGlm` fits a GLM for the mean
  and a gamma GLM with the log link for the dispersion, each on its own
  design, for the Gaussian, Poisson, gamma and Tweedie, by ML (the
  default, as `dglm`) or REML (`.reml()`). `DoubleGlmFit` gives both sets of coefficients
  with their covariances, the fitted means and dispersions, `dglm`'s
  `m2loglik`, the REML criterion and AIC. At parity with R `dglm` where it
  is right; where its mean step is not (a non-canonical link such as the
  gamma's or Tweedie's log, or an offset), with its loop corrected (#353).
- The double GLMs in Python and R: `prospicio.models.DoubleGlm` and
  `DoubleGlmFit`, fitted on a mean and a dispersion `Design`, and R
  `dglm_fit(formula, dformula, data, ...)`, `dglm()`'s order of arguments,
  returning a `dglm_model` whose `predict()` gives means or, with
  `type = "dispersion"`, dispersions (#353).
- Elastic-net coefficient bounds: `ElasticNet::lower_limits` and
  `upper_limits` bound each coefficient on the design's scale, as glmnet's
  `lower.limits` and `upper.limits` (every bound 0 makes a step basis
  monotone). The relaxed lasso: `ElasticNetFit::relax(design, y, gamma)`
  refits the kept columns without the penalty and blends, and
  `ElasticNet::cross_validate_relaxed` chooses `(lambda, gamma)` as
  `cv.glmnet(relax = TRUE)` (glmnet parity). Paths on nearly collinear
  columns, such as hinges at neighbouring knots, converge about fifty
  times faster. An artifact with bounds or a relaxed fit is format
  version 2; one without stays version 1 (#356).
- The elastic net's bounds and relaxed lasso in Python and R: Python
  `ElasticNet(lower_limits=, upper_limits=)`, `ElasticNetFit.relax` and
  `.gamma`, and `ElasticNet.cross_validate_relaxed` with its `RelaxedCv`;
  R `elastic_net_fit(lower_limits, upper_limits, relax = TRUE)`, `coef()`
  and `predict()` with `gamma`, and `elastic_net_cv(relax = TRUE, gamma)`
  (#356).
- Model-agnostic explanations that need only `predict()`: permutation
  importance (single columns or groups shuffled together, with a score's
  increase per feature and its percent of the largest), the pairwise
  interaction screen `L_A + L_B - L_AB`, and partial dependence, on one
  fitted model or averaged over folds like `cross_validate`. Python
  `prospicio.models.permutation_importance`, `interaction_screen` and
  `partial_dependence` on a dict of raw columns and a design builder, with
  `PermutationImportance`, `InteractionScreen` and `PartialDependence`
  (`combine` for folds whose designs differ); R `permutation_importance()`,
  `interaction_screen()` and `partial_dependence()` on a data frame. Both
  shuffle with the new `permutation(n, seed, stream)`
  (`prospicio_models::resample::permutation`), so they give the same values
  on the same rows (#355).
- Randomized quantile residuals (Dunn and Smyth): `GlmFit::quantile_residuals(design, y, seed)` and
  `metrics::quantile_residuals`, the normal scores of the randomized PIT, exact for the Gaussian,
  gamma and inverse Gaussian and drawn by seed for the counts and the Tweedie's zero; `statmod::qresiduals`
  parity on quine. Python `GlmFit.quantile_residuals()` and `models.quantile_residuals()`, R
  `residuals(m, type = "quantile", seed = 0)` and `quantile_residuals()` (#464).
- MARS knot selection for hinge terms: `prospicio_glm::mars::Mars` runs R earth's forward pass with
  `degree = 1` (hinge pairs, earth's minimum and end spans, thresholds and GCV, weights optional) and an
  optional backward pass by GCV; `MarsFit::add_terms` puts the knots into `Terms` as `numeric` and
  `hinges`. With `glm(family, link)` each step scores candidates on the GLM's working response and
  compares deviances. earth parity on the Chalk portfolio: the knots, RSS and GCV of default,
  weighted, `nk = 15` and pruned runs. Python `models.mars()` and `MarsFit`, R `mars_knots()` and
  `mars_formula()` (#465).

### Aggregate and reinsurance

- `Tower::on_grid` sets probabilities below machine epsilon to 0 in its
  ceded and net grids: the FFT round-off there made the exponential and
  Esscher premiums of a stop loss's net loss read the top of the grid
  (509.4 for *Pricing Insurance Risk*'s 50.420). `Grid::remove_fuzz(eps)`,
  Python `Grid.remove_fuzz()`, R `remove_fuzz()` (#367).
- `TowerModel::diagnostics` warnings print each position at six
  significant figures (`the piece from 1.28998`, not every digit of the
  float); Python `TowerModel.diagnostics()` and R `tower_diagnostics()`
  carry the same text (#276).
- Multi-year contract terms: `Tower::apply_term` applies a tower to
  consecutive years of the same simulated paths (one `EventSet` per year;
  `EventSet::term_years` splits one simulation into the years of a term),
  and `apply_aggregate_term` does the same on aggregate losses; the result
  has a `year` dimension. A profit commission can carry its deficits
  forward (`Layer::profit_commission_carry_forward`) and a sliding scale
  its deficits, credits or both (`Layer::sliding_scale_carry_forward`),
  to extinction or for a number of years, used up oldest first;
  `profit_commissions_over` and `ceding_commissions_over` give the
  commissions over consecutive years. Tower documents with a carry are
  format version 2. Python `Layer.with_profit_commission_carry_forward`,
  `with_sliding_scale_carry_forward`, `Tower.apply_term`,
  `apply_aggregate_term`, `EventSet.term_years`; R
  `with_profit_commission_carry_forward()`,
  `with_sliding_scale_carry_forward()`, `apply_tower_term()`,
  `term_years()` (#301).
- The normal power and Wilson–Hilferty approximations of a compound sum
  (Bahnemann, CAS Monograph 2, section 4.4), defined by their quantiles:
  `ApproximationKind::NormalPower` and `WilsonHilferty`; Python
  `CompoundMoments.approximate("normal_power")` and
  `"wilson_hilferty"`, R `moment_approximation(m, "normal_power")` and
  `"wilson_hilferty"` (#329).
- `audit_grid` and `audit_portfolio_grid` (Python
  `prospicio.aggregate.audit_grid`, R `audit_grid()`) audit an FFT grid as
  `aggregate` 1.2.0's `sharpen` does: each cell scored by six moment
  errors against the exact moments, the step doubled and halved around the
  center, the best cell that loses no mass taken, preferring fewer points.
  `GridAudit` keeps every cell and the reason for the choice. With
  `recommend_portfolio_grid`, a grid for several units. `aggregate`'s
  scores and choices on its books, to six digits (#317).
- A layer's annual terms can split the year's recovery pro rata across its events
  (`LimitAllocation::ProRata`, `Layer::allocation`): every event's recovery is scaled by the annual
  recovery after the terms over the one before them, instead of using the limit up in event order.
  The annual total is unchanged; the split by event, and so what an inuring stage sees, differs.
  Refused with reinstatements pro rata as to time; written to a tower's JSON only when set. Python
  `Layer(..., allocation="pro_rata")`, R `xol_layer(..., allocation = "pro_rata")` (#476, #493).
- Labels on simulated events, a peril or a class, carried through a tower (`EventSet::with_labels`,
  `from_long_labeled`, kept by `merge`): with labels `Tower::apply` gives gross, each layer's ceded
  loss and net by label, with the label dimensions after `kind` and `layer` and the annual premiums
  and commissions labelled `""`; under an annual limit the split follows the layer's allocation.
  Unlabelled results are unchanged. Python `EventSet.with_labels`, `from_long(..., labels=)`,
  `labels`, R `with_labels()`, `event_labels()`, `event_set_from_long(..., labels =)` (#475, #493).
- `Tower::apply_annual` applies a tower to plain annual losses, one per simulation, with no
  `EventSet` or `PredictiveDistribution` to build; a contract stated on the loss ratio (a cap and
  a sliding scale on subject premium) is a stop loss with `premium_rate(1, P)` and a sliding
  scale. Python `Tower.apply_annual`, R `apply_tower()` on a numeric vector (#477, #493).
- A vendor's year loss table read, checked and expanded:
  `prospicio_aggregate::ylt::YearLossTable` takes each row's year (from
  1), loss, event slot and labels, checks the losses, the years, repeated
  year and slot within the labels named (`check_unique`) and the expected
  label values (`check_values`), orders the rows by year and slot, and
  expands the years to a run's simulations by tiling (a whole multiple) or
  resampling each simulation from its own stream. The source file's hash
  (`ylt::hash_bytes`) is the events' `EventSet::input_hash` and every
  result's provenance input hash. Python `prospicio.aggregate.read_ylt`
  and R `read_ylt()` read CSV or Parquet (with pyarrow or arrow);
  `EventSet.from_ylt` and `event_set_from_ylt()` take columns (#484, #496).
- The docs say where Panjer and the FFT assume independence (the count
  independent of the severities, i.i.d. severities, independent units on a
  grid) and name the dependent alternatives: `fft_bivariate`, the clash
  model, a mixed count on a severity mixture and `reorder_groups` on
  simulations; in the crate docs, `aggregate.md` and the Python and R help
  for `panjer`, `fft`, `fft_auto` and `compound_distribution()` (#491).
- R: `moment_approximation()` also takes a `compound_moments()` result
  alone, `moment_approximation(m, "gamma")`; calls with a count and a
  severity are unchanged (#327, #348).

### Distributions and probability

- Iman–Conover with other reference scores and shuffled rows:
  `copula::iman_conover_with` takes `ImanConover { scores, shuffle_rows }`,
  with `Scores::StudentT(dof)` (more joint extremes than normal scores at
  the same correlation, Monograph 15 section 8.2.3, `aggregate`'s `dof`)
  or `Scores::Given`, and `shuffle_rows` to undo the first component's
  ascending order, now documented on `iman_conover`. Python
  `iman_conover(..., dof=, scores=, shuffle_rows=)`, R `iman_conover(...,
  dof, scores, shuffle_rows)`. `prospicio_math::special::student_t_quantile`
  (#378).
- Threshold diagnostics for peaks over threshold:
  `prospicio_prob::evt::threshold_diagnostics` gives, at each threshold,
  the mean excess with its standard error and the generalized Pareto
  fitted to the exceedances, with the shape, the modified scale
  `beta - xi * u` and their standard errors from the observed information,
  the numbers behind R `ismev`'s `mrl.plot` and `gpd.fitrange` (parity).
  `Gpd::fit_with_covariance` returns a fit with the covariance of
  `(xi, beta)`. Python `prospicio.risk.threshold_diagnostics` (a dict of
  columns) and R `threshold_diagnostics()` (a data frame) (#282).
- A results table's statistics in one call: `prospicio_prob::statistics::StatisticsGrid`
  gives the mean, SD, CoV, min, max, variance, the percentiles, TVaRs and
  mean plus or minus multiples of the SD under stable names
  (`"Percentile 99.5"`, `"TVaR 99"`, `"Mean + 2 SD"`), of a `Sampled` or of
  each component of a `PredictiveDistribution`. Quantiles by the inverted
  cdf (the default) or linearly as numpy and R `type = 7`
  (`QuantileMethod`), TVaR exact (the default) or as the integral of the
  linear quantile function (`TvarMethod`); `Sampled::var_with` and
  `tvar_with`, `risk::var_linear_sorted` and `tvar_linear_sorted`. Python
  `prospicio.risk.statistics` and `Sampled.var(p, method=)`,
  `.tvar(p, method=)`; R `statistics()` and `VaR(x, p, method = )`,
  `TVaR(x, p, method = )` on a `sampled` (#474, #492).
- A Spearman target for `reorder_groups`: `reorder_groups_with(dim,
  correlation, CorrelationKind::Spearman, seed)` converts the matrix to
  `2 sin(π ρ / 6)` before Iman-Conover, so the achieved rank correlation
  of the group totals meets it (the default stays the normal scores'
  correlation). `correlation_from_pairs` builds the matrix from a long
  table `(group, group, rho)` with every pair once, `groups` lists the
  groups in matrix order and `group_spearman` gives the achieved
  Spearman's rho with ties at their average rank. Python
  `reorder_groups(..., kind="spearman")`, `groups`,
  `correlation_from_pairs`, `group_spearman`; R `reorder_groups(..., kind
  = "spearman")` with a matrix or a three-column data frame, and
  `group_spearman()` (#481, #493).
- Any copula's dependence on groups already simulated:
  `PredictiveDistribution::reorder_groups_by_copula(dim, copula, seed)`
  sorts each group's rows by its total and gives them the ranks of a
  column of the copula's draws, so the totals take the t, Gumbel, Clayton,
  nested or vine copula's rank pattern and joint tails, rows moving whole
  as under `reorder_groups`. Python
  `PredictiveDistribution.reorder_groups_by_copula`, R
  `reorder_groups_by_copula()` (#482, #495).
- Risk ratios on draws: `prospicio_prob::statistics::RiskRatios` gives the
  Sharpe ratio (population SD), the Sortino ratio over the downside
  deviation or, as a reference pricing model computes it, the SD of the
  values below the threshold (`Sortino`), the Omega ratio of partial
  moments or of counts (`Omega`) and the TVaR ratio `(mean - t) /
  TVaR_q(-x)` under `TvarMethod`, about a threshold, of a `Sampled` or
  each component of a `PredictiveDistribution`. Python
  `prospicio.risk.risk_ratios`, R `risk_ratios()` (#486, #495).
- Items read at a driver's percentiles: `StatisticsGrid::co_statistics`
  reads each item on the simulation the driver's mean, percentile or TVaR
  selects (the stable order of the driver, `h = q (n - 1)` rounded by
  `IndexRule`: halves to even as Python's `round`, floor or ceil; the TVaR
  over `o[ceil(h):]`), and `co_components` reads every component of a
  `PredictiveDistribution` at one component's or the total's ranks.
  Python `prospicio.risk.co_statistics`, R `co_statistics()` (#487, #495).
- `PanjerClass` (a count chosen by mean and dispersion) forwards its third
  moment and cumulant, `cdf`, `quantile` and `sample` to its family. The
  third cumulant had fallen to the trait's default, which cancels at large
  means: compound skewness at 1,000 claims was 1.2e-7 off; it is now exact
  (#307).
- TVaR of every parametric severity: `Severity::tvar(p)`, `VaR(p) +
  E[(X - VaR(p))+] / (1 - p)`, infinite where the mean is (closed-form
  parity for the lognormal, Pareto and exponential). Python severities gain
  `var` and `tvar`; R's `VaR()` and `TVaR()` take every parametric severity
  and `VaR()` a `grid_distribution` (#366 item 2, #373, #375; #397).
- The covariance allocation of the margin:
  `AllocationMethod::CovarianceMargin` gives each unit its expected loss
  plus a share of `ρ(S) − E[S]` by covariance with the total (Kalkbrener's
  allocation; *Pricing Insurance Risk* Exercise 217 exact), beside
  `Covariance`, which spreads the whole measure by covariance. Python
  `capital(..., "covariance_margin")`, R
  `capital_allocation(..., "covariance_margin")`.
- The Shapley value of any cost, not only a distortion:
  `PredictiveDistribution::shapley_with(cost)` with a closure of the
  sub-portfolio's draws (a standard deviation premium, an expected
  utility). Python `prospicio.risk.shapley(pd, cost)` with a callable, R
  `shapley_allocation(x, cost)` with a function (#369).
- The haircut and equal risk allocations of an exogenous total (a
  premium, a capital amount) by stand-alone VaR, TVaR or EPD at one level:
  `PredictiveDistribution::haircut(total, measure, level)` scales the
  stand-alone measures at a given level, `equal_risk(total, measure)`
  finds the level at which they add up (*Pricing Insurance Risk*
  Exercises 205 and 206 exact). Python `prospicio.risk.haircut` and
  `equal_risk` (a `LevelAllocation`), R `haircut_allocation()` and
  `equal_risk_allocation()` (#369).
- Nested GLM tests: `prospicio_glm::nested::f_test(small, big)` and
  `likelihood_ratio_test(small, big)`, R `anova(small, big, test = "F")`
  and `test = "Chisq"` (parity on quine); `GlmFit::bic`, counting the
  coefficients as `aic` does; `metrics::vif`, variance inflation factors of
  a design's columns. Python `f_test`, `likelihood_ratio_test`, `vif` and
  `GlmFit.bic`; R `f_test()`, `likelihood_ratio_test()`, `vif()` and
  `glm_model@bic` (#351).
- Charts for comparing and validating models (CAS Monograph 5, chapter 7):
  `metrics::double_lift` (bands by the ratio of two models' predictions,
  relativities to each overall average), `loss_ratio_chart` (bands by
  predicted loss ratio against current premium), `roc` with the area
  under the curve, `confusion_matrix` and `normalized_gini`. Python
  `double_lift`, `loss_ratio_chart`, `roc`, `confusion_matrix`,
  `normalized_gini`; R `double_lift_table()`, `loss_ratio_table()`,
  `roc_curve()`, `confusion_matrix()`, `normalized_gini()` (#351).
- GLM diagnostics and prediction errors: `GlmFit::diagnostics` (response,
  Pearson, deviance and working residuals, working weights, hat values,
  standardized residuals, Cook's distances; R `glm` parity),
  `prediction_se` (as `predict(type = "response", se.fit = TRUE)`),
  `sum_prediction_error` (the RMSEP of a reserve by the delta method plus
  process variance), `gcv` and `log_likelihood_at` a held dispersion.
  Python methods of the same names on `GlmFit`; R `residuals()`,
  `hatvalues()`, `rstandard()` and `cooks.distance()` on a `glm_model`,
  and `glm_diagnostics()`, `prediction_se()`, `sum_prediction_error()`,
  `glm_gcv()`, `log_likelihood_at()` (#334).
- Rating-variable terms (CAS Monographs 5 and 16): `Terms::interaction`
  (numeric and factor in any combination, named as R's `a:b`), `poly` (raw
  powers), `hinges` (`max(0, x - k)`), `steps` (`1[x >= k]`), `bins` (a
  numeric column cut into right-closed bins and coded as a factor, labels
  as R's `cut`), `log` (`ln(x / base)`) and `ordinal` (an ordered factor in
  cumulative coding). A `Coding` rebuilds them on new data. Python
  `Terms.interaction`, `.poly`, `.hinges`, `.steps`, `.bins`, `.log`,
  `.ordinal`; in R they are formula terms (#350, #357).
- Terms that learn from the training data: `Terms::orthogonal_poly`, R's
  `poly(x, degree)` recurrence (parity, on new data too), and
  `natural_spline`, a natural cubic spline in the restricted cubic basis,
  which spans R's `splines::ns`: fitted values and predictions agree with
  `lm(y ~ ns(x, knots))`, coefficients differ. Python
  `Terms.orthogonal_poly` and `.natural_spline` (#350).
- Out-of-fold target encoding of a high-cardinality factor with
  credibility (CAS Monograph 16, chapter 6):
  `prospicio_models::encode::target_encode` gives each training row its
  level's actual-to-expected ratio from the other folds, shrunk to 1 by
  `z = n / (n + K)`, and a `TargetEncoding` from all rows for new data.
  Python `target_encode` and `TargetEncoding`; R `target_encode()` and
  `encode_target()` (#357).
- The worst VaR of a sum with unknown dependence by the Rearrangement
  Algorithm (Puccetti and Rüschendorf): `rearrangement::rearrange` on any
  columns and `worst_var` on `n` quantiles of each marginal, from a start
  given as is, rotated or shuffled by seed, so a result is reproducible
  (*Pricing Insurance Risk* Exercise 59 as its guide prints it). Python
  `prospicio.risk.rearrange` and `worst_var`, R `rearrange()` and
  `worst_var()` (#366).
- The third moment of a layer of any severity,
  `Severity::layer_third_moment(limit, attachment)`, by adaptive
  quadrature of the survival function (the limited third moment at
  attachment 0, the unlimited layer at an infinite limit); Python
  `layer_third_moment` on every severity, R `layer_third_moment()` (#329).
- A normal distribution, `prospicio_prob::Normal`, on the whole real line,
  with its closed-form TVaR `mu + sigma phi(z_p) / (1 - p)` (*Pricing
  Insurance Risk*, section 4.3.3), and the ogive of grouped data,
  `prospicio_prob::Ogive` (`new` from tabulated `(x, F)`, `from_counts`
  from counts per band), a severity whose limited moments, VaR and TVaR
  are exact band by band (Exercise 50). Python
  `prospicio.distributions.Normal` and `Ogive`, R `normal_distribution()`,
  `ogive()` and `ogive_from_counts()` (#366).
- A severity scaled and shifted, `prospicio_prob::Affine` (`scale * X +
  shift`, `aggregate`'s `10 * lognorm 1 + 20`), whose layer moments, LEV
  and stop loss stay exact from the inner severity's, saved as JSON with
  it. Python `prospicio.distributions.Affine`, R `affine_distribution()`
  (#318).
- `fit_severity` fits the Lomax, `S(x) = (1 + x / scale)^(-alpha)`
  (Bahnemann's Pareto, CAS Monograph 2), as a Burr with `gamma = 1`:
  `Family::Lomax`, Python `fit_severity("lomax", ...)`, R
  `severity_fit("lomax", ...)` (#328).
- Losses truncated from above: `LargeLosses::truncated_above(u)` conditions
  each loss on `r < X <= u` (claims kept only up to a bound), which
  `fit_severity` uses for every family, a censored loss included;
  `goodness_of_fit` conditions both sides on a bound common to every loss
  and refuses bounds that differ. The Pareto fits refuse such data rather
  than ignore it. Python and R `truncated_above` on `fit_severity` /
  `severity_fit` and `goodness_of_fit` (#328).
- Grouped losses: `prospicio_prob::grouped_fit::fit_grouped` fits the
  lognormal, gamma, Weibull, Burr or Lomax to counts in size bands by
  maximum likelihood or minimum chi-square, conditioned on the range of the
  bands (a deductible's truncation, a truncation from above), with
  standard errors from the grouped Fisher information and the chi-square
  test; `fit_lev_distance` fits by the minimum distance to empirical
  limited expected values (Bahnemann, CAS Monograph 2). Python
  `fit_grouped`, `fit_lev_distance`, `grouped_goodness_of_fit`; R
  `grouped_fit()`, `lev_distance_fit()`, `grouped_goodness_of_fit()`
  (#328).
- `Distortion::complement(s)`, `1 - g(s)` computed from `1 - s` so that it
  keeps its relative accuracy as `s` approaches 1 (#330).
- `distortion::calibrate` merges equal values before it solves, so a
  portfolio's totals capped at the assets count once: calibrating Wang on
  *Pricing Insurance Risk*'s 2^19-point Hurricane/SCS portfolio takes
  0.25 s, not about 9 s (#331).
- `SizeOfLoss` (Python `prospicio.distributions.SizeOfLoss`, R
  `size_of_loss()`), a severity from claim counts and loss totals by size
  band, with each band's claims at its average so the limited expected
  value is exact at the band edges. `ilf`, `loss_elimination_ratio` and the
  layer functions take it, and it saves as JSON: Werner and Modlin's
  LAS(100K) 77,046, ILF 1.59 and LER 0.135 (#338).
- The beta and Sichel mixings of mixed Poisson counts, `aggregate`'s
  `mixed beta`, `sichel`, `sichel.gamma` and `sichel.ig`:
  `Mixing::Beta { cv, max }` (the Poisson mean `max` times a beta
  variable), `Mixing::Sichel { cv, shape }` (a generalized inverse
  Gaussian of the given shape) and `Mixing::SichelSkewness { cv, skewness }`
  (the shape solved for a skewness), with `Mixing::sichel_gamma(cv, f)`
  and `Mixing::sichel_ig(cv, f)`; `MixedPoisson::gig` gives the solved
  parameters and `Mixing::cv` the coefficient of variation. At parity with
  `aggregate` 1.2.0 for the Sichel and `sichel.ig`, and with quadrature
  and SciPy where `aggregate` is wrong (its beta probabilities are off by
  up to 8e-9, and `sichel.gamma 0.4 0.3` solves to a count of mean 14.76,
  not 10). `prospicio_math::special::ln_bessel_k`, `ln K_nu(x)` for real
  order. Python `Count.mixed_poisson(..., mixing="beta", max=)`,
  `mixing="sichel"` with `shape=` or `skewness=`, `"sichel_gamma"` and
  `"sichel_ig"` with `f=`, and `Count.gig()`; R `mixed_poisson_count()`
  with the same arguments and `gig_parameters()`.
- `PredictiveDistribution::scale_to_totals` rescales each simulation's components pro rata to a new
  total (factor 1 when they sum to 0): a loss-ratio cap over several classes or a stop loss on a mix
  of perils carried back onto its components. Inside a tower, a cover on the annual total with
  `LimitAllocation::ProRata` on labelled events does the same. Python
  `PredictiveDistribution.scale_to_totals`, R `scale_to_totals()` (#478, #493).
- R: `survival()`, `layer_variance()` and `layer_second_moment()` on
  `lognormal()`; a new generic `layer_second_moment(dist, limit, attachment)`
  on every family with `layer_variance()` (the Pareto family, gamma, Tweedie,
  Weibull, log-logistic, inverse gamma, inverse Gaussian, Burr, beta,
  truncated, policy limits, mixture, custom); `std_dev()`, the standard
  deviation of anything with a `variance()` (#327, #270, #348).
- Python: `Lognormal.layer_second_moment(limit, attachment)` (#327, #348).

### Pricing

- Log-log interpolation of tabulated exposure curves:
  `TabulatedCurve::with_interpolation(x, g, Interpolation::LogLog)` runs
  the curve as a power law between points (the first piece stays
  straight) and refuses a table it would make non-concave, naming the
  point where the elasticity rises. Linear stays the default. Python
  `TabulatedCurve(x, g, interpolation="log-log")`, R
  `tabulated_curve(x, g, interpolation = "log-log")` (#282).
- R `pentagon()` returns the risk discount `d = M / (a - L)` too, as
  Python's `Pentagon.discount` (#365). `Portfolio::price` (Python
  `Portfolio.price`, R `natural_price()`) documents its two conventions,
  both `aggregate` 1.0.1's: a total exactly at the assets keeps its own
  weight in the linear allocation, and the bottom layer of capital takes
  the units' margins at `g'(1) / (1 - g'(1))` (#368, #379; #398).
- Casualty exposure rating in `prospicio_pricing::casualty`: a
  `CasualtyProfile` of policies (attachment, limit, share, premium, ELR,
  and a curve each: a severity, an `IlfTable` or an `ElfTable`), rated
  for a per-policy treaty in the ILF form (`rate`) or the frequency form
  (`rate_by_frequency`), with ALAE pro rata, on top of the capped loss,
  or in the severity (`Alae`), kept policy by policy so a tagged segment
  can be left out. Clark's workers compensation example reproduces
  exactly (13,695, 3.42%) (#312).
- Casualty exposure checks: `CasualtyProfile::occurrences_above(t)` (each
  policy's counts above a threshold from its own curve), `sweep` (the
  frequency form at each alpha of a Pareto from the threshold),
  `ParetoAnchor::fit` (the censored Pareto fit to the trended claims, its
  standard error and the exceedances it rests on), `expected_claims` and
  `count_test` (expected against observed claims at a few sizes, with
  the one-sided Poisson probability, `poisson_tail`), and
  `Prepared::without(ids)` and `Prepared::trended` for the burning cost
  with and without a segment (#390).
- Casualty exposure rating in Python and R. Python
  `prospicio.pricing.CasualtyProfile` (`rate`, `rate_by_frequency`,
  `exposure_factors`, `occurrences_above`, `sweep`, `expected_claims`,
  `count_test`), `IlfTable`, `ElfTable`, `CasualtyRating` (`rows`,
  `without`, `by_tag`), `ParetoAnchor.fit`, `poisson_tail`, and
  `PreparedLosses.trended` and `.without`. R `casualty_profile()`,
  `ilf_table()`, `elf_table()`, `casualty_rate()`,
  `casualty_rate_by_frequency()`, `casualty_occurrences()`,
  `casualty_sweep()`, `casualty_expected_claims()`,
  `casualty_count_test()`, `pareto_anchor()` and `poisson_tail()`. The
  ALAE treatment is a required argument (#396).
- The book's conventions for the natural allocation as settings:
  `Portfolio::price_with(g, a, PriceSettings { allocation, bottom_layer,
  atom })`. `AtomAtAssets::Lumped` puts an outcome whose total is exactly
  the assets with the defaults in the linear allocation, as *Pricing
  Insurance Risk* does (Example 235's 4.3 and 5.7, #368);
  `BottomLayer::Zero` gives the layer the loss always reaches no capital,
  Monograph 15's equation 8.5 (#379). The defaults stay `aggregate`'s,
  which 1.2.0 leaves unchanged from 1.0.1. Python `Portfolio.price(...,
  bottom_layer="none", atom="lumped")`, R `natural_price(..., bottom_layer
  = "none", atom = "lumped")`.
- The natural allocation's capital by unit no longer moves with the last
  bit of the distortion's parameter: each layer's capital per unit of
  margin, `(1 - g) / (g - S)`, is computed from `1 - S`, where it cancelled
  near `S = 1`. Under the CCoC each unit's capital is now its margin over
  `r` to rounding; on *Pricing Insurance Risk*'s gross Cat/Non-Cat case the
  lifted Cat capital is 197.62767 at `r = 0.1` and at the calibrated `r`
  (197.62748 or 197.62785 before, #330).
- `Portfolio::premium_bounds` (Python `Portfolio.premium_bounds`, R
  `premium_bounds()`) runs in about the time of one pricing: each knot's
  TVaR prices come from prefix sums and the best BiTVaR for each unit from
  a convex hull, with the same bounds as the search over every pair. On
  *Pricing Insurance Risk*'s Cat/Non-Cat grid (71,795 totals) it takes
  0.05 s; it did not finish in two minutes (#368).
- `Portfolio::allocation_set` (Python `Portfolio.allocation_set`, R
  `allocation_set()`) gives each unit's smallest and largest premium over
  the natural allocation set, every order of the outcomes whose totals
  capped at the assets tie; *Pricing Insurance Risk*'s Example 259 (6.2009
  to 6.2085) and Exercise 260 exact. A portfolio built from rows only
  (#368).
- `Portfolio::bodoff_price` (Python `Portfolio.bodoff_price`, R
  `bodoff_price()`) prices Bodoff's percentile layers by any distortion:
  each layer's cost and capital are shared in Bodoff's proportions, and the
  premiums add up to the price of `X ∧ a`; Bodoff's Thought Experiment 1
  under the CCoC gives the book's 24.8729 and 5.8635 (#368).
- `layer::censored_las` (Python `prospicio.pricing.censored_las`, R
  `censored_las()`) estimates the limited average severity at each policy
  limit from claims censored at their limits, layer by layer, one
  `SizeOfLoss` table per limit: Werner and Modlin's Table 11.6 layers
  77,046, 44,575 and 15,213 (#338).
- Return metrics on simulated cash flows (`prospicio_pricing::returns`): the NPV at an annual
  rate, the IRR searched on a grid with a flag for whether a root was found and the number of
  roots seen (the one nearest 0 kept), monthly or actual/365 offsets, MOIC, leverage of premium
  on capital and a fee tiered on the margin. Python `npv`, `irr` (`IrrResult`), `moic`,
  `leverage`, `tiered_fee` in `prospicio.pricing`; R `npv()`, `irr()`, `moic()`, `leverage()`,
  `tiered_fee()` (#480, #494).
- R: `portfolio_kappa(port, unit)` and `portfolio_assets(port, p)`, a
  `capital_portfolio`'s conditional expectations by unit and its assets at
  a capital standard, as Python's `Portfolio.kappa(unit)` and
  `Portfolio.assets(p)` (#322, #347).

### Excel

- Aggregate covers in the Excel add-in: `PRO.AGG.COVER`, a stop loss or
  any annual limit xs retention on the year's total, ground up, net of
  per-occurrence layers or on their ceded total, with the chances of
  attaching and exhausting, the ceded moments and the loss on line (at
  parity with `aggregate`'s `reins_stats_df`) (#404).
- Aggregate distributions in the Excel add-in: `PRO.AGG.STATS`, the
  annual total of a count (by mean and dispersion) and a severity, ground
  up or to a per-occurrence layer, with its moments, VaR and TVaR on an
  FFT grid sized automatically or by `step` and `points`, and
  `PRO.AGG.LAYERS`, per-occurrence layer statistics one column per layer;
  at parity with `aggregate`'s `reins_stats_df` and compound moments (#302).
- `PRO.BAYES.ODP`'s `prior_sd` help and the README warn that a tight
  prior lifts the late development columns toward the first column's
  level and raises the reserve (#407).
- `PRO.BAYES.ODP` in the Excel add-in: the Bayesian ODP chain ladder
  sampled by MCMC (NUTS, four chains on worker threads by default), with
  three views of one fit: the reserve exhibit, the parameters' posterior
  with R-hat and ESS, and the sampler's diagnostics and warnings. Recent
  fits are kept, so a sheet showing all three samples once. At parity
  with the same model in Stan on RAA and GenIns (#310).
- `PRO.BURNINGCOST` in the Excel add-in: experience rating from large
  losses, each accident year's trended losses through one or more layers
  (with annual aggregate deductibles and limits), developed by the year's
  excess factor and divided by its on-level premium, with the total and
  the mean loss cost; a reporting threshold drops losses the older years
  would not have reported. At parity with Clark's per-risk example (#300).
- `PRO.CLARK` and `PRO.CLARK.PARAMS` in the Excel add-in: Clark's growth
  curves, the LDF method or Cape Cod with an exposure column, log-logistic
  or Weibull, with the reserve's process, parameter and total standard
  errors and the fitted parameters; at parity with R ChainLadder's
  `ClarkLDF` and `ClarkCapeCod` (#290).
- Copulas: `PRO.COPULA.CDF` (a Gaussian or Archimedean copula's distribution function at points), `PRO.COPULA.SIM` (uniforms simulated from a Gaussian, t or Archimedean copula, reproducible by seed and drawn as the Python and R packages draw them) and `PRO.IMAN.CONOVER` (simulated columns reordered to a rank correlation, normal or t scores); mpmath parity.
- Classical credibility in the Excel add-in: the limited fluctuation
  standards (`PRO.CRED.STANDARD`, frequency, severity or pure premium, a
  column of probabilities against a row of tolerances giving the table)
  and the square root rule (`PRO.CRED.Z`), and the Bühlmann–Straub model
  (`PRO.BUHLMANN`, `PRO.BUHLMANN.PARAMS`) with the unbiased or iterative
  estimator and blank cells as missing; at parity with R's `actuar` (#297).
- `PRO.DISTORTION`, `PRO.DISTORTION.PRICE` and `PRO.TOWER.PRICE` take the distortions with several parameters, given in one range: the bi-TVaR (`bitvar`: `p0, p1, w`), the beta (`a, b`) and the weighted TVaR (`wtdtvar`: levels and weights as two columns); `PRO.TOWER.PRICE` also takes the capped families, with `r0` and the parameter in one range (`aggregate` parity).
- Distortions in the Excel add-in: `PRO.DISTORTION` (g(s) for
  `aggregate`'s families), `PRO.DISTORTION.PRICE` (the risk measure of
  outcomes or simulated draws) and `PRO.DISTORTION.CALIBRATE` (each
  family's parameter that gives a target premium), at `aggregate` parity.
- The Excel add-in is built and smoke-tested by CI (`.github/workflows/xll.yml`)
  and kept as the `prospicio-xll` artifact with the test workbook; a version
  tag attaches it to a draft GitHub release. It is signed when a
  code-signing certificate is set as a repository secret (#284).
- Extreme value theory in the Excel add-in: `PRO.EVT.THRESHOLDS` (the
  mean excess and the GPD's shape and modified scale at each threshold,
  with standard errors, R `ismev` parity), `PRO.EVT.FIT` (the GPD fitted
  above a threshold with its VaR and TVaR) and `PRO.EVT.HILL` (Hill's tail
  index and Pareto alpha).
- Link-ratio exclusions and the geometric average: the `average` argument of the chain-ladder, Mack, expected-loss and Mack bootstrap functions takes `"geometric"` or a row `kind, n_periods, drop_high, drop_low` (such as `{"volume",5,1,1}`, the latest five calendar periods with the high and low link ratio left out), and `PRO.ODP`, `PRO.ODP.SIMS` and `PRO.ODP.CDR` take the same row as a last argument (chainladder-python and R ChainLadder parity).
- The expected-loss methods in the Excel add-in: `PRO.EL`, `PRO.BF`,
  `PRO.BENKTANDER` and `PRO.CAPECOD`, on a triangle with a premium
  (exposure) column, each an exhibit with the a priori loss ratio,
  ultimate and reserve per origin and a total; at parity with
  chainladder-python on GenIns (#287).
- Exposure curves in the Excel add-in: `PRO.EXPOSURE.CURVE`,
  `PRO.EXPOSURE.SHARE` (limits, attachments and MPLs broadcast, so a risk
  profile's column of sums insured gives each band's share of a layer),
  `PRO.EXPOSURE.QUANTILE` and `PRO.EXPOSURE.STATS` on Swiss Re, MBBEFD and
  tabulated curves, and the Swiss Re `c` of a mean rate or of a layer's
  share (`PRO.SWISSRE.C`, `PRO.SWISSRE.C.LAYER`); at parity with mpmath's
  quadrature of the MBBEFD curves (#295).
- The rate indication in the Excel add-in: the loss ratio and pure
  premium methods' experience (`PRO.LR.EXPERIENCE`, `PRO.PP.EXPERIENCE`),
  the indicated change and rate (`PRO.INDICATED.CHANGE`,
  `PRO.INDICATED.RATE`), the permissible loss ratio, the trended present
  rates complement, a credibility-weighted estimate and the premium-based
  expense split (`PRO.PERMISSIBLE.LR`, `PRO.TRENDED.PRESENT`,
  `PRO.CRED.WEIGHTED`, `PRO.EXPENSES`); Werner and Modlin's Appendices A
  to C reproduced end to end (#299).
- Layer pricing in the Excel add-in: `PRO.LEV`, `PRO.LAYER`, `PRO.LAYER.CV`
  and `PRO.ILF` on a Pareto (optionally truncated) or lognormal severity.
  Limits and attachments can be ranges; a column of attachments against a
  row of limits gives a layer table from one formula (#281).
- `PRO.MACK.BOOT` and `PRO.MACK.BOOT.SIMS` in the Excel add-in: Mack's
  bootstrap (England, Verrall and Wuthrich) as an exhibit laid out like
  `PRO.ODP`'s and as its draws, with gamma, lognormal, normal or residual
  process error, or parameter error only (#286).
- Options on the Excel add-in's chain-ladder functions and `PRO.MACK`:
  selected factors (`selected`, `kind` and `weight`: age-to-age factors,
  factors to ultimate or percents developed, blended with the estimates)
  and Mack's rule for the last sigma (`sigma`, also on `PRO.MACK.BOOT`);
  appended as new last arguments, so saved formulas are unchanged (#288).
- The occurrence and aggregate PML curves in the Excel add-in:
  `PRO.AGG.PML`, by return period, the loss the year's largest occurrence
  exceeds (any Panjer count, at parity with `aggregate`'s `oep`) beside
  the annual total's VaR on the FFT grid.
- `PRO.ODP` in the Excel add-in: the ODP bootstrap exhibit (R's
  `BootChainLadder`), with the chain-ladder reserve and the simulated
  mean, SD, CV and percentiles per origin and in total, reproducible by
  seed, with gamma process error or parameter error only (#283).
- The ODP GLM on a sheet: `PRO.ODP.GLM` (reserves with their analytic process and parameter standard errors, RMSEP and CV, Taylor and McGuire's Table 5-4), `PRO.ODP.GLM.PARAMS` (coefficients, standard errors and dispersion) and `PRO.ODP.GLM.RESIDUALS` (residuals and leverage by cell, or the standardized residuals grouped by origin, development or calendar period); R `glm` quasi-Poisson parity on GenIns and ABC.
- `PRO.ODP.SIMS` in the Excel add-in: the ODP bootstrap's simulated
  reserves, one row per simulation with a column per origin and the total,
  the same draws `PRO.ODP` summarizes for the same seed (at most 100,000)
  (#285).
- The one-year view in the Excel add-in: `PRO.MACK.CDR` and
  `PRO.MACK.CDR.YEARS` (Merz and Wuthrich's CDR standard errors, at parity
  with R ChainLadder's `CDR()`), and `PRO.ODP.CDR` and `PRO.MACK.BOOT.CDR`
  (the simulated claims development result of both bootstraps, re-reserving
  with the chain ladder) (#289).
- The Pareto helpers in the Excel add-in: `PRO.PARETO.EXTRAPOLATE` (from
  one layer to a table of others), `PRO.PARETO.ALPHA`,
  `PRO.PARETO.ALPHA.FREQLAYER` and `PRO.PARETO.ALPHA.FREQS`, optionally
  truncated; at parity with the R package Pareto (#293).
- The piecewise Pareto (`"piecewise_pareto"`, thresholds and alphas as
  ranges, truncated on the last piece or the whole distribution) and
  Riegel's generalized Pareto (`"gen_pareto"`) as severity families in the
  Excel add-in, and `PRO.LAYER.VAR`, a layer's variance; at parity with
  the R package Pareto (#294).
- A prototype Excel add-in, `prospicio.xll`, for 64-bit Excel on Windows
  (`crates/prospicio-xll`, `docs/design/xll.md`): the chain ladder and
  Mack's model as worksheet functions (`PRO.CL`, `PRO.CL.LDF`,
  `PRO.CL.CDF`, `PRO.CL.ULTIMATE`, `PRO.CL.RESERVE`, `PRO.CL.FACTORS`,
  `PRO.MACK`, `PRO.VERSION`), built with `cargo xtask xll`. Not yet
  distributed; unsigned (#278).
- A cover's quote in the Excel add-in: `PRO.QUOTE`, the technical premium
  of a risk premium grossed up for expenses, brokerage and a fixed
  expense, its build-up, capital and return, the loss ratio, multiple,
  rate on line and payback, and the same cover at quoted premiums.
- Rate level and trend in the Excel add-in: on-level factors and the
  parallelogram's cells (`PRO.ONLEVEL`, `PRO.RATELEVEL`, from changes or
  printed levels, written or in-force), average written and earned dates
  for trend lengths (`PRO.AVGDATE`), exponential and linear trend fits
  (`PRO.TREND.FIT`) and dates on the half-month grid
  (`PRO.YEARFRACTION`); every date argument takes an Excel date or a year
  fraction. At parity with Werner and Modlin's chapters 5 and 6 and
  Appendices A and C (#298).
- `PRO.PROFILE` in the Excel add-in: an exposure-rated risk profile, each
  band's mean sum insured, expected claims, surplus cession and loss to a
  per-risk layer with a total, on one curve or a Swiss Re `c` per band,
  with sums insured spread between bounds (uniform, tilted or auto) and a
  surplus treaty underneath; it matches `RiskProfile` in Rust, Python and
  R (#296).
- Every parametric severity family in the Excel add-in: the gamma,
  Weibull, log-logistic, inverse gamma, inverse Gaussian, Burr and
  generalized Pareto join the Pareto and lognormal in `PRO.LEV`,
  `PRO.LAYER`, `PRO.LAYER.CV` and `PRO.ILF`, with new `PRO.SEV.CDF`,
  `PRO.SEV.SURVIVAL`, `PRO.SEV.QUANTILE`, `PRO.SEV.STATS` and `PRO.LER`.
  The layer functions' last argument, formerly `truncation`, is now `p3`
  (the same for the Pareto; the Burr's scale) (#292).
- Severity fits in the Excel add-in: `PRO.SEV.FIT` fits the lognormal,
  gamma, Weibull and Burr by maximum likelihood to a column of large
  losses with reporting thresholds, policy-limit censoring and weights,
  and the Pareto, piecewise Pareto and Riegel's generalized Pareto by the
  R package Pareto's estimators; the parameters come out in the order the
  severity functions take them, with standard errors, AIC and BIC, P-P and
  Q-Q against the product-limit estimate, and the body families compared
  (#413).
- The newer severity fits: the Lomax (`"lomax"`, alpha and scale) in `PRO.SEV.FIT` and every severity function, truncation from above in `PRO.SEV.FIT` (a last `upper` argument), `PRO.SEV.FIT.GROUPED` (a body family fitted to counts in size bands by maximum likelihood or minimum chi-square, with the chi-square test) and `PRO.SEV.FIT.LEV` (fitted by the minimum distance to empirical limited expected values); Bahnemann parity. `PRO.SEV.FIT(..., "compare")` now has a fifth row, the Lomax.
- Multi-year contract terms: `PRO.PROFIT.COMMISSION.TERM` and `PRO.SLIDING.SCALE.TERM` give each year's commission over a term, one term a row of ceded losses, with profit commission deficits and a sliding scale's deficits and credits carried forward, used oldest first, for a number of years or until used up.
- Contract terms in the Excel add-in: `PRO.SWING` (swing-rated and retro
  premiums), `PRO.SLIDING.SCALE` and `PRO.PROFIT.COMMISSION` at a range of
  losses, and `PRO.LAYER.TERMS`, a layer's premium, reinstatement premium
  and ceding and profit commissions in expectation over its annual loss,
  at `aggregate` parity.
- The time value of money in the Excel add-in: `PRO.PAYOUT.PATTERN`,
  `PRO.PAYOUT.STATS` and `PRO.PAYOUT.PRICE`, a payment pattern from
  shares, the share paid to date or a chain ladder's factors to ultimate,
  its discount factor, duration and discounted capital years, and a
  cost-of-capital price restated over the payout.
- Towers priced on a grid in the Excel add-in: `PRO.TOWER.PRICE`, the
  gross, each per-occurrence layer (with annual deductibles and limits),
  the layers' total, a stop loss on the net and the net, each priced on
  its own by a distortion (CCoC, PH, Wang, dual, TVaR, exponential), at
  parity with `aggregate`'s `reins_price_df` (#406).
- Triangle helpers in the Excel add-in: `PRO.TRI.INCREMENTAL`,
  `PRO.TRI.CUMULATIVE`, `PRO.TRI.LINKRATIOS` and `PRO.TRI.LATEST` (#291).

### Examples

- R versions of `one_year_view` and `reinsurance_pricing` as Quarto documents
  (`examples/*.qmd`), with the same numbers as the notebooks.
- `examples/reading_path.ipynb` and its R twin `reading_path.qmd`: a reading path
  through the guides, one small synthetic book carried from severity and claim
  count through the aggregate, a layer price, a rate indication, reserving and
  capital allocation, each station naming the guides to read next (#387).

### Guides

- `docs/guides/prospicio-for-aggregate-users.qmd`, the R twin of the guide for
  users of Mildenhall's `aggregate`: its twenty-minute tour run with the R
  package, every number equal to the Python guide's. CI's R job runs its
  chunks and checks every printed number (`R/prospicio/tests/test-guides.R`) (#314).
- New guide `prospicio-for-aggregate-users.md`: the twenty-minute tour of Mildenhall's `aggregate` package run in prospicio, with both libraries' numbers side by side, a DecL-to-prospicio table and the gaps.
- R twins of the two guides reproducing CAS Monograph 2, Bahnemann's
  *Distributions for Actuaries*: `distributions-for-actuaries-claim-size.qmd`
  and `distributions-for-actuaries-counts-aggregates.qmd`, every number equal
  to the Python guides', checked by `R/prospicio/tests/test-guides.R` (#325).
- Two guides reproducing CAS Monograph 2, Bahnemann's *Distributions for
  Actuaries*, with the monograph's printed numbers beside prospicio's:
  `distributions-for-actuaries-claim-size.md` (limited expected values, fits
  to grouped, censored and truncated data, excess severity and layers, the
  leveraging effect of inflation, increased limit factors with ALAE and risk
  loads, deductibles) and `distributions-for-actuaries-counts-aggregates.md`
  (Poisson, mixed Poisson and negative binomial counts, contagion, moment
  approximations against Panjer, the FFT and simulation, aggregate limits)
  (#324).
- R twin of the compartmental guide, `reserving-compartmental.qmd`: Gesmann
  and Morris' models 1 and 2 on company 337 with `compartmental()`,
  `compartmental_model_2()`, `compartmental_prior_predictive()`,
  `bayes_loo()` and the one-year views, every checked chunk printing the
  Python guide's text (#384).
- New guide `reserving-compartmental.md` on Gesmann and Morris' hierarchical
  compartmental reserving models 1 and 2 on company 337: paid and
  outstanding as two compartments, the prior predictive check, both fits
  with the paper's printed posteriors beside prospicio's (model 1 checked in
  CI, model 2 recorded), the fitted development and test diagonal, the
  reserve against the outcome, why their elpd_loo values do not compare,
  and the one-year view of both models (#381).
- `docs/guides/glm-ratemaking-goldburd.qmd` and
  `docs/guides/glm-ratemaking-goldburd-validation.qmd`, the R twins of the
  guides to Goldburd, Khare, Tevet and Guller's GLM monograph (CAS Monograph 5):
  every number equal to the Python guides'. CI's R job runs their chunks and
  checks every printed number (`R/prospicio/tests/test-guides.R`). The R gaps
  found are filed as #360.
- Two guides following Goldburd, Khare, Tevet and Guller's *Generalized Linear
  Models for Insurance Rating* (CAS Monograph 5) with `prospicio.models`:
  `glm-ratemaking-goldburd.md` (the GLM as a multiplicative rating plan, base
  levels, offsets and weights, frequency, severity and Tweedie families, the
  F-test, AIC and BIC, what prospicio does not support) and
  `glm-ratemaking-goldburd-validation.md` (continuous variables, interactions,
  residuals, holdout lift, double lift and loss ratio charts, Gini and ROC,
  stability and the elastic net), with the monograph's printed arithmetic
  beside prospicio's and the workflow run on `validation/data/glm_policies.csv` (#354).
- R twins of the two CAS Monograph 16 guides (Chalk, Deacon, Guillen and
  Martinelli, *From GLMs to Comprehensive Insurance Pricing*),
  `glm-to-comprehensive-pricing-chalk.qmd` and
  `glm-to-comprehensive-pricing-chalk-2.qmd`, with the same numbers as the
  Python guides (the gradient boosting ones included), run by
  `R/prospicio/tests/test-guides.R`. The Python guide's simulated portfolio is
  shipped as `examples/data/chalk_portfolio.csv` and `chalk_toys.csv`, written
  by `examples/data/make_chalk_portfolio.py`, so R reads numpy's draws.
- Two guides to CAS Monograph 16 (Chalk, Deacon, Guillen and Martinelli, *From GLMs to Comprehensive Insurance Pricing*): `docs/guides/glm-to-comprehensive-pricing-chalk.md` (chapters 3 to 6: pseudo-R2, penalized regression, step functions and GAMs, target encoding) and `docs/guides/glm-to-comprehensive-pricing-chalk-2.md` (chapters 7 to 10: correlated features, two-stage models, learning from a boosting model), with every printed worked example recomputed (#359).
- The docs site's front page links only the Python, R and Rust sites; the guides are in the Python site's User Guide and the R site's Articles.
- R twins of the CAS Monograph 15 chapter 8 and chapters 1, 2 and 7 guides,
  `capital-modeling-monograph-15-iman-conover.qmd` and
  `capital-modeling-monograph-15-practice.qmd`, with the same numbers as the
  Python guides, run by `R/prospicio/tests/test-guides.R`; the chapters 3 and
  4 R guide's "leaves out" list now points to them. R has no
  `pit_from_draws()` or `pit_histogram()` yet, so the practice guide writes
  them out (#382).
- `docs/guides/capital-modeling-monograph-15-iman-conover.md`: CAS Monograph
  15, chapter 8. Parameter uncertainty, Iman-Conover Examples 1 and 2 with
  the achieved correlations against sampling ranges, the algorithm by hand
  with normal and t scores, Iman-Conover against the normal copula, block
  Iman-Conover by `reorder_groups` (Table 8.4 exactly), capital by layer
  (Tables 8.8 to 8.10 exactly; three conventions for Table 8.11's CCoC and
  PH) and Bodoff's allocation (Table 8.12 exactly); gaps filed as #378 and
  #379 (#380).
- `docs/guides/capital-modeling-monograph-15-practice.md`: CAS Monograph 15,
  chapters 1, 2 and 7. The parts of a capital model mapped to prospicio,
  InsCo's Table 2.1 and funding equation, loss ratio, leverage and return by
  unit, reinsurance as a negative unit, the capital structure and the pro
  formas of Tables 2.2 to 2.4 for every outcome, back testing by PIT, and the
  platform checklist of section 7.3 answered for prospicio (#380).
- `docs/guides/capital-modeling-monograph-15.qmd`, the R twin of the guide
  that reproduces CAS Monograph 15's chapter 9 (InsCo's five calibrated
  distortions, the natural allocation, the 35 xs 65 stop loss and the
  correlated Auto, GL and WC book): every number equal to the Python guide's,
  the seeded Iman-Conover samples included. CI's R job runs its chunks and
  checks every printed number (`R/prospicio/tests/test-guides.R`) (#321).
- R twins `docs/guides/capital-modeling-monograph-15-theory.qmd` and
  `capital-modeling-monograph-15-applications.qmd` of the CAS Monograph 15
  chapters 3 to 6 guides, every printed number equal to the Python guides';
  the R gap (`pentagon()` has no risk discount) is #365 (#372).
- `docs/guides/capital-modeling-monograph-15-theory.md` and
  `capital-modeling-monograph-15-applications.md`: CAS Monograph 15 (Major
  and Mildenhall), chapters 3 to 6, on the InsCo example. VaR and TVaR by the
  monograph's algorithms, coTVaR, the pentagon, layer pricing, the five
  calibrated distortions and their weights, the natural allocation by hand
  and by `Portfolio`, the industry standard approach; then performance
  assessment, the 35 xs 65 reinsurance decision, new business and acquisition
  pricing, portfolio optimization by grid search, and the range of unit
  premiums by `premium_bounds`, with the monograph's printed values beside
  prospicio's (#364).
- New guide `capital-modeling-monograph-15.md`: CAS Monograph 15's InsCo example, reinsurance decision and correlated three-unit book reproduced in prospicio, with the monograph's numbers beside ours.
- `docs/guides/pir-price-allocation.qmd` and `docs/guides/pir-price-allocation-practice.qmd`,
  the R twins of the *Pricing Insurance Risk* Part III allocation guides, every
  number equal to the Python guides'. CI's R job runs their chunks and checks
  every printed number (`R/prospicio/tests/test-guides.R`) (#374).
- `docs/guides/pir-price-allocation.md` and `docs/guides/pir-price-allocation-practice.md`:
  Part III of Mildenhall and Major's *Pricing Insurance Risk* reproduced, the
  classical allocations, the natural allocation linear and lifted, margin and
  capital by unit, Bodoff's percentile layer of capital, pricing bounds, the
  Cat/Non-Cat case-study tables and chapter 19's reinsurance test, with the
  book's printed numbers beside prospicio's, checked in CI (#370).
- R version of the Pricing Insurance Risk case studies guide (`pricing-insurance-risk-case-studies.qmd`), checked by `R/prospicio/tests/test-guides.R`.
- New guide `pricing-insurance-risk-case-studies.md`: the case studies of *Pricing Insurance Risk* (Discrete, Tame, Cat/Non-Cat and Hurricane/SCS, gross and net of reinsurance) priced by five distortions and allocated to units in prospicio, with `aggregate` 0.30.1's numbers beside ours (#323).
- `docs/guides/pir-risk-measures.qmd` and `docs/guides/pir-portfolio-pricing.qmd`,
  the R twins of the guides to Parts I and II of Mildenhall and Major's
  *Pricing Insurance Risk* (risk measures; portfolio pricing): every number
  equal to the Python guides', run by `R/prospicio/tests/test-guides.R`. The R
  gaps found are `VaR()` on grids (#373) and `VaR()` and `TVaR()` on
  parametric severities (#375).
- `docs/guides/pir-risk-measures.md` and `docs/guides/pir-portfolio-pricing.md`:
  Parts I and II of Mildenhall and Major's *Pricing Insurance Risk* run in
  prospicio (risk measures, VaR and TVaR on the case studies, classical
  premium principles, the portfolio CCoC price, ruin and the natural scale,
  the five distortions and the Blend priced by unit), with the book's printed
  numbers beside prospicio's and every mismatch explained (#371).
- `cargo xtask r` shows an R article that `_pkgdown.yml` does not list under a last group, "More articles", instead of failing the site build.
- The R site lists the reading-path example among its worked examples, which the Pages build needs (it fails on an article missing from `_pkgdown.yml`).
- The R twins of the guides (`docs/guides/*.qmd`) and of the notebooks
  (`examples/*.qmd`) are pages of the R docs site, under "Articles", grouped
  as in `docs/guides/reading-list.md`. `cargo xtask r` stages them as pkgdown
  articles (the `.qmd` files stay the only source) and points their links to
  the Python guides at the Python site's pages. A guide's chunks are shown
  with the output that `test-guides.R` checks, not re-run; the two examples
  run in full, and the build fails if `reinsurance_pricing.qmd` reports a
  comparison outside its tolerance. The site index links to them (#388).
- New page `reading-list.md`: each source text, its guides, knowledge file and open issues.
- The guides README points to the reading-path notebook.
- `docs/guides/reserving-friedland.qmd`, the R twin of the guide that
  reproduces Friedland's *Estimating Unpaid Claims Using Basic Techniques*
  (development, expected claims, Bornhuetter-Ferguson, Benktander and Cape
  Cod on U.S. Industry Auto and XYZ Insurer): every number equal to the
  Python guide's, checked by `R/prospicio/tests/test-guides.R` (#344).
- `reserving-friedland.md`: Friedland's *Estimating Unpaid Claims Using
  Basic Techniques* (CAS) reproduced on its U.S. Industry Auto and XYZ
  Insurer data: age-to-age averages and selected factors, the development,
  expected claims, Bornhuetter-Ferguson, Benktander and Cape Cod techniques
  and the book's reasonableness checks, with its printed ultimates and IBNR
  beside prospicio's (#337).
- R version of the Meyers Monograph 8 guide, `reserving-meyers.qmd`: the
  same sections, checks and recorded numbers with `meyers()`, `bayes_loo()`,
  `meyers_one_year()`, `mack()` and `odp_bootstrap()`, checked by
  `R/prospicio/tests/test-guides.R` (#346).
- New guide `reserving-meyers.md` reproducing CAS Monograph 8, Meyers'
  *Stochastic Loss Reserving Using Bayesian MCMC Models*: the model sequence
  from Mack and the ODP bootstrap to CRC, CSR and CAY, the validation by the
  outcome's percentile, p-p plots and the KS test, and his printed results on
  group 353 beside prospicio's, with the Monte Carlo error of each (#343).
- `docs/guides/reserving-taylor-mcguire.qmd`, the R twin of the Taylor and
  McGuire (CAS Monograph 3) guide: the cross-classified ODP GLM, delta-method
  and parametric-bootstrap prediction errors, residual heat maps, reduced
  models and AIC, BIC and GCV with `glm_fit()`, `chain_ladder()`, `mack()` and
  `odp_bootstrap()`, every printed number equal to the Python guide's (#341).
- `docs/guides/reserving-taylor-mcguire.md`: Taylor and McGuire's *Stochastic
  Loss Reserving Using Generalized Linear Models* (CAS Monograph 3)
  reproduced. The chain ladder as a cross-classified ODP GLM (Tables 3-1 to
  3-5 and 5-1 exactly), the delta-method prediction error by accident year and
  in total (Table 5-4 exactly), the parametric bootstrap (Tables 5-5, 7-2, 7-4,
  7-6 within noise), residual heat maps, the reduced models with interactions
  (Tables 7-1, 7-3, 7-5) and AIC, BIC and GCV (Table 7-7), with the gaps filed
  as #333 and #334 (#335).
- The reading list and the README's R-twin table list the Monograph 15 chapters 1, 2, 7 and 8 guides and the compartmental reserving guide, with their R twins.
- R twins of the two Werner and Modlin ratemaking guides,
  `ratemaking-werner-modlin.qmd` and
  `ratemaking-werner-modlin-indications.qmd`, with the same numbers as the
  Python guides, run by `R/prospicio/tests/test-guides.R` (#345).
- Two guides reproducing Werner and Modlin's *Basic Ratemaking* (CAS study
  note), with the book's printed numbers beside prospicio's:
  `ratemaking-werner-modlin.md` (half-month dates and trend lengths, the
  parallelogram method with law changes, premium and loss trend fits, loss
  development with `prospicio.reserving`, expenses and the permissible loss
  ratio, limited fluctuation and Buhlmann credibility) and
  `ratemaking-werner-modlin-indications.md` (Appendices A to C end to end,
  chapter 11's increased limits and deductibles, what prospicio does not
  cover) (#340).

## 0.0.3 (2026-10-15)

### Breaking changes

- `PreparedLoss` gains `development`, `ultimate` and `fx`, and its
  `trended` is now the developed amount trended; code that builds or
  matches it in full must name the new fields (#224, #225).
- `CompartmentalPriors` has three new public fields (`ker_effect_sd`,
  `kp_effect_sd`, `development_effect_sd`); code that builds it in full
  adds them or uses `..CompartmentalPriors::gesmann_morris()` (#254).
- Rust: `Development` is no longer `Copy` or `Eq`; `ChainLadder`, `Mack`
  and the four expected-loss settings are no longer `Copy`;
  `Development`, `OdpBootstrap`, `MackBootstrap`, `DevelopmentFit` and
  `ChainLadderFit` gain public fields, and `Error` gains `Pattern` (#264).
- The exponential premium principle's value changes at small `k`: it kept
  its loading where it used to fall below the mean (#258).

### Distributions and probability

- Third raw moments: `Severity::third_moment()` (`None` where a family has
  no closed form, infinite where the moment diverges) and
  `Counting::third_moment()` with `Counting::third_cumulant()`, for
  compound moments to third order (#183).
- `aggregate`'s named counts: `EmpiricalCount::fixed`,
  `Binomial::bernoulli` and `NegativeBinomial::geometric`; Python
  `Count.fixed`, `Count.bernoulli`, `Count.geometric`; R `fixed_count()`,
  `bernoulli_count()`, `geometric_count()` (#202).
- TVaR of a grid and `tvar_threshold`, the level at which TVaR reaches the
  VaR at `p` (`aggregate`'s `tvar_threshold`): `Grid::tvar`,
  `Grid::tvar_threshold`, `natural::Portfolio::tvar_threshold`,
  `risk::{var_weighted, tvar_weighted, tvar_threshold}`. Python
  `Grid.tvar`, `Grid.tvar_threshold`, `Portfolio.tvar_threshold`; R
  `TVaR()` on a `grid_distribution` and `tvar_threshold()` (#202).
- `Distortion::family()` and `Distortion::parameters()`: a distortion's
  family and its scalar parameters by name (for example the `rho` that
  `calibrate` returns). Python `Distortion.family`, `Distortion.parameters`;
  R `d@parameters` (#210).
- A copula's distribution function, `Copula::cdf`: the Archimedean
  copulas' in any dimension (with `ArchimedeanCopula::inverse_generator`)
  and the Gaussian copula's in two, from a new bivariate normal cdf
  (`prospicio_math::special::bivariate_norm_cdf`, Genz's BVND, about
  `1e-15`); an error for the t copula. Python `cdf` on the copula classes;
  R `copula_cdf()` (#234).
- A distribution on finitely many values of either sign and any spacing,
  `prospicio_prob::Discrete`, for the rows of a P&L: equal values merged,
  the lower and upper quantiles, TVaR and the return-period table as
  `aggregate` 1.0.1's `GridDistribution`, `pmf` and skewness. Python
  `Discrete`; R `discrete_distribution()`, `quantile_upper()`,
  `discrete_skewness()` and `return_periods()` (#241).
- Nested (hierarchical) Archimedean copulas,
  `prospicio_prob::NestedArchimedeanCopula`: Clayton or Gumbel copulas
  joining groups of dimensions inside a weaker outer one, to any depth,
  with the distribution function and McNeil's sampling. Python
  `prospicio.risk.NestedArchimedeanCopula`; R
  `nested_archimedean_copula()` and `nac_node()` (#261).
- Vine copulas, `prospicio_prob::vine`: pair copulas (independence,
  Gaussian, Clayton, Gumbel, Frank, Joe, with rotations) with their
  conditional distributions and densities, and C- and D-vines built from
  them (log-density, inverse Rosenblatt transform, sampling), matching
  pyvinecopulib 1.0.1. Python `prospicio.risk.PairCopula` and
  `VineCopula`; R `pair_copula()`, `vine_copula()` and
  `copula_log_density()` (#262).
- A severity under a mix of policy limits, `prospicio_prob::PolicyLimits`
  (#213): `min(X, L)` with each limit's weight, exact layer moments from
  the base severity's and an atom at each finite limit, so it discretizes,
  compounds and saves as JSON like any severity. Python `PolicyLimits`; R
  `policy_limits_distribution()`.
- Standard errors of the Pareto fits from the observed information
  (#213): `Pareto::fit_with_standard_error` (truncated or not) and
  `PiecewisePareto::fit_with_standard_errors` (uncorrelated by piece).
  Python `Pareto.fit_with_standard_error`,
  `PiecewisePareto.fit_with_standard_errors`; R `pareto_standard_error()`,
  `piecewise_pareto_standard_errors()`.
- Maximum likelihood fits of the lognormal, gamma, Weibull and Burr to
  large losses with reporting thresholds (left truncation), policy limits
  (censoring) and weights (#194):
  `prospicio_prob::severity_fit::fit_severity`, returning the
  distribution, parameters, standard errors and covariance,
  log-likelihood, AIC and BIC. Python `fit_severity` and `SeverityFit`; R
  `severity_fit()`.
- Goodness of fit for large losses with reporting thresholds and
  censoring (#194): the product-limit (Kaplan–Meier) estimator with left
  truncation, `severity_fit::product_limit`, and
  `severity_fit::goodness_of_fit`, which sets any severity against it
  (P-P, Q-Q and the largest distance). Python `product_limit`,
  `goodness_of_fit`; R `product_limit()`, `goodness_of_fit()`.

### Aggregate and reinsurance

- The occurrence exceedance (OEP) curve and occurrence PMLs for any claim
  count, `prospicio_aggregate::pml::{occurrence_exceedance,
  occurrence_pml}`, and each simulated year's largest loss,
  `EventSet::maxima()`, matching `aggregate` 1.0.1's `oep` on Poisson
  books. Python `prospicio.aggregate`; R `occurrence_exceedance()`,
  `occurrence_pml()`, `annual_maxima()` (#182).
- Compound moments to third order by cumulants and the moment
  approximations of `aggregate`'s `approximate` (normal, gamma, lognormal,
  shifted gamma, shifted lognormal), `prospicio_aggregate::moments`.
  Python `prospicio.aggregate.compound_moments`; R `compound_moments()`,
  `moment_approximation()`, `approximation_cdf()`,
  `approximation_quantile()` (#184).
- Per-occurrence layer statistics, `aggregate`'s `reins_stats_df`: the
  chances of attaching, detaching and a loss in the year, the loss on line,
  and the moments of the excess count, the layer severity and the annual
  ceded loss (`prospicio_aggregate::occurrence_layer_statistics`); and
  each view of a tower on the grid priced under a distortion,
  `reins_price_df` (`TowerGrids::price`). Python
  `prospicio.reinsurance.occurrence_layer_statistics`,
  `TowerGrids.price`; R `occurrence_layer_statistics()`, `tower_price()` (#187).
- The senior and junior priority ladder for capital, `aggregate`'s
  `priority_epd_df` and `priority_kappa`: each unit's recovery and EPD
  ratio from the assets when it ranks ahead of the other units, pro rata
  or behind them, and where the shortfall falls given the total
  (`natural::Portfolio::priority`, `priority_kappa`); exact under any
  dependence for a portfolio built from scenarios. Python
  `Portfolio.priority`, `Portfolio.priority_kappa`; R `priority_ladder()`,
  `priority_kappa()` (#201).
- Eventual ruin of the classical surplus process by the
  Pollaczeck–Khinchine formula, `aggregate`'s `pollaczeck_khinchine` and
  `eventual_ruin`: `ψ(u)` on the severity's grid, the capital for a ruin
  probability, and the Lundberg exponent, with the severity capped, in
  excess or stopped (`prospicio_aggregate::ruin`). Python
  `prospicio.aggregate.eventual_ruin`; R `eventual_ruin()` (#201).
- Layer loss picks, `aggregate`'s `Aggregate.picks`: a severity grid
  reshaped so that its expected loss to each layer of a tower is a pick,
  rescaling the survival function within each layer and keeping the tail
  (`prospicio_aggregate::picks::layer_loss_picks`). Infeasible picks are
  an error. Python `prospicio.aggregate.layer_loss_picks`; R
  `layer_loss_picks()` (#201).
- Pricing bounds for other risks, `aggregate`'s `PricingBounds` and
  `Bounds`: the range of a risk's price over every distortion that prices
  a reference risk at a premium, with the BiTVaRs attaining it, and the
  envelope of those distortions, both exact
  (`prospicio_pricing::bounds`). Python `prospicio.pricing.PricingBounds`,
  `distortion_envelope`; R `pricing_bounds()`, `distortion_envelope()` (#201).
- The total cession of a stage with several layers on the grid,
  `TowerGrids::stage_ceded`, priced in `TowerPrice::stages`; its statistics
  (`aggregate`'s `Ceded` column), `occurrence_cession_statistics`; and an
  aggregate cover's statistics on any annual grid,
  `aggregate_layer_statistics`. Python `TowerGrids.stage_ceded`,
  `TowerPrice.stages`, `prospicio.reinsurance.occurrence_cession_statistics`,
  `aggregate_layer_statistics`; R `tower_on_grid()`'s `stage_ceded`,
  `tower_price()`'s `stage_<k>` rows, `occurrence_cession_statistics()`,
  `aggregate_layer_statistics()` (#201).
- Each unit priced alone against the book priced together,
  `aggregate`'s `Portfolio.price_stand_alone`
  (`natural::Portfolio::price_stand_alone`), and Denuit et al.'s (2025)
  comonotonic allocation, `make_comonotonic_allocations`
  (`natural::Portfolio::comonotonic_kappa`). Python
  `Portfolio.price_stand_alone` (`StandAlone`), `Portfolio.comonotonic_kappa`;
  R `price_stand_alone()`, `comonotonic_allocation()` (#202).
- A tower on an annual loss distribution, `Tower::on_annual_grid`
  (`AnnualGrids`): a quota share or stop-loss on a loss-ratio grid, with its
  expected sliding-scale and profit commission, each year's total treated
  as one loss. Python `Tower.on_annual_grid`; R `tower_on_annual_grid()` (#210).
- Risk-loaded prices on a grid at any capital standard,
  `risk_load::price_grid`: Python `prospicio.pricing.price` and R
  `risk_loaded_price()` take a grid, such as a layer's exact annual loss (#210).
- A cover's quote, `prospicio_pricing::quote::Quote`: the technical
  premium from a risk-loaded price grossed up for internal expenses,
  brokerage and a fixed expense, with the rate on line, payback, multiple
  and loss ratio, and `Quote::at` for what a quoted premium gives the
  reinsurer (its return on capital). Python `prospicio.pricing.Quote`; R
  `technical_quote()`, `implied_quote()` (#222).
- Experience rating from individual large losses,
  `prospicio_pricing::experience`: each loss trended to the prospective
  period by a rate or a factor per loss, a reporting threshold carried to
  the level at which every year is complete, each year's tower applied to
  the year's trended losses, the layer totals developed and divided by the
  premium (the burning cost), reproducing Clark's study note. Python
  `LargeLossExperience`, `PreparedLosses`, `BurningCost`; R
  `trend_large_losses()`, `burning_cost()` (#223).
- Experience rating: each known large loss developed to ultimate (IBNER)
  by a `DevelopmentPattern` at its age at the valuation date or by a factor
  per loss, a pattern estimated from each claim's valuations (the chain
  ladder on the claims seen at both ages), and each year's count IBNR
  factor from a count pattern (`YearBasis::from_count_pattern`).
  `PreparedLoss` gains `development` and `ultimate`, and its `trended` is
  the developed amount trended (**breaking** for code that builds or
  matches `PreparedLoss` in full). Python `DevelopmentPattern` and
  `prepare(..., development=)`; R `development_pattern()`,
  `development_pattern_from_history()`, `pattern_factor()` (#224).
- Experience rating: losses converted to the pricing currency at a stated
  rate or a rate per loss (`Experience::with_fx`), and an index
  (stability) clause on the treaty, full or severe inflation, with a
  franchise (`IndexClause`, `Prepared::burning_cost_indexed`), for covers
  whose terms are all per occurrence. `PreparedLoss` gains `fx`
  (**breaking** for code that builds or matches it in full). Python
  `IndexClause`, `LargeLossExperience(..., fx=)` and `burning_cost(...,
  index=, ratios=)`; R `index_clause()` and the `fx`, `index` and `ratios`
  arguments (#225).
- As-if history: each year's large losses as reported under the program in
  force that year, beside the trended losses under the proposed tower, to
  reconcile with the cedant's recoveries (`Prepared::as_if`, `AsIf`).
  Python `PreparedLosses.as_if`; R `as_if_history()` (#226).
- The time value of money in a price, `prospicio_pricing::payout`: payment
  patterns (from shares or from factors to ultimate such as a paid chain
  ladder's), discount factors, the share outstanding, and a one-period
  price restated over the payout with capital held while losses are
  outstanding (`price_over_payout`). Python `PaymentPattern`,
  `price_over_payout`, `PayoutPrice`; R `payment_pattern()`,
  `payment_pattern_from_development()`, `pattern_values()`,
  `price_over_payout()` (#228).
- Catastrophe model output: an event loss table with secondary uncertainty
  (a beta on each event's exposure), its occurrence exceedance curve and
  PMLs, and simulated years of events with each loss's event
  (`prospicio_aggregate::cat::EventLossTable`), and a year loss table in
  long form (`EventSet::from_long`). Python `EventLossTable`,
  `EventSet.from_long`; R `event_loss_table()`, `elt_oep()`,
  `elt_occurrence_pml()`, `simulate_elt()`, `event_set_from_long()` (#229).
- Event sets merged year by year (a catastrophe set with a non-catastrophe
  one, interleaved by time) and grouped into occurrences by an hours clause
  (`EventSet::merge`, `EventSet::grouped`). Python `EventSet.merge`,
  `EventSet.grouped`; R `merge_event_sets()`, `hours_clause()` (#231).
- Two lines hit by the same events: their joint annual loss by a 2-D
  compound FFT from a joint per-event severity
  (`prospicio_aggregate::bivariate::{fft_bivariate, JointGrid}`, with each
  line's marginal, the total, mixed moments and the correlation), and
  `aggregate`'s `clash` model, given by the expected events hitting line
  A only, line B only and both (`Clash`), matching `aggregate` 1.0.1.
  Python `fft_bivariate`, `JointGrid`, `Clash`; R `fft_bivariate()`,
  `clash_model()`, `clash_severity()`, `clash_aggregate()` (#233).
- Two lines whose losses in an event are coupled by a copula: the joint
  per-event severity as the copula's rectangle probabilities over the two
  marginals (`JointGrid::from_copula`, with `bivariate::triggered` for a
  line an event hits with some probability), for `fft_bivariate`,
  matching `aggregate` 1.0.1's copula `bivariate`. Python
  `JointGrid.from_copula`; R `copula_severity()` (#235).
- Renewal claim counts: the number of claims by a horizon when the waits
  between claims follow any law (`prospicio_aggregate::renewal::renewal_count`,
  a `Wait` that is a distribution or discrete waits, possibly defective,
  with waits of zero arriving in clusters), on `aggregate`'s wait grid or
  a given one, matching `aggregate` 1.0.1's `years ... wait ...`
  frequency. Python `Count.renewal`; R `renewal_count()` (#236).
- Eventual ruin for renewal claims: `ψ(u)` when the waits between claims
  follow any law and the premium comes in at `(1 + ρ)` times the expected
  loss rate, by the cepstral Wiener–Hopf factorization of the claim walk
  (`prospicio_aggregate::ruin::wiener_hopf`, returning `RenewalRuin`),
  matching `aggregate` 1.0.1's `wiener_hopf`. Python `wiener_hopf`; R
  `wiener_hopf()` (#237).
- The P&L ledger, as `aggregate` 1.0.1's `PnL`
  (`prospicio_aggregate::pnl::{Ledger, Group, Leg}`): signed legs in books
  that sell or buy cover over a source, every row (legs, side totals,
  results, running nets, spans and net-of blocks, grand rows, the covers'
  total impact) a sum at each atom with its exact distribution, the card,
  and the scenario columns that condition each row on the grand result.
  Python `prospicio.aggregate.Ledger`, `Group`, `Leg`; R `pnl_ledger()`,
  `pnl_group()`, `pnl_leg()` and `ledger_sheet()`, `ledger_card()`,
  `ledger_marginal()`, `ledger_legs()`, `ledger_result()`, `ledger_row()`
  (#242).
- Legs rebucketed on a step (`Leg::step`, with `Ledger::validation`
  comparing the exact and grid means), the ratio exhibit
  (`Ledger::ratios`: premium, loss, expense and result by block, ratios of
  means and means of ratios, shares) and the stack of several ledgers'
  results with their differences (`pnl::stack`), as `aggregate` 1.0.1's
  `bs`, `economic_ratios_df` and `stack_marginal_pnls`. Python
  `Leg(step=)`, `Ledger.validation`, `Ledger.ratios`, `stack`; R
  `pnl_leg(step =)`, `ledger_validation()`, `ledger_ratios()`,
  `ledger_stack()` (#244).
- The ledger of a seller's book and the annual covers it buys, as
  `aggregate` 1.0.1's `pnl` and `xpnl` builders
  (`prospicio_aggregate::pnl::{Book, Expense, Face, derive_premium}`):
  consolidated (the net position over the net loss), walked (one group
  per stage of the tower) or peeled (one group per layer, top down or
  bottom up), with expenses on a fixed, premium or loss basis and
  `derive premium`. `pnl::Source` now keeps probabilities that already
  sum to 1 (to `1e-12`). Python `Book`, `Expense`, `derive_premium`; R
  `pnl_book()`, `pnl_expense()`, `book_ledger()`, `derive_premium()`
  (#245).
- Occurrence covers in the P&L ledger: a joint source of the annual
  gross and occurrence-ceded loss (`pnl::Source::joint`,
  `pnl::occurrence_source` by a 2-D FFT of each claim and its cession,
  `aggregate`'s `occ_bivariate`), legs of both coordinates
  (`pnl::Leg::of2`), and `Book::occurrence_ledger`: the walk through the
  occurrence covers and then aggregate covers on their net, the peel and
  the consolidated face, matching `aggregate` 1.0.1. Python
  `occurrence_source`, `Book.occurrence_ledger`; R `occurrence_source()`,
  `book_occurrence_ledger()` (#246).
- Variable terms in the P&L builders: a swing-rated premium, a
  sliding-scale or profit commission and a loss corridor booked as legs
  at each atom from the layer's own terms, on every face; retro rating of
  the book (`Book::retro`); and `pnl::pnl_program`, the premium and each
  layer's deposit from target combined ratios, with `size_premium` and
  `round_premium`, matching `aggregate` 1.0.1. The builders no longer
  refuse loss-sensitive terms (reinstatement premiums still are). Python
  `Book(retro=)`, `pnl_program`, `size_premium`, `round_premium`; R
  `pnl_book(retro =)`, `pnl_program()`, `size_premium()`,
  `round_premium()` (#247).
- Reinstatement premiums in the P&L ledger: an occurrence cover of one
  layer with annual terms and paid reinstatements over the joint of the
  gross loss and its unlimited annual cession, its recovery `A(R)` and
  premium `D + h(R)` varying with the year on every face, and an
  aggregate cover behind it, matching `aggregate` 1.0.1. Annual covers
  now refuse paid reinstatements with a pointer to
  `Book::occurrence_ledger` (#248).
- `pnl::pnl_program` also respells each priced deposit as a rate
  (`Program::rates`, `aggregate`'s `premium_style='rate'`), and
  `pnl::pnl_program_stated` prices an engine that states its own technical
  premium; `ProgramLayer` says its tier. Python `pnl_program_stated` and a
  `rates` entry; R `pnl_program_stated()` and `rates` (#249).
- The P&L's consolidated face books a loss-basis expense at each year's
  gross loss, so its margin is the walk's net margin year by year (the
  mean is unchanged; its spread now includes the expense's).
  `Book::expense_at_expected_loss` keeps the expected-loss booking of
  `aggregate`'s guaranteed-cost builder. Python
  `Book(expense_at_expected_loss=)`; R `pnl_book(expense_at_expected_loss
  =)` (#250).
- The stitched peel of several occurrence layers, `aggregate`'s `xpnl ...
  peel` with two or more occurrence layers: `Book::occurrence_peel` builds
  every row as its own exact marginal (compounds of per-claim maps, and
  pushforwards for aggregate covers behind them) through
  `pnl::Ledger::stitched`; the mean column foots, the other statistics and
  the card are each row's own, with no scenario columns. Python
  `Book.occurrence_peel` and `Ledger.is_stitched`; R
  `book_occurrence_peel()` and `ledger_is_stitched()` (#251).
- `PredictiveDistribution::select(dim, keys)`: the components with given
  values on one dimension, joint structure kept, so a tower's ceded part
  can be joined to a book (#193). Python `PredictiveDistribution.select`, R
  `select_components()`; `Tower.apply`'s docs now warn against summing over
  `kind`.
- Tower matching diagnostics (#211): `TowerModel::diagnostics` reports
  alphas below a minimum, jumps between neighbouring alphas and an
  infinite mean, and `fit_references_bounded` keeps the alpha between
  reference points within bounds; `fit_references` no longer says
  `references = NaN` when it refuses. Python `TowerModel.diagnostics()` and
  `fit_references(..., alpha_bounds=)`; R `tower_diagnostics()` and
  `fit_references(alpha_bounds =)`.
- Exposure-rating helpers (#212): `TabulatedCurve::from_severity` (a
  severity's exposure curve, tabulated where it bends),
  `Mbbefd::swiss_re_for_mean_rate` and `swiss_re_for_layer_share` (the
  Swiss Re `c` calibrated to a mean destruction rate or to one layer's
  share), and `layer::layer_cv` (the CV of a layer's loss per loss that
  reaches it). A risk profile's band may start at 0, and
  `Band::with_spread_bounds` keeps the band's sum insured as the mean
  when it can. The Python and R risk profiles now warn when the default
  uniform spread moves a band's mean, and take `spread="auto"`. Python
  `TabulatedCurve.from_severity`, `Mbbefd.swiss_re_for_mean_rate`,
  `Mbbefd.swiss_re_for_layer_share`, `layer_cv`; R
  `tabulated_curve_from_severity()`, `swiss_re_for_mean_rate()`,
  `swiss_re_for_layer_share()`, `layer_cv()` and
  `risk_profile(spread = "auto")`.
- Small fixes from the guides (#214): `LimitedFluctuation::p` and `k`
  (Python `LimitedFluctuation.p`, `.k`); `year_fraction`'s docs warn
  against assigning accident years with it; the exponential premium
  principle keeps its loading at a very small `k` (it fell below the
  mean at `k = 1e-12`); `Portfolio::epd` no longer returns a tiny
  negative ratio.

### Examples

- `examples/reinsurance_pricing.ipynb` prices an occurrence tower, a
  program and a share, an aggregate limit with reinstatements, stop loss and
  aggregate deductible, retro, swing, profit commission and sliding scale,
  and distortion premiums, and compares 343 results (321 independent
  comparisons and 22 consistency checks) with Mildenhall's `aggregate` 1.0.1 (reference file `validation/reference/re_pricing_aggregate.csv`) (#186).
- The notebooks in `examples/` are committed without outputs, and an
  nbstripout git filter (`.gitattributes`; setup in `examples/README.md`)
  keeps a local run out of commits; the docs site executes them and
  publishes their results, with the guides, in the Python site's User Guide (#215).
- `examples/submission_evaluation.ipynb` evaluates a synthetic per-risk
  excess of loss submission end to end, from the raw files (premium and rate
  changes, a large-loss listing, a loss triangle, a risk profile, treaty
  terms; generated by `examples/data/make_submission.py`) to a decision to
  quote, counter, decline or refer to the portfolio view: on-level premium
  and trend, chain ladder and Cape Cod, large-loss development, trend and a
  Pareto fit with truncation and censoring, experience and exposure rating
  blended by credibility, the program on the grid, risk metrics, technical
  premium against the broker's quote and against one price of risk,
  sensitivity and parameter uncertainty, and the estimates against the
  generator's true parameters (`examples/data/answer_key.json`). It needs a
  prospicio newer than 0.0.2 (#217).
- `examples/submission_evaluation.ipynb` uses the same features in place of
  by-hand steps, and `examples/reinsurance_pricing.ipynb` checks calibrated
  distortions through `Distortion.parameters` (#232).

### Guides

- `docs/guides/reinsurance-price.md`, "What a reinsurance price is": the pricing pentagon, an excess of loss layer's technical premium and reinstatements, where a layer sits in a tower, a quota share's technical commission with a sliding scale and a profit commission, how price follows a risk profile, a contract alone and in a book, soft and hard markets, and what the model leaves out. CI runs its code and checks every printed number (`python/tests/test_guides.py`) (#203).

- Six "when to use what" guides in `docs/guides/`: `choosing-a-method.md` (the map from a situation to the method and the calls), `risk-loads-and-distortions.md`, `experience-and-exposure-rating.md`, `aggregate-distributions.md`, `severity-and-frequency.md` and `reserving-methods.md`. CI runs their code and checks every printed number (`python/tests/test_guides.py`); `docs/guides/README.md` lists them (#216).
- The guides and notebooks use the features from #193, #194 and #211 to
  #214: `PredictiveDistribution.select` for a tower's ceded part,
  `fit_severity` and `goodness_of_fit` for the body,
  `Pareto.fit_with_standard_error`, `TowerModel.diagnostics` and
  `alpha_bounds`, `TabulatedCurve.from_severity`, the Swiss Re `c`
  calibrations, `layer_cv`, `PolicyLimits` in the submission evaluation,
  and an explicit `spread` on risk profiles with bounds. The gaps tables
  keep #192 and clash across more than two lines (#265).
- The pricing, experience and exposure rating, risk load, distribution,
  severity and reserving guides, and the method map, use the current API:
  large-loss preparation with currency, development, trend and an index
  clause, as-if history under past programs, payment patterns and the price
  over the payout, event loss tables and year loss tables, the quote,
  `price` on a grid at a capital standard, a distortion's `family` and
  `parameters`, and towers on an annual loss grid. The list of what the
  library does not do yet keeps only the open issues (#232).

### Reserving

- The one-year view of both bootstraps accepts a bootstrap tail, which
  must equal the re-reserving method's tail: an origin whose tail step falls
  in the year closes at its simulated ultimate, and the tail's parameter
  error is drawn as in the lifetime view. R gains `bootstrap_tail`,
  `tail_std_err` and `tail_sigma` on `odp_one_year()` and
  `mack_one_year()` (#176).
- The Bayesian chain ladder, `prospicio_reserving::bayes::BayesOdp`, behind
  a new `bayes` feature of `prospicio-reserving` (on in the umbrella's
  `bayes` feature; the default build and the WASM scope are unchanged):
  England and Verrall's over-dispersed Poisson model with log-linear
  origin and development effects, normal priors and the chain ladder's
  Pearson scale plugged in, sampled with NUTS on a scale whitened at the
  posterior mode (target acceptance 0.95 by default). Reserves are a
  `PredictiveDistribution` by origin, as the ODP
  bootstrap's, with Gamma process error or none; the fit carries
  the posterior draws, R-hat, ESS and divergence warnings, and PSIS-LOO.
  Checked against the same model in Stan (CmdStan 2.39.0) on GenIns and
  RAA, and against the chain ladder and the ODP bootstrap on GenIns
  (`docs/design/reserving-bayes.md`). In Python as
  `prospicio.reserving.BayesOdp` and in R as `bayes_odp()`, for a
  single-segment triangle, with the posterior summary, the long tables,
  PSIS-LOO (`BayesOdpFit.loo()`, R's `bayes_loo()`), and failed
  diagnostics raised as a Python `UserWarning` or an R `warning()` (#188).
- Meyers' correlated chain ladder (CRC) and changing settlement rate (CSR)
  models, `prospicio_reserving::bayes::Meyers` with `MeyersModel::{Crc,
  Csr}`, behind the `bayes` feature: log-normal cumulative values with
  premium from a measure column, his priors and his floor on each variance
  step, sampled with NUTS on a dense scale from a pilot run. Ultimates and
  reserves are `PredictiveDistribution`s by origin; the fit carries the
  model (`MeyersFit::model()`), posterior draws, diagnostics and PSIS-LOO
  on Meyers' log scale. Checked against Meyers' own Stan code
  (transcribed, CmdStan 2.39.0) and his printed results on his group 353 Commercial Auto data, now in
  `validation/data/clrd_comauto_353.csv` (`docs/design/reserving-bayes.md`,
  PR 3). In Python as `prospicio.reserving.Meyers` and in R as `meyers()`,
  for a single-segment triangle, with the ultimates and reserves, the
  posterior summary, the long tables, PSIS-LOO (`MeyersFit.loo()`, R's
  `bayes_loo()`), and failed diagnostics raised as a Python `UserWarning`
  or an R `warning()` (#200).
- The Bayesian Bornhuetter–Ferguson, `prospicio_reserving::bayes::BayesBf`,
  behind the `bayes` feature: England, Verrall and Wüthrich's (2012) ODP
  model with a gamma prior on each origin's row parameter (mean `apriori`
  times exposure from a measure column, coefficient of variation `cv`,
  flat when infinite) and a vague prior on the unnormalised development
  pattern, so the expected ultimate is the row parameter times the
  pattern's sum and `apriori` is a loss ratio only when that sum is 1.
  A flat prior gives the Bayesian chain ladder; a tight one EVW's
  strong-prior limit, Mack's Bornhuetter–Ferguson with the raw pattern,
  which is the classical one only when the prior ultimates are the chain
  ladder's. Checked against the same model in Stan on GenIns with premium
  and on EVW's triangle, now in `validation/data/evw2012.csv`, and against
  their printed totals (`docs/design/reserving-bayes.md`, PR 4). The mode
  finder of the Bayesian models no longer stalls under a tight prior. In
  Python (`prospicio.reserving.BayesBf`, `fit(triangle, column, exposure)`)
  and R (`bayes_bf()`), for one segment, with the fit's `exposure` and
  `prior_ultimate`, the posterior summary, the long tables, PSIS-LOO and
  failed diagnostics raised as a Python `UserWarning` or an R `warning()` (#208).
- `BayesOdp` over segments and in the diagonal backtest
  (`docs/design/reserving-bayes.md`, PR 2): `BayesOdp::fit_segments`
  fits each segment's own posterior into a joint `BayesOdpFits`, with
  `SegmentDependence::Independent` (each segment on its own derived
  seeds) or `RankCorrelation` (Iman–Conover, as in the bootstraps);
  `Synchronized` is an error, since a posterior has no residuals to share.
  A bad `RankCorrelation` matrix (the wrong size for the segments, not
  symmetric, or not positive definite after `2 sin(pi rho / 6)`) fails
  before any sampling; the bootstraps now also report a matrix that is not
  positive definite as `InvalidSetting`.
  `BayesOdp` implements `TriangleModel`, so `diagonal_backtest` scores it
  next to the ODP GLM, with log predictive densities for stacking. The
  posterior mode's Newton method now also stops at the rounding floor on
  very large values (a line summed over companies). In Python,
  `BayesOdp(dependence=..., spearman=...).fit_segments()` returns a
  `BayesOdpFits` (joint reserves, frames, `diagnostics_frame()`,
  `segment(...)`); in R, `bayes_odp_segments()` returns a
  `bayes_odp_fits` (#207).
- Meyers' correlated accident year model (CAY), `MeyersModel::Cay`: CRC
  with `rho` times the previous origin's residual at the same age added to
  each mean (a moving average of order 1 down each age's column, so
  neighbouring origins correlate), Meyers' `Beta(2, 2)` prior on
  `(1 - rho) / 2` (`Meyers::rho_shape`, default 2), and ultimates drawn
  origin by origin down the oldest age. Checked on group 353 incurred
  against Meyers' own `CAY.R` Stan code (transcribed, CmdStan 2.39.0),
  including the correlation of neighbouring origins' reserves, and against
  his printed results (`docs/design/reserving-bayes.md`, PR 5). In Python
  `Meyers("cay", rho_shape=2.0)` and in R
  `meyers(..., model = "cay", rho_shape = 2)`; `rho_shape` follows
  `gamma_sd` in both signatures (#218).
- Meyers' models (CRC, CSR and CAY) over segments and in the diagonal
  backtest (`docs/design/reserving-bayes.md`, PR 6), in Rust:
  `Meyers::fit_segments(triangle, column, premium)` fits each segment's
  own posterior into a joint `MeyersFits` (ultimates and reserves, with
  `to_long`, `totals`, `development_table` and `segment`), with
  `Meyers::dependence`, a new field: `Independent` (each segment on its own
  derived seeds) or `RankCorrelation` (Iman–Conover on the reserves, each
  segment's ultimates moving with them); `Synchronized` and a bad matrix
  fail before any sampling. `Meyers` implements `TriangleModel`
  (`meyers_crc`, `meyers_csr`, `meyers_cay`): it rebuilds each origin's
  cumulative values from the train increments, reads premium from the
  frame's exposure, simulates each origin's path jointly (a test
  increment is the difference of consecutive values on one path) and gives
  the log-normal log predictive density of a held-out diagonal. It
  reports no diagnostics; on group 353's small held-out triangles the
  default target acceptance 0.95 sometimes fails to converge, and its
  doc advises 0.99 there.
  `validation/data/clrd_lines.csv` gains each line's net earned premium
  (`validation/scripts/clrd_lines.py`). In Python,
  `Meyers(..., dependence="independent", spearman=None)` and
  `Meyers.fit_segments(triangle, column, premium)` returning `MeyersFits`
  (joint `ultimates` and `reserves`, `premium`, the frames,
  `diagnostics_frame()`, `segment(...)`); in R,
  `meyers_segments(triangle, column, premium, ..., dependence, spearman)`
  returns a `meyers_fits`. The diagonal backtest stays Rust-only (#227).
- `Compartmental` fits Gesmann and Morris' model 2: the log-normal
  likelihood (`CompartmentalLikelihood::LogNormal`), loss ratios to
  premium (`Scale::LossRatio`) and incremental paid
  (`PaidResponse::Incremental`, each increment the closed form at its age
  less the same at the age before, with the cell's own parameters) are
  built, in any combination with the normal likelihood, amounts and
  cumulative paid. The hierarchy is in the priors: `CompartmentalPriors`
  gains `ker_effect_sd`, `kp_effect_sd` (origin effects on the rates) and
  `development_effect_sd` (development-year effects on any of the four
  parameters), all `None` in `gesmann_morris()`, and
  `CompartmentalPriors::gesmann_morris_2()` is the paper's model 2 preset.
  A development age with no observed cell takes a new effect from the
  population in each simulation, as an origin with none does.
  `CompartmentalFit::cell_rate_swaps` reports the share of draws and cells
  with `k_er <= k_p` when the rates vary by cell. At parity with the
  paper's model 2 code reconstructed in Stan and its printed posterior on
  company 337. `CompartmentalPriors` has three new public fields
  (**breaking** for code that builds it in full; `..CompartmentalPriors::gesmann_morris()`
  keeps model 1's hierarchy) (#254).
- Python: `CompartmentalPriors` takes `ker_effect_sd`, `kp_effect_sd` and
  `development_effect_sd` (all `None` by default, model 1's hierarchy),
  with `CompartmentalPriors.gesmann_morris_2()` and, in `replace`, `False`
  to remove an effect; `Compartmental.model_2(priors, ...)` sets model 2's
  likelihood, scale, paid response and sampler; the log-normal, loss
  ratios and incremental paid no longer raise "not yet built";
  `CompartmentalFit.cell_rate_swaps` (#254).
- R: `compartmental_priors()` takes `ker_effect_sd`, `kp_effect_sd` and
  `development_effect_sd` (a named list over `k_er`, `k_p`, `rlr` and
  `rrf`), with `gesmann_morris_2_priors()`; `compartmental_model_2()`;
  the fit's `cell_rate_swaps`; `compartmental_prior_predictive()` takes
  `paid_response`; the log-normal, loss ratios and incremental paid no
  longer stop with "not yet built" (#254).
- Gesmann and Morris' hierarchical compartmental reserving model with one
  outstanding compartment (`prospicio_reserving::bayes::Compartmental`,
  behind the `bayes` feature): paid and outstanding fitted jointly with
  NUTS from a reporting rate, a payment rate and each origin's reported
  loss ratio and reserve robustness factor (correlated origin effects),
  with priors a required argument (`CompartmentalPriors`, and the paper's
  case study as `CompartmentalPriors::gesmann_morris()`), a prior
  predictive check (`prior_predictive`), the reserve to the square's last
  age as a joint `PredictiveDistribution`, the future paid and outstanding
  cells, PSIS-LOO split by type, and a warning when a draw's rates are in
  the swapped order (or `RateOrder::Ordered`). The likelihood, scale, paid
  response and fitted pair are settings; this release fits the paper's
  model 1 (normal on amounts, cumulative paid, paid and outstanding), at
  parity with the paper's code reconstructed in Stan and its printed
  posterior on company 337; the other values are an error saying they are
  not yet built. In Python as `prospicio.reserving.Compartmental`
  (`CompartmentalPriors.gesmann_morris()`, `fit(..., incurred=...)` or
  `outstanding=...`, `prior_predictive`) and in R as `compartmental()`
  (`gesmann_morris_priors()`, `compartmental_prior_predictive()`), with
  the settings as strings, the diagnostics as one warning, and
  `bayes_loo()` in R (#243).
- The one-year view of the Bayesian ODP chain ladder,
  `BayesOdp::one_year` and `BayesOdp::one_year_segments` (behind the
  `bayes` feature, settings in `BayesOneYear`): the claims development
  result over the coming year by Pareto smoothed importance sampling of
  the posterior draws, each simulated year reweighting them; a
  simulation whose Pareto k is above `k_threshold` (0.7) is refitted
  with NUTS on the extended triangle (`OneYearRefit::RefitFlagged`, the
  default; `ReweightOnly` and `RefitAll` too). Each simulation's Pareto
  k and the flagged share are reported; segments are independent or rank
  correlated. Its standard deviation agrees with the ODP bootstrap's
  one-year view on RAA and GenIns on average over seeds. A triangle with
  no cells in the coming year gives a zero CDR, Pareto k of 0 and no
  refits. In Python `BayesOdp.one_year` and
  `one_year_segments` (`k_threshold=0.7`, `refit="refit_flagged"`,
  `"reweight_only"` or `"refit_all"`) return `BayesOneYearFit` and
  `BayesOneYearFits`; in R `bayes_odp_one_year()` and
  `bayes_odp_one_year_segments()` return a `bayes_one_year_fit` and a
  `bayes_one_year_fits`, with `fit`, `cdr`, `pareto_k`, `refitted`,
  `flagged_share` and the frames. Refits that fail their MCMC
  diagnostics raise a warning (#252).
- Selected development patterns: `SelectedPattern` on `Development`
  (`pattern`) gives benchmark factors that replace, or blend with, the
  factors estimated from the triangle, as age-to-age factors, factors to
  ultimate or percents developed (`PatternKind`, named by the user), for
  all ages or some (the rest stay estimated), with a weight per pattern or
  per age (`PatternWeight`). The chain ladder, `ExpectedLoss`,
  `BornhuetterFerguson`, `Benktander`, `CapeCod`, `Mack` (with Merz and
  Wüthrich's one-year view) and `MackBootstrap` take it through their
  `Development`, `OdpBootstrap` through its own `pattern`. A selected
  factor carries no parameter error and a blended one `(1 - w)` of the
  estimate's, so a pattern lowers the reported uncertainty; the process
  error stays the triangle's. A cumulative pattern (factors to ultimate
  or percents developed) with skipped ages is interpolated, `ln CDF`
  linear in age between the given ages. Its level at its last age selects
  no factor: with no tail and a last age that is the oldest, the level is
  the tail, a constant with no parameter error, as chainladder-python's
  `DevelopmentConstant` keeps it (`ChainLadderFit::pattern_tail`, also a
  `pattern_tail` column of `totals()`); otherwise it is rescaled to the
  factors in use and the tail (`ChainLadderFit::pattern_rescale`; with a
  tail, percent developed at the oldest age times the tail factor is
  100%). In the one-year view a bootstrap without a pattern takes the
  method's (`OneYearPattern::Method`, the default, on `OdpBootstrap` and
  `MackBootstrap`; `OneYearPattern::Estimated` simulates from the
  triangle's own factors) and reads it as the method does (no implied
  tail when the method has a tail of its own); a bootstrap's own pattern
  must select the same factors and weights as the method's, in any order,
  kind or form of weight, and the tail it implies must be the one the
  method uses (its own, or the one its pattern implies). The ODP bootstrap
  keeps the origins' level error with every factor selected (about 73% of
  the unblended standard deviation on RAA); Mack's bootstrap keeps only
  the process error. Fits
  report the estimate next to the selection (`DevelopmentFit::selected`,
  `weight`, `blended_ldf()`, `blended_std_err()`, and five new columns of
  `development_table()`). Parity with chainladder-python's
  `DevelopmentConstant` and with R's `MackChainLadder` on the triangle's
  own factors (`reserving-v02.md`, decision 12). **Breaking (Rust):**
  `Development` is no longer `Copy` or `Eq`, and `ChainLadder`, `Mack`
  and the four expected-loss settings are no longer `Copy`; `Development`,
  `OdpBootstrap`, `MackBootstrap`, `DevelopmentFit` and
  `ChainLadderFit` gain public fields, and `Error` gains `Pattern` (#264).
- Selected development patterns in Python and R: `ChainLadder`, `Mack`,
  `ExpectedLoss`, `BornhuetterFerguson`, `Benktander`, `CapeCod`,
  `OdpBootstrap` and `MackBootstrap` (in R `chain_ladder()`, `mack()`,
  `expected_loss()`, `bornhuetter_ferguson()`, `benktander()`,
  `cape_cod()`, `odp_bootstrap()`, `mack_bootstrap()`, `odp_one_year()`
  and `mack_one_year()`) take `ldf` (a dict, or a named vector or
  data.frame in R, from age in months to value), `ldf_kind` (`"ldf"`,
  `"cdf"` or `"percent_developed"`, required with `ldf`) and `ldf_weight`
  (a number or a weight by age); the one-year views (R's
  `odp_one_year()` and `mack_one_year()`, Python's `OdpBootstrap.one_year`
  and `MackBootstrap.one_year`) also take `bootstrap_ldf`, default true:
  a bootstrap without a pattern of its own takes the method's. Chain-ladder and Mack fits gain `selected_ldf`,
  `ldf_weight`, `blended_ldf`, `estimated_std_err`, `pattern_rescale` and
  `pattern_tail`,
  and their `std_err` is the standard error of the factor in use
  (unchanged without a pattern) (#264).

### Pricing

- Ratemaking in `prospicio_pricing::ratemaking`: premium at the current
  rate level by the parallelogram method (`RateHistory`, for calendar or
  policy periods, any term, and changes to written or in-force policies),
  trend fits (`TrendFit`, exponential or linear) and factors, and the
  average written and earned dates that set trend lengths (`Period`,
  `year_fraction`), after Werner and Modlin's *Basic Ratemaking*. Python
  `prospicio.pricing`; R `rate_level()`, `trend_fit()`, `trend_factor()`,
  `average_written_date()`, `average_earned_date()`, `year_fraction()`
  (#179).
- The overall rate indication in `prospicio_pricing::ratemaking`: the loss
  ratio and pure premium methods over experience years projected to the
  forecast period (`LossRatioExperience`, `PurePremiumExperience`),
  expense and profit provisions with the permissible loss ratio
  (`Provisions`, `premium_based_expenses`), and credibility against a
  complement (`credibility_weighted`, `trended_present_rates`), matching
  Werner and Modlin's Appendices A to C. Python `prospicio.pricing`; R
  `loss_ratio_experience()`, `pure_premium_experience()`,
  `indicated_change()`, `indicated_rate()`, `permissible_loss_ratio()`,
  `premium_based_expenses()`, `credibility_weighted()`,
  `trended_present_rates()` (#180).

### Models

- REML smoothing for GAMs (`Smoothing::Reml`, mgcv's `method = "REML"`):
  the Laplace-approximate restricted likelihood with the observed
  information, an unknown dispersion estimated with the smoothing
  parameters (`GamFit::reml_scale`), for the Gaussian, Poisson, gamma,
  inverse Gaussian and binomial families, matching mgcv 1.9.1. Python
  `Gam(..., smoothing="reml")` and `GamFit.reml_scale`; R
  `gam_fit(smoothing = "reml")` and `@reml_scale`. New in
  `prospicio-math`: `linalg::symmetric_eigenvalues` (#263).
- Tensor product smooths for GAMs (`TensorSpline`, `Gam::tensor`):
  mgcv's `te(..., bs = "ps", np = FALSE)`, a smoothing parameter per
  margin, with GCV, UBRE or REML, matching mgcv 1.9.1. Searches with
  several smoothing parameters now polish a coarse coordinate pass
  jointly. Python `Gam(..., tensors=)`; R `gam_fit(tensors =,
  tensor_basis =)` (#263).

## 0.0.2 (2026-10-12)

### Breaking changes

- `MackBootstrap` centres the residuals by default (`centre_residuals`
  is `true` in Rust, `True` in Python, `TRUE` in R), so its results change.
  Pass `false` for England and Verrall's Appendix 1 as written (#163).
- Every Gamma draw changes: `Gamma::sample` uses Marsaglia and Tsang instead
  of the inverse transform. This moves `Dist::Gamma`, the Gamma process of
  the ODP and Mack bootstraps, and the bindings' `sample` at a fixed seed.
  Sampler versions are now recorded in `Provenance::samplers` (#164, #169).
- `Distortion` is `Clone`, no longer `Copy` (#165).
- `Counting::panjer_ab` returns an `Option` and means the (a, b, 1) class
  (#165).
- `OdpBootstrap` and `MackBootstrap` are no longer `Copy` or `Eq`; struct
  literals need `..Default::default()` (#171).
- The Python and R layer constructors keep a `premium` given without paid
  reinstatements; it used to be dropped (#170).

### Reserving

- `MackBootstrap::fit` and `fit_segments`: the lifetime (run-off) view,
  `MackBootstrapFit` and `MackBootstrapFits` (#159).
- `SegmentDependence` for joint reserves across lines in both bootstraps:
  `Independent` (default), `Synchronized` residuals, or
  `RankCorrelation { spearman }`, in the lifetime and one-year views (#171).
- Tails in both bootstraps' lifetime view: an estimated tail refitted per
  simulation, a constant one drawn with its standard error, Mack's
  `tail_sigma` or the ODP's process past the oldest age; reconciled with
  R's `MackChainLadder(tail = ...)` (#172).
- Closed-form tests pinning the Mack bootstrap against Merz and Wüthrich
  (#160).

### Distributions and probability

- Severities: `InverseGamma`, `InverseGaussian`, `Burr`, scaled `Beta`,
  `Truncated` and splicing (#167).
- Claim counts beyond (a, b, 0): zero-modified and zero-truncated,
  logarithmic, mixed Poisson (negative binomial, Delaporte,
  Poisson-inverse Gaussian), Neyman A, Pólya-Aeppli and empirical, under
  the new `CountDist` (#165).
- Distortions: `Ccoc`, `BiTvar`, `WeightedTvar`, `CappedLinear`,
  `CappedLogLinear`, `Lep`, `LinearYield`, `Beta`, `Mixture`, `Minimum`,
  `Convex`, with `g_inv`, `g_dual`, `mass`, `slope_at_one` and
  `distortion::calibrate` (#165).
- A faster Gamma sampler: the Mack bootstrap on ABC takes 0.02 s instead of
  39 s (#164).
- `Provenance::samplers`, `shares_streams`, `replays_same_draws` and
  `current_samplers()`; older files still load (#169).

### Aggregate and reinsurance

- Seasonal event times (`EventSet::with_seasonal_times`), loss corridors
  (`Layer::loss_corridor`) and sums insured tilted within a band
  (`Band::with_tilted_bounds`) (#162).
- The FFT grid sized automatically: `recommend_grid`, `fft_auto`,
  `round_bucket` (#168).
- Contract terms on `Layer`: deposit premium, rate on line, premium rate,
  ceding, sliding-scale and profit commissions, and swing rating
  (`LossSensitivePremium`) (#170).

### Pricing

- Natural allocation (`prospicio_pricing::natural`: `Portfolio`,
  `NaturalPrice`, `Pentagon`) and classical premium principles with
  calibration (`prospicio_pricing::classical`) (#165).
- Classical credibility (`prospicio_pricing::credibility`): limited
  fluctuation and Bühlmann–Straub, checked against `actuar::cm`; Python
  `prospicio.credibility` (#175).

### Packaging

- The R package installs with
  `remotes::install_github("EKtheSage/prospicio", subdir = "R/prospicio")`
  (#174).
- crates.io publishing through trusted publishing only (#173).

## 0.0.1 (2026-10-07)

The first release, to PyPI (`prospicio`) and crates.io (11 crates).

- **Reserving**: `Triangle`; chain ladder, Mack, ODP bootstrap and ODP GLM;
  expected loss, Bornhuetter–Ferguson, Benktander, Cape Cod, tails and
  Clark; the Merz–Wüthrich one-year view and a simulated one-year view by
  re-reserving on the ODP or Mack bootstrap; diagonal backtest and segment
  fits. Parity with R ChainLadder and chainladder-python.
- **Distributions**: severities and counts, `Grid`, the Pareto family,
  `PredictiveDistribution` (join, reorder, blend, Arrow IPC), distortions,
  allocation, copulas, EVT, the `Dist` enum, `Custom`, JSON save and load.
- **Aggregate and reinsurance**: compound distributions, collective model,
  towers, layer rating and tower matching, exposure curves (MBBEFD,
  tabulated, severity), sums insured on events, surplus treaties, risk
  profiles, reinstatements pro rata as to time, towers as JSON.
- **Pricing**: risk-loaded pricing.
- **Models**: GLM, GAM, elastic net and Tweedie (statsmodels parity,
  freMTPL2); `prospicio-nn`; `prospicio-bayes` (NUTS, Bayesian GLM,
  stacking); LightGBM and XGBoost adapters in Python and R.
- **Bindings**: Python and R across every lane.
