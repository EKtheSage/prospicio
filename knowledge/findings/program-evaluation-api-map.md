---
type: Finding
title: Evaluating a reinsurance submission with prospicio, what the API covers
description: Each step of evaluating an incoming reinsurance program (on-level and trend, development, experience rating, exposure rating, blending, loss model, treaty terms, risk metrics, pricing, portfolio impact, sensitivity, monitoring) mapped onto the Python API of main on 2026-10-12, with what ran, the caveats met and the gaps (issues 190 to 197).
tags: [reinsurance, pricing, python, api, experience-rating, exposure-rating, gaps]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-12T18:00:00Z }
sources:
  - id: stub
    resource: ../python/prospicio/prospicio_native.pyi
    title: prospicio_native.pyi, the signatures quoted here
  - id: aggregate-design
    resource: ../docs/design/aggregate.md
    title: docs/design/aggregate.md, the Next list (multi-year carry-forward is item 2)
  - id: issues
    resource: https://github.com/EKtheSage/prospicio/issues/190
    title: Issues 190 to 197, one per gap below
  - id: notebook
    resource: ../examples/submission_evaluation.ipynb
    title: submission_evaluation.ipynb, the steps run end to end (2026-10-13)
---

# What was checked

Two mapping passes ran snippets on invented inputs for each step of
evaluating a submission, and a verifier then re-ran the claims neither
had run and corrected them. The build was `prospicio` from `main` at
2026-10-12 in a scratch virtual environment, not CI, so nothing here
carries `verified`. Only the API behaviour matters; the numbers came from
made-up data. Status: **verified** (a snippet ran), **partial** (the
pieces exist, the assembly is by hand), **gap** (nothing in the API).

Two conventions to know first. Several names that look like methods are
properties, so `grids.net`, `pd.dims`, `pd.components`,
`layer.needs_sums_insured`, `pareto.alpha` and `glm_fit.coefficients` are
used without parentheses (calling them raises `TypeError`). Dates are
year fractions (`year_fraction(y, m, d)` converts a calendar date).

# The steps

## 1. On-level premium and trend

* `RateHistory(effective, changes, applies=None)` or
  `RateHistory.from_levels(...)`, then
  `.rate_level(Period.calendar(a, b) | Period.policy(a, b), term).on_level_factor`.
  Verified. Uniform writing and earning is assumed, you multiply the
  premium by the factor, and nothing checks that the period type fits a
  risks-attaching treaty. There is no loader for a submission's rate table.
* `trend_factor(rate, years)`, `TrendFit.exponential(times, values)` and
  `.linear`, `Period.average_written_date(term)` and
  `.average_earned_date(term)` for the trend periods. Verified.
  Frequency and severity trends combine by hand, `(1+f)*(1+s)-1`;
  `loss_trend` in `LossRatioExperience` is a factor, not a rate.
* A statistical trend with its standard error:
  `Glm("poisson", "log", dispersion="pearson")` with a log-exposure
  offset and a year covariate (`std_errors`, `p_values`), or a gamma GLM
  for severity. Verified for frequency. No combined trend object.
* `LossRatioExperience(earned_premium, losses, on_level_factor,
  premium_trend, development, loss_trend, lae_factor)` and
  `PurePremiumExperience`. Verified, but they are primary ratemaking
  tools with no layer or treaty notion.

## 2. Development

* `Triangle.from_long(origin, development, values, ...)`,
  `ChainLadder(average, sigma_interpolation, tail).fit(triangle, column)`,
  `Mack`, `BornhuetterFerguson(apriori, ...)`, `CapeCod`, `Benktander`,
  `ExpectedLoss`, `ClarkLdf`, `ClarkCapeCod`. Verified (Mack's attributes
  were listed and its total standard error checked). Development ages
  must be in months (12, 24, ...); 1, 2, ... raise "not on the
  development grid".
* BF and Cape Cod take their exposure as a triangle column, the latest
  value per origin, so the on-level premium goes in as a constant column.
  `apriori` is one float, but a per-year prior works: put
  `ELR_i * premium_i` in the exposure column and use `apriori=1.0`
  (ultimates [65.0, 78.0, 72.7, 74.8, 82.9] on the test triangle). This
  is not a gap.
* Uncertainty: `OdpBootstrap`, `MackBootstrap`, `BayesOdp`, each giving a
  `PredictiveDistribution`. Exists; not run in this pass.
* Reserve-side covers (ADC, LPT) on the same program go through
  `Tower.apply_aggregate` on a bootstrap result, joined with
  `join(same_simulations=True)`. Exists; not run.
* Excess-layer development: only a chain ladder on a triangle of layer
  losses, which is unstable on sparse data. No excess LDFs from a
  ground-up triangle, no benchmark blending, no Pareto-informed BF prior.
  Gap (#192).

## 3. Large-loss experience rating

* Trend each loss: `trend_factor(rate, years)` in a Python loop, with
  IBNER factors, index or stability clauses, FX and franchises supplied
  by hand. No feature. Gap (#190).
* As-if burning cost per layer and year: a loop of
  `Layer(name, limit, attachment, share, aggregate_deductible,
  aggregate_limit, reinstatements, premium, reinstatement_rates,
  pro_rata_time).ceded(losses)`, `Tower.ceded(losses)` or
  `.ceded_by_event(losses)`. Partial: annual terms are applied (4m, 2m
  and 4m recoveries with a 1m AAD and an 8m annual limit gave 8m), but
  there is no loss-record or experience-table object and no
  reconciliation of claims, triangle and premium. Gap (#190).
* As-if under past structures (changed retentions and limits, a prior
  quota share inuring, a changed policy-limit mix): `Tower.inuring` and
  `Layer.ceded` apply the current structure only. Gap (#191).
* Count IBNR for late-reported large claims is not modelled (#190).

## 4. Severity and frequency fitting

* Pareto family: `Pareto.fit(losses, t, reporting_thresholds=None,
  censored=None, weights=None, truncation=None)`,
  `PiecewisePareto.fit(...)`, `GeneralizedPareto.fit_riegel(...)`.
  Verified, with left truncation per loss and right censoring at a
  policy limit (alpha 1.85 uncensored, 1.757 with two censored losses).
* `Gpd.fit(exceedances)`, `PotTail.fit`, `mean_excess(draws, thresholds)`
  and `hill(draws, ks)`. Verified; exceedances only, no truncation or
  censoring. `LogAffinePareto` has no fit, and
  `local_pareto_to_piecewise(t, alpha)` takes a callable `alpha(x)`, not
  the object.
* Other families: an intercept-only
  `Glm("gamma" | "inverse_gaussian", "log")` fits the mean by maximum
  likelihood (shape from `1/dispersion`), and a lognormal is a Gaussian
  GLM on `log(y)`. Partial: no truncation or censoring outside the
  Pareto family, no Burr or Weibull fit. Gap (#194).
* Attritional plus large: `Mixture.splice([(w, body), (1-w, Pareto)],
  [0, t, inf])` and `Truncated(sev, lower, upper)`. Verified.
* Frequency: `claim_count(mean, dispersion)` (below 1 binomial, 1
  Poisson, above 1 negative binomial), `Count.mixed_poisson(mean, cv,
  mixing, shift)`, `Poisson`, `NegativeBinomial`, `Binomial`. Verified.
  Exposure-adjusted counts are `count * current / year` by hand; no
  over-dispersion test and no count trend adjustment.

## 5. Exposure rating

* Property: `RiskProfile(sums_insured, risks, curves, expected_losses,
  premiums, loss_ratio, lower, upper, spread)` with `Mbbefd(b, g)`,
  `Mbbefd.swiss_re(c)` or `TabulatedCurve(x, g)`;
  `.expected_loss()`, `.expected_layer_loss(limit, attachment,
  surplus_retention, surplus_lines)`, `.expected_surplus_loss(...)`.
  Verified, including bands spread between `lower` and `upper`. No curve
  fitting; log-log interpolation for tabulated curves is next in the
  Aggregate lane.
* Casualty: `ilf(severity, limit, basic_limit)`,
  `loss_elimination_ratio(severity, deductible)`,
  `severity_exposure_curve(severity, mpl, x)`. Verified. One curve per
  limit band is needed (a shared curve is wrong across bands), the
  profile is per risk (no per-policy aggregate or clash), and there is
  no ALAE treatment or ISO table.

## 6. Blending and matching a tower

* `LimitedFluctuation(p, k).credibility(n, standard)`,
  `buhlmann_straub`, `credibility_weighted(estimate, credibility,
  complement)`. Verified; the blend is by hand, per layer.
* `match_tower(attachments, layer_losses, frequencies=None,
  rule="minimize")`, `fit_references(layers, frequencies,
  default_alpha=2.0, rule)`, `fit_pml_curve(return_periods, amounts,
  tail_alpha=2.0, truncation=None)`, giving a `TowerModel` with
  `.frequency`, `.severity`, `.layer_loss(limit, attachment)`. Verified
  (a matched tower reproduced 3m xs 2m exactly). **Caveat**: they raise a
  `ValueError` only when no decreasing, convex excess-loss function
  meets the inputs. Other inconsistent inputs pass silently with wild
  alphas: `fit_references` on layer losses 6e5, 6e5, 1e5 returned alphas
  17.84 and 0.74, and `match_tower([1e6,2e6,5e6], [6e5,6e5,1e5])` gave
  0.97 and 5.09. Check the alphas.

## 7. The loss model

* `CollectiveModel(frequency, severity)` with `.layer_mean`,
  `.layer_std`, `.excess_frequency`, `.simulate(n_sims, seed)`. Verified.
  `pareto_extrapolation`, `alpha_between_layers`,
  `alpha_between_frequencies` and `ilf` extrapolate to other layers.
* `fft_auto(frequency, severity, log2=16, p, p_star) -> (Grid,
  CompoundReport, GridSize)`, `recommend_grid`,
  `fft(frequency, severity, points)`, with severities on a grid from
  `Grid.rounding(sev, step, points)`. Verified.
* Events: `simulate_events(frequency, severity, n_sims, seed)`,
  `RiskProfile.simulate`, `EventSet.with_uniform_times()`,
  `.with_seasonal_times(weights)`, `EventSet.from_years(years,
  sums_insured, seed, times)` (a user's year loss table enters here),
  `.totals()` and `.maxima()` for the AEP and OEP. Verified.
* `compound_moments(frequency, severity)` and `.approximate` need three
  finite moments and fail on a Pareto with alpha 1.6 or a matched tower
  with alpha below 3 ("the severity's first three moments must be
  finite"). Use the grid or simulation.

## 8. Treaty terms, on the grid or on events

* Terms: `Layer(...)` with `.with_rate_on_line(rol)`,
  `.with_premium_rate(rate, subject_premium)`,
  `.with_deposit_premium(amount)`, `Layer.quota_share`, `.surplus`,
  `.stop_loss`, `.with_sliding_scale`, `.with_profit_commission`,
  `.with_ceding_commission`, `.with_loss_corridor`,
  `.with_swing_rating`. Only one of `aggregate_limit`, `reinstatements`
  and `reinstatement_rates` is allowed; a sliding scale cannot combine
  with swing rating or paid reinstatements; a profit commission has no
  deficit or credit carry-forward.
* On events: `Tower(layers).apply(events) -> PredictiveDistribution`
  with dims `["kind", "layer"]` and kinds `gross`, `ceded`, `net`,
  `reinstatement_premium`, `swing_premium`, `ceding_commission`,
  `profit_commission`. Verified. Pro rata as to time needs events with
  times, else it raises. Surplus needs sums insured on the events (works
  on `RiskProfile.simulate` events, refuses others).
  `Tower.inuring([[stage 1], [stage 2]])` verified.
  `Tower.apply_aggregate(pd)` applies stop-loss, adverse-development or
  quota-share covers to any `PredictiveDistribution`. Verified.
* On the grid: `Tower.on_grid(frequency, severity, points) -> TowerGrids`
  with `.ceded`, `.gross`, `.net`, `.expected_reinstatement_premium`,
  `.expected_ceding_commission`, `.expected_profit_commission`. Verified.
  **Caveats**: `on_grid` refuses `pro_rata_time` ("a compound
  distribution has no event times"), `net` is `None` when a layer has
  annual terms, and a stage with annual terms cannot inure to a later
  stage. On one example the reinstatement premium was 369k exact as to
  amount against 188k simulated as to time, so the two bases differ by
  about 2x.

## 9. Risk metrics

* On events: `res.marginal(["ceded", name]) -> Sampled` with `.mean()`,
  `.variance()`, `.var(p)`, `.tvar(p)`, `.cdf(x)`, `.quantile(p)`, and
  `res.aggregate(["kind"])` for the whole tower. Verified.
* On the grid: `Grid.quantile`, `.cdf`, `.mean`, `.variance`, and
  `Distortion.tvar(p).measure(grid)` for TVaR (18.31m against 18.35m
  simulated). A `Grid` has no skewness method.
* `occurrence_layer_statistics(frequency, severity, share, limit,
  attachment)`, `occurrence_exceedance`, `occurrence_pml`. Verified.
  `pr_attach` is per claim, the simulated probability of attaching is
  per year; `pr_loss` agrees. Statistics are for occurrence layers only.

## 10. Pricing and the quote

* `price(losses, assets, *, cost_of_capital=None, distortion=None) ->
  Price` (`.premium`, `.capital`, `.assets`, `.loss_ratio`,
  `.return_on_capital`, `.margin`), `Distortion.tvar | wang | ccoc |
  proportional_hazard | dual_power`, `calibrate(family, dist, premium,
  assets, r0)`, `TowerGrids.price(distortion) -> TowerPrice`,
  `Portfolio.from_independent(units, grids)` and `.from_predictive(pd)`
  with `.calibrate` and `.price(distortion, assets, p, allocation)`.
  Verified.
* Expenses: `Layer.premium_for`, `.ceding_commission_for`,
  `.profit_commission_for`, `TowerGrids.expected_premium`,
  `premium_based_expenses`, `Provisions(variable, profit)`. Partial:
  brokerage and the assembly from loss cost to a gross and net quote are
  hand arithmetic. Gap (#197).
* Comparing with the cedant's price: `with_rate_on_line` or
  `with_premium_rate`, then `price` or `calibrate(family, dist, premium)`
  gives the implied distortion or return. Partial: no quote-against-
  indication exhibit (#197).
* Time value of money: no payment pattern, discounting or investment
  income. (`Pentagon.discount` is something else.) Gap (#195).

## 11. Portfolio impact

* `simulate(GaussianCopula(...), marginals, n_sims, seed, keys, dims)`,
  `PredictiveDistribution.join(parts, dim, same_simulations=False)`,
  `.reorder_groups(dim, correlation, seed)`, `capital(pd, distortion,
  method="euler")`, `price_portfolio(pd, assets, *, cost_of_capital)`.
  Verified (book TVaR99 110.88m, 115.03m with the treaty, marginal 4.15m
  against a stand-alone 12.86m).
* **Caveat**: a `Tower.apply` result cannot be cut down to its ceded
  component. Joining `res.aggregate(["kind"])` carries every kind, and
  its `total()` was 18.68m against the correct 10.09m (book plus ceded),
  a silent double count. The workaround is a hand-built
  `PredictiveDistribution(["segment"], [("new treaty",)], draws)` from
  `Sampled.draws`. Gap (#193).

## 12. Sensitivity and parameter uncertainty

* Frequency: `Count.mixed_poisson(mean, cv, mixing)` into
  `simulate_events`, `fft_auto` or `on_grid` (same mean, sd 2,524,156 to
  2,544,045 for Poisson against a mixing cv of 25%). Verified.
* Severity tail and trend: a manual loop over `Pareto(t, alpha)` or a
  scaled threshold `Pareto(t * trend_factor(...), alpha)`; no sweep
  helper. Verified. Scaling the threshold holds the excess frequency
  fixed, which is a modelling choice.
* Scenario weighting: `PredictiveDistribution.blend(models, weights,
  seed)` over scenario `Tower.apply` results. Verified; it is a mixture
  of scenarios, not a posterior.
* `GlmFit.covariance`, `.predict_distribution`, `simulate_from_means` and
  `BayesGlm` add parameter risk to frequency and severity GLMs. Exist.
  Bayesian credibility is a candidate with v0.7.

## 13. Monitoring after binding

`actual_vs_expected`, `pit` and `crps`. Exist; not run in this pass.

# Gaps

Each has a GitHub issue.

* **#190 Large-loss preparation.** Trend by date, IBNER and large-loss
  development, indexation or stability clauses, FX, franchises, count
  IBNR, and a loss-record or as-if burning-cost object with
  reconciliation of claims, triangle and premium. This feeds every
  excess price and is the step hurt most.
* **#191 As-if under past structures.** Re-stating history under past
  retentions, limits and inuring quota shares.
* **#192 Excess-layer development.** Excess LDFs from a ground-up
  triangle plus a severity curve, benchmark blending, a Pareto-informed
  BF prior.
* **#193 Select a component from a `Tower.apply` result.** So a ceded
  total can enter a portfolio join without a hand-built distribution
  and without the 18.68m double count.
* **#194 Severity fitting beyond Pareto.** Truncation and censoring for
  lognormal, gamma, Burr and Weibull, plus goodness of fit and threshold
  selection for large losses.
* **#195 Time value of money.** Payment patterns, discounting,
  investment income in the price.
* **#196 Cat XL inputs.** Event sets from a cat model, secondary
  uncertainty, the hours clause, merging a cat set with a non-cat one,
  and clash across lines (parked as large). Multi-year carry-forward for
  profit commissions and sliding scales is already planned in
  `docs/design/aggregate.md` (Next, item 2).
* **#197 Quote assembly and comparison with the cedant's price.**
  Loss cost to gross and net premium (brokerage, commission, margin) and
  a quote-against-indication exhibit.

Smaller items already on the roadmap or not worth an issue: log-log
interpolation for tabulated exposure curves, a skewness method on a
`Grid`, an over-dispersion test, a combined frequency and severity trend
object, grid-path statistics of aggregate covers, and a pro rata as to
time on the grid (simulation is the only route today).

# Run end to end (2026-10-13)

`examples/submission_evaluation.ipynb` ran every step above on a synthetic
submission ([the data](/datasets/synthetic-submission.md)); the run added
these, none of them in the first two passes. `test_submission_evaluation`
(`python/tests/test_examples.py`) runs the notebook on every PR, with fewer
simulations; the numbers below are from a local run at the default, and the
test asserts only the looser facts noted.

* **Policy limits.** A claim over a threshold is cut at its own risk's sum
  insured, so the severity of a book is a mixture of Pareto distributions cut
  at different limits. There is no object for it; the notebook averages one
  `Grid.rounding(Pareto(t, alpha), step, n)` per claim limit. Not in issues
  190 to 197, and no issue yet; nor is a fit of an exposure curve to data or a
  credibility function made for layer losses.
* **Truncation matters.** `Pareto.fit` with the per-claim `reporting_thresholds`
  gave alpha 1.720 on 100 developed, trended claims (true 1.6); without them
  1.242, with the censored flags also dropped 1.255 (the test asserts alpha
  within two standard errors of 1.6 and at least 0.2 above the fit without
  the thresholds). It returns no standard error (no issue yet), so alpha over
  the square root of the uncensored claims stands in.
* **Development from the listing is selected on the outcome.** A listing of
  claims whose latest incurred is over a threshold leaves out claims that were
  large early and fell below it, which biases age-to-age factors measured on
  every listed claim upward. Measuring each step only on claims over a higher
  threshold at the earlier valuation (750,000 against a 500,000 listing) avoids
  most of it; here the factors moved by under one point (1.044 to 1.035 at the
  second step).
* **Counts.** A `Triangle` of claim counts by report year developed with
  `ChainLadder` gives the factor for claims not yet reported (1.24 on the
  latest year), and an intercept-only Poisson `Glm` with a log offset gives the
  frequency level and a Pearson dispersion; a `year` term tests a trend in the
  adjusted counts.
* **`fit_references` error text.** `fit_references([(5.0, 5.0, 3.0)],
  [(5.0, 0.2)])` raises `ValueError: invalid parameter references = NaN: are
  inconsistent: ...`; the message prints `NaN` where the references should be.
  The inconsistent case (a layer loss above limit times frequency) is
  correctly refused. With layers only, as the notebook calls it, the matching
  returned a model for every one of 200 parameter draws and derived the
  claim count over the lowest attachment itself (0.60 a year against the
  experience model's 0.49); the pieces' alphas still run from 0.83 to 6.62.
  Only the layer means are pinned: against a smooth shape (the experience
  severity with its frequency scaled to each layer's loss) the top layer's
  99% TVaR was 18.66m against 15.86m and its technical rate on line 10.8%
  against 9.4%; the two lower layers moved by under half a point.
* **Smaller things.** `RiskProfile` refuses `lower=0` for a band (`lower`
  must be positive; use the smallest sum insured). `year_fraction(2016, 12, 31)`
  is 2017.0 (the grid puts the last day at the next year's start), so take an
  accident year from the date's own year, not from its fraction.
  `LimitedFluctuation` has no `p` or `k` attributes to read back.
  `CapeCod(trend).fit(...).trended_apriori` is at the price level of the
  valuation date (the end of the last origin year), one year earlier than the
  middle of the treaty year the notebook trends to. The notebook pools the
  Cape Cod ultimates trended to the treaty year instead, which is the same
  ratio (Cape Cod's estimate is the pooled ratio of its trended ultimates).
* **Where the price comes from.** With ten years and a few claims over the
  attachments, the credibility of the experience was 0.16, 0.07 and 0.03 for
  three layers, so the technical rate on line moved most with the Swiss Re
  curve parameter (47.1% at c=2, 35.1% at c=4, 41.1% at c=3 for 5m xs 5m) and
  with the clustering of claims (47.5% at a count dispersion of 3; the
  Pearson estimate from ten counts was 0.80 against a true 1.97), and by under
  two points with the tail alpha, the frequency, the trend, the loss ratio, the
  analysis threshold or a tripled credibility: gaps #190 and #194 matter less
  to the price here than the exposure curve, which the library cannot fit.
  The limited-fluctuation standard measures a burning cost's volatility, so
  it is a conservative weight for a fitted experience model.
* **Stand-alone capital against one price of risk.** At a 99% TVaR stand-alone
  standard the 20m xs 20m layer's technical premium (2.19m) was 2.3 times the
  premium under one proportional hazard distortion calibrated to the working
  layer (0.94m), because its TVaR (18.66m) is nearly its limit. Quote
  assembly at a capital standard is by hand (issues #197, #204).
