---
type: Finding
title: Evaluating a reinsurance submission with prospicio, what the API covers
description: Each step of evaluating an incoming reinsurance program (on-level and trend, development, experience rating, exposure rating, blending, loss model, treaty terms, risk metrics, pricing, portfolio impact, sensitivity, monitoring) mapped onto the Python API of main on 2026-10-12, updated 2026-10-13, with what ran, the caveats met and the gaps (issues 190 to 197, most now closed).
tags: [reinsurance, pricing, python, api, experience-rating, exposure-rating, gaps]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-13T18:00:00Z }
sources:
  - id: stub
    resource: ../python/prospicio/prospicio_native.pyi
    title: prospicio_native.pyi, the signatures quoted here
  - id: aggregate-design
    resource: ../docs/design/aggregate.md
    title: docs/design/aggregate.md, the Next list (multi-year carry-forward is item 2)
  - id: issues
    resource: https://github.com/EKtheSage/prospicio/issues/190
    title: Issues 190 to 197, one per gap below, and 204 to 206
  - id: experience-design
    resource: ../docs/design/experience-rating.md
    title: docs/design/experience-rating.md, large-loss preparation and the burning cost (#190, #191)
  - id: time-value-design
    resource: ../docs/design/time-value.md
    title: docs/design/time-value.md, payment patterns and the price over the payout (#195)
  - id: cat-inputs-design
    resource: ../docs/design/cat-inputs.md
    title: docs/design/cat-inputs.md, event loss tables, year loss tables, merging and the hours clause (#196; clash parked)
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

Update 2026-10-13: issues 190, 191, 195, 196 (except clash across lines),
197, 204, 205 and 206 are closed (PRs 223 to 226, 228, 229, 231, 222 and
210). The sections below keep the first-pass findings and say where each
is now closed. The new calls were run on main at 2168d76 in a scratch
build, not by CI.

Two conventions to know first. Several names that look like methods are
properties, so `grids.net`, `pd.dims`, `pd.components`,
`layer.needs_sums_insured`, `pareto.alpha` and `glm_fit.coefficients` are
used without parentheses (calling them raises `TypeError`); so are the
newer `BurningCost.loss_cost`, `PreparedLosses.complete_above`, every
`Quote` field, `Distortion.family` and `.parameters`, while
`EventSet.counts()` and `DevelopmentPattern.at(age)` are methods. Dates are
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

* The loss record: `LargeLossExperience(ids, accident, amount, *,
  threshold, valuation, fx)` (accident dates as year fractions, `fx` one
  rate or one per loss), then `.prepare(rate=..., to=...)`, where `to` is
  the average accident date of the prospective period, or
  `.prepare(factors=...)`, with `development=` a `DevelopmentPattern` (read
  at each loss's age at the valuation) or a factor per loss for IBNER.
  It returns `PreparedLosses`: `.complete_above` (the threshold times the
  largest trend factor, the level above which every year is complete;
  losses whose trended amount falls below it are left out) and
  `.to_dict()` (columns id, year, accident, amount, fx, development,
  ultimate, factor, trended). Closed (#190, PRs 223 to 225). This replaces
  the first pass's loop of `trend_factor(rate, years)` with IBNER and FX
  by hand.
* The burning cost: `PreparedLosses.burning_cost(towers, years, *,
  premium, development, index=None, ratios=None) -> BurningCost`, one
  tower or one per year applied to the trended losses, the layer totals
  developed (a factor per year, or a `DevelopmentPattern` read at each
  year's average age) and divided by the on-level premium. `development`
  is Clark's excess factor per year or, when `prepare` already developed
  each loss (IBNER), the count IBNR factor only; an excess factor that
  holds IBNER after IBNER in `prepare` counts it twice.
  `.loss_cost` is premium weighted per layer, `.mean_loss_cost` weights
  the years equally, `.to_dict()` gives each year's row. Annual terms on
  the layers still apply (`Layer.ceded` gave 8m from 4m, 2m and 4m
  recoveries with a 1m AAD and an 8m annual limit). An index clause is
  `IndexClause(franchise, kind="full" | "severe")` with a ratio per loss,
  on per-occurrence terms only (annual terms raise). Closed (#190, PRs 223
  to 225). Reconciling the claims with the triangle and the premium is
  still by hand; `examples/submission_evaluation.ipynb` does it in its
  section 1 checks.
* Development patterns: `DevelopmentPattern(ages, factors)` takes factors
  to ultimate (`.at(age)`, `.age_to_age`) at ages in years from the
  accident date (not the months of a `Triangle`), and
  `DevelopmentPattern.from_history(ids, age, amount, *, ages)` estimates
  one from each claim's valuations (the volume-weighted chain ladder on
  claims valued at both ages; no selection threshold, see "Development
  from the listing" below). Count IBNR is a pattern built from a
  `ChainLadder` fit on a claim-count triangle, passed as
  `burning_cost(development=pattern)`: an accident year's average age at
  12 months is half a year, so the cdf at 12, 24 and later months goes at
  ages 0.5, 1.5 and so on. Closed (#190, PR 224).
* As-if under past structures: `PreparedLosses.as_if(historical, years,
  proposed) -> AsIf` applies the program in force each year to that
  year's losses as reported (untrended, undeveloped, in the reported
  currency) and puts the result beside the trended losses under the
  proposed tower, to reconcile with the cedant's reported recoveries
  (`.to_dict()`: year,
  reported_losses, trended_losses, historical_layers, historical,
  proposed). Closed (#191, PR 226). A changed policy-limit mix is not
  restated (#191 closed without it; #213 covers the severity under a mix
  of policy limits).

## 4. Severity and frequency fitting

* Pareto family: `Pareto.fit(losses, t, reporting_thresholds=None,
  censored=None, weights=None, truncation=None)`,
  `PiecewisePareto.fit(...)`, `GeneralizedPareto.fit_riegel(...)`.
  Verified, with left truncation per loss and right censoring at a
  policy limit (alpha 1.85 uncensored, 1.757 with two censored losses).
  No standard error and no severity under a mix of policy limits (#213;
  since added: `Pareto.fit_with_standard_error`,
  `PiecewisePareto.fit_with_standard_errors` and `PolicyLimits`).
* `Gpd.fit(exceedances)`, `PotTail.fit`, `mean_excess(draws, thresholds)`
  and `hill(draws, ks)`. Verified; exceedances only, no truncation or
  censoring. `LogAffinePareto` has no fit, and
  `local_pareto_to_piecewise(t, alpha)` takes a callable `alpha(x)`, not
  the object.
* Other families: an intercept-only
  `Glm("gamma" | "inverse_gaussian", "log")` fits the mean by maximum
  likelihood (shape from `1/dispersion`), and a lognormal is a Gaussian
  GLM on `log(y)`. Partial: no truncation or censoring outside the
  Pareto family, no Burr or Weibull fit. Gap (#194; since added:
  `fit_severity("lognormal" | "gamma" | "weibull" | "burr", ...)` with
  reporting thresholds, censoring and weights, and standard errors).
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
* Cat model inputs: `EventLossTable(rate, mean, sd_independent,
  sd_correlated, exposure)` (secondary uncertainty a beta on
  [0, exposure] per row, whose sd is `sd_independent + sd_correlated`,
  added rather than combined in quadrature, with no correlation modelled
  across events; a mean and sd no such beta can have raise an error naming
  the row) with `.expected_annual_loss`, the exact
  `.exceedance_frequency(x)`, `.oep(x)` and `.occurrence_pml(p)`, and
  `.simulate(n_years, seed) -> (EventSet, row ids)` ready for any
  `Tower`. A year loss table in long form is `EventSet.from_long(years,
  losses, n_years, seed)`; `EventSet.merge(other)` joins a cat set with a
  non-cat set year by year (interleaved by time when both carry times);
  `EventSet.grouped(window_hours)` groups losses into occurrences by an
  hours clause (times are fractions of a year of 8760 hours). Note that
  `counts()`, `events(i)` and `times(i)` are methods. Closed (#196, PRs
  229 and 231), except clash across lines, which is not built (parked in
  `docs/design/cat-inputs.md` and `docs/roadmap.md`).
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
* On an annual loss grid: `Tower.on_annual_grid(annual) -> AnnualGrids`
  applies the tower to an annual loss already on a grid (a loss ratio
  times premium, a book total), each year's total as one loss: a quota
  share with a sliding scale or profit commission, or a stop loss, on a
  loss-ratio grid, with no claim count. It has `.ceded` (a list of `Grid`,
  one per layer at the placed share, so a layer with share c is on step c
  times h), `.gross` and `.net` (one `Grid` each),
  `.expected_premium`, `.expected_ceding_commission`,
  `.expected_profit_commission`, `.expected_reinstatement_premium` (one
  per layer) and `.on_points`. It replaces a loop over the grid with
  `Layer.ceding_commission_for` and `.profit_commission_for`; not for
  per-occurrence layers with a severity (use `on_grid`). Closed (#206, PR
  210).

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

* `price(losses: Sampled | PredictiveDistribution | Grid, assets, *,
  cost_of_capital=None, distortion=None) -> Price` (`.premium`,
  `.expected_loss`, `.capital`, `.assets`, `.loss_ratio`,
  `.return_on_capital`, `.margin`), with the assets a distortion risk
  measure of the loss (for example `Distortion.tvar(0.99)`). A `Grid` is
  accepted since PR 210 (#204), which replaces
  `Pentagon.solve(loss=g.mean(), assets=TAIL.measure(g),
  return_on_capital=r)` for a grid. Also `Distortion.tvar | wang | ccoc |
  proportional_hazard | dual_power`, `calibrate(family, dist, premium,
  assets, r0)`, `TowerGrids.price(distortion) -> TowerPrice`,
  `Portfolio.from_independent(units, grids)` and `.from_predictive(pd)`
  with `.calibrate` and `.price(distortion, assets, p, allocation)`.
  Verified. A calibrated distortion's `.family` and `.parameters` (both
  properties; `parameters` is empty for weighted TVaR, mixtures, minimums
  and envelopes) read it back without parsing `repr()` (#205, PR 210).
* The quote: `Quote.technical(price, *, expenses=0.0, brokerage=0.0,
  fixed=0.0, limit=None)` grosses the risk premium up, premium = (risk
  premium + fixed) / (1 - expenses - brokerage), and reports
  `.rate_on_line` and `.payback` (with a limit), `.multiple`,
  `.loss_ratio`, `.return_on_capital`, `.net_premium` and `.to_dict()`.
  Closed (#197, PR 222). Superseded first-pass text: "brokerage and the
  assembly from loss cost to a gross and net quote are hand arithmetic".
  `Layer.premium_for`, `.ceding_commission_for`, `.profit_commission_for`,
  `TowerGrids.expected_premium`, `premium_based_expenses` and
  `Provisions(variable, profit)` remain for the terms.
* Comparing with the cedant's price: `Quote.at(premium)` keeps the loss
  and the assets and gives the return at the quoted premium
  (`.return_on_capital`), and `.risk_premium` (what is left after the
  loads) is the premium to give `calibrate(family, dist, premium)` for the
  implied distortion. Closed (#197, PR 222). Superseded first-pass text:
  "no quote-against-indication exhibit".
* Time value of money: `PaymentPattern(times, shares)` or
  `PaymentPattern.from_development(ages, factors)` (factors to ultimate,
  for example a paid chain ladder's; the last factor must be 1), with
  `.duration`, `.discount_factor(rate)` and `.capital_years(rate)`, and
  `price_over_payout(price, pattern, *, rate=0.0, cost_of_capital=None,
  distortion=None) -> PayoutPrice` restates a one-period `Price` over the
  payout (give the same rule the `Price` was made with), with `.premium`,
  `.expected_loss`, `.discounted_loss`, `.discount_factor`,
  `.capital_years`, `.margin`, `.capital` and `.assets`. With a pricing
  distortion the premium is discounted with the losses, and a pattern
  that pays everything within the first year at a zero rate gives back the
  one-period price. Investment
  income is the discounting of the losses, and the cost of capital is a
  spread over the discount rate (`docs/design/time-value.md`, decision 5).
  Closed (#195, PR 228). Superseded first-pass text: "no payment pattern,
  discounting or investment income". (`Pentagon.discount` is something
  else.)

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

Each has a GitHub issue. Closed ones keep their first-pass text and say
what closed them.

* **#190 Large-loss preparation. Closed** (PRs 223 to 225). Trend by date,
  IBNER and large-loss development, indexation or stability clauses, FX,
  franchises, count IBNR, and a loss-record or as-if burning-cost object.
  Now `LargeLossExperience`, `PreparedLosses`, `BurningCost`,
  `DevelopmentPattern` and `IndexClause` (section 3). Reconciling claims,
  triangle and premium stays by hand.
* **#191 As-if under past structures. Closed** (PR 226). Re-stating
  history under past retentions, limits and inuring quota shares: now
  `PreparedLosses.as_if` (section 3). A changed policy-limit mix is not
  restated (#191 closed without it; #213 covers the severity under a limit
  mix).
* **#192 Excess-layer development.** Excess LDFs from a ground-up
  triangle plus a severity curve, benchmark blending, a Pareto-informed
  BF prior.
* **#193 Select a component from a `Tower.apply` result.** So a ceded
  total can enter a portfolio join without a hand-built distribution
  and without the 18.68m double count.
* **#194 Severity fitting beyond Pareto.** Truncation and censoring for
  lognormal, gamma, Burr and Weibull, plus goodness of fit and threshold
  selection for large losses.
* **#195 Time value of money. Closed** (PR 228). Payment patterns,
  discounting, investment income in the price: now `PaymentPattern` and
  `price_over_payout` (section 10).
* **#196 Cat XL inputs. Closed except clash** (PRs 229 and 231). Event
  sets from a cat model, secondary uncertainty, the hours clause and
  merging a cat set with a non-cat one are `EventLossTable`,
  `EventSet.from_long`, `.merge` and `.grouped` (section 7). Clash across
  lines is not built: it stays parked (`docs/design/cat-inputs.md`, Next,
  and `docs/roadmap.md`). Multi-year carry-forward for profit commissions
  and sliding scales is already planned in `docs/design/aggregate.md`
  (Next, item 2).
* **#197 Quote assembly and comparison with the cedant's price. Closed**
  (PR 222). Loss cost to gross and net premium (brokerage, commission,
  margin) and a quote-against-indication exhibit: now `Quote.technical`
  and `Quote.at` (section 10).
* **#204, #205, #206. Closed** (PR 210). `price` on a `Grid` at a chosen
  capital standard (#204), a distortion's `.family` and `.parameters`
  (#205), and `Tower.on_annual_grid` for a tower on an annual loss grid
  (#206).

Issues opened after the first pass: #211 (tower matching should warn on
implausible fits, the caveat in section 6), #212 (exposure rating
helpers: a tabulated curve from a severity, calibrating the Swiss Re c,
the CV of a layer loss, and `RiskProfile` pitfalls), #213 (severity under
a mix of policy limits, and a standard error from `Pareto.fit`) and #214
(small consistency items, some of the "Smaller things" bullet below).

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
  `Grid.rounding(Pareto(t, alpha), step, n)` per claim limit (issue #213).
  Nor is there a fit of an exposure curve to data (calibrating the Swiss Re
  c to experience is issue #212); no issue covers a credibility function
  made for layer losses.
* **Truncation matters.** `Pareto.fit` with the per-claim `reporting_thresholds`
  gave alpha 1.720 on 100 developed, trended claims (true 1.6); without them
  1.242, with the censored flags also dropped 1.255 (the test asserts alpha
  within two standard errors of 1.6 and at least 0.2 above the fit without
  the thresholds). It returns no standard error (issue #213), so alpha over
  the square root of the uncensored claims stands in.
* **Development from the listing is selected on the outcome.** A listing of
  claims whose latest incurred is over a threshold leaves out claims that were
  large early and fell below it, which biases age-to-age factors measured on
  every listed claim upward. Measuring each step only on claims over a higher
  threshold at the earlier valuation (750,000 against a 500,000 listing) avoids
  most of it; here the factors moved by under one point (1.044 to 1.035 at the
  second step). `DevelopmentPattern.from_history` measures every listed claim
  (1.044 at the second step) and has no selection threshold, so it does not
  correct for this; the notebook keeps its thresholded helper.
* **Counts.** A `Triangle` of claim counts by report year developed with
  `ChainLadder` gives the factor for claims not yet reported (1.24 on the
  latest year), and an intercept-only Poisson `Glm` with a log offset gives the
  frequency level and a Pearson dispersion; a `year` term tests a trend in the
  adjusted counts. The count pattern, as
  `DevelopmentPattern([0.5 + k for k in range(len(cdf))], cdf)` from the
  `ChainLadder` fit, can be passed as `burning_cost(development=pattern)`,
  which reproduced the notebook's factors (1.0172, 1.0404 and 1.2385 for 2023
  to 2025, with the record built with `valuation=2026.0`).
* **`fit_references` error text.** `fit_references([(5.0, 5.0, 3.0)],
  [(5.0, 0.2)])` raises `ValueError: invalid parameter references = NaN: are
  inconsistent: ...`; the message prints `NaN` where the references should be.
  The inconsistent case (a layer loss above limit times frequency) is
  correctly refused (the NaN message and warnings on implausible fits are
  issue #211). With layers only, as the notebook calls it, the matching
  returned a model for every one of 200 parameter draws and derived the
  claim count over the lowest attachment itself (0.60 a year against the
  experience model's 0.49); the pieces' alphas still run from 0.83 to 6.62.
  Only the layer means are pinned: against a smooth shape (the experience
  severity with its frequency scaled to each layer's loss) the top layer's
  99% TVaR was 18.66m against 15.86m and its technical rate on line 10.8%
  against 9.4%; the two lower layers moved by under half a point.
* **Smaller things.** `RiskProfile` refuses `lower=0` for a band (`lower`
  must be positive; use the smallest sum insured; issue #212).
  `year_fraction(2016, 12, 31)`
  is 2017.0 (the grid puts the last day at the next year's start), so take an
  accident year from the date's own year, not from its fraction.
  `LimitedFluctuation` has no `p` or `k` attributes to read back (issue
  #214 holds this, the `year_fraction` point and the Cape Cod price level
  below).
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
  analysis threshold or a tripled credibility: the gap #194 matters less to the
  price here than the exposure curve, which the library cannot fit (#212);
  #190 is closed.
  The limited-fluctuation standard measures a burning cost's volatility, so
  it is a conservative weight for a fitted experience model.
* **Stand-alone capital against one price of risk.** At a 99% TVaR stand-alone
  standard the 20m xs 20m layer's technical premium (2.19m) was 2.3 times the
  premium under one proportional hazard distortion calibrated to the working
  layer (0.94m), because its TVaR (18.66m) is nearly its limit. Quote
  assembly at a capital standard is now `price(grid, TAIL,
  cost_of_capital=r)` then `Quote.technical` (issues #197 and #204, closed);
  the notebook uses them.
