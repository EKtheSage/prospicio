# Changelog

What each release of prospicio contains. Every crate, the Python package
`prospicio` and the R package share one version (`docs/release.md`). A
release PR adds its section here; work merged since the last release goes
under "Unreleased" until then.

## Unreleased

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
