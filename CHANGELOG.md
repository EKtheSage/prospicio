# Changelog

What each release of prospicio contains. Every crate, the Python package
`prospicio` and the R package share one version (`docs/release.md`). A
release PR adds its section here; work merged since the last release goes
under "Unreleased" until then.

## Unreleased

### Distributions and probability

- Third raw moments: `Severity::third_moment()` (`None` where a family has
  no closed form, infinite where the moment diverges) and
  `Counting::third_moment()` with `Counting::third_cumulant()`, for
  compound moments to third order (#183).

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
  `priority_kappa()`.
- Eventual ruin of the classical surplus process by the
  Pollaczeck–Khinchine formula, `aggregate`'s `pollaczeck_khinchine` and
  `eventual_ruin`: `ψ(u)` on the severity's grid, the capital for a ruin
  probability, and the Lundberg exponent, with the severity capped, in
  excess or stopped (`prospicio_aggregate::ruin`). Python
  `prospicio.aggregate.eventual_ruin`; R `eventual_ruin()`.
- Layer loss picks, `aggregate`'s `Aggregate.picks`: a severity grid
  reshaped so that its expected loss to each layer of a tower is a pick,
  rescaling the survival function within each layer and keeping the tail
  (`prospicio_aggregate::picks::layer_loss_picks`). Infeasible picks are
  an error. Python `prospicio.aggregate.layer_loss_picks`; R
  `layer_loss_picks()`.
- Pricing bounds for other risks, `aggregate`'s `PricingBounds` and
  `Bounds`: the range of a risk's price over every distortion that prices
  a reference risk at a premium, with the BiTVaRs attaining it, and the
  envelope of those distortions, both exact
  (`prospicio_pricing::bounds`). Python `prospicio.pricing.PricingBounds`,
  `distortion_envelope`; R `pricing_bounds()`, `distortion_envelope()`.
- The total cession of a stage with several layers on the grid,
  `TowerGrids::stage_ceded`, priced in `TowerPrice::stages`; its statistics
  (`aggregate`'s `Ceded` column), `occurrence_cession_statistics`; and an
  aggregate cover's statistics on any annual grid,
  `aggregate_layer_statistics`. Python `TowerGrids.stage_ceded`,
  `TowerPrice.stages`, `prospicio.reinsurance.occurrence_cession_statistics`,
  `aggregate_layer_statistics`; R `tower_on_grid()`'s `stage_ceded`,
  `tower_price()`'s `stage_<k>` rows, `occurrence_cession_statistics()`,
  `aggregate_layer_statistics()`.

### Examples

- `examples/reinsurance_pricing.ipynb` prices an occurrence tower, a
  program and a share, an aggregate limit with reinstatements, stop loss and
  aggregate deductible, retro, swing, profit commission and sliding scale,
  and distortion premiums, and compares 343 results (321 independent
  comparisons and 22 consistency checks) with Mildenhall's `aggregate` 1.0.1 (reference file `validation/reference/re_pricing_aggregate.csv`) (#186).

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
  diagnostics raised as a Python `UserWarning` or an R `warning()`.

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
