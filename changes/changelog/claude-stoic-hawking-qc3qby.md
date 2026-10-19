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

### Pricing

- Log-log interpolation of tabulated exposure curves:
  `TabulatedCurve::with_interpolation(x, g, Interpolation::LogLog)` runs
  the curve as a power law between points (the first piece stays
  straight) and refuses a table it would make non-concave, naming the
  point where the elasticity rises. Linear stays the default. Python
  `TabulatedCurve(x, g, interpolation="log-log")`, R
  `tabulated_curve(x, g, interpolation = "log-log")` (#282).

### Models

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

### Breaking changes

- Rust: `PSpline` gains a public `kind` field (`SmoothKind`); code that
  builds a `PSpline` in full adds it or uses `PSpline::new` (#282).

### Aggregate and reinsurance

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

### Breaking changes

- Rust: `Layer` gains the public fields `profit_commission_carry` and
  `sliding_scale_carry`; code that builds it in full adds them (`None`)
  or uses the builders (#301).

### Models

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

### Distributions and probability

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

### Models

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

### Pricing

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

### Models

- The negative binomial GLM with its `theta` estimated:
  `prospicio_glm::negbin::NegativeBinomialGlm` alternates the GLM's IRLS
  with Newton's method on `theta` as `MASS::glm.nb` does, with `MASS`'s
  standard error of `theta`, the log-likelihood and an AIC that counts
  `theta` (parity on `quine`). Python `NegativeBinomialGlm` and the
  Tweedie GLM with its power estimated (`TweedieGlm`, `tweedie_profile`);
  R `glm_nb_fit()`, `tweedie_glm_fit()` and `tweedie_profile()` (#352).
### Distributions and probability

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

### Breaking changes

- `prospicio_models::Term` has new variants (`Interaction`, `Poly`,
  `Hinges`, `Steps`, `Bins`, `Log`, `Ordinal`, `OrthogonalPoly`,
  `NaturalSpline`), so an exhaustive `match`
  on it needs new arms (#350).
### Distributions and probability

- The worst VaR of a sum with unknown dependence by the Rearrangement
  Algorithm (Puccetti and Rüschendorf): `rearrangement::rearrange` on any
  columns and `worst_var` on `n` quantiles of each marginal, from a start
  given as is, rotated or shuffled by seed, so a result is reproducible
  (*Pricing Insurance Risk* Exercise 59 as its guide prints it). Python
  `prospicio.risk.rearrange` and `worst_var`, R `rearrange()` and
  `worst_var()` (#366).

### Aggregate and reinsurance

- The normal power and Wilson–Hilferty approximations of a compound sum
  (Bahnemann, CAS Monograph 2, section 4.4), defined by their quantiles:
  `ApproximationKind::NormalPower` and `WilsonHilferty`; Python
  `CompoundMoments.approximate("normal_power")` and
  `"wilson_hilferty"`, R `moment_approximation(m, "normal_power")` and
  `"wilson_hilferty"` (#329).

### Distributions and probability

- The third moment of a layer of any severity,
  `Severity::layer_third_moment(limit, attachment)`, by adaptive
  quadrature of the survival function (the limited third moment at
  attachment 0, the unlimited layer at an infinite limit); Python
  `layer_third_moment` on every severity, R `layer_third_moment()` (#329).

### Breaking changes

- `prospicio_aggregate::ApproximationKind` and `Approximation` have the
  variants `NormalPower` and `WilsonHilferty`, so an exhaustive `match` on
  them needs new arms (#329).

### Distributions and probability

- A normal distribution, `prospicio_prob::Normal`, on the whole real line,
  with its closed-form TVaR `mu + sigma phi(z_p) / (1 - p)` (*Pricing
  Insurance Risk*, section 4.3.3), and the ogive of grouped data,
  `prospicio_prob::Ogive` (`new` from tabulated `(x, F)`, `from_counts`
  from counts per band), a severity whose limited moments, VaR and TVaR
  are exact band by band (Exercise 50). Python
  `prospicio.distributions.Normal` and `Ogive`, R `normal_distribution()`,
  `ogive()` and `ogive_from_counts()` (#366).
