### Aggregate and reinsurance

- `TowerModel::diagnostics` warnings print each position at six
  significant figures (`the piece from 1.28998`, not every digit of the
  float); Python `TowerModel.diagnostics()` and R `tower_diagnostics()`
  carry the same text (#276).

### Distributions and probability

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

### Pricing

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
