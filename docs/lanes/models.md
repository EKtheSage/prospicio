# Models lane

**Session:** Cloud (the Probability session). **Owns:** `crates/prospicio-models/`, `crates/prospicio-glm/`, `crates/prospicio-nn/`, `crates/prospicio-bayes/`, `validation/tests/models.rs`, `validation/tests/fremtpl2.rs`, `docs/design/models.md`

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

## Next

- the issues filed by the Guides lane on 2026-10-18 (#353, #355, #356, #358)
- [survival-models] survival models when claim-level reserving resumes (`docs/design/models.md`)
- [triangle-design-bridge] The triangle-to-design bridge and the diagonal backtest stay in the Reserving lane

## Done

- `prospicio-models`, `prospicio-glm` (GLM, GAM with P-splines, tensor product, cyclic and monotone smooths (mgcv and scam parity), GCV, UBRE or REML smoothing, elastic net, Tweedie; statsmodels parity incl. freMTPL2), `prospicio-nn`, `prospicio-bayes` (NUTS, Bayesian GLM, stacking), boosting adapters over LightGBM and XGBoost (Python and R) with quantile and dispersion models
- the model artifact format decided (versioned JSON, one envelope, per-format versions stamped at the oldest that reads the file)
- Bayesian hierarchical credibility (`prospicio_bayes::credibility`: Bühlmann–Straub and Jewell's hierarchy with the means integrated out, `actuar` parity at given structure parameters and grid-integration parity of the posterior; `credibility.md`; Rust, Python and R)
- posterior predictive checks (`prospicio_bayes::ppc`: test statistics with p-values, grouped too, and the LOO-PIT at parity with R loo; on `BayesGlmFit` and for any model's replicates; Rust, Python and R)
- ONNX import (`prospicio_nn::onnx`: files parsed at run time by Burn's `onnx-ir`, run in `f64` for the operators a tabular network exports, `OnnxPredictor` behind `Fitted`; onnxruntime parity)
- the Python and R bindings for `prospicio-nn` (Python `Cann`, `AttentionCann`, `TokenAttention`, `OnnxModel`, `OnnxPredictor` in `prospicio.models`; R `cann_fit()`, `attention_cann_fit()`, `attention_weights()`, `network_correction()`, `onnx_model()`, `onnx_run()`, `onnx_predictor()`; behind the bindings' default `nn` feature, user 2026-10-16)
- network artifacts (`CannFit` and `AttentionCannFit` `to_json` / `from_json` with every weight by its path, bit-exact; the shared envelope moved to `prospicio_models::artifact`; Python pickling, R `save_model()` / `load_model()`)
- a GAM's log-likelihood and AIC, R `deviance_score()`, `pit_from_draws()` and `pit_histogram()` (#360, #382)
- the negative binomial GLM with theta estimated (`NegativeBinomialGlm`, MASS `glm.nb` parity) and the Tweedie power estimate in Python and R (#352)
- nested GLM tests and diagnostics (`prospicio_glm::nested::f_test` and `likelihood_ratio_test`, `GlmFit::bic`, `metrics::vif`; R `anova` and `BIC` parity on quine) and the validation charts (`metrics::double_lift`, `loss_ratio_chart`, `roc` with its AUROC, `confusion_matrix`, `normalized_gini`); Python and R (#351)
- GLM residuals, hat values, Cook's distances and prediction errors (`GlmFit::diagnostics`, `prediction_se`, `sum_prediction_error`, `gcv`, `log_likelihood_at`; R `glm` parity on quine; Python, and R's `residuals()`, `hatvalues()`, `rstandard()`, `cooks.distance()` on a `glm_model`; #334)
- rating-variable terms (`Terms::interaction`, `poly`, `hinges`, `steps`, `bins`, `log`, `ordinal`, `orthogonal_poly` as R's `poly`, `natural_spline` spanning R's `ns`; Python `Terms`; R through its formula; #350) and out-of-fold target encoding with credibility (`encode::target_encode`; Python `target_encode`, R `target_encode()`; #357)
