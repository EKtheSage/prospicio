# Design note: models and their life cycle

Status: **In progress** · v0.5 onward · Depends on: `distributions.md`, `predictive-distribution.md`, `triangle.md` · Lane: Models

## Goal

Fit predictive models of any kind — GLM, GAM, neural network, gradient
boosting, Bayesian — through one interface. Run each through the same life
cycle: specify, preprocess, resample, fit, evaluate, tune, compare, version
and monitor. Every model returns the shared distribution objects, so its
output feeds aggregation, reinsurance, risk measures and capital like any
other result. The same machinery fits a reserving triangle with several
models and compares them.

## What exists

- `prospicio-models`:
  - `Family` (Gaussian, Poisson, gamma, inverse Gaussian, binomial,
    negative binomial, Tweedie) with variance function, unit deviance,
    log-likelihood as statsmodels defines it, and `draw` (the response's
    process noise, through the `prospicio-prob` distributions);
  - `Link` (identity, log, logit, probit, cloglog, inverse, inverse
    squared, power);
  - `Terms` → `Coding` → `Design`: factor coding learned on training data
    and replayed on new data (unseen levels are an error), with offset and
    weights;
  - the `Model` and `Fitted` traits;
  - `metrics`: deviance, mean deviance, RMSE, MAE, Gini of the ordered
    Lorenz curve, lift bands, CRPS of draws, interval coverage;
  - `resample`: k-fold, grouped k-fold and time-ordered (calendar
    diagonal) splits, `cross_validate` and `grid_search`.
  - `compare`: `compare(candidates, design, y, splits, metrics)` scores
    every `Candidate` (any `Model`, or a fit-and-predict function, so
    different engines sit in one table) on the same splits and metrics.
    `Comparison` gives per-split scores, means, standard errors and the
    paired standard error of each model's difference from the best.
    Python `compare` / `Comparison`, R `compare_models`. The diagonal
    backtest on a triangle stays in the Reserving lane.
  - `stack`: `stacking_weights` (Yao et al., 2018: maximize the log score
    of the mixture over the simplex; EM, then active-set Newton checked
    against the KKT conditions) and `pseudo_bma_weights` (softmax of elpd,
    or pseudo-BMA+ with the Bayesian bootstrap), from pointwise held-out
    log densities of any model: PSIS-LOO for Bayesian fits, cross-validated
    log densities otherwise. `PredictiveDistribution::blend` mixes the
    models' simulations row by row. Parity
    (`validation/scripts/stacking_weights.py`): SLSQP as in BayesBlend
    (MIT), polished by Newton, to 1e-10; `loo::stacking_weights` stops
    early and agrees to about 1e-3.
  - Bayesian and hierarchical stacking (`prospicio_bayes::stacking`), sampled by
    NUTS: `BayesStacking` (a Dirichlet prior on one weight vector, by the
    additive-logistic map with its Jacobian) and `HierarchicalStacking`
    (Yao, Pirš, Vehtari and Gelman, 2022: `wᵢ = softmax(α + Bᵀ xᵢ)` against
    the last model, non-centred, with the priors of BayesBlend's
    `HierarchicalBayesStacking`, both its models: no pooling, and partial
    pooling (`Pooling`), where each model's slopes on the discrete
    covariates, and separately on the continuous ones, are drawn around a
    model-level mean drawn around a global mean, with half-normal slope
    scales; the first `discrete` covariates are the dummy codes. Adaptive
    priors multiply the prior scales by `δ = N^λ`, `λ ~ Exponential(rate)`,
    as BayesBlend's `adaptive`). `StackingFit`
    gives posterior mean weights at any covariates, the draws, R̂ and ESS.
    `PredictiveDistribution::blend_by_component` blends with per-component
    weights, every component drawing its model from the simulation's
    common uniform. Parity (`validation/scripts/stacking_grid.py`): exact
    posterior moments by grid integration, on the claim models' held-out
    densities (weakly informative) and on a synthetic case where the
    covariate decides the weights; means within 0.1 posterior sd, sds
    within 8%. The pooling model has too many parameters for a grid: its
    gradient is checked against finite differences, scales of 0 against
    the slopes they fix, and a four-region fit for convergence and the
    weights it should find. As BayesBlend warns, pooling with fewer than
    three covariates gives a funnel and divergences.
- `prospicio-glm`: `Glm` (family, link, dispersion fixed, Pearson or
  deviance-based) fitted by IRLS with offsets and prior weights,
  step-halving, and convergence on both the deviance and the coefficients;
  `GlmFit` with coefficients, standard errors, p-values, deviance, null
  deviance, dispersion, log-likelihood and AIC (statsmodels' conventions);
  `predict`, and `predict_distribution` drawing `β` from its normal
  approximation and each row's response from the family.
  `Glm::over_dispersed_poisson()` reproduces the Chain Ladder on a
  triangle's origin and development factors (tested).
- Parity: `validation/reference/glm_statsmodels.csv`
  (`validation/scripts/statsmodels_glm.py`): 10 GLMs on a synthetic
  600-policy portfolio (Poisson with exposure offset, quasi-Poisson, gamma
  with log and inverse links and with weights, inverse Gaussian, binomial
  with trials, negative binomial, Gaussian, Tweedie with weights), 144
  values at 1e-7 to 1e-9.
- Parity at scale: `validation/reference/fremtpl2_statsmodels.csv`
  (`validation/scripts/statsmodels_fremtpl2.py`), the claim frequency GLM
  of Noll, Salzmann and Wüthrich (2018) on all 678,013 policies of
  freMTPL2freq (OpenML 41214, fetched and prepared by
  `validation/scripts/fetch_fremtpl2.py`, checked by SHA-256, not
  committed): 49 coefficients from `Terms` (Area, VehPower, VehAge and
  DrivAge classes, BonusMalus, VehBrand, VehGas, log Density, Region),
  offset log(Exposure). Poisson coefficients at 1e-8, standard errors,
  quasi-Poisson and HC0 standard errors at 1e-7, deviance, null deviance,
  log-likelihood and AIC at 1e-10: 202 values. On the same design without
  the offset, a logistic regression of whether a policy has a claim
  (ClaimNb > 0) over all policies, and a gamma GLM with log link of the
  average claim amount on freMTPL2sev (OpenML 41215, also SHA-256 checked)
  aggregated per policy and joined to the portfolio (24,944 policies),
  weighted by the number of amounts (statsmodels' `var_weights`), each with
  HC0 standard errors and the gamma's Pearson scale at 1e-8: 303 more
  values, 505 in all. The fits take about 30 s in
  a release build (statsmodels: 2 m 30 s); `validation/tests/fremtpl2.rs`
  skips unoptimized builds and a missing file unless
  `RISK_RS_REQUIRE_FREMTPL2` is set, as CI's `fremtpl2` job sets it. Not
  done: glum (its elastic-net objective differs from glmnet's, which
  `ElasticNet` follows).
- Sandwich covariance (`GlmFit::robust_covariance`, `Robust::{Hc0, Hc1,
  Cluster}`): the bread is the inverse observed information, so a
  non-canonical link matches statsmodels (R's `sandwich` uses the expected
  information; the two agree for canonical links). HC1 scales by
  `n / (n - p)`; clusters use statsmodels' and Stata's
  `G / (G - 1) · (n - 1) / (n - p)`. Parity:
  `validation/reference/glm_robust_statsmodels.csv`
  (`validation/scripts/statsmodels_glm_robust.py`), HC0 and age-band
  clusters for all 10 GLMs, 100 values at 1e-7. statsmodels' GLM reports
  HC0 for `"HC1"`, so HC1 is tested as the rescaling. Python
  `GlmFit.robust_covariance` / `robust_std_errors`, R `robust_vcov`.
- GAM (`prospicio_glm::gam`): `Gam` = a `Glm` plus `PSpline` smooths, each
  replacing a numeric design column with a cubic B-spline basis on
  mgcv's `"ps"` knots, a second-order difference penalty, and mgcv's
  sum-to-zero reparameterization. Penalized IRLS; smoothing parameters by
  GCV (estimated dispersion) or UBRE (fixed), mgcv's `GCV.Cp`, searched on
  `ln λ` (one smooth) or by coordinate passes (several); or fixed.
  `GamFit` has effective degrees of freedom, the Bayesian posterior
  covariance `φ (XᵀWX + S)⁻¹` (mgcv's `Vp`), and the same `predict` and
  `predict_distribution`. Parity: `validation/reference/gam_mgcv.csv`
  (`validation/scripts/r_gam.R`), five GAMs (Gaussian, Poisson with
  offset, gamma): deviance, edf, scale, score, a coefficient and fitted
  values, at 1e-5 to 1e-6 where the smooth is clearly non-linear and
  looser where the optimal smoothing is effectively infinite.
- REML smoothing (`Smoothing::Reml`, decided 2026-10-14): mgcv's
  `method = "REML"`, the Laplace approximation to the restricted
  likelihood (Wood 2011), `D_p/(2φ) − l_s(φ) + ½ ln|XᵀWX + S| − ½ ln|S|₊ −
  (M_p/2) ln(2πφ)`. Checked in R against mgcv's own `gcv.ubre` before
  building: it matches only with `W` the observed (Newton) weights, which
  differ from the expected ones for a non-canonical link such as the gamma
  with a log link, and with `φ` mgcv's `reml.scale`, not `sig2`. So the
  determinant uses the observed information at the fit (the β̂ is the same
  PIRLS solution either way), `l_s` is mgcv's `fix.family.ls` (Gaussian,
  Poisson, gamma, inverse Gaussian, binomial; negative binomial and
  Tweedie are refused), `ln|S|₊ = Σ r_j ln λ_j + ln|P_j|₊` with each
  smooth's rank and log pseudo-determinant from its eigenvalues (Jacobi,
  `prospicio_math::linalg::symmetric_eigenvalues`), and an unknown `φ` is
  minimized with the score at each trial: `D_p/(n − M_p)` in closed form
  for the Gaussian and inverse Gaussian, by Brent on `ln φ` for the gamma.
  `GamFit::reml_scale` reports it; `dispersion` stays the `Dispersion`
  estimate (mgcv's `sig2`). The search is the coordinate search on
  `ln λ`, then Nelder–Mead jointly for several smooths. Parity: six REML
  cases in `gam_mgcv.csv` (the five GAMs above and two smooths). mgcv
  stops once the score settles to about 3e-7 relative; where the score is
  flat in `λ` prospicio reaches a lower score (by up to 2e-3 in 4,700), so
  those cases compare the score to 1e-6 and the fit loosely, and the test
  asserts that the score is never above mgcv's. The wavy single smooths
  agree to 1e-6.
- Tensor product smooths (`TensorSpline`, `Gam::tensor`, decided
  2026-10-14): mgcv's `te(x, z, bs = "ps", np = FALSE)`. Each margin is a
  cubic B-spline basis on mgcv's `"ps"` knots (5 functions by default),
  the basis is the row-wise Kronecker product (the last margin's index
  fastest), margin `i` contributes `I ⊗ … ⊗ P_i ⊗ … ⊗ I` with its own
  smoothing parameter, and the whole product is centered to sum to zero
  like a P-spline. `np = FALSE` matters: mgcv's default reparameterizes
  each margin before forming the Kronecker penalties, which is a
  different model, not a different basis for the same one; with it off,
  how mgcv scales the penalties or picks the constraint's basis leaves the
  fit unchanged. Smoothing parameters run over the P-splines, then each
  tensor's margins. For REML, a block with several penalties takes
  `ln|Σ λ_i P_i|₊` from its top `r` eigenvalues at each trial, `r` the rank
  of `Σ P_i`. The search for several smoothing parameters is now one coarse
  coordinate pass (a 64-cell scan of `ln λ`, then Brent to 1e-4) polished
  jointly by Nelder–Mead, for every criterion, and each trial starts IRLS
  from the previous trial's means; a single smooth keeps the full-precision
  scan, so its results are unchanged. Parity:
  `validation/reference/gam_tensor_mgcv.csv`
  (`validation/scripts/r_gam_tensor.R`, data `validation/data/
  gam_surface.csv`): a surface by REML and GCV, Poisson counts by REML,
  and a P-spline next to the tensor by REML, at 1e-5 and 1e-6 for the
  score; GCV with two parameters is flat and mgcv stops about 1e-9 above
  prospicio's score, so its fitted values are compared to 2e-4. Python
  `Gam(..., tensors=[["x", "z"]])`; R `gam_fit(..., tensors = list(c("x",
  "z")), tensor_basis = 5)`.
- Cyclic smooths (`PSpline::cyclic`, `SmoothKind::Cyclic`, decided
  2026-10-15): mgcv's `s(x, bs = "cp")`. `k` cyclic cubic B-splines on
  `k + 1` equally spaced knots from the period's start to its end (no
  0.1% widening, unlike `"ps"`), each function that starts below the
  period wrapped onto its top (`prospicio_math::spline::
  cyclic_bspline_basis`, as mgcv's `cSplineDes`), and the second-order
  difference penalty taken around the circle (`cyclic_difference_penalty`),
  which leaves only a constant unpenalized; the centering constraint then
  removes it, so a cyclic block's penalty has full rank. The period is
  `PSpline::period(lo, hi)` (mgcv's `knots = list(x = c(lo, hi))`) or the
  data's range, and new data outside it wraps around (mgcv's `cwrap`), so
  the fit and its derivatives meet at the ends. The GAM artifact writes
  the smooth's `kind` and version 2 for it (see "Model artifacts"). Parity:
  `validation/reference/gam_cyclic_mgcv.csv`
  (`validation/scripts/r_gam_cyclic.R`, data computed in the script and
  the test): the raw basis at the period's ends and inside, and the
  penalty, at 1e-12; Gaussian by GCV with the period and over the data's
  range, Gaussian and Poisson by REML: deviance, edf, scale, score, fitted
  values and the mean at both ends at 1e-6 (1e-5 for edf). Python
  `Gam(glm, [{"column": "month", "bs": "cp", "period": (0, 12)}])`; R
  `gam_fit(..., bs = "cp", period = list(month = c(0, 12)))`.
- Monotone smooths (`PSpline::increasing`, `PSpline::decreasing`,
  decided 2026-10-15): scam's `s(x, bs = "mpi")` and `"mpd"` (Pya and
  Wood 2015), not mgcv's `mono.con` and `pcls`, which hold only for a
  Gaussian response with fixed smoothing. Cubic B-splines on `k − 3`
  equal intervals over the data's range (not widened), coefficients
  `β₁ + Σ_{i≤j} exp(γᵢ)`: the basis is the cumulative sums
  `Σ_{i≥j} Bᵢ` for `j ≥ 2` (the constant first dropped, negated for
  decreasing), centered by its training column means as scam does (not by
  the Householder constraint), with a first-order difference penalty on
  `γ`, which leaves a straight line unpenalized. Fitting is Gauss–Newton
  IRLS in `θ` (`γ` on the monotone columns, the coefficients elsewhere):
  the working model is linearized at `θ` through the Jacobian
  `X diag(∂β/∂θ)`, each step capped so no `γ` moves by more than 2 (the
  linearized `exp` is poor further out, and uncapped steps halved for
  hundreds of iterations), and convergence is judged on `β`, since an
  increment that wants to be 0 sends its `γ` towards `−∞` in steps that
  never shrink. The fixed point solves the penalized score equations, so
  at the same smoothing parameter it is scam's estimate (scam itself takes
  full Newton steps). The edf is scam's, `tr((JᵀWJ − E + S)⁻¹ JᵀWJ)` with
  the Newton Hessian (`E` diagonal on the monotone columns,
  `Σᵢ wᵢ (yᵢ − μᵢ) μ'ᵢ/Vᵢ Xᵢⱼ exp(γⱼ)`), so GCV and UBRE select as scam
  does; REML is refused, as scam has none. The fit reports `β = exp(γ)`
  on the monotone columns, so prediction stays linear in the expanded
  design, and their covariance by the delta method from
  `φ (JᵀWJ + S)⁻¹`. New data outside the training range is clamped to
  it, so the smooth extrapolates flat and stays monotone (scam
  extrapolates the B-splines, which stop summing to 1 there and can turn
  back). scam scales each penalty, `S = c DᵀD`: a scam `sp`
  is `λ = c sp` here. Parity: `validation/reference/gam_monotone_scam.csv`
  (`validation/scripts/r_gam_monotone.R`, data computed in the script and
  the test): at scam's fixed `sp`, an increasing Gaussian and a decreasing
  Poisson fit, deviance, edf, scale and fitted values at 1e-7 and the
  increments at 1e-6; selected by GCV and UBRE, against scam at 1e-3 and
  against its score's refined minimum (`optimize` over scam's own fixed-sp
  fits, since scam's BFGS stops about 1e-4 short in edf) at 1e-9 for the
  score, 1e-6 for the fit and 1e-5 for the edf, where the score is flat;
  every fit is checked monotone on a grid. Python
  `Gam(glm, [{"column": "x", "bs": "mpi"}])`; R `gam_fit(..., bs = "mpi")`.
- `prospicio-bayes`: MCMC diagnostics for any sampler's draws, the estimators
  of Vehtari et al. (2021) as R's `posterior` implements them:
  rank-normalized split `rhat` (the larger of the bulk and folded
  versions), `ess_bulk`, `ess_tail`, `ess_quantile`, `ess_mean` and
  `mcse_mean`, with Geyer's initial monotone sequence on FFT
  autocovariances. Parity: `validation/reference/mcmc_posterior.csv`
  (`validation/scripts/r_mcmc_diagnostics.R`), five sets of four chains
  (independent, AR(1), a shifted chain, a wider chain, ties), 25 values at
  1e-10.
- `prospicio_bayes::elpd`: the expected log pointwise predictive density from
  a fit's pointwise log-likelihood draws: `lppd`, `waic` and `loo`
  (Pareto-smoothed importance sampling: the largest importance ratios of
  each observation replaced by the expected order statistics of a
  generalized Pareto fitted by Zhang and Stephens' estimator with Vehtari
  et al.'s weakly informative prior, truncated at the largest raw ratio),
  with `k̂` per observation as the reliability diagnostic. Written from the
  papers; parity with R's `loo` 2.6.0 (`validation/scripts/r_loo.R`):
  every estimate, pointwise ELPD to `1e-9` and `k̂` to `1e-6`. The
  smoothing itself is public as `psis(log_ratios, r_eff)`, returning the
  unnormalized smoothed log weights (shifted so the largest raw ratio is
  0) and `k̂`, for reweighting posterior draws outside LOO; the Bayesian
  one-year view in `prospicio-reserving` uses it (decision 13 of
  `reserving-bayes.md`, decided by the user 2026-10-15). A unit test checks
  that `loo` is `psis` applied to `-log p(yᵢ | θₛ)`, bit for bit.
- **Sampling: nutpie** (decided 2026-10-04). Native models (Bayesian GLM,
  Bayesian chain ladder, credibility) will sample with nutpie's Rust core
  `nuts-rs`, so R gets NUTS too. The nutpie Python package itself is not
  used. Only our own NUTS draws reach the diagnostics and ELPD today; taking
  external draws (nutpie or Stan traces, arviz `InferenceData`) is a
  longer-term idea (`reserving-bayes.md`, open decision 11), not built.
- **Bayesian GLM** (`prospicio_bayes::glm`): `BayesGlm` samples a GLM's
  posterior with `nuts-rs` 0.19 (MIT): normal priors on the coefficients
  (`intercept_sd`, `prior_sd`), and for the Gaussian, gamma and inverse
  Gaussian a half-normal prior on the dispersion, sampled on the log scale
  with an analytic gradient (`prospicio_math::special::digamma` for the gamma).
  Chains run on Rayon, start at the maximum-likelihood fit with a small
  jitter, and replay exactly: chain `c` keys its ChaCha20 generator from
  stream `c` of the seed. `BayesGlmFit` gives draws, a summary (mean, sd,
  quantiles, R̂, bulk and tail ESS), divergences, the pointwise
  log-likelihood, PSIS-LOO with `r_eff` from the chains, posterior-mean
  predictions and posterior predictive draws. Parity
  (`validation/scripts/bayes_glm_grid.py`): exact posterior moments by
  grid integration for a Poisson with exposure and a Gaussian with
  sampled dispersion; means within 0.1 posterior sd, sds within 8%.
  Python `BayesGlm` / `BayesGlmFit`, R `bayes_glm_fit`, `bayes_loo`.
- **Any posterior** (`prospicio_bayes::nuts`): the NUTS driver is public. A
  model implements `LogDensity` (dimension, log density and gradient on an
  unconstrained vector, `None` outside the support) and `sample(&density,
  start, Sampler)` returns `PosteriorDraws`: chains, divergences, a
  summary, `transform` to the natural scale (keeping chains, so the
  diagnostics are on that scale), and `predictive`, which picks a
  posterior draw per simulation and lets the model simulate the outcome,
  giving a `PredictiveDistribution` with parameter and process
  uncertainty. Bayesian models in other crates (for example a Bayesian
  reserving model) sample through it without touching `nuts-rs`.
  `BayesGlm` and stacking share the driver. Tests: a correlated normal,
  a half-line support, exact replay, and the gamma-Poisson posterior
  predictive against the exact negative binomial. No Python or R
  binding: a density written in Python or R would call back into the
  interpreter at every gradient. Passing external traces to the
  diagnostics is not built yet (see above).
- **Bayesian hierarchical credibility** (`prospicio_bayes::credibility`):
  Bühlmann–Straub's model and Jewell's hierarchy of it as a normal
  random-effects model, the group and node means integrated out in closed
  form so NUTS samples only the standard deviations; the premiums are
  the credibility formula averaged over their posterior. Designed and
  validated in `docs/design/credibility.md` ("Bayesian hierarchical
  credibility"): parity with `actuar::cm` at given structure parameters,
  and with grid integration of the posterior.
- Decision: families and links are closed enums, like `Distortion`, so a
  fitted model serializes as data.

## Crates

Crates split by dependency weight, never by concept (`architecture.md`:
"split a crate only when compile time, dependency weight, or independent
release cadence forces it"). One light crate holds the interface and the
life cycle; each heavy engine sits in its own crate behind a feature flag.

| Crate | Holds | Heavy dependencies |
|---|---|---|
| `prospicio-models` | The `Model` trait, model specs, `Design`, `Family` and links, resampling, metrics, tuning, comparison and stacking, model artifacts | None |
| `prospicio-glm` | GLM by IRLS; GAM as a penalized GLM on the same solver (P-splines, tensor smooths, GCV/REML) | `faer` |
| `prospicio-nn` | Neural networks on Burn: CANN (a GLM offset plus a network correction) and the attention CANN (a transformer over feature tokens) | Burn, opt-in feature |
| `prospicio-bayes` | Bayesian model specs and diagnostics (R-hat, ESS, divergences, ELPD by PSIS-LOO and WAIC); sampling through `nuts-rs` (nutpie's Rust core) | Samplers, opt-in feature |
| *(no crate)* | Gradient boosting: Python and R adapters over LightGBM and XGBoost that implement the interface and return shared objects | None in Rust |

- **Families live in `prospicio-models`, distributions in `prospicio-prob`.** A
  `Family` (variance function, unit deviance, log-likelihood, canonical
  link) is shared by the GLM, the network's loss, the boosters' objective
  and the Bayesian likelihood. It names the `prospicio-prob` distribution it
  predicts: `Poisson`, `NegativeBinomial`, `Binomial`, `Gamma`, `Tweedie`,
  and later the inverse Gaussian. `prospicio-glm` does not own families, so
  `prospicio-nn` never depends on `prospicio-glm`.
- **GAM lives inside `prospicio-glm`.** A GAM is a penalized GLM: same IRLS loop,
  plus a penalty and smoothing-parameter selection. A separate crate would
  duplicate the solver.
- **Chain Ladder never compiles Burn or a sampler.** `prospicio-nn` and
  `prospicio-bayes` are opt-in features of the bindings, and nothing in the
  default build depends on them.

## Life cycle

The stages are modules of `prospicio-models`, with matching submodules in Python
(`prospicio.models.*`) and R. This takes tidymodels' life cycle, not its
package split: a stage becomes a crate only if it acquires a heavy
dependency.

| Stage | Module | tidymodels analogue | What it does |
|---|---|---|---|
| Specify | `spec` | parsnip | An engine-agnostic spec (family, link, terms, penalty, hyperparameters) that compiles to an engine |
| Preprocess | `design` | recipes | Builds the design matrix (factor coding, splines, interactions, offset, exposure, weights). Fitted on training data and replayed exactly on new data |
| Resample | `resample` | rsample | K-fold, grouped and time-ordered splits, plus calendar-diagonal splits for triangles |
| Fit and predict | `Model` | parsnip, workflows | `fit`, `predict`, `predict_distribution`, `score`, `diagnostics` |
| Evaluate | `metrics` | yardstick | Deviance, log-likelihood, log score, Gini, Lorenz, lift, double lift, CRPS, randomized PIT (with a histogram and KS distance from uniform), interval coverage, actual vs expected; for Bayesian fits, ELPD (PSIS-LOO, WAIC) |
| Tune | `tune` | tune, dials | Grid or random search over a parameter space, each candidate scored on the resamples. Folds run in parallel on the Rayon pool (`resample::map_splits`, results in split order) and replay from a seed |
| Compare and stack | `compare`, `stack` | stacks | One table across engines; stacking weights from out-of-sample scores |
| Version and monitor | `artifact` | vetiver | A serialized fitted model with its provenance (spec, design, data hash, seed, package version), and a monitor of actual vs expected on new periods |

### The interface

```rust
trait Model {
    type Fitted: Fitted;
    fn fit(&self, design: &Design, y: &[f64]) -> Result<Self::Fitted>;
}

trait Fitted {
    fn predict(&self, design: &Design) -> Result<Vec<f64>>;
    /// Joint across the rows of `design`: draws × rows, with process and
    /// parameter uncertainty.
    fn predict_distribution(&self, design: &Design, n_sims: usize, seed: u64)
        -> Result<PredictiveDistribution>;
    fn score(&self, design: &Design, y: &[f64], metric: &Metric) -> Result<f64>;
    fn diagnostics(&self) -> Diagnostics;
}
```

Python and R wrap this with the conventions of each language (sklearn's
`fit`/`predict`/`get_params` in Python, S7 generics in R).
`predict_distribution` is joint: anything that is summed later (cells to
reserves, policies to a portfolio) needs the joint draws, not per-row
marginals.

- **GLM:** parameter draws from the estimator's asymptotic normal, plus
  the family's process noise; or a bootstrap.
- **Bayesian:** posterior predictive draws, natively.
- **Neural network, gradient boosting:** bootstrap refits, or a quantile or
  distributional head where the engine has one.

### Model artifacts

A fitted model serializes with what governance needs to reproduce and
audit it: the spec, the fitted design (factor levels, knots, scaling), the
estimates, a hash of the training data, the seed and stream scheme, and
the package version. It reuses the `Provenance` that `PredictiveDistribution`
carries. Monitoring compares a stored model's predictions with actuals as
new periods arrive, and reports actual vs expected and calibration drift.

Done for GLMs: `GlmFit::to_json` / `from_json` write and read a JSON
artifact (`format: "risk_rs.glm_fit"`, `format_version: 1`) with the spec,
estimates, unscaled covariance, fit statistics, fitted values and
provenance (crate version, `input_hash` of the design, offset, weights and
response, recorded at fit time). Numbers round-trip bit for bit
(serde_json's `float_roundtrip`); non-finite values are the strings
`"NaN"`, `"inf"`, `"-inf"`. A reader refuses a newer format version. The
artifact holds no training data, so a loaded fit predicts and simulates
but gives no sandwich covariance. The coding (factor levels) stays with
the binding: Python keeps its `Coding`, R's `save_model` / `load_model`
store the formula terms and levels beside the artifact in an RDS file,
and Python's `GlmFit` pickles through the artifact. `GamFit`
(`"risk_rs.gam_fit"`, with each smooth's knots and centering constraint,
so new data gets exactly the training basis) and `ElasticNetFit`
(`"risk_rs.elastic_net_fit"`, one per `λ`; R saves the whole path) work
the same way.

**The format, decided 2026-10-15.** Model artifacts are versioned JSON,
for every engine; CBOR and Arrow are not used for them.

- **One envelope.** Every artifact is a JSON object with `format`
  (`"risk_rs.<kind>"`), `format_version` and `provenance` (crate version,
  `input_hash`), then the kind's own fields, as `"risk_rs.distribution"`
  and `"risk_rs.tower"` already are. The name keeps its `risk_rs` prefix:
  renaming it would break every saved file for no gain.
- **Why JSON.** An artifact is read by people auditing a model and by
  tools that are not prospicio (a reviewer's script, a model inventory),
  so it must be readable without our code; JSON is, and it diffs. Numbers
  round-trip bit for bit already. Size does not argue for CBOR: the
  arrays are the coefficients, the covariance (`p²` values), the knots
  and constraints, and the fitted values (one per training row), at about
  20 bytes a number; only the fitted values grow with the data, a few
  megabytes for a million rows. If an artifact ever needs arrays far
  larger than that (draws, a large network's weights), those arrays go
  into an Arrow IPC file beside the JSON under a new format version, as
  `PredictiveDistribution` already stores draws, rather than turning the
  whole artifact binary.
- **Versions are per format.** Each format has its own version and each
  reader the newest it knows (`artifact.rs`: `FORMAT_VERSION`, and
  `GAM_FORMAT_VERSION` = 2 since cyclic and monotone smooths). A writer
  stamps the oldest version whose readers read that artifact correctly,
  so a GAM with only ordinary smooths stays version 1 and older builds
  still load it. A reader refuses a newer version. A new field that an
  older reader would ignore and so misread (a smooth's `kind`) bumps the
  version; one an older reader cannot ignore without failing its own
  checks (the tensors, which change the number of smoothing parameters)
  or that changes nothing it computes needs no bump.
- **Delegated and later engines.** A boosting adapter, when it saves,
  puts the engine's own text model (LightGBM's model string, XGBoost's
  JSON) in a field of the same envelope; a network (`prospicio-nn`) or a
  Bayesian fit (`prospicio-bayes`) uses the envelope with its weights or
  draws as above. None of them saves yet. The JSON helpers stay in
  `prospicio_glm::artifact` until a second crate writes an artifact; then
  the envelope (the format check, the version rule, the non-finite
  numbers) moves to `prospicio_models::artifact`, the module the life
  cycle table names.
- **The coding stays with the bindings**, as above: factor levels and
  formula terms are front-end objects (Python's `Coding`, R's `terms`), so
  the R package stores them beside the artifact in its RDS file and the
  Rust artifact carries only the numeric design it was fitted on.

Monitoring (`prospicio_models::monitor::actual_vs_expected`): a stored model's
predictions against actuals by period, `A = Σ w y` and `E = Σ w μ`, each
period's z-score `(A - E) / √(φ Σ w V(μ))` from the model's own variance
function, and a drift test: the precision-weighted slope of `A / E - 1`
across periods with its standard error. `metrics::lift` gives the same
split by predicted rate. Python and R `actual_vs_expected`.

## Fitting a triangle with several models

Models never see a `Triangle`. The Reserving lane owns a bridge that turns
a triangle into a design, and the backtest that scores models on it.

1. **Triangle to design** (`prospicio-reserving`). One row per origin ×
   development cell: the incremental value, exposure, and features (origin,
   development and calendar period, as factors or numbers). Observed cells
   are the training rows; future cells, below the latest diagonal, are the
   prediction rows.
2. **Any model fits the design.**
   - The ODP model is a quasi-Poisson GLM with origin and development
     factors. Its fitted future cells reproduce Chain Ladder exactly, which
     is a built-in parity test.
   - A GAM smooths over development age; a Bayesian hierarchical model
     pools across origins; a CANN or a booster adds what the GLM misses.
3. **`predict_distribution(future cells)`** returns draws × cells, joint.
4. **The shared tools apply to every model.** `aggregate(["origin"])`
   gives reserves by origin, then the total, VaR, TVaR, `capital()` and the
   one-year view.
5. **`compare(models, triangle)` backtests on calendar diagonals.** It
   refits with the latest k diagonals removed, predicts them, and scores
   CRPS, interval coverage and actual vs expected. Diagonals, not random
   cells, because reserving forecasts the next calendar period. The scores
   can weight a stacked blend.

## Build order

1. `prospicio-models`: `Design`, `Family` (Poisson, negative binomial, binomial,
   gamma, Tweedie), links, `Model`, metrics.
2. `prospicio-glm`: IRLS with offsets, weights and exposure; quasi-likelihood
   dispersion; elastic net. Parity with statsmodels and glum on freMTPL2.
3. The triangle bridge and the ODP GLM in the Reserving lane, checked
   against Chain Ladder.
4. Resampling, tuning and comparison.
5. GAM in `prospicio-glm`.
6. `prospicio-nn` on Burn: CANN first, then the attention CANN.
7. `prospicio-bayes`, and the boosting adapters.

## Elastic net

`prospicio_glm::net::ElasticNet` minimizes glmnet's objective,
`Σ wᵢ dᵢ / (2 Σ w) + λ Σ pfⱼ ((1 - α)/2 bⱼ² + α |bⱼ|)`, with `bⱼ` the
coefficient of column `j` standardized to unit weighted (population)
standard deviation, by coordinate descent inside IRLS, so it serves every
family and link the GLM does. The first all-ones column is the
unpenalized intercept; penalty factors are rescaled as glmnet does;
coefficients are reported on the design's scale. `lambda_max`,
`lambda_path` (log-spaced, glmnet's defaults) and `path` (warm starts)
build the regularization path. `cross_validate` scores the whole path on
resamples with warm starts (glmnet's `cv.glmnet`: the folds' mean deviance
weighted by fold weight, its standard error, `lambda_min` and the sparser
`lambda_1se`); α takes a small grid (0, 0.5, 1) over the same splits. Predictive distributions
carry process uncertainty only: penalized estimates have no standard
errors, so parameter uncertainty comes from bootstrapping the fit.

Parity with glmnet (`validation/scripts/r_glmnet.R`; glmnet is GPL-2 and
used for reference values only): Gaussian (standardized and not,
weighted), Poisson with offset (lasso and ridge), binomial and gamma with
log link, at four λ each. Two notes:

- For the Gaussian, glmnet scales `y` by its standard deviation `s_y`
  before fitting, which divides the ridge part of its penalty by `s_y`.
  Our objective is the stated one; the parity test maps glmnet's
  `(λ, α)` to ours. The lasso (`α = 1`) is unaffected.
- With correlated columns coordinate descent creeps: glmnet's default
  `thresh = 1e-7` stops with coefficients off in the third digit, and
  even `1e-14` leaves them 5e-6 off. The references use `1e-20`.

## Tweedie power

A Tweedie GLM's power `p` is a hyperparameter: IRLS gives the
coefficients for a given `p`, and `φ` only scales the variance.
`prospicio_glm::tweedie::tweedie_profile` chooses `p` by profile likelihood, as
R's `tweedie.profile`: for each `p` on a grid it fits the GLM, maximizes
the exact Tweedie log-likelihood (Dunn and Smyth's series density from
`prospicio-prob`, row `i` at `φ / wᵢ`) over `φ`, and refines the best `p` by
golden-section search between its grid neighbours.

`prospicio_glm::tweedie::TweedieGlm` estimates `p` as part of the fit. Since the
coefficients at a fixed `p` do not depend on `φ`, the joint maximum
likelihood over coefficients, `p` and `φ` (what H2O's GLM does with a free
variance power) is the maximum of this one-dimensional profile; Brent's
method (`prospicio_math::optimize::brent`) finds it in about a dozen GLM fits,
each warm-started from the last. The fit reports a 95% profile-likelihood
interval for `p` (where the profile is 1.92 below its maximum, found by
the Illinois method) and flags a power at a search bound: with no zero
claims the likelihood pushes `p` towards 2, and the bound is then not an
interior maximum. Standard errors and predictive draws are conditional on
`p`. Parity: statsmodels fits plus an independent NumPy series density
(`validation/scripts/statsmodels_tweedie_profile.py`). Cross-validated
deviance cannot choose `p`, since each `p` has its own deviance scale.

**Decided: no Fourier-inversion density.** Dunn and Smyth (2008) evaluate
the Tweedie density by Fourier inversion where their series struggles, and
H2O uses both. Against 40-digit mpmath, `prospicio-prob`'s series is accurate to
`5e-11` or better, in under 0.4 ms per point, for `p` from 1.01 to 1.99,
`φ` from 0.001 to 50 and `y` from 0.01 to 10 times the mean. It degrades
only at `φ = 1e-6` with `p` near 2 (`5e-8`, about 11 ms): a coefficient of
variation near 0.001, which claims and pure premiums never have. The
inversion's oscillatory integrals and series acceleration are not worth
that. If such regimes ever matter, compute the series' large log-gamma
differences more precisely instead.

## Attention

`prospicio_nn::AttentionCann` is a CANN whose correction is a transformer over
feature tokens (Gorishniy et al.'s FT-Transformer in Wüthrich and Merz's
CANN setting; Richman, Scognamiglio and Wüthrich's Credibility Transformer
develops the same pairing). Each numeric column is a token, and each
factor's indicator columns (`region[B]`, `region[C]`, …) together form one
token, so every level gets a learned embedding. A learned `[CLS]` token
joins them; pre-norm blocks of multi-head self-attention and a
feed-forward layer mix them; the `[CLS]` state, through a zero-initialized
head, corrects the GLM's linear predictor. Training starts exactly at the
GLM.

The point for actuaries is that attention is readable per risk:
`AttentionCannFit::attention` returns, for every row and head, the last
block's weight from `[CLS]` to each feature. Averaged over a segment's
rows, it shows which rating factors the model leans on there, which is how
attention reaches "different parts of the data": the weights differ by
row, so they differ by segment, and different heads can specialize. It is
a diagnostic, not a decomposition of the prediction.

Not yet: attention along development periods or accident years (a
sequence model over a triangle's cells), which needs the triangle bridge
from the Reserving lane, and attention across claims of one policy.

Tuning the networks: `EarlyStopping` (on `Cann` and `AttentionCann`)
holds out a share of the training rows, scores the family's mean deviance
on them after each epoch, and keeps the best epoch once `patience` epochs
pass without improvement; `best_epoch` and `validation_history` record it.
Width, depth, learning rate and the like go through
`prospicio_models::resample::random_search` (with `log_uniform` and
`uniform_int` draws), which covers several hyperparameters better than a
grid of the same size. Fits vary by seed, so an average over a few seeds
often beats finer tuning.

Seeding: Burn's generator is global and its parameters initialize lazily,
so `prospicio-nn` seeds, builds and initializes every parameter under one lock.
Without it, fits running in parallel (tests, cross-validation) interleave
their draws and stop being reproducible.

## Decisions

- **Burn for neural networks** (replaces PyTorch-only, `architecture.md`).
  Networks are plain Rust, so Python, R and the WASM target all get them,
  results are reproducible under our RNG streams, and actuarial networks
  (tabular MLPs with embeddings, CANN) are small enough for CPU training.
  The default backend is `ndarray` on the CPU; GPU (`wgpu`) is opt-in.
  Models trained in PyTorch come in for inference through Burn's ONNX
  import.
- **Gradient boosting stays an adapter.** LightGBM and XGBoost are mature
  and fast; a Rust reimplementation would not change what users can do.
  Done in Python (`prospicio.boosting`): `Booster(family, engine,
  power, n_rounds, learning_rate, params, n_boot, seed)` over LightGBM or
  XGBoost for the Poisson, gamma, Tweedie (log link) and Gaussian. The
  design's offset is the engine's starting score (`init_score`,
  `base_margin`) plus a constant that starts the trees at the weighted
  mean (the engines skip `boost_from_average` once an offset is given);
  weights are sample weights. `BoosterFit.predict` gives means;
  `predict_distribution` gives joint draws keyed `row`, process noise from
  the family with Pearson's dispersion (divisor `n`), and with `n_boot`
  bootstrap refits each simulation takes one refit's means. Draws go
  through `prospicio_models::simulate::from_means` (Python
  `models.simulate_from_means`), which any mean-only engine can use. A
  `Booster` works in `compare`, `cross_validate` and the searches.
  Tests (`python/tests/test_boosting.py`): the offset is the exposure for
  both engines, the LightGBM adapter equals a direct LightGBM call, the
  gamma dispersion is recovered, bootstrap refits widen the draws.
  Done in R (`boosting.R`): `booster_fit(formula, data, family, engine,
  power, n_rounds, learning_rate, params, n_boot, seed, offset, weights)`
  over the lightgbm and xgboost packages (in `Suggests`), with the same
  starting score, dispersion and bootstrap; the design is `model.matrix()`
  without its intercept; `predict()` and `predict_distribution()` methods,
  and `simulate_from_means()`. Seeding leaves R's generator state as it
  was. Tests (`R/prospicio/tests/test-boosting.R`) mirror the Python
  ones. Quantile objective (both languages): `family = "quantile"` with
  `alpha` (LightGBM `quantile`, XGBoost `reg:quantileerror` with
  `quantile_alpha`), started from the weighted `alpha` quantile of `y`,
  no offset (refused with a message); one quantile per fit, so it works
  in `compare` and cross-validation unchanged, scored by
  `prospicio_models::metrics::pinball` (Python `pinball_loss`/`pinball_score`,
  R `pinball_loss()`). `predict_quantiles` combines fits into quantile
  sets and sorts each row across levels (the rearrangement of
  Chernozhukov, Fernández-Val and Galichon, 2010), so they never cross.
  Distributional head (both languages): `dispersion_model = True` fits a
  second booster (gamma objective, log link) to the Pearson residuals
  `w (y − μ)² / V(μ)` of 5-fold cross-fitted means, so each row gets its
  own dispersion (`predict_dispersion`), as a double GLM does (Smyth,
  1989); `predict_distribution` passes it to
  `prospicio_models::simulate::from_means`, whose `dispersion` is now one value
  or one per row. Gamma, Tweedie and Gaussian only: the Poisson's
  dispersion is 1.
- **Samplers are delegated.** No home-grown NUTS (`architecture.md`).

## Open questions

- (Decided 2026-10-15: model artifacts are versioned JSON; see "Model
  artifacts".)
- Whether `Design` stores dense or sparse columns (high-cardinality factors
  favour sparse).
- Burn version pinning, given its API churn.
