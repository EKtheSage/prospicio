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

- `act-models`:
  - `Family` (Gaussian, Poisson, gamma, inverse Gaussian, binomial,
    negative binomial, Tweedie) with variance function, unit deviance,
    log-likelihood as statsmodels defines it, and `draw` (the response's
    process noise, through the `act-prob` distributions);
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
- `act-glm`: `Glm` (family, link, dispersion fixed, Pearson or
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
- GAM (`act_glm::gam`): `Gam` = a `Glm` plus `PSpline` smooths, each
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
- `act-bayes`: MCMC diagnostics for any sampler's draws, the estimators
  of Vehtari et al. (2021) as R's `posterior` implements them:
  rank-normalized split `rhat` (the larger of the bulk and folded
  versions), `ess_bulk`, `ess_tail`, `ess_quantile`, `ess_mean` and
  `mcse_mean`, with Geyer's initial monotone sequence on FFT
  autocovariances. Parity: `validation/reference/mcmc_posterior.csv`
  (`validation/scripts/r_mcmc_diagnostics.R`), five sets of four chains
  (independent, AR(1), a shifted chain, a wider chain, ties), 25 values at
  1e-10.
- `act_bayes::elpd`: the expected log pointwise predictive density from
  a fit's pointwise log-likelihood draws: `lppd`, `waic` and `loo`
  (Pareto-smoothed importance sampling: the largest importance ratios of
  each observation replaced by the expected order statistics of a
  generalized Pareto fitted by Zhang and Stephens' estimator with Vehtari
  et al.'s weakly informative prior, truncated at the largest raw ratio),
  with `k̂` per observation as the reliability diagnostic. Written from the
  papers; parity with R's `loo` 2.6.0 (`validation/scripts/r_loo.R`):
  every estimate, pointwise ELPD to `1e-9` and `k̂` to `1e-6`.
- **Sampling: nutpie** (decided 2026-10-04). Native models (Bayesian GLM,
  Bayesian chain ladder, credibility) will sample with nutpie's Rust core
  `nuts-rs`, so R gets NUTS too; Python users can hand nutpie traces of
  PyMC or Stan models to the same diagnostics and ELPD.
- Decision: families and links are closed enums, like `Distortion`, so a
  fitted model serializes as data.

## Crates

Crates split by dependency weight, never by concept (`architecture.md`:
"split a crate only when compile time, dependency weight, or independent
release cadence forces it"). One light crate holds the interface and the
life cycle; each heavy engine sits in its own crate behind a feature flag.

| Crate | Holds | Heavy dependencies |
|---|---|---|
| `act-models` | The `Model` trait, model specs, `Design`, `Family` and links, resampling, metrics, tuning, comparison and stacking, model artifacts | None |
| `act-glm` | GLM by IRLS; GAM as a penalized GLM on the same solver (P-splines, tensor smooths, GCV/REML) | `faer` |
| `act-nn` | Neural networks on Burn: CANN (a GLM offset plus a network correction) and the attention CANN (a transformer over feature tokens) | Burn, opt-in feature |
| `act-bayes` | Bayesian model specs and diagnostics (R-hat, ESS, divergences, ELPD by PSIS-LOO and WAIC); sampling through nutpie | Samplers, opt-in feature |
| *(no crate)* | Gradient boosting: Python and R adapters over LightGBM and XGBoost that implement the interface and return shared objects | None in Rust |

- **Families live in `act-models`, distributions in `act-prob`.** A
  `Family` (variance function, unit deviance, log-likelihood, canonical
  link) is shared by the GLM, the network's loss, the boosters' objective
  and the Bayesian likelihood. It names the `act-prob` distribution it
  predicts: `Poisson`, `NegativeBinomial`, `Binomial`, `Gamma`, `Tweedie`,
  and later the inverse Gaussian. `act-glm` does not own families, so
  `act-nn` never depends on `act-glm`.
- **GAM lives inside `act-glm`.** A GAM is a penalized GLM: same IRLS loop,
  plus a penalty and smoothing-parameter selection. A separate crate would
  duplicate the solver.
- **Chain Ladder never compiles Burn or a sampler.** `act-nn` and
  `act-bayes` are opt-in features of the bindings, and nothing in the
  default build depends on them.

## Life cycle

The stages are modules of `act-models`, with matching submodules in Python
(`actuarialrs.models.*`) and R. This takes tidymodels' life cycle, not its
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

## Fitting a triangle with several models

Models never see a `Triangle`. The Reserving lane owns a bridge that turns
a triangle into a design, and the backtest that scores models on it.

1. **Triangle to design** (`act-reserving`). One row per origin ×
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

1. `act-models`: `Design`, `Family` (Poisson, negative binomial, binomial,
   gamma, Tweedie), links, `Model`, metrics.
2. `act-glm`: IRLS with offsets, weights and exposure; quasi-likelihood
   dispersion; elastic net. Parity with statsmodels and glum on freMTPL2.
3. The triangle bridge and the ODP GLM in the Reserving lane, checked
   against Chain Ladder.
4. Resampling, tuning and comparison.
5. GAM in `act-glm`.
6. `act-nn` on Burn: CANN first, then the attention CANN.
7. `act-bayes`, and the boosting adapters.

## Elastic net

`act_glm::net::ElasticNet` minimizes glmnet's objective,
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
`act_glm::tweedie::tweedie_profile` chooses `p` by profile likelihood, as
R's `tweedie.profile`: for each `p` on a grid it fits the GLM, maximizes
the exact Tweedie log-likelihood (Dunn and Smyth's series density from
`act-prob`, row `i` at `φ / wᵢ`) over `φ`, and refines the best `p` by
golden-section search between its grid neighbours.

`act_glm::tweedie::TweedieGlm` estimates `p` as part of the fit. Since the
coefficients at a fixed `p` do not depend on `φ`, the joint maximum
likelihood over coefficients, `p` and `φ` (what H2O's GLM does with a free
variance power) is the maximum of this one-dimensional profile; Brent's
method (`act_math::optimize::brent`) finds it in about a dozen GLM fits,
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
H2O uses both. Against 40-digit mpmath, `act-prob`'s series is accurate to
`5e-11` or better, in under 0.4 ms per point, for `p` from 1.01 to 1.99,
`φ` from 0.001 to 50 and `y` from 0.01 to 10 times the mean. It degrades
only at `φ = 1e-6` with `p` near 2 (`5e-8`, about 11 ms): a coefficient of
variation near 0.001, which claims and pure premiums never have. The
inversion's oscillatory integrals and series acceleration are not worth
that. If such regimes ever matter, compute the series' large log-gamma
differences more precisely instead.

## Attention

`act_nn::AttentionCann` is a CANN whose correction is a transformer over
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
`act_models::resample::random_search` (with `log_uniform` and
`uniform_int` draws), which covers several hyperparameters better than a
grid of the same size. Fits vary by seed, so an average over a few seeds
often beats finer tuning.

Seeding: Burn's generator is global and its parameters initialize lazily,
so `act-nn` seeds, builds and initializes every parameter under one lock.
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
- **Samplers are delegated.** No home-grown NUTS (`architecture.md`).

## Open questions

- Serialization format for model artifacts (a versioned JSON or CBOR
  schema, or Arrow for large coefficient sets).
- Whether `Design` stores dense or sparse columns (high-cardinality factors
  favour sparse).
- Burn version pinning, given its API churn.
