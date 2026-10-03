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
| `act-nn` | Neural networks on Burn, starting with CANN (a GLM offset plus a network correction) | Burn, opt-in feature |
| `act-bayes` | Bayesian model specs and diagnostics (R-hat, ESS, divergences, LOO); sampling through nutpie or BridgeStan | Samplers, opt-in feature |
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
| Evaluate | `metrics` | yardstick | Deviance, log-likelihood, Gini, Lorenz, lift, double lift, CRPS, PIT, interval coverage, actual vs expected |
| Tune | `tune` | tune, dials | Grid or random search over a parameter space, each candidate scored on the resamples. Parallel over the outer loop and reproducible from a seed |
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
6. `act-nn` on Burn: CANN first.
7. `act-bayes`, and the boosting adapters.

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
