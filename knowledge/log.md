# Bundle history

## 2026-10-06

* **Update**: [Property exposure curves beyond MBBEFD](/references/property-exposure-curves.md): the exposure-rated per-risk XL net of a surplus, from the risk-profile simulator.
* **Creation**: [Property exposure curves beyond MBBEFD](/references/property-exposure-curves.md), from surveying the curves for tabulated exposure curves.
* **Update**: [LightGBM and XGBoost](/references/lightgbm-xgboost.md): why the dispersion model cross-fits its residuals, and the positive-label floor.
* **Update**: [LightGBM and XGBoost](/references/lightgbm-xgboost.md): the quantile objectives' parameter names and the crossing fix, from the quantile boosters.
* **Creation**: [R callbacks must stay on R's main thread](/findings/r-callbacks-main-thread.md), from building `Custom` and its R binding.
* **Creation**: [Local Windows R](/environment/local-windows-r.md), from setting up local R binding checks for reserving v0.2.
* **Verification**: [LightGBM and XGBoost](/references/lightgbm-xgboost.md) verified again by CI's R job on #128, which now fails unless both engines install, so `test-boosting.R` ran both.
* **Update**: [LightGBM and XGBoost](/references/lightgbm-xgboost.md): the R packages' prediction calls and how `init_score` and `base_margin` come back, from building R's `booster_fit()`; `verified` is dropped until CI runs the R tests.
* **Verification**: [MBBEFD exposure curves](/references/mbbefd.md) verified by CI's Rust job on #124 (the mpmath parity test).
* **Update**: [Cloud network](/environment/cloud-network.md): CRAN is now allowed; Posit's binary redirect host is still refused, so R packages build from source.
* **Verification**: [LightGBM and XGBoost](/references/lightgbm-xgboost.md) verified by CI's Python job on #123.
* **Creation**: [MBBEFD exposure curves](/references/mbbefd.md), from building `act_pricing::exposure`.
* **Update**: [Cloud network](/environment/cloud-network.md): CRAN is refused in the container.
* **Creation**: [LightGBM and XGBoost](/references/lightgbm-xgboost.md), from building the boosting adapters: offsets as starting scores, the mean-starting constant, XGBoost's single precision, the CPU-only wheel.
* **Update**: AGENTS.md now asks every session to update this bundle when it finishes a work item, before it reports; no concepts changed in this entry.

## 2026-10-05

* **Initialization**: Created the bundle with [datasets](/datasets/index.md), [references](/references/index.md), [findings](/findings/index.md) and [environment](/environment/index.md), from knowledge collected while building the Probability, Aggregate and Models lanes (PRs #104 to #116).
* **Expected-loss reference**: Added [chainladder-python expected-loss estimators](/references/chainladder-python-expected-loss.md), from the Reserving lane's expected-loss methods (branch claude/v02-expected-loss).
* **Expected-loss reference update**: [chainladder-python expected-loss estimators](/references/chainladder-python-expected-loss.md) now records the Benktander closed form, its divergence below cdf 1/2, and how incremental exposure and off-diagonal origins differ from act_reserving (branch claude/v02-expected-loss).
