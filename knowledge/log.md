# Bundle history

## 2026-10-06

* **Correction**: [The ODP one-year bootstrap against Merz-Wuthrich](/findings/one-year-bootstrap-vs-merz-wuthrich.md): the next cell is now projected from the pseudo latest value, as the lifetime ODP bootstrap does, so the ratios changed (RAA total 0.46 to 0.61); the one-cell origin now matches BootChainLadder without an allowance; the Mack-bootstrap re-reserving check is a CI unit test (within 0.6% on GenIns), and the ratio test is labelled a seed-pinned regression. From the review of branch claude/one-year-bootstrap.
* **Update**: [Local checks](/environment/local-checks.md): how to check a hand-edited `extendr-wrappers.R` against `wrap__make_actuarialrs_wrappers`, and roxygen's link warnings for a new topic; and [The ODP one-year bootstrap against Merz-Wuthrich](/findings/one-year-bootstrap-vs-merz-wuthrich.md): the R and Python tests hold the same ratios. From R's `odp_one_year()` (branch claude/one-year-bootstrap).
* **Creation**: [The ODP one-year bootstrap against Merz-Wuthrich](/findings/one-year-bootstrap-vs-merz-wuthrich.md), from building `OdpBootstrap::one_year` (reserving v0.2, decision 8, branch claude/one-year-bootstrap): the measured ratios, why the ODP and Mack differ, and what England, Verrall and Wuthrich (2019) and Boumezoued et al. (2011) publish.
* **Update**: [Local checks](/environment/local-checks.md): the roxygen churn is a link-target rewrite, so a real change to those Rd files is kept and only the link restored.
* **Update**: [R ChainLadder CDR](/references/r-chainladder-cdr.md): what counts as no tail for the one-year view now that the chain ladder fits a `TailFit` (a factor of 1 that replaces no estimated factor), from merging `main` into claude/v02-one-year.
* **Verification**: [Property exposure curves beyond MBBEFD](/references/property-exposure-curves.md) verified by CI's Rust job on #140 (tabulated-curve and risk-profile tests).
* **Verification**: [LightGBM and XGBoost](/references/lightgbm-xgboost.md) verified again by CI's R and Python jobs on #139 (quantile objectives and the dispersion model, both engines).
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
* **Addition**: [R ChainLadder CDR](/references/r-chainladder-cdr.md), the reference for the one-year view (Reserving lane, v0.2 decision 4).
* **Update**: [R ChainLadder CDR](/references/r-chainladder-cdr.md) records how R and act-reserving treat an interior hole, and that act-reserving deviates on purpose.
* **Tails**: Added [Tails in R ChainLadder and chainladder-python](/references/chainladder-tails.md) and [Bondy least squares](/findings/bondy-least-squares-stop.md), from building act_reserving::Tail (reserving v0.2, decision 3).
* **Tails review**: Updated [Tails in R ChainLadder and chainladder-python](/references/chainladder-tails.md): a tail below 1 now follows chainladder-python, and the note records Python's positional age indexing and its ignored attachment at the youngest age.
* **Expected-loss reference**: Added [chainladder-python expected-loss estimators](/references/chainladder-python-expected-loss.md), from the Reserving lane's expected-loss methods (branch claude/v02-expected-loss).
* **Clark references**: Added [R ChainLadder ClarkLDF and ClarkCapeCod](/references/r-chainladder-clark.md) and [chainladder-python ClarkLDF](/references/chainladder-python-clark.md), from the Reserving lane's Clark methods (branch claude/v02-clark).
* **Clark ELR cap**: Recorded in [R ChainLadder ClarkLDF and ClarkCapeCod](/references/r-chainladder-clark.md) that R bounds the Cape Cod ELR at 10 without a warning, where act_reserving is unbounded (branch claude/v02-clark).
* **Expected-loss reference update**: [chainladder-python expected-loss estimators](/references/chainladder-python-expected-loss.md) now records the Benktander closed form, its divergence below cdf 1/2, and how incremental exposure and off-diagonal origins differ from act_reserving (branch claude/v02-expected-loss).
