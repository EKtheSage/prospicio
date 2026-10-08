# Bundle history

## 2026-10-08

* **Update**: [Reinstatements pro rata as to time](/findings/reinstatement-pro-rata-time.md): seasonal event times (`with_seasonal_times`) map the sorted uniform draws through the season's increasing quantile, so they stay exact; equal weights give the uniform times; one exhausting loss a year in the second half only costs a quarter of the amount-only premium. Not yet run by CI. From branch claude/stoic-hawking-qc3qby.
* **Update**: [Local checks](/environment/local-checks.md): installing R (with `libuv1-dev` for `fs`), roxygen2, uv and the wasm target in a container that lacks them.

## 2026-10-07

* **Correction**: [The simulated one-year view against Merz-Wuthrich, ODP and Mack's process](/findings/one-year-bootstrap-vs-merz-wuthrich.md): at a quarterly split the ODP's SD halves through its scale, not through quarters carrying a sixteenth of the variance each: the Pearson chi-square is unchanged, so the scale falls by exactly the ratio of degrees of freedom (36/171 on RAA and GenIns, 45/210 on ABC) and the process SD by about 0.46; Mack's zero-sigma first-year quarterly links no longer put zeros in its residual pool (they cut its mean square to 0.854), and the remeasured ratios are ODP 0.43 to 0.47 per origin (0.44 to 0.47 total), Mack 0.48 to 0.72 (0.53 to 0.55); a Gamma or lognormal draw out of floating point range, reachable when Mack's chained quarterly draws come out near zero, is now its limit, zero, not a panic. From the review of branch claude/one-year-grain.
* **Creation**: [Publishing to PyPI and crates.io](/environment/publishing.md), from setting up the release workflow (branch claude/sharp-wright-fyp7ta): the name check, the workspace dry run, trusted publishing on each registry, and the sdist and wheel tests.
* **Update**: the crates are renamed `act-*` to `prospicio-*` and the project `risk-rs` to `prospicio`; concept files use the new crate paths and repository URL. Earlier entries keep the old names; saved-file format tags keep `risk_rs.*`.
* **Update**: the Python and R packages are renamed `actuarialrs` to `prospicio`; concept files now use the new paths (`python/prospicio`, `R/prospicio`). Earlier entries keep the old name.
* **Creation**: [WebAssembly builds](/environment/wasm-builds.md), from settling the WASM scope: the core crates build for wasm32, their tests run under WASI with Node, and Rayon falls back to the calling thread.

## 2026-10-06

* **Update**: [The simulated one-year view against Merz-Wuthrich, ODP and Mack's process](/findings/one-year-bootstrap-vs-merz-wuthrich.md): any development grain and lagging origins; split into quarters, RAA, GenIns and ABC keep the annual opening reserve and get about half the annual one-year SD under both models (their quarters are independent, the split's move together); an exact pattern split stays at zero; a lagging RAA origin's SD grows 1.98 (ODP) and 1.38 (Mack) times. From branch claude/one-year-grain.
* **Correction**: [The simulated one-year view against Merz-Wuthrich, ODP and Mack's process](/findings/one-year-bootstrap-vs-merz-wuthrich.md): the reconciliation under Mack's process is of the standard deviation; EVW's uncentred residual pool (mean 0.14 on RAA) biases the mean CDR by -0.214, -0.038 and +0.176 SD on RAA, GenIns and ABC (now pinned in the validation test), and the new `MackBootstrap::centre_residuals` brings it to within Monte Carlo error of zero while the SD still reconciles; the `Residuals` process carries the pool's mean and variance. From the review of branch claude/one-year-mack.
* **Update**: [The simulated one-year view against Merz-Wuthrich, ODP and Mack's process](/findings/one-year-bootstrap-vs-merz-wuthrich.md), retitled: `MackBootstrap::one_year` (Mack's process, EVW 2019 Appendix 1) reconciles with R's `CDR(1)S.E.` on RAA, GenIns and ABC within five Monte Carlo standard errors and with EVW's Table 4; a reconciliation table of ODP, Mack's process and Merz-Wuthrich; RAA's young origins 0.4% to 1.2% above R at 200,000 simulations. From branch claude/one-year-mack.
* **Verification**: [Reinstatements pro rata as to time](/findings/reinstatement-pro-rata-time.md) verified by CI's Rust, Python and R jobs on #146.
* **Creation**: [Reinstatements pro rata as to time](/findings/reinstatement-pro-rata-time.md), from dating events and pro rata as to time reinstatement premiums.
* **Update**: [Property exposure curves beyond MBBEFD](/references/property-exposure-curves.md): a band spread between bounds cedes on its SI-weighted average, not its mean risk (3/8 against 1/3 in the test), from band bounds in the risk profile.
* **Correction**: [The ODP one-year bootstrap against Merz-Wuthrich](/findings/one-year-bootstrap-vs-merz-wuthrich.md): the next cell is now projected from the pseudo latest value, as the lifetime ODP bootstrap does, so the ratios changed (RAA total 0.46 to 0.61); the one-cell origin now matches BootChainLadder without an allowance; the Mack-bootstrap re-reserving check is a CI unit test (within 0.6% on GenIns), and the ratio test is labelled a seed-pinned regression. From the review of branch claude/one-year-bootstrap.
* **Update**: [Local checks](/environment/local-checks.md): how to check a hand-edited `extendr-wrappers.R` against `wrap__make_actuarialrs_wrappers`, and roxygen's link warnings for a new topic; and [The ODP one-year bootstrap against Merz-Wuthrich](/findings/one-year-bootstrap-vs-merz-wuthrich.md): the R and Python tests hold the same ratios. From R's `odp_one_year()` (branch claude/one-year-bootstrap).
* **Creation**: [The ODP one-year bootstrap against Merz-Wuthrich](/findings/one-year-bootstrap-vs-merz-wuthrich.md), from building `OdpBootstrap::one_year` (reserving v0.2, decision 8, branch claude/one-year-bootstrap): the measured ratios, why the ODP and Mack differ, and what England, Verrall and Wuthrich (2019) and Boumezoued et al. (2011) publish.
* **Creation**: [Error and uncertainty terminology](/references/error-terminology.md), a glossary of the error terms used in reserving, statistics and machine learning, and how model bias and systemic risk differ from parameter error.
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
