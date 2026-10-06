# Reference implementations

* [statsmodels](statsmodels.md) - Python statsmodels, the reference for act-glm's GLMs, robust standard errors and Tweedie profiles; its conventions and quirks.
* [R loo](r-loo.md) - The R package loo 2.6.0, the reference for ELPD by PSIS-LOO and WAIC; its stacking_weights stops early.
* [BayesBlend](bayesblend.md) - Ledger Investing's BayesBlend 0.0.8 (MIT), whose Stan models act_bayes::stacking follows, including partial pooling and adaptive priors.
* [R Pareto](r-pareto.md) - The R package Pareto 2.4.5 (GPL), used only for reference values; where its tower matching departs from Riegel (2018).
* [nuts-rs](nuts-rs.md) - nutpie's Rust core, nuts-rs 0.19 (MIT), the sampler behind act_bayes::nuts, BayesGlm and stacking; how act-bayes drives it.
* [LightGBM and XGBoost](lightgbm-xgboost.md) - The engines behind actuarialrs.boosting; how offsets, starting scores and precision behave, and which wheels to install.
* [MBBEFD exposure curves](mbbefd.md) - The MBBEFD class behind act_pricing::exposure; its four closed-form cases, the Swiss Re c curves, and numerical points found while building it.
* [chainladder-python expected-loss estimators](chainladder-python-expected-loss.md) - chainladder-python 0.10.1's ExpectedLoss, BornhuetterFerguson, Benktander and CapeCod, the reference for act_reserving's expected-loss family; how its arithmetic and Cape Cod trend work.
* [Property exposure curves beyond MBBEFD](property-exposure-curves.md) - First-loss scales, PSOLD, Lloyd's and reinsurer curves, and how act_pricing::exposure covers them with MBBEFD, tabulated and severity curves.
