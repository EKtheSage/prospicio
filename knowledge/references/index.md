# Reference implementations

* [statsmodels](statsmodels.md) - Python statsmodels, the reference for prospicio-glm's GLMs, robust standard errors and Tweedie profiles; its conventions and quirks.
* [R loo](r-loo.md) - The R package loo 2.6.0, the reference for ELPD by PSIS-LOO and WAIC; its stacking_weights stops early.
* [BayesBlend](bayesblend.md) - Ledger Investing's BayesBlend 0.0.8 (MIT), whose Stan models prospicio_bayes::stacking follows, including partial pooling and adaptive priors.
* [R Pareto](r-pareto.md) - The R package Pareto 2.4.5 (GPL), used only for reference values; where its tower matching departs from Riegel (2018).
* [nuts-rs](nuts-rs.md) - nutpie's Rust core, nuts-rs 0.19 (MIT), the sampler behind prospicio_bayes::nuts, BayesGlm and stacking; how prospicio-bayes drives it.
* [R ChainLadder CDR](r-chainladder-cdr.md) - R ChainLadder 0.2.21's CDR.MackChainLadder, the reference for Merz-Wuthrich's claims development result; its conventions, and where it and the 2008 paper differ.
* [Tails in R ChainLadder and chainladder-python](chainladder-tails.md) - R's tail = TRUE rule and tail_SE, chainladder-python's TailConstant, TailCurve and TailBondy; the quirks prospicio_reserving::Tail reproduces or avoids.
* [LightGBM and XGBoost](lightgbm-xgboost.md) - The engines behind prospicio.boosting; how offsets, starting scores and precision behave, and which wheels to install.
* [MBBEFD exposure curves](mbbefd.md) - The MBBEFD class behind prospicio_pricing::exposure; its four closed-form cases, the Swiss Re c curves, and numerical points found while building it.
* [chainladder-python expected-loss estimators](chainladder-python-expected-loss.md) - chainladder-python 0.10.1's ExpectedLoss, BornhuetterFerguson, Benktander and CapeCod, the reference for prospicio_reserving's expected-loss family; how its arithmetic and Cape Cod trend work.
* [Property exposure curves beyond MBBEFD](property-exposure-curves.md) - First-loss scales, PSOLD, Lloyd's and reinsurer curves, and how prospicio_pricing::exposure covers them with MBBEFD, tabulated and severity curves.
* [R ChainLadder ClarkLDF and ClarkCapeCod](r-chainladder-clark.md) - R ChainLadder 0.2.21's Clark growth-curve methods, the reference for prospicio_reserving's ClarkLdf and ClarkCapeCod; their age, scale and reporting definitions, a loose optimizer stop, a silent cap of 10 on the Cape Cod ELR, and a wrong Weibull second derivative.
* [chainladder-python ClarkLDF](chainladder-python-clark.md) - chainladder-python 0.10.1's ClarkLDF, the secondary reference for prospicio_reserving's Clark methods; what it shares with R ChainLadder and where its outputs are not comparable.
* [Error and uncertainty terminology](error-terminology.md) - What standard error, process, parameter, model, systemic and systematic error, epistemic and aleatoric uncertainty, bias and variance mean, where they overlap, and which prospicio-reserving outputs measure which.
* [Mildenhall's aggregate package, Pricing Insurance Risk and CAS Monograph 15](aggregate-mildenhall.md) - The Python aggregate package (1.0.1) as the reference for distortions, calibration, natural allocation and contract terms; where the Pricing Insurance Risk and Monograph 15 examples live, and the API and tolerance quirks met reproducing them.
