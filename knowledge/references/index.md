# Reference implementations

* [statsmodels](statsmodels.md) - Python statsmodels, the reference for act-glm's GLMs, robust standard errors and Tweedie profiles; its conventions and quirks.
* [R loo](r-loo.md) - The R package loo 2.6.0, the reference for ELPD by PSIS-LOO and WAIC; its stacking_weights stops early.
* [BayesBlend](bayesblend.md) - Ledger Investing's BayesBlend 0.0.8 (MIT), whose Stan models act_bayes::stacking follows, including partial pooling and adaptive priors.
* [R Pareto](r-pareto.md) - The R package Pareto 2.4.5 (GPL), used only for reference values; where its tower matching departs from Riegel (2018).
* [nuts-rs](nuts-rs.md) - nutpie's Rust core, nuts-rs 0.19 (MIT), the sampler behind act_bayes::nuts, BayesGlm and stacking; how act-bayes drives it.
* [R ChainLadder CDR](r-chainladder-cdr.md) - R ChainLadder 0.2.21's CDR.MackChainLadder, the reference for Merz-Wuthrich's claims development result; its conventions, and where it and the 2008 paper differ.
