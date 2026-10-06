# Reference implementations

* [statsmodels](statsmodels.md) - Python statsmodels, the reference for act-glm's GLMs, robust standard errors and Tweedie profiles; its conventions and quirks.
* [R loo](r-loo.md) - The R package loo 2.6.0, the reference for ELPD by PSIS-LOO and WAIC; its stacking_weights stops early.
* [BayesBlend](bayesblend.md) - Ledger Investing's BayesBlend 0.0.8 (MIT), whose Stan models act_bayes::stacking follows, including partial pooling and adaptive priors.
* [R Pareto](r-pareto.md) - The R package Pareto 2.4.5 (GPL), used only for reference values; where its tower matching departs from Riegel (2018).
* [nuts-rs](nuts-rs.md) - nutpie's Rust core, nuts-rs 0.19 (MIT), the sampler behind act_bayes::nuts, BayesGlm and stacking; how act-bayes drives it.
* [chainladder-python expected-loss estimators](chainladder-python-expected-loss.md) - chainladder-python 0.10.1's ExpectedLoss, BornhuetterFerguson, Benktander and CapeCod, the reference for act_reserving's expected-loss family; how its arithmetic and Cape Cod trend work.
* [R ChainLadder ClarkLDF and ClarkCapeCod](r-chainladder-clark.md) - R ChainLadder 0.2.21's Clark growth-curve methods, the reference for act_reserving's ClarkLdf and ClarkCapeCod; their age, scale and reporting definitions, a loose optimizer stop, a silent cap of 10 on the Cape Cod ELR, and a wrong Weibull second derivative.
* [chainladder-python ClarkLDF](chainladder-python-clark.md) - chainladder-python 0.10.1's ClarkLDF, the secondary reference for act_reserving's Clark methods; what it shares with R ChainLadder and where its outputs are not comparable.
