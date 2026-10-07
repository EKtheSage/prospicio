---
type: Reference
title: Error and uncertainty terminology
description: What standard error, standard deviation, process, parameter, model, systemic and systematic error, epistemic and aleatoric uncertainty, bias, variance and irreducible error mean in actuarial, statistical and machine-learning writing, where they overlap, and which act-reserving outputs measure which.
resource: https://doi.org/10.2143/AST.23.2.2005092
tags: [reserving, terminology, uncertainty, msep, risk-margin]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-06T23:00:00Z }
sources:
  - id: mack
    resource: https://doi.org/10.2143/AST.23.2.2005092
    title: Mack (1993), Distribution-free calculation of the standard error of chain ladder reserve estimates, ASTIN Bulletin 23(2)
  - id: ev
    resource: https://doi.org/10.1017/S1357321700003809
    title: England and Verrall (2002), Stochastic claims reserving in general insurance, British Actuarial Journal 8(3)
  - id: kd
    resource: https://doi.org/10.1016/j.strusafe.2008.06.020
    title: Der Kiureghian and Ditlevsen (2009), Aleatory or epistemic? Does it matter?, Structural Safety 31(2)
  - id: risk-margin
    resource: citation:Marshall, Collings, Hodson and O'Dowd (2008), A risk margin framework for outstanding claims liabilities, Actuaries Institute General Insurance Seminar
    title: Marshall et al. (2008), A risk margin framework for outstanding claims liabilities
  - id: esl
    resource: citation:Hastie, Tibshirani and Friedman (2009), The Elements of Statistical Learning, 2nd ed., section 7.3
    title: Hastie, Tibshirani and Friedman, The Elements of Statistical Learning, the bias-variance decomposition
  - id: code
    resource: ../crates/act-reserving/src/mack.rs
    title: act_reserving::Mack, MackFit
---

# The identity behind most of the terms

For a prediction `Ŷ` of an outcome `Y`, the mean squared error of
prediction (MSEP) splits in three:[^mack][^esl]

```
MSEP = E[(Y − Ŷ)²] = Var(Y) + Var(Ŷ) + Bias(Ŷ)²
                     process  parameter  systematic
```

Mack's formula and the ODP bootstrap estimate the first two terms and
assume the third is zero.[^mack][^ev]

# The terms

| Term | Meaning | Also called |
|---|---|---|
| Process error | Randomness of the outcome given the true model and parameters; more data does not reduce it | aleatoric, irreducible error, Mack's process risk |
| Parameter error | Uncertainty from estimating parameters (factors, ELR, ω, θ) on finite data; shrinks as data grows | estimation error, Mack's parameter risk, the "variance" of bias–variance; part of epistemic |
| Model error | The model's form is wrong (e.g. chain ladder when case reserving practice changed); does not shrink with more of the same data | specification error, model risk; shows up as bias; part of epistemic |
| Aleatoric | Machine-learning and engineering name for inherent randomness[^kd] | process error |
| Epistemic | Uncertainty from lack of knowledge, reducible in principle[^kd] | parameter + model error |
| Irreducible error | The term of the bias–variance decomposition no model removes[^esl] | process variance |
| Bias | The estimate's mean is shifted from the truth in a consistent direction | systematic error |
| Systematic error | Statistics' name for error that repeats in one direction, against random error | bias |
| Systemic risk | Actuarial risk-margin term: external drivers (inflation shocks, legal and social change, claims-handling change) and model specification error, which hit every claim, origin and line together and do not diversify[^risk-margin] | finance's "systematic risk" (market, beta); **not** statistics' systematic error |
| Independent risk | Risk-margin term for the random, diversifiable part[^risk-margin] | process + parameter error |
| Variance | Spread of a random quantity; says nothing until you name the quantity (the outcome, or an estimator) | — |
| Standard deviation | Square root of a variance; a property of a distribution, such as the reserve's | — |
| Standard error | Statistics: the standard deviation of an estimator (parameter error only). Reserving (Mack's S.E.): the square root of MSEP, process and parameter together | the second sense is prediction error, root MSEP |

# Common confusions

* **Standard error has two meanings.** `DevelopmentFit::std_err` is each
  factor's standard error (parameter only); `MackFit::standard_error` is
  root MSEP (process and parameter).[^code]
* **Systemic is not systematic.** In statistics, systematic means bias. In
  actuarial risk margins, systemic means correlated, non-diversifying
  risk; finance calls that systematic and keeps "systemic" for the
  collapse of a financial system.
* **"Variance" differs between decompositions.** In bias–variance it is
  the estimator's variance, i.e. parameter error; for a reserve
  distribution it usually means the total.
* **The horizon is a separate choice.** Mack and the full bootstrap give
  the ultimate (run-off) view; Merz–Wüthrich and the simulated claims
  development result give the one-year view. Each has its own process and
  parameter parts.

# Are model bias and systemic error like parameter error?

All three are epistemic: they come from what we do not know, not from
inherent randomness. Beyond that they differ in the ways that matter:

| | Parameter error | Model bias | Systemic risk |
|---|---|---|---|
| Shrinks with more data of the same kind | yes | no: a wrong model converges to the wrong answer | no: future shocks are not in past data |
| Mean | zero (an unbiased estimator) | non-zero by definition | can be either; often skewed (inflation) |
| Visible inside the model's own standard error | yes | no | no |
| Diversifies across lines | partly | no | no, by definition |
| Estimated by | Mack, bootstrap, Fisher information | backtests, comparing models, expert judgement | scenario analysis, benchmarks, judgement (risk-margin frameworks) |

Two links between them:

* Parameter error is already correlated across origins in one triangle,
  because every origin shares the same estimated factors (Mack's total
  parameter risk has these covariance terms). That is the one way it
  resembles systemic risk.
* A systemic driver can be turned into parameter error by modelling it:
  give future inflation a parameter and a prior, and its uncertainty
  becomes posterior parameter uncertainty in a Bayesian model. Model
  uncertainty can likewise be partly brought inside by averaging or
  stacking several models. What stays outside any model stays model or
  systemic risk.

# What act-reserving measures

| Output | Contains |
|---|---|
| `MackFit` process, parameter and standard error | process / parameter / root MSEP, ultimate view |
| `OdpBootstrap` reserves | process and parameter, as joint draws |
| `ClaimsDevelopmentResult` | root MSEP of the one-year view (Merz–Wüthrich) |
| Simulated one-year view (decision 8 of `docs/design/reserving-v02.md`) | process and parameter, one-year view, any method |
| Diagonal backtest, model stacking | evidence of model error |
| Systemic risk | not measured |

[^mack]: Mack (1993)
[^ev]: England and Verrall (2002)
[^kd]: Der Kiureghian and Ditlevsen (2009)
[^risk-margin]: Marshall et al. (2008)
[^esl]: Hastie, Tibshirani and Friedman, section 7.3
[^code]: act_reserving::Mack, MackFit
