---
type: Reference
title: England, Verrall and Wüthrich (2012), the Bayesian ODP and the Bornhuetter–Ferguson
description: EVW's Model 3.1 (gamma priors on the row parameters, vague priors on the pattern) is the model BayesBf builds; its strong-prior limit is Mack's raw-pattern BF, not the classical BF, and their printed totals on the Wüthrich–Merz triangle are reproducible to within their Monte Carlo error.
tags: [bayes, reserving, bornhuetter-ferguson, odp, reference, evw]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-12T23:30:00-07:00 }
sources:
  - id: paper
    resource: https://openaccess.city.ac.uk/id/eprint/3801/
    title: England, Verrall and Wüthrich (2012), Bayesian Overdispersed Poisson Model and the Bornhuetter-Ferguson Claims Reserving Method, Annals of Actuarial Science 6(2), 258-283 (City Research Online preprint, October 2010)
  - id: data
    resource: ../validation/data/evw2012.csv
    title: Their Table 1 (Wüthrich and Merz 2008, Tables 2.2-2.4), cumulated
  - id: test
    resource: ../validation/tests/reserving_bayes_bf.rs
    title: The parity tests, england_verrall_wuthrich_2012
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decision 9)
---

# What it is

* The paper's ODP model has means `mu_i gamma_j` (an unnormalised
  pattern), dispersion `phi` plugged in, and two prior families. Model
  2.1 puts flat priors on `log mu_i` and `log gamma_j`: its mode is the
  chain ladder. Model 3.1 puts `mu_i ~ Gamma(mean m_i, shape a_i)` and a
  vague gamma prior (`b -> 0`, flat on `log gamma_j`) on the pattern;
  `a_i` is the "degree of information", so the prior's coefficient of
  variation is `1 / sqrt(a_i)`. They sample with Gibbs (both full
  conditionals are gamma).[^paper]
* The strong-prior limit `a -> infinity` fixes `mu_i = m_i` and makes the
  `gamma_j` independent `Gamma(S_j / phi, M_j / phi)`, with `S_j` the
  observed increments of column `j` and `M_j` the prior ultimates of the
  origins observed there. The posterior mean reserve is then
  `m_i sum S_j / M_j` over the future cells: Mack's (2008)
  Bornhuetter–Ferguson with the raw pattern ("Mack1" in their sec. 3.4),
  not the classical BF `m_i (1 - 1 / cdf_i)`, which takes the chain
  ladder's pattern. The paper says so explicitly: priors on the rows
  reshape the pattern even under vague column priors, and Verrall's (2004)
  hybrid that keeps the chain ladder's pattern exactly is not
  statistically optimal.[^paper] The two coincide when `m_i` are the chain
  ladder's ultimates, since the chain ladder's column equations are
  `gamma_j sum mu_i = S_j`.
* Their `m_i` is the prior mean of the row parameter `mu_i`, not of the
  expected ultimate `mu_i sum_j gamma_j`: the pattern is unnormalised, and
  under a strong prior its sum is `sum_j S_j / M_j` over all columns, 0.885
  on their Table 1 triangle (computed from `validation/data/evw2012.csv`).
  A prior "ultimate" built as a loss ratio times premium therefore implies
  an ultimate about 11.5% lower there; `BayesBf`'s `apriori` inherits
  this.[^paper]
* The MAP of the gamma parametrisation differs from the posterior mean
  (their Remarks 2.3 and the univariate example of sec. 5.1): the mode
  is not invariant under reparametrisation, so they report MMSE
  predictors. On the log scale with a flat prior the mode of `gamma_j`
  at `a = infinity` is the raw pattern itself.

# The example

* Data: Wüthrich and Merz (2008), Tables 2.2-2.4, as their Table 1 prints
  it, 10 x 10 incremental paid with prior ultimates `m_i`, kept as
  `validation/data/evw2012.csv` (cumulated, origins 2001 to 2010, the
  prior repeated on each age).[^data] `phi = 14,714` in the paper; the
  chain ladder's Pearson scale is 14,714.085, so it is the same plug-in.
* Their totals (total reserve, MSEP^(1/2)): chain ladder 6,047,059; `a = 0`
  6,049,398 (430,160), Gibbs; `a = 100` 6,145,526 (422,526), Gibbs;
  `a = infinity` 6,644,047 (395,012), closed form. The classical BF with
  the same `m_i` gives 7,356,578, far from any of them.[^paper][^test]
* Reproduced (`BayesBf` with `development_sd` 100 for their flat pattern
  prior; Stan of the same model at 4 x 25,000 draws): `a = 100` ours
  6,146,087 (415,078), Stan 6,145,195 (422,771); `a = 0` ours 6,050,384
  (426,478), Stan 6,051,957 (429,728); the closed form at `a = infinity`
  computed from the data matches theirs to 1.3 on the mean and 1.5 on the
  MSEP (their rounded `phi`), and a fit at cv 0.001 gives 6,639,993
  (399,035).[^test] Their Gibbs runs agree with Stan to 0.05% on the
  means and 0.1% on the MSEPs, so their printed numbers are a usable
  second reference.
* With the default `development_sd` 10 the vague normal prior on
  `log gamma_j` lifts the one-cell last column (`S_10 / phi` about 1.07)
  by about 6.5% at `a = infinity` and the total by 0.27%; at sd 100 the
  lift is 3e-5. Stan at `a = 100` gives 6,162,974 at sd 10 against
  6,145,195 at sd 100.
