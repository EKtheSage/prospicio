# Guides

Explanations for actuaries, in prose, written around what the library does.
Each guide starts from a question an actuary asks (what is a reinsurance price,
how is a submission evaluated) and answers it with a running example, so a
reader who knows primary pricing can follow it without the reinsurance
vocabulary. The notebooks in [`examples/`](../../examples/README.md) are
worked analyses; the guides are the reading that goes before them.

The guides are also published on the [docs site](https://ekthesage.github.io/prospicio/),
in the Python site's User Guide beside the notebooks, which the site runs so
their results show there (the repository keeps the notebooks without outputs).
`cargo xtask python` builds that User Guide from this folder: every `.md` file
here but this README becomes a page, its first `#` heading the page's title,
and a link to a notebook in `examples/` points to the notebook's page.

Every guide's code is tested. `python/tests/test_guides.py` runs each guide's
```` ```python ```` blocks in order in one namespace, and when a block is
followed by a ```` ```text ```` block, requires the block to print exactly that
text, so a number in a guide cannot go stale when the library changes. A block
marked ```` ```python no-run ```` is shown but not run. To run the tests:

```sh
cd python
maturin develop
pytest tests/test_guides.py
```

Start here: [`choosing-a-method.md`](choosing-a-method.md). It maps each
situation (a rate indication, a per-risk or casualty layer, a catastrophe cover,
a quota share, a stop loss, a reserve, capital) to the method and the calls,
and sends you to the guide that goes deeper.

| Guide | What it answers |
|---|---|
| [`choosing-a-method.md`](choosing-a-method.md) | Which prospicio tool for which job: a decision table over the situations an actuary meets, the calls, the traps and where the library stops |

**Pricing**

| Guide | What it answers |
|---|---|
| [`reinsurance-price.md`](reinsurance-price.md) | What a reinsurance price is: what is priced and what comes out, the technical premium of an excess of loss layer and a quota share, how price follows a risk profile without a rating plan, a contract alone and in a book, hard and soft markets, what the reinsurer decides, and what the model leaves out |
| [`experience-and-exposure-rating.md`](experience-and-exposure-rating.md) | Rating a layer from the cedant's own losses (preparing them as-if for currency, development, trend and an index clause, then the burning cost, as-if history under past programs, a fitted Pareto), from its exposure (curves, increased limit factors, risk profiles), and how to blend the two |
| [`risk-loads-and-distortions.md`](risk-loads-and-distortions.md) | Which risk load or distortion to use (constant cost of capital, proportional hazard, Wang, TVaR and others), how to calibrate it to quotes, and how to price a contract in a portfolio |
| [`capital-modeling-monograph-15.md`](capital-modeling-monograph-15.md) | CAS Monograph 15 (Major and Mildenhall) reproduced: the InsCo example priced by five distortions and allocated to units, the 35 xs 65 reinsurance decision, and the correlated three-unit book, with the monograph's printed numbers beside prospicio's |
| [`capital-modeling-monograph-15-theory.md`](capital-modeling-monograph-15-theory.md) | CAS Monograph 15, chapters 3 and 4: VaR and TVaR by the monograph's algorithms, coTVaR, the pentagon, pricing layers, the five calibrated distortions and their event weights, the natural allocation by hand and by `Portfolio`, and the industry standard approach, on InsCo with the printed tables beside prospicio's |
| [`capital-modeling-monograph-15-applications.md`](capital-modeling-monograph-15-applications.md) | CAS Monograph 15, chapters 5 and 6: performance by unit, the 35 xs 65 reinsurance decision, new business and acquisition pricing, portfolio optimization by grid search, and choosing a distortion from the range of unit premiums (`premium_bounds`), with the printed tables beside prospicio's |
| [`pricing-insurance-risk-case-studies.md`](pricing-insurance-risk-case-studies.md) | The case studies of *Pricing Insurance Risk* (Mildenhall and Major): Discrete, Tame, Cat/Non-Cat gross and net of an aggregate cover, Hurricane/SCS gross and net of an occurrence cover, each priced by five distortions and allocated to units, with `aggregate` 0.30.1's numbers beside prospicio's |
| [`pir-price-allocation.md`](pir-price-allocation.md) | Part III of *Pricing Insurance Risk* (Mildenhall and Major): the classical allocations (haircut, equal risk, coTVaR, Merton-Perold, Shapley, covariance), equal priority, the natural allocation linear and lifted, margin and capital by unit, Bodoff's percentile layer of capital, pricing bounds, with the book's printed numbers beside prospicio's |
| [`pir-price-allocation-practice.md`](pir-price-allocation-practice.md) | *Pricing Insurance Risk* in practice: the Cat/Non-Cat case study allocated by chapter 13's nine classical recipes and by chapter 15's natural allocation and PLC (Tables 13.3, 15.36, 15.39), and chapter 19's test of a reinsurance cover, with the book's printed numbers beside prospicio's |
| [`pir-risk-measures.md`](pir-risk-measures.md) | *Pricing Insurance Risk* (Mildenhall and Major), Part I: risk represented two ways, quantiles and VaR, a hurricane PML table, VaR's failures of subadditivity and the worst VaR by rearrangement, TVaR and CTE, VaR, TVaR and EPD for the Tame and Cat/Non-Cat case studies, the properties of risk measures, and how a `Distortion` represents a spectral risk measure, with the book's printed numbers beside prospicio's |
| [`pir-portfolio-pricing.md`](pir-portfolio-pricing.md) | *Pricing Insurance Risk*, Part II: the portfolio constant cost of capital price, ten classical premium principles stand-alone and pooled, ruin theory and the natural scale of a market, Bernoulli layer statistics, the five distortions and the Blend calibrated and applied to each unit, gross and net, with the book's printed numbers beside prospicio's |
| [`ratemaking-werner-modlin.md`](ratemaking-werner-modlin.md) | Werner and Modlin's *Basic Ratemaking* (CAS study note), the steps of the overall rate indication reproduced: dates on the book's half-month grid, premium at the current rate level by the parallelogram method, premium and loss trend fits, loss development, expenses and the permissible loss ratio, and limited fluctuation and Bühlmann credibility, with the book's printed numbers beside prospicio's |
| [`ratemaking-werner-modlin-indications.md`](ratemaking-werner-modlin-indications.md) | Werner and Modlin's three worked indications end to end (Appendix A auto by the loss ratio method, B homeowners by the pure premium method, C medical malpractice with a net trend), chapter 11's increased limits and deductibles, what prospicio does not cover, and every place the book's rounding moves a printed digit |
| [`glm-to-comprehensive-pricing-chalk.md`](glm-to-comprehensive-pricing-chalk.md) | CAS Monograph 16 (Chalk, Deacon, Guillen and Martinelli), chapters 3 to 6: pseudo-R2 and folds from the policy number, null, baseline and boosting benchmark models, ridge, lasso and the penalty chosen by cross-validation, step functions picked by the lasso against a GAM, and credibility-weighted target encoding of a 300-level make, on a simulated portfolio with a known truth and with the monograph's printed worked examples beside prospicio's |
| [`glm-to-comprehensive-pricing-chalk-2.md`](glm-to-comprehensive-pricing-chalk-2.md) | CAS Monograph 16, chapters 7 to 10: highly correlated features under a GLM, ridge and the lasso, the chapter 8 offset example reproduced from its printed cell counts, offsets and residuals in two-stage models, and learning from a boosting model (permutation importance, partial dependence, an interaction screen), with what prospicio does not cover |
| [`glm-ratemaking-goldburd.md`](glm-ratemaking-goldburd.md) | CAS Monograph 5 (Goldburd, Khare, Tevet and Guller), part 1: the GLM as a multiplicative rating plan, base levels, offsets and weights, Poisson, negative binomial, gamma, inverse Gaussian and Tweedie, the F-test, AIC and BIC, with the monograph's printed arithmetic beside prospicio's and the list of what prospicio does not support |
| [`glm-ratemaking-goldburd-validation.md`](glm-ratemaking-goldburd-validation.md) | CAS Monograph 5, part 2: partial residuals, bins, polynomials, hinges and GAM splines, interactions, deviance and binned working residuals, holdout lift, double lift and loss ratio charts, the Gini index and ROC, stability and the elastic net |

**Distributions**

| Guide | What it answers |
|---|---|
| [`severity-and-frequency.md`](severity-and-frequency.md) | Choosing and fitting a severity and a claim count: lognormal, Pareto family, mixtures, censoring and truncation, and count families by dispersion |
| [`aggregate-distributions.md`](aggregate-distributions.md) | The distribution of a year's total loss: FFT, Panjer, simulation or moments, the grid, and when each is exact, fast or unsafe |
| [`prospicio-for-aggregate-users.md`](prospicio-for-aggregate-users.md) | The twenty-minute tour of Mildenhall's `aggregate` package run in prospicio: the same books, with both libraries' numbers side by side, a DecL-to-prospicio table and what prospicio does not have |
| [`distributions-for-actuaries-claim-size.md`](distributions-for-actuaries-claim-size.md) | CAS Monograph 2 (Bahnemann), chapters 2, 5 and 6 reproduced: limited expected values, fits to grouped, censored and truncated data, excess severity and layers, the leveraging effect of inflation, increased limit factors with ALAE and risk loads, and straight, franchise and diminishing deductibles, with the monograph's printed numbers beside prospicio's |
| [`distributions-for-actuaries-counts-aggregates.md`](distributions-for-actuaries-counts-aggregates.md) | CAS Monograph 2 (Bahnemann), chapters 3 and 4 and its aggregate limits reproduced: Poisson, mixed Poisson and negative binomial counts with contagion, aggregate moments and moment-matched approximations against Panjer's recursion, the FFT and simulation, and aggregate limits priced exactly and by the monograph's shifted gamma |

**Reserving**

| Guide | What it answers |
|---|---|
| [`reserving-methods.md`](reserving-methods.md) | Which reserving method when: chain ladder, Mack, Bornhuetter-Ferguson, bootstraps, Bayesian models, tails and the one-year view, with the reserve's uncertainty as a distribution |
| [`reserving-meyers.md`](reserving-meyers.md) | CAS Monograph 8 (Meyers) reproduced: from Mack and the ODP bootstrap to the Bayesian CRC, CSR and CAY models, what each changes, the validation by the outcome's percentile, p-p plots and the KS test, and his printed results on group 353 beside prospicio's with their Monte Carlo error |
| [`reserving-friedland.md`](reserving-friedland.md) | Friedland's *Estimating Unpaid Claims Using Basic Techniques* (CAS) reproduced on its own data: age-to-age averages and selected factors, the development, expected claims, Bornhuetter-Ferguson, Benktander and Cape Cod techniques on U.S. Industry Auto and XYZ Insurer, and the book's reasonableness checks, with its printed ultimates and IBNR beside prospicio's and the rounding conventions that explain every difference |
| [`reserving-taylor-mcguire.md`](reserving-taylor-mcguire.md) | CAS Monograph 3 (Taylor and McGuire) reproduced: the chain ladder as an over-dispersed Poisson GLM, its prediction error by the delta method and the parametric bootstrap, residual diagnostics and heat maps, and the reduced models with accident-year and development-year curves and interactions, with the monograph's printed numbers beside prospicio's |

## R versions

Every guide below has an R twin, a Quarto document next to it with the same name
and the same sections, models and numbers, written with the R package like the R
twins in [`examples/`](../../examples/README.md):

| R twin | Python guide |
|---|---|
| [`prospicio-for-aggregate-users.qmd`](prospicio-for-aggregate-users.qmd) | `prospicio-for-aggregate-users.md` |
| [`capital-modeling-monograph-15.qmd`](capital-modeling-monograph-15.qmd) | `capital-modeling-monograph-15.md` |
| [`capital-modeling-monograph-15-theory.qmd`](capital-modeling-monograph-15-theory.qmd) | `capital-modeling-monograph-15-theory.md` |
| [`capital-modeling-monograph-15-applications.qmd`](capital-modeling-monograph-15-applications.qmd) | `capital-modeling-monograph-15-applications.md` |
| [`pricing-insurance-risk-case-studies.qmd`](pricing-insurance-risk-case-studies.qmd) | `pricing-insurance-risk-case-studies.md` |
| [`pir-price-allocation.qmd`](pir-price-allocation.qmd) | `pir-price-allocation.md` |
| [`pir-price-allocation-practice.qmd`](pir-price-allocation-practice.qmd) | `pir-price-allocation-practice.md` |
| [`distributions-for-actuaries-claim-size.qmd`](distributions-for-actuaries-claim-size.qmd) | `distributions-for-actuaries-claim-size.md` |
| [`distributions-for-actuaries-counts-aggregates.qmd`](distributions-for-actuaries-counts-aggregates.qmd) | `distributions-for-actuaries-counts-aggregates.md` |
| [`ratemaking-werner-modlin.qmd`](ratemaking-werner-modlin.qmd) | `ratemaking-werner-modlin.md` |
| [`ratemaking-werner-modlin-indications.qmd`](ratemaking-werner-modlin-indications.qmd) | `ratemaking-werner-modlin-indications.md` |
| [`reserving-friedland.qmd`](reserving-friedland.qmd) | `reserving-friedland.md` |
| [`reserving-taylor-mcguire.qmd`](reserving-taylor-mcguire.qmd) | `reserving-taylor-mcguire.md` |
| [`reserving-meyers.qmd`](reserving-meyers.qmd) | `reserving-meyers.md` |
| [`glm-ratemaking-goldburd.qmd`](glm-ratemaking-goldburd.qmd) | `glm-ratemaking-goldburd.md` |
| [`glm-ratemaking-goldburd-validation.qmd`](glm-ratemaking-goldburd-validation.qmd) | `glm-ratemaking-goldburd-validation.md` |
| [`glm-to-comprehensive-pricing-chalk.qmd`](glm-to-comprehensive-pricing-chalk.qmd) | `glm-to-comprehensive-pricing-chalk.md` |
| [`glm-to-comprehensive-pricing-chalk-2.qmd`](glm-to-comprehensive-pricing-chalk-2.qmd) | `glm-to-comprehensive-pricing-chalk-2.md` |

Each R chunk is followed by the text it prints, and
`R/prospicio/tests/test-guides.R` (run by CI's R job with the other R tests) runs
every `.qmd` here and requires each chunk to print exactly that text. The `.qmd`
files are not pages of the Python site. With the R package installed, render one
with `quarto render docs/guides/prospicio-for-aggregate-users.qmd`, or check them
all from the repository root with `Rscript R/prospicio/tests/test-guides.R`.
Two blocks in `reserving-meyers` (the heavy Bayesian fits) are plain code fences
with recorded numbers, which neither harness re-runs.

Money in the guides is in millions of a synthetic currency, and every example is
synthetic. The guides write about the library and do not change it; a gap found
while writing one becomes an issue for the lane that owns the code
(`AGENTS.md`).
