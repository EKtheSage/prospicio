# Reading list: the source texts and the guides that reproduce them

The guides in this folder follow published actuarial texts and recompute their
worked examples with prospicio, printing the text's value beside the library's.
This page maps each text to its guides (Python, with an R twin of the same name
and numbers), to the knowledge file that records its quirks and the parity
checks behind it, and to the open issues for what prospicio does not yet do.
The texts themselves are not in the repository; each guide explains in its own
words and says where it cannot reproduce a printed number.

Knowledge files are in the repository's
[`knowledge/`](https://github.com/EKtheSage/prospicio/tree/main/knowledge)
bundle; issue numbers are on GitHub.

## Pricing and capital

| Source | Guides | Knowledge | Open issues |
|---|---|---|---|
| Mildenhall's `aggregate` package, documentation 1.2.0 | [prospicio for aggregate users](prospicio-for-aggregate-users.md) | [coverage map](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/aggregate-docs-coverage.md), [aggregate reference](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/aggregate-mildenhall.md) | none open |
| Mildenhall and Major, *Pricing Insurance Risk* (2022): case studies | [the case studies](pricing-insurance-risk-case-studies.md) | [aggregate reinsurance pricing](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/aggregate-reinsurance-pricing.md) | none open |
| *Pricing Insurance Risk*: Parts I to III (risk measures, portfolio pricing, price allocation) | [risk measures](pir-risk-measures.md), [portfolio pricing](pir-portfolio-pricing.md), [price allocation](pir-price-allocation.md), [allocation practice](pir-price-allocation-practice.md) | [Parts I and II](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/pricing-insurance-risk-parts-1-2.md), [Part III](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/pir-price-allocation.md) | none open |
| Major and Mildenhall, CAS Monograph 15, *Introduction to Capital Modeling and Portfolio Management* (2026) | [chapter 9, the `aggregate` code](capital-modeling-monograph-15.md); [chapters 3 and 4](capital-modeling-monograph-15-theory.md); [chapters 5 and 6](capital-modeling-monograph-15-applications.md); [chapters 1, 2 and 7](capital-modeling-monograph-15-practice.md); [chapter 8, Iman-Conover and capital by layer](capital-modeling-monograph-15-iman-conover.md) | [chapters 3 to 6](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/major-mildenhall-monograph-15-chapters-3-6.md) | none open |
| Werner and Modlin, *Basic Ratemaking* (CAS, 5th ed. 2016) | [the steps](ratemaking-werner-modlin.md), [the indications and increased limits](ratemaking-werner-modlin-indications.md) | [parity and rounding](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/werner-modlin-ratemaking.md), [guides note](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/werner-modlin-guides.md) | none open |
| Clark, *Basics of Reinsurance Pricing* (CAS) | [experience and exposure rating](experience-and-exposure-rating.md) | [Clark note](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/clark-reinsurance-pricing.md) | none open |
| Riegel, *Matching tower information with piecewise Pareto*; the local Pareto distribution | [severity and frequency](severity-and-frequency.md), [experience and exposure rating](experience-and-exposure-rating.md) | [R Pareto](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/r-pareto.md) | none open |

## Distributions and GLMs

| Source | Guides | Knowledge | Open issues |
|---|---|---|---|
| Bahnemann, CAS Monograph 2, *Distributions for Actuaries* (2015) | [claim size, excess claims, limits and deductibles](distributions-for-actuaries-claim-size.md), [claim counts and aggregates](distributions-for-actuaries-counts-aggregates.md) | [Bahnemann note](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/bahnemann-distributions-for-actuaries.md), [R API map](https://github.com/EKtheSage/prospicio/blob/main/knowledge/findings/bahnemann-r-api-map.md) | none open |
| Goldburd, Khare, Tevet and Guller, CAS Monograph 5, *GLMs for Insurance Rating* (2nd ed.) | [the rating plan](glm-ratemaking-goldburd.md), [validation](glm-ratemaking-goldburd-validation.md) | [Monograph 5 note](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/goldburd-glm-monograph-5.md) | none open |
| Chalk, Deacon, Guillen and others, CAS Monograph 16, *From GLMs to Comprehensive Insurance Pricing* | [chapters 3 to 6](glm-to-comprehensive-pricing-chalk.md), [chapters 7 to 10](glm-to-comprehensive-pricing-chalk-2.md) | [Monograph 16 note](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/chalk-monograph-16.md), [R API map](https://github.com/EKtheSage/prospicio/blob/main/knowledge/findings/chalk-monograph-16-r-api-map.md) | none open |

## Reserving

| Source | Guides | Knowledge | Open issues |
|---|---|---|---|
| Friedland, *Estimating Unpaid Claims Using Basic Techniques* (CAS) | [reserving with Friedland's techniques](reserving-friedland.md) | [Friedland note](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/friedland-estimating-unpaid-claims.md), [chainladder-python expected loss](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/chainladder-python-expected-loss.md) | none open |
| Taylor and McGuire, CAS Monograph 3, *Stochastic Loss Reserving Using GLMs* | [the ODP GLM reproduced](reserving-taylor-mcguire.md) | [Monograph 3 note](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/taylor-mcguire-monograph-3.md), [R API map](https://github.com/EKtheSage/prospicio/blob/main/knowledge/findings/taylor-mcguire-r-api-map.md) | none open |
| Meyers, CAS Monograph 8, *Stochastic Loss Reserving Using Bayesian MCMC Models* (2nd ed.) | [Meyers' models reproduced](reserving-meyers.md) | [Monograph 8 parity](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/meyers-monograph-8.md), [reproduced](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/meyers-monograph-8-reproduced.md) | none open |
| Gesmann and Morris (2020), compartmental reserving; England, Verrall and Wuthrich (2012) | [compartmental models reproduced](reserving-compartmental.md), [reserving methods](reserving-methods.md) | [Gesmann and Morris](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/gesmann-morris-2020.md), [EVW 2012](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/evw-2012-bayesian-bf.md) | none open |
| Bloom and Patrik (CAS E-Forum, 2013), Pinto and Gogol (PCAS 1987) and Bear's discussion (PCAS 1992), excess loss development | [excess-layer loss development](reserving-excess-layers.md) | [excess development literature](https://github.com/EKtheSage/prospicio/blob/main/knowledge/references/excess-development-literature.md), [simulation study](https://github.com/EKtheSage/prospicio/blob/main/knowledge/findings/excess-simulation-study.md), [Bayesian prior](https://github.com/EKtheSage/prospicio/blob/main/knowledge/findings/excess-bayes-prior.md) | none open |

## How the texts agree with the library

- Where a text prints enough to recompute a number, the guide does, and nearly
  every printed value agrees once the text's own rounding is applied. The
  guides name each rounding, and call a printed value a misprint only with
  evidence from the text itself.
- Seeded MCMC and bootstrap digits differ by platform and toolchain, so the
  checked blocks of the Bayesian guides print the text's value, a tolerance and
  whether they agree; heavy fits are recorded numbers that CI does not re-run.
- The capital by unit under the lifted allocation for CCoC and PH differs
  between `aggregate` 0.30.1 and the Monograph 15 tables on one side and
  `aggregate` 1.0.1 and prospicio on the other; premiums and margins agree
  everywhere. See [chapter 9 of Monograph 15](capital-modeling-monograph-15.md),
  section 5.
