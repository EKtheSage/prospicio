---
type: Reference
title: Chalk, Deacon, Guillen and Martinelli, CAS Monograph 16, reproduced in the guides
description: What the two Monograph 16 guides found; the case-study data (FAA-NTSB, STATS19, the simulated auto rows) is not available offline, so only the printed worked examples reproduce; chapter 8's offset example reproduces from its printed cell counts; three printed slips (section 5.2's three-step frequencies, the 222.22 of section 6.7.1, truncated digits in 4.7.2 and Table 5.7); ElasticNet is slow on hinge bases.
resource: https://www.casact.org/monographs
tags: [pricing, glm, elastic-net, lasso, gam, target-encoding, boosting, guides, parity]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-18T07:30:00-07:00 }
verified: { by: process:ci, at: 2026-10-18T07:30:00-07:00 }
sources:
  - id: m16
    resource: https://www.casact.org/monographs
    title: Chalk, Deacon, Guillen and Martinelli, From GLMs to Comprehensive Insurance Pricing - Techniques and Challenges, CAS Monograph 16 (2025), ISBN 978-1-7370028-9-5
  - id: guide1
    resource: ../docs/guides/glm-to-comprehensive-pricing-chalk.md
    title: The guide to chapters 3 to 6
  - id: guide2
    resource: ../docs/guides/glm-to-comprehensive-pricing-chalk-2.md
    title: The guide to chapters 7 to 10 and the gaps
  - id: test
    resource: ../python/tests/test_guides.py
    title: The harness that runs both guides and checks every printed number in CI
---

# What it is

The CAS monograph that follows Goldburd et al.'s GLM monograph: pseudo-R2 on
held-out folds, penalized regression, step and hinge functions chosen by the
lasso, target encoding of high cardinality factors, correlated features,
two-stage models with offsets, and learning from a gradient boosting benchmark
with permutation importance, partial dependence and a permutation interaction
screen.

# Facts

| Fact | Value |
|---|---|
| Data | The case studies use FAA registry plus NTSB accidents (an R package on GitHub, not offline), STATS19 summarized by the authors, and simulated auto rows that are not published; their fitted numbers cannot be recomputed |
| Pseudo-R2 | `1 - D_model / D_null` with the null at the validation data's own mean; Table 3.1's 47% reproduces with `models.deviance` |
| Chapter 8 | Tables 8.2 (cell counts, 1,000,000 policies) and 8.3 (truth, base 10%) specify the data; a Poisson GLM on the 25 cells with expected claims gives the truth for age and NCB and, with the marketing NCB offset, intercept 0.1125 and age 0.849, 0.918, 0.983, 1.059 against the printed 0.112 and 0.849, 0.925, 0.980, 1.057 (the monograph's simulation noise) |
| Section 5.2 slip | The three-step frequencies print about 2% below those the printed coefficients give (0.00783 against 0.00799, all four bands by the same factor); an intercept near -4.85 rather than -4.8292 fits them |
| Section 6.7.1 slip | The worked sentence multiplies z by 222.22; Table 6.5's AvE ratio 223.75 is what gives the printed 3.58. Table 6.5's exposures are rounded (Marlin's z 0.0122 implies about 12.65, not 13) and its caption states the AvE ratio upside down |
| Truncated digits | Section 4.7.2 prints 1000/50005 as 0.019 and the frequency as 1.9 (exact 0.019998, 1.9998); Table 5.7 prints 2.06 where Table 5.8 prints 2.07 (exact 2.067) |
| Table 9.11 | Columns are L_AB, L_A, L_B and the strength; recomputed 0.097 and 0.036 against printed 0.096 and 0.037 (inputs rounded) |
| Area groups | Section 8.9.2's step 1.0162 is 1.01615 rounded; 1.0162^99 = 4.908 overshoots the 4.884 range |
| Ridge and credibility | ElasticNet(alpha=0, standardize=False) on one factor's dummies with no intercept gives `n_k (ybar_k - ybar) / (n_k + K)` with `K = lam * n`, section 6.8.1's credibility form |
| Speed | ElasticNet on 20 hinge columns of 441 aggregated cells took about 40 s against 0.3 s for 20 step columns; 10 hinges about 4 s (Windows, release build) |
