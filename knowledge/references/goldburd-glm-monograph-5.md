---
type: Reference
title: Goldburd, Khare, Tevet and Guller, GLMs for Insurance Rating (CAS Monograph 5), reproduced in the GLM ratemaking guides
description: What of CAS Monograph 5 can be recomputed (its printed arithmetic, all of which prospicio matches up to the monograph's rounding) and what cannot (every coefficient table is on insurers' data not in the PDF); two rounding gaps, three wrong figure references, and the identities checked with prospicio.
resource: https://www.casact.org/sites/default/files/2021-01/05-Goldburd-Khare-Tevet.pdf
tags: [glm, ratemaking, guides, parity, gini, tweedie]
status: stable
generated: { by: claude-code/guides-session, at: 2026-10-17T12:00:00Z }
sources:
  - id: mono
    resource: https://www.casact.org/sites/default/files/2021-01/05-Goldburd-Khare-Tevet.pdf
    title: Goldburd, Khare, Tevet and Guller, Generalized Linear Models for Insurance Rating, CAS Monograph 5, 2nd ed. (2025 revision)
  - id: guide1
    resource: ../docs/guides/glm-ratemaking-goldburd.md
    title: The GLM ratemaking guide, part 1
  - id: guide2
    resource: ../docs/guides/glm-ratemaking-goldburd-validation.md
    title: The GLM ratemaking guide, part 2 (refinement and validation)
  - id: data
    resource: ../validation/scripts/statsmodels_glm.py
    title: The generator of validation/data/glm_policies.csv
---

# What can be reproduced

The monograph's worked examples (Tables 4 to 12, Figure 9, Table 14, the
Gini of 56.1%, the AUROC of 0.857) are outputs on insurers' data that the PDF
does not contain, so no coefficient table can be refitted. What can be
recomputed is the arithmetic it prints from those outputs, and prospicio
matches all of it:[^mono][^guide1][^guide2]

| Monograph | Value | Note |
|---|---|---|
| 2.1.3 gamma severity example | 8.15, 3,463.38; factors 330.30, 12.182, 0.861; 3,464.42; 10,938 | 3,463.38 is the exact mean, 3,464.42 the product of the rounded factors |
| 2.2 variances | 3.6 and 35.9 million | `Gamma.from_mean_cv(mu, sqrt(phi)).variance()` |
| 2.4 factors | 3.421, 0.741, +242%, -25.9%; AOI 1.54; odds +27% | |
| 2.6 offsets | -0.0513, -0.1054, 5.4744 | |
| 2.7.3 equations 7 to 9 | Tweedie mean, power, dispersion from Poisson-gamma | equal to `Tweedie.from_poisson_gamma` |
| 6.2.1 F-test | F = 2.412, 95.2th percentile of F(4, 962) | 962 = 972 rows - 10 parameters of the big model; the F cdf is `Beta(d1/2, d2/2).cdf(d1 f / (d1 f + d2))` |
| 5.4.4 hinge slopes | 1.225, -1.044 | |
| 5.6 interactions | 10/20/15/33, -26.3%, -25.1%, 0.563 | |
| 7.3.1 confusion matrices | 0.358, 0.956, 0.044, 68.8%, 14.6% | |

# Rounding gaps (not errors)

- Table 5's truck coefficient (van as base) is 0.88, while Table 4's printed
  two-decimal coefficients give 0.57 + 0.30 = 0.87: the unrounded estimates
  must differ by at least 0.875.
- Table 12 (log AOI centred at 200,000) follows from Table 11 by adding the
  slope times ln 200,000 = 12.206: -8.9456 + 0.4239 x 12.206 = -3.7714 against
  the printed -3.7710, and 0.7447 - 0.1032 x 12.206 = -0.5150 against -0.5153.
  A rounding of 0.00005 in a four-decimal slope moves the product by 0.0006.

# Wrong figure references in the text

- Section 7.1 says the actual-versus-predicted plot is on a log scale "as was
  done in Figure 20"; the plot is Figure 21 (Figure 20 is residuals over
  exposure).
- Section 5.6.2 says "the right panel of Figure 13" for the centred-AOI model;
  the panel is in Figure 14 (Figure 13 is the two-by-two interaction table).
- Section 10.4 says the hand-built hinge model of section 5.4.4 had cut points
  at 2.75 and 3.5; section 5.4.4 and Table 7 use 2.75 and 3.6.

# Identities checked with prospicio

- Footnote 17, AUROC = 0.5 x normalized Gini + 0.5 with the normalized Gini
  the model's over the perfect model's: exact (to 1e-6 printed) with `gini`
  (ties grouped) against the Mann-Whitney AUROC on a logistic model.
- `gini` is the monograph's Gini (7.2.4): sorted by predicted rate, exposure
  share against loss share, twice the area to the diagonal.
- `lift` bands by predicted rate `pred / exposure` with equal exposure, the
  simple quantile plot of 7.2.1.
- Section 2.6's equivalence of a Poisson count model with a log exposure
  offset and a Poisson frequency model with exposure weights holds to 1e-9 in
  coefficients and standard errors.

# The data used instead

`validation/data/glm_policies.csv` (600 synthetic policies, known true
model)[^data]: its `claims` total 155 where the true model expects 134.0, so
fitted base rates sit about 1.8 standard deviations above the truth; the
region effects (true factors 1.2, 0.8, 1.5) are not significant on it, a
useful illustration of section 2.10 and 5.3. freMTPL2 is not available
offline and cannot be used in a guide that CI runs.

[^mono]: Goldburd, Khare, Tevet and Guller, CAS Monograph 5, 2nd ed. (2025 revision).
[^guide1]: docs/guides/glm-ratemaking-goldburd.md.
[^guide2]: docs/guides/glm-ratemaking-goldburd-validation.md.
[^data]: validation/scripts/statsmodels_glm.py.
