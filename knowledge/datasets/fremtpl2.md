---
type: Dataset
title: freMTPL2freq and freMTPL2sev (French motor third-party liability)
description: 678,013 French motor TPL policies with claim counts and exposure, and their 26,639 claim amounts; the GLM parity portfolio, fetched from OpenML, not committed.
resource: https://www.openml.org/d/41214
tags: [glm, parity, claim-frequency, openml]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-05T22:05:00Z }
verified: { by: process:ci, at: 2026-10-05T16:34:15Z }
sources:
  - id: openml
    resource: https://api.openml.org/api/v1/json/data/41214
    title: OpenML dataset 41214 (freMTPL2freq, version 1)
    author: team:openml
  - id: openml-sev
    resource: https://api.openml.org/api/v1/json/data/41215
    title: OpenML dataset 41215 (freMTPL2sev, version 1)
    author: team:openml
  - id: noll
    resource: https://ssrn.com/abstract=3164764
    title: Noll, Salzmann and Wüthrich (2018), Case Study - French Motor Third-Party Liability Claims
  - id: fetch
    resource: ../validation/scripts/fetch_fremtpl2.py
    title: fetch_fremtpl2.py
  - id: parity
    resource: ../validation/tests/fremtpl2.rs
    title: freMTPL2 parity test
---

# Facts

| Fact | Value |
|---|---|
| Policies | 678,013 |
| OpenML file | ARFF, 36 MB, SHA-256 `a45363e056e2ea56408b38eeb9d4d04d7f6c6982eb7a14ed5e807c7c71807cdd`[^openml] |
| Columns | IDpol, ClaimNb, Exposure, Area (A–F), VehPower, VehAge, DrivAge, BonusMalus, VehBrand (B1–B14), VehGas, Density, Region (22 levels) |
| Origin | The R package CASdatasets (Charpentier) |
| Policies with a claim | 34,060 (ClaimNb > 0) |
| freMTPL2sev | 26,639 claim amounts (IDpol, ClaimAmount) on 24,950 policies; ARFF, 0.4 MB, SHA-256 `047632016b87e132247124f449a95d6d0e9f8ba05f2fe0947ef76ab8aaa10e0a`[^openml-sev] |
| Join | 24,944 policies, 26,444 amounts: 6 policies of freMTPL2sev (195 amounts) are not in freMTPL2freq; 9,116 policies with ClaimNb > 0 have no amounts; on 9 joined policies the number of amounts differs from ClaimNb |

# Download

OpenML's metadata gives the file at `https://openml.org/data/v1/download/20649148/freMTPL2freq.arff`,
on bare `openml.org`. The same path on `www.openml.org` serves it, so the
fetch script uses that host, as for freMTPL2sev
(`/data/v1/download/20649149/freMTPL2sev.arff`), and needs only `www.openml.org` (and
`api.openml.org` for metadata) on the network allowlist (see
[cloud network](/environment/cloud-network.md)).[^fetch]

# Preparation (the GLM of Noll, Salzmann and Wüthrich)

ClaimNb capped at 4, Exposure at 1; Area as its rank 1–6; VehPower capped
at 9, a factor; VehAge classes 0-5, 6-12, 13+ (reference 6-12); DrivAge
classes 18-20, 21-25, 26-30, 31-40, 41-50, 51-70, 71+ (reference 41-50);
BonusMalus capped at 150; log Density; VehBrand, VehGas, Region factors
(reference R24). 49 coefficients with the intercept.[^noll][^fetch]

The severity file aggregates freMTPL2sev per IDpol (NSev amounts, their
mean AvgClaim) and inner-joins it to the prepared rows, in freMTPL2freq's
order. Every factor level of the design occurs among the 24,944 joined
policies, so the severity design has the same 49 columns.[^fetch]

# Results

* Poisson GLM, offset log(Exposure): deviance 216,564.175, null deviance
  223,932.323, AIC 285,978.609; quasi-Poisson dispersion 2.585.
* `prospicio_glm` matches statsmodels 0.15.0 on all 202 values (coefficients at
  1e-8, standard errors at 1e-7, deviances at 1e-10).[^parity]
* Logistic regression of ClaimNb > 0 on the same design without an
  offset: deviance 265,572.373, null deviance 270,129.437, AIC
  265,670.373.
* Gamma GLM, log link, of AvgClaim on the same design without an offset,
  `var_weights` NSev (prospicio's prior weights): deviance 39,130.019,
  null deviance 45,021.223, Pearson scale 18.631, AIC 497,529.346.
* Both match statsmodels 0.15.0 at the Poisson case's tolerances, HC0
  standard errors included (303 more values): largest relative
  differences on 2026-10-12, Windows, coefficients 1.0e-9 (logit) and
  1.8e-9 (gamma), standard errors 3.0e-11, the gamma scale 5.2e-11,
  deviances below 1.1e-11.[^parity]
* Release-build time for the three tests about 20 s, against 2 m 30 s for
  statsmodels on the same machine; unoptimized several minutes, so the
  test runs in release only (CI job `fremtpl2`).
* Regenerating the reference on Windows (numpy 2.5.4, scipy 1.18.1)
  moves the Poisson rows written on Linux by about 1e-13 relative, far
  inside their tolerances; the existing rows were kept as written.

See [statsmodels](/references/statsmodels.md).

[^openml]: OpenML dataset 41214 (freMTPL2freq, version 1)
[^openml-sev]: OpenML dataset 41215 (freMTPL2sev, version 1)
[^noll]: Noll, Salzmann and Wüthrich (2018)
[^fetch]: fetch_fremtpl2.py
[^parity]: freMTPL2 parity test
