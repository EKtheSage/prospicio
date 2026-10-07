---
type: Dataset
title: freMTPL2freq (French motor third-party liability)
description: 678,013 French motor TPL policies with claim counts and exposure; the GLM parity portfolio, fetched from OpenML, not committed.
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

# Download

OpenML's metadata gives the file at `https://openml.org/data/v1/download/20649148/freMTPL2freq.arff`,
on bare `openml.org`. The same path on `www.openml.org` serves it, so the
fetch script uses that host and needs only `www.openml.org` (and
`api.openml.org` for metadata) on the network allowlist (see
[cloud network](/environment/cloud-network.md)).[^fetch]

# Preparation (the GLM of Noll, Salzmann and Wüthrich)

ClaimNb capped at 4, Exposure at 1; Area as its rank 1–6; VehPower capped
at 9, a factor; VehAge classes 0-5, 6-12, 13+ (reference 6-12); DrivAge
classes 18-20, 21-25, 26-30, 31-40, 41-50, 51-70, 71+ (reference 41-50);
BonusMalus capped at 150; log Density; VehBrand, VehGas, Region factors
(reference R24). 49 coefficients with the intercept.[^noll][^fetch]

# Results

* Poisson GLM, offset log(Exposure): deviance 216,564.175, null deviance
  223,932.323, AIC 285,978.609; quasi-Poisson dispersion 2.585.
* `prospicio_glm` matches statsmodels 0.15.0 on all 202 values (coefficients at
  1e-8, standard errors at 1e-7, deviances at 1e-10).[^parity]
* Release-build fit time about 30 s, against 1 m 45 s for statsmodels on
  the same machine; unoptimized about 6 minutes, so the test runs in
  release only (CI job `fremtpl2`).

See [statsmodels](/references/statsmodels.md).

[^openml]: OpenML dataset 41214 (freMTPL2freq, version 1)
[^noll]: Noll, Salzmann and Wüthrich (2018)
[^fetch]: fetch_fremtpl2.py
[^parity]: freMTPL2 parity test
