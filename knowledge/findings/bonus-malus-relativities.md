---
type: Finding
title: Bonus-malus stationary levels and Norberg relativities against closed forms and a SciPy reimplementation
description: On two levels and on -1/+top the bottom level's stationary probability and relativity under a gamma Theta have closed forms, (a/(a+n lambda))^a and a/(a+n lambda), met to 1e-12; an independent SciPy implementation agrees on three scales to 1e-15 (stationary), 2e-12 (relativities) and 6e-10 (efficiency); no printed table from Lemaire or Denuit et al. was available, so nothing is at parity with a published source.
tags: [prospicio-pricing, bonus-malus, markov-chain, credibility, motor]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-27T22:00:00Z }
verified: { by: process:ci, at: 2026-10-27T22:00:00Z }
sources:
  - id: module
    resource: ../crates/prospicio-pricing/src/bonus_malus.rs
    title: prospicio_pricing::bonus_malus (BonusMalus, Portfolio)
  - id: script
    resource: ../validation/scripts/bonus_malus_reference.py
    title: The SciPy reimplementation behind validation/reference/bonus_malus_scipy.csv
  - id: test
    resource: ../validation/tests/pricing.rs
    title: bonus_malus_matches_scipy
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/563
    title: Issue #563, bonus-malus and no-claim-discount systems as Markov chains
---

# Finding

A scale's bottom level is reached only by `n` claim-free years in a row
(`n = 1` on two levels, `n = s` on `-1/+top` with top level `s`), so under
a gamma `Θ` with shape and rate `a`, `π_0 = E[e^{-nλΘ}] = (a/(a+nλ))^a` and
Norberg's `r_0 = E[Θ | L = 0] = a/(a+nλ)`. `prospicio_pricing::bonus_malus`
meets both to `1e-12`, which checks the quadrature on `ln θ` and the
relativity's weighting at once.

An independent implementation (`validation/scripts/bonus_malus_reference.py`:
the table rebuilt from the rule, least squares for `π`, QUADPACK for the
integrals, a central difference for Loimaranta's efficiency) on `-1/+top`
with six levels (gamma `cv` 1), `-1/+2` with ten levels (inverse Gaussian
`cv` 0.8) and `-1/+3` with twelve levels and three a priori classes (gamma
`cv` 1.3) agrees to:

| quantity | largest difference |
|---|---|
| stationary distribution over the portfolio | 9e-16 |
| stationary distribution at one frequency | 2e-15 |
| optimal relativities | 1.7e-12 |
| Loimaranta efficiency | 5.9e-10, the reference's difference error |
| years to stationarity | 0 |

The relativities balance, `Σ π_l r_l = 1`, to `1e-12` in every case, which
is the identity that tells `E[(Θ - r_L)²]` from the `λ_k²`-weighted
convention once there are a priori classes.

Lemaire's (1995) and Denuit, Maréchal, Pitrebois and Walhin's (2007)
printed tables were not available (no copy in `knowledge/references/`, and
the session could not reach the publisher), so the reference is a second
implementation of the same formulas, not a published source.
