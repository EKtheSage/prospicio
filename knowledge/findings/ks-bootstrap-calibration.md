---
type: Finding
title: The classical KS p-value of a fitted severity rejects a sixth as often as it should; the parametric bootstrap is calibrated, and needs every loss's policy limit
description: "On 200 samples of 40 lognormal losses with reporting thresholds 0, 5,000 and 20,000 and a 100,000 limit, each tested against its own lognormal fit, the classical KS p-value (kstwo) fell at or below 0.10 three times, the parametric bootstrap with 99 replicates 19 times (20 expected; mean p-value 0.54). The bootstrap's null distribution reproduces Stephens' tables for the exponential and the normal with estimated parameters within 1.2% at n = 40, and its p-values agree with SciPy's goodness_of_fit(statistic='ks') within two standard errors. A loss that did not reach its limit was still exposed to it: replicates drawn without that limit show losses the data could not, so the limits are an argument."
tags: [prospicio-prob, severity, goodness-of-fit, bootstrap, scipy]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-12T12:00:00Z }
verified: { by: process:ci, at: 2026-11-12T12:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-prob/src/severity_fit.rs
    title: ks_bootstrap, KsBootstrap
  - id: test
    resource: ../validation/tests/severity_fit.rs
    title: ks_bootstrap_p_values_match_scipys_goodness_of_fit, the_bootstrap_null_distribution_is_stephens, the_bootstrap_p_value_is_calibrated_under_truncation_and_censoring
  - id: script
    resource: ../validation/scripts/scipy_ks_bootstrap.py
    title: the SciPy goodness_of_fit reference
  - id: stephens
    resource: https://doi.org/10.1080/01621459.1974.10480196
    title: Stephens, M. A. (1974), EDF statistics for goodness of fit and some comparisons, Journal of the American Statistical Association 69, 730-737, Table 1A; D'Agostino and Stephens (1986), Goodness-of-Fit Techniques, Tables 4.7 and 4.9
---

# The parametric bootstrap of a fit's KS p-value

#805 gave every severity fit the classical KS p-value, which assumes the
distribution was fixed before the data were seen. A fitted distribution
lies closer to its own data, so its distance is smaller and the classical
p-value too large (Lilliefors, 1967 and 1969).

## How far off the classical p-value is

`the_bootstrap_p_value_is_calibrated_under_truncation_and_censoring`
draws 200 samples of 40 lognormal losses (log mean 9.6, log sd 1.2), each
reported above 0, 5,000 or 20,000 in turn and capped at 100,000, fits a
lognormal to each and tests the fit against its own sample. Under this
null a calibrated p-value falls at or below 0.10 about 20 times. The
bootstrap's (99 replicates) did 19 times, its mean 0.54; the classical
one did 3 times.

## Agreement with the tables and with SciPy

With the exponential's mean, or the normal's two parameters, estimated,
the null distribution of `D` is the same whatever the true parameters, and
Stephens tabulates its upper points after a modification nearly free of
`n`: `(D - 0.2/n)(√n + 0.26 + 0.5/√n)` at 0.990, 1.094 and 1.308 (10%, 5%,
1%) for the exponential, and `D (√n - 0.01 + 0.85/√n)` at 0.819, 0.895 and
1.035 for the normal (the lognormal's logarithms, KS being invariant under
a monotone transform). With 9,999 replicates at `n = 40` the bootstrap's
points are 0.984, 1.085, 1.293 and 0.821, 0.890, 1.041. On 40 untruncated
gamma-like losses the five body families SciPy can fit with `loc` fixed
agree with `scipy.stats.goodness_of_fit(..., statistic="ks",
n_mc_samples=9999)` within two standard errors of the difference (for
example the exponential's 0.3531 against SciPy's 0.3485).

## Policy limits

`LargeLosses` records a limit only where it capped a loss. The replicate
of an uncensored loss under a limit must be censored at that limit too,
or the replicates contain losses larger than any the data could show and
the replicate distances are not the data's null distribution. The limits
are therefore an optional argument of `ks_bootstrap`; without it a
censored loss's limit is its own value and an uncensored loss has none.
