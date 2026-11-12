---
type: Finding
title: A gamma fitted above a high reporting threshold runs to shape 0, where one Nelder-Mead simplex never meets its tolerance
description: "On #800's 89 losses reported above 150,000 the gamma's truncated likelihood rises toward its supremum only as the shape goes to 0: f(y)/S(r) tends to y^-1 e^(-y/θ) / E1(r/θ), a proper distribution on (r, ∞). Nelder-Mead over log(shape) and log(scale) crawls along that ridge, each move changing the likelihood by about the shape (1e-8), and the simplex never shrinks to its tolerance: all 40,000 moves, 5.2 s a fit in release. Run in rounds of 250 moves from the best point, stopping when a round improves the negative log-likelihood by less than 1e-9 relative, it stops in 83 ms at a log-likelihood 3.0e-5 higher than the single simplex reached (-1174.4525365 against -1174.4525666; SciPy's reference -1174.4525871, tolerance 1e-4) and the bootstrap p-value is unchanged; fits that converge within a round take the same path as before. A bootstrap replicate refitted from the data's estimate does not walk the ridge again."
tags: [prospicio-prob, severity, optimization, nelder-mead, gamma, performance]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-12T16:00:00Z }
verified: { by: process:ci, at: 2026-11-12T16:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-prob/src/severity_fit.rs
    title: minimize_in_rounds, fit_body
  - id: test
    resource: ../validation/tests/severity_fit.rs
    title: every_family_matches_scipy (the gamma on plain, checked on its log-likelihood)
---

# The gamma at its boundary

`scipy_severity_fits.py` already noted that on the `plain` losses (89
reported above 150,000) the gamma's fit runs to shape 0, and checks it on
its log-likelihood alone. As the shape `a` goes to 0 with the scale `θ`
fixed, `f(y) / S(r) = y^(a-1) e^(-y/θ) / (θ^a Γ(a, r/θ))` tends to
`y^-1 e^(-y/θ) / E1(r/θ)`, a proper density on `(r, ∞)`, so the likelihood
has a supremum at the boundary, not a maximum inside it.

## What the simplex did

Over `(log a, log θ)` the simplex moves `log a` down a ridge on which the
negative log-likelihood falls by about `a` per unit: from 1174.452567 to
1174.452536 over 2,000 moves at `log a` near -18. Its tolerance needs both
the values and the diameter to shrink, and along the ridge the diameter
does not, so it ran all 40,000 moves allowed: 5.2 seconds a fit, and
`PRO.SEV.FIT.KS` would have spent a few minutes on the gamma alone.

## The fix

`minimize_in_rounds` runs 250 moves at a time, each round restarted from
the best point, and stops when a round meets the tolerance (as one simplex
did) or improves the value by less than 1e-9 relative. The gamma stops at
`log a` near -18 in 83 ms, its log-likelihood 3.0e-5 higher than after the
single simplex's 40,000 moves. Rounds of 2,000 moves with a 1e-10 test took 520
ms, since rounds near the boundary still improved by 1e-8 relative. A fit
that converges within its first round is unchanged, because the first
round is the same simplex as before.

For the bootstrap, each replicate refits a body family from the data's
estimate alone, with no standard errors; for the gamma that start is
already on the ridge. 999 replicates on the 89 losses take 13 seconds on
four threads in release, every other family under 2.
