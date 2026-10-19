---
type: Finding
title: On Pareto-like large losses the Burr fit runs to its Pareto limit, with the best likelihood and no standard errors
description: fit_severity("burr") on R Pareto's 14 example losses (1,100 to 40,000) stops at alpha 1.6e-9, gamma 4.0e8 and scale 1,100.0, a Pareto from the smallest loss with alpha 0.64; its log-likelihood (-140.32) beats the lognormal's (-140.98), its information is singular (no standard errors), and Burr::survival overflows there ((x/scale)^gamma is infinite), so the KS distance reads 0.93; PRO.SEV.FIT's note names the limit.
tags: [severity, fitting, burr, pareto, boundary, excel]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-19T09:00:00Z }
verified: { by: process:ci, at: 2026-10-19T09:30:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-prob/src/severity_fit.rs
    title: fit_severity's Burr, started from the Weibull fit and two heavier tails
  - id: test
    resource: ../crates/prospicio-xll/tests/severity_fit.rs
    title: compare_fits_every_body_family (the note on the Burr row)
  - id: data
    resource: ../validation/reference/pareto_fit_r.csv
    title: R Pareto 2.4.5's example losses (validation/scripts/r_pareto_fit.R)
---

The Burr's survival is `S(x) = (1 + (x/θ)^γ)^(−α)`. As `α → 0` and
`γ → ∞` with `αγ = a` held, it tends to `(θ/x)^a` above `θ` and to 1
below: a single-parameter Pareto from `θ`. Large losses that are
Pareto-like above a threshold are fitted best by that limit. The
likelihood keeps improving toward it, so the simplex runs out along the
ridge.

On R Pareto's 14 example losses, untruncated and uncensored,
`fit_severity(Family::Burr, ..)` gives:

- alpha 1.59e-9, gamma 4.01e8 and scale 1,100.0, which is the smallest
  loss.
- The product `αγ` is 0.64, close to the Pareto alpha from 1,000 (0.60,
  R's estimate).
- A log-likelihood of -140.32, against the lognormal's -140.98, the
  Weibull's -142.14 and the gamma's -142.18. It loses on AIC only through
  its third parameter (286.6 against 285.96).
- `converged` is false here. With the weights, thresholds and censoring of
  the add-in's test it reports true at the same kind of point.
- The observed information is singular, so the standard errors are NaN.

`Burr::survival` evaluates `(x/θ)^γ` directly, so at `γ = 4e8` it is
infinite for any `x > θ` and the survival is 0. The goodness of fit then
reads a distance of 0.93, and a layer priced on the fitted Burr would be
wrong. Above `θ` the exact value is `(θ/x)^{αγ}`, about 0.90 at 1,300.

`PRO.SEV.FIT` therefore adds a note to a body fit. For a Burr with alpha
below 1e-4 and gamma above 1e4, the note names the Pareto limit and its
alpha and says to fit `"pareto"` instead. Any other fit without standard
errors is noted as at a boundary. In `"compare"` the note sits on each
row, so the Burr is not picked on likelihood alone.
