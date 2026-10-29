---
type: Finding
title: A Hawkes year started from an empty history has fewer events than the stationary rate, and the stationary count variance needs its window correction when the decay is slow
description: "For a Hawkes process with an exponential kernel, a year simulated from no history has mean count mu T / (1 - n) - mu n (1 - exp(-beta (1 - n) T)) / (beta (1 - n)^2), 3.92 rather than 4 at mu 2, alpha 12, beta 24; a burn-in of a few times 1 / (beta (1 - n)) years restores the stationary rate. Hawkes and Oakes' stationary variance of the count over a window, mu / (1 - n)^3 (T - n (2 - n) (1 - exp(-beta (1 - n) T)) / (beta (1 - n))), matches 60,000 simulated years within 3% at mu 2, alpha 2, beta 4, where the correction is a third of mu T / (1 - n)^3."
tags: [aggregate, catastrophe, hawkes, clustering, event-sets, prospicio-aggregate]
status: draft
verified: { by: process:ci, at: 2026-10-29T12:00:00Z }
generated: { by: claude-code/cloud-session, at: 2026-10-29T12:00:00Z }
sources:
  - id: design
    resource: ../docs/design/cat-inputs.md
    title: cat-inputs.md decision 17
  - id: code
    resource: ../crates/prospicio-aggregate/src/hawkes.rs
    title: the Hawkes module and its tests
---

# Hawkes years

`Hawkes::event_set` draws each year independently. From an empty history
the first events of a year have nothing to excite them, so the mean count
is below the stationary `μ / (1 − n)`; `mean_count(1)` gives it exactly.
With `burn_in` years simulated first and dropped, the year starts near the
stationary state; the excitation forgets at rate `β (1 − n)`, so a burn-in
of a few times `1 / (β (1 − n))` years is enough.

The stationary variance of a window's count is about `μ T / (1 − n)³`
only when `T` is many times `1 / β`. At `μ = 2, α = 2, β = 4` over one
year the window correction is a third of it, and the full formula matches
60,000 simulated years within 3% (the unit test
`the_mean_and_variance_match_the_closed_forms`).
