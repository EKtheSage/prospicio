---
type: Finding
title: A blended development pattern in Merz-Wuthrich's one-year view and in the bootstraps
description: With a selected factor blended as w s + (1 - w) e, Merz-Wuthrich's terms scale by g = 1 - w, but not simply by g^2. The new cell's link ratio carries the blend's parameter variance g^2 tau, not tau, and in later years the scales depend on the share k of the year-start volume already there at the valuation. The coming year's scales carried to later years are 3% to 6% too low against a simulation on GenIns; the derived ones agree within Monte Carlo error. The ODP bootstrap keeps the origins' level error with every factor selected (73% of the SD on RAA), Mack's bootstrap does not.
tags: [reserving, pattern, mack, merz-wuthrich, one-year, cdr, bootstrap, odp]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-14T20:00:00-07:00 }
sources:
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 12, items 6 to 8 and As built
  - id: closed-form
    resource: ../crates/prospicio-reserving/src/one_year.rs
    title: MackFit::claims_development_result with a selected pattern
  - id: test
    resource: ../validation/tests/reserving_patterns.rs
    title: Exact, simulated and bootstrap checks of the blended one-year view
  - id: unit
    resource: ../crates/prospicio-reserving/src/pattern.rs
    title: Unit tests of patterns through Mack and the bootstraps
---

# Finding

A selected pattern is taken as known: the used factor `w s + g e`
(`g = 1 - w`) has parameter variance `g^2 sigma^2 / S`, and Mack's
lifetime parameter terms scale by `g^2`.[^design] The one-year view of
Merz and Wüthrich does not scale so simply, because the next year's
estimate takes in a new cell whose link ratio scatters around the true
factor, and the true factor is now the blend's, with variance
`g^2 sigma^2 / S`, not the estimate's.[^closed-form] In the coming year, with
`s` the new cell's share and `t` R's parameter term:

* own link `g^2 t`; a later re-estimated link `g^2 (1 - s (1 - g^2)) s t`;
  the cross term of an origin's own link with a younger origin's
  re-estimate of it `g (1 - s (1 - g^2)) t`.

In a later year the estimate at the year's start already holds new cells,
so with `k` the share of its volume there at the valuation (R's `kept`)
and `a = 1 - k` the own term is `g^2 ((1 - g a)^2 / k + a) t`, a later
link's `g^2 (1 - s k (1 - g^2)) s t`, and the cross term
`g (1 - s (1 - g^2 (1 - g a) - g a)) t`. All are 1 at `w = 0` and 0 at
`w = 1`.

# Evidence

* The coming year equals the first-order covariance of the model computed
  directly from the triangle's cells to 1e-9 on RAA, GenIns and ABC, for a
  uniform and an age-by-age blend of the triangle's own factors.[^test]
* Every year of the run-off agrees with 40,000 simulated run-offs of the
  model on GenIns within five Monte Carlo standard errors (largest gaps
  0.9% without a pattern, 1.1% with `w = 0.5` and by age). The coming
  year's scales carried to later years are 3% to 6% below the simulation
  from the second year on.[^test]
* Mack's bootstrap re-reserving with the same blend reconciles with the
  coming year within five Monte Carlo standard errors on GenIns.[^test]
* Strictly between 0 and 1 the used factor follows the data less than the
  estimate does, so the years' CDRs are correlated and their variances need
  not add up to Mack's; at `w = 1` they add up to Mack's process risk.[^unit]
* The ODP bootstrap projects each origin from its pseudo latest value, so
  with every factor selected and no process error its reserve still varies
  (on RAA, 73% of the unblended SD, seed 5): the ODP's parameter error in
  the origin effects. Mack's bootstrap projects from the observed latest
  value and gives the deterministic reserve in every simulation.[^unit]

[^design]: Design note, reserving v0.2, decision 12, items 6 to 8 and As built
[^closed-form]: `MackFit::claims_development_result` in
    `crates/prospicio-reserving/src/one_year.rs`.
[^test]: `validation/tests/reserving_patterns.rs`:
    `merz_wuthrich_coming_year_is_the_linearised_blend`,
    `merz_wuthrich_later_years_match_a_simulation_of_the_blend` and
    `mack_bootstrap_one_year_reconciles_with_the_blended_closed_form`.
[^unit]: Unit tests in `crates/prospicio-reserving/src/pattern.rs`
    (`merz_wuthrich_at_the_limits`, `odp_bootstrap_keeps_the_levels_error_only`,
    `mack_bootstrap_with_a_fully_selected_pattern`).
