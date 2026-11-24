# Models lane

**Session:** Cloud (the Probability session). **Owns:** `crates/prospicio-models/`, `crates/prospicio-glm/`, `crates/prospicio-nn/`, `crates/prospicio-bayes/`, `validation/tests/models.rs`, `validation/tests/fremtpl2.rs`, `docs/design/models.md`

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/models.md](done/models.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next

- [survival-models] survival models, parked with claim-level reserving by the user on 2026-10-29 until the user has researched how to approach them; no lane works on them until the user picks them up (`docs/design/models.md`)
- [triangle-design-bridge] The triangle-to-design bridge and the diagonal backtest stay in the Reserving lane
- [glm-perf-gam-tweedie] #969 the rest of the issue: the GAM's REML score reuses IRLS's information for a canonical link and searches the gamma dispersion without a pass over the rows a step, and `tweedie_profile` fits its power grid in parallel (stacked on the first #969 PR)
- [glm-gam-edf-trace] the GAM's effective degrees of freedom take `tr(A⁻¹ JᵀWJ)` from a full `cholesky_inverse` of the penalised information at each REML step (`gam.rs`, both the shape-constrained and the plain path), O(p³) a step; triangular solves for the trace instead, when a GAM with many basis columns makes it matter (left out of #969, whose benchmarks had p of 10 to 30)
