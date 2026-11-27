# Models lane

**Session:** Cloud (the Probability session). **Owns:** `crates/prospicio-models/`, `crates/prospicio-glm/`, `crates/prospicio-nn/`, `crates/prospicio-bayes/`, `validation/tests/models.rs`, `validation/tests/fremtpl2.rs`, `docs/design/models.md`

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/models.md](done/models.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next

- [survival-models] survival models, parked with claim-level reserving by the user on 2026-10-29 until the user has researched how to approach them; no lane works on them until the user picks them up (`docs/design/models.md`)
- [triangle-design-bridge] The triangle-to-design bridge and the diagonal backtest stay in the Reserving lane
- [glm-gam-edf-trace] the GAM's effective degrees of freedom take `tr(A⁻¹ JᵀWJ)` from a full `cholesky_inverse` of the penalised information at each REML step (`gam.rs`, both the shape-constrained and the plain path), O(p³) a step; triangular solves for the trace instead, when a GAM with many basis columns makes it matter (left out of #969, whose benchmarks had p of 10 to 30)
- [977-numerics] #977 the numerical routines left in domain crates, into `prospicio-math`: `gauss_hermite` (`glmm.rs`) into `integrate`, `solve_semidefinite` (`monthly_reading.rs`) into `linalg`, the Levenberg-Marquardt `gauss_newton` (`curve.rs`) into `optimize`
- [978-glm-artifacts] #978 the GLM family's remaining JSON artifacts and pickling: `GlmmFit`, `DglmFit`, `MarsFit`, `QuantileFit`, `ExpectileFit`, `ZeroInflatedFit`, `HurdleFit` and `NestedGlmFit` in `artifact.rs`, and `BayesGlmFit`/`BayesGlmmFit` in `prospicio-bayes`
