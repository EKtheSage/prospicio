# Probability lane

**Session:** Cloud. **Owns:** `crates/prospicio-prob/`, `crates/prospicio-math/`, `validation/tests/distributions.rs`

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/probability.md](done/probability.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next
- [faer-decision] #974 waits for the user: the evaluation recommends not adopting faer now (`knowledge/findings/faer-evaluation.md`); on a yes to that, close the issue, and on a no, put faer behind `prospicio_math::linalg` for matrices of 32 by 32 and more
- [eigen-tridiagonal] a tridiagonal reduction with implicit QL in place of cyclic Jacobi for `symmetric_eigen` and `symmetric_eigenvalues`, only when a caller's profile shows the eigendecomposition (`knowledge/findings/faer-evaluation.md`); not started
- [978-rest] #978 the rest of the issue: pickling for `Triangle`, `EventSet`, `Design`, the reserving fits and the other fitted models, JSON artifacts for the GLM fits `artifact.rs` lacks (`GlmmFit`, `DglmFit`, `MarsFit`, `QuantileFit`, `ExpectileFit`, `ZeroInflatedFit`, `HurdleFit`, `NestedGlmFit`) and the Bayesian GLM fits, R matrices returned as `RMatrix` instead of reshaped with `matrix(..., byrow = TRUE)`, and NumPy or Arrow zero-copy transfers in Python (the scope waits on the user)
