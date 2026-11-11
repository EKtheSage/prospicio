# Probability lane

**Session:** Cloud. **Owns:** `crates/prospicio-prob/`, `crates/prospicio-math/`, `validation/tests/distributions.rs`

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/probability.md](done/probability.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next
- #804 the fit-curves table: every family's conditional survival against the empirical one, families in AIC order (Rust, Python `fit_curves`, R; then the Excel lane's `PRO.SEV.FIT` `"curves"`)
- #805 the Kolmogorov p-value, tail mean and tail CV on a severity fit's goodness of fit (Rust, Python, R; then the Excel lane's columns)
