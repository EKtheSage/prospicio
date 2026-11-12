# Probability lane

**Session:** Cloud. **Owns:** `crates/prospicio-prob/`, `crates/prospicio-math/`, `validation/tests/distributions.rs`

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/probability.md](done/probability.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next

- [reference-generators] #844 items 2 and 3: pinned environments for the Python generators (PEP 723 inline metadata or one requirements file per reference package family, so `uv run validation/scripts/<name>.py` rebuilds a reference) and a weekly job, outside the required checks, that reruns the cheap pure-Python generators and compares their output with the committed files within each row's tolerance
- [numeric-helpers-rest] #843: the Spearman conversion and indexed reorder in reserving's dependence (group 1), group 2's least-squares line (slope, intercept, R²) for the six hand-written fits and the two `TrendFit` types, and group 3 (a fallible `try_bisect` and the 13 hand-written bisections, one site per PR, each keeping its tolerance; `tweedie` onto `golden_section`)
- [bindings-core-838] #838 items 2, 4 and 5: `HierarchicalBfFit::to_long` and `totals` in the core, the `read_ylt` day and hour rule in `YearLossTable` (a non-numeric day an error naming the row, the user's decision of 2026-11-11), and `underwriting_paid` as monthly paid shares from `prospicio-reserving`
- [option-names-step-2] #834 step 2: `NAMES` and `from_name` (with `name`) on each user-facing core enum, as `CountDistribution` has them, one family per PR: sigma interpolation and tail kind, grain, mixing, pattern kind, family and link, average kind, distortion family; the front ends' match blocks replaced by a call and their `unknown_name` message
- [linalg-helpers] #843 group 2, first PR (branch `claude/cool-shannon-fqpoy8-linalg`): `prospicio_math::linalg::mat_mul`, `inverse` (by unit-vector solves, the caller keeping its failure policy) and `cholesky_log_det`, replacing four `p × p` products (credibility, moment matching, the robust sandwich, Kolmogorov's matrix power), three inverses (Clark with its `rcond` test, the two severity fits' covariances) and four log determinants (GAM, GLMM, double GLM, moment matching); same summation order, so every parity row is unchanged
