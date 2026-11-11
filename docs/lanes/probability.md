# Probability lane

**Session:** Cloud. **Owns:** `crates/prospicio-prob/`, `crates/prospicio-math/`, `validation/tests/distributions.rs`

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/probability.md](done/probability.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next

- [numeric-helpers-rest] #843: group 1's Fisher-Yates shuffle as a `StreamRng` method (a `prospicio-core` PR of its own) and the Spearman conversion and indexed reorder in reserving's dependence, then group 2 (`linalg` additions: `mat_mul`, the inverse by unit-vector solves, `cholesky_log_det`, a least-squares line) and group 3 (a fallible `try_bisect` and the 13 hand-written bisections, one site per PR, each keeping its tolerance; `tweedie` onto `golden_section`)
- [bindings-core-838] #838 items 2, 4 and 5: `HierarchicalBfFit::to_long` and `totals` in the core, the `read_ylt` day and hour rule in `YearLossTable` (a non-numeric day an error naming the row, the user's decision of 2026-11-11), and `underwriting_paid` as monthly paid shares from `prospicio-reserving`
- [option-names-step-2] #834 step 2: `from_name` and `name` on each user-facing core enum (count distribution, sigma interpolation and tail kind, grain, mixing, pattern kind, family and link, average kind), one enum family per PR, the front ends' match blocks replaced by a call
