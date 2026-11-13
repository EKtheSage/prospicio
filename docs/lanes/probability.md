# Probability lane

**Session:** Cloud. **Owns:** `crates/prospicio-prob/`, `crates/prospicio-math/`, `validation/tests/distributions.rs`

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/probability.md](done/probability.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next

- [bindings-core-838] #838 items 2, 4 and 5: `HierarchicalBfFit::to_long` and `totals` in the core, the `read_ylt` day and hour rule in `YearLossTable` (a non-numeric day an error naming the row, the user's decision of 2026-11-11), and `underwriting_paid` as monthly paid shares from `prospicio-reserving`
- [option-names-step-2] #834 step 2: `NAMES` and `from_name` (with `name`) on each user-facing core enum, as `CountDistribution` has them, one family per PR: tail kind, mixing, family and link, average kind, distortion family; the front ends' match blocks replaced by a call and their `unknown_name` message
