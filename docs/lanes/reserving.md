# Reserving lane

**Session:** Local. **Owns:** `crates/prospicio-reserving/`, `validation/tests/reserving*.rs`, `docs/design/triangle.md`, `docs/design/reserving-v02.md`, `docs/design/reserving-bayes.md`, `docs/design/reserving-compartmental.md`, `docs/design/reserving-excess.md`, `docs/design/reserving-ifrs17.md`

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/reserving.md](done/reserving.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next
- [split-r-reserving] #963 the R half: `crates/prospicio-r/src/reserving.rs` (8,265 lines) split into a `reserving/` module tree the same way as the Python bindings, with no change to the R API
