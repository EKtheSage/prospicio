# Reserving lane

**Session:** Local. **Owns:** `crates/prospicio-reserving/`, `validation/tests/reserving*.rs`, `docs/design/triangle.md`, `docs/design/reserving-v02.md`, `docs/design/reserving-bayes.md`, `docs/design/reserving-compartmental.md`, `docs/design/reserving-excess.md`, `docs/design/reserving-ifrs17.md`, `docs/design/reserving-ifrs17-measurement.md`, `docs/design/reserving-inflation.md`

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/reserving.md](done/reserving.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next

- [option-names-mack] #990 next family: `MackProcess` names on the type (`NAMES`, `from_name`, `name`) used by the Python, R and Excel parsers, the error's list of names in one order (`docs/design/bindings.md`)
- [ifrs17-measurement] #982 IFRS 17 measurement of a group of contracts: the GMM with the CSM and the loss component, the OCI option, the PAA, reinsurance held (`reserving-ifrs17-measurement.md`, PRs 1 to 6); waits for the user's decisions on the note
