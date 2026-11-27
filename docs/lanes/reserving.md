# Reserving lane

**Session:** Local. **Owns:** `crates/prospicio-reserving/`, `validation/tests/reserving*.rs`, `docs/design/triangle.md`, `docs/design/reserving-v02.md`, `docs/design/reserving-bayes.md`, `docs/design/reserving-compartmental.md`, `docs/design/reserving-excess.md`, `docs/design/reserving-ifrs17.md`, `docs/design/reserving-ifrs17-measurement.md`, `docs/design/reserving-inflation.md`

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/reserving.md](done/reserving.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next

- [option-names-mack] #990 next family: `MackProcess` names on the type (`NAMES`, `from_name`, `name`) used by the Python, R and Excel parsers, the error's list of names in one order (`docs/design/bindings.md`)
- [ifrs17-measurement] #982 IFRS 17 measurement of a group of contracts: the GMM with the CSM and the loss component, the OCI option, the PAA, reinsurance held (`reserving-ifrs17-measurement.md`, PRs 1 to 6); waits for the user's decisions on the note
- [option-names-pair-family] #990 next family after `MackProcess`: `PairFamily`, then `Archimedean`, `AnnualReading`, `Statistic`, `Smoothing` and `CorrelationKind`, each its names on the type used by the front ends' parsers (`docs/design/bindings.md`)
- [978-reserving-fits] #978 the other reserving fits after the chain ladder: JSON artifacts and Python pickling for `MackFit`, the Bornhuetter-Ferguson, expected-loss, Benktander and Cape Cod fits, `OdpBootstrapFit` and `MackBootstrapFit`, one PR each on the writers in `fit_json.rs`
- [ifrs17-oci] #982 IFRS 17 PR 2: the current curve beside the locked-in one (B97(a), Example 6) and the OCI option for insurance finance income or expense (Example 15, read first), then the PAA (PR 3, with the LIC), reinsurance held (PR 4), bindings and the JSON state (PR 5) and the guide (PR 6) (`reserving-ifrs17-measurement.md`)
- [ifrs17-add-contracts] #982 `Group::add_contracts`: contracts added within the cohort's year at a weighted-average locked-in rate (B72(b), decision 6) (`reserving-ifrs17-measurement.md`)
