# Reserving lane

**Session:** Local. **Owns:** `crates/prospicio-reserving/`, `validation/tests/reserving*.rs`, `docs/design/triangle.md`, `docs/design/reserving-v02.md`, `docs/design/reserving-bayes.md`, `docs/design/reserving-compartmental.md`, `docs/design/reserving-excess.md`, `docs/design/reserving-ifrs17.md`, `docs/design/reserving-inflation.md`

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/reserving.md](done/reserving.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next

- [inflation-index] #979 indexation, Rust (`reserving-inflation.md` PR 1): `CalendarIndex`, `Indexation` (`restate`, `inflate`, `reinflate`), `FutureCells` from a chain ladder or an `OdpGlm` fit, exact-identity validation
- [inflation-index-bindings] #979 Python and R bindings for the indexation (`reserving-inflation.md` PR 2), in new module files
- [separation-arithmetic] #979 Taylor's arithmetic separation method, Rust (`reserving-inflation.md` PR 3), validated on Taylor (1977) Example 1 and a statsmodels Poisson GLM reference
- [separation-geometric] #979 Taylor's geometric separation method, Rust (`reserving-inflation.md` PR 4), after its open decisions 2 and 3
- [separation-bindings] #979 Python and R bindings for the separation method (`reserving-inflation.md` PR 5)
- [apc-glm] #979 the age-period-cohort reserving GLM, Rust (`reserving-inflation.md` PR 6): canonical identification, I2, I1 and I0 calendar extrapolation, after its open decision 5
- [apc-glm-bindings] #979 Python and R bindings for the APC GLM (`reserving-inflation.md` PR 7)
- [reinflate-draws] #979 stochastic re-inflation (`reserving-inflation.md` PR 8, decision 11): `Indexation::reinflate_draws` over a `PredictiveDistribution` of future cells with one index path per simulation, in Rust, Python and R, after PR 2
