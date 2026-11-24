### Reserving

- Indexation in Python and R (#979, PR 2): `CalendarIndex`, `Indexation`
  (`restate`, `inflate`, `reinflate`), `FutureCells` and `Reinflated` in
  `prospicio.reserving`; `calendar_index()`, `indexation()`, `restate()`,
  `inflate()`, `reinflate()` and `future_cells()` in R. Python builds
  `FutureCells` from columns, a chain-ladder fit (`segment=` chooses one of
  several) or an ODP GLM fit.
- **Breaking (Rust, unreleased):** `FutureCells::from_odp_glm` returns a
  `Result` instead of panicking.
