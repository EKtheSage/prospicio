### Reserving

- Indexation of a triangle by a calendar index (#979, Rust): `CalendarIndex`
  (one value per period of its grain, extended at a constant annual rate),
  `Indexation::restate` and `inflate` (each increment at the price level of a
  base period, by the index of the period containing its valuation month, in
  every segment and either basis) and `Indexation::reinflate`, which takes
  `FutureCells` from a chain ladder or an `OdpGlm` fit back to the money of
  their payment periods. New error variant `Error::Inflation`.
