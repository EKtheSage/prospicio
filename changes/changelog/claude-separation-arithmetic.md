### Reserving

- Taylor's (1977) arithmetic separation method (#979, Rust): `Separation`
  with an optional normaliser per origin and future calendar indices from a
  rate, a `CalendarIndex` or the exponential trend of the last fitted
  indices (`FutureInflation`), giving `SeparationFit` with the development
  proportions, the calendar indices, the fitted and future cells
  (`FutureCells`, ready for `Indexation::reinflate`) and the reserves; also
  `fit_segments`. On a triangle restated by an economic index the indices
  measure superimposed inflation.
