### Reserving

- Latest diagonals over segments: `LatestDiagonals` holds a latest diagonal
  per segment, from a long table (ages, one valuation or a valuation per
  row) or any triangle, and `fit_diagonal_segments` and
  `fit_diagonal_segments_from` on the chain ladder, `ExpectedLoss`,
  `BornhuetterFerguson`, `Benktander` and `CapeCod` fit each segment, with
  each segment's development taken from a source fit by label, by a map
  (`SegmentMap::Pairs`) or from a source of one segment for every segment
  (#544, #622).
- Python: `LatestDiagonal` (from arrays, `at_valuation`, `from_frame` or
  `from_triangle`, with `with_exposure` and `past_pattern_end`),
  `fit_diagonal(diagonal, development=None, segment_map=None)` on the five
  methods, `ChainLadderFit.to_chain_ladder()`, and `ldf_origin` on every
  method that takes `ldf` for a pattern measured on another origin basis
  (#544, #622).
- R: `bordereau()` and `as_bordereau()`, taken by `chain_ladder()`,
  `expected_loss()`, `bornhuetter_ferguson()`, `benktander()` and
  `cape_cod()` with `development` and `segment_map`; `with_exposure()`,
  `with_past_pattern_end()`, `segment()` on a bordereau, `to_chain_ladder()`
  and `ldf_origin`. The expected-loss functions' `column` and `exposure` now
  default to `NULL` (still required for a triangle) (#544, #622).
- The `ldf` help in Python and R no longer says every age must be a
  development age of the triangle, and warns that one off-grid age changes
  how a sparse age-to-age pattern's other factors are read (#544, #622).
- `SelectedPattern::averaged_over_origins` with a weight by age (and
  `ldf_origin` with an `ldf_weight` by age in Python and R) is still an
  error, now ending "give one weight for every factor" rather than naming
  `with_weight` (#544, #622).
