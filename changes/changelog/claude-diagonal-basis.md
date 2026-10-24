### Reserving

- `SelectedPattern::averaged_over_origins` turns a pattern measured on one
  origin basis (accident months) into the pattern of a coarser one
  (accident years or quarters), with the exposure even through the period,
  for both the triangle methods and a latest diagonal (#544, #614). Rust only
  for now.
- On a triangle, a selected pattern may now give ages the triangle does not
  have: a monthly pattern on an annual triangle selects its annual factors,
  a misaligned pattern is read between its ages, and a pattern that runs
  past the oldest age folds what it gives there, with the tail, into a
  constant tail (`ChainLadderFit::pattern_tail`, keeping a constant tail's
  decay), so a development fitted on an older book projects a younger one
  on the plain triangle path too; such a pattern must be fully selected
  (#544, #614). Rust only for now.
- On a latest diagonal, an origin older than the pattern's end is developed
  as `LatestDiagonal::with_past_pattern_end` says: an error by default
  (whose message names the settings), a factor of 1
  (`PastPatternEnd::NoDevelopment`), or an a-priori or estimated tail spread
  from the pattern's end at one step per origin period
  (`PastPatternEnd::Tail`) (#544, #614). Rust only for now.
