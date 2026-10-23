### Reserving

- `ChainLadderFit::to_chain_ladder` freezes a fitted development (its
  factors in use and its tail factor) as a chain ladder, and
  `LatestDiagonal::from_triangle` takes a triangle's latest diagonal with
  an optional exposure column, so a development fitted on one book
  projects another book or a bordereau through `fit_diagonal`, by the
  chain ladder and the expected-loss methods, as chainladder-python's
  `fit(source).predict(target)` does. Rust only for now (#544).
