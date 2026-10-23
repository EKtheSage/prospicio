### Breaking changes

- Rust: `prospicio_reserving::Error` has the variant `Diagonal`, so an
  exhaustive `match` on it needs an arm for it (#544).

### Reserving

- `LatestDiagonal` holds a bordereau's latest value per origin at its own
  age (or from a valuation date) with an optional exposure, and
  `fit_diagonal` on `ChainLadder`, `ExpectedLoss`, `BornhuetterFerguson`,
  `Benktander` and `CapeCod` projects it with a fully selected pattern
  from elsewhere. Ages between the pattern's are read log-linearly, so
  monthly or half-year ages work with an annual pattern; ages outside the
  pattern are an error. Rust only for now (#544).
