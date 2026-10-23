### Breaking changes

- Rust: `prospicio_reserving::Error` has the variant `LatestDiagonalOnly`,
  so an exhaustive `match` on it needs an arm for it (#544, #571).

### Reserving

- A latest diagonal only (one value per origin at its own age, as a
  bordereau holds) is projected by the chain ladder, the expected loss
  method, Bornhuetter-Ferguson, Benktander and Cape Cod when the pattern is
  fully selected: a fully selected age with no link ratio is no longer an
  error. Mack, Mack's bootstrap and the ODP bootstrap reject such a
  triangle with `Error::LatestDiagonalOnly`, which names the method (#544, #571).
