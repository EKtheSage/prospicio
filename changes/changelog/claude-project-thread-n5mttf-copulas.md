### Distributions and probability

- `PairFamily::NAMES` and `Archimedean::NAMES`, with `from_name` and
  `name`: the copula family names defined once in Rust and shared by the
  Python, R and Excel parsers. The accepted names are unchanged; an
  unknown family's error now reads `family must be ..., got "..."` (and
  `pair family must be ...`) in Python and R alike (#1035).
