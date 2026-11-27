### Reserving

- `SegmentDependence::NAMES`, `from_name` and `name`: the segment
  dependence names defined once in Rust and shared by the Python and R
  parsers; `"rank_correlation"` comes back with an empty matrix for the
  front end to fill. Names and messages are unchanged (#1035).
