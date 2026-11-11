### Reserving

- The tables of `HierarchicalBf`'s fit and one-year view come from the
  Rust core (`HierarchicalBfFit::totals`, and `to_long` and `totals` on its
  one-year view), the same columns in Python's `to_frame` and
  `totals_frame` and R's `as.data.frame()` as before; the one-year view's
  claims development result moments are now computed as every other
  model's (#838).
