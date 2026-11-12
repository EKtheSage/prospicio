### Bindings

- A selected pattern's kind takes its names from one table
  (`PatternKind::NAMES` with `from_name` and `name` in
  `prospicio-reserving`): `ldf_kind` in Python and R, the benchmark curve's
  `kind` and Excel's `kind` accept the same spellings, and an unknown name's
  error lists them the same way in all three (#834).
