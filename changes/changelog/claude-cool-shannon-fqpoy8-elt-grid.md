### Aggregate and reinsurance

- An event loss table as the severity of a compound Poisson on a grid
  (#781, #921): `EventLossTable::count`, `severity_grid` (aggregate's two
  rules: the nearest node for a table with secondary uncertainty, the
  mean-preserving linear scatter for its means alone) and `aggregate_grid`
  by FFT, for `fft`, `panjer` and `Tower::on_grid`; parity with
  `aggregate` 2.1.0. Python `EventLossTable.count`, `severity_grid` and
  `aggregate_grid`; R `elt_severity_grid()` and `elt_aggregate_grid()`.
