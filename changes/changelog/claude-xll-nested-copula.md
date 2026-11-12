### Excel

- `PRO.CAP.NESTED.CDF(points, family, nodes, members)` and
  `PRO.CAP.NESTED.SIM(family, nodes, members, [sims], [seed])`: nested
  Archimedean copulas (Clayton or Gumbel) on a sheet, the tree as a table
  of nodes (name, parent, theta) and the node each dimension joins; the
  simulation runs asynchronously and draws what Python's
  `NestedArchimedeanCopula.sample` draws at the same seed.
