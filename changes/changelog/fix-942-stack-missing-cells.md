### Reserving

- `Triangle::stack` of incremental programs keeps each program's own
  unobserved cells. A program valued earlier than the others, or missing a
  cell, no longer gains observed zero increments there, so `HierarchicalBf`
  built from incremental triangles with unequal valuations no longer reads
  them as a year of zero development (#942).
