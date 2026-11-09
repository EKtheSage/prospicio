### Aggregate and reinsurance

- `aggregate` 2.1.0's lattice rule for a discrete severity's bucket:
  `dyadic_lattice_step` (the dyadic step the atoms lie on),
  `lattice_candidates` (the buckets tried for it) and `lattice_bucket` (the
  first that holds the extent within `2^log2` points, or a coarser one with
  a warning naming the `log2` needed); in Python and R (#776, #818).
