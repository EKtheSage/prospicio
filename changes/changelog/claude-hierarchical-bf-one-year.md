### Reserving

- `HierarchicalBf.one_year()` (R `bayes_hierarchical_bf_one_year()`, Rust
  `HierarchicalBf::one_year`): the claims development result over the coming
  year of every program and origin at once, as one joint distribution with
  dimensions `program` and `origin`, by Pareto smoothed importance sampling
  with moment matching and NUTS refits of the whole panel. A program's next
  diagonal moves the others' closing reserves, so it is one view and not one
  per program. On the reference panel every simulation needs a refit, so it
  runs at most 2,000 simulations by default (#599, #758).
