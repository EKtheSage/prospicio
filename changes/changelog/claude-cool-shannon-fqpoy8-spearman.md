### Distributions and probability

- A Spearman target of 1 between two groups is accepted by
  `PredictiveDistribution::reorder_groups_with` (Python and R through the
  same call) and gives near-comonotone group totals. Before, only the
  diagonal's 1 stayed 1 in `CorrelationKind::Spearman`'s conversion and an
  off-diagonal 1 made the matrix singular, which was refused. Reserving's
  rank-correlated segments already accepted it and now use the same
  conversion (#843).
