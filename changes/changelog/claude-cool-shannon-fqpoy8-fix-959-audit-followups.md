### Distributions and probability

- `risk::var_linear_sorted` gives NaN for draws that hold a NaN, in debug
  builds as in release builds. Before, a debug build panicked on its
  sortedness assertion, which reached the posterior summaries of
  `prospicio-bayes` (#959, #1024).
- `PredictiveDistribution::reorder_groups` and the reserving
  `SegmentDependence::RankCorrelation` built on it now state their rule for
  simulations whose group totals tie exactly: they rank in their order, the
  earlier one lower. The results are unchanged (#959, #1024).
- `optimize::golden_section_by` returns NaN at once for a bracket that is
  not finite. Before, with no step cap, the search could not end (#959, #1024).
