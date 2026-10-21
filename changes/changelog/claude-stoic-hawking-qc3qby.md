### Distributions and probability

- Bootstrap standard errors of a statistics grid
  (`StatisticsGrid::bootstrap_se`): each replicate resamples the
  simulations once for every column, so the items' errors share their
  resamples, and is read off counts on each column sorted once;
  `bootstrap_se_holding` keeps a fixed part of each simulation and
  resamples only the rest. Python `bootstrap_se` (with `hold_fixed`), R
  `bootstrap_se()` (with `fixed`) (#488, #503).
- Where each simulation came from in a reorder:
  `PredictiveDistribution::reorder_groups_indexed` and
  `reorder_groups_by_copula_indexed` return each group's source
  simulations beside the result (Python `return_index=True`, R
  `index = TRUE`), so anything simulated with a group follows it without
  matching values (#483, #504).

### Aggregate and reinsurance

- `EventSet::permute` takes a set's years in another order, losses, sums
  insured, times and labels together: with a group's index from a reorder
  it keeps a class's large events with its reordered totals (Python
  `EventSet.permute`, R `permute_events()`) (#483, #504).
- An aggregate part on an `EventSet` (`EventSet::with_aggregate_part`):
  amounts a year with no event or time, attritional losses by peril or
  class, which quota shares and stop losses cede from
  (`Layer::sees_aggregate_part`) and per-occurrence layers skip; it comes
  before the events in the year under annual terms, and gross, net and
  `totals` include it (Python `EventSet.with_aggregate_part`,
  `aggregate_part`, `Layer.sees_aggregate_part`; R `with_aggregate_part()`,
  `aggregate_part()`) (#483, #505).
