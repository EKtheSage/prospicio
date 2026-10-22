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
- A deal's deductions on a layer: brokerage, flat (`Layer::brokerage`) or
  on a loss-ratio scale (`brokerage_scale`), and `Deduction`s (a TPA fee,
  tax, reinsurance cost, fronting, a manager's fee) on the premium, the
  ceded loss or an amount, inside or outside NMBF, each its own component
  of a tower's results. `Tower::deal_ratios` gives each simulation's loss
  ratio, each deduction, the total, the combined ratio, NMBF and the
  margin, by year over a term. `Layer::scales_before_annual_terms` keys the
  commission and brokerage scales on the loss ratio before the annual terms
  and adds `loss_before_terms` to the results.
  `brokerage_within_commission_range` holds the brokerage key in the
  commission scale's range. Towers with them are JSON format version 3
  (Python `with_brokerage`, `with_brokerage_scale`, `with_deduction`,
  `with_scales_before_annual_terms`, `Tower.deal_ratios`; R
  `with_brokerage()`, `with_deduction()`, `deal_ratios()`) (#485, #477, #511).
- A deal-level scale on several layers together (`Tower::deal_scale` with a
  `DealScale`): a sliding-scale commission and a brokerage read on the sum
  of the sections' keys over the sum of their premiums, paid on that
  premium and added to each section's own, with its own carry over a term.
  The results gain `(ceding_commission, <deal>)` and `(brokerage,
  <deal>)`, and `Tower::deal_ratios` takes a deal's name. Towers with a deal
  are JSON format version 4; the grids and the P&L ledger refuse them
  (Python `Tower.with_deal_scale`, `Tower.deals`; R `with_deal_scale()`)
  ([deal-scale], #477).
