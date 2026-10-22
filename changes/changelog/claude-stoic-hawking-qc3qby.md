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

- `Grid::normalized` rescales probabilities that do not sum to 1, which
  `Grid::new` rejects (Python `Grid(..., normalize=True)`, R
  `grid_distribution(..., normalize = TRUE)`), and
  `Distortion::parameter` gives a one-parameter family's parameter (Python
  `Distortion.parameter`, R `@parameter`) (#214).

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
- The reinsurer's P&L ledger of a layer or a deal (`Tower::deal_ledger`):
  one sell group over the annual gross losses with the premium, the ceded
  loss, the commissions, the brokerage and each deduction as legs, the
  amounts `deal_ratios` divides by the premium (Python `Tower.deal_ledger`;
  R `deal_ledger()`) ([deal-ledger], #477).
- A monthly cash-flow engine for collateralised deals
  (`prospicio_pricing::cashflow`): a trust account and an investor's
  operating account per simulation, from its ultimate loss ratio and
  commission, with every rule a setting (the collateral requirement as
  loads on named bases with a floor and a switch to the limit, a
  commutation requirement, premium collection and adjustments, funding
  dates, rule and source, release timing, interest, opening capital,
  expenses, a buffer and a tiered performance fee); `CashflowTerms::run`
  gives the investor's monthly flows for `npv_rows` and `irr_rows`, and
  `trace` every account line of one simulation (#479); in Python (`prospicio.pricing.CashflowTerms`, `CollateralRequirement`,
  `PremiumAdjustment`, `Cashflows`) and R (`cashflow_terms()`,
  `cashflow_requirement()`, `cashflow_adjustment()`, `cashflow_run()`,
  `cashflow_trace()`).
- `PaymentPattern::unpaid_from(age)`: the pattern of what is unpaid at an
  age, times from the age and shares over the outstanding (Python
  `PaymentPattern.unpaid_from`, R `payment_pattern_unpaid()`), and
  Pentagon's loss under Price's name, `expected_loss`, in Python (#214).
- A book of deals on one shared simulation (`prospicio_pricing::book`):
  `DealRun` holds a deal's premium, loss and deductions in each
  simulation (from `Tower::deal_columns`, the amounts behind
  `deal_ratios`, or from arrays), its inception in months and its
  investor's flows; `DealBook` selects and shifts deals and gives every
  amount by deal and in total with ratios as sums over sums, the
  distribution of an item with each deal's contribution and marginal
  effect on either tail, the investors' flows on one calendar with the
  book's IRR, NPV and MOIC on the summed flows, and premium, paid loss
  and flows by calendar year (#490); in Python (`DealRun`, `DealBook`,
  `BookCashflows`) and R (`deal_run()`, `deal_book()` and the `book_*()`
  functions).
- Underwriting years with a rank correlation for multi-year terms:
  `EventSet::correlated_years` reorders whole simulated years so their
  totals have a Spearman or normal-score correlation, every year keeping the
  one-year distribution, and `calibrate_year_correlation` finds the
  correlation for a target ratio of the term's spread to one year's (#489);
  in Python (`EventSet.correlated_years`, `.calibrate_year_correlation`,
  `year_correlation_for_ratio`) and R (`correlated_years()`,
  `calibrate_year_correlation()`, `year_correlation_for_ratio()`).
