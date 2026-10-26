### Breaking changes

- Rust: `TowerGrids` and `AnnualGrids` gain the public fields
  `expected_deal_ceding_commission` and `expected_deal_brokerage`; code
  that builds either in full adds them (empty `Vec`s for a tower without
  deals) (#541).

### Aggregate and reinsurance

- A deal's scales on the grid and in the cedant's P&L ledger: the grid
  methods and the ledger no longer refuse a tower with a `DealScale`.
  `Tower::on_annual_grid` gives each deal's expected sliding-scale
  commission and brokerage, read on its sections' loss ratio together as
  `apply_annual` reads it; `Tower::on_grid` does when no section has
  annual terms and names the section otherwise. The cedant's ledger books
  each deal's commission with the cover that buys its last section; the
  stitched peel, which is guaranteed cost, refuses it. Python
  `TowerGrids.expected_deal_ceding_commission` and
  `expected_deal_brokerage` (by deal name), R `tower_on_grid()` and
  `tower_on_annual_grid()`'s `expected_deal_ceding_commission` and
  `expected_deal_brokerage` (#541).
