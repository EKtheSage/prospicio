### Aggregate and reinsurance

- Reinsurer default on a tower in Python and R. Python binds
  `prospicio.reinsurance.Reinsurer(name, probability, recovery,
  collateral=0.0, correlation=0.0)` and `Panel(reinsurers)`, with
  `with_line`, `with_driver`, `apply(result, seed=0)` (the tower's
  one-year result with `credit_loss` and `recovered` by layer and
  `net_after_default`) and `by_reinsurer(result, seed=0)`. R binds
  `reinsurer()`, `reinsurer_panel()`, `panel_line()`, `panel_driver()`,
  `apply_panel()` and `panel_by_reinsurer()` (#980, #1084).
