### Aggregate and reinsurance

- Reinsurer default over a contract term: `Panel::apply_term` and
  `Panel::by_reinsurer_term` take a term's paths (`Tower::apply_term`,
  `apply_aggregate_term`, `apply_timeline`) and the number of years. A
  default is final for the rest of the path, and the collateral is held
  once per reinsurer and path and used up year by year. Python binds
  `Panel.apply_term(result, n_years, seed=0)` and
  `Panel.by_reinsurer_term(result, n_years, seed=0)`, R
  `apply_panel_term()` and `panel_by_reinsurer_term()`; R's
  `reinsurer_panel` gains a `reinsurers` property (#980).
