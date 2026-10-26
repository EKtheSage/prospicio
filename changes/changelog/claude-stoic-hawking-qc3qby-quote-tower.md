### Breaking changes

- Rust: `TowerGrids` gains the public field `total_ceded`; code that
  builds one in full adds it.

### Pricing

- A whole tower quoted at once: `TowerQuote::technical` quotes each layer
  standalone from its ceded grid and the total cession as one cover, the
  diversification being the layers' deposits less the total's. The total
  comes from the new `TowerGrids::total_ceded` (the whole tower's cession
  as one compound, `None` for several layers with annual terms or an
  aggregate cover after per-occurrence stages, when the quote is refused).
  Python `TowerQuote.technical` and `TowerGrids.total_ceded`; R
  `tower_quote()` and `tower_on_grid()$total_ceded`. Quotes in R now also
  carry `reinstatement_rate` and `commission_rate`.
