### Breaking changes

- `TowerQuote::technical` (Python `TowerQuote.technical`, R `tower_quote()`)
  now solves each layer's deposit under its sliding-scale or profit
  commission. A layer with either term gets a higher technical deposit than
  before, when those commissions were left out; the total cession still
  carries only the flat commissions. A layer whose scale is read before
  annual terms (`scales_before_annual_terms`) is now refused there, where
  its scale was left out before.

### Pricing

- A layer's technical deposit solved under loss-sensitive commissions:
  `Quote::technical_for_layer` finds the deposit at which the loads and the
  expected sliding-scale and profit commission, read off the layer's ceded
  grid at that deposit (`Layer::expected_loss_sensitive_commissions`), leave
  the price's risk premium. A steep scale with several such deposits gives
  the lowest. A swing-rated layer is refused, since it has no deposit to
  solve for. Python `Quote.technical_for_layer`; R `layer_quote()`.
