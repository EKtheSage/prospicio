### Aggregate and reinsurance

- `TowerModel::diagnostics` warnings print each position at six
  significant figures (`the piece from 1.28998`, not every digit of the
  float); Python `TowerModel.diagnostics()` and R `tower_diagnostics()`
  carry the same text.
