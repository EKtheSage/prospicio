### Reserving

- Excess-layer development, PR 1 (Rust only; issue #192, `docs/design/reserving-excess.md`):
  `ExcessPattern::from_ground_up` derives a layer's age-to-age factors from a
  ground-up incurred triangle, an optional claim count column, the ground-up
  tail and the ultimate severity by a scale shift of the severity, reproducing
  Bloom and Patrik's worked example exactly; `ExcessPattern::selected` turns it
  into a `SelectedPattern`. `layer_share` and `layer_exposure` give the layer
  share of a severity and a layer exposure column for a BF or Cape Cod prior.
  A paid basis is an error. Python, R and the credibility blending follow (#507).
- A fully selected age whose own factor cannot be estimated (every earlier value
  zero, as in the early ages of a sparse layer) is no longer an error in
  `Development`: its own factor is missing (NaN) and the selected factor is used.
  Every result that was an estimate before is unchanged; Merz and Wuthrich's closed
  form and the bootstraps reject such a fit.
