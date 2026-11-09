### Breaking changes

- Rust: `prospicio_aggregate::Layer` has the public field
  `disappearing: Option<f64>`, so a struct literal of `Layer` needs it;
  build layers with `Layer::xol` and the builders, or add
  `disappearing: None` (#772).

### Aggregate and reinsurance

- A disappearing deductible on a layer, `aggregate` 2.1.0's `y xs d1 to d2
  disappearing` (`Layer::disappearing`; Python `Layer.with_disappearing`, R
  `with_disappearing()`), on events and on the grid, saved in the tower's
  JSON (format version 10), and the exact first three moments of one
  occurrence's recovery under an ordinary, franchise or disappearing
  deductible (`Layer::occurrence_moments`; Python and R
  `occurrence_moments`), at parity with `aggregate` 2.1.0's analytic
  severity moments (#772).
