### Reserving

- Excess-layer development, PR 3 (Python and R; issue #192, `docs/design/reserving-excess.md`):
  `prospicio.reserving` has `ExcessLayer`, `ScaleShift`, `ParetoShape`,
  `ExcessPattern.from_ground_up` (with `selected`, `tail`, `ldf` and
  `credibility_weights`), `layer_share`, `layer_exposure` and `layer_at_shift`;
  R has `excess_layer()`, `scale_shift()`, `pareto_shape()`, `excess_pattern()`,
  `excess_selected()`, `excess_credibility_weights()`, `layer_share()`,
  `layer_exposure()` and `layer_at_shift()`. A layer's development pattern
  derived from the ground-up triangle goes into `ChainLadder`, `Mack` and the
  expected-loss methods through their `ldf`, `ldf_kind`, `ldf_weight` and `tail`
  arguments. Bloom and Patrik's, Pinto and Gogol's and Bear's examples are
  reproduced in both languages (#513).
