### Reserving

- The geometric separation method (#979, Rust):
  `SeparationKind::Geometric(Retransformation)` fits `ln s_ij` by least
  squares on development and calendar effects, with the proportions summing
  to 1 as the arithmetic method's, and takes the fit back to amounts by the
  geometric means (the default) or the lognormal means;
  `SeparationFit::log_variance` reports the residual variance on the log
  scale.
