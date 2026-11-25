### Reserving

- Stochastic re-inflation of reserve draws (#979, PR 8): `IndexPaths`, one
  future index path per simulation, and `Indexation::reinflate_draws`,
  which re-inflates a `PredictiveDistribution` of future cells (the draws
  of `OdpGlmFit` and `ApcGlmFit`) cell by cell, each simulation under its
  own path, or under the deterministic index without paths. In Python,
  `IndexPaths` and `Indexation.reinflate_draws`; in R, `index_paths()` and
  `reinflate_draws()`.
