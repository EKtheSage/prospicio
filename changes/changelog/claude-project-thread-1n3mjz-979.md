### Reserving

- Stochastic re-inflation (#979, #1097): `Indexation::reinflate_draws` takes
  simulated future cells at the base level (as an `OdpGlmFit`'s
  `predict_distribution()` gives) and one calendar index path per
  simulation, and returns the draws in the money of their payment periods;
  Python's `Indexation.reinflate_draws(draws, paths=None)` and R's
  `reinflate_draws()`.
