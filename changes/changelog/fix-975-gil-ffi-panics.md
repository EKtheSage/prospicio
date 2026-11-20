### Bindings

- Python: `fit_curves`, `Grid.local_moment`, `Grid.rounding`, `Grid.lower`,
  `ParetoAnchor.fit` and `PredictiveDistribution.join` release the GIL
  while they compute, so other Python threads keep running.
