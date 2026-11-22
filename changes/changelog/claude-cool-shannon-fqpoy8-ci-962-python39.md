### Models

- Python: the means and standard errors of `Comparison`,
  `PermutationImportance`, `InteractionScreen` and `PartialDependence` are
  the same on every Python version. They were taken with the built-in
  `sum`, which adds floats with compensation from Python 3.12 only, so a
  value could differ in its last bit between 3.9 to 3.11 and 3.12; they
  now use `math.fsum` (#962, #1033).
