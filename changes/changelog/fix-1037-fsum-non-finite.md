### Models

- Python: the means and standard errors of `Comparison`, `PermutationImportance`, `InteractionScreen` and `PartialDependence` no longer raise for scores that are infinite or overflow. A score of +inf on one split and -inf on another gives a NaN mean, and scores of 1e308 give an infinite mean, as before #1033; finite scores are unchanged (#1037, #1038).
