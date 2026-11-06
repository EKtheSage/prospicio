### Reserving

- `HierarchicalBf` (R `bayes_hierarchical_bf()`): a Bornhuetter-Ferguson over
  several programs, one segment of a triangle each, or named triangles in
  Python and R. An industry hyperprior anchors the loss ratio, programs are
  shrunk towards it by a sampled spread, the development pattern is
  partially pooled and the ODP scale is pooled per unit of premium. The fit
  gives each program's reserves, loss ratio and shrinkage, the
  hyperparameters' posterior and one joint distribution of all reserves
  (#599).
