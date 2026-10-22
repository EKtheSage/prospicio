### Reserving

- Excess-layer development, PR 4 (#192): `LayerSharePrior` (Rust, Python and R:
  `layer_share_prior()`, `bayes_bf_layer()`, `bayes_bf_layer_one_year()`) turns
  a layer share and its uncertainty into the gamma prior of the normalised
  Bayesian Bornhuetter-Ferguson, centred on the ground-up loss ratio times the
  share, with the coefficient of variation of a product of independent
  variables. A wider share cv widens the layer's predictive interval.
