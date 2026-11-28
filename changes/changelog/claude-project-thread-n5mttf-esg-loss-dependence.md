### Aggregate and reinsurance

- `dfa::LossDependence` and `Projection::with_loss_dependence`: the year's
  new losses depend on the year's growth of the claims (or price) index
  through a Gaussian, Student's t or Archimedean copula on their ranks,
  each year's losses keeping their distribution (#983).
