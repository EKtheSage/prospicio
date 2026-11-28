### Aggregate and reinsurance

- `prospicio-esg` simulates economic inflation (an Ornstein-Uhlenbeck rate,
  `Inflation`) and superimposed claims inflation above it (an AR(1) annual
  excess, `Superimposed`), their shocks correlated with the short rate's
  through `Esg::with_correlation`. `Scenarios` gives the inflation rate, the
  price and claims index levels and each year's excess, and
  `index_levels` gives the levels by year, quarter or month for
  `prospicio_reserving::IndexPaths`, so reserve cells re-inflate along the
  scenarios through `Indexation::reinflate_draws`. Rust only for now (#983).
