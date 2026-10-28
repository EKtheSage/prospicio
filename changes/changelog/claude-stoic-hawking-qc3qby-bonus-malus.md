### Pricing

- Bonus-malus and no-claim-discount scales as Markov chains:
  `bonus_malus::BonusMalus` from a rule or a table of next levels, with
  the transition matrix, the stationary distribution, the years to reach
  it and Loimaranta's efficiency at a claim frequency, and over a mixed
  Poisson portfolio (a priori classes and a gamma or inverse Gaussian
  heterogeneity) the stationary distribution of the levels and Norberg's
  optimal relativities. Python `prospicio.pricing.BonusMalus`, R
  `bonus_malus()` and `bonus_malus_portfolio()` (#563, #656).
