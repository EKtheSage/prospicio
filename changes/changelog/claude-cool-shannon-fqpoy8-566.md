### Aggregate and reinsurance

- Importance sampling of the tail: `simulate_events_weighted(frequency, severity, n_sims, count, tail, seed)` draws years from a proposal claim count of any family and a defensive tail mixture for the losses (`TailMixture`), with each year's likelihood ratio; without either the years are `simulate_events`'s. On a compound Poisson the TVaR at 0.999 has a standard error 14 times smaller than plain Monte Carlo's (`aggregate.md`, "Importance sampling of the tail"). Python `prospicio.aggregate.simulate_events_weighted`, R `simulate_events_weighted()` (#566, #703).

### Distributions and probability

- Weighted draws: `prospicio_prob::weighted::WeightedSample` gives an importance sample's self-normalised mean, VaR and TVaR with batch-means standard errors and the effective sample size. Python `prospicio.risk.WeightedSample`, R `weighted_tail()` (#566, #703).
