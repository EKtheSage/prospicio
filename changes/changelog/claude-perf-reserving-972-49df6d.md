### Reserving

- The ODP, Mack and one-year bootstraps and the Bayesian models draw each
  process error and prior value without allocating a one-element vector
  (`Distribution::sample_one`); their seeded numbers are unchanged (#972, #1056).
- `HierarchicalBf` computes its log posterior's gradient by hand in one
  pass instead of several passes of the chunked dual numbers, which makes
  its fit and its one-year refits faster (#972, #1056).

### Breaking changes

- `HierarchicalBf`'s seeded draws, reserves and one-year views change in
  their last digits: the hand-written gradient agrees with the dual one to
  rounding, not bit for bit, so the sampler's trajectories differ (#972, #1056).
