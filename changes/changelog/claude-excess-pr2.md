### Reserving

- Excess-layer development, PR 2 (Rust only; issue #192, `docs/design/reserving-excess.md`):
  `ExcessPattern::credibility_weights` gives the weight on the derived layer
  pattern for a blend with the layer triangle's own factors, `1 - n/(n + k)`
  with `n` the layer claims (or origins with a positive value) and `k`
  required, capped so that removing one origin moves the blended factor by at
  most `max_influence` of it, and exactly 1 where the own factor cannot be
  estimated. `SeverityByMaturity::ParetoShape` with `ShapeBase` derives layer
  factors from Pinto and Gogol's Pareto shape drift and a base development
  (the counts above the threshold or a base layer's losses), reproducing their
  and Bear's examples. A 500-book simulation study checks the derived pattern
  and the blend against the layer's own chain ladder (#510).
