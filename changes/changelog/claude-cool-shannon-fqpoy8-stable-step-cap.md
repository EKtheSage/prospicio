### Distributions and probability

- Fixed: `invert_chf` on a stable law with two power tails (`|beta| < 1`)
  and a given `step` above `pi scale / (104 ln 2)^(1/alpha)` returned
  masses that were no distribution's (a sum of 21.2 at `alpha = 1.5`,
  `beta = 0.5`, `step = 0.5`), with nothing flagged. It now refuses the
  step and names the largest one the law can take; the automatic step is
  unchanged (#937).
