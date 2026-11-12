### Excel

- `PRO.SEV.FIT.KS(losses, [family], [t], [thresholds], [censored],
  [weights], [upper], [limits], [replicates], [seed])`: each fitted
  family's KS distance with its classical p-value and the p-value by the
  parametric bootstrap, in `PRO.SEV.FIT(..., "compare")`'s order.

### Distributions and probability

- A body severity fit that runs to a boundary of its family stops there:
  the simplex runs in rounds and stops when one no longer improves the
  likelihood. The gamma fitted to losses above a high threshold took 5.2
  seconds and now takes under 0.1; fits that converged before are
  unchanged. `ks_bootstrap` refits each replicate from the data's
  estimate, several times faster.
