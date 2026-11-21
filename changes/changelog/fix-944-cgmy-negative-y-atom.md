### Distributions and probability

- A CGMY (`TemperedStable`) law with `Y < 0` is compound Poisson with an atom
  at `mu`; the characteristic-function engine now takes the atom out before
  inversion, so the bucket at `mu` and its neighbours no longer ring. New
  `TemperedStable::atom`. Masses near `mu` change for `Y < 0` only. A grid
  that does not hold `mu` (one the user gives with `step` and `x_min`, or an
  automatic window that leaves out a negligible atom when the jumps go mostly
  one way) leaves the atom in, as before, instead of failing; so does a
  Tweedie's grid without 0 (#944, #1007).
