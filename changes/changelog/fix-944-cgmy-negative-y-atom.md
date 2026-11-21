### Distributions and probability

- A CGMY (`TemperedStable`) law with `Y < 0` is compound Poisson with an atom
  at `mu`; the characteristic-function engine now takes the atom out before
  inversion, so the bucket at `mu` and its neighbours no longer ring. New
  `TemperedStable::atom`. Masses near `mu` change for `Y < 0` only.
