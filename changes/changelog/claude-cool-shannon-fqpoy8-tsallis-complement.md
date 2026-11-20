### Distributions and probability

- Fixed: `Distortion::complement` (`1 - g(s)`) of a Tsallis distortion
  lost every digit at orders above about 100 for `s` between 0.75 and 1
  (0.0453 for 0.2362 at `alpha = 200`, `s = 0.76`), which moved the
  natural allocation's capital per unit of margin under a thin Tsallis
  margin. It is now accurate to 1e-13 for orders up to 1e6 (#938).
