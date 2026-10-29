### Reserving

- A development pattern per simulation from Meyers' CRC, CSR and CAY
  (#678): `patterns` (`DevelopmentPatterns`) on `MeyersFit` and
  `MeyersFits`, each simulation's `exp(beta_d g)` of the posterior draw it
  took (`g` the speed-up of the origin after the newest in CSR, 1 in CRC
  and CAY), ending at 1 at the oldest age and carried through a rank
  correlation of segments, in Python (`fit.patterns`) and R
  (`pattern_draws()`, `monthly_pattern_draws()`). The ultimates and
  reserves are unchanged. A share usually falls somewhere and often
  passes 1 (on Meyers' group 353 data), so `monthly` needs
  `Monotone::RunningMax`. **Breaking** in Rust: `MeyersFit`
  and `MeyersFits` gain the public `patterns` field and can no longer be
  built with a struct literal that leaves it out.
