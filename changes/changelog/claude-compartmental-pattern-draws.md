### Reserving

- A development pattern per simulation from `Compartmental`, models 1 and
  2 (#679): `patterns` (`DevelopmentPatterns`) on `CompartmentalFit` and
  `CompartmentalFits`, each simulation's expected cumulative paid of the
  ODE at each age of the square over that at its last age, for an origin
  with nothing observed at the population's rates (with each observed
  age's development effects in model 2), ending at 1 where the reserves
  end and carried through a rank correlation of segments, in Python
  (`fit.patterns`) and R (`pattern_draws()`, `monthly_pattern_draws()`).
  The reserves and cells are unchanged. The paid curve rises, so the
  shares never fall and `monthly`'s default takes them. **Breaking** in
  Rust: `CompartmentalFit` and `CompartmentalFits` gain the public
  `patterns` field and can no longer be built with a struct literal that
  leaves it out.
