### Reserving

- A paid development pattern per posterior draw from `Compartmental`, models
  1 and 2 (#679): `CompartmentalFit.patterns` and `CompartmentalFits.patterns`
  (`DevelopmentPatterns`), row `i` the pattern of the draw that simulation `i`
  took, the ODE's paid curve for the draw's rates over the same at the
  square's last age, for the cash-flow engine's `PatternDraws`. Python
  `fit.patterns`; R `pattern_draws()` and `monthly_pattern_draws()` take the
  fits. The reserves are unchanged. **Breaking** in Rust: `CompartmentalFit`
  and `CompartmentalFits` gain the public `patterns` field, so a struct
  literal that leaves it out no longer compiles.
