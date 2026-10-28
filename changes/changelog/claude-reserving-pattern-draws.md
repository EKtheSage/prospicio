### Reserving

- A development pattern per simulation, aligned with the reserve draws, for
  the cash-flow engine's paid patterns (#600, #673): `DevelopmentPatterns` on the
  ODP and Mack bootstraps, `BayesOdp`, `BayesBf` and `BayesBfNormalised`
  (one fit or segments, carried through a rank correlation), with
  `pattern_draws` and `monthly` (on the origin's own basis or an
  underwriting basis, a falling pattern refused or held with
  `Monotone::RunningMax`), whose rows go to the engine's `PatternDraws`.
  The reserves are unchanged. **Breaking** in Rust: the fit structs that
  gain the public `patterns` field can no longer be built with a struct
  literal that leaves it out.
