### Reserving

- Mack's calendar-year effect test and development-factor correlation test (Mack 1994,
  appendices H and G; R ChainLadder's `cyEffTest` and `dfCorTest`): `calendar_year_effect` and
  `development_correlation` on a triangle's link ratios, and as `MackFit` methods on the link
  ratios the fit averaged, in Rust, Python (`prospicio.reserving.calendar_year_effect`,
  `development_correlation`, `MackFit.calendar_year_effect()`, `.development_correlation()`)
  and R (`calendar_year_effect()`, `development_correlation()`, with `print` and `summary`)
  (#664).
