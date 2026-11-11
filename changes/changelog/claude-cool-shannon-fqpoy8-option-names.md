### Bindings

- One vocabulary for option names in Python, R and Excel (#834, step 1).
  Each front end now accepts every spelling another one did: `negbin` and
  `negative_binomial` for the zero-inflated and hurdle count part;
  `log-linear`, `loglinear` and `log_linear` for Mack's and the Munich
  chain ladder's sigma, the paid-incurred chain's variance extrapolation
  and Excel's tail; a grain as `M`, `Q`, `S` or `Y` or as `month`,
  `quarter`, `semiannual` or `year`, in any case; the mixing alias `ig`
  in R; a selected pattern's kind as `percent`, `percent developed`,
  `age-to-age` or `age-to-ultimate` in Python and R. Distortion families
  have one parser, `Family::from_name` in `prospicio-prob`, which all three
  call: short or long names, in any case, with spaces or hyphens. R keeps
  `match.arg()`'s partial matching (`"vol"` for `"volume"`), which Python
  and Excel do not offer; nothing that was accepted before is refused.
