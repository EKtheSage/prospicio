### Guides

- `docs/guides/pir-risk-measures.qmd` and `docs/guides/pir-portfolio-pricing.qmd`,
  the R twins of the guides to Parts I and II of Mildenhall and Major's
  *Pricing Insurance Risk* (risk measures; portfolio pricing): every number
  equal to the Python guides', run by `R/prospicio/tests/test-guides.R`. The R
  gaps found are `VaR()` on grids (#373) and `VaR()` and `TVaR()` on
  parametric severities (#375).
