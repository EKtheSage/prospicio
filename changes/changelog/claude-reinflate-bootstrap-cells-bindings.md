### Reserving

- The bootstraps' reserve draws by future cell in Python and R (#979):
  `OdpBootstrap.fit(..., cells=True)` and `MackBootstrap.fit(..., cells=True)`
  keep them in the fit's `cells`, a `PredictiveDistribution` by the segment
  keys, `origin` and `development` (`None` otherwise); in R,
  `odp_bootstrap(..., cells = TRUE)` and `mack_bootstrap(..., cells = TRUE)`
  and the fit's `cells` (`NULL` otherwise). They sum to the reserves, which
  are unchanged, and `Indexation.reinflate_draws` (`reinflate_draws()` in R)
  re-inflates them.
