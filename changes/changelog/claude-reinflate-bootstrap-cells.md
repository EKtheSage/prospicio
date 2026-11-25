### Reserving

- The ODP and Mack bootstraps can keep their reserve draws by future cell
  (#979): `fit_with_cells` and `fit_segments_with_cells` on `OdpBootstrap`
  and `MackBootstrap` return the usual fit with `cells` set, the simulated
  future increments by `origin` and `development` (after the segment keys),
  which sum to the reserves and which `Indexation::reinflate_draws` can
  re-inflate. The reserves are unchanged; `fit` and `fit_segments` leave
  `cells` empty.
