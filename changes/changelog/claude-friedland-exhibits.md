### Breaking changes

- Rust: `prospicio_reserving::Error` has the variant `Exhibit`, so an
  exhaustive `match` on it needs an arm for it (#557, #735).

### Reserving

- Friedland's exhibits and a weighted selection of ultimates
  (`docs/design/reserving-v02.md`, decision 34): `DevelopmentExhibit` (link
  ratios, Friedland's six averages, selected factors, factors to ultimate
  and percents developed), `UltimateExhibit` (each method's ultimate, IBNR
  and unpaid beside the claims to date) and `UltimateExhibit::select` with
  `MethodWeights` by origin, by age or one method, each origin's weights
  summing to 1; Friedland's chapter 7 and 15 exhibits to the printed digits
  and chainladder-python's `VotingChainladder` on GenIns. Python
  `DevelopmentExhibit`, `UltimateExhibit`, `SelectedUltimates` and
  `select_ultimates`; R `development_exhibit()`, `ultimate_exhibit()` and
  `select_ultimates()` (#557, #735).
