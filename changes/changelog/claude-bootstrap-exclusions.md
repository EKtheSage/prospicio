### Reserving

- The ODP bootstrap, Mack's bootstrap and their one-year views follow
  `n_periods`, `drop_high` and `drop_low`. `OdpBootstrap` gets the three
  settings (volume-weighted chain ladder only): the fitted values are
  back-cast with the factors under the exclusions, an incremental cell has a
  residual only if the link ratio it closes is kept, and the pseudo factors
  average the kept links only. The scale follows chainladder-python's rule
  (the kept squared residuals over `N - p`), so it is biased low under
  exclusions (RAA, `n_periods = 3`: 694 against 984). Mack's bootstrap uses
  the settings it already took: its standard deviation under them is Mack's
  standard error under the same settings. The drops stay fixed on the
  observed triangle in the simulations. A one-year view takes the
  re-reserving method's exclusions; a bootstrap that sets different ones is
  an error naming both, and the mean of the claims development result is
  not zero under the exclusions. Merz and Wuthrich's closed form still
  rejects them. The geometric average stays rejected. In Python
  (`OdpBootstrap(n_periods, drop_high, drop_low)`) and R (`odp_bootstrap()`;
  `odp_one_year()`, `mack_bootstrap()` and `mack_one_year()` no longer
  raise) (#401). **Breaking (Rust):** `OdpBootstrap` has three new public
  fields, so struct literals need `..Default::default()`; `Error` has a new
  variant, `ExclusionsMismatch`.
