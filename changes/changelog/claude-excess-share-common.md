### Reserving

- **Breaking (default changes):** the excess-layer BF prior (`LayerSharePrior`,
  Python `LayerSharePrior`, R `layer_share_prior()`) now treats the share's
  error as one error common to all origins (#518, `reserving-excess.md`
  decision 23(a)). `BayesBfNormalised` gets an optional multiplier `s` (Rust
  `common_cv`, Python `common_cv=`, R `common_cv =`, a draw named `log_share`)
  with mean 1 and the share's cv, sampled jointly with the other parameters so
  the data update the share for every origin. The total's prior interval is
  about three times as wide as with independent priors, but its posterior
  interval is narrower on the test layer (the origins pool what they say about
  the share), so layer fits with a share cv change by default. Pass
  `share_dependence="per_origin"` (R `share_dependence = "per_origin"`, Rust
  `ShareDependence::PerOrigin`) for the previous independent priors. A common
  share error needs `apriori_cv > 0`; `LayerSharePrior::model` returns a
  `Result` in Rust. New: `prior_total` (Python `prior_total()`, R
  `layer_prior_total()`), the closed-form mean and sd of the layer's total
  prior ultimate. Models without `common_cv` are unchanged bit for bit.
