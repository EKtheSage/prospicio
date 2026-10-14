### Reserving

- `Compartmental` fits Gesmann and Morris' model 2: the log-normal
  likelihood (`CompartmentalLikelihood::LogNormal`), loss ratios to
  premium (`Scale::LossRatio`) and incremental paid
  (`PaidResponse::Incremental`, each increment the closed form at its age
  less the same at the age before, with the cell's own parameters) are
  built, in any combination with the normal likelihood, amounts and
  cumulative paid. The hierarchy is in the priors: `CompartmentalPriors`
  gains `ker_effect_sd`, `kp_effect_sd` (origin effects on the rates) and
  `development_effect_sd` (development-year effects on any of the four
  parameters), all `None` in `gesmann_morris()`, and
  `CompartmentalPriors::gesmann_morris_2()` is the paper's model 2 preset.
  A development age with no observed cell takes a new effect from the
  population in each simulation, as an origin with none does.
  `CompartmentalFit::cell_rate_swaps` reports the share of draws and cells
  with `k_er <= k_p` when the rates vary by cell. At parity with the
  paper's model 2 code reconstructed in Stan and its printed posterior on
  company 337. `CompartmentalPriors` has three new public fields
  (**breaking** for code that builds it in full; `..CompartmentalPriors::gesmann_morris()`
  keeps model 1's hierarchy).
- Python: `CompartmentalPriors` takes `ker_effect_sd`, `kp_effect_sd` and
  `development_effect_sd` (all `None` by default, model 1's hierarchy),
  with `CompartmentalPriors.gesmann_morris_2()` and, in `replace`, `False`
  to remove an effect; `Compartmental.model_2(priors, ...)` sets model 2's
  likelihood, scale, paid response and sampler; the log-normal, loss
  ratios and incremental paid no longer raise "not yet built";
  `CompartmentalFit.cell_rate_swaps`.
- R: `compartmental_priors()` takes `ker_effect_sd`, `kp_effect_sd` and
  `development_effect_sd` (a named list over `k_er`, `k_p`, `rlr` and
  `rrf`), with `gesmann_morris_2_priors()`; `compartmental_model_2()`;
  the fit's `cell_rate_swaps`; `compartmental_prior_predictive()` takes
  `paid_response`; the log-normal, loss ratios and incremental paid no
  longer stop with "not yet built".
