### Models

- MCMC draws from outside prospicio in its Bayes diagnostics and PSIS-LOO
  (#395): `PosteriorDraws::from_chains`, `log_likelihood`,
  `relative_efficiency` and `loo`, and `prospicio_bayes::stan_csv::read`
  for CmdStan's CSV files (warmup dropped, sampler columns kept apart,
  divergences counted). Python `Draws`, `read_stan_csv`,
  `draws_from_arrays` and `draws_from_inference_data` (arviz, nutpie and
  PyMC), R `read_stan_csv()`, `mcmc_draws()` (a cmdstanr or rstan fit, a
  posterior draws object or an array), `draws_summary()` and
  `draws_loo()`. R-hat, ESS and LOO match posterior 1.7.0 and loo 2.10.1
  on a CmdStan run.
