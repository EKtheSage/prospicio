### Models

- `prospicio_bayes::elpd::psis(log_ratios, r_eff)`: Pareto smoothed
  importance sampling of any log importance ratios, public. It returns the
  smoothed log weights (unnormalized, the largest raw ratio shifted to 0)
  and the Pareto k, by the smoothing `loo` applies, with the tail length
  `min(ceil(0.2 S), ceil(3 sqrt(S / r_eff)))`; ratios that are all -inf
  (every weight zero) are an error. `loo` and `waic` are
  unchanged (#273).
