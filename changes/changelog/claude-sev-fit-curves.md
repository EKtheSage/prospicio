### Distributions and probability

- The fit-curves table: `fit_curves` sets every fitted family's survival
  conditional on a threshold beside the plotting positions
  `(n - i + 0.5) / n`, the families in AIC order, thinned to numpy's rows;
  `compare_fits` and `rank_by_aic` fit and rank any families, refusing a
  comparison whose likelihoods condition on different thresholds. Python
  `prospicio.distributions.fit_curves` and R `fit_curves()` (#804, #867).

### Excel

- `PRO.SEV.FIT(..., "curves")` spills the same table, with a tenth
  argument `points` (#804, #867).
