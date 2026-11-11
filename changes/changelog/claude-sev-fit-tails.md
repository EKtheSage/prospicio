### Distributions and probability

- The classical two-sided Kolmogorov-Smirnov p-value
  (`prospicio_math::kolmogorov::kolmogorov_sf`, matching SciPy's `kstwo`)
  on every goodness of fit (`GoodnessOfFit::ks_p_value`), and the tail
  mean and CV of a severity above a threshold (`tail_moments`); Python and
  R `goodness_of_fit` return `ks_p_value`, `tail_mean` and `tail_cv` (#805).

### Breaking changes

- Excel: `PRO.SEV.FIT(..., "compare")` has three more columns (KS p, Tail
  mean, Tail CV) before Converged and Note, and `"params"` three more rows
  before Converged, so a formula that picked Converged or Note by position
  moves (#805).
