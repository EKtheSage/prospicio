### Distributions and probability

- The KS p-value of a severity fit by the parametric bootstrap
  (`severity_fit::ks_bootstrap`, Python and R `ks_bootstrap`): samples
  drawn from the fit under the data's reporting thresholds, bounds from
  above and policy limits, each refitted, so the p-value accounts for the
  fitted parameters, which `goodness_of_fit`'s classical `ks_p_value` does
  not. Reproducible by seed on any number of threads.
