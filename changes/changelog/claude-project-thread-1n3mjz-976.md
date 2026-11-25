### Models

- `prospicio_glm::wald` (`p_value`, `critical_value`, `interval`) and
  `prospicio_glm::quantile::rows_by_tau`, the normal Wald inference and the
  non-crossing rearrangement that the Excel add-in used to compute itself
  (#976, #1095).

### Distributions and probability

- `prospicio_prob::statistics::sample_moments`: a sample's mean, SD, CV and
  skewness on the population moments (#976, #1095).

### Aggregate and reinsurance

- `CompoundMoments::covering_points`: the FFT points that cover the mean
  plus 12 standard deviations and a reach, as the Excel add-in sizes its
  grids (#976, #1095).

### Pricing

- `prospicio_pricing::payout::discount`, the present value of 1 paid at a
  time (#976, #1095).
