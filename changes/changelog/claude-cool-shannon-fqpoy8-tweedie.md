### Distributions and probability

- Tweedie parameter conversions, as `aggregate` 2.1.0's
  `TweedieConversion`: a Tweedie from its power, mean and coefficient of
  variation (`Tweedie::from_cv`), from its mean, coefficient of variation
  and dispersion (`from_cv_dispersion`, the power then follows) and from a
  Poisson count with a gamma severity's mean and coefficient of variation
  (`from_compound`), with the accessors `cv`, `severity_mean`,
  `severity_cv` and `p0`; Python `Tweedie.from_cv`, `.cv` and the others,
  R `tweedie_from_cv()`, `tweedie_from_cv_dispersion()`,
  `tweedie_from_compound()` and `@cv` (#775).
