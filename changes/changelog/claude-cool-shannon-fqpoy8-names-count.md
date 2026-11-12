### Bindings

- The zero-inflated and hurdle count part's names come from one table,
  `CountDistribution::NAMES` with `from_name` in `prospicio-glm`, which
  Python, R and Excel read; an unknown name's error reads the same in all
  three, `count must be "poisson", "negative_binomial" or "negbin", got
  "binomial"` (R's argument is `dist`) (#834).
