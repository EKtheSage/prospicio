### Models

- Posterior draws (`PosteriorDraws`, Python's `Draws`) and `BayesGlmmFit`
  save to and load from versioned JSON artifacts (`to_json`, `from_json`)
  in Rust and Python, and pickle in Python, `Draws` with its sampler
  columns (#978).
