### Models

- `GlmmFit` saves to and loads from a versioned JSON artifact (`to_json`,
  `from_json`) in Rust and Python, random effects included, so a loaded
  fit predicts with the same groups; it pickles in Python (#978).
