### Models

- `BayesGlmFit` saves to and loads from a versioned JSON artifact
  (`to_json`, `from_json`) in Rust and Python, with every posterior draw,
  so a loaded fit summarizes and predicts as the original; it pickles in
  Python (#978).
