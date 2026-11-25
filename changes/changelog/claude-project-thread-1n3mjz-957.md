### Bindings

- R: every native function whose Rust signature returns a `Result` now
  raises its error in R from the generated wrapper, so no call site can
  return an `extendr_error` object as a value; `ale()` with a bins count
  the core rejects, and `shap()` when the exact enumeration fails, now
  stop with the core's message (#957). An error from a reserving binding
  keeps the condition class `prospicio_not_supported` whichever R
  function raises it.
