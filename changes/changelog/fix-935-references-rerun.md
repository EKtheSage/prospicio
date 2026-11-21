### Examples

- The weekly references rerun no longer fails on `sobol_scipy.csv` (only its SciPy
  version label moved, now listed as a moved source) or on `hawkes_scipy.csv`
  (an optimiser's last digits: the fitted parameters are now compared to 1e-6
  relative, where the Rust test allows 1e-4, and the log-likelihood to 1e-7
  absolute, inside the Rust test's 1e-6) (#935, #1017).
