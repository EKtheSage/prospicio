### Examples

- The weekly references rerun no longer fails on `sobol_scipy.csv` (only its SciPy
  version label moved, now listed as a moved source) or on `hawkes_scipy.csv`
  (an optimiser's last digits, now compared to 1e-6 relative, as the Rust test
  already allows 1e-4).
