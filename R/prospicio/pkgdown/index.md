# prospicio

Actuarial and risk modeling on a Rust core, for R and Python.

- **Reserving**: triangles, Chain Ladder, Mack, Bornhuetter–Ferguson,
  Benktander, Cape Cod, Clark, tails, the ODP bootstrap, the Bayesian models
  and the Merz–Wüthrich one-year view, checked against R ChainLadder and
  chainladder-python.
- **Distributions**: parametric, discretized and sampled distributions,
  predictive distributions, risk measures, distortions, copulas and extreme
  values.
- **Aggregate loss and reinsurance**: Panjer, FFT and Monte Carlo compound
  distributions, layers and towers, reinstatements.
- **Pricing**: layer rating, exposure curves, tower matching and risk-loaded
  pricing.
- **Models**: GLMs, GAMs, elastic nets and Bayesian GLMs with NUTS.

```r
remotes::install_github("EKtheSage/prospicio", subdir = "R/prospicio")
```

The package is pre-alpha: the API will change before v1.0.

The same library for Python and Rust: <https://ekthesage.github.io/prospicio/>.
Source: <https://github.com/EKtheSage/prospicio>.
