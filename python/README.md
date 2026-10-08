# prospicio

Actuarial and risk modeling on a Rust core, for Python and R.

- **Reserving**: triangles, Chain Ladder, Mack, Bornhuetter–Ferguson,
  Benktander, Cape Cod, Clark, tails, the ODP bootstrap and the
  Merz–Wüthrich one-year view, checked against R ChainLadder and
  chainladder-python.
- **Distributions**: parametric, discretized and sampled distributions,
  `PredictiveDistribution`, risk measures, distortions, copulas and
  extreme values.
- **Aggregate loss and reinsurance**: Panjer, FFT and Monte Carlo
  compound distributions, layers and towers, reinstatements.
- **Pricing**: layer rating, exposure curves, tower matching and
  risk-loaded pricing.
- **Models**: GLM, GAM, elastic net and Tweedie (statsmodels parity),
  Bayesian GLMs with NUTS, and adapters over LightGBM and XGBoost.

```
pip install prospicio
```

The package is pre-alpha: the API will change before v1.0.

Documentation: <https://ekthesage.github.io/prospicio/>.
Source, and the same library for R and Rust:
<https://github.com/EKtheSage/prospicio>.

Licensed under either of MIT or Apache-2.0, at your option.
