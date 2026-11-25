### Distributions and probability

- `Stable` implements `Distribution` (#977): `cdf`, `survival` and
  `quantile` by Nolan's integral representation, each tail computed
  directly so that far tails keep their relative precision, checked against
  SciPy's `levy_stable` and the Lévy, Cauchy and normal closed forms.
  `prospicio_math::integrate::adaptive_gauss_legendre` is new.
- The `prospicio` crate's documentation now describes its `io` feature
  (triangles from CSV and Excel files).
