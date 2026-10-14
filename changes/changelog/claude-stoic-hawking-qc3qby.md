### Distributions and probability

- A copula's distribution function, `Copula::cdf`: the Archimedean
  copulas' in any dimension (with `ArchimedeanCopula::inverse_generator`)
  and the Gaussian copula's in two, from a new bivariate normal cdf
  (`prospicio_math::special::bivariate_norm_cdf`, Genz's BVND, about
  `1e-15`); an error for the t copula. Python `cdf` on the copula classes;
  R `copula_cdf()` (#234).

### Aggregate and reinsurance

- Two lines hit by the same events: their joint annual loss by a 2-D
  compound FFT from a joint per-event severity
  (`prospicio_aggregate::bivariate::{fft_bivariate, JointGrid}`, with each
  line's marginal, the total, mixed moments and the correlation), and
  `aggregate`'s `clash` model, given by the expected events hitting line
  A only, line B only and both (`Clash`), matching `aggregate` 1.0.1.
  Python `fft_bivariate`, `JointGrid`, `Clash`; R `fft_bivariate()`,
  `clash_model()`, `clash_severity()`, `clash_aggregate()` (#233).
- Two lines whose losses in an event are coupled by a copula: the joint
  per-event severity as the copula's rectangle probabilities over the two
  marginals (`JointGrid::from_copula`, with `bivariate::triggered` for a
  line an event hits with some probability), for `fft_bivariate`,
  matching `aggregate` 1.0.1's copula `bivariate`. Python
  `JointGrid.from_copula`; R `copula_severity()` (#235).
- Renewal claim counts: the number of claims by a horizon when the waits
  between claims follow any law (`prospicio_aggregate::renewal::renewal_count`,
  a `Wait` that is a distribution or discrete waits, possibly defective,
  with waits of zero arriving in clusters), on `aggregate`'s wait grid or
  a given one, matching `aggregate` 1.0.1's `years ... wait ...`
  frequency. Python `Count.renewal`; R `renewal_count()` (#236).
- Eventual ruin for renewal claims: `ψ(u)` when the waits between claims
  follow any law and the premium comes in at `(1 + ρ)` times the expected
  loss rate, by the cepstral Wiener–Hopf factorization of the claim walk
  (`prospicio_aggregate::ruin::wiener_hopf`, returning `RenewalRuin`),
  matching `aggregate` 1.0.1's `wiener_hopf`. Python `wiener_hopf`; R
  `wiener_hopf()` (#237).
