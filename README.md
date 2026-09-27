# risk-rs

Actuarial and risk modeling in Rust.

## Build

```bash
cargo build
cargo test
```

## Modules

- `triangle` — cumulative loss triangles (ragged rows, incremental ↔ cumulative).
- `development` — age-to-age factors (volume-weighted, simple average) and
  age-to-ultimate factors.
- `chain_ladder` — chain-ladder ultimates and reserves, with an optional
  tail, and Mack's (1993) standard errors.

All results are checked against the RAA triangle as published in R's
`ChainLadder` package.

## Roadmap

1. **Reserving** — factor exclusions and n-year averages, tail fitting
   (exponential / inverse power), Bornhuetter-Ferguson, Cape Cod, bootstrap ODP.
2. **Frequency and severity** — Poisson / negative binomial, lognormal /
   gamma / Pareto fitting, limited expected values, increased limits factors.
3. **Aggregate loss** — Panjer recursion, FFT, Monte Carlo compound
   distributions.
4. **Risk measures** — VaR, TVaR, and capital allocation.
5. **Life contingencies** — mortality tables, annuities, and insurance
   present values.
