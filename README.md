# risk-rs

Actuarial and risk modeling in Rust.

The plan: [docs/architecture.md](docs/architecture.md).

## Build

```bash
cargo build
cargo test
```

## Provisional code

> **Provisional.** Everything in `src/` is a sandbox to experiment with while
> the real architecture is built. It will be replaced by the `act-*` crates
> described in [docs/architecture.md](docs/architecture.md); in particular,
> `Triangle` will be replaced by the four-axis, masked, Arrow-backed design.
> Do not build on these APIs.

- `triangle` — cumulative loss triangles (ragged rows, incremental ↔ cumulative).
- `development` — age-to-age factors (volume-weighted, simple average) and
  age-to-ultimate factors.
- `chain_ladder` — chain-ladder ultimates and reserves, with an optional
  tail, and Mack's (1993) standard errors.

Results are checked against the RAA triangle as published in R's
`ChainLadder` package.

The roadmap lives in [docs/architecture.md](docs/architecture.md).
