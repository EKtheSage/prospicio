# risk-rs

Actuarial and risk modeling on a Rust core, exposed to Python and R.

The plan: [docs/architecture.md](docs/architecture.md). Design notes for the
core abstractions: [docs/design/](docs/design/).

## Layout

| Path | What |
|---|---|
| `crates/act-core` | Error type, reproducible RNG streams |
| `crates/act-math` | Numerical engine (Phase 0: normal special functions) |
| `crates/act-prob` | Distributions (Phase 0: `Distribution` trait, `Lognormal`) |
| `crates/act-reserving` | Reserving (empty until v0.1) |
| `crates/act-aggregate` | Aggregate loss and reinsurance (empty until v0.3) |
| `crates/act-python`, `python/` | Python package `actuarialrs` (PyO3 + maturin) |
| `crates/act-r`, `R/actuarialrs` | R package `actuarialrs` (extendr) |
| `validation/` | Parity harness: reference datasets and results from SciPy and R ChainLadder |
| `src/` | **Provisional** reserving sandbox, see below |

## Build and test

Rust (all pure-Rust crates, the sandbox and the parity suite):

```bash
cargo test
```

Python (needs Python ≥ 3.9):

```bash
cd python
python -m venv .venv && . .venv/bin/activate
pip install maturin pytest scipy
maturin develop
pytest tests
```

R (needs R ≥ 4.2 and Cargo):

```bash
R CMD INSTALL R/actuarialrs
Rscript R/actuarialrs/tests/test-distributions.R
```

The same distribution, driven by the same Rust code, from all three:

```python
import actuarialrs as ar
d = ar.distributions.Lognormal.from_mean_cv(1000, 0.5)
d.quantile(0.995), d.sample(3, seed=42, stream=0)
```

```r
library(actuarialrs)
d <- lognormal_from_mean_cv(1000, 0.5)
quantile(d, 0.995); draws(d, 3, seed = 42, stream = 0)
```

Errors raised in Rust surface as Python `ValueError`s and R errors. In R,
extendr also prints the Rust panic message it uses to carry the error.

## Provisional code

> **Provisional.** Everything in `src/` is a sandbox to experiment with while
> the real architecture is built. It will be replaced by `act-reserving`
> ([docs/design/triangle.md](docs/design/triangle.md) describes the
> replacement); in particular, `Triangle` will be replaced by the four-axis,
> masked, Arrow-backed design. Do not build on these APIs.

- `triangle`: cumulative loss triangles (ragged rows, incremental ↔ cumulative).
- `development`: age-to-age factors (volume-weighted, simple average) and
  age-to-ultimate factors.
- `chain_ladder`: chain-ladder ultimates and reserves, with an optional
  tail, and Mack's (1993) standard errors.

Results are checked against R `ChainLadder` on RAA in `validation/`.
