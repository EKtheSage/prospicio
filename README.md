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

### Prerequisites

| | macOS / Linux | Windows (PowerShell) |
|---|---|---|
| Rust ≥ 1.85 | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` | `winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"` then `winget install Rustlang.Rustup` |
| uv (Python) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `winget install astral-sh.uv` |
| R ≥ 4.2 | from CRAN | R from CRAN, **Rtools** matching your R version, then `rustup target add x86_64-pc-windows-gnu` |
| Quarto (docs only) | from [quarto.org](https://quarto.org/docs/get-started/) | `winget install Posit.Quarto` |

Open a new terminal after installing so `cargo`, `uv` and `R` are on `PATH`.

### Rust

All pure-Rust crates, the sandbox and the parity suite:

```bash
cargo test
```

### Python (uv)

```bash
cd python
uv sync              # creates .venv, installs dev deps, builds the Rust extension
uv run pytest tests
uv run python        # a Python shell with actuarialrs installed
```

`uv run` rebuilds the extension automatically when Rust sources change.

### R

From the repository root (a terminal where `cargo --version` works). The
package's R API is built on [S7](https://rconsortium.github.io/S7/), so
install that first:

```bash
Rscript -e 'install.packages("S7")'
R CMD INSTALL R/actuarialrs
Rscript R/actuarialrs/tests/test-distributions.R
```

**Windows PowerShell:** `R` is PowerShell's alias for `Invoke-History`, so
call `R.exe CMD INSTALL R\actuarialrs` instead. To put R on `PATH` for your
user (newest installed R; rerun after upgrading R), then open a new window:

```powershell
$rbin = (Get-ChildItem "C:\Program Files\R" -Directory | Sort-Object Name | Select-Object -Last 1).FullName + "\bin"
[Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", "User") + ";$rbin", "User")
```

On Windows, R builds packages with Rtools' MinGW compiler, so the Rust code
is compiled for the `x86_64-pc-windows-gnu` target (see
`R/actuarialrs/src/Makevars.win`); the build stops with a message if that
target is not installed.

### Docs

Building a binding regenerates its docs (see "Documentation" in
[docs/architecture.md](docs/architecture.md)). Use these instead of the
plain builds above whenever you change a doc comment or a public API:

```bash
Rscript -e 'install.packages(c("S7", "roxygen2", "pkgdown"))'  # once
cargo xtask python   # build + stubs, test, great-docs site in python/great-docs/_site
cargo xtask r        # install, roxygen2 man/ + NAMESPACE, test, pkgdown site in R/actuarialrs/docs
cargo xtask docs     # both plus rustdoc, collected into target/docs-site
```

Commit the regenerated `python/actuarialrs/actuarialrs_native.pyi`,
`R/actuarialrs/man/` and `R/actuarialrs/NAMESPACE`; never edit them by hand.
`cargo xtask docs --check` fails if they are stale.

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
d@meanlog; d@sdlog   # read-only S7 properties
```

Errors raised in Rust surface as Python `ValueError`s and ordinary R errors.

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
