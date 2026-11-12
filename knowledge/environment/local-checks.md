---
type: Environment Fact
title: Local checks in the cloud container and their noise
description: What runs locally before a push, what cannot, how to install R and uv when the container lacks them, and the generated-file churn cargo xtask r leaves to revert.
tags: [environment, ci, bindings, r, python]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-06T18:30:00Z }
updated: { by: claude-code/cloud-session, at: 2026-11-12T00:00:00Z }
stale_after: 2027-01-05T00:00:00Z
sources:
  - id: agents
    resource: ../AGENTS.md
    title: AGENTS.md, Checks
---

# What runs

* `cargo fmt --all --check`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test`, and clippy on `prospicio-python` and `prospicio-r`.[^agents]
* `cargo xtask python`: builds the extension and runs pytest; its final
  docs render fails because Quarto is not installed (CI's `docs` job
  covers it).
* `cargo xtask r`: roxygen, install, then every R test file through
  `validation/scripts/run_r_tests.py` (#845, #849), which runs each file
  in its own process, one per core with the Bayesian and boosting files
  alone at the end, keeps going after a failure and ends with each
  failure's log, the skipped optional-package blocks and the slowest
  files; the pkgdown step fails on the network (see
  [cloud network](/environment/cloud-network.md)). The "boom" errors in
  its output are expected test errors. Run some files with
  `python3 validation/scripts/run_r_tests.py <name fragment>`. Until #845
  the parity blocks guarded by actuar, glmnet, statmod, ChainLadder,
  dglm, posterior or loo never ran in CI, which installed none of them;
  the nightly run now installs them.

* Shared test helpers (#845): `R/prospicio/tests/helper.R`, sourced by
  every R test file, holds `with_package()`, `expect_error_like()`,
  `with_warnings()`, `validation()` and the two factories each file takes
  its `near()` from: the 38 files had defined `near()` 19 ways (relative
  or absolute, floors of 1e-300, 1 or 0 under |b|, an absolute floor,
  exact equality allowed, lengths checked), and
  `near_rel(rel, abs, floor, exact, same_length, unname)` and
  `near_abs(tol, same_length, unname)` reproduce each one; a grid of
  147,074 cases (zeros, NaN, Inf, near misses, recycled lengths, names)
  gave the same pass or fail for every file's old and new `near()`.
  `test-option-names.R` keeps its own `expect_error_like()`, which matches
  a regular expression, not fixed text. `python/tests/helpers.py` holds
  `VALIDATION`, `read_csv`, `data_rows`, `dataset`, `ranks`, `pearson` and
  `spearman`; the four files' `close()` keep their own tolerance rules.

# When the container has no R or uv

A fresh container may lack R, uv and the wasm target (2026-10-08). These
installed and ran the R and Python checks:

* `apt-get install -y r-base-core libuv1-dev libxml2-dev
  libcurl4-openssl-dev libssl-dev` (R 4.3; `fs`, a roxygen2 dependency,
  fails to configure without `libuv1-dev`), then
  `install.packages(c("S7", "roxygen2", "pkgload"))` from CRAN.
  `cargo xtask r` then passes up to its pkgdown step, which fails for
  want of pkgdown; CI's docs job covers it.
* `pip install uv`, then `cargo xtask python`'s steps by hand in
  `python/`: `uv sync --group docs`, `uv run maturin develop --uv
  --release --generate-stubs`, `uv run pytest tests`.
* `rustup target add wasm32-unknown-unknown`.

When R is installed but `install.packages()` stops with "lib is not
writable" (2026-10-15), create the user library first:
`mkdir -p ~/R/x86_64-pc-linux-gnu-library/4.3`. CRAN then installs
reference packages such as `ismev`, `evir` and `scam` through the proxy.

That install (R 4.3.3, roxygen2 8.1.0) left none of the noise below:
`cargo xtask r` changed only the files the code change called for.

# Noise to revert after `cargo xtask r`

The container's roxygen2 and R differ from the committed output. Undo
only these:

* the `RoxygenNote: 7.3.1` line it adds to `R/prospicio/DESCRIPTION`
  (keep a real `Collate` or `Suggests` change);
* the link targets it rewrites in `man/gamma_distribution.Rd`,
  `man/lognormal.Rd` and `man/weibull_distribution.Rd`: this R resolves
  `[stats::dgamma()]` to `\link[stats:GammaDist]`, `dlnorm` to
  `stats:Lognormal` and `dweibull` to `stats:Weibull`, where the committed
  files (and CI) have `stats:dgamma`, `stats:dlnorm`, `stats:dweibull`.
  When one of these files changes for real (a new argument), keep that
  change and restore only the link; reverting the whole file loses it;
* `NAMESPACE`'s multi-line `importFrom(stats, aggregate, coef, predict,
  quantile)`, which it splits into four lines.

Keep the other regenerated `man/` and `NAMESPACE` changes: they come from
the code change.

# Also

* `R/prospicio/R/extendr-wrappers.R` is generated, not edited by hand
  (#836, 2026-11-10): `cargo xtask r` and `cargo xtask generated` install
  the package, call `wrap__make_prospicio_wrappers` and write the file
  with the doc lines dropped and the free functions and classes sorted by
  name, so Windows' other function order no longer shows; CI's `docs` job
  fails when the committed file is stale. Without the full docs toolchain,
  `cargo xtask generated` needs only uv, R with roxygen2, pkgload and
  pkgdown. The first regeneration found one drift from the hand-edited
  file, `ChainLadderFit$tail <-function()` missing a space. In the cloud
  container `apt-get install r-base-core r-base-dev` gives R 4.3.3 and
  `install.packages("S7")` the package's one dependency.
* maturin 1.16.0's `--generate-stubs` on Windows (a uv venv, 2026-11-02)
  writes a stub unlike the committed one: an `__all__` line, blank
  docstring lines without their indentation and a newline escape in a
  docstring doubled. Keep the committed stub and add only the new classes' hunks
  (`git diff -w` shows them), with blank docstring lines indented like the
  rest, and write an optional type as `X |None` (no space after the
  bar), as the generator on CI writes it everywhere in the stub;
  `X | None` fails CI's docs job, which checks the result.
* `cargo clippy -p prospicio-r` panics in extendr-api's build script
  (`DEP_R_R_VERSION_MAJOR`: `NotPresent`) when the target directory
  holds an `extendr-ffi` build from a run without R: setting `R_HOME`
  afterwards does not rerun it. `cargo clean -p extendr-ffi`, then the
  clippy run with `R_HOME` set and R's `bin/x64` on the path, fixes it
  (Windows, 2026-10-13).
* A topic that a new roxygen block adds and other blocks link to makes
  `cargo xtask r` print "Could not resolve link to topic" for those
  links on the run that creates it; the links in the written Rd files
  are correct.
* The freMTPL2 parity runs only in release builds
  (`cargo test --release -p prospicio-validation --test fremtpl2`).

[^agents]: AGENTS.md, Checks
