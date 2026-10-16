# Environment

* [Cloud session disk space](cloud-disk.md) - The writable disk is a fixed allowance that a full build and test fills (target/debug reached 25 GB); deleting target/debug/incremental frees about 9 GB.
* [Cloud network](cloud-network.md) - Which hosts the cloud environment reaches, which need allowlisting, and the OpenML host quirk.
* [Installing the R package from GitHub](r-install-from-github.md) - remotes::install_github(subdir=) builds a tarball of R/prospicio alone, so the Rust crates must be copied into it first; pkgbuild's Config/build/bootstrap runs bootstrap.R to do that.
* [Local checks](local-checks.md) - What runs locally before a push, what cannot, how to install R and uv when the container lacks them, and the generated-file churn cargo xtask r leaves to revert.
* [Local Windows R](local-windows-r.md) - How to build the R package, run R tests and regenerate binding docs locally on Windows, with the system R that is installed but not on PATH.
* [Bayesian tests run in release](slow-tests-in-release.md) - The NUTS parity tests sit behind the slow-bayes feature of prospicio-validation and prospicio-reserving's NUTS unit tests behind its slow-tests feature, so a debug cargo test skips them; CI's bayes job runs both in release; how to run them locally and add one.
* [Notebooks without outputs, executed on the docs site](notebooks-on-site.md) - Notebooks in examples/ are committed without outputs under an nbstripout git filter; cargo xtask python stages the guides and the executed notebooks as the great-docs User Guide; the filter's git status quirk and great-docs 0.17's user guide limits.
* [Publishing](publishing.md) - How the crates and the Python package package for PyPI and crates.io, and what trusted publishing on each needs.
* [Signing the Excel XLL](xll-signing.md) - Authenticode signing with osslsigncode on Linux, checked with a self-signed certificate, and which certificates fit a CI secret.
* [Building and testing the Excel XLL](xll-builds.md) - Cross-building the 64-bit XLL from Linux with MinGW, its system-only imports, and the fake-Excel host under Wine that drives the C API.
* [prospicio-nn in the Python bindings](nn-in-bindings.md) - The wheel links Burn through the default nn feature: about 8 MB more native module (18.3 MB to 26.5 MB, release), Rust 1.92 to build (the bindings' rust-version stays 1.85 for clippy), and the fitted networks are Send and Sync, so frozen pyclasses need no lock.
* [WebAssembly builds](wasm-builds.md) - Which crates build for wasm32, how to run their tests under WASI with Node, and how Rayon behaves without threads.
* [CmdStan on the local Windows machine](local-cmdstan.md) - CmdStan 2.39.0 at ~/.cmdstan builds models with RTools 4.5's make and g++ from PowerShell or cmd only; cmdstanpy 1.3.0 needs MAKE set, since it looks for mingw32-make.
