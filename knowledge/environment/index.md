# Environment

* [Cloud network](cloud-network.md) - Which hosts the cloud environment reaches, which need allowlisting, and the OpenML host quirk.
* [Installing the R package from GitHub](r-install-from-github.md) - remotes::install_github(subdir=) builds a tarball of R/prospicio alone, so the Rust crates must be copied into it first; pkgbuild's Config/build/bootstrap runs bootstrap.R to do that.
* [Local checks](local-checks.md) - What runs locally before a push, what cannot, how to install R and uv when the container lacks them, and the generated-file churn cargo xtask r leaves to revert.
* [Local Windows R](local-windows-r.md) - How to build the R package, run R tests and regenerate binding docs locally on Windows, with the system R that is installed but not on PATH.
* [Publishing](publishing.md) - How the crates and the Python package package for PyPI and crates.io, and what trusted publishing on each needs.
* [WebAssembly builds](wasm-builds.md) - Which crates build for wasm32, how to run their tests under WASI with Node, and how Rayon behaves without threads.
* [CmdStan on the local Windows machine](local-cmdstan.md) - CmdStan 2.39.0 at ~/.cmdstan builds models with RTools 4.5's make and g++ from PowerShell or cmd only; cmdstanpy 1.3.0 needs MAKE set, since it looks for mingw32-make.
