---
type: Environment Fact
title: Installing the R package from GitHub, and why it needs bootstrap.R
description: remotes::install_github(subdir=) builds a tarball of R/prospicio alone, so the Rust crates must be copied into it first; pkgbuild's Config/build/bootstrap runs bootstrap.R to do that.
tags: [environment, r, bindings, release]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-12T02:00:00Z }
stale_after: 2027-04-12T00:00:00Z
sources:
  - id: pkgbuild
    resource: https://cran.r-project.org/web/packages/pkgbuild/refman/pkgbuild.html
    title: pkgbuild reference manual, build() and Config/build/bootstrap
  - id: makevars
    resource: ../../R/prospicio/src/Makevars
    title: R/prospicio/src/Makevars
---

# What install_github does

`remotes::install_github("EKtheSage/prospicio", subdir = "R/prospicio")`
downloads the whole repository, then builds a source tarball of
`R/prospicio` with `pkgbuild::build` and installs that tarball from a
temporary directory. The tarball holds only the package directory, so a
Makevars that builds from `../../../crates` (as `R CMD INSTALL R/prospicio`
in a checkout can) fails with "manifest path ... does not exist". Reproduced
on 2026-10-12 with `remotes::install_local(<repo tarball>, subdir =
"R/prospicio")`, which takes the same path after the download.

# The fix

`Config/build/bootstrap: TRUE` in DESCRIPTION makes pkgbuild run
`Rscript bootstrap.R` in the package's source directory before
`R CMD build`.[^pkgbuild] There the checkout is still around it, so
`R/prospicio/bootstrap.R` copies into `src/rust/`:

* `prospicio-r` and the workspace crates it depends on, found by reading
  each crate's `[dependencies]` for `prospicio-*.workspace = true` (ten
  crates in October 2026; `prospicio-nn` and `prospicio-python` are left out);
* the root `Cargo.toml` with `members` cut to `crates/prospicio-r`, so the
  crates still inherit `[workspace.package]` and `[workspace.dependencies]`;
* the root `Cargo.lock`, so third-party versions match the workspace.

`src/Makevars` and `Makevars.win` build from `src/rust/` when its
`Cargo.toml` exists and from the workspace otherwise.[^makevars] Third-party
crates still download from crates.io during the install; a CRAN build must
vendor them for offline builds.

A plain `R CMD build R/prospicio` does not run the bootstrap, so a tarball
made that way still cannot build on its own.

Not checked: whether r-universe runs the bootstrap. It needs a registry
repository (`<user>.r-universe.dev`) that the owner sets up, and had not been
tried as of 2026-10-12.

[^pkgbuild]: [pkgbuild reference manual, build() and Config/build/bootstrap](https://cran.r-project.org/web/packages/pkgbuild/refman/pkgbuild.html)
[^makevars]: [R/prospicio/src/Makevars](../../R/prospicio/src/Makevars)
