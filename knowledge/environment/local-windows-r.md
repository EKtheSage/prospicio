---
type: Environment Fact
title: R bindings and docs on the local Windows machine
description: How to build the R package, run R tests and regenerate binding docs locally on Windows, with the system R that is installed but not on PATH.
tags: [environment, bindings, r, python, windows]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-06T00:00:00Z }
stale_after: 2027-01-06T00:00:00Z
sources:
  - id: setup
    resource: process:local-session setup agent, reserving v0.2 (PRs #131 to #136)
    title: Local R build check, 2026-10-05
---

# Facts (2026-10-05)

* R 4.5.3 is installed at `C:/Program Files/R/R-4.5.3` but is not on PATH.
  Rtools45 is at `C:/rtools45`, and R finds it on its own. The user library
  already has S7, roxygen2, pkgload, pkgdown and ChainLadder. The Rust target
  `x86_64-pc-windows-gnu` is installed.[^setup]
* From a worktree root, in Git Bash:

  ```bash
  export R_HOME="C:/Program Files/R/R-4.5.3" PATH="/c/Program Files/R/R-4.5.3/bin:$PATH" R_LIBS="<worktree>/target/rlib" PYTHONUTF8=1
  mkdir -p target/rlib
  cargo clean -p extendr-ffi   # once, if act-r was ever built without R visible
  cargo xtask docs --check     # over 10 minutes; run it in the background
  ```

* Without `cargo clean -p extendr-ffi`, extendr-api's build script panics
  with `NotPresent`: extendr-ffi caches its failed probe for R and does not
  rerun when `R_HOME` changes.
* `R_LIBS` per worktree keeps parallel sessions from overwriting each
  other's installed `prospicio` in the shared user library.
* `PYTHONUTF8=1` is needed: without it great-docs fails at "Prepare freeze
  cache" with `'utf-8' codec can't decode byte 0x96`.
* `cargo xtask r` and `cargo xtask python` (without `--check`) regenerate
  `man/`, `NAMESPACE` and the `.pyi` stub. A full run leaves no tracked-file
  churn.
* A conda-forge R from pixi has no matching toolchain and cannot build the
  package; it can still run ChainLadder parity scripts.

[^setup]: Local R build check
