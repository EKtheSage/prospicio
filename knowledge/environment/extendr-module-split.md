---
type: Environment Fact
title: Splitting an extendr binding file into modules
description: extendr 0.9's #[extendr] generates a private meta__<item> function that extendr_module! calls by name, so a split file registers its own items in an extendr_module! of its own and the parent module uses it; cargo xtask sorts extendr-wrappers.R by name and extendr skips the hidden metadata functions, so the nesting leaves the wrappers, NAMESPACE and man/ unchanged.
tags: [environment, r, bindings, extendr, refactor]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-24T20:00:00Z }
stale_after: 2027-05-24T00:00:00Z
sources:
  - id: split
    resource: process:local session claude/split-r-reserving-963, 2026-11-24
    title: cargo xtask generated --check and run_r_tests.py reserving before and after splitting crates/prospicio-r/src/reserving.rs (#963)
  - id: macros
    resource: https://github.com/extendr/extendr/blob/extendr-macros-v0.9.0/extendr-macros/src/extendr_module.rs
    title: extendr-macros 0.9.0, extendr_module.rs and wrappers.rs
---

# Facts (2026-11-24)

* `#[extendr]` on a function or an impl generates a public `#[no_mangle]`
  `wrap__*` C wrapper and a private `fn meta__<name>`, and
  `extendr_module! { mod m; fn f; impl T; }` calls `meta__f` and `meta__T`
  by bare name. A private function cannot be re-exported, so a glob
  re-export of a submodule, as the Python bindings' split used, does not
  reach them: the registration has to sit in the module that holds the
  item.
* A split file therefore ends with its own
  `extendr_module! { mod <file>; fn ...; impl ...; }`, and the parent's
  module lists `use <file>;` for it, the nesting `lib.rs` already uses for
  `reserving`, `reserving_counts` and the other modules. The parent's
  metadata appends each used module's functions and impls after its own.
* The order that nesting gives does not reach the generated R files: `cargo
  xtask r` and `cargo xtask generated` sort the free functions and classes
  of `R/prospicio/R/extendr-wrappers.R` by name, and extendr's
  `make_r_wrappers` skips the `hidden` metadata functions
  (`get_<mod>_metadata`, `make_<mod>_wrappers`) that each nested module
  adds. Splitting the R bindings' 8,265-line `reserving.rs` into 23 nested
  modules left `extendr-wrappers.R`, `NAMESPACE` and `man/` unchanged, and
  the 36 R reserving test files passed.
* Each nested module also exports `#[no_mangle]` symbols named after it
  (`R_init_<mod>_extendr`, `wrap__get_<mod>_metadata`,
  `wrap__make_<mod>_wrappers`), so two `extendr_module!`s of the same name
  anywhere in the crate would clash at link time; `cargo check` and clippy
  do not link, so a clash shows only in `R CMD INSTALL`. The reserving
  split's file names (`triangle`, `mack`, `curves`, ...) are unique in the
  crate.
* `prospicio-r` builds locally on Windows only with R visible: with R
  installed but off `PATH`, extendr-api's build script panics with
  `NotPresent` until `R_HOME` and `PATH` point at R and `cargo clean -p
  extendr-ffi` drops its cached failed probe
  ([Local Windows R](/environment/local-windows-r.md)). Without Pandoc,
  `cargo xtask generated --check` stops at its last step, pkgdown's
  reference index, after the wrappers, `NAMESPACE` and `man/` have been
  regenerated, so `git status` still shows whether they changed.
