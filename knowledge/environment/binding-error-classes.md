---
type: Environment Fact
title: Error classes across the Python and R bindings
description: A Rust error reaches Python as an exception class of its own through pyo3's create_exception and pymodule_export, but reaches R only as a message, since extendr 0.9's result_condition error carries a string; the reserving wrappers mark a not-supported error with a message prefix that an R helper turns into the condition class prospicio_not_supported.
tags: [environment, bindings, python, r, pyo3, extendr, errors]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-02T00:00:00Z }
stale_after: 2027-05-02T00:00:00Z
sources:
  - id: build
    resource: process:local-session, issue #580 (branch claude/unify-not-supported)
    title: Building the bindings with one not-supported class, 2026-11-02
  - id: raise
    resource: process:cloud-session, issue #957 (branch claude/project-thread-1n3mjz-957)
    title: The generated R wrappers raising Rust errors, 2026-11-25
  - id: note
    resource: ../docs/design/reserving-v02.md
    title: reserving-v02.md decision 29, items 6 and 7
---

# Facts (2026-11-02)

* pyo3 0.29: `pyo3::create_exception!(prospicio, NotSupportedError,
  PyValueError, "doc")` in a lane's module file defines a subclass of
  `ValueError` whose `__module__` is `prospicio`, and
  `#[pymodule_export] use super::reserving::NotSupportedError;` in the
  declarative `prospicio_native` module adds it like a class. The package
  re-exports it from `prospicio/__init__.py` and `prospicio/reserving.py`.
  `maturin develop --generate-stubs` (maturin 1.15.0, the version
  `python/uv.lock` pins; 1.16.0 rewrites the whole stub's whitespace) does
  not list the exception in `prospicio_native.pyi`; it adds `from _typeshed
  import Incomplete` and a module-level `def __getattr__(name: str) ->
  Incomplete: ...` instead. [^build]
* extendr 0.9 with the `result_condition` feature returns an error to R as
  an object of class `extendr_error` whose `value` is the message;
  `extendr_api::Error::Other(String)` carries nothing else, so a Rust error
  cannot choose an R condition class. `rust_result()` in
  `R/prospicio/R/distributions.R` raises every such object as a plain
  `errorCondition`. [^build]
* The reserving wrappers therefore prefix the message of a not-supported
  error with `[prospicio_not_supported] ` (`NOT_SUPPORTED_PREFIX` in
  `crates/prospicio-r/src/reserving/common.rs`). Since #957
  `rust_result()` itself removes the prefix and signals
  `errorCondition(msg, class = "prospicio_not_supported")`, so the class
  survives whichever lane's call raises it; `reserving_result()` in
  `R/prospicio/R/reserving.R` is now only `rust_result()`. [^note] [^raise]
* Since #957 the generated `R/prospicio/R/extendr-wrappers.R` passes the
  `.Call` of every `#[extendr]` function or method whose Rust signature
  returns a `Result` through `rust_native_result()`, so an `extendr_error`
  never comes back as a value. `cargo xtask` finds them by reading the
  signatures in `crates/prospicio-r/src` (rustfmt's layout: `#[extendr]`
  at column 0, methods at column 4); extendr's own generated wrappers do
  not say which functions are fallible. `rust_native_result()` leaves the
  error to the caller when the wrapper runs inside a `rust_result()` or
  `reserving_result()` call (some frame between the wrapper and its caller
  is one), because those calls choose the error's `call`, which the R tests
  check with `conditionCall()`; otherwise it raises in the name of the
  wrapper's caller, `sys.call(sys.parent())` from the wrapper. `sys.call(-1)`
  would name `rust_result()` itself when the wrapper is a promise that
  `rust_result()` forces, since `-1` counts frames on the stack, not
  parents. [^raise]
* `R CMD INSTALL` and `cargo check -p prospicio-r` need R visible to
  extendr-ffi's build script; after a check without it, `cargo clean -p
  extendr-ffi` is needed once (as in [Local Windows R](/environment/local-windows-r.md)).
  [^build]
