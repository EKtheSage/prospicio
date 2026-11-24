---
type: Environment Fact
title: Splitting a pyo3 binding file leaves the stub unchanged; macros cannot sit inside pymethods
description: The Python stub that maturin extracts with pyo3 0.29's experimental-inspect depends on each class's name and module attributes, its docs and signatures, not on the Rust module or file it is defined in, so moving pyclasses into submodules re-exported by glob leaves prospicio_native.pyi byte-identical; pyo3 0.29 rejects a macro invocation inside a pymethods block, and without multiple-pymethods each class keeps one pymethods block, so macro_rules helpers for binding glue go on plain impl blocks only.
tags: [environment, python, bindings, pyo3, stub, macros]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-24T12:00:00Z }
stale_after: 2027-05-24T00:00:00Z
sources:
  - id: split
    resource: process:local session claude/split-py-reserving-963-2a853d, 2026-11-24
    title: maturin develop --release --generate-stubs before and after splitting crates/prospicio-python/src/reserving.rs (#963)
  - id: pyimpl
    resource: https://github.com/PyO3/pyo3/blob/v0.29.2/pyo3-macros-backend/src/pyimpl.rs
    title: pyo3-macros-backend 0.29.2, pyimpl.rs, the ImplItem::Macro arm
---

# Splitting a pyo3 binding file

**Fact.** The stub `python/prospicio/prospicio_native.pyi` is extracted from
the built module by pyo3's `experimental-inspect` feature
(`maturin develop --generate-stubs`). It lists classes and functions in
alphabetical order and names each by its `#[pyclass(name = ..., module =
...)]` attributes, so it does not depend on which Rust module or file holds
the item. Splitting the 22,305-line `crates/prospicio-python/src/reserving.rs`
into 28 files under `reserving/`, each starting with `use super::*;` and
re-exported from `reserving.rs` with `pub(crate) use <file>::*;`, left the
stub byte-identical on Windows, and `#[pymodule_export]` in `lib.rs` kept
finding the classes and the `#[pyfunction]`s through the glob re-exports.

**What a split needs.** A glob re-export carries only items visible to the
re-exporting module, so the moved items another file names become
`pub(crate)`, and so do the fields and methods a sibling file reaches
(`.inner` of a pyclass built or read by another file's code). Private
`use` imports of the parent reach its children through `use super::*;`
without an unused-import warning. A `macro_rules!` used by another file
gets `pub(crate) use <macro>;` after its definition.

**Macros.** pyo3 0.29 rejects a macro invocation inside a `#[pymethods]`
block ("macros cannot be used as items in `#[pymethods]` impl blocks"), and
without the `multiple-pymethods` feature each class has one `#[pymethods]`
block, so a macro cannot add Python methods to a class beside its other
methods. Macros that remove repeated binding glue therefore generate plain
`impl` blocks (Rust-side helpers, such as `impl_resolved!` and
`impl_one_segment!` in `reserving/`), and the repeated Python getters stay
written out, each with its own docstring.
