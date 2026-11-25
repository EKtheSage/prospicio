---
type: Environment Fact
title: The numpy crate 0.29 works with pyo3 0.29's abi3 build and lends a frozen class's buffer to NumPy
description: numpy 0.29 builds against pyo3 0.29.2 with abi3-py39 and experimental-inspect; PyArray2::borrow_from_array gives NumPy a view of a pyclass's Vec<f64> that holds a reference to the object, safe for a frozen class, and setting flags.writeable to False makes it read-only; a #[derive(FromPyObject)] enum of PyReadonlyArray2 and Vec<Vec<f64>> takes both arrays and lists, an int64 array falling through to the list arm; the stub types numpy arguments and returns as Incomplete.
tags: [environment, python, bindings, pyo3, numpy]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-25T12:00:00Z }
stale_after: 2027-05-25T00:00:00Z
sources:
  - id: build
    resource: process:cloud session claude/project-thread-1n3mjz-978-numpy-pd, 2026-11-25
    title: maturin develop --release --generate-stubs with numpy 0.29 added to prospicio-python (#978)
  - id: borrow
    resource: https://docs.rs/numpy/0.29.0/numpy/array/struct.PyArray.html#method.borrow_from_array
    title: numpy 0.29, PyArray::borrow_from_array
---

# NumPy in the Python bindings

**Fact.** `numpy = "0.29"` builds in `prospicio-python` next to
`pyo3 = "0.29.2"` with the `abi3-py39` and `experimental-inspect` features,
and the wheel stays a single abi3 wheel. The Python package then needs NumPy
at run time (`dependencies` in `python/pyproject.toml`).

- **Views without a copy.** `PyArray2::borrow_from_array(&view, owner)`
  (unsafe) makes an array over memory that `owner` holds and keeps `owner`
  alive through the array's base. It is sound only for a buffer that never
  changes or moves, so it is used on `frozen` classes whose numbers live in
  one immutable `Vec<f64>` (`PredictiveDistribution.to_numpy()`), and the
  array's `flags.writeable` is set to `False` so Python cannot write into
  the Rust buffer. Deleting the Python object and collecting leaves the
  view valid.
- **Arrays or lists in.** A `#[derive(FromPyObject)]` enum with a
  `PyReadonlyArray2<f64>` arm first and a `Vec<Vec<f64>>` arm second takes
  a float64 array in one copy (`as_slice` for a C-contiguous one, row order
  otherwise) and a list of rows as before. An `int64` array fails the first
  arm and is read by the second, row by row, as it was before NumPy.
- **Stub.** The stub generated with `--generate-stubs` types a numpy
  argument or return as `Incomplete`, so the docstring's Returns section
  carries the type.

`crates/prospicio-python/src/arrays.rs` holds both helpers, and `python/tests/test_distributions_numpy.py` checks them.
