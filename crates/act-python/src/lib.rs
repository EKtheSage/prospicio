//! Python bindings, built with maturin from `python/`.
//!
//! Wrappers convert arguments, release the GIL around any real work, and map
//! [`act_core::Error`] to Python exceptions. They hold no numerical code.
//! Each lane keeps its wrappers in its own module; this file only registers
//! them.
//!
//! The `///` comments on `#[pyclass]` and `#[pymethods]` items are the Python
//! docstrings, written in numpydoc style. `cargo xtask python` copies them
//! into `python/actuarialrs/actuarialrs_native.pyi` and the Python docs site
//! (docs/architecture.md, "Documentation").

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod aggregate;
mod distributions;

fn to_py(e: act_core::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pymodule]
mod actuarialrs_native {
    #[pymodule_export]
    use super::aggregate::{
        PyCompoundReport, PyEventSet, PyLayer, PyTower, fft, panjer, simulate_events,
    };
    #[pymodule_export]
    use super::distributions::{
        PyDiscretizationReport, PyGrid, PyLognormal, PyNegativeBinomial, PyPoisson,
        PyPredictiveDistribution, PySampled,
    };
}
