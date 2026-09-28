//! Python bindings, built with maturin from `python/`.
//!
//! Wrappers convert arguments, release the GIL around any real work, and map
//! [`act_core::Error`] to Python exceptions. They hold no numerical code.

use act_core::StreamRng;
use act_prob::Distribution;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn to_py(e: act_core::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Lognormal distribution: ``ln X ~ Normal(meanlog, sdlog**2)``.
#[pyclass(name = "Lognormal", module = "actuarialrs.distributions", frozen)]
struct PyLognormal {
    inner: act_prob::Lognormal,
}

#[pymethods]
impl PyLognormal {
    #[new]
    fn new(meanlog: f64, sdlog: f64) -> PyResult<Self> {
        let inner = act_prob::Lognormal::new(meanlog, sdlog).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Lognormal with the given mean and coefficient of variation.
    #[staticmethod]
    fn from_mean_cv(mean: f64, cv: f64) -> PyResult<Self> {
        let inner = act_prob::Lognormal::from_mean_cv(mean, cv).map_err(to_py)?;
        Ok(Self { inner })
    }

    #[getter]
    fn meanlog(&self) -> f64 {
        self.inner.meanlog()
    }

    #[getter]
    fn sdlog(&self) -> f64 {
        self.inner.sdlog()
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    fn std(&self) -> f64 {
        self.inner.std_dev()
    }

    fn cdf(&self, x: f64) -> f64 {
        self.inner.cdf(x)
    }

    fn quantile(&self, p: f64) -> PyResult<f64> {
        self.inner.quantile(p).map_err(to_py)
    }

    /// ``n`` draws from stream ``stream`` of the generator keyed by ``seed``.
    ///
    /// The same ``(seed, stream)`` gives the same draws in Python, R and Rust.
    #[pyo3(signature = (n, seed, stream = 0))]
    fn sample(&self, py: Python<'_>, n: usize, seed: u64, stream: u64) -> Vec<f64> {
        let inner = self.inner;
        py.detach(|| inner.sample(&mut StreamRng::new(seed, stream), n))
    }

    fn __getnewargs__(&self) -> (f64, f64) {
        (self.inner.meanlog(), self.inner.sdlog())
    }

    fn __repr__(&self) -> String {
        format!(
            "Lognormal(meanlog={:?}, sdlog={:?})",
            self.inner.meanlog(),
            self.inner.sdlog()
        )
    }
}

#[pymodule]
fn actuarialrs_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLognormal>()?;
    Ok(())
}
