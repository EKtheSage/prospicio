//! `actuarialrs.distributions`: wrappers over `act_prob` distributions.

use act_core::StreamRng;
use act_prob::Distribution;
use pyo3::prelude::*;

use crate::to_py;

/// Lognormal distribution: ``ln X ~ Normal(meanlog, sdlog**2)``.
///
/// Parameters
/// ----------
/// meanlog : float
///     Mean of ``ln X``.
/// sdlog : float
///     Standard deviation of ``ln X``; must be positive.
///
/// Raises
/// ------
/// ValueError
///     If ``meanlog`` is not finite or ``sdlog`` is not positive and finite.
///
/// Examples
/// --------
/// >>> from actuarialrs.distributions import Lognormal
/// >>> d = Lognormal.from_mean_cv(1000.0, 0.5)
/// >>> round(d.mean(), 6)
/// 1000.0
#[pyclass(name = "Lognormal", module = "actuarialrs.distributions", frozen)]
pub(crate) struct PyLognormal {
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
    ///
    /// Parameters
    /// ----------
    /// mean : float
    ///     Mean of ``X``; must be positive.
    /// cv : float
    ///     Coefficient of variation of ``X``; must be positive.
    ///
    /// Returns
    /// -------
    /// Lognormal
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``mean`` or ``cv`` is not positive and finite.
    #[staticmethod]
    fn from_mean_cv(mean: f64, cv: f64) -> PyResult<Self> {
        let inner = act_prob::Lognormal::from_mean_cv(mean, cv).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Mean of ``ln X``.
    #[getter]
    fn meanlog(&self) -> f64 {
        self.inner.meanlog()
    }

    /// Standard deviation of ``ln X``.
    #[getter]
    fn sdlog(&self) -> f64 {
        self.inner.sdlog()
    }

    /// Mean of the distribution.
    ///
    /// Returns
    /// -------
    /// float
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// Variance of the distribution.
    ///
    /// Returns
    /// -------
    /// float
    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// Standard deviation of the distribution.
    ///
    /// Returns
    /// -------
    /// float
    fn std(&self) -> f64 {
        self.inner.std_dev()
    }

    /// Distribution function ``P(X <= x)``.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///
    /// Returns
    /// -------
    /// float
    fn cdf(&self, x: f64) -> f64 {
        self.inner.cdf(x)
    }

    /// Quantile: the smallest ``x`` with ``P(X <= x) >= p``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///     Probability in ``[0, 1]``; ``quantile(1.0)`` is ``inf``.
    ///
    /// Returns
    /// -------
    /// float
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``p`` is outside ``[0, 1]``.
    fn quantile(&self, p: f64) -> PyResult<f64> {
        self.inner.quantile(p).map_err(to_py)
    }

    /// ``n`` draws from stream ``stream`` of the generator keyed by ``seed``.
    ///
    /// The same ``(seed, stream)`` gives the same draws in Python, R and Rust.
    ///
    /// Parameters
    /// ----------
    /// n : int
    ///     Number of draws.
    /// seed : int
    ///     Generator seed.
    /// stream : int, default 0
    ///     Stream id; distinct streams are independent.
    ///
    /// Returns
    /// -------
    /// list of float
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
