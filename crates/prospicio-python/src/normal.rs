//! `prospicio.distributions.Normal` and `Ogive` (Probability lane), over
//! `prospicio_prob::Normal` and `prospicio_prob::Ogive`.

use prospicio_core::StreamRng;
use prospicio_prob::{Distribution, Normal, Ogive, Severity};
use pyo3::prelude::*;

use crate::to_py;

/// The normal distribution, on the whole real line: asset returns, log
/// returns and other outcomes of either sign.
///
/// Its TVaR is ``mu + sigma * phi(z_p) / (1 - p)`` with ``z_p`` the standard
/// normal quantile (*Pricing Insurance Risk*, section 4.3.3): VaR plus a
/// multiple of ``sigma`` that depends only on ``p``.
///
/// Parameters
/// ----------
/// mu : float
///     The mean.
/// sigma : float
///     The standard deviation, positive.
///
/// Raises
/// ------
/// ValueError
///     If ``mu`` is not finite or ``sigma`` not finite and positive.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Normal
/// >>> x = Normal(0.0, 1.0)
/// >>> round(x.quantile(0.99), 6), round(x.tvar(0.99), 6)
/// (2.326348, 2.665214)
#[pyclass(name = "Normal", module = "prospicio.distributions", frozen)]
pub(crate) struct PyNormal {
    inner: Normal,
}

#[pymethods]
impl PyNormal {
    #[new]
    fn new(mu: f64, sigma: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Normal::new(mu, sigma).map_err(to_py)?,
        })
    }

    /// The mean.
    #[getter]
    fn mu(&self) -> f64 {
        self.inner.mu()
    }

    /// The standard deviation.
    #[getter]
    fn sigma(&self) -> f64 {
        self.inner.sigma()
    }

    /// The mean.
    ///
    /// Returns
    /// -------
    /// float
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// The variance.
    ///
    /// Returns
    /// -------
    /// float
    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// The standard deviation.
    ///
    /// Returns
    /// -------
    /// float
    fn std_dev(&self) -> f64 {
        self.inner.std_dev()
    }

    /// The density.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///
    /// Returns
    /// -------
    /// float
    fn pdf(&self, x: f64) -> f64 {
        self.inner.pdf(x)
    }

    /// ``P(X <= x)``.
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

    /// ``P(X > x)``, precise far in the upper tail.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///
    /// Returns
    /// -------
    /// float
    fn survival(&self, x: f64) -> f64 {
        self.inner.survival(x)
    }

    /// The quantile at ``p``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///     In ``[0, 1]``.
    ///
    /// Returns
    /// -------
    /// float
    fn quantile(&self, p: f64) -> PyResult<f64> {
        self.inner.quantile(p).map_err(to_py)
    }

    /// Tail value at risk at ``p``, ``E[X | X >= VaR(p)]``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///     In ``[0, 1)``.
    ///
    /// Returns
    /// -------
    /// float
    fn tvar(&self, p: f64) -> PyResult<f64> {
        self.inner.tvar(p).map_err(to_py)
    }

    /// ``n`` draws from stream ``stream`` of the generator keyed by ``seed``.
    ///
    /// Parameters
    /// ----------
    /// n : int
    /// seed : int
    /// stream : int, default 0
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (n, seed, stream = 0))]
    fn sample(&self, py: Python<'_>, n: usize, seed: u64, stream: u64) -> Vec<f64> {
        let inner = self.inner;
        py.detach(|| inner.sample(&mut StreamRng::new(seed, stream), n))
    }

    fn __repr__(&self) -> String {
        format!(
            "Normal(mu={}, sigma={})",
            self.inner.mu(),
            self.inner.sigma()
        )
    }
}

/// An ogive: the piecewise linear distribution function through tabulated
/// points ``(x, F(x))``, the usual reading of grouped loss data (losses
/// uniform within each band), as in *Pricing Insurance Risk*, Exercise 50.
///
/// Its limited moments are exact, band by band. Build one from grouped
/// counts with :meth:`Ogive.from_counts`.
///
/// Parameters
/// ----------
/// x : list of float
///     At least two band boundaries, finite, non-negative and strictly
///     increasing.
/// cdf : list of float
///     ``F`` at each boundary, non-decreasing from 0 to 1.
///
/// Raises
/// ------
/// ValueError
///     If the table breaks any of these.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Ogive
/// >>> o = Ogive.from_counts([0, 1000, 5000, 10000], [40, 50, 10])
/// >>> o.cdf(3000.0), round(o.quantile(0.95), 6), round(o.tvar(0.95), 6)
/// (0.65, 7500.0, 8750.0)
#[pyclass(name = "Ogive", module = "prospicio.distributions", frozen)]
pub(crate) struct PyOgive {
    inner: Ogive,
}

#[pymethods]
impl PyOgive {
    #[new]
    fn new(x: Vec<f64>, cdf: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: Ogive::new(x, cdf).map_err(to_py)?,
        })
    }

    /// The ogive of grouped data: ``counts[k]`` losses (or any weights)
    /// between ``bounds[k]`` and ``bounds[k + 1]``, spread uniformly.
    ///
    /// Parameters
    /// ----------
    /// bounds : list of float
    ///     One more than the counts.
    /// counts : list of float
    ///     Non-negative, with a positive total.
    ///
    /// Returns
    /// -------
    /// Ogive
    #[staticmethod]
    fn from_counts(bounds: Vec<f64>, counts: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: Ogive::from_counts(&bounds, &counts).map_err(to_py)?,
        })
    }

    /// The band boundaries.
    #[getter]
    fn x(&self) -> Vec<f64> {
        self.inner.x().to_vec()
    }

    /// ``F`` at the boundaries.
    #[getter]
    fn cdf_values(&self) -> Vec<f64> {
        self.inner.f().to_vec()
    }

    /// The mean.
    ///
    /// Returns
    /// -------
    /// float
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// The variance.
    ///
    /// Returns
    /// -------
    /// float
    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// The standard deviation.
    ///
    /// Returns
    /// -------
    /// float
    fn std_dev(&self) -> f64 {
        self.inner.std_dev()
    }

    /// The density, constant on each band and 0 outside.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///
    /// Returns
    /// -------
    /// float
    fn pdf(&self, x: f64) -> f64 {
        self.inner.pdf(x)
    }

    /// ``P(X <= x)``, linear between the boundaries.
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

    /// ``P(X > x)``.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///
    /// Returns
    /// -------
    /// float
    fn survival(&self, x: f64) -> f64 {
        self.inner.survival(x)
    }

    /// The lower quantile, the smallest ``x`` with ``F(x) >= p``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///     In ``[0, 1]``.
    ///
    /// Returns
    /// -------
    /// float
    fn quantile(&self, p: f64) -> PyResult<f64> {
        self.inner.quantile(p).map_err(to_py)
    }

    /// Tail value at risk at ``p``, the mean of the worst ``1 - p``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///     In ``[0, 1)``.
    ///
    /// Returns
    /// -------
    /// float
    fn tvar(&self, p: f64) -> PyResult<f64> {
        self.inner.tvar(p).map_err(to_py)
    }

    /// Limited expected value ``E[min(X, limit)]``.
    ///
    /// Parameters
    /// ----------
    /// limit : float
    ///
    /// Returns
    /// -------
    /// float
    fn lev(&self, limit: f64) -> f64 {
        self.inner.lev(limit)
    }

    /// Expected loss to the layer ``limit`` xs ``attachment``.
    ///
    /// Parameters
    /// ----------
    /// limit : float
    /// attachment : float
    ///
    /// Returns
    /// -------
    /// float
    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        self.inner.layer(limit, attachment)
    }

    /// Variance of the loss to the layer ``limit`` xs ``attachment``.
    ///
    /// Parameters
    /// ----------
    /// limit : float
    /// attachment : float
    ///
    /// Returns
    /// -------
    /// float
    fn layer_variance(&self, limit: f64, attachment: f64) -> f64 {
        self.inner.layer_variance(limit, attachment)
    }

    /// ``n`` draws from stream ``stream`` of the generator keyed by ``seed``.
    ///
    /// Parameters
    /// ----------
    /// n : int
    /// seed : int
    /// stream : int, default 0
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (n, seed, stream = 0))]
    fn sample(&self, py: Python<'_>, n: usize, seed: u64, stream: u64) -> Vec<f64> {
        let inner = &self.inner;
        py.detach(|| inner.sample(&mut StreamRng::new(seed, stream), n))
    }

    fn __len__(&self) -> usize {
        self.inner.x().len()
    }

    fn __repr__(&self) -> String {
        let x = self.inner.x();
        format!(
            "Ogive(n={}, from {} to {}, mean={})",
            x.len(),
            x[0],
            x[x.len() - 1],
            self.inner.mean()
        )
    }
}
