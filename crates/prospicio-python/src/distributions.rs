//! `prospicio.distributions`: wrappers over `prospicio_prob` distributions.

use prospicio_core::StreamRng;
use prospicio_prob::{
    ComponentKey, Counting, DiscretizationReport, Dist, Distribution, Empirical, Grid, KeyValue,
    PredictiveDistribution, Provenance, Sampled, Severity, SeverityDist,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
/// >>> from prospicio.distributions import Lognormal
/// >>> d = Lognormal.from_mean_cv(1000.0, 0.5)
/// >>> round(d.mean(), 6)
/// 1000.0
#[pyclass(name = "Lognormal", module = "prospicio.distributions", frozen)]
pub(crate) struct PyLognormal {
    pub(crate) inner: prospicio_prob::Lognormal,
}

#[pymethods]
impl PyLognormal {
    #[new]
    fn new(meanlog: f64, sdlog: f64) -> PyResult<Self> {
        let inner = prospicio_prob::Lognormal::new(meanlog, sdlog).map_err(to_py)?;
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
        let inner = prospicio_prob::Lognormal::from_mean_cv(mean, cv).map_err(to_py)?;
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

    /// Value at risk, the lower quantile at ``p``: the same as ``quantile``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///
    /// Returns
    /// -------
    /// float
    fn var(&self, p: f64) -> PyResult<f64> {
        self.inner.quantile(p).map_err(to_py)
    }

    /// Tail value at risk at ``p`` in ``[0, 1)``, the mean of the worst
    /// ``1 - p``: ``VaR(p) + E[(X - VaR(p))+] / (1 - p)``; ``inf`` where the
    /// mean is.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///
    /// Returns
    /// -------
    /// float
    fn tvar(&self, p: f64) -> PyResult<f64> {
        self.inner.tvar(p).map_err(to_py)
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

    /// Expected excess over a retention, ``E[max(X - retention, 0)]``,
    /// accurate far into the tail.
    ///
    /// Parameters
    /// ----------
    /// retention : float
    ///
    /// Returns
    /// -------
    /// float
    fn stop_loss(&self, retention: f64) -> f64 {
        self.inner.stop_loss(retention)
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
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import Lognormal
    /// >>> d = Lognormal(7.0, 0.5)
    /// >>> abs(d.layer(1000.0, 0.0) - d.lev(1000.0)) < 1e-9
    /// True
    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        self.inner.layer(limit, attachment)
    }

    /// Second moment of the loss to the layer ``limit`` xs
    /// ``attachment``, in closed form.
    ///
    /// Parameters
    /// ----------
    /// limit : float
    /// attachment : float
    ///
    /// Returns
    /// -------
    /// float
    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        self.inner.layer_second_moment(limit, attachment)
    }

    /// Third moment of the loss to the layer ``limit`` xs ``attachment``,
    /// ``E[min(max(X - a, 0), l)^3]``, by quadrature of the survival
    /// function (relative accuracy about 1e-10); ``limit = inf`` for the
    /// unlimited layer, ``attachment = 0`` for the limited third moment.
    ///
    /// Parameters
    /// ----------
    /// limit : float
    /// attachment : float
    ///
    /// Returns
    /// -------
    /// float
    fn layer_third_moment(&self, limit: f64, attachment: f64) -> f64 {
        self.inner.layer_third_moment(limit, attachment)
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

/// Any native distribution: a parametric family, a ``Mixture``, a ``Grid``
/// or a ``Sampled``.
pub(crate) fn extract_dist(obj: &Bound<'_, PyAny>) -> PyResult<Dist> {
    use crate::pareto::{
        PyAffine, PyBeta, PyBurr, PyCustom, PyGamma, PyGeneralizedPareto, PyInverseGamma,
        PyInverseGaussian, PyLogAffinePareto, PyLoglogistic, PyMixture, PyPareto,
        PyPiecewisePareto, PyPolicyLimits, PySizeOfLoss, PyTruncated, PyTweedie, PyWeibull,
    };
    if let Ok(d) = obj.extract::<PyRef<'_, PyLognormal>>() {
        return Ok(d.inner.into());
    }
    if let Ok(g) = obj.extract::<PyRef<'_, PyGrid>>() {
        return Ok(g.inner.clone().into());
    }
    if let Ok(d) = obj.extract::<PyRef<'_, PyPareto>>() {
        return Ok(d.inner.into());
    }
    if let Ok(d) = obj.extract::<PyRef<'_, PyPiecewisePareto>>() {
        return Ok(d.inner.clone().into());
    }
    if let Ok(d) = obj.extract::<PyRef<'_, PyLogAffinePareto>>() {
        return Ok(d.inner.into());
    }
    if let Ok(d) = obj.extract::<PyRef<'_, PyGeneralizedPareto>>() {
        return Ok(d.inner.into());
    }
    if let Ok(d) = obj.extract::<PyRef<'_, PyGamma>>() {
        return Ok(d.inner.into());
    }
    if let Ok(d) = obj.extract::<PyRef<'_, PyTweedie>>() {
        return Ok(d.inner.into());
    }
    if let Ok(d) = obj.extract::<PyRef<'_, PyWeibull>>() {
        return Ok(d.inner.into());
    }
    if let Ok(d) = obj.extract::<PyRef<'_, PyLoglogistic>>() {
        return Ok(d.inner.into());
    }
    if let Ok(d) = obj.extract::<PyRef<'_, PyInverseGamma>>() {
        return Ok(d.inner.into());
    }
    if let Ok(d) = obj.extract::<PyRef<'_, PyInverseGaussian>>() {
        return Ok(d.inner.into());
    }
    if let Ok(d) = obj.extract::<PyRef<'_, PyBurr>>() {
        return Ok(d.inner.into());
    }
    if let Ok(d) = obj.extract::<PyRef<'_, PyBeta>>() {
        return Ok(d.inner.into());
    }
    if let Ok(d) = obj.extract::<PyRef<'_, PyTruncated>>() {
        return Ok(Dist::Truncated(d.inner.clone()));
    }
    if let Ok(d) = obj.extract::<PyRef<'_, PyAffine>>() {
        return Ok(Dist::Affine(d.inner.clone()));
    }
    if let Ok(d) = obj.extract::<PyRef<'_, PyPolicyLimits>>() {
        return Ok(Dist::PolicyLimits(d.inner.clone()));
    }
    if let Ok(d) = obj.extract::<PyRef<'_, PySizeOfLoss>>() {
        return Ok(Dist::SizeOfLoss(d.inner.clone()));
    }
    if let Ok(d) = obj.extract::<PyRef<'_, PyMixture>>() {
        return Ok(d.inner.clone().into());
    }
    if let Ok(s) = obj.extract::<PyRef<'_, PySampled>>() {
        return Ok(s.inner.clone().into());
    }
    if let Ok(d) = obj.extract::<PyRef<'_, PyCustom>>() {
        return Ok(d.inner.clone().into());
    }
    Err(PyTypeError::new_err(format!(
        "expected a distribution ({SEVERITIES} or Sampled), got {}",
        type_name(obj)
    )))
}

/// A distribution back as the Python class of its family.
pub(crate) fn dist_to_py(py: Python<'_>, d: Dist) -> PyResult<Py<PyAny>> {
    use crate::pareto::{
        PyAffine, PyBeta, PyBurr, PyGamma, PyGeneralizedPareto, PyInverseGamma, PyInverseGaussian,
        PyLogAffinePareto, PyLoglogistic, PyMixture, PyPareto, PyPiecewisePareto, PyPolicyLimits,
        PySizeOfLoss, PyTruncated, PyTweedie, PyWeibull,
    };
    Ok(match d {
        Dist::Lognormal(inner) => Py::new(py, PyLognormal { inner })?.into_any(),
        Dist::Pareto(inner) => Py::new(py, PyPareto { inner })?.into_any(),
        Dist::PiecewisePareto(inner) => Py::new(py, PyPiecewisePareto { inner })?.into_any(),
        Dist::LogAffinePareto(inner) => Py::new(py, PyLogAffinePareto { inner })?.into_any(),
        Dist::GeneralizedPareto(inner) => Py::new(py, PyGeneralizedPareto { inner })?.into_any(),
        Dist::Gamma(inner) => Py::new(py, PyGamma { inner })?.into_any(),
        Dist::Tweedie(inner) => Py::new(py, PyTweedie { inner })?.into_any(),
        Dist::Weibull(inner) => Py::new(py, PyWeibull { inner })?.into_any(),
        Dist::Loglogistic(inner) => Py::new(py, PyLoglogistic { inner })?.into_any(),
        Dist::InverseGamma(inner) => Py::new(py, PyInverseGamma { inner })?.into_any(),
        Dist::InverseGaussian(inner) => Py::new(py, PyInverseGaussian { inner })?.into_any(),
        Dist::Burr(inner) => Py::new(py, PyBurr { inner })?.into_any(),
        Dist::Beta(inner) => Py::new(py, PyBeta { inner })?.into_any(),
        Dist::Truncated(inner) => Py::new(py, PyTruncated { inner })?.into_any(),
        Dist::Affine(inner) => Py::new(py, PyAffine { inner })?.into_any(),
        Dist::PolicyLimits(inner) => Py::new(py, PyPolicyLimits { inner })?.into_any(),
        Dist::Mixture(inner) => Py::new(py, PyMixture { inner })?.into_any(),
        Dist::Grid(inner) => Py::new(py, PyGrid { inner })?.into_any(),
        Dist::SizeOfLoss(inner) => Py::new(py, PySizeOfLoss { inner })?.into_any(),
        Dist::Sampled(inner) => Py::new(py, PySampled { inner })?.into_any(),
        Dist::Custom(_) => {
            return Err(PyTypeError::new_err(
                "a custom distribution cannot be loaded",
            ));
        }
    })
}

/// A distribution as a JSON document: the family and the parameters its
/// constructor takes, versioned, numbers bit for bit. ``from_json`` reads
/// it back to an equal distribution of the same class.
///
/// A ``Custom`` cannot be saved: it is a Python function.
///
/// Parameters
/// ----------
/// dist : a distribution
///     Any distribution class, ``Sampled`` and ``Mixture`` included.
///
/// Returns
/// -------
/// str
///
/// Raises
/// ------
/// ValueError
///     For a ``Custom``.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Lognormal, from_json, to_json
/// >>> text = to_json(Lognormal(7.0, 0.5))
/// >>> from_json(text).mean() == Lognormal(7.0, 0.5).mean()
/// True
#[pyfunction]
pub(crate) fn to_json(dist: &Bound<'_, PyAny>) -> PyResult<String> {
    extract_dist(dist)?.to_json().map_err(to_py)
}

/// A distribution from a document written by ``to_json``, as the class of
/// its family.
///
/// Parameters
/// ----------
/// text : str
///
/// Returns
/// -------
/// a distribution
///
/// Raises
/// ------
/// ValueError
///     If the document is malformed, of another format or a newer
///     version, or its parameters are out of range.
#[pyfunction]
pub(crate) fn from_json(py: Python<'_>, text: &str) -> PyResult<Py<PyAny>> {
    dist_to_py(py, Dist::from_json(text).map_err(to_py)?)
}

/// The classes accepted as a severity, for error messages.
const SEVERITIES: &str = "Lognormal, Gamma, Tweedie, Weibull, Loglogistic, InverseGamma, \
                          InverseGaussian, Burr, Beta, Truncated, Affine, PolicyLimits, Mixture, Grid, \
                          SizeOfLoss, Pareto, \
                          PiecewisePareto, LogAffinePareto, GeneralizedPareto or Custom";

fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map_or_else(|_| "an unknown type".into(), |n| n.to_string())
}

/// A severity accepted wherever a parametric or discretized loss
/// distribution can be used: any distribution but ``Sampled``, which has no
/// exact layer moments.
pub(crate) fn extract_severity(obj: &Bound<'_, PyAny>) -> PyResult<SeverityDist> {
    let dist = extract_dist(obj).map_err(|_| {
        PyTypeError::new_err(format!(
            "expected a severity ({SEVERITIES}), got {}",
            type_name(obj)
        ))
    })?;
    SeverityDist::try_from(dist).map_err(|_| {
        PyTypeError::new_err(format!(
            "expected a severity ({SEVERITIES}); a Sampled has no exact layer moments"
        ))
    })
}

/// Poisson claim counts with mean ``lam``.
///
/// Parameters
/// ----------
/// lam : float
///     Mean number of claims; must be finite and non-negative.
///
/// Raises
/// ------
/// ValueError
///     If ``lam`` is negative or not finite.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Poisson
/// >>> n = Poisson(3.0)
/// >>> round(n.pmf(0), 6)
/// 0.049787
#[pyclass(name = "Poisson", module = "prospicio.distributions", frozen)]
pub(crate) struct PyPoisson {
    pub(crate) inner: prospicio_prob::Poisson,
}

#[pymethods]
impl PyPoisson {
    #[new]
    fn new(lam: f64) -> PyResult<Self> {
        let inner = prospicio_prob::Poisson::new(lam).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// The mean number of claims.
    #[getter]
    fn lam(&self) -> f64 {
        self.inner.lambda()
    }

    /// ``P(N = k)``.
    ///
    /// Parameters
    /// ----------
    /// k : int
    ///
    /// Returns
    /// -------
    /// float
    fn pmf(&self, k: u64) -> f64 {
        self.inner.pmf(k)
    }

    /// ``P(N <= k)``.
    ///
    /// Parameters
    /// ----------
    /// k : int
    ///
    /// Returns
    /// -------
    /// float
    fn cdf(&self, k: u64) -> f64 {
        self.inner.cdf(k)
    }

    /// Mean of the claim count.
    ///
    /// Returns
    /// -------
    /// float
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// Variance of the claim count.
    ///
    /// Returns
    /// -------
    /// float
    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// Smallest ``k`` with ``P(N <= k) >= p``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///     Probability in ``[0, 1)``.
    ///
    /// Returns
    /// -------
    /// int
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``p`` is outside ``[0, 1]``.
    fn quantile(&self, p: f64) -> PyResult<u64> {
        self.inner.quantile(p).map_err(to_py)
    }

    /// ``n`` claim counts from stream ``stream`` of the generator keyed by
    /// ``seed``.
    ///
    /// Parameters
    /// ----------
    /// n : int
    /// seed : int
    /// stream : int, default 0
    ///
    /// Returns
    /// -------
    /// list of int
    #[pyo3(signature = (n, seed, stream = 0))]
    fn sample(&self, py: Python<'_>, n: usize, seed: u64, stream: u64) -> Vec<u64> {
        let inner = self.inner;
        py.detach(|| inner.sample(&mut StreamRng::new(seed, stream), n))
    }

    fn __getnewargs__(&self) -> (f64,) {
        (self.inner.lambda(),)
    }

    fn __repr__(&self) -> String {
        format!("Poisson(lam={:?})", self.inner.lambda())
    }
}

/// Negative binomial claim counts: mean ``r * beta``, variance
/// ``r * beta * (1 + beta)`` (Klugman, Panjer & Willmot).
///
/// SciPy's ``nbinom(n=r, p=1/(1+beta))`` is the same distribution.
///
/// Parameters
/// ----------
/// r : float
///     Shape; must be positive.
/// beta : float
///     Scale; must be positive.
///
/// Raises
/// ------
/// ValueError
///     If ``r`` or ``beta`` is not positive and finite.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import NegativeBinomial
/// >>> n = NegativeBinomial.from_mean_variance(10.0, 30.0)
/// >>> round(n.variance(), 9)
/// 30.0
#[pyclass(name = "NegativeBinomial", module = "prospicio.distributions", frozen)]
pub(crate) struct PyNegativeBinomial {
    pub(crate) inner: prospicio_prob::NegativeBinomial,
}

#[pymethods]
impl PyNegativeBinomial {
    #[new]
    fn new(r: f64, beta: f64) -> PyResult<Self> {
        let inner = prospicio_prob::NegativeBinomial::new(r, beta).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// The negative binomial with this mean and variance.
    ///
    /// Parameters
    /// ----------
    /// mean : float
    ///     Must be positive.
    /// variance : float
    ///     Must exceed the mean.
    ///
    /// Returns
    /// -------
    /// NegativeBinomial
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the mean is not positive or the variance does not exceed it.
    #[staticmethod]
    fn from_mean_variance(mean: f64, variance: f64) -> PyResult<Self> {
        let inner =
            prospicio_prob::NegativeBinomial::from_mean_variance(mean, variance).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Shape ``r``.
    #[getter]
    fn r(&self) -> f64 {
        self.inner.r()
    }

    /// Scale ``beta``.
    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta()
    }

    /// ``P(N = k)``.
    ///
    /// Parameters
    /// ----------
    /// k : int
    ///
    /// Returns
    /// -------
    /// float
    fn pmf(&self, k: u64) -> f64 {
        self.inner.pmf(k)
    }

    /// ``P(N <= k)``.
    ///
    /// Parameters
    /// ----------
    /// k : int
    ///
    /// Returns
    /// -------
    /// float
    fn cdf(&self, k: u64) -> f64 {
        self.inner.cdf(k)
    }

    /// Mean of the claim count.
    ///
    /// Returns
    /// -------
    /// float
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// Variance of the claim count.
    ///
    /// Returns
    /// -------
    /// float
    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// Smallest ``k`` with ``P(N <= k) >= p``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///     Probability in ``[0, 1)``.
    ///
    /// Returns
    /// -------
    /// int
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``p`` is outside ``[0, 1]``.
    fn quantile(&self, p: f64) -> PyResult<u64> {
        self.inner.quantile(p).map_err(to_py)
    }

    /// ``n`` claim counts from stream ``stream`` of the generator keyed by
    /// ``seed``.
    ///
    /// Parameters
    /// ----------
    /// n : int
    /// seed : int
    /// stream : int, default 0
    ///
    /// Returns
    /// -------
    /// list of int
    #[pyo3(signature = (n, seed, stream = 0))]
    fn sample(&self, py: Python<'_>, n: usize, seed: u64, stream: u64) -> Vec<u64> {
        let inner = self.inner;
        py.detach(|| inner.sample(&mut StreamRng::new(seed, stream), n))
    }

    fn __getnewargs__(&self) -> (f64, f64) {
        (self.inner.r(), self.inner.beta())
    }

    fn __repr__(&self) -> String {
        format!(
            "NegativeBinomial(r={:?}, beta={:?})",
            self.inner.r(),
            self.inner.beta()
        )
    }
}

/// How a distribution was discretized, and the error that introduced.
///
/// Returned with the grid by ``Grid.local_moment``, ``Grid.rounding`` and
/// ``Grid.lower``.
#[pyclass(
    name = "DiscretizationReport",
    module = "prospicio.distributions",
    frozen
)]
pub(crate) struct PyDiscretizationReport {
    inner: DiscretizationReport,
}

#[pymethods]
impl PyDiscretizationReport {
    /// Method: ``"local_moment"``, ``"rounding"`` or ``"lower"``.
    #[getter]
    fn method(&self) -> &'static str {
        match self.inner.method {
            prospicio_prob::Discretization::LocalMoment => "local_moment",
            prospicio_prob::Discretization::Rounding => "rounding",
            prospicio_prob::Discretization::Lower => "lower",
        }
    }

    /// Grid step.
    #[getter]
    fn step(&self) -> f64 {
        self.inner.step
    }

    /// Number of grid points.
    #[getter]
    fn points(&self) -> usize {
        self.inner.points
    }

    /// Probability the source puts above the last grid point, lumped onto it.
    #[getter]
    fn tail_mass(&self) -> f64 {
        self.inner.tail_mass
    }

    /// Mean of the source distribution.
    #[getter]
    fn source_mean(&self) -> f64 {
        self.inner.source_mean
    }

    /// Mean of the grid.
    #[getter]
    fn grid_mean(&self) -> f64 {
        self.inner.grid_mean
    }

    /// ``grid_mean - source_mean``.
    ///
    /// Returns
    /// -------
    /// float
    fn mean_error(&self) -> f64 {
        self.inner.mean_error()
    }

    fn __repr__(&self) -> String {
        format!(
            "DiscretizationReport(method={:?}, points={}, tail_mass={:e}, mean_error={:e})",
            self.method(),
            self.inner.points,
            self.inner.tail_mass,
            self.inner.mean_error()
        )
    }
}

/// A distribution on the points ``0, step, 2*step, ...``: the discretized
/// representation that FFT and Panjer aggregation work on.
///
/// Parameters
/// ----------
/// step : float
///     Grid step; must be positive.
/// probs : list of float
///     Probabilities at ``0, step, ...``; non-negative, summing to 1.
///
/// Raises
/// ------
/// ValueError
///     If the step is not positive or the probabilities are invalid.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Grid, Lognormal
/// >>> grid, report = Grid.local_moment(Lognormal(7.0, 0.5), 100.0, 200)
/// >>> report.tail_mass < 1e-8
/// True
#[pyclass(name = "Grid", module = "prospicio.distributions", frozen)]
pub(crate) struct PyGrid {
    pub(crate) inner: Grid,
}

fn discretized(
    result: prospicio_core::Result<(Grid, DiscretizationReport)>,
) -> PyResult<(PyGrid, PyDiscretizationReport)> {
    let (inner, report) = result.map_err(to_py)?;
    Ok((PyGrid { inner }, PyDiscretizationReport { inner: report }))
}

#[pymethods]
impl PyGrid {
    #[new]
    fn new(step: f64, probs: Vec<f64>) -> PyResult<Self> {
        let inner = Grid::new(step, probs).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// The distribution of ``f(X)`` on the same step.
    ///
    /// Each point's mass moves to ``f(x)``. A value between two points is
    /// split between them so its mean is kept, so the mean is always exact
    /// and the whole distribution is exact when every value lands on a
    /// point. ``f`` is a Python callable, evaluated once per point with
    /// mass.
    ///
    /// Parameters
    /// ----------
    /// f : callable
    ///     Maps a loss to a finite, non-negative value.
    ///
    /// Returns
    /// -------
    /// tuple of (Grid, bool)
    ///     The grid and whether every value landed on a grid point.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``f`` returns a negative or non-finite value.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import Grid
    /// >>> x = Grid(1.0, [0.2, 0.3, 0.3, 0.2])
    /// >>> layer, exact = x.map(lambda v: min(max(v - 1.0, 0.0), 1.0))
    /// >>> layer.probs, exact
    /// ([0.5, 0.5], True)
    fn map(&self, f: &Bound<'_, PyAny>) -> PyResult<(PyGrid, bool)> {
        let mut error = None;
        let result = self
            .inner
            .map(|x| match f.call1((x,)).and_then(|v| v.extract::<f64>()) {
                Ok(v) => v,
                Err(e) => {
                    error.get_or_insert(e);
                    f64::NAN
                }
            });
        if let Some(e) = error {
            return Err(e);
        }
        let (inner, exact) = result.map_err(to_py)?;
        Ok((PyGrid { inner }, exact))
    }

    /// Discretizes a severity by local moment matching on the mean.
    ///
    /// Parameters
    /// ----------
    /// severity : Lognormal, Grid, Pareto, PiecewisePareto, LogAffinePareto or GeneralizedPareto
    /// step : float
    /// points : int
    ///
    /// Returns
    /// -------
    /// tuple of (Grid, DiscretizationReport)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``step`` is not positive or ``points`` is 0.
    #[staticmethod]
    fn local_moment(
        severity: &Bound<'_, PyAny>,
        step: f64,
        points: usize,
    ) -> PyResult<(PyGrid, PyDiscretizationReport)> {
        let sev = extract_severity(severity)?;
        discretized(Grid::local_moment(&sev, step, points))
    }

    /// Discretizes a severity by rounding each loss to the nearest point.
    ///
    /// Parameters
    /// ----------
    /// severity : Lognormal, Grid, Pareto, PiecewisePareto, LogAffinePareto or GeneralizedPareto
    /// step : float
    /// points : int
    ///
    /// Returns
    /// -------
    /// tuple of (Grid, DiscretizationReport)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``step`` is not positive or ``points`` is 0.
    #[staticmethod]
    fn rounding(
        severity: &Bound<'_, PyAny>,
        step: f64,
        points: usize,
    ) -> PyResult<(PyGrid, PyDiscretizationReport)> {
        let sev = extract_severity(severity)?;
        discretized(Grid::rounding(&sev, step, points))
    }

    /// Discretizes a severity by moving each cell's mass to its left end: a
    /// stochastic lower bound.
    ///
    /// Parameters
    /// ----------
    /// severity : Lognormal, Grid, Pareto, PiecewisePareto, LogAffinePareto or GeneralizedPareto
    /// step : float
    /// points : int
    ///
    /// Returns
    /// -------
    /// tuple of (Grid, DiscretizationReport)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``step`` is not positive or ``points`` is 0.
    #[staticmethod]
    fn lower(
        severity: &Bound<'_, PyAny>,
        step: f64,
        points: usize,
    ) -> PyResult<(PyGrid, PyDiscretizationReport)> {
        let sev = extract_severity(severity)?;
        discretized(Grid::lower(&sev, step, points))
    }

    /// Grid step.
    #[getter]
    fn step(&self) -> f64 {
        self.inner.step()
    }

    /// Probabilities at ``0, step, 2*step, ...``.
    #[getter]
    fn probs(&self) -> Vec<f64> {
        self.inner.probs().to_vec()
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

    /// Smallest grid point ``x`` with ``P(X <= x) >= p``.
    ///
    /// Parameters
    /// ----------
    /// p : float
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

    /// Tail value at risk at ``p``: the mean of the worst ``1 - p`` of the
    /// distribution, the point at the VaR counted with its mass above
    /// ``p``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///
    /// Returns
    /// -------
    /// float
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``p`` is outside ``[0, 1]``.
    fn tvar(&self, p: f64) -> PyResult<f64> {
        self.inner.tvar(p).map_err(to_py)
    }

    /// The level ``t`` with ``TVaR(t) = VaR(p)`` (the lower VaR), as
    /// ``aggregate``'s ``tvar_threshold``; ``t <= p``, and ``p = 0`` gives 0.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///
    /// Returns
    /// -------
    /// float
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``p`` is outside ``[0, 1]``.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import Grid
    /// >>> g = Grid(1.0, [0.4, 0.3, 0.2, 0.1])
    /// >>> t = g.tvar_threshold(0.85)
    /// >>> round(g.tvar(t), 12) == g.quantile(0.85)
    /// True
    fn tvar_threshold(&self, p: f64) -> PyResult<f64> {
        self.inner.tvar_threshold(p).map_err(to_py)
    }

    /// The grid with every probability below ``eps`` set to 0: the
    /// round-off an FFT leaves where the true mass is zero, as
    /// ``aggregate``'s ``remove_fuzz``. ``Tower.on_grid`` already removes
    /// it from its ceded and net grids.
    ///
    /// Parameters
    /// ----------
    /// eps : float, default machine epsilon (2.22e-16)
    ///
    /// Returns
    /// -------
    /// Grid
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import Grid
    /// >>> Grid(1.0, [0.5, 0.5 - 1e-20, 1e-20]).remove_fuzz().probs[2]
    /// 0.0
    #[pyo3(signature = (eps = f64::EPSILON))]
    fn remove_fuzz(&self, eps: f64) -> Self {
        Self {
            inner: self.inner.remove_fuzz(eps),
        }
    }

    /// Limited expected value ``E[min(X, limit)]``, exact on the grid.
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

    /// Expected excess ``E[max(X - retention, 0)]``, exact on the grid.
    ///
    /// Parameters
    /// ----------
    /// retention : float
    ///
    /// Returns
    /// -------
    /// float
    fn stop_loss(&self, retention: f64) -> f64 {
        self.inner.stop_loss(retention)
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

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getnewargs__(&self) -> (f64, Vec<f64>) {
        (self.inner.step(), self.inner.probs().to_vec())
    }

    fn __repr__(&self) -> String {
        format!(
            "Grid(step={:?}, points={})",
            self.inner.step(),
            self.inner.len()
        )
    }
}

/// A distribution known only through equally weighted draws.
///
/// Parameters
/// ----------
/// draws : list of float
///     Non-empty, all finite.
///
/// Raises
/// ------
/// ValueError
///     If ``draws`` is empty or holds a value that is not finite.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Sampled
/// >>> s = Sampled([1.0, 2.0, 3.0, 4.0])
/// >>> s.tvar(0.5)
/// 3.5
#[pyclass(name = "Sampled", module = "prospicio.distributions", frozen)]
pub(crate) struct PySampled {
    pub(crate) inner: Sampled,
}

#[pymethods]
impl PySampled {
    #[new]
    fn new(draws: Vec<f64>) -> PyResult<Self> {
        let inner = Sampled::new(draws).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// The draws, in simulation order.
    #[getter]
    fn draws(&self) -> Vec<f64> {
        self.inner.draws().to_vec()
    }

    /// Mean of the draws.
    ///
    /// Returns
    /// -------
    /// float
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// Variance of the draws (dividing by ``n``).
    ///
    /// Returns
    /// -------
    /// float
    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// Empirical distribution function ``P(X <= x)``.
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

    /// Inverted empirical cdf (R ``type = 1``).
    ///
    /// Parameters
    /// ----------
    /// p : float
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

    /// Value at risk at level ``p``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    /// method : str
    ///     ``"inverted_cdf"`` (the default, R ``type = 1``) or ``"linear"``
    ///     (numpy's default, R ``type = 7``).
    ///
    /// Returns
    /// -------
    /// float
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``p`` is outside ``[0, 1]`` or ``method`` is unknown.
    #[pyo3(signature = (p, method = "inverted_cdf"))]
    fn var(&self, p: f64, method: &str) -> PyResult<f64> {
        self.inner
            .var_with(p, crate::risk::quantile_method(method)?)
            .map_err(to_py)
    }

    /// Tail value at risk at level ``p``: the mean of the worst ``1 - p``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    /// method : str
    ///     ``"exact"`` (the default: the empirical distribution, mass ``1/n``
    ///     per draw) or ``"linear"`` (the mean of the linear quantile
    ///     function over ``[p, 1]``).
    ///
    /// Returns
    /// -------
    /// float
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``p`` is outside ``[0, 1]`` or ``method`` is unknown.
    #[pyo3(signature = (p, method = "exact"))]
    fn tvar(&self, p: f64, method: &str) -> PyResult<f64> {
        self.inner
            .tvar_with(p, crate::risk::tvar_method(method)?)
            .map_err(to_py)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getnewargs__(&self) -> (Vec<f64>,) {
        (self.inner.draws().to_vec(),)
    }

    fn __repr__(&self) -> String {
        format!("Sampled(n={})", self.inner.len())
    }
}

/// One value of a component key, as Python passes it.
#[derive(FromPyObject)]
pub(crate) enum KeyArg {
    Int(i64),
    Text(String),
}

pub(crate) fn key_from_py(key: Vec<KeyArg>) -> ComponentKey {
    key.into_iter()
        .map(|v| match v {
            KeyArg::Int(i) => KeyValue::Int(i),
            KeyArg::Text(t) => KeyValue::Text(t),
        })
        .collect()
}

/// Groups and a matrix over them, one row per group.
type GroupMatrix<'py> = (Vec<Bound<'py, PyAny>>, Vec<Vec<f64>>);

/// Group labels as Python values, as `key_to_py` gives a key's entries.
fn groups_to_py<'py>(py: Python<'py>, groups: &[KeyValue]) -> PyResult<Vec<Bound<'py, PyAny>>> {
    groups
        .iter()
        .map(|g| key_to_py(py, &vec![g.clone()])?.get_item(0))
        .collect()
}

pub(crate) fn key_to_py<'py>(py: Python<'py>, key: &ComponentKey) -> PyResult<Bound<'py, PyAny>> {
    let values: Vec<Bound<'py, PyAny>> = key
        .iter()
        .map(|v| {
            // Converting an int or a str cannot fail.
            match v {
                KeyValue::Int(i) => {
                    let Ok(o) = i.into_pyobject(py);
                    o.into_any()
                }
                KeyValue::Text(t) => {
                    let Ok(o) = t.into_pyobject(py);
                    o.into_any()
                }
                KeyValue::Period(p) => {
                    let Ok(o) = p.to_string().into_pyobject(py);
                    o.into_any()
                }
            }
        })
        .collect();
    Ok(pyo3::types::PyTuple::new(py, values)?.into_any())
}

/// The joint result every model returns: draws for each simulation (row)
/// and component (column), keyed by dimension values.
///
/// The ``mean``, ``quantile``, ``var`` and ``tvar`` methods describe the
/// total over all components, computed from row sums.
///
/// Parameters
/// ----------
/// dims : list of str
///     Dimension names, e.g. ``["lob", "origin"]``.
/// components : list of tuple
///     One key per component, with one ``int`` or ``str`` per dimension.
/// draws : list of list of float
///     One row per simulation, one value per component.
///
/// Raises
/// ------
/// ValueError
///     If the keys or the draws do not fit together.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import PredictiveDistribution
/// >>> pd = PredictiveDistribution(["line"], [("A",), ("B",)],
/// ...                             [[0.0, 0.0], [0.0, 0.0], [0.0, 100.0], [100.0, 0.0]])
/// >>> pd.var(0.75)
/// 100.0
/// >>> pd.marginal(("A",)).var(0.75)
/// 0.0
#[pyclass(
    name = "PredictiveDistribution",
    module = "prospicio.distributions",
    frozen
)]
pub(crate) struct PyPredictiveDistribution {
    pub(crate) inner: PredictiveDistribution,
}

impl PyPredictiveDistribution {
    /// A reorder's result, with its index by group when asked for.
    fn reordered<'py>(
        &self,
        py: Python<'py>,
        dim: &str,
        (inner, index): (PredictiveDistribution, Vec<Vec<usize>>),
        return_index: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let result = Bound::new(py, Self { inner })?.into_any();
        if !return_index {
            return Ok(result);
        }
        let by_group = PyDict::new(py);
        let groups = self.inner.groups(dim).map_err(to_py)?;
        for (g, rows) in groups_to_py(py, &groups)?.into_iter().zip(index) {
            by_group.set_item(g, rows)?;
        }
        Ok((result, by_group).into_pyobject(py)?.into_any())
    }
}

#[pymethods]
impl PyPredictiveDistribution {
    #[new]
    fn new(
        dims: Vec<String>,
        components: Vec<Vec<KeyArg>>,
        draws: Vec<Vec<f64>>,
    ) -> PyResult<Self> {
        let width = components.len();
        if let Some(bad) = draws.iter().position(|row| row.len() != width) {
            return Err(PyValueError::new_err(format!(
                "row {bad} has {} values, expected one per component ({width})",
                draws[bad].len()
            )));
        }
        let inner = PredictiveDistribution::from_draws(
            dims,
            components.into_iter().map(key_from_py).collect(),
            draws.into_iter().flatten().collect(),
            Provenance::new("python"),
        )
        .map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Dimension names.
    #[getter]
    fn dims(&self) -> Vec<String> {
        self.inner.dims().to_vec()
    }

    /// Component keys, one tuple per column.
    ///
    /// Returns
    /// -------
    /// list of tuple
    fn components<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.inner
            .components()
            .iter()
            .map(|k| key_to_py(py, k))
            .collect()
    }

    /// Number of simulations (rows).
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims()
    }

    /// Number of components (columns).
    #[getter]
    fn n_components(&self) -> usize {
        self.inner.n_components()
    }

    /// All draws, one row per simulation.
    ///
    /// Returns
    /// -------
    /// list of list of float
    fn draw_matrix(&self) -> Vec<Vec<f64>> {
        self.inner
            .draw_matrix()
            .chunks_exact(self.inner.n_components())
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// Joins distributions of different models into one portfolio, with a
    /// leading dimension ``dim`` holding each part's label, followed by the
    /// union of the parts' dimensions (``""`` where a part lacks one).
    /// Simulation ``i`` of the result is simulation ``i`` of every part.
    ///
    /// Parameters
    /// ----------
    /// parts : list of (str, PredictiveDistribution)
    /// dim : str
    /// same_simulations : bool, default False
    ///     ``False``: the parts were simulated separately, and two with the
    ///     same seed and stream scheme (which would share random numbers)
    ///     are refused. ``True``: the parts come from the same scenarios (a
    ///     cover applied to a reserve) and keep their pairing.
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import PredictiveDistribution
    /// >>> a = PredictiveDistribution(["origin"], [(2023,), (2024,)], [[10.0, 20.0], [12.0, 25.0]])
    /// >>> b = PredictiveDistribution(["lob"], [("motor",)], [[50.0], [40.0]])
    /// >>> p = PredictiveDistribution.join([("reserve", a), ("premium", b)], "risk")
    /// >>> p.dims, p.total().draws
    /// (['risk', 'origin', 'lob'], [80.0, 77.0])
    #[staticmethod]
    #[pyo3(signature = (parts, dim, same_simulations = false))]
    fn join(
        parts: Vec<(String, PyRef<'_, PyPredictiveDistribution>)>,
        dim: &str,
        same_simulations: bool,
    ) -> PyResult<Self> {
        use prospicio_prob::portfolio::Pairing;
        let refs: Vec<(&str, &PredictiveDistribution)> =
            parts.iter().map(|(l, p)| (l.as_str(), &p.inner)).collect();
        let pairing = if same_simulations {
            Pairing::SameSimulations
        } else {
            Pairing::Independent
        };
        let inner = PredictiveDistribution::join(&refs, dim, pairing).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Sets the dependence between the groups of dimension ``dim`` by
    /// Iman–Conover on the groups' totals, moving each group's simulations
    /// as whole rows: every group keeps its distribution and internal joint
    /// structure, and the group totals take a rank correlation close to
    /// ``correlation``.
    ///
    /// Parameters
    /// ----------
    /// dim : str
    /// correlation : list of list of float
    ///     One row and column per group, in order of first appearance
    ///     (``groups(dim)``); ``correlation_from_pairs`` builds it from a
    ///     long table.
    /// seed : int
    /// kind : str
    ///     ``"pearson"`` (the default: the normal scores' correlation, so
    ///     Spearman's rho comes out near ``(6/π) asin(r/2)``) or
    ///     ``"spearman"`` (a Spearman target, converted to ``2 sin(π ρ /
    ///     6)`` first).
    /// return_index : bool
    ///     Also return where each simulation came from (see below).
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution, or (PredictiveDistribution, dict)
    ///     With ``return_index``, also a dict from each group to the list
    ///     of input simulations its rows came from: simulation ``i`` of the
    ///     result holds the group's row of input simulation
    ///     ``index[group][i]``. ``EventSet.permute(index[group])`` keeps a
    ///     class's events with its reordered totals, with no matching of
    ///     values (which tied draws defeat).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the matrix (after conversion) is not a positive definite
    ///     correlation matrix, or ``kind`` is unknown.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import PredictiveDistribution
    /// >>> n = 4000
    /// >>> pd = PredictiveDistribution(
    /// ...     ["seg"], [("a",), ("b",)], [[i, (i * 7919) % n] for i in range(n)])
    /// >>> r = pd.reorder_groups("seg", [[1, 0.6], [0.6, 1]], 5, kind="spearman")
    /// >>> groups, rho = r.group_spearman("seg")
    /// >>> groups, abs(rho[0][1] - 0.6) < 0.02
    /// (['a', 'b'], True)
    /// >>> r, index = pd.reorder_groups("seg", [[1, 0.6], [0.6, 1]], 5, return_index=True)
    /// >>> before, after = pd.draw_matrix(), r.draw_matrix()
    /// >>> all(after[i][0] == before[index["a"][i]][0] for i in range(n))
    /// True
    #[pyo3(signature = (dim, correlation, seed, kind = "pearson", return_index = false))]
    fn reorder_groups<'py>(
        &self,
        py: Python<'py>,
        dim: &str,
        correlation: Vec<Vec<f64>>,
        seed: u64,
        kind: &str,
        return_index: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        use prospicio_prob::portfolio::CorrelationKind;
        let kind = match kind {
            "pearson" => CorrelationKind::Pearson,
            "spearman" => CorrelationKind::Spearman,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "kind must be \"pearson\" or \"spearman\", not {kind:?}"
                )));
            }
        };
        let flat: Vec<f64> = correlation.into_iter().flatten().collect();
        let pd = &self.inner;
        let reordered = py
            .detach(|| pd.reorder_groups_indexed(dim, &flat, kind, seed))
            .map_err(to_py)?;
        self.reordered(py, dim, reordered, return_index)
    }

    /// Sets the dependence between the groups of dimension ``dim`` by a
    /// copula: ``n_sims`` draws of uniforms from ``copula`` (draw ``i`` on
    /// stream ``i`` of ``seed``, as ``copula.sample``), and each group's
    /// simulations, whole rows sorted by the group's total, take the ranks
    /// of its column. The group totals then have the copula's rank pattern,
    /// so a t, Gumbel or Clayton copula gives its joint tails to results
    /// already simulated; every group keeps its distribution and internal
    /// joint structure, as under ``reorder_groups``.
    ///
    /// Parameters
    /// ----------
    /// dim : str
    /// copula : GaussianCopula, StudentTCopula, ArchimedeanCopula, NestedArchimedeanCopula or VineCopula
    ///     One dimension per group, in order of first appearance
    ///     (``groups(dim)``).
    /// seed : int
    /// return_index : bool
    ///     Also return where each simulation came from, as
    ///     ``reorder_groups`` does.
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution, or (PredictiveDistribution, dict)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the copula's dimension is not the number of groups.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import PredictiveDistribution
    /// >>> from prospicio.risk import ArchimedeanCopula
    /// >>> n = 4000
    /// >>> pd = PredictiveDistribution(
    /// ...     ["seg"], [("a",), ("b",)], [[i, (i * 7919) % n] for i in range(n)])
    /// >>> clayton = ArchimedeanCopula("clayton", 2.0, 2)  # Spearman's rho near 0.68
    /// >>> r = pd.reorder_groups_by_copula("seg", clayton, 5)
    /// >>> abs(r.group_spearman("seg")[1][0][1] - 0.68) < 0.03
    /// True
    #[pyo3(signature = (dim, copula, seed, return_index = false))]
    fn reorder_groups_by_copula<'py>(
        &self,
        py: Python<'py>,
        dim: &str,
        copula: &Bound<'_, PyAny>,
        seed: u64,
        return_index: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let copula = crate::risk::AnyCopula::extract(copula)?;
        let pd = &self.inner;
        let reordered = py
            .detach(|| pd.reorder_groups_by_copula_indexed(dim, copula.as_copula(), seed))
            .map_err(to_py)?;
        self.reordered(py, dim, reordered, return_index)
    }

    /// The groups of dimension ``dim`` in order of first appearance: the
    /// rows and columns of ``reorder_groups``' matrix.
    ///
    /// Parameters
    /// ----------
    /// dim : str
    ///
    /// Returns
    /// -------
    /// list of int or str
    fn groups<'py>(&self, py: Python<'py>, dim: &str) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let groups = self.inner.groups(dim).map_err(to_py)?;
        groups_to_py(py, &groups)
    }

    /// The correlation matrix between the groups of ``dim`` from a long
    /// pairwise table, in the order of ``groups(dim)``. Each pair of
    /// distinct groups appears once, in either order (both orders only
    /// with the same value); a diagonal entry may be given and must be 1.
    ///
    /// Parameters
    /// ----------
    /// dim : str
    /// pairs : list of tuple
    ///     ``(group, group, value)`` rows, values in ``[-1, 1]``.
    ///
    /// Returns
    /// -------
    /// list of list of float
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a pair is missing, given twice with different values, names
    ///     an unknown group, or a value is outside ``[-1, 1]``.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import PredictiveDistribution
    /// >>> pd = PredictiveDistribution(["peril"], [("wind",), ("quake",)], [[0.0, 1.0]] * 3)
    /// >>> pd.correlation_from_pairs("peril", [("quake", "wind", 0.2)])
    /// [[1.0, 0.2], [0.2, 1.0]]
    fn correlation_from_pairs(
        &self,
        dim: &str,
        pairs: Vec<(KeyArg, KeyArg, f64)>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let one = |k: KeyArg| key_from_py(vec![k]).remove(0);
        let pairs: Vec<(KeyValue, KeyValue, f64)> = pairs
            .into_iter()
            .map(|(a, b, v)| (one(a), one(b), v))
            .collect();
        let m = self
            .inner
            .correlation_from_pairs(dim, &pairs)
            .map_err(to_py)?;
        let g = (m.len() as f64).sqrt().round() as usize;
        Ok(m.chunks(g.max(1)).map(<[f64]>::to_vec).collect())
    }

    /// Spearman's rho between the totals of the groups of ``dim``, ties
    /// given their average rank as ``scipy.stats.spearmanr``: what
    /// ``reorder_groups`` achieved.
    ///
    /// Parameters
    /// ----------
    /// dim : str
    ///
    /// Returns
    /// -------
    /// tuple of (list, list of list of float)
    ///     The groups, as ``groups(dim)``, and the matrix.
    fn group_spearman<'py>(&self, py: Python<'py>, dim: &str) -> PyResult<GroupMatrix<'py>> {
        let pd = &self.inner;
        let (groups, m) = py.detach(|| pd.group_spearman(dim)).map_err(to_py)?;
        let g = groups.len();
        Ok((
            groups_to_py(py, &groups)?,
            m.chunks(g.max(1)).map(<[f64]>::to_vec).collect(),
        ))
    }

    /// Blends several models' predictive distributions: simulation ``i`` is
    /// simulation ``i`` of model ``k``, with ``k`` drawn with probability
    /// ``weights[k]`` from stream ``i`` of ``seed``. Rows stay whole, so sums
    /// across components remain coherent. Use weights from
    /// ``stacking_weights`` or ``pseudo_bma_weights``.
    ///
    /// Parameters
    /// ----------
    /// models : list of PredictiveDistribution
    ///     Same dimensions, components and number of simulations.
    /// weights : list of float
    ///     Non-negative, not all zero; normalized.
    /// seed : int
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import PredictiveDistribution
    /// >>> a = PredictiveDistribution(["lob"], [("x",)], [[0.0]] * 1000)
    /// >>> b = PredictiveDistribution(["lob"], [("x",)], [[1.0]] * 1000)
    /// >>> mix = PredictiveDistribution.blend([a, b], [0.25, 0.75], seed=7)
    /// >>> abs(mix.mean() - 0.75) < 0.05
    /// True
    #[staticmethod]
    fn blend(
        py: Python<'_>,
        models: Vec<PyRef<'_, PyPredictiveDistribution>>,
        weights: Vec<f64>,
        seed: u64,
    ) -> PyResult<Self> {
        let refs: Vec<&PredictiveDistribution> = models.iter().map(|m| &m.inner).collect();
        let inner = py
            .detach(|| PredictiveDistribution::blend(&refs, &weights, seed))
            .map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Blends models with weights that differ by component, as
    /// ``HierarchicalStacking`` gives them: in simulation ``i`` every
    /// component draws its model from the same uniform against its own
    /// cumulative weights, so components with equal weights take the same
    /// model and dependence is kept as far as the weights allow.
    ///
    /// Parameters
    /// ----------
    /// models : list of PredictiveDistribution
    /// weights : list of list of float
    ///     One weight vector per component (in ``components()`` order), one
    ///     weight per model.
    /// seed : int
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    #[staticmethod]
    fn blend_by_component(
        py: Python<'_>,
        models: Vec<PyRef<'_, PyPredictiveDistribution>>,
        weights: Vec<Vec<f64>>,
        seed: u64,
    ) -> PyResult<Self> {
        let refs: Vec<&PredictiveDistribution> = models.iter().map(|m| &m.inner).collect();
        let inner = py
            .detach(|| PredictiveDistribution::blend_by_component(&refs, &weights, seed))
            .map_err(to_py)?;
        Ok(Self { inner })
    }

    /// One component's draws, or ``None`` if no component has this key.
    ///
    /// An origin period is named by its label, as a string or an integer:
    /// ``("2021",)`` or ``(2021,)`` for a year, ``("2021Q3",)`` for a
    /// quarter.
    ///
    /// Parameters
    /// ----------
    /// key : tuple
    ///
    /// Returns
    /// -------
    /// Sampled or None
    fn marginal(&self, key: Vec<KeyArg>) -> Option<PySampled> {
        self.inner
            .marginal(&key_from_py(key))
            .map(|inner| PySampled { inner })
    }

    /// Sums the components within each simulation over every dimension not
    /// in ``keep``, keeping the joint structure.
    ///
    /// Parameters
    /// ----------
    /// keep : list of str
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``keep`` names an unknown dimension or repeats one.
    fn aggregate(&self, keep: Vec<String>) -> PyResult<Self> {
        let keep: Vec<&str> = keep.iter().map(String::as_str).collect();
        let inner = self.inner.aggregate(&keep).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// The components whose value on dimension ``dim`` is one of ``keys``,
    /// with every dimension and the draws' joint structure kept.
    ///
    /// Take one part of a result this way before joining it to another:
    /// ``Tower.apply(...).select("kind", ["ceded"])`` is a treaty's cession,
    /// ready to add to a book. Aggregating over ``kind`` instead adds gross,
    /// ceded and net together.
    ///
    /// Parameters
    /// ----------
    /// dim : str
    /// keys : list of int or str
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     An unknown dimension, no keys, or a key no component has.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import PredictiveDistribution
    /// >>> pd = PredictiveDistribution(
    /// ...     ["kind", "layer"], [("gross", "ground_up"), ("ceded", "5x5"), ("net", "retained")],
    /// ...     [[10.0, 3.0, 7.0], [4.0, 0.0, 4.0]])
    /// >>> pd.select("kind", ["ceded"]).mean()
    /// 1.5
    fn select(&self, dim: &str, keys: Vec<KeyArg>) -> PyResult<Self> {
        let keys = key_from_py(keys);
        let inner = self.inner.select(dim, &keys).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Every simulation's components rescaled pro rata to a new total.
    ///
    /// Component ``k`` in simulation ``s`` becomes ``c[s][k] * totals[s] /
    /// sum_k c[s][k]`` (unchanged when the components sum to 0). This
    /// carries a total computed from the components, a loss-ratio cap over
    /// several classes or a stop loss on a mix of perils, back onto them in
    /// proportion to their shares; it is not a capital allocation. The keys
    /// and dimensions are kept.
    ///
    /// Parameters
    /// ----------
    /// totals : list of float
    ///     One total per simulation.
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Not one finite total per simulation, or a non-zero total where
    ///     the components sum to 0.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import PredictiveDistribution
    /// >>> pd = PredictiveDistribution(["class"], [("motor",), ("home",)], [[6.0, 9.0], [3.0, 1.0]])
    /// >>> pd.scale_to_totals([10.0, 4.0]).draw_matrix()
    /// [[4.0, 6.0], [3.0, 1.0]]
    fn scale_to_totals(&self, totals: Vec<f64>) -> PyResult<Self> {
        let inner = self.inner.scale_to_totals(&totals).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// The total over all components, one value per simulation.
    ///
    /// Returns
    /// -------
    /// Sampled
    fn total(&self) -> PySampled {
        PySampled {
            inner: self.inner.total().clone(),
        }
    }

    /// Mean of the total.
    ///
    /// Returns
    /// -------
    /// float
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// Variance of the total.
    ///
    /// Returns
    /// -------
    /// float
    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// Quantile of the total.
    ///
    /// Parameters
    /// ----------
    /// p : float
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

    /// Value at risk of the total at level ``p``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///
    /// Returns
    /// -------
    /// float
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``p`` is outside ``[0, 1]``.
    fn var(&self, p: f64) -> PyResult<f64> {
        self.inner.var(p).map_err(to_py)
    }

    /// Tail value at risk of the total at level ``p``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///
    /// Returns
    /// -------
    /// float
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``p`` is outside ``[0, 1]``.
    fn tvar(&self, p: f64) -> PyResult<f64> {
        self.inner.tvar(p).map_err(to_py)
    }

    /// Where this result came from: model, parameters, seed, stream scheme,
    /// samplers, crate versions and input hash.
    ///
    /// ``stream_scheme`` names how simulations map to random-number
    /// streams; ``samplers`` lists ``(family, sampler id)`` pairs, e.g.
    /// ``[("gamma", "marsaglia-tsang/2026-10")]``: the sampler table of the
    /// build that made the draws, so every sampler they may have used. A
    /// family not listed uses its first sampler (inverse transform for a
    /// distribution, the documented method for a copula's frailty). A
    /// result replays only with the same seed, stream scheme and samplers.
    /// ``samplers`` is ``None`` when not recorded: a result without draws,
    /// one computed from draws made elsewhere, or one made by a build from
    /// before samplers were recorded (2026-10-08).
    ///
    /// Returns
    /// -------
    /// dict
    fn provenance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let p = self.inner.provenance();
        let d = PyDict::new(py);
        d.set_item("model", &p.model)?;
        d.set_item("parameters", p.parameters.clone())?;
        d.set_item("seed", p.seed)?;
        d.set_item("stream_scheme", p.stream_scheme.clone())?;
        d.set_item("samplers", p.samplers.clone())?;
        d.set_item("versions", p.versions.clone())?;
        d.set_item("input_hash", p.input_hash.clone())?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "PredictiveDistribution(dims={:?}, n_sims={}, n_components={})",
            self.inner.dims(),
            self.inner.n_sims(),
            self.inner.n_components()
        )
    }
}
