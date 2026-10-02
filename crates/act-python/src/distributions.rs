//! `actuarialrs.distributions`: wrappers over `act_prob` distributions.

use act_core::StreamRng;
use act_prob::{
    ComponentKey, Counting, DiscretizationReport, Distribution, Empirical, Grid, KeyValue,
    PredictiveDistribution, Provenance, Sampled, Severity,
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
    /// >>> from actuarialrs.distributions import Lognormal
    /// >>> d = Lognormal(7.0, 0.5)
    /// >>> abs(d.layer(1000.0, 0.0) - d.lev(1000.0)) < 1e-9
    /// True
    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        self.inner.layer(limit, attachment)
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

/// A severity accepted wherever a parametric or discretized loss
/// distribution can be used.
pub(crate) enum AnySeverity {
    Lognormal(act_prob::Lognormal),
    Grid(Grid),
}

impl AnySeverity {
    pub(crate) fn extract(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(d) = obj.extract::<PyRef<'_, PyLognormal>>() {
            return Ok(Self::Lognormal(d.inner));
        }
        if let Ok(g) = obj.extract::<PyRef<'_, PyGrid>>() {
            return Ok(Self::Grid(g.inner.clone()));
        }
        Err(PyTypeError::new_err("expected a Lognormal or a Grid"))
    }
}

impl Distribution for AnySeverity {
    fn mean(&self) -> f64 {
        match self {
            Self::Lognormal(d) => d.mean(),
            Self::Grid(g) => g.mean(),
        }
    }
    fn variance(&self) -> f64 {
        match self {
            Self::Lognormal(d) => d.variance(),
            Self::Grid(g) => g.variance(),
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        match self {
            Self::Lognormal(d) => d.cdf(x),
            Self::Grid(g) => g.cdf(x),
        }
    }
    fn quantile(&self, p: f64) -> act_core::Result<f64> {
        match self {
            Self::Lognormal(d) => d.quantile(p),
            Self::Grid(g) => g.quantile(p),
        }
    }
}

impl Severity for AnySeverity {
    fn lev(&self, limit: f64) -> f64 {
        match self {
            Self::Lognormal(d) => d.lev(limit),
            Self::Grid(g) => g.lev(limit),
        }
    }
    fn stop_loss(&self, retention: f64) -> f64 {
        match self {
            Self::Lognormal(d) => d.stop_loss(retention),
            Self::Grid(g) => g.stop_loss(retention),
        }
    }
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
/// >>> from actuarialrs.distributions import Poisson
/// >>> n = Poisson(3.0)
/// >>> round(n.pmf(0), 6)
/// 0.049787
#[pyclass(name = "Poisson", module = "actuarialrs.distributions", frozen)]
pub(crate) struct PyPoisson {
    pub(crate) inner: act_prob::Poisson,
}

#[pymethods]
impl PyPoisson {
    #[new]
    fn new(lam: f64) -> PyResult<Self> {
        let inner = act_prob::Poisson::new(lam).map_err(to_py)?;
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
/// >>> from actuarialrs.distributions import NegativeBinomial
/// >>> n = NegativeBinomial.from_mean_variance(10.0, 30.0)
/// >>> round(n.variance(), 9)
/// 30.0
#[pyclass(
    name = "NegativeBinomial",
    module = "actuarialrs.distributions",
    frozen
)]
pub(crate) struct PyNegativeBinomial {
    pub(crate) inner: act_prob::NegativeBinomial,
}

#[pymethods]
impl PyNegativeBinomial {
    #[new]
    fn new(r: f64, beta: f64) -> PyResult<Self> {
        let inner = act_prob::NegativeBinomial::new(r, beta).map_err(to_py)?;
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
            act_prob::NegativeBinomial::from_mean_variance(mean, variance).map_err(to_py)?;
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
    module = "actuarialrs.distributions",
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
            act_prob::Discretization::LocalMoment => "local_moment",
            act_prob::Discretization::Rounding => "rounding",
            act_prob::Discretization::Lower => "lower",
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
/// >>> from actuarialrs.distributions import Grid, Lognormal
/// >>> grid, report = Grid.local_moment(Lognormal(7.0, 0.5), 100.0, 200)
/// >>> report.tail_mass < 1e-8
/// True
#[pyclass(name = "Grid", module = "actuarialrs.distributions", frozen)]
pub(crate) struct PyGrid {
    pub(crate) inner: Grid,
}

fn discretized(
    result: act_core::Result<(Grid, DiscretizationReport)>,
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

    /// Discretizes a severity by local moment matching on the mean.
    ///
    /// Parameters
    /// ----------
    /// severity : Lognormal or Grid
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
        let sev = AnySeverity::extract(severity)?;
        discretized(Grid::local_moment(&sev, step, points))
    }

    /// Discretizes a severity by rounding each loss to the nearest point.
    ///
    /// Parameters
    /// ----------
    /// severity : Lognormal or Grid
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
        let sev = AnySeverity::extract(severity)?;
        discretized(Grid::rounding(&sev, step, points))
    }

    /// Discretizes a severity by moving each cell's mass to its left end: a
    /// stochastic lower bound.
    ///
    /// Parameters
    /// ----------
    /// severity : Lognormal or Grid
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
        let sev = AnySeverity::extract(severity)?;
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
/// >>> from actuarialrs.distributions import Sampled
/// >>> s = Sampled([1.0, 2.0, 3.0, 4.0])
/// >>> s.tvar(0.5)
/// 3.5
#[pyclass(name = "Sampled", module = "actuarialrs.distributions", frozen)]
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

    /// Tail value at risk at level ``p``: the mean of the worst ``1 - p``.
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

fn key_to_py<'py>(py: Python<'py>, key: &ComponentKey) -> PyResult<Bound<'py, PyAny>> {
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
/// >>> from actuarialrs.distributions import PredictiveDistribution
/// >>> pd = PredictiveDistribution(["line"], [("A",), ("B",)],
/// ...                             [[0.0, 0.0], [0.0, 0.0], [0.0, 100.0], [100.0, 0.0]])
/// >>> pd.var(0.75)
/// 100.0
/// >>> pd.marginal(("A",)).var(0.75)
/// 0.0
#[pyclass(
    name = "PredictiveDistribution",
    module = "actuarialrs.distributions",
    frozen
)]
pub(crate) struct PyPredictiveDistribution {
    pub(crate) inner: PredictiveDistribution,
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

    /// One component's draws, or ``None`` if no component has this key.
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
    /// crate versions and input hash.
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
