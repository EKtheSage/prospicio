//! `prospicio.risk`: distortion risk measures, allocation, copulas and
//! Iman-Conover, over `prospicio_prob`.

use prospicio_core::StreamRng;
use prospicio_prob::capital::{Allocation, AllocationMethod};
use prospicio_prob::copula::{self, Copula};
use prospicio_prob::evt::{Gpd, PotTail};
use prospicio_prob::{
    Archimedean, ArchimedeanCopula, Distortion, Empirical, GaussianCopula, Provenance,
    StudentTCopula,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::distributions::{
    KeyArg, PyGrid, PyPredictiveDistribution, PySampled, extract_severity, key_from_py,
};
use crate::to_py;

/// A distortion risk measure: ``rho(X) = integral of g(S(x)) dx`` for a
/// concave distortion ``g`` of the survival function.
///
/// Make one with ``Distortion.tvar``, ``Distortion.wang``,
/// ``Distortion.proportional_hazard``, ``Distortion.dual_power`` or
/// ``Distortion.exponential``. Every one is coherent, and each has a
/// parameter value that gives the mean (or a limit that does).
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Sampled
/// >>> from prospicio.risk import Distortion
/// >>> x = Sampled([1.0, 2.0, 3.0, 4.0])
/// >>> Distortion.tvar(0.5).measure(x)
/// 3.5
/// >>> Distortion.tvar(0.5).weights(4)
/// [0.0, 0.0, 0.5, 0.5]
#[pyclass(name = "Distortion", module = "prospicio.risk", frozen)]
pub(crate) struct PyDistortion {
    pub(crate) inner: Distortion,
}

#[pymethods]
impl PyDistortion {
    /// Tail value at risk at level ``p``: ``g(s) = min(s / (1 - p), 1)``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///     In ``[0, 1]``.
    ///
    /// Returns
    /// -------
    /// Distortion
    #[staticmethod]
    fn tvar(p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Distortion::tvar(p).map_err(to_py)?,
        })
    }

    /// Wang transform: ``g(s) = Phi(Phi^-1(s) + lambda)``.
    ///
    /// Parameters
    /// ----------
    /// lam : float
    ///     Market price of risk, ``>= 0``.
    ///
    /// Returns
    /// -------
    /// Distortion
    #[staticmethod]
    fn wang(lam: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Distortion::wang(lam).map_err(to_py)?,
        })
    }

    /// Proportional hazard transform: ``g(s) = s**rho``.
    ///
    /// Parameters
    /// ----------
    /// rho : float
    ///     In ``(0, 1]``.
    ///
    /// Returns
    /// -------
    /// Distortion
    #[staticmethod]
    fn proportional_hazard(rho: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Distortion::proportional_hazard(rho).map_err(to_py)?,
        })
    }

    /// Dual power transform: ``g(s) = 1 - (1 - s)**beta``.
    ///
    /// Parameters
    /// ----------
    /// beta : float
    ///     ``>= 1``.
    ///
    /// Returns
    /// -------
    /// Distortion
    #[staticmethod]
    fn dual_power(beta: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Distortion::dual_power(beta).map_err(to_py)?,
        })
    }

    /// Exponential spectral measure: ``g(s) = (1 - exp(-k s)) / (1 - exp(-k))``,
    /// risk aversion that grows exponentially towards the worst outcomes.
    ///
    /// Parameters
    /// ----------
    /// k : float
    ///     Risk aversion, positive; the mean as ``k -> 0``.
    ///
    /// Returns
    /// -------
    /// Distortion
    #[staticmethod]
    fn exponential(k: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Distortion::exponential(k).map_err(to_py)?,
        })
    }

    /// The distortion ``g(s)`` of a survival probability ``s``.
    ///
    /// Parameters
    /// ----------
    /// s : float
    ///
    /// Returns
    /// -------
    /// float
    fn g(&self, s: f64) -> f64 {
        self.inner.g(s)
    }

    /// Weights for ``n`` equally likely values sorted ascending.
    ///
    /// Parameters
    /// ----------
    /// n : int
    ///
    /// Returns
    /// -------
    /// list of float
    ///     Non-negative, summing to 1.
    fn weights(&self, n: usize) -> Vec<f64> {
        self.inner.weights(n)
    }

    /// The risk measure of a distribution.
    ///
    /// Parameters
    /// ----------
    /// dist : Sampled, Grid or PredictiveDistribution
    ///     A predictive distribution is measured on its total.
    ///
    /// Returns
    /// -------
    /// float
    fn measure(&self, dist: &Bound<'_, PyAny>) -> PyResult<f64> {
        if let Ok(s) = dist.extract::<PyRef<'_, PySampled>>() {
            return Ok(s.inner.distortion(&self.inner));
        }
        if let Ok(g) = dist.extract::<PyRef<'_, PyGrid>>() {
            return Ok(g.inner.distortion(&self.inner));
        }
        if let Ok(p) = dist.extract::<PyRef<'_, PyPredictiveDistribution>>() {
            return Ok(p.inner.distortion(&self.inner));
        }
        Err(PyTypeError::new_err(
            "expected a Sampled, Grid or PredictiveDistribution",
        ))
    }

    fn __repr__(&self) -> String {
        match self.inner {
            Distortion::Tvar(p) => format!("Distortion.tvar({p:?})"),
            Distortion::Wang(l) => format!("Distortion.wang({l:?})"),
            Distortion::ProportionalHazard(r) => format!("Distortion.proportional_hazard({r:?})"),
            Distortion::DualPower(b) => format!("Distortion.dual_power({b:?})"),
            Distortion::Exponential(k) => format!("Distortion.exponential({k:?})"),
        }
    }
}

/// Allocates a distortion risk measure of the total to the components.
///
/// Euler allocation by co-measure: simulations are ranked by their total
/// and each component gets the distortion-weighted sum of its own draws.
/// The contributions sum to ``distortion.measure(pd)``; for
/// ``Distortion.tvar(p)`` they are the CoTVaRs. Components must add up to
/// the portfolio being allocated.
///
/// Parameters
/// ----------
/// pd : PredictiveDistribution
/// distortion : Distortion
///
/// Returns
/// -------
/// list of float
///     One contribution per component, in ``pd.components()`` order.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import PredictiveDistribution
/// >>> from prospicio.risk import Distortion, allocate
/// >>> pd = PredictiveDistribution(["lob"], [("motor",), ("property",)],
/// ...                             [[1.0, 2.0], [4.0, 1.0], [2.0, 5.0], [3.0, 6.0]])
/// >>> allocate(pd, Distortion.tvar(0.5))
/// [2.5, 5.5]
#[pyfunction]
pub(crate) fn allocate(
    py: Python<'_>,
    pd: PyRef<'_, PyPredictiveDistribution>,
    distortion: PyRef<'_, PyDistortion>,
) -> Vec<f64> {
    let (pd, d) = (&pd.inner, distortion.inner);
    py.detach(|| pd.allocate(&d))
}

/// The draws of a ``Sampled``, the total of a ``PredictiveDistribution``,
/// or a list of floats.
fn draws_of(dist: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(s) = dist.extract::<PyRef<'_, PySampled>>() {
        return Ok(s.inner.draws().to_vec());
    }
    if let Ok(p) = dist.extract::<PyRef<'_, PyPredictiveDistribution>>() {
        return Ok(p.inner.total().draws().to_vec());
    }
    dist.extract::<Vec<f64>>().map_err(|_| {
        PyTypeError::new_err("expected a Sampled, PredictiveDistribution or list of float")
    })
}

/// Entropic risk measure ``(1 / theta) log E[exp(theta X)]``: the certainty
/// equivalent of a loss under exponential utility.
///
/// It rises from the mean (``theta -> 0``) to the largest draw
/// (``theta -> inf``); for a normal loss it is ``mu + theta sigma**2 / 2``.
///
/// Parameters
/// ----------
/// dist : Sampled, PredictiveDistribution or list of float
///     A predictive distribution is measured on its total.
/// theta : float
///     Risk aversion, positive.
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> import math
/// >>> from prospicio.risk import entropic
/// >>> round(entropic([0.0, 1.0], math.log(2.0)), 12) == round(math.log2(1.5), 12)
/// True
#[pyfunction]
pub(crate) fn entropic(dist: &Bound<'_, PyAny>, theta: f64) -> PyResult<f64> {
    prospicio_prob::risk::entropic(&draws_of(dist)?, theta).map_err(to_py)
}

/// Esscher premium ``E[X exp(h X)] / E[exp(h X)]``: the mean after tilting
/// probability towards large losses.
///
/// The mean at ``h = 0``; ``mu + h sigma**2`` for a normal loss.
///
/// Parameters
/// ----------
/// dist : Sampled, PredictiveDistribution or list of float
///     A predictive distribution is measured on its total.
/// h : float
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> import math
/// >>> from prospicio.risk import esscher
/// >>> round(esscher([0.0, 1.0], math.log(3.0)), 12)
/// 0.75
#[pyfunction]
pub(crate) fn esscher(dist: &Bound<'_, PyAny>, h: f64) -> PyResult<f64> {
    prospicio_prob::risk::esscher(&draws_of(dist)?, h).map_err(to_py)
}

/// Marginal expected shortfall of each component at level ``p``: its mean
/// over the simulations where the total is in its worst ``1 - p``.
///
/// The same as ``allocate(pd, Distortion.tvar(p))``; it sums to the
/// total's TVaR.
///
/// Parameters
/// ----------
/// pd : PredictiveDistribution
/// p : float
///
/// Returns
/// -------
/// list of float
///     One per component, in ``pd.components()`` order.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import PredictiveDistribution
/// >>> from prospicio.risk import marginal_expected_shortfall
/// >>> pd = PredictiveDistribution(["lob"], [("motor",), ("property",)],
/// ...                             [[1.0, 2.0], [4.0, 1.0], [2.0, 5.0], [3.0, 6.0]])
/// >>> marginal_expected_shortfall(pd, 0.5)
/// [2.5, 5.5]
#[pyfunction]
pub(crate) fn marginal_expected_shortfall(
    py: Python<'_>,
    pd: PyRef<'_, PyPredictiveDistribution>,
    p: f64,
) -> PyResult<Vec<f64>> {
    let pd = &pd.inner;
    py.detach(|| pd.marginal_expected_shortfall(p))
        .map_err(to_py)
}

/// CoVaR of a component: the total's VaR at level ``q`` over the
/// simulations where the component is at or above its own VaR at ``p``.
///
/// Compare it with the total's unconditional VaR at ``q`` to see how much
/// one segment's bad years drag the portfolio.
///
/// Parameters
/// ----------
/// pd : PredictiveDistribution
/// key : tuple
///     The component's key.
/// p : float
///     The component's distress level.
/// q : float
///     The level of the total's VaR.
///
/// Returns
/// -------
/// float
///
/// Raises
/// ------
/// ValueError
///     If there is no component ``key``.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import PredictiveDistribution
/// >>> from prospicio.risk import covar
/// >>> pd = PredictiveDistribution(["lob"], [("a",), ("b",)],
/// ...                             [[1.0, 0.0], [2.0, 1.0], [3.0, 5.0], [4.0, 1.0]])
/// >>> covar(pd, ("a",), 0.75, 0.5)
/// 5.0
#[pyfunction]
pub(crate) fn covar(
    py: Python<'_>,
    pd: PyRef<'_, PyPredictiveDistribution>,
    key: Vec<KeyArg>,
    p: f64,
    q: f64,
) -> PyResult<f64> {
    let (pd, key) = (&pd.inner, key_from_py(key));
    py.detach(|| pd.covar(&key, p, q)).map_err(to_py)
}

/// Esscher allocation: each component's mean under the Esscher transform
/// of the total, ``E[X_j exp(h S)] / E[exp(h S)]``.
///
/// The contributions sum to ``esscher(pd, h)``; at ``h = 0`` they are the
/// means.
///
/// Parameters
/// ----------
/// pd : PredictiveDistribution
/// h : float
///
/// Returns
/// -------
/// list of float
///     One per component, in ``pd.components()`` order.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import PredictiveDistribution
/// >>> from prospicio.risk import esscher_allocation
/// >>> pd = PredictiveDistribution(["lob"], [("motor",), ("property",)],
/// ...                             [[1.0, 2.0], [4.0, 1.0], [2.0, 5.0], [3.0, 6.0]])
/// >>> esscher_allocation(pd, 0.0)
/// [2.5, 3.5]
#[pyfunction]
pub(crate) fn esscher_allocation(
    py: Python<'_>,
    pd: PyRef<'_, PyPredictiveDistribution>,
    h: f64,
) -> PyResult<Vec<f64>> {
    let pd = &pd.inner;
    py.detach(|| pd.esscher_allocation(h)).map_err(to_py)
}

/// Capital allocation of a distortion risk measure, from ``capital``.
#[pyclass(name = "Allocation", module = "prospicio.risk", frozen)]
pub(crate) struct PyAllocation {
    inner: Allocation,
}

#[pymethods]
impl PyAllocation {
    /// The allocation method, as passed to ``capital``.
    #[getter]
    fn method(&self) -> &'static str {
        method_name(self.inner.method)
    }

    /// The portfolio's measure, ``rho(S)``.
    #[getter]
    fn total(&self) -> f64 {
        self.inner.total
    }

    /// Stand-alone measure ``rho(X_j)`` of each component.
    #[getter]
    fn standalone(&self) -> Vec<f64> {
        self.inner.standalone.clone()
    }

    /// Allocated capital of each component.
    #[getter]
    fn allocated(&self) -> Vec<f64> {
        self.inner.allocated.clone()
    }

    /// ``sum(standalone) - total``: the capital saved by holding the
    /// components together.
    ///
    /// Returns
    /// -------
    /// float
    fn diversification_benefit(&self) -> f64 {
        self.inner.diversification_benefit()
    }

    /// ``standalone - allocated`` per component: each one's share of the
    /// diversification benefit.
    ///
    /// Returns
    /// -------
    /// list of float
    fn diversification(&self) -> Vec<f64> {
        self.inner.diversification()
    }

    fn __repr__(&self) -> String {
        format!(
            "Allocation(method={:?}, total={}, allocated={:?})",
            self.method(),
            self.inner.total,
            self.inner.allocated
        )
    }
}

const METHODS: [(&str, AllocationMethod); 5] = [
    ("euler", AllocationMethod::Euler),
    ("covariance", AllocationMethod::Covariance),
    ("proportional", AllocationMethod::Proportional),
    ("marginal", AllocationMethod::Marginal),
    ("shapley", AllocationMethod::Shapley),
];

fn method_name(method: AllocationMethod) -> &'static str {
    METHODS
        .iter()
        .find(|(_, m)| *m == method)
        .map(|(name, _)| *name)
        .expect("every method has a name")
}

/// Allocates the distortion risk measure of a portfolio's total to its
/// components.
///
/// Methods:
///
/// - ``"euler"``: co-measure (for TVaR, the CoTVaRs); consistent with
///   marginal changes to the portfolio.
/// - ``"covariance"``: ``rho(S) Cov(X_j, S) / Var(S)``.
/// - ``"proportional"``: stand-alone measures scaled to ``rho(S)``.
/// - ``"marginal"``: ``rho(S) - rho(S - X_j)`` (Merton-Perold); does not
///   add up to ``rho(S)``.
/// - ``"shapley"``: Shapley value of ``v(T) = rho(sum of T)``; at most 12
///   components.
///
/// Parameters
/// ----------
/// pd : PredictiveDistribution
///     Components that add up to the portfolio.
/// distortion : Distortion
/// method : str, default "euler"
///
/// Returns
/// -------
/// Allocation
///
/// Raises
/// ------
/// ValueError
///     For an unknown method, a constant total (``"covariance"``),
///     stand-alone measures summing to 0 (``"proportional"``) or more than
///     12 components (``"shapley"``).
///
/// Examples
/// --------
/// >>> from prospicio.distributions import PredictiveDistribution
/// >>> from prospicio.risk import Distortion, capital
/// >>> pd = PredictiveDistribution(["lob"], [("motor",), ("property",)],
/// ...                             [[1.0, 2.0], [4.0, 1.0], [2.0, 5.0], [3.0, 6.0]])
/// >>> a = capital(pd, Distortion.tvar(0.5))
/// >>> a.total, a.standalone, a.allocated
/// (8.0, [3.5, 5.5], [2.5, 5.5])
/// >>> a.diversification_benefit()
/// 1.0
#[pyfunction]
#[pyo3(signature = (pd, distortion, method = "euler"))]
pub(crate) fn capital(
    py: Python<'_>,
    pd: PyRef<'_, PyPredictiveDistribution>,
    distortion: PyRef<'_, PyDistortion>,
    method: &str,
) -> PyResult<PyAllocation> {
    let method = METHODS
        .iter()
        .find(|(name, _)| *name == method)
        .map(|(_, m)| *m)
        .ok_or_else(|| {
            PyValueError::new_err(
                "method must be one of euler, covariance, proportional, marginal, shapley",
            )
        })?;
    let (pd, d) = (&pd.inner, distortion.inner);
    let inner = py.detach(|| pd.capital(&d, method)).map_err(to_py)?;
    Ok(PyAllocation { inner })
}

/// One of the copula classes, as a Rust copula.
enum AnyCopula {
    Gaussian(GaussianCopula),
    StudentT(StudentTCopula),
    Archimedean(ArchimedeanCopula),
}

impl AnyCopula {
    fn extract(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(c) = obj.extract::<PyRef<'_, PyGaussianCopula>>() {
            return Ok(Self::Gaussian(c.inner.clone()));
        }
        if let Ok(c) = obj.extract::<PyRef<'_, PyStudentTCopula>>() {
            return Ok(Self::StudentT(c.inner.clone()));
        }
        if let Ok(c) = obj.extract::<PyRef<'_, PyArchimedeanCopula>>() {
            return Ok(Self::Archimedean(c.inner.clone()));
        }
        Err(PyTypeError::new_err(
            "expected a GaussianCopula, StudentTCopula or ArchimedeanCopula",
        ))
    }

    fn as_copula(&self) -> &dyn Copula {
        match self {
            Self::Gaussian(c) => c,
            Self::StudentT(c) => c,
            Self::Archimedean(c) => c,
        }
    }
}

/// ``n`` draws of uniforms; draw ``i`` uses stream ``i`` of ``seed``.
fn sample_rows(c: &dyn Copula, n: usize, seed: u64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let mut u = vec![0.0; c.dim()];
            c.sample(&mut StreamRng::new(seed, i as u64), &mut u);
            u
        })
        .collect()
}

/// A correlation matrix from nested lists, row-major.
fn flatten_square(m: Vec<Vec<f64>>) -> PyResult<(Vec<f64>, usize)> {
    let d = m.len();
    if m.iter().any(|row| row.len() != d) {
        return Err(PyValueError::new_err("correlation must be a square matrix"));
    }
    Ok((m.into_iter().flatten().collect(), d))
}

/// The Gaussian copula with correlation matrix ``correlation``.
///
/// Parameters
/// ----------
/// correlation : list of list of float
///     Symmetric, unit diagonal, positive definite.
///
/// Raises
/// ------
/// ValueError
///     If the matrix is not a valid correlation matrix.
///
/// Examples
/// --------
/// >>> from prospicio.risk import GaussianCopula
/// >>> c = GaussianCopula([[1.0, 0.5], [0.5, 1.0]])
/// >>> u = c.sample(3, seed=1)
/// >>> len(u), all(0.0 < x < 1.0 for row in u for x in row)
/// (3, True)
#[pyclass(name = "GaussianCopula", module = "prospicio.risk", frozen)]
pub(crate) struct PyGaussianCopula {
    inner: GaussianCopula,
}

#[pymethods]
impl PyGaussianCopula {
    #[new]
    fn new(correlation: Vec<Vec<f64>>) -> PyResult<Self> {
        let (r, d) = flatten_square(correlation)?;
        Ok(Self {
            inner: GaussianCopula::new(&r, d).map_err(to_py)?,
        })
    }

    /// Number of dimensions.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// ``n`` draws of uniforms; draw ``i`` uses stream ``i`` of ``seed``.
    ///
    /// Parameters
    /// ----------
    /// n : int
    /// seed : int
    ///
    /// Returns
    /// -------
    /// list of list of float
    fn sample(&self, py: Python<'_>, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let c = &self.inner;
        py.detach(|| sample_rows(c, n, seed))
    }
}

/// The Student t copula with correlation matrix ``correlation`` and ``nu``
/// degrees of freedom: Gaussian-like correlation with joint extremes.
///
/// Parameters
/// ----------
/// correlation : list of list of float
///     Symmetric, unit diagonal, positive definite.
/// nu : float
///     Degrees of freedom, positive.
///
/// Raises
/// ------
/// ValueError
///     If the matrix or ``nu`` is invalid.
///
/// Examples
/// --------
/// >>> from prospicio.risk import StudentTCopula
/// >>> StudentTCopula([[1.0, 0.5], [0.5, 1.0]], 4.0).dim
/// 2
#[pyclass(name = "StudentTCopula", module = "prospicio.risk", frozen)]
pub(crate) struct PyStudentTCopula {
    inner: StudentTCopula,
}

#[pymethods]
impl PyStudentTCopula {
    #[new]
    fn new(correlation: Vec<Vec<f64>>, nu: f64) -> PyResult<Self> {
        let (r, d) = flatten_square(correlation)?;
        Ok(Self {
            inner: StudentTCopula::new(&r, d, nu).map_err(to_py)?,
        })
    }

    /// Number of dimensions.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Degrees of freedom.
    #[getter]
    fn nu(&self) -> f64 {
        self.inner.nu()
    }

    /// ``n`` draws of uniforms; draw ``i`` uses stream ``i`` of ``seed``.
    ///
    /// Parameters
    /// ----------
    /// n : int
    /// seed : int
    ///
    /// Returns
    /// -------
    /// list of list of float
    fn sample(&self, py: Python<'_>, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let c = &self.inner;
        py.detach(|| sample_rows(c, n, seed))
    }
}

/// An exchangeable Archimedean copula: Clayton, Gumbel, Frank or Joe.
///
/// Parameters
/// ----------
/// family : {"clayton", "gumbel", "frank", "joe"}
/// theta : float
///     Positive for Clayton and Frank; at least 1 for Gumbel and Joe.
/// dim : int
///
/// Raises
/// ------
/// ValueError
///     If the family is unknown or ``theta`` is out of range.
///
/// Examples
/// --------
/// >>> from prospicio.risk import ArchimedeanCopula
/// >>> c = ArchimedeanCopula("clayton", 2.0, 3)  # Kendall's tau 0.5
/// >>> c.dim, c.family
/// (3, 'clayton')
#[pyclass(name = "ArchimedeanCopula", module = "prospicio.risk", frozen)]
pub(crate) struct PyArchimedeanCopula {
    inner: ArchimedeanCopula,
}

#[pymethods]
impl PyArchimedeanCopula {
    #[new]
    fn new(family: &str, theta: f64, dim: usize) -> PyResult<Self> {
        let family = match family {
            "clayton" => Archimedean::Clayton,
            "gumbel" => Archimedean::Gumbel,
            "frank" => Archimedean::Frank,
            "joe" => Archimedean::Joe,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown family {other:?}: use \"clayton\", \"gumbel\", \"frank\" or \"joe\""
                )));
            }
        };
        Ok(Self {
            inner: ArchimedeanCopula::new(family, theta, dim).map_err(to_py)?,
        })
    }

    /// Number of dimensions.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Family name.
    #[getter]
    fn family(&self) -> &'static str {
        match self.inner.family() {
            Archimedean::Clayton => "clayton",
            Archimedean::Gumbel => "gumbel",
            Archimedean::Frank => "frank",
            Archimedean::Joe => "joe",
        }
    }

    /// Copula parameter.
    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta()
    }

    /// ``n`` draws of uniforms; draw ``i`` uses stream ``i`` of ``seed``.
    ///
    /// Parameters
    /// ----------
    /// n : int
    /// seed : int
    ///
    /// Returns
    /// -------
    /// list of list of float
    fn sample(&self, py: Python<'_>, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let c = &self.inner;
        py.detach(|| sample_rows(c, n, seed))
    }
}

/// Simulates marginals joined by a copula.
///
/// In simulation ``i``, draws uniforms from ``copula`` with stream ``i`` of
/// ``seed`` and applies each marginal's quantile function.
///
/// Parameters
/// ----------
/// copula : GaussianCopula, StudentTCopula or ArchimedeanCopula
/// marginals : list of Lognormal, Grid or Pareto-family severities
///     One per copula dimension.
/// n_sims : int
/// seed : int
/// keys : list of tuple, optional
///     One component key per marginal; defaults to ``(0,), (1,), ...``.
/// dims : list of str, default ["component"]
///
/// Returns
/// -------
/// PredictiveDistribution
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Lognormal
/// >>> from prospicio.risk import GaussianCopula, simulate
/// >>> c = GaussianCopula([[1.0, 0.4], [0.4, 1.0]])
/// >>> pd = simulate(c, [Lognormal.from_mean_cv(100.0, 0.2), Lognormal.from_mean_cv(50.0, 1.0)],
/// ...               10_000, 42, keys=[("motor",), ("property",)], dims=["lob"])
/// >>> abs(pd.mean() - 150.0) < 3.0
/// True
#[pyfunction]
#[pyo3(signature = (copula, marginals, n_sims, seed, keys = None, dims = None))]
pub(crate) fn simulate(
    py: Python<'_>,
    copula: &Bound<'_, PyAny>,
    marginals: Vec<Bound<'_, PyAny>>,
    n_sims: usize,
    seed: u64,
    keys: Option<Vec<Vec<KeyArg>>>,
    dims: Option<Vec<String>>,
) -> PyResult<PyPredictiveDistribution> {
    let copula = AnyCopula::extract(copula)?;
    let marginals: Vec<prospicio_prob::SeverityDist> = marginals
        .iter()
        .map(extract_severity)
        .collect::<PyResult<_>>()?;
    let components = match keys {
        Some(keys) => keys.into_iter().map(key_from_py).collect(),
        None => (0..marginals.len())
            .map(|j| vec![prospicio_prob::KeyValue::Int(j as i64)])
            .collect(),
    };
    let dims = dims.unwrap_or_else(|| vec!["component".into()]);
    let inner = py
        .detach(|| {
            let refs: Vec<&(dyn prospicio_prob::Distribution + Sync)> = marginals
                .iter()
                .map(|m| m as &(dyn prospicio_prob::Distribution + Sync))
                .collect();
            copula::simulate(
                copula.as_copula(),
                &refs,
                dims,
                components,
                n_sims,
                seed,
                Provenance::new("copula"),
            )
        })
        .map_err(to_py)?;
    Ok(PyPredictiveDistribution { inner })
}

/// Reorders each component's draws to a target correlation (Iman-Conover).
///
/// Every component keeps exactly its own draws; only their pairing across
/// simulations changes. The correlation of the result's normal scores is
/// close to ``correlation``, and Spearman's rho close to
/// ``(6 / pi) asin(correlation / 2)``.
///
/// Parameters
/// ----------
/// pd : PredictiveDistribution
/// correlation : list of list of float
///     One row and column per component.
/// seed : int
///
/// Returns
/// -------
/// PredictiveDistribution
///
/// Examples
/// --------
/// >>> from prospicio.distributions import PredictiveDistribution
/// >>> from prospicio.risk import iman_conover
/// >>> rows = [[float(i), float((i * 7919) % 1000)] for i in range(1000)]
/// >>> pd = PredictiveDistribution(["lob"], [(0,), (1,)], rows)
/// >>> joined = iman_conover(pd, [[1.0, 0.7], [0.7, 1.0]], seed=3)
/// >>> sorted(joined.marginal((1,)).draws) == sorted(pd.marginal((1,)).draws)
/// True
#[pyfunction]
pub(crate) fn iman_conover(
    py: Python<'_>,
    pd: PyRef<'_, PyPredictiveDistribution>,
    correlation: Vec<Vec<f64>>,
    seed: u64,
) -> PyResult<PyPredictiveDistribution> {
    let (r, _) = flatten_square(correlation)?;
    let pd = &pd.inner;
    let inner = py
        .detach(|| copula::iman_conover(pd, &r, seed))
        .map_err(to_py)?;
    Ok(PyPredictiveDistribution { inner })
}

/// The generalized Pareto distribution, as SciPy's
/// ``genpareto(c=xi, scale=beta)``.
///
/// ``P(X > x) = (1 + xi x / beta)**(-1 / xi)`` for ``x >= 0``.
///
/// Parameters
/// ----------
/// xi : float
///     Shape; moments of order ``1 / xi`` and above are infinite.
/// beta : float
///     Scale, positive.
///
/// Examples
/// --------
/// >>> from prospicio.risk import Gpd
/// >>> g = Gpd(0.5, 2.0)
/// >>> g.mean()
/// 4.0
/// >>> fit = Gpd.fit([g.quantile((i - 0.5) / 1000) for i in range(1, 1001)])
/// >>> round(fit.xi, 2), round(fit.beta, 2)
/// (0.5, 2.0)
#[pyclass(name = "Gpd", module = "prospicio.risk", frozen)]
pub(crate) struct PyGpd {
    inner: Gpd,
}

#[pymethods]
impl PyGpd {
    #[new]
    fn new(xi: f64, beta: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Gpd::new(xi, beta).map_err(to_py)?,
        })
    }

    /// Maximum likelihood fit to exceedances (values over a threshold,
    /// minus the threshold).
    ///
    /// Parameters
    /// ----------
    /// exceedances : list of float
    ///     At least 3, non-negative, not all equal.
    ///
    /// Returns
    /// -------
    /// Gpd
    #[staticmethod]
    fn fit(py: Python<'_>, exceedances: Vec<f64>) -> PyResult<Self> {
        let inner = py.detach(|| Gpd::fit(&exceedances)).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Shape.
    #[getter]
    fn xi(&self) -> f64 {
        self.inner.xi()
    }

    /// Scale.
    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta()
    }

    /// Mean, ``beta / (1 - xi)``; infinite for ``xi >= 1``.
    fn mean(&self) -> f64 {
        prospicio_prob::Distribution::mean(&self.inner)
    }

    /// Distribution function.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///
    /// Returns
    /// -------
    /// float
    fn cdf(&self, x: f64) -> f64 {
        prospicio_prob::Distribution::cdf(&self.inner, x)
    }

    /// Quantile function.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///
    /// Returns
    /// -------
    /// float
    fn quantile(&self, p: f64) -> PyResult<f64> {
        prospicio_prob::Distribution::quantile(&self.inner, p).map_err(to_py)
    }

    fn __repr__(&self) -> String {
        format!("Gpd({:?}, {:?})", self.inner.xi(), self.inner.beta())
    }
}

/// A peaks-over-threshold tail: draws above a threshold modelled by a
/// fitted generalized Pareto distribution, for VaR and TVaR beyond the
/// draws.
///
/// Make one with ``PotTail.fit(draws, level)``, which takes the threshold
/// at the empirical ``level`` quantile.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Lognormal, Sampled
/// >>> from prospicio.risk import PotTail
/// >>> d = Lognormal(0.0, 1.0)
/// >>> s = Sampled([d.quantile((i - 0.5) / 100_000) for i in range(1, 100_001)])
/// >>> tail = PotTail.fit(s, 0.95)
/// >>> abs(tail.var(0.999) / d.quantile(0.999) - 1) < 0.02
/// True
#[pyclass(name = "PotTail", module = "prospicio.risk", frozen)]
pub(crate) struct PyPotTail {
    inner: PotTail,
}

#[pymethods]
impl PyPotTail {
    /// Fits a tail to the draws above their empirical ``level`` quantile.
    ///
    /// Parameters
    /// ----------
    /// draws : Sampled
    /// level : float
    ///     For example 0.95 for the top 5%.
    ///
    /// Returns
    /// -------
    /// PotTail
    #[staticmethod]
    fn fit(py: Python<'_>, draws: PyRef<'_, PySampled>, level: f64) -> PyResult<Self> {
        let s = &draws.inner;
        let inner = py.detach(|| PotTail::fit(s, level)).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Threshold ``u``.
    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold()
    }

    /// Share of draws above the threshold.
    #[getter]
    fn p_exceed(&self) -> f64 {
        self.inner.p_exceed()
    }

    /// The fitted GPD for the exceedances.
    #[getter]
    fn gpd(&self) -> PyGpd {
        PyGpd {
            inner: *self.inner.gpd(),
        }
    }

    /// VaR at ``p >= 1 - p_exceed``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///
    /// Returns
    /// -------
    /// float
    fn var(&self, p: f64) -> PyResult<f64> {
        self.inner.var(p).map_err(to_py)
    }

    /// TVaR at ``p >= 1 - p_exceed``; infinite when ``xi >= 1``.
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

    fn __repr__(&self) -> String {
        format!(
            "PotTail(threshold={:?}, p_exceed={:?}, gpd={})",
            self.inner.threshold(),
            self.inner.p_exceed(),
            self.gpd().__repr__()
        )
    }
}

/// The empirical mean-excess function ``e(u) = E[X - u | X > u]`` at each
/// threshold, linear above a threshold where a GPD fits.
///
/// Parameters
/// ----------
/// draws : list of float
/// thresholds : list of float
///
/// Returns
/// -------
/// list of (float, float, int)
///     ``(u, e(u), number of draws above u)``; ``e(u)`` is NaN when none are.
///
/// Examples
/// --------
/// >>> from prospicio.risk import mean_excess
/// >>> mean_excess([1.0, 2.0, 3.0, 4.0], [2.0])
/// [(2.0, 1.5, 2)]
#[pyfunction]
pub(crate) fn mean_excess(draws: Vec<f64>, thresholds: Vec<f64>) -> Vec<(f64, f64, usize)> {
    prospicio_prob::evt::mean_excess(&draws, &thresholds)
}

/// Hill estimates of the tail index ``xi`` (``1 / alpha``) from the ``k``
/// largest draws, for each ``k``.
///
/// Parameters
/// ----------
/// draws : list of float
/// ks : list of int
///
/// Returns
/// -------
/// list of float
///
/// Raises
/// ------
/// ValueError
///     If a ``k`` is 0 or not below the number of draws, or the ``k + 1``
///     largest draws are not all positive.
#[pyfunction]
pub(crate) fn hill(draws: Vec<f64>, ks: Vec<usize>) -> PyResult<Vec<f64>> {
    prospicio_prob::evt::hill(&draws, &ks).map_err(to_py)
}
