//! `prospicio.risk`: distortion risk measures, allocation, copulas and
//! Iman-Conover, over `prospicio_prob`.

use prospicio_core::StreamRng;
use prospicio_prob::capital::{Allocation, AllocationMethod, LevelAllocation, LevelMeasure};
use prospicio_prob::copula::{self, Copula};
use prospicio_prob::evt::{Gpd, PotTail};
use prospicio_prob::{
    Archimedean, ArchimedeanCopula, Distortion, Empirical, Family, GaussianCopula, Provenance,
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
/// ``Distortion.proportional_hazard``, ``Distortion.dual_power``,
/// ``Distortion.exponential``, ``Distortion.ccoc`` (constant cost of
/// capital), ``Distortion.bitvar``, ``Distortion.weighted_tvar``,
/// ``Distortion.capped_linear``, ``Distortion.capped_log_linear``,
/// ``Distortion.lep``, ``Distortion.linear_yield``, ``Distortion.beta``,
/// ``Distortion.mixture``, ``Distortion.minimum`` or ``Distortion.convex``.
/// Every one is concave, so every measure is coherent. ``calibrate``
/// solves for the parameter that gives a target price.
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

    /// Constant cost of capital ``r``: ``g(s) = min(1, d + (1 - d) s)`` for
    /// ``s > 0``, ``d = r / (1 + r)``. The price is ``(E[X] + r max X) / (1 + r)``.
    ///
    /// Parameters
    /// ----------
    /// r : float
    ///     Return on capital, ``>= 0``.
    ///
    /// Returns
    /// -------
    /// Distortion
    #[staticmethod]
    fn ccoc(r: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Distortion::ccoc(r).map_err(to_py)?,
        })
    }

    /// Weight ``1 - w`` on ``TVaR(p0)`` and ``w`` on ``TVaR(p1)``.
    ///
    /// Parameters
    /// ----------
    /// p0, p1 : float
    ///     Levels with ``0 <= p0 <= p1 <= 1``.
    /// w : float
    ///     Weight on ``TVaR(p1)``, in ``[0, 1]``.
    ///
    /// Returns
    /// -------
    /// Distortion
    #[staticmethod]
    fn bitvar(p0: f64, p1: f64, w: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Distortion::bitvar(p0, p1, w).map_err(to_py)?,
        })
    }

    /// A weighted average of TVaRs.
    ///
    /// Parameters
    /// ----------
    /// ps : list of float
    ///     Strictly ascending levels in ``[0, 1]``.
    /// wts : list of float
    ///     Non-negative weights summing to 1.
    ///
    /// Returns
    /// -------
    /// Distortion
    #[staticmethod]
    fn weighted_tvar(ps: Vec<f64>, wts: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: Distortion::weighted_tvar(ps, wts).map_err(to_py)?,
        })
    }

    /// Capped linear: ``g(s) = min(1, r0 + slope s)`` for ``s > 0``.
    ///
    /// Parameters
    /// ----------
    /// r0 : float
    ///     Mass on the largest outcome, in ``[0, 1)``.
    /// slope : float
    ///     With ``r0 + slope >= 1``.
    ///
    /// Returns
    /// -------
    /// Distortion
    #[staticmethod]
    fn capped_linear(r0: f64, slope: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Distortion::capped_linear(r0, slope).map_err(to_py)?,
        })
    }

    /// Capped log-linear: ``g(s) = min(1, exp(r0) s**b)``.
    ///
    /// Parameters
    /// ----------
    /// r0 : float
    ///     ``>= 0``.
    /// b : float
    ///     In ``(0, 1]``.
    ///
    /// Returns
    /// -------
    /// Distortion
    #[staticmethod]
    fn capped_log_linear(r0: f64, b: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Distortion::capped_log_linear(r0, b).map_err(to_py)?,
        })
    }

    /// Leverage-equivalent pricing:
    /// ``g(s) = min(1, d + (1 - d) s + (delta - d) sqrt(s (1 - s)))`` with
    /// ``d = r0 / (1 + r0)`` and ``delta = r / (1 + r)``.
    ///
    /// Parameters
    /// ----------
    /// r0 : float
    ///     Minimum rate on line, ``>= 0``.
    /// r : float
    ///     Target return, ``>= r0``.
    ///
    /// Returns
    /// -------
    /// Distortion
    #[staticmethod]
    fn lep(r0: f64, r: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Distortion::lep(r0, r).map_err(to_py)?,
        })
    }

    /// Linear yield: ``g(s) = (r0 + (1 + r) s) / (1 + r0 + r s)``.
    ///
    /// Parameters
    /// ----------
    /// r0, r : float
    ///     Both ``>= 0``.
    ///
    /// Returns
    /// -------
    /// Distortion
    #[staticmethod]
    fn linear_yield(r0: f64, r: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Distortion::linear_yield(r0, r).map_err(to_py)?,
        })
    }

    /// The Beta(a, b) distribution function, concave for ``a <= 1 <= b``.
    ///
    /// Parameters
    /// ----------
    /// a : float
    ///     In ``(0, 1]``.
    /// b : float
    ///     ``>= 1``.
    ///
    /// Returns
    /// -------
    /// Distortion
    #[staticmethod]
    fn beta(a: f64, b: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Distortion::beta(a, b).map_err(to_py)?,
        })
    }

    /// A weighted average of distortions.
    ///
    /// Parameters
    /// ----------
    /// distortions : list of Distortion
    /// weights : list of float
    ///     Non-negative, summing to 1.
    ///
    /// Returns
    /// -------
    /// Distortion
    #[staticmethod]
    fn mixture(distortions: Vec<PyRef<'_, PyDistortion>>, weights: Vec<f64>) -> PyResult<Self> {
        if distortions.len() != weights.len() {
            return Err(PyValueError::new_err("give one weight per distortion"));
        }
        let parts = weights
            .into_iter()
            .zip(distortions.iter().map(|d| d.inner.clone()))
            .collect();
        Ok(Self {
            inner: Distortion::mixture(parts).map_err(to_py)?,
        })
    }

    /// The pointwise minimum of distortions.
    ///
    /// Parameters
    /// ----------
    /// distortions : list of Distortion
    ///
    /// Returns
    /// -------
    /// Distortion
    #[staticmethod]
    fn minimum(distortions: Vec<PyRef<'_, PyDistortion>>) -> PyResult<Self> {
        let parts = distortions.iter().map(|d| d.inner.clone()).collect();
        Ok(Self {
            inner: Distortion::minimum(parts).map_err(to_py)?,
        })
    }

    /// The smallest concave distortion above the points ``(s, g)``, for
    /// example layers' exceedance probabilities and their prices per unit
    /// of limit (a cat bond's expected loss and spread).
    ///
    /// Parameters
    /// ----------
    /// points : list of (float, float)
    ///     In the unit square, with ``g >= s``.
    ///
    /// Returns
    /// -------
    /// Distortion
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.risk import Distortion
    /// >>> Distortion.convex([(0.1, 0.3), (0.5, 0.55)]).g(0.05)
    /// 0.15
    #[staticmethod]
    fn convex(points: Vec<(f64, f64)>) -> PyResult<Self> {
        Ok(Self {
            inner: Distortion::convex(&points).map_err(to_py)?,
        })
    }

    /// The smallest ``s`` with ``g(s) >= y``.
    ///
    /// Parameters
    /// ----------
    /// y : float
    ///
    /// Returns
    /// -------
    /// float
    fn g_inv(&self, y: f64) -> f64 {
        self.inner.g_inv(y)
    }

    /// The dual distortion ``1 - g(1 - s)``, which gives the bid.
    ///
    /// Parameters
    /// ----------
    /// s : float
    ///
    /// Returns
    /// -------
    /// float
    fn g_dual(&self, s: f64) -> f64 {
        self.inner.g_dual(s)
    }

    /// The probability mass on the largest outcome, ``g(0+)``.
    #[getter]
    fn mass(&self) -> f64 {
        self.inner.mass()
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

    /// The family, as its constructor is named, for example
    /// ``"proportional_hazard"`` or ``"capped_linear"``.
    #[getter]
    fn family(&self) -> &'static str {
        self.inner.family()
    }

    /// The scalar parameters by name, as the constructor takes them, for
    /// example ``{"rho": 0.7}``; empty for weighted TVaR, mixtures, minimums
    /// and convex envelopes, whose inputs are lists.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.risk import Distortion
    /// >>> d = Distortion.proportional_hazard(0.7)
    /// >>> d.family, d.parameters
    /// ('proportional_hazard', {'rho': 0.7})
    #[getter]
    fn parameters(&self) -> std::collections::BTreeMap<&'static str, f64> {
        self.inner.parameters().into_iter().collect()
    }

    fn __repr__(&self) -> String {
        repr(&self.inner)
    }
}

fn repr(d: &Distortion) -> String {
    match d {
        Distortion::Tvar(p) => format!("Distortion.tvar({p:?})"),
        Distortion::Wang(l) => format!("Distortion.wang({l:?})"),
        Distortion::ProportionalHazard(r) => format!("Distortion.proportional_hazard({r:?})"),
        Distortion::DualPower(b) => format!("Distortion.dual_power({b:?})"),
        Distortion::Exponential(k) => format!("Distortion.exponential({k:?})"),
        Distortion::Ccoc(r) => format!("Distortion.ccoc({r:?})"),
        Distortion::BiTvar { p0, p1, w } => format!("Distortion.bitvar({p0:?}, {p1:?}, {w:?})"),
        Distortion::WeightedTvar { ps, wts } => {
            format!("Distortion.weighted_tvar({ps:?}, {wts:?})")
        }
        Distortion::CappedLinear { r0, slope } => {
            format!("Distortion.capped_linear({r0:?}, {slope:?})")
        }
        Distortion::CappedLogLinear { r0, b } => {
            format!("Distortion.capped_log_linear({r0:?}, {b:?})")
        }
        Distortion::Lep { r0, r } => format!("Distortion.lep({r0:?}, {r:?})"),
        Distortion::LinearYield { r0, r } => format!("Distortion.linear_yield({r0:?}, {r:?})"),
        Distortion::Beta { a, b } => format!("Distortion.beta({a:?}, {b:?})"),
        Distortion::Mixture(parts) => {
            let ds: Vec<String> = parts.iter().map(|(_, d)| repr(d)).collect();
            let ws: Vec<f64> = parts.iter().map(|(w, _)| *w).collect();
            format!("Distortion.mixture([{}], {ws:?})", ds.join(", "))
        }
        Distortion::Minimum(parts) => {
            let ds: Vec<String> = parts.iter().map(repr).collect();
            format!("Distortion.minimum([{}])", ds.join(", "))
        }
        Distortion::Convex(knots) => format!("Distortion.convex({knots:?})"),
    }
}

/// The values, ascending, and their probabilities, of a Sampled, Grid or
/// PredictiveDistribution (its total), capped at ``assets`` when given.
pub(crate) fn discrete_of(
    dist: &Bound<'_, PyAny>,
    assets: Option<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (mut x, p): (Vec<f64>, Vec<f64>) = if let Ok(s) = dist.extract::<PyRef<'_, PySampled>>() {
        let v = s.inner.sorted().to_vec();
        let n = v.len() as f64;
        let p = vec![1.0 / n; v.len()];
        (v, p)
    } else if let Ok(g) = dist.extract::<PyRef<'_, PyGrid>>() {
        let v = (0..g.inner.len()).map(|j| g.inner.x(j)).collect();
        (v, g.inner.probs().to_vec())
    } else if let Ok(pd) = dist.extract::<PyRef<'_, PyPredictiveDistribution>>() {
        let total = pd.inner.total();
        let v = total.sorted().to_vec();
        let n = v.len() as f64;
        let p = vec![1.0 / n; v.len()];
        (v, p)
    } else {
        return Err(PyTypeError::new_err(
            "expected a Sampled, Grid or PredictiveDistribution",
        ));
    };
    if let Some(a) = assets {
        x.iter_mut().for_each(|v| *v = v.min(a));
    }
    Ok((x, p))
}

/// The member of a distortion family whose price of ``dist`` is ``premium``.
///
/// Parameters
/// ----------
/// family : str
///     ``"ccoc"``, ``"ph"``, ``"wang"``, ``"dual"``, ``"tvar"``, ``"exp"``,
///     ``"clin"``, ``"cll"``, ``"lep"`` or ``"ly"``.
/// dist : Sampled, Grid or PredictiveDistribution
///     A predictive distribution is priced on its total.
/// premium : float
///     The target price, strictly between the (capped) mean and maximum.
/// assets : float, optional
///     Price the loss capped at the assets, ``min(X, assets)``.
/// r0 : float, default 0.0
///     The fixed ``r0`` of the ``clin``, ``cll``, ``lep`` and ``ly`` families.
///
/// Returns
/// -------
/// Distortion
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Sampled
/// >>> from prospicio.risk import calibrate
/// >>> x = Sampled([22, 28, 36, 40, 40, 40, 40, 55, 65, 100])
/// >>> premium = (46.6 + 0.15 * 100) / 1.15
/// >>> d = calibrate("proportional_hazard", x, premium)
/// >>> round(d.measure(x), 9) == round(premium, 9)
/// True
#[pyfunction]
#[pyo3(signature = (family, dist, premium, assets=None, r0=0.0))]
pub(crate) fn calibrate(
    family: &str,
    dist: &Bound<'_, PyAny>,
    premium: f64,
    assets: Option<f64>,
    r0: f64,
) -> PyResult<PyDistortion> {
    let family = family_from(family, r0)?;
    let (x, p) = discrete_of(dist, assets)?;
    Ok(PyDistortion {
        inner: prospicio_prob::distortion::calibrate(family, &x, &p, premium).map_err(to_py)?,
    })
}

pub(crate) fn family_from(name: &str, r0: f64) -> PyResult<Family> {
    Ok(match name {
        "ccoc" => Family::Ccoc,
        "ph" | "proportional_hazard" => Family::ProportionalHazard,
        "wang" => Family::Wang,
        "dual" | "dual_power" => Family::DualPower,
        "tvar" => Family::Tvar,
        "exp" | "exponential" => Family::Exponential,
        "clin" | "capped_linear" => Family::CappedLinear { r0 },
        "cll" | "capped_log_linear" => Family::CappedLogLinear { r0 },
        "lep" => Family::Lep { r0 },
        "ly" | "linear_yield" => Family::LinearYield { r0 },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown distortion family {other:?}"
            )));
        }
    })
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
    let (pd, d) = (&pd.inner, distortion.inner.clone());
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

const METHODS: [(&str, AllocationMethod); 6] = [
    ("euler", AllocationMethod::Euler),
    ("covariance", AllocationMethod::Covariance),
    ("covariance_margin", AllocationMethod::CovarianceMargin),
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
/// - ``"covariance"``: ``rho(S) Cov(X_j, S) / Var(S)``, which spreads the
///   expected loss by covariance too.
/// - ``"covariance_margin"``: ``E[X_j] + (rho(S) - E[S]) Cov(X_j, S) /
///   Var(S)``, each unit's mean plus a share of the margin by covariance
///   (Kalkbrener's allocation; *Pricing Insurance Risk*, Exercise 217).
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
///     For an unknown method, a constant total (``"covariance"``,
///     ``"covariance_margin"``),
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
                "method must be one of euler, covariance, covariance_margin, proportional, \
                 marginal, shapley",
            )
        })?;
    let (pd, d) = (&pd.inner, distortion.inner.clone());
    let inner = py.detach(|| pd.capital(&d, method)).map_err(to_py)?;
    Ok(PyAllocation { inner })
}

/// The Shapley value of the game whose cost is ``cost`` of each
/// sub-portfolio, for any cost function, not only a distortion: a
/// standard deviation premium, a variance, an expected utility.
///
/// ``cost`` is called with a list of the sub-portfolio's total in each
/// simulation, in simulation order, once for each of the ``2**m - 1``
/// non-empty sub-portfolios (the empty one costs 0), and returns a float.
///
/// Parameters
/// ----------
/// pd : PredictiveDistribution
///     Components that add up to the portfolio; at most 12.
/// cost : callable
///     ``cost(draws: list[float]) -> float``.
///
/// Returns
/// -------
/// Allocation
///     ``total`` is ``cost`` of the whole portfolio, ``standalone`` that of
///     each component, ``allocated`` the Shapley values, summing to
///     ``total``; ``method`` is ``"shapley"``.
///
/// Raises
/// ------
/// ValueError
///     For more than 12 components or a cost that is not finite.
///
/// Examples
/// --------
/// >>> import statistics
/// >>> from prospicio.distributions import PredictiveDistribution
/// >>> from prospicio.risk import shapley
/// >>> pd = PredictiveDistribution(["lob"], [("motor",), ("property",)],
/// ...                             [[1.0, 2.0], [4.0, 1.0], [2.0, 5.0], [3.0, 6.0]])
/// >>> a = shapley(pd, lambda x: statistics.fmean(x) + statistics.pstdev(x))
/// >>> round(sum(a.allocated) - a.total, 12)
/// 0.0
#[pyfunction]
pub(crate) fn shapley(
    pd: PyRef<'_, PyPredictiveDistribution>,
    cost: &Bound<'_, PyAny>,
) -> PyResult<PyAllocation> {
    let mut error = None;
    let result = pd.inner.shapley_with(|x| {
        if error.is_some() {
            return 0.0;
        }
        match cost.call1((x.to_vec(),)).and_then(|v| v.extract::<f64>()) {
            Ok(v) => v,
            Err(e) => {
                error = Some(e);
                0.0
            }
        }
    });
    if let Some(e) = error {
        return Err(e);
    }
    Ok(PyAllocation {
        inner: result.map_err(to_py)?,
    })
}

/// An allocation of an exogenous total by stand-alone measures at one
/// level, from ``haircut`` or ``equal_risk``.
#[pyclass(name = "LevelAllocation", module = "prospicio.risk", frozen)]
pub(crate) struct PyLevelAllocation {
    inner: LevelAllocation,
}

#[pymethods]
impl PyLevelAllocation {
    /// The stand-alone measure: ``"var"``, ``"tvar"`` or ``"epd"``.
    #[getter]
    fn measure(&self) -> &'static str {
        level_measure_name(self.inner.measure)
    }

    /// The level of the stand-alone measures: a probability for VaR and
    /// TVaR, an EPD ratio for EPD.
    #[getter]
    fn level(&self) -> f64 {
        self.inner.level
    }

    /// The amount allocated.
    #[getter]
    fn total(&self) -> f64 {
        self.inner.total
    }

    /// Each component's stand-alone measure at ``level``.
    #[getter]
    fn standalone(&self) -> Vec<f64> {
        self.inner.standalone.clone()
    }

    /// The allocation per component, summing to ``total``.
    #[getter]
    fn allocated(&self) -> Vec<f64> {
        self.inner.allocated.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "LevelAllocation(measure={:?}, level={}, total={}, allocated={:?})",
            self.measure(),
            self.inner.level,
            self.inner.total,
            self.inner.allocated
        )
    }
}

const LEVEL_MEASURES: [(&str, LevelMeasure); 3] = [
    ("var", LevelMeasure::Var),
    ("tvar", LevelMeasure::Tvar),
    ("epd", LevelMeasure::Epd),
];

fn level_measure_name(measure: LevelMeasure) -> &'static str {
    LEVEL_MEASURES
        .iter()
        .find(|(_, m)| *m == measure)
        .map(|(name, _)| *name)
        .expect("every measure has a name")
}

fn level_measure(name: &str) -> PyResult<LevelMeasure> {
    LEVEL_MEASURES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, m)| *m)
        .ok_or_else(|| PyValueError::new_err("measure must be one of var, tvar, epd"))
}

/// The haircut allocation of an exogenous ``total`` (a premium, a capital
/// amount): each component's stand-alone ``measure`` at ``level``, scaled
/// to add up to ``total`` (*Pricing Insurance Risk*, section 12.2.3).
///
/// Parameters
/// ----------
/// pd : PredictiveDistribution
///     Components that add up to the portfolio.
/// total : float
/// measure : str
///     ``"var"``, ``"tvar"`` (``level`` a probability) or ``"epd"``
///     (``level`` the EPD ratio ``r``: the assets at which
///     ``E[(X - a)+] = r E[X]``).
/// level : float
///
/// Returns
/// -------
/// LevelAllocation
///
/// Raises
/// ------
/// ValueError
///     For a level out of range, a mean that is not positive (EPD) or
///     stand-alone measures summing to 0.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import PredictiveDistribution
/// >>> from prospicio.risk import haircut
/// >>> pd = PredictiveDistribution(["lob"], [("motor",), ("property",)],
/// ...                             [[1.0, 2.0], [4.0, 1.0], [2.0, 5.0], [3.0, 6.0]])
/// >>> haircut(pd, 18.0, "tvar", 0.5).allocated
/// [7.0, 11.0]
#[pyfunction]
pub(crate) fn haircut(
    py: Python<'_>,
    pd: PyRef<'_, PyPredictiveDistribution>,
    total: f64,
    measure: &str,
    level: f64,
) -> PyResult<PyLevelAllocation> {
    let measure = level_measure(measure)?;
    let pd = &pd.inner;
    let inner = py
        .detach(|| pd.haircut(total, measure, level))
        .map_err(to_py)?;
    Ok(PyLevelAllocation { inner })
}

/// The equal risk allocation of an exogenous ``total``: the one level at
/// which the components' stand-alone ``measure``s add up to ``total``,
/// each getting its own (*Pricing Insurance Risk*, section 12.2.4). For VaR
/// and TVaR this prices the components as if comonotonic; for EPD each
/// has the same EPD ratio.
///
/// A VaR of draws moves in steps: where no level adds up exactly, the level
/// is the lowest that reaches ``total`` and its VaRs are scaled to it.
///
/// Parameters
/// ----------
/// pd : PredictiveDistribution
/// total : float
/// measure : str
///     ``"var"``, ``"tvar"`` or ``"epd"``.
///
/// Returns
/// -------
/// LevelAllocation
///     With the ``level`` found.
///
/// Raises
/// ------
/// ValueError
///     When no level makes the stand-alone measures add up to ``total``.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import PredictiveDistribution
/// >>> from prospicio.risk import equal_risk
/// >>> pd = PredictiveDistribution(["lob"], [("motor",), ("property",)],
/// ...                             [[1.0, 2.0], [4.0, 1.0], [2.0, 5.0], [3.0, 6.0]])
/// >>> round(equal_risk(pd, 9.0, "tvar").level, 9)
/// 0.5
#[pyfunction]
pub(crate) fn equal_risk(
    py: Python<'_>,
    pd: PyRef<'_, PyPredictiveDistribution>,
    total: f64,
    measure: &str,
) -> PyResult<PyLevelAllocation> {
    let measure = level_measure(measure)?;
    let pd = &pd.inner;
    let inner = py.detach(|| pd.equal_risk(total, measure)).map_err(to_py)?;
    Ok(PyLevelAllocation { inner })
}

/// One of the copula classes, as a Rust copula.
pub(crate) enum AnyCopula {
    Gaussian(GaussianCopula),
    StudentT(StudentTCopula),
    Archimedean(ArchimedeanCopula),
    Nested(copula::NestedArchimedeanCopula),
    Vine(prospicio_prob::vine::VineCopula),
}

impl AnyCopula {
    pub(crate) fn extract(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(c) = obj.extract::<PyRef<'_, PyGaussianCopula>>() {
            return Ok(Self::Gaussian(c.inner.clone()));
        }
        if let Ok(c) = obj.extract::<PyRef<'_, PyStudentTCopula>>() {
            return Ok(Self::StudentT(c.inner.clone()));
        }
        if let Ok(c) = obj.extract::<PyRef<'_, PyArchimedeanCopula>>() {
            return Ok(Self::Archimedean(c.inner.clone()));
        }
        if let Ok(c) = obj.extract::<PyRef<'_, PyNestedArchimedeanCopula>>() {
            return Ok(Self::Nested(c.inner.clone()));
        }
        if let Ok(c) = obj.extract::<PyRef<'_, PyVineCopula>>() {
            return Ok(Self::Vine(c.inner.clone()));
        }
        Err(PyTypeError::new_err(
            "expected a GaussianCopula, StudentTCopula, ArchimedeanCopula, \
             NestedArchimedeanCopula or VineCopula",
        ))
    }

    pub(crate) fn as_copula(&self) -> &dyn Copula {
        match self {
            Self::Gaussian(c) => c,
            Self::StudentT(c) => c,
            Self::Archimedean(c) => c,
            Self::Nested(c) => c,
            Self::Vine(c) => c,
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

    /// The copula's distribution function at the point ``u``, each
    /// coordinate in ``[0, 1]``.
    ///
    /// Parameters
    /// ----------
    /// u : list of float
    ///     One value per dimension.
    ///
    /// Returns
    /// -------
    /// float
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``u`` has the wrong length or a value outside ``[0, 1]``,
    ///     or the copula has no closed-form distribution function here.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.risk import GaussianCopula
    /// >>> round(GaussianCopula([[1.0, 0.5], [0.5, 1.0]]).cdf([0.5, 0.5]), 12)
    /// 0.333333333333
    fn cdf(&self, u: Vec<f64>) -> PyResult<f64> {
        self.inner.cdf(&u).map_err(to_py)
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

    /// The copula's distribution function at the point ``u``, each
    /// coordinate in ``[0, 1]``.
    ///
    /// Parameters
    /// ----------
    /// u : list of float
    ///     One value per dimension.
    ///
    /// Returns
    /// -------
    /// float
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``u`` has the wrong length or a value outside ``[0, 1]``,
    ///     or the copula has no closed-form distribution function here.
    ///
    /// The t copula has none yet: this always raises ``ValueError``.
    fn cdf(&self, u: Vec<f64>) -> PyResult<f64> {
        self.inner.cdf(&u).map_err(to_py)
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

    /// The copula's distribution function at the point ``u``, each
    /// coordinate in ``[0, 1]``.
    ///
    /// Parameters
    /// ----------
    /// u : list of float
    ///     One value per dimension.
    ///
    /// Returns
    /// -------
    /// float
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``u`` has the wrong length or a value outside ``[0, 1]``,
    ///     or the copula has no closed-form distribution function here.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.risk import ArchimedeanCopula
    /// >>> c = ArchimedeanCopula("clayton", 2.0, 2)
    /// >>> round(c.cdf([0.3, 0.6]), 12) == round((0.3**-2 + 0.6**-2 - 1) ** -0.5, 12)
    /// True
    fn cdf(&self, u: Vec<f64>) -> PyResult<f64> {
        self.inner.cdf(&u).map_err(to_py)
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

fn archimedean_family(family: &str) -> PyResult<Archimedean> {
    Ok(match family {
        "clayton" => Archimedean::Clayton,
        "gumbel" => Archimedean::Gumbel,
        "frank" => Archimedean::Frank,
        "joe" => Archimedean::Joe,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown family {other:?}: use \"clayton\", \"gumbel\", \"frank\" or \"joe\""
            )));
        }
    })
}

/// A node ``(theta, members)``, each member a dimension (an int) or a node.
fn nested_node(obj: &Bound<'_, PyAny>) -> PyResult<copula::NestedNode> {
    let (theta, members): (f64, Vec<Bound<'_, PyAny>>) = obj.extract().map_err(|_| {
        PyTypeError::new_err("a node is a tuple (theta, members), each member an int or a node")
    })?;
    let (mut leaves, mut children) = (Vec::new(), Vec::new());
    for m in members {
        match m.extract::<usize>() {
            Ok(j) => leaves.push(j),
            Err(_) => children.push(nested_node(&m)?),
        }
    }
    Ok(copula::NestedNode::new(theta, leaves, children))
}

fn node_to_py<'py>(py: Python<'py>, node: &copula::NestedNode) -> PyResult<Bound<'py, PyAny>> {
    let mut members: Vec<Bound<'py, PyAny>> = Vec::new();
    for &j in &node.leaves {
        members.push(j.into_pyobject(py)?.into_any());
    }
    for c in &node.children {
        members.push(node_to_py(py, c)?);
    }
    Ok((node.theta, members).into_pyobject(py)?.into_any())
}

/// A nested (hierarchical) Archimedean copula: groups of dimensions joined
/// by their own Clayton or Gumbel copula, and the groups joined by an outer
/// one with weaker dependence, to any depth. Kendall's tau between two
/// dimensions is that of the node where they first meet.
///
/// Parameters
/// ----------
/// family : {"clayton", "gumbel"}
/// tree : tuple
///     The root node ``(theta, members)``: each member a dimension
///     (0-based) or a node. Every dimension appears once, every node has at
///     least two members, and no child's ``theta`` is below its parent's.
///
/// Examples
/// --------
/// >>> from prospicio.risk import NestedArchimedeanCopula
/// >>> c = NestedArchimedeanCopula("gumbel", (1.5, [0, (3.0, [1, 2]), (2.0, [3, 4])]))
/// >>> c.dim, c.family
/// (5, 'gumbel')
/// >>> c.cdf([1.0, 0.5, 1.0, 1.0, 1.0])
/// 0.5
#[pyclass(name = "NestedArchimedeanCopula", module = "prospicio.risk", frozen)]
pub(crate) struct PyNestedArchimedeanCopula {
    inner: copula::NestedArchimedeanCopula,
}

#[pymethods]
impl PyNestedArchimedeanCopula {
    #[new]
    fn new(family: &str, tree: &Bound<'_, PyAny>) -> PyResult<Self> {
        let family = archimedean_family(family)?;
        let root = nested_node(tree)?;
        Ok(Self {
            inner: copula::NestedArchimedeanCopula::new(family, root).map_err(to_py)?,
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

    /// The tree, as given: ``(theta, members)``, leaves first.
    #[getter]
    fn tree<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        node_to_py(py, self.inner.root())
    }

    /// The copula's distribution function at the point ``u``.
    ///
    /// Parameters
    /// ----------
    /// u : list of float
    ///
    /// Returns
    /// -------
    /// float
    fn cdf(&self, u: Vec<f64>) -> PyResult<f64> {
        self.inner.cdf(&u).map_err(to_py)
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

fn pair_family(name: &str) -> PyResult<prospicio_prob::vine::PairFamily> {
    use prospicio_prob::vine::PairFamily;
    Ok(match name {
        "independence" => PairFamily::Independence,
        "gaussian" => PairFamily::Gaussian,
        "clayton" => PairFamily::Clayton,
        "gumbel" => PairFamily::Gumbel,
        "frank" => PairFamily::Frank,
        "joe" => PairFamily::Joe,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown pair family {other:?}: use \"independence\", \"gaussian\", \
                 \"clayton\", \"gumbel\", \"frank\" or \"joe\""
            )));
        }
    })
}

/// A bivariate copula for a vine: a family, its parameter and a rotation.
///
/// Parameters
/// ----------
/// family : {"independence", "gaussian", "clayton", "gumbel", "frank", "joe"}
/// parameter : float, default 0.0
///     Gaussian correlation in (-1, 1); Clayton above 0; Gumbel and Joe at
///     least 1; Frank not 0 (negative for negative dependence).
/// rotation : {0, 90, 180, 270}, default 0
///     For Clayton, Gumbel and Joe: 90 and 270 give negative dependence,
///     180 moves the tail dependence to the other corner (pyvinecopulib's
///     conventions).
///
/// Examples
/// --------
/// >>> from prospicio.risk import PairCopula
/// >>> c = PairCopula("clayton", 2.0)
/// >>> round(c.h2(0.3, 0.6), 12) == round(0.6**-3 * (0.3**-2 + 0.6**-2 - 1) ** -1.5, 12)
/// True
#[pyclass(name = "PairCopula", module = "prospicio.risk", frozen)]
pub(crate) struct PyPairCopula {
    inner: prospicio_prob::vine::PairCopula,
}

#[pymethods]
impl PyPairCopula {
    #[new]
    #[pyo3(signature = (family, parameter = 0.0, rotation = 0))]
    fn new(family: &str, parameter: f64, rotation: u16) -> PyResult<Self> {
        let f = pair_family(family)?;
        Ok(Self {
            inner: prospicio_prob::vine::PairCopula::new(f, parameter, rotation).map_err(to_py)?,
        })
    }

    /// Family name.
    #[getter]
    fn family(&self) -> &'static str {
        use prospicio_prob::vine::PairFamily;
        match self.inner.family() {
            PairFamily::Independence => "independence",
            PairFamily::Gaussian => "gaussian",
            PairFamily::Clayton => "clayton",
            PairFamily::Gumbel => "gumbel",
            PairFamily::Frank => "frank",
            PairFamily::Joe => "joe",
        }
    }

    /// The parameter.
    #[getter]
    fn parameter(&self) -> f64 {
        self.inner.parameter()
    }

    /// The rotation in degrees.
    #[getter]
    fn rotation(&self) -> u16 {
        self.inner.rotation()
    }

    /// The distribution function ``C(u, v)``.
    fn cdf(&self, u: f64, v: f64) -> f64 {
        self.inner.cdf(u, v)
    }

    /// The density ``c(u, v)``.
    fn density(&self, u: f64, v: f64) -> f64 {
        self.inner.density(u, v)
    }

    /// ``h2(u | v) = dC(u, v)/dv``: the first argument given the second.
    fn h2(&self, u: f64, v: f64) -> f64 {
        self.inner.h2(u, v)
    }

    /// ``h1(v | u) = dC(u, v)/du``: the second argument given the first.
    fn h1(&self, v: f64, u: f64) -> f64 {
        self.inner.h1(v, u)
    }

    /// The ``u`` with ``h2(u | v) = p``.
    fn h2_inverse(&self, p: f64, v: f64) -> f64 {
        self.inner.h2_inverse(p, v)
    }

    /// The ``v`` with ``h1(v | u) = p``.
    fn h1_inverse(&self, p: f64, u: f64) -> f64 {
        self.inner.h1_inverse(p, u)
    }

    fn __repr__(&self) -> String {
        format!(
            "PairCopula({:?}, {}, rotation={})",
            self.family(),
            self.inner.parameter(),
            self.inner.rotation()
        )
    }
}

/// A C- or D-vine copula built from bivariate pair copulas (Aas et al.
/// 2009), with each edge's first argument its earlier variable in the
/// order.
///
/// Parameters
/// ----------
/// kind : {"c", "d"}
///     A C-vine: tree ``t`` joins the ``t``-th variable in the order to
///     every later one, given the first ``t``. A D-vine: tree ``t`` joins
///     variables ``t + 1`` places apart in the order, given those between.
/// order : list of int
///     The dimensions (0-based) in vine order.
/// pairs : list of list of PairCopula
///     Tree ``t``'s ``d - 1 - t`` pair copulas, edge by edge. Fewer trees
///     truncate the vine (the rest independence).
///
/// Examples
/// --------
/// >>> from prospicio.risk import PairCopula, VineCopula
/// >>> v = VineCopula("d", [0, 1, 2], [[PairCopula("gumbel", 2.0), PairCopula("clayton", 1.5, 90)],
/// ...                                 [PairCopula("frank", 3.0)]])
/// >>> v.dim
/// 3
#[pyclass(name = "VineCopula", module = "prospicio.risk", frozen)]
pub(crate) struct PyVineCopula {
    inner: prospicio_prob::vine::VineCopula,
}

#[pymethods]
impl PyVineCopula {
    #[new]
    fn new(
        kind: &str,
        order: Vec<usize>,
        pairs: Vec<Vec<PyRef<'_, PyPairCopula>>>,
    ) -> PyResult<Self> {
        use prospicio_prob::vine::{VineCopula, VineKind};
        let kind = match kind {
            "c" => VineKind::Canonical,
            "d" => VineKind::Drawable,
            other => {
                return Err(PyValueError::new_err(format!(
                    "kind must be \"c\" or \"d\", not {other:?}"
                )));
            }
        };
        let pairs = pairs
            .iter()
            .map(|tree| tree.iter().map(|p| p.inner).collect())
            .collect();
        Ok(Self {
            inner: VineCopula::new(kind, order, pairs).map_err(to_py)?,
        })
    }

    /// Number of dimensions.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// ``"c"`` or ``"d"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind() {
            prospicio_prob::vine::VineKind::Canonical => "c",
            prospicio_prob::vine::VineKind::Drawable => "d",
        }
    }

    /// The order.
    #[getter]
    fn order(&self) -> Vec<usize> {
        self.inner.order().to_vec()
    }

    /// The log-density at the point ``u``.
    ///
    /// Parameters
    /// ----------
    /// u : list of float
    ///
    /// Returns
    /// -------
    /// float
    fn log_density(&self, u: Vec<f64>) -> PyResult<f64> {
        self.inner.log_density(&u).map_err(to_py)
    }

    /// The inverse Rosenblatt transform: the point whose conditional
    /// distributions, in vine order, are the uniforms ``w`` (in vine order).
    ///
    /// Parameters
    /// ----------
    /// w : list of float
    ///
    /// Returns
    /// -------
    /// list of float
    fn inverse_rosenblatt(&self, w: Vec<f64>) -> PyResult<Vec<f64>> {
        if w.len() != self.inner.dim() {
            return Err(PyValueError::new_err("give one uniform per dimension"));
        }
        Ok(self.inner.inverse_rosenblatt(&w))
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
/// dof : float, optional
///     Student t reference scores with these degrees of freedom, which give
///     more joint extremes than the default normal scores (Monograph 15,
///     section 8.2.3; ``aggregate``'s ``dof``).
/// scores : list of float, optional
///     Reference scores of your own, one per simulation, in any order.
///     Give ``dof`` or ``scores``, not both.
/// shuffle_rows : bool, default False
///     Shuffle the result's rows. Without it the first component comes
///     back sorted ascending, which matters to anything that reads the
///     rows in order or a slice of them; whole-sample statistics do not
///     notice.
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
#[pyo3(signature = (pd, correlation, seed, *, dof = None, scores = None, shuffle_rows = false))]
pub(crate) fn iman_conover(
    py: Python<'_>,
    pd: PyRef<'_, PyPredictiveDistribution>,
    correlation: Vec<Vec<f64>>,
    seed: u64,
    dof: Option<f64>,
    scores: Option<Vec<f64>>,
    shuffle_rows: bool,
) -> PyResult<PyPredictiveDistribution> {
    let (r, _) = flatten_square(correlation)?;
    let scores = match (dof, scores) {
        (None, None) => copula::Scores::Normal,
        (Some(nu), None) => copula::Scores::StudentT(nu),
        (None, Some(s)) => copula::Scores::Given(s),
        (Some(_), Some(_)) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "give dof or scores, not both",
            ));
        }
    };
    let options = copula::ImanConover {
        scores,
        shuffle_rows,
    };
    let pd = &pd.inner;
    let inner = py
        .detach(|| copula::iman_conover_with(pd, &r, seed, &options))
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

/// Threshold diagnostics for peaks over threshold: at each threshold the
/// mean excess and the generalized Pareto fitted to the exceedances, with
/// standard errors (the numbers behind ``ismev``'s ``mrl.plot`` and
/// ``gpd.fitrange``).
///
/// Above a threshold where a GPD fits, the mean excess is linear in the
/// threshold and the shape ``xi`` and modified scale ``beta - xi * u`` are
/// constant up to their sampling error: pick the lowest threshold above
/// which they are stable, then fit a :class:`PotTail` there. Standard
/// errors come from the observed information and hold for ``xi > -0.5``.
/// A threshold with too few values above it gives NaN where it cannot
/// estimate (2 for the mean excess's standard error, 3 for the GPD).
///
/// Parameters
/// ----------
/// draws : list of float
/// thresholds : list of float
///
/// Returns
/// -------
/// dict of str to list
///     Columns ``threshold``, ``n_above``, ``mean_excess``,
///     ``mean_excess_se``, ``xi``, ``xi_se``, ``beta``,
///     ``modified_scale`` and ``modified_scale_se``, one entry per
///     threshold (``pandas.DataFrame`` takes it as is).
///
/// Examples
/// --------
/// >>> from prospicio.risk import threshold_diagnostics
/// >>> d = threshold_diagnostics([1.0, 2.0, 3.0, 4.0, 5.0], [2.5])
/// >>> d["n_above"], d["mean_excess"]
/// ([3], [1.5])
#[pyfunction]
pub(crate) fn threshold_diagnostics<'py>(
    py: Python<'py>,
    draws: Vec<f64>,
    thresholds: Vec<f64>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let rows = py.detach(|| prospicio_prob::evt::threshold_diagnostics(&draws, &thresholds));
    let d = pyo3::types::PyDict::new(py);
    d.set_item(
        "threshold",
        rows.iter().map(|r| r.threshold).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "n_above",
        rows.iter().map(|r| r.n_above).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "mean_excess",
        rows.iter().map(|r| r.mean_excess).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "mean_excess_se",
        rows.iter().map(|r| r.mean_excess_se).collect::<Vec<_>>(),
    )?;
    d.set_item("xi", rows.iter().map(|r| r.xi).collect::<Vec<_>>())?;
    d.set_item("xi_se", rows.iter().map(|r| r.xi_se).collect::<Vec<_>>())?;
    d.set_item("beta", rows.iter().map(|r| r.beta).collect::<Vec<_>>())?;
    d.set_item(
        "modified_scale",
        rows.iter().map(|r| r.modified_scale).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "modified_scale_se",
        rows.iter().map(|r| r.modified_scale_se).collect::<Vec<_>>(),
    )?;
    Ok(d)
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
