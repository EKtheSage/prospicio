//! `prospicio.distributions`: the Pareto family for treaty pricing
//! (Probability lane; `docs/design/pareto.md`), the gamma and Tweedie
//! (compound Poisson-gamma) distributions, claim counts chosen by
//! dispersion, and `Custom`, a severity from a Python cdf.

use prospicio_core::StreamRng;
use prospicio_prob::{Counting, Distribution, LargeLosses, Severity, Truncation};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::distributions::{PyNegativeBinomial, PyPoisson};
use crate::to_py;

/// The methods every Pareto-family severity shares, generated into the
/// class's single `#[pymethods]` block after its own items.
macro_rules! severity_class {
    ($ty:ident { $($own:tt)* }) => {
        #[pymethods]
        impl $ty {
            $($own)*

            /// Mean of the distribution (``inf`` if it does not exist).
            ///
            /// Returns
            /// -------
            /// float
            fn mean(&self) -> f64 {
                self.inner.mean()
            }

            /// Variance of the distribution (``inf`` if it does not exist).
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

            /// Survival function ``P(X > x)``, accurate far into the tail.
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

            /// Quantile: the smallest ``x`` with ``P(X <= x) >= p``.
            ///
            /// Parameters
            /// ----------
            /// p : float
            ///     Probability in ``[0, 1]``.
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

            /// ``n`` draws from stream ``stream`` of the generator keyed by
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
            /// list of float
            #[pyo3(signature = (n, seed, stream = 0))]
            fn sample(&self, py: Python<'_>, n: usize, seed: u64, stream: u64) -> Vec<f64> {
                let inner = self.inner.clone();
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

            /// Expected excess over a retention, ``E[max(X - retention, 0)]``.
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
            ///     ``inf`` for an unlimited layer.
            /// attachment : float
            ///
            /// Returns
            /// -------
            /// float
            fn layer(&self, limit: f64, attachment: f64) -> f64 {
                self.inner.layer(limit, attachment)
            }

            /// Second moment of the loss to the layer ``limit`` xs
            /// ``attachment``.
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
        }
    };
}

/// Large-loss data from Python arguments.
pub(crate) fn large_losses(
    losses: Vec<f64>,
    reporting_thresholds: Option<Vec<f64>>,
    censored: Option<Vec<bool>>,
    weights: Option<Vec<f64>>,
) -> PyResult<LargeLosses> {
    let mut data = LargeLosses::new(losses).map_err(to_py)?;
    if let Some(r) = reporting_thresholds {
        data = data.reporting_thresholds(r).map_err(to_py)?;
    }
    if let Some(c) = censored {
        data = data.censored(c).map_err(to_py)?;
    }
    if let Some(w) = weights {
        data = data.weights(w).map_err(to_py)?;
    }
    Ok(data)
}

fn truncation_kind(kind: &str) -> PyResult<Truncation> {
    match kind {
        "lp" => Ok(Truncation::LastPiece),
        "wd" => Ok(Truncation::WholeDistribution),
        _ => Err(PyValueError::new_err(
            "truncation_type must be \"lp\" (last piece) or \"wd\" (whole distribution)",
        )),
    }
}

fn truncation_name(kind: Truncation) -> &'static str {
    match kind {
        Truncation::LastPiece => "lp",
        Truncation::WholeDistribution => "wd",
    }
}

/// Single-parameter Pareto: ``P(X > x) = (t / x) ** alpha`` for ``x >= t``,
/// optionally truncated (conditioned on ``X < truncation``).
///
/// Parameters
/// ----------
/// t : float
///     Threshold; finite and positive.
/// alpha : float
///     Pareto alpha; finite and positive.
/// truncation : float, optional
///     Truncation point above ``t``.
///
/// Raises
/// ------
/// ValueError
///     If a parameter is out of range.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Pareto
/// >>> p = Pareto(500.0, 2.0)
/// >>> round(p.layer(4000.0, 1000.0), 9)
/// 200.0
#[pyclass(name = "Pareto", module = "prospicio.distributions", frozen)]
pub(crate) struct PyPareto {
    pub(crate) inner: prospicio_prob::Pareto,
}

severity_class!(PyPareto {
    #[new]
    #[pyo3(signature = (t, alpha, truncation = None))]
    fn new(t: f64, alpha: f64, truncation: Option<f64>) -> PyResult<Self> {
        let p = prospicio_prob::Pareto::new(t, alpha).map_err(to_py)?;
        let inner = match truncation {
            None => p,
            Some(tr) => p.truncated(tr).map_err(to_py)?,
        };
        Ok(Self { inner })
    }

    /// Maximum likelihood fit of the alpha to large losses at or above
    /// ``t``.
    ///
    /// Parameters
    /// ----------
    /// losses : list of float
    /// t : float
    ///     Threshold of the fitted Pareto.
    /// reporting_thresholds : list of float, optional
    ///     Per-loss thresholds below which a loss would not have been
    ///     reported; raised to ``t``.
    /// censored : list of bool, optional
    ///     ``True`` where a loss was capped by a policy limit.
    /// weights : list of float, optional
    /// truncation : float, optional
    ///
    /// Returns
    /// -------
    /// Pareto
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import Pareto
    /// >>> round(Pareto.fit([1500.0, 2500.0, 4000.0], 1000.0).alpha, 6)
    /// 1.10524
    #[staticmethod]
    #[pyo3(signature = (losses, t, reporting_thresholds = None, censored = None, weights = None, truncation = None))]
    fn fit(
        losses: Vec<f64>,
        t: f64,
        reporting_thresholds: Option<Vec<f64>>,
        censored: Option<Vec<bool>>,
        weights: Option<Vec<f64>>,
        truncation: Option<f64>,
    ) -> PyResult<Self> {
        let data = large_losses(losses, reporting_thresholds, censored, weights)?;
        let inner = prospicio_prob::Pareto::fit(t, &data, truncation).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Threshold ``t``.
    #[getter]
    fn t(&self) -> f64 {
        self.inner.t()
    }

    /// Pareto alpha.
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }

    /// Truncation point, or ``None``.
    #[getter]
    fn truncation(&self) -> Option<f64> {
        self.inner.truncation()
    }

    fn __getnewargs__(&self) -> (f64, f64, Option<f64>) {
        (self.inner.t(), self.inner.alpha(), self.inner.truncation())
    }

    fn __repr__(&self) -> String {
        let tr = self.inner.truncation().map_or(String::new(), |t| format!(", truncation={t:?}"));
        format!("Pareto(t={:?}, alpha={:?}{tr})", self.inner.t(), self.inner.alpha())
    }
});

/// Gamma distribution with shape ``alpha`` and scale ``theta``: mean
/// ``alpha * theta``, variance ``alpha * theta**2``.
///
/// Parameters
/// ----------
/// shape : float
/// scale : float
///
/// Raises
/// ------
/// ValueError
///     If a parameter is not finite and positive.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Gamma
/// >>> g = Gamma.from_mean_cv(1000.0, 0.5)
/// >>> g.shape, round(g.std(), 9)
/// (4.0, 500.0)
#[pyclass(name = "Gamma", module = "prospicio.distributions", frozen)]
pub(crate) struct PyGamma {
    pub(crate) inner: prospicio_prob::Gamma,
}

severity_class!(PyGamma {
    #[new]
    fn new(shape: f64, scale: f64) -> PyResult<Self> {
        let inner = prospicio_prob::Gamma::new(shape, scale).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Gamma with the given mean and coefficient of variation.
    ///
    /// Parameters
    /// ----------
    /// mean : float
    /// cv : float
    ///
    /// Returns
    /// -------
    /// Gamma
    #[staticmethod]
    fn from_mean_cv(mean: f64, cv: f64) -> PyResult<Self> {
        let inner = prospicio_prob::Gamma::from_mean_cv(mean, cv).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Gamma with mean ``mu`` and GLM dispersion ``phi`` (variance
    /// ``phi * mu**2``): shape ``1 / phi``.
    ///
    /// Parameters
    /// ----------
    /// mean : float
    /// dispersion : float
    ///
    /// Returns
    /// -------
    /// Gamma
    #[staticmethod]
    fn from_mean_dispersion(mean: f64, dispersion: f64) -> PyResult<Self> {
        let inner = prospicio_prob::Gamma::from_mean_dispersion(mean, dispersion).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Shape ``alpha``.
    #[getter]
    fn shape(&self) -> f64 {
        self.inner.shape()
    }

    /// Scale ``theta``.
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale()
    }

    /// Log density at ``x``.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///
    /// Returns
    /// -------
    /// float
    fn ln_pdf(&self, x: f64) -> f64 {
        self.inner.ln_pdf(x)
    }

    fn __getnewargs__(&self) -> (f64, f64) {
        (self.inner.shape(), self.inner.scale())
    }

    fn __repr__(&self) -> String {
        format!("Gamma(shape={:?}, scale={:?})", self.inner.shape(), self.inner.scale())
    }
});

/// Tweedie distribution with mean ``mu``, dispersion ``phi`` and power
/// ``1 < p < 2``: variance ``phi * mu**p``, a point mass at 0 and a
/// continuous density above it.
///
/// It is a Poisson number of gamma losses, the GLM family for pure
/// premium. ``P(Y = 0) = exp(-lambda_)``.
///
/// Parameters
/// ----------
/// mean : float
/// dispersion : float
/// power : float
///     In ``(1, 2)``.
///
/// Raises
/// ------
/// ValueError
///     If a parameter is out of range.
///
/// Examples
/// --------
/// >>> import math
/// >>> from prospicio.distributions import Tweedie
/// >>> y = Tweedie(500.0, 40.0, 1.6)
/// >>> abs(y.cdf(0.0) - math.exp(-y.lambda_)) < 1e-15
/// True
#[pyclass(name = "Tweedie", module = "prospicio.distributions", frozen)]
pub(crate) struct PyTweedie {
    pub(crate) inner: prospicio_prob::Tweedie,
}

severity_class!(PyTweedie {
    #[new]
    fn new(mean: f64, dispersion: f64, power: f64) -> PyResult<Self> {
        let inner = prospicio_prob::Tweedie::new(mean, dispersion, power).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// The Tweedie equal to a Poisson(``lambda_``) number of
    /// Gamma(``shape``, ``scale``) losses.
    ///
    /// Parameters
    /// ----------
    /// lambda_ : float
    /// shape : float
    /// scale : float
    ///
    /// Returns
    /// -------
    /// Tweedie
    #[staticmethod]
    fn from_poisson_gamma(lambda_: f64, shape: f64, scale: f64) -> PyResult<Self> {
        let inner = prospicio_prob::Tweedie::from_poisson_gamma(lambda_, shape, scale).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Dispersion ``phi``.
    #[getter]
    fn dispersion(&self) -> f64 {
        self.inner.dispersion()
    }

    /// Power ``p``.
    #[getter]
    fn power(&self) -> f64 {
        self.inner.power()
    }

    /// Poisson mean of the number of losses.
    #[getter]
    fn lambda_(&self) -> f64 {
        self.inner.lambda()
    }

    /// The gamma distribution of each loss.
    #[getter]
    fn severity(&self) -> PyGamma {
        PyGamma {
            inner: self.inner.severity(),
        }
    }

    /// Log density at ``y > 0``; at ``y = 0``, the log of the point mass.
    ///
    /// Parameters
    /// ----------
    /// y : float
    ///
    /// Returns
    /// -------
    /// float
    fn ln_pdf(&self, y: f64) -> f64 {
        self.inner.ln_pdf(y)
    }

    fn __getnewargs__(&self) -> (f64, f64, f64) {
        (self.inner.mean(), self.inner.dispersion(), self.inner.power())
    }

    fn __repr__(&self) -> String {
        format!(
            "Tweedie(mean={:?}, dispersion={:?}, power={:?})",
            self.inner.mean(),
            self.inner.dispersion(),
            self.inner.power()
        )
    }
});

/// Weibull distribution with shape ``k`` and scale ``lam``:
/// ``P(X > x) = exp(-(x / lam) ** k)``, as SciPy's ``weibull_min``.
///
/// Parameters
/// ----------
/// shape : float
/// scale : float
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Weibull
/// >>> Weibull(1.0, 2.0).mean()
/// 2.0
#[pyclass(name = "Weibull", module = "prospicio.distributions", frozen)]
pub(crate) struct PyWeibull {
    pub(crate) inner: prospicio_prob::Weibull,
}

severity_class!(PyWeibull {
    #[new]
    fn new(shape: f64, scale: f64) -> PyResult<Self> {
        let inner = prospicio_prob::Weibull::new(shape, scale).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Shape ``k``.
    #[getter]
    fn shape(&self) -> f64 {
        self.inner.shape()
    }

    /// Scale ``lam``.
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale()
    }

    fn __getnewargs__(&self) -> (f64, f64) {
        (self.inner.shape(), self.inner.scale())
    }

    fn __repr__(&self) -> String {
        format!("Weibull(shape={:?}, scale={:?})", self.inner.shape(), self.inner.scale())
    }
});

/// Loglogistic (Fisk) distribution with shape ``alpha`` and scale
/// ``theta`` (the median): ``F(x) = (x/theta)**alpha / (1 + (x/theta)**alpha)``,
/// as SciPy's ``fisk``. Its ``cdf`` is Clark's loglogistic growth curve.
/// The mean is infinite for ``alpha <= 1`` and the variance for
/// ``alpha <= 2``; limited and layer moments always exist.
///
/// Parameters
/// ----------
/// shape : float
/// scale : float
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Loglogistic
/// >>> Loglogistic(1.0, 2.0).cdf(3.0)
/// 0.6
#[pyclass(name = "Loglogistic", module = "prospicio.distributions", frozen)]
pub(crate) struct PyLoglogistic {
    pub(crate) inner: prospicio_prob::Loglogistic,
}

severity_class!(PyLoglogistic {
    #[new]
    fn new(shape: f64, scale: f64) -> PyResult<Self> {
        let inner = prospicio_prob::Loglogistic::new(shape, scale).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Shape ``alpha``.
    #[getter]
    fn shape(&self) -> f64 {
        self.inner.shape()
    }

    /// Scale ``theta``, the median.
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale()
    }

    fn __getnewargs__(&self) -> (f64, f64) {
        (self.inner.shape(), self.inner.scale())
    }

    fn __repr__(&self) -> String {
        format!("Loglogistic(shape={:?}, scale={:?})", self.inner.shape(), self.inner.scale())
    }
});

/// A finite mixture of severities: component ``i`` with probability
/// ``w_i``, such as attritional plus large losses.
///
/// Parameters
/// ----------
/// components : list of (float, severity)
///     Weights (positive, summing to 1) and severities.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Lognormal, Mixture, Pareto
/// >>> m = Mixture([(0.9, Lognormal.from_mean_cv(1e4, 1.0)), (0.1, Pareto(1e5, 2.0))])
/// >>> round(m.mean(), 6)
/// 29000.0
#[pyclass(name = "Mixture", module = "prospicio.distributions", frozen)]
pub(crate) struct PyMixture {
    pub(crate) inner: std::sync::Arc<prospicio_prob::Mixture>,
}

severity_class!(PyMixture {
    #[new]
    fn new(components: Vec<(f64, Bound<'_, PyAny>)>) -> PyResult<Self> {
        let parts = components
            .iter()
            .map(|(w, s)| Ok((*w, crate::distributions::extract_severity(s)?)))
            .collect::<PyResult<Vec<_>>>()?;
        let inner = prospicio_prob::Mixture::from_dists(parts).map_err(to_py)?;
        Ok(Self {
            inner: std::sync::Arc::new(inner),
        })
    }

    /// Component weights.
    #[getter]
    fn weights(&self) -> Vec<f64> {
        self.inner.weights().to_vec()
    }

    fn __repr__(&self) -> String {
        format!("Mixture(weights={:?})", self.inner.weights())
    }
});

/// Piecewise Pareto: alpha ``alpha[k]`` above threshold ``t[k]``, the
/// general large-loss model and the result of tower matching.
///
/// Parameters
/// ----------
/// t : list of float
///     Strictly increasing positive thresholds.
/// alpha : list of float
///     One alpha per threshold; interior ones may be 0, the last must be
///     positive.
/// truncation : float, optional
///     Truncation point above the last threshold.
/// truncation_type : {"lp", "wd"}, default "lp"
///     Truncate the last piece only, or the whole distribution.
///
/// Raises
/// ------
/// ValueError
///     If a parameter is out of range.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import PiecewisePareto
/// >>> pp = PiecewisePareto([1000.0, 2000.0], [1.0, 2.0])
/// >>> round(pp.survival(4000.0), 12)
/// 0.125
#[pyclass(name = "PiecewisePareto", module = "prospicio.distributions", frozen)]
pub(crate) struct PyPiecewisePareto {
    pub(crate) inner: prospicio_prob::PiecewisePareto,
}

severity_class!(PyPiecewisePareto {
    #[new]
    #[pyo3(signature = (t, alpha, truncation = None, truncation_type = "lp"))]
    fn new(
        t: Vec<f64>,
        alpha: Vec<f64>,
        truncation: Option<f64>,
        truncation_type: &str,
    ) -> PyResult<Self> {
        let kind = truncation_kind(truncation_type)?;
        let pp = prospicio_prob::PiecewisePareto::new(t, alpha).map_err(to_py)?;
        let inner = match truncation {
            None => pp,
            Some(tr) => pp.truncated(tr, kind).map_err(to_py)?,
        };
        Ok(Self { inner })
    }

    /// Maximum likelihood fit of the alphas for thresholds ``t`` to large
    /// losses at or above ``t[0]``.
    ///
    /// Parameters
    /// ----------
    /// losses : list of float
    /// t : list of float
    ///     Thresholds of the fitted distribution.
    /// reporting_thresholds : list of float, optional
    /// censored : list of bool, optional
    /// weights : list of float, optional
    /// truncation : float, optional
    /// truncation_type : {"lp", "wd"}, default "lp"
    ///     Truncate the last piece only (each alpha a closed form or a
    ///     one-dimensional solve), or the whole distribution (the alphas
    ///     are coupled and solved together).
    ///
    /// Returns
    /// -------
    /// PiecewisePareto
    #[staticmethod]
    #[pyo3(signature = (losses, t, reporting_thresholds = None, censored = None, weights = None, truncation = None, truncation_type = "lp"))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        py: Python<'_>,
        losses: Vec<f64>,
        t: Vec<f64>,
        reporting_thresholds: Option<Vec<f64>>,
        censored: Option<Vec<bool>>,
        weights: Option<Vec<f64>>,
        truncation: Option<f64>,
        truncation_type: &str,
    ) -> PyResult<Self> {
        let data = large_losses(losses, reporting_thresholds, censored, weights)?;
        let kind = truncation_kind(truncation_type)?;
        let truncation = truncation.map(|tr| (tr, kind));
        let inner = py
            .detach(|| prospicio_prob::PiecewisePareto::fit(t, &data, truncation))
            .map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Thresholds.
    #[getter]
    fn t(&self) -> Vec<f64> {
        self.inner.thresholds().to_vec()
    }

    /// Alphas, one per threshold.
    #[getter]
    fn alpha(&self) -> Vec<f64> {
        self.inner.alphas().to_vec()
    }

    /// Truncation point, or ``None``.
    #[getter]
    fn truncation(&self) -> Option<f64> {
        self.inner.truncation().map(|(tr, _)| tr)
    }

    /// ``"lp"`` or ``"wd"`` when truncated, else ``None``.
    #[getter]
    fn truncation_type(&self) -> Option<&'static str> {
        self.inner.truncation().map(|(_, k)| truncation_name(k))
    }

    fn __getnewargs__(&self) -> (Vec<f64>, Vec<f64>, Option<f64>, &'static str) {
        let (tr, kind) = match self.inner.truncation() {
            Some((tr, k)) => (Some(tr), truncation_name(k)),
            None => (None, "lp"),
        };
        (self.t(), self.alpha(), tr, kind)
    }

    fn __repr__(&self) -> String {
        format!("PiecewisePareto(t={:?}, alpha={:?})", self.t(), self.alpha())
    }
});

/// Log-affine local Pareto: the local alpha
/// ``alpha0 * (1 + gamma * ln(x / t))`` rises linearly in the log of the
/// amount, so ``P(X > x) = exp(-alpha0 L - alpha0 gamma L**2 / 2)`` with
/// ``L = ln(x / t)``.
///
/// Parameters
/// ----------
/// t : float
///     Threshold; finite and positive.
/// alpha0 : float
///     Local alpha at ``t``; finite and positive.
/// gamma : float
///     Non-negative; 0 gives the Pareto.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import LogAffinePareto
/// >>> d = LogAffinePareto.from_delta(1e6, 1.5, 0.5)
/// >>> round(d.local_alpha(2e6), 12)
/// 2.0
#[pyclass(name = "LogAffinePareto", module = "prospicio.distributions", frozen)]
pub(crate) struct PyLogAffinePareto {
    pub(crate) inner: prospicio_prob::LogAffinePareto,
}

severity_class!(PyLogAffinePareto {
    #[new]
    fn new(t: f64, alpha0: f64, gamma: f64) -> PyResult<Self> {
        let inner = prospicio_prob::LogAffinePareto::new(t, alpha0, gamma).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// The distribution from ``delta = alpha0 * gamma * ln 2``, the rise
    /// in the local alpha each time the amount doubles.
    ///
    /// Parameters
    /// ----------
    /// t : float
    /// alpha0 : float
    /// delta : float
    ///
    /// Returns
    /// -------
    /// LogAffinePareto
    #[staticmethod]
    fn from_delta(t: f64, alpha0: f64, delta: f64) -> PyResult<Self> {
        let inner = prospicio_prob::LogAffinePareto::from_delta(t, alpha0, delta).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Threshold ``t``.
    #[getter]
    fn t(&self) -> f64 {
        self.inner.t()
    }

    /// Local alpha at the threshold.
    #[getter]
    fn alpha0(&self) -> f64 {
        self.inner.alpha0()
    }

    /// ``gamma``.
    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    /// ``delta = alpha0 * gamma * ln 2``.
    #[getter]
    fn delta(&self) -> f64 {
        self.inner.delta()
    }

    /// The local Pareto alpha ``-x S'(x) / S(x)`` at ``x``.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///
    /// Returns
    /// -------
    /// float
    fn local_alpha(&self, x: f64) -> f64 {
        self.inner.local_alpha(x)
    }

    fn __getnewargs__(&self) -> (f64, f64, f64) {
        (self.inner.t(), self.inner.alpha0(), self.inner.gamma())
    }

    fn __repr__(&self) -> String {
        format!(
            "LogAffinePareto(t={:?}, alpha0={:?}, gamma={:?})",
            self.inner.t(),
            self.inner.alpha0(),
            self.inner.gamma()
        )
    }
});

/// Generalized Pareto severity with a location (Riegel's parameterization
/// via :meth:`GeneralizedPareto.riegel`): ``P(X > x) =
/// (1 + xi (x - location) / beta) ** (-1 / xi)`` above the location.
///
/// For tail estimation from draws, see :class:`prospicio.risk.Gpd`;
/// this class is the same distribution as a pricing severity.
///
/// Parameters
/// ----------
/// xi : float
///     Shape.
/// beta : float
///     Scale; finite and positive.
/// location : float, default 0.0
///
/// Examples
/// --------
/// >>> from prospicio.distributions import GeneralizedPareto
/// >>> g = GeneralizedPareto.riegel(1000.0, 2.0, 1.5)
/// >>> round(g.survival(2000.0), 12) == round((7 / 3) ** -1.5, 12)
/// True
#[pyclass(name = "GeneralizedPareto", module = "prospicio.distributions", frozen)]
pub(crate) struct PyGeneralizedPareto {
    pub(crate) inner: prospicio_prob::evt::Gpd,
}

severity_class!(PyGeneralizedPareto {
    #[new]
    #[pyo3(signature = (xi, beta, location = 0.0))]
    fn new(xi: f64, beta: f64, location: f64) -> PyResult<Self> {
        let inner = prospicio_prob::evt::Gpd::new(xi, beta)
            .and_then(|g| g.shifted(location))
            .map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Riegel's generalized Pareto: local alpha ``alpha_ini`` at the
    /// threshold ``t``, tending to ``alpha_tail`` far out.
    ///
    /// Parameters
    /// ----------
    /// t : float
    /// alpha_ini : float
    /// alpha_tail : float
    ///
    /// Returns
    /// -------
    /// GeneralizedPareto
    #[staticmethod]
    fn riegel(t: f64, alpha_ini: f64, alpha_tail: f64) -> PyResult<Self> {
        let inner = prospicio_prob::evt::Gpd::riegel(t, alpha_ini, alpha_tail).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Maximum likelihood fit of Riegel's generalized Pareto with threshold
    /// ``t`` to large losses at or above ``t``.
    ///
    /// Parameters
    /// ----------
    /// losses : list of float
    /// t : float
    /// reporting_thresholds : list of float, optional
    /// censored : list of bool, optional
    /// weights : list of float, optional
    ///
    /// Returns
    /// -------
    /// GeneralizedPareto
    ///     Read the alphas as ``t / beta`` (initial) and ``1 / xi`` (tail).
    #[staticmethod]
    #[pyo3(signature = (losses, t, reporting_thresholds = None, censored = None, weights = None))]
    fn fit_riegel(
        py: Python<'_>,
        losses: Vec<f64>,
        t: f64,
        reporting_thresholds: Option<Vec<f64>>,
        censored: Option<Vec<bool>>,
        weights: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let data = large_losses(losses, reporting_thresholds, censored, weights)?;
        let inner = py
            .detach(|| prospicio_prob::evt::Gpd::fit_riegel(t, &data))
            .map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Shape ``xi``.
    #[getter]
    fn xi(&self) -> f64 {
        self.inner.xi()
    }

    /// Scale ``beta``.
    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta()
    }

    /// Location, where the support starts.
    #[getter]
    fn location(&self) -> f64 {
        self.inner.location()
    }

    fn __getnewargs__(&self) -> (f64, f64, f64) {
        (self.inner.xi(), self.inner.beta(), self.inner.location())
    }

    fn __repr__(&self) -> String {
        format!(
            "GeneralizedPareto(xi={:?}, beta={:?}, location={:?})",
            self.inner.xi(),
            self.inner.beta(),
            self.inner.location()
        )
    }
});

/// Binomial claim counts: ``n`` risks, each claiming with probability
/// ``p``.
///
/// Parameters
/// ----------
/// n : int
///     Number of trials.
/// p : float
///     Claim probability in ``[0, 1)``.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Binomial
/// >>> Binomial(10, 0.3).mean()
/// 3.0
#[pyclass(name = "Binomial", module = "prospicio.distributions", frozen)]
pub(crate) struct PyBinomial {
    pub(crate) inner: prospicio_prob::Binomial,
}

#[pymethods]
impl PyBinomial {
    #[new]
    fn new(n: u64, p: f64) -> PyResult<Self> {
        let inner = prospicio_prob::Binomial::new(n, p).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Number of trials.
    #[getter]
    fn n(&self) -> u64 {
        self.inner.n()
    }

    /// Claim probability per trial.
    #[getter]
    fn p(&self) -> f64 {
        self.inner.p()
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
    ///
    /// Returns
    /// -------
    /// int
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

    fn __getnewargs__(&self) -> (u64, f64) {
        (self.inner.n(), self.inner.p())
    }

    fn __repr__(&self) -> String {
        format!("Binomial(n={}, p={:?})", self.inner.n(), self.inner.p())
    }
}

/// The claim count with this mean and dispersion ``Var[N] / E[N]``:
/// binomial below 1, Poisson at 1, negative binomial above 1.
///
/// A binomial needs a whole number of trials, so below 1 the trials are
/// ``mean / (1 - dispersion)`` rounded up: the mean is kept and the
/// dispersion moves up to the nearest attainable value.
///
/// Parameters
/// ----------
/// mean : float
/// dispersion : float
///     Positive.
///
/// Returns
/// -------
/// Binomial or Poisson or NegativeBinomial
///
/// Examples
/// --------
/// >>> from prospicio.distributions import claim_count
/// >>> claim_count(4.0, 2.5)
/// NegativeBinomial(r=2.6666666666666665, beta=1.5)
#[pyfunction]
pub(crate) fn claim_count(py: Python<'_>, mean: f64, dispersion: f64) -> PyResult<Py<PyAny>> {
    let n = prospicio_prob::PanjerClass::from_mean_dispersion(mean, dispersion).map_err(to_py)?;
    Ok(match n {
        prospicio_prob::PanjerClass::Binomial(inner) => {
            Py::new(py, PyBinomial { inner })?.into_any()
        }
        prospicio_prob::PanjerClass::Poisson(inner) => Py::new(py, PyPoisson { inner })?.into_any(),
        prospicio_prob::PanjerClass::NegativeBinomial(inner) => {
            Py::new(py, PyNegativeBinomial { inner })?.into_any()
        }
    })
}

/// Converts the local Pareto distribution with local alpha ``alpha(x)``
/// above ``t`` to a piecewise Pareto that matches its survival function
/// exactly at the thresholds and within ``rel_tolerance`` between them.
///
/// Parameters
/// ----------
/// t : float
///     Threshold; ``P(X > x) = 1`` below it.
/// alpha : callable
///     ``alpha(x) -> float``, finite and non-negative, positive where the
///     conversion stops.
/// rel_tolerance : float, default 1e-4
/// stop_survival : float, default 1e-9
///     Stop once the survival function falls below this.
/// stop_at : float, default inf
///     Stop at this amount.
///
/// Returns
/// -------
/// tuple of (PiecewisePareto, float, float)
///     The approximation, the largest relative error found, and where the
///     approximated range ends (the last alpha continues above it).
///
/// Examples
/// --------
/// >>> import math
/// >>> from prospicio.distributions import local_pareto_to_piecewise
/// >>> pp, err, end = local_pareto_to_piecewise(1000.0, lambda x: 1.5 + 0.3 * math.log(x / 1000.0))
/// >>> err <= 1e-4
/// True
#[pyfunction]
#[pyo3(signature = (t, alpha, rel_tolerance = 1e-4, stop_survival = 1e-9, stop_at = f64::INFINITY))]
pub(crate) fn local_pareto_to_piecewise(
    t: f64,
    alpha: &Bound<'_, PyAny>,
    rel_tolerance: f64,
    stop_survival: f64,
    stop_at: f64,
) -> PyResult<(PyPiecewisePareto, f64, f64)> {
    // A Python error inside the callable becomes NaN for the Rust side,
    // which rejects it; the original error is raised instead.
    let failure = std::cell::RefCell::new(None);
    let call = |x: f64| -> f64 {
        match alpha.call1((x,)).and_then(|v| v.extract::<f64>()) {
            Ok(v) => v,
            Err(e) => {
                failure.borrow_mut().get_or_insert(e);
                f64::NAN
            }
        }
    };
    let options = prospicio_prob::LocalParetoConversion {
        rel_tolerance,
        stop_survival,
        stop_at,
    };
    let result = prospicio_prob::local_pareto_to_piecewise(t, call, options);
    if let Some(e) = failure.into_inner() {
        return Err(e);
    }
    let approx = result.map_err(to_py)?;
    Ok((
        PyPiecewisePareto {
            inner: approx.severity,
        },
        approx.max_relative_error,
        approx.approximated_to,
    ))
}

/// Turns a Python callable of one float into a [`prospicio_prob::custom::Callback`].
/// A Python exception becomes the callback's error message.
fn py_callback(f: Py<PyAny>) -> prospicio_prob::custom::Callback {
    std::sync::Arc::new(move |x: f64| {
        Python::attach(|py| {
            f.call1(py, (x,))
                .and_then(|v| v.extract::<f64>(py))
                .map_err(|e| e.to_string())
        })
    })
}

/// A loss severity defined by your own distribution function: the slow
/// path for a distribution the library does not have.
///
/// Give the cdf, and the quantile function if you have one (sampling
/// inverts the cdf by bisection otherwise, about a hundred cdf calls per
/// draw). The mean, variance, limited expected values and layer moments
/// are computed by Gauss–Legendre quadrature of the survival function
/// between the distribution's own quantiles, ignoring the probability
/// above the ``1 - 1e-12`` quantile. A ``Custom`` goes anywhere a severity
/// does (layers, compound distributions, simulated events, copula
/// marginals, mixtures); calculations that meet one run single-threaded,
/// since every value calls back into Python.
///
/// Parameters
/// ----------
/// cdf : callable
///     ``cdf(x) -> float``: ``P(X <= x)`` for ``x >= 0``, in ``[0, 1]``
///     and non-decreasing. Losses are non-negative.
/// quantile : callable, optional
///     ``quantile(p) -> float``: the smallest ``x`` with ``cdf(x) >= p``.
/// name : str, default "custom"
///     Shown in errors and ``repr``.
///
/// Raises
/// ------
/// ValueError
///     If a callable raises or returns a value out of range, or the cdf
///     never reaches ``1 - 1e-12`` at a finite loss. An error in a later
///     call makes that value ``nan``; ``last_error`` says why.
///
/// Examples
/// --------
/// >>> import math
/// >>> from prospicio.distributions import Custom
/// >>> d = Custom(lambda x: 1 - math.exp(-x / 100), name="exponential")
/// >>> round(d.mean(), 6)
/// 100.0
/// >>> round(d.lev(50), 6) == round(100 * (1 - math.exp(-0.5)), 6)
/// True
#[pyclass(name = "Custom", module = "prospicio.distributions", frozen)]
pub(crate) struct PyCustom {
    pub(crate) inner: prospicio_prob::Custom,
    cdf: Py<PyAny>,
    quantile_fn: Option<Py<PyAny>>,
}

severity_class!(PyCustom {
    #[new]
    #[pyo3(signature = (cdf, quantile = None, name = "custom"))]
    fn new(py: Python<'_>, cdf: Py<PyAny>, quantile: Option<Py<PyAny>>, name: &str) -> PyResult<Self> {
        for (what, f) in [("cdf", Some(&cdf)), ("quantile", quantile.as_ref())] {
            if let Some(f) = f
                && !f.bind(py).is_callable()
            {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "{what} must be callable"
                )));
            }
        }
        let inner = prospicio_prob::Custom::new(
            name,
            py_callback(cdf.clone_ref(py)),
            quantile.as_ref().map(|q| py_callback(q.clone_ref(py))),
            false,
        )
        .map_err(to_py)?;
        Ok(Self {
            inner,
            cdf,
            quantile_fn: quantile,
        })
    }

    /// The name given at construction.
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Whether a quantile function was given.
    #[getter]
    fn has_quantile(&self) -> bool {
        self.inner.has_quantile()
    }

    /// The ``1 - 1e-12`` quantile, where the moment integrals stop.
    #[getter]
    fn upper(&self) -> f64 {
        self.inner.upper()
    }

    /// The first error raised by a callable since construction, or ``None``.
    #[getter]
    fn last_error(&self) -> Option<String> {
        self.inner.error()
    }

    fn __getnewargs__(&self, py: Python<'_>) -> (Py<PyAny>, Option<Py<PyAny>>, String) {
        (
            self.cdf.clone_ref(py),
            self.quantile_fn.as_ref().map(|q| q.clone_ref(py)),
            self.inner.name().to_string(),
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "Custom(name={:?}, mean={:?}, has_quantile={})",
            self.inner.name(),
            self.inner.mean(),
            if self.inner.has_quantile() { "True" } else { "False" }
        )
    }
});
