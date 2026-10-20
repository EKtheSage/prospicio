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

/// `data` with per-loss bounds from above, if given.
fn bounded_above(data: LargeLosses, upper: Option<Vec<f64>>) -> PyResult<LargeLosses> {
    match upper {
        Some(u) => data.truncated_above(u).map_err(to_py),
        None => Ok(data),
    }
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

    /// ``Pareto.fit`` with the standard error of the alpha from the
    /// observed information. Untruncated it is ``alpha / sqrt(n)``, ``n``
    /// the uncensored weight; truncated, the conditioning terms add to the
    /// information. Parameters as ``fit``.
    ///
    /// Returns
    /// -------
    /// tuple of (Pareto, float)
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import Pareto
    /// >>> fit, se = Pareto.fit_with_standard_error([1500.0, 2500.0, 4000.0], 1000.0)
    /// >>> round(se * 3 ** 0.5 / fit.alpha, 12)
    /// 1.0
    #[staticmethod]
    #[pyo3(signature = (losses, t, reporting_thresholds = None, censored = None, weights = None, truncation = None))]
    fn fit_with_standard_error(
        losses: Vec<f64>,
        t: f64,
        reporting_thresholds: Option<Vec<f64>>,
        censored: Option<Vec<bool>>,
        weights: Option<Vec<f64>>,
        truncation: Option<f64>,
    ) -> PyResult<(Self, f64)> {
        let data = large_losses(losses, reporting_thresholds, censored, weights)?;
        let (inner, se) = prospicio_prob::Pareto::fit_with_standard_error(t, &data, truncation)
            .map_err(to_py)?;
        Ok((Self { inner }, se))
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

/// Inverse gamma distribution with shape ``alpha`` and scale ``theta``:
/// ``X = theta / G`` for ``G`` a unit-scale gamma, as SciPy's ``invgamma``.
/// The tail is Pareto-like: the mean is infinite for ``alpha <= 1`` and the
/// variance for ``alpha <= 2``; limited and layer moments always exist.
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
/// >>> from prospicio.distributions import InverseGamma
/// >>> round(InverseGamma(3.0, 2000.0).mean(), 9)
/// 1000.0
#[pyclass(name = "InverseGamma", module = "prospicio.distributions", frozen)]
pub(crate) struct PyInverseGamma {
    pub(crate) inner: prospicio_prob::InverseGamma,
}

severity_class!(PyInverseGamma {
    #[new]
    fn new(shape: f64, scale: f64) -> PyResult<Self> {
        let inner = prospicio_prob::InverseGamma::new(shape, scale).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Inverse gamma with the given mean and coefficient of variation:
    /// ``alpha = 2 + 1 / cv**2``.
    ///
    /// Parameters
    /// ----------
    /// mean : float
    /// cv : float
    ///
    /// Returns
    /// -------
    /// InverseGamma
    #[staticmethod]
    fn from_mean_cv(mean: f64, cv: f64) -> PyResult<Self> {
        let inner = prospicio_prob::InverseGamma::from_mean_cv(mean, cv).map_err(to_py)?;
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

    fn __getnewargs__(&self) -> (f64, f64) {
        (self.inner.shape(), self.inner.scale())
    }

    fn __repr__(&self) -> String {
        format!("InverseGamma(shape={:?}, scale={:?})", self.inner.shape(), self.inner.scale())
    }
});

/// Inverse Gaussian distribution with mean ``mu`` and shape ``lam``:
/// variance ``mu**3 / lam``. SciPy's ``invgauss(mu / lam, scale=lam)``.
///
/// Parameters
/// ----------
/// mean : float
/// shape : float
///
/// Raises
/// ------
/// ValueError
///     If a parameter is not finite and positive.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import InverseGaussian
/// >>> d = InverseGaussian.from_mean_cv(1000.0, 0.5)
/// >>> d.shape, round(d.std(), 9)
/// (4000.0, 500.0)
#[pyclass(name = "InverseGaussian", module = "prospicio.distributions", frozen)]
pub(crate) struct PyInverseGaussian {
    pub(crate) inner: prospicio_prob::InverseGaussian,
}

severity_class!(PyInverseGaussian {
    #[new]
    fn new(mean: f64, shape: f64) -> PyResult<Self> {
        let inner = prospicio_prob::InverseGaussian::new(mean, shape).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Inverse Gaussian with the given mean and coefficient of variation:
    /// ``lam = mean / cv**2``.
    ///
    /// Parameters
    /// ----------
    /// mean : float
    /// cv : float
    ///
    /// Returns
    /// -------
    /// InverseGaussian
    #[staticmethod]
    fn from_mean_cv(mean: f64, cv: f64) -> PyResult<Self> {
        let inner = prospicio_prob::InverseGaussian::from_mean_cv(mean, cv).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Shape ``lam``.
    #[getter]
    fn shape(&self) -> f64 {
        self.inner.shape()
    }

    fn __getnewargs__(&self) -> (f64, f64) {
        (self.inner.mean_param(), self.inner.shape())
    }

    fn __repr__(&self) -> String {
        format!(
            "InverseGaussian(mean={:?}, shape={:?})",
            self.inner.mean_param(),
            self.inner.shape()
        )
    }
});

/// Burr (type XII) distribution with tail shape ``alpha``, power ``gamma``
/// and scale ``theta``: ``P(X > x) = (1 + (x / theta)**gamma)**(-alpha)``,
/// as SciPy's ``burr12(c=gamma, d=alpha, scale=theta)``. With ``alpha = 1``
/// it is the ``Loglogistic``; with ``gamma = 1``, the Lomax. Moments exist
/// below ``alpha * gamma``; limited and layer moments always do.
///
/// Parameters
/// ----------
/// alpha : float
/// gamma : float
/// scale : float
///
/// Raises
/// ------
/// ValueError
///     If a parameter is not finite and positive.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Burr
/// >>> round(Burr(3.0, 1.0, 2000.0).survival(2000.0), 12)
/// 0.125
#[pyclass(name = "Burr", module = "prospicio.distributions", frozen)]
pub(crate) struct PyBurr {
    pub(crate) inner: prospicio_prob::Burr,
}

severity_class!(PyBurr {
    #[new]
    fn new(alpha: f64, gamma: f64, scale: f64) -> PyResult<Self> {
        let inner = prospicio_prob::Burr::new(alpha, gamma, scale).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Tail shape ``alpha``.
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }

    /// Power ``gamma``.
    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    /// Scale ``theta``.
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale()
    }

    fn __getnewargs__(&self) -> (f64, f64, f64) {
        (self.inner.alpha(), self.inner.gamma(), self.inner.scale())
    }

    fn __repr__(&self) -> String {
        format!(
            "Burr(alpha={:?}, gamma={:?}, scale={:?})",
            self.inner.alpha(),
            self.inner.gamma(),
            self.inner.scale()
        )
    }
});

/// Beta distribution with shapes ``a`` and ``b`` on ``[0, scale]``, as
/// SciPy's ``beta(a, b, scale=scale)``: a bounded severity, such as a
/// damage ratio times a sum insured.
///
/// Parameters
/// ----------
/// a : float
/// b : float
/// scale : float, default 1.0
///
/// Raises
/// ------
/// ValueError
///     If a parameter is not finite and positive.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Beta
/// >>> Beta(1.0, 1.0, 10.0).mean()
/// 5.0
#[pyclass(name = "Beta", module = "prospicio.distributions", frozen)]
pub(crate) struct PyBeta {
    pub(crate) inner: prospicio_prob::Beta,
}

severity_class!(PyBeta {
    #[new]
    #[pyo3(signature = (a, b, scale = 1.0))]
    fn new(a: f64, b: f64, scale: f64) -> PyResult<Self> {
        let inner = prospicio_prob::Beta::new(a, b, scale).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// First shape ``a``.
    #[getter]
    fn a(&self) -> f64 {
        self.inner.a()
    }

    /// Second shape ``b``.
    #[getter]
    fn b(&self) -> f64 {
        self.inner.b()
    }

    /// Scale, the top of the support.
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale()
    }

    fn __getnewargs__(&self) -> (f64, f64, f64) {
        (self.inner.a(), self.inner.b(), self.inner.scale())
    }

    fn __repr__(&self) -> String {
        format!(
            "Beta(a={:?}, b={:?}, scale={:?})",
            self.inner.a(),
            self.inner.b(),
            self.inner.scale()
        )
    }
});

/// A severity conditioned on ``lower < X <= upper``, as ``aggregate``'s
/// ``sev_lb`` and ``sev_ub``: its distribution function is
/// ``(F(x) - F(lower)) / (F(upper) - F(lower))`` on the window. Layer
/// moments are exact, from the inner severity's.
///
/// Parameters
/// ----------
/// severity : a severity
///     Any distribution but ``Sampled``.
/// lower : float, default 0.0
/// upper : float, default inf
///
/// Raises
/// ------
/// ValueError
///     If ``0 <= lower < upper`` fails or the window has no probability.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Lognormal, Truncated
/// >>> t = Truncated(Lognormal.from_mean_cv(1000.0, 1.0), upper=5000.0)
/// >>> t.cdf(5000.0), t.mean() < 1000.0
/// (1.0, True)
#[pyclass(name = "Truncated", module = "prospicio.distributions", frozen)]
pub(crate) struct PyTruncated {
    pub(crate) inner: std::sync::Arc<prospicio_prob::Truncated>,
}

severity_class!(PyTruncated {
    #[new]
    #[pyo3(signature = (severity, lower = 0.0, upper = f64::INFINITY))]
    fn new(severity: &Bound<'_, PyAny>, lower: f64, upper: f64) -> PyResult<Self> {
        let inner = crate::distributions::extract_severity(severity)?;
        let t = prospicio_prob::Truncated::new(inner, lower, upper).map_err(to_py)?;
        Ok(Self {
            inner: std::sync::Arc::new(t),
        })
    }

    /// Lower end of the window (excluded).
    #[getter]
    fn lower(&self) -> f64 {
        self.inner.lower()
    }

    /// Upper end of the window (included).
    #[getter]
    fn upper(&self) -> f64 {
        self.inner.upper()
    }

    /// ``P(lower < X <= upper)`` under the inner severity.
    #[getter]
    fn probability(&self) -> f64 {
        self.inner.probability()
    }

    /// The severity before conditioning.
    #[getter]
    fn severity(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::distributions::dist_to_py(py, self.inner.inner().dist().clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "Truncated({}, lower={:?}, upper={:?})",
            self.inner.inner().dist().family(),
            self.inner.lower(),
            self.inner.upper()
        )
    }
});

/// A severity scaled and shifted, ``scale * X + shift``: ``aggregate``'s
/// ``10 * lognorm 1 + 20``, keeping the severity's exact layer moments.
///
/// Parameters
/// ----------
/// severity : severity
///     Any severity but ``Sampled``.
/// scale : float, default 1.0
///     Positive.
/// shift : float, default 0.0
///     Non-negative, so that the result is a loss.
///
/// Raises
/// ------
/// ValueError
///     If ``scale`` is not positive or ``shift`` is negative.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Affine, Lognormal
/// >>> x = Lognormal(0.0, 1.0)
/// >>> y = Affine(x, scale=10.0, shift=20.0)
/// >>> round(y.mean() - (10 * x.mean() + 20), 9), y.cdf(20.0)
/// (0.0, 0.0)
#[pyclass(name = "Affine", module = "prospicio.distributions", frozen)]
pub(crate) struct PyAffine {
    pub(crate) inner: std::sync::Arc<prospicio_prob::Affine>,
}

severity_class!(PyAffine {
    #[new]
    #[pyo3(signature = (severity, scale = 1.0, shift = 0.0))]
    fn new(severity: &Bound<'_, PyAny>, scale: f64, shift: f64) -> PyResult<Self> {
        let inner = crate::distributions::extract_severity(severity)?;
        let t = prospicio_prob::Affine::new(inner, scale, shift).map_err(to_py)?;
        Ok(Self {
            inner: std::sync::Arc::new(t),
        })
    }

    /// The scale.
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale()
    }

    /// The shift.
    #[getter]
    fn shift(&self) -> f64 {
        self.inner.shift()
    }

    /// The severity before scaling and shifting.
    #[getter]
    fn severity(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::distributions::dist_to_py(py, self.inner.inner().dist().clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "Affine({}, scale={:?}, shift={:?})",
            self.inner.inner().dist().family(),
            self.inner.scale(),
            self.inner.shift()
        )
    }
});

/// A severity read from a size-of-loss table: claim counts and loss totals
/// by size band, as Werner and Modlin's *Basic Ratemaking* chapter 11
/// prices increased limits and deductibles. Each band's claims sit at the
/// band's average, so the limited expected value is exact at every band
/// edge, ``(losses below b + b * claims above b) / claims``, and ``ilf``,
/// ``loss_elimination_ratio`` and every layer function take it. The mean is
/// the table's; the variance sees only the band averages.
///
/// Parameters
/// ----------
/// bounds : list of float
///     ``len(counts) + 1`` band bounds, from 0 or above, strictly
///     increasing; the last may be ``inf``.
/// counts : list of float
///     Claims in each band.
/// losses : list of float
///     Their loss total. Each band's average must lie within its bounds (a
///     claim capped at a policy limit belongs to the band below the limit).
///
/// Raises
/// ------
/// ValueError
///     If the lengths differ, the bounds do not increase, or a band's
///     average lies outside its bounds.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import SizeOfLoss
/// >>> from prospicio.pricing import ilf
/// >>> t = SizeOfLoss([0, 100e3, 250e3, 500e3, 1e6], [2324, 1923, 680, 73],
/// ...                [117_629_223, 307_599_929, 222_793_514, 43_047_470])
/// >>> round(t.lev(100e3)), round(ilf(t, 250e3, 100e3), 2)
/// (77046, 1.59)
#[pyclass(name = "SizeOfLoss", module = "prospicio.distributions", frozen)]
pub(crate) struct PySizeOfLoss {
    pub(crate) inner: prospicio_prob::SizeOfLoss,
}

severity_class!(PySizeOfLoss {
    #[new]
    fn new(bounds: Vec<f64>, counts: Vec<f64>, losses: Vec<f64>) -> PyResult<Self> {
        let inner = prospicio_prob::SizeOfLoss::new(&bounds, &counts, &losses).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// The band bounds.
    #[getter]
    fn bounds(&self) -> Vec<f64> {
        self.inner.bounds().to_vec()
    }

    /// The claim counts by band.
    #[getter]
    fn counts(&self) -> Vec<f64> {
        self.inner.counts().to_vec()
    }

    /// The loss totals by band.
    #[getter]
    fn losses(&self) -> Vec<f64> {
        self.inner.losses().to_vec()
    }

    /// The number of claims.
    #[getter]
    fn claims(&self) -> f64 {
        self.inner.claims()
    }

    fn __repr__(&self) -> String {
        format!(
            "SizeOfLoss({} bands, {} claims)",
            self.inner.counts().len(),
            self.inner.claims()
        )
    }
});

/// A severity capped by a mix of policy limits: the loss ``min(X, L)``
/// with ``L = limits[j]`` with probability ``weights[j]``, as a cedant's
/// losses censored at its policies' limits. Every layer moment is exact,
/// from the base severity's, and the distribution function has an atom at
/// each finite limit, so ``Grid.rounding`` and ``CollectiveModel`` take it
/// like any severity.
///
/// Parameters
/// ----------
/// severity : a severity
///     The losses before the limits; any distribution but ``Sampled``.
/// limits : list of float
///     Positive; ``inf`` for an unlimited policy.
/// weights : list of float
///     Non-negative, normalized to sum to 1 (policy or loss counts do).
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Pareto, PolicyLimits
/// >>> c = PolicyLimits(Pareto(100.0, 2.0), [1000.0, 5000.0], [3, 1])
/// >>> round(c.mean(), 9)  # 0.75 * 100 * (2 - 0.1) + 0.25 * 100 * (2 - 0.02)
/// 192.0
#[pyclass(name = "PolicyLimits", module = "prospicio.distributions", frozen)]
pub(crate) struct PyPolicyLimits {
    pub(crate) inner: std::sync::Arc<prospicio_prob::PolicyLimits>,
}

severity_class!(PyPolicyLimits {
    #[new]
    fn new(severity: &Bound<'_, PyAny>, limits: Vec<f64>, weights: Vec<f64>) -> PyResult<Self> {
        let base = crate::distributions::extract_severity(severity)?;
        let p = prospicio_prob::PolicyLimits::new(base, limits, weights).map_err(to_py)?;
        Ok(Self {
            inner: std::sync::Arc::new(p),
        })
    }

    /// The policy limits.
    #[getter]
    fn limits(&self) -> Vec<f64> {
        self.inner.limits().to_vec()
    }

    /// Each limit's weight, summing to 1.
    #[getter]
    fn weights(&self) -> Vec<f64> {
        self.inner.weights().to_vec()
    }

    /// The severity before the limits.
    #[getter]
    fn severity(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::distributions::dist_to_py(py, self.inner.base().dist().clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "PolicyLimits({}, limits={:?}, weights={:?})",
            self.inner.base().dist().family(),
            self.inner.limits(),
            self.inner.weights()
        )
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

    /// A splice: component ``i`` conditioned on
    /// ``(breaks[i], breaks[i + 1]]`` with weight ``w_i``, such as a body
    /// below a threshold and a Pareto tail above it.
    ///
    /// Parameters
    /// ----------
    /// components : list of (float, severity)
    ///     Weights (positive, summing to 1) and severities.
    /// breaks : list of float
    ///     One more than the components: 0, the joins, and the top (which
    ///     may be ``inf``).
    ///
    /// Returns
    /// -------
    /// Mixture
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the breaks do not fit the components or a piece's window has
    ///     no probability.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import Lognormal, Mixture, Pareto
    /// >>> body, tail = Lognormal.from_mean_cv(50.0, 1.0), Pareto(100.0, 1.8)
    /// >>> s = Mixture.splice([(0.9, body), (0.1, tail)], [0.0, 100.0, float("inf")])
    /// >>> round(s.cdf(100.0), 12)
    /// 0.9
    #[staticmethod]
    fn splice(components: Vec<(f64, Bound<'_, PyAny>)>, breaks: Vec<f64>) -> PyResult<Self> {
        let parts = components
            .iter()
            .map(|(w, s)| Ok((*w, crate::distributions::extract_severity(s)?)))
            .collect::<PyResult<Vec<_>>>()?;
        let inner = prospicio_prob::Truncated::splice(parts, &breaks).map_err(to_py)?;
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

    /// ``PiecewisePareto.fit`` with the standard error of each alpha. The
    /// likelihood separates by piece, so the alphas are uncorrelated:
    /// ``alpha_k / sqrt(n_k)``, ``n_k`` the uncensored weight ending in
    /// piece ``k`` (NaN where none does). A truncated last piece takes the
    /// truncated Pareto's; ``truncation_type="wd"`` couples the alphas and
    /// is refused. Parameters as ``fit``.
    ///
    /// Returns
    /// -------
    /// tuple of (PiecewisePareto, list of float)
    #[staticmethod]
    #[pyo3(signature = (losses, t, reporting_thresholds = None, censored = None, weights = None, truncation = None, truncation_type = "lp"))]
    #[allow(clippy::too_many_arguments)]
    fn fit_with_standard_errors(
        losses: Vec<f64>,
        t: Vec<f64>,
        reporting_thresholds: Option<Vec<f64>>,
        censored: Option<Vec<bool>>,
        weights: Option<Vec<f64>>,
        truncation: Option<f64>,
        truncation_type: &str,
    ) -> PyResult<(Self, Vec<f64>)> {
        let data = large_losses(losses, reporting_thresholds, censored, weights)?;
        let kind = truncation_kind(truncation_type)?;
        let truncation = truncation.map(|tr| (tr, kind));
        let (inner, se) =
            prospicio_prob::PiecewisePareto::fit_with_standard_errors(t, &data, truncation)
                .map_err(to_py)?;
        Ok((Self { inner }, se))
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

/// A severity fitted by maximum likelihood to large losses, from
/// ``fit_severity``: the distribution, its parameters with standard errors
/// and covariance (from the observed information), and the fit's
/// log-likelihood with AIC and BIC.
#[pyclass(name = "SeverityFit", module = "prospicio.distributions", frozen)]
pub(crate) struct PySeverityFit {
    inner: prospicio_prob::severity_fit::SeverityFit,
}

#[pymethods]
impl PySeverityFit {
    /// The family's name.
    #[getter]
    fn family(&self) -> &'static str {
        self.inner.family().name()
    }

    /// The fitted distribution.
    #[getter]
    fn distribution(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::distributions::dist_to_py(py, self.inner.dist().clone())
    }

    /// The parameters by name.
    #[getter]
    fn parameters(&self) -> std::collections::BTreeMap<&'static str, f64> {
        let names = self.inner.family().parameter_names();
        names
            .iter()
            .copied()
            .zip(self.inner.parameters().iter().copied())
            .collect()
    }

    /// Each parameter's standard error, by name; NaN where the observed
    /// information is singular.
    #[getter]
    fn standard_errors(&self) -> std::collections::BTreeMap<&'static str, f64> {
        let names = self.inner.family().parameter_names();
        names
            .iter()
            .copied()
            .zip(self.inner.standard_errors())
            .collect()
    }

    /// The parameters' covariance, one row per parameter in the family's
    /// order (``meanlog, sdlog``; ``shape, scale``; ``alpha, gamma, scale``).
    #[getter]
    fn covariance(&self) -> Vec<Vec<f64>> {
        let n = self.inner.parameters().len();
        self.inner
            .covariance()
            .chunks(n)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// The maximized log-likelihood.
    #[getter]
    fn log_likelihood(&self) -> f64 {
        self.inner.log_likelihood()
    }

    /// Akaike's information criterion.
    #[getter]
    fn aic(&self) -> f64 {
        self.inner.aic()
    }

    /// The Bayesian information criterion, with the losses' total weight.
    #[getter]
    fn bic(&self) -> f64 {
        self.inner.bic()
    }

    /// Whether the optimizer met its tolerance.
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged()
    }

    fn __repr__(&self) -> String {
        format!(
            "SeverityFit({}, {:?}, log_likelihood={:.6})",
            self.inner.family().name(),
            self.inner.parameters(),
            self.inner.log_likelihood()
        )
    }
}

/// Maximum likelihood fit of a lognormal, gamma, Weibull, Burr or Lomax
/// severity to large losses, each conditioned on exceeding its reporting
/// threshold (left truncation), with losses capped by a policy limit
/// censored, and optionally truncated from above (each loss conditioned on
/// not exceeding its bound). The Lomax, ``S(x) = (1 + x / scale) **
/// -alpha`` (Bahnemann's Pareto), is fitted as a :class:`Burr` with
/// ``gamma = 1``.
///
/// Parameters
/// ----------
/// family : {"lognormal", "gamma", "weibull", "burr", "lomax"}
/// losses : list of float
/// reporting_thresholds : list of float, optional
///     Per-loss thresholds below which a loss would not have been
///     reported.
/// censored : list of bool, optional
///     ``True`` where a policy limit capped the loss.
/// weights : list of float, optional
/// truncated_above : list of float, optional
///     Per-loss bounds from above: a loss is conditioned on not exceeding
///     its bound (a study of claims up to 50,000). ``math.inf`` means none.
///
/// Returns
/// -------
/// SeverityFit
///
/// Examples
/// --------
/// >>> import math
/// >>> from prospicio.distributions import fit_severity
/// >>> y = [120.0, 300.0, 450.0, 800.0, 1500.0, 2600.0, 4000.0]
/// >>> fit = fit_severity("lognormal", y)
/// >>> round(fit.parameters["meanlog"], 6) == round(sum(map(math.log, y)) / 7, 6)
/// True
#[pyfunction]
#[pyo3(signature = (
    family, losses, reporting_thresholds = None, censored = None, weights = None,
    truncated_above = None
))]
pub(crate) fn fit_severity(
    py: Python<'_>,
    family: &str,
    losses: Vec<f64>,
    reporting_thresholds: Option<Vec<f64>>,
    censored: Option<Vec<bool>>,
    weights: Option<Vec<f64>>,
    truncated_above: Option<Vec<f64>>,
) -> PyResult<PySeverityFit> {
    use prospicio_prob::severity_fit::{Family, fit_severity};
    let family = Family::from_name(family).map_err(to_py)?;
    let data = bounded_above(
        large_losses(losses, reporting_thresholds, censored, weights)?,
        truncated_above,
    )?;
    let inner = py.detach(|| fit_severity(family, &data)).map_err(to_py)?;
    Ok(PySeverityFit { inner })
}

/// The product-limit (Kaplan–Meier) estimator for left-truncated,
/// right-censored losses: at each value where an uncensored loss lies, the
/// weight at risk (losses reported below it and not below it), the
/// uncensored weight there and ``S(x)``, the survival conditional on the
/// lowest reporting threshold.
///
/// Parameters
/// ----------
/// losses : list of float
/// reporting_thresholds : list of float, optional
/// censored : list of bool, optional
/// weights : list of float, optional
///
/// Returns
/// -------
/// dict of str to list of float
///     ``x``, ``at_risk``, ``events`` and ``survival``.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import product_limit
/// >>> product_limit([100.0, 200.0, 300.0, 400.0])["survival"]
/// [0.75, 0.5, 0.25, 0.0]
#[pyfunction]
#[pyo3(signature = (losses, reporting_thresholds = None, censored = None, weights = None))]
pub(crate) fn product_limit(
    losses: Vec<f64>,
    reporting_thresholds: Option<Vec<f64>>,
    censored: Option<Vec<bool>>,
    weights: Option<Vec<f64>>,
) -> PyResult<std::collections::BTreeMap<&'static str, Vec<f64>>> {
    let data = large_losses(losses, reporting_thresholds, censored, weights)?;
    let steps = prospicio_prob::severity_fit::product_limit(&data);
    Ok([
        ("x", steps.iter().map(|s| s.x).collect()),
        ("at_risk", steps.iter().map(|s| s.at_risk).collect()),
        ("events", steps.iter().map(|s| s.events).collect()),
        ("survival", steps.iter().map(|s| s.survival).collect()),
    ]
    .into_iter()
    .collect())
}

/// A severity against the product-limit estimate of large losses, each
/// conditioned on its reporting threshold: P-P (``survival`` against
/// ``model_survival``, the model's survival conditional on the lowest
/// reporting threshold), Q-Q (``x`` against ``model_quantile``, the
/// model's conditional quantile at the middle of each step) and
/// ``max_distance``, the largest gap between the two survival functions.
/// With a bound from above common to every loss, both sides are
/// conditioned on it too; bounds that differ are refused.
///
/// Parameters
/// ----------
/// severity : a distribution
///     A fit's ``distribution``, a Pareto fit, or any other.
/// losses : list of float
/// reporting_thresholds : list of float, optional
/// censored : list of bool, optional
/// weights : list of float, optional
/// truncated_above : list of float, optional
///     Per-loss bounds from above: a loss is conditioned on not exceeding
///     its bound (a study of claims up to 50,000). ``math.inf`` means none.
///
/// Returns
/// -------
/// dict
///     ``x``, ``at_risk``, ``survival``, ``model_survival``,
///     ``model_quantile`` (lists) and ``max_distance`` (float).
///
/// Examples
/// --------
/// >>> from prospicio.distributions import fit_severity, goodness_of_fit
/// >>> y = [120.0, 300.0, 450.0, 800.0, 1500.0, 2600.0]
/// >>> gof = goodness_of_fit(fit_severity("lognormal", y).distribution, y)
/// >>> len(gof["x"]), gof["max_distance"] < 0.3
/// (6, True)
#[pyfunction]
#[pyo3(signature = (
    severity, losses, reporting_thresholds = None, censored = None, weights = None,
    truncated_above = None
))]
pub(crate) fn goodness_of_fit<'py>(
    py: Python<'py>,
    severity: &Bound<'py, PyAny>,
    losses: Vec<f64>,
    reporting_thresholds: Option<Vec<f64>>,
    censored: Option<Vec<bool>>,
    weights: Option<Vec<f64>>,
    truncated_above: Option<Vec<f64>>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use pyo3::types::PyDictMethods;
    let dist = crate::distributions::extract_dist(severity)?;
    let data = bounded_above(
        large_losses(losses, reporting_thresholds, censored, weights)?,
        truncated_above,
    )?;
    let g = prospicio_prob::severity_fit::goodness_of_fit(&dist, &data).map_err(to_py)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("x", g.steps.iter().map(|s| s.x).collect::<Vec<_>>())?;
    d.set_item(
        "at_risk",
        g.steps.iter().map(|s| s.at_risk).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "survival",
        g.steps.iter().map(|s| s.survival).collect::<Vec<_>>(),
    )?;
    d.set_item("model_survival", g.model_survival)?;
    d.set_item("model_quantile", g.model_quantile)?;
    d.set_item("max_distance", g.max_distance)?;
    Ok(d)
}
