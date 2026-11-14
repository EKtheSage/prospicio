//! `prospicio.distributions.invert_chf` and `ChfDistribution` (Probability
//! lane), over `prospicio_prob::chf`: a law's distribution on a grid from
//! its characteristic function (#779).

use std::sync::Arc;

use prospicio_prob::Distribution;
use prospicio_prob::chf::{
    BUCKET_SIZING_P, ChfDistribution, ChfFn, ChfLaw, ChfOptions, Complex64, Tilt, UserChf, invert,
};
use prospicio_prob::levy::{NormalInverseGaussian, TemperedStable, TweediePower, VarianceGamma};
use prospicio_prob::{Stable, StableParameterization};
use pyo3::prelude::*;

use crate::distributions::{PyGrid, extract_dist};
use crate::normal::PyNormal;
use crate::to_py;

/// A law's bucket masses on a grid, from its characteristic function: what
/// ``invert_chf`` returns.
///
/// Bucket ``k`` holds the law's probability within half a step of
/// ``x_min + k * step`` (the rounding discretization). The masses are as
/// inverted, not renormalized; the statistics are those of the masses on
/// the window.
#[pyclass(name = "ChfDistribution", module = "prospicio.distributions", frozen)]
pub(crate) struct PyChfDistribution {
    inner: ChfDistribution,
}

#[pymethods]
impl PyChfDistribution {
    /// The first grid point.
    #[getter]
    fn x_min(&self) -> f64 {
        self.inner.x_min()
    }

    /// The bucket.
    #[getter]
    fn step(&self) -> f64 {
        self.inner.step()
    }

    /// The grid has ``2 ** log2`` buckets.
    #[getter]
    fn log2(&self) -> u32 {
        self.inner.report().log2
    }

    /// Whether the bucket was given rather than sized from the law's tails.
    #[getter]
    fn step_given(&self) -> bool {
        self.inner.report().step_given
    }

    /// The grid points.
    #[getter]
    fn xs(&self) -> Vec<f64> {
        self.inner.xs()
    }

    /// The bucket masses, as inverted.
    #[getter]
    fn masses(&self) -> Vec<f64> {
        self.inner.masses().to_vec()
    }

    /// ``|phi(pi / step)|`` of the continuous part: the frequencies the
    /// inversion drops. Above 1e-6 the masses ring; lower the step.
    #[getter]
    fn truncation_error(&self) -> f64 {
        self.inner.report().truncation_error
    }

    /// Whether ``truncation_error`` is above 1e-6.
    #[getter]
    fn is_truncated(&self) -> bool {
        self.inner.report().is_truncated()
    }

    /// The Chernoff bound on the law's mass below the window, which wraps
    /// onto its top; 0 on a grid whose left edge was given.
    #[getter]
    fn outside_below(&self) -> f64 {
        self.inner.report().outside_below
    }

    /// The Chernoff bound on the law's mass above the window, which wraps
    /// onto its bottom; 0 on a grid whose left edge was given.
    #[getter]
    fn outside_above(&self) -> f64 {
        self.inner.report().outside_above
    }

    /// ``(left, right)``: the e-folds of damping across the window on each
    /// side, ``(0, tau)`` for a right power tail tilted, ``(tau, 0)`` for a
    /// left one, ``(0, 0)`` untilted.
    #[getter]
    fn tilt(&self) -> (f64, f64) {
        self.inner.report().tilt
    }

    /// The modelled worst relative error of a bucket at the heavy end of a
    /// law with one power tail; ``nan`` for any other law.
    #[getter]
    fn edge_error(&self) -> f64 {
        self.inner.report().edge_error
    }

    /// The sum of the masses.
    #[getter]
    fn total_mass(&self) -> f64 {
        self.inner.report().total_mass
    }

    /// The law's mean (``nan`` for your own function).
    #[getter]
    fn law_mean(&self) -> f64 {
        self.inner.report().law_mean
    }

    /// The law's variance (``nan`` for your own function).
    #[getter]
    fn law_variance(&self) -> f64 {
        self.inner.report().law_variance
    }

    /// The mean of the masses on the window.
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// The variance of the masses on the window.
    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// The standard deviation of the masses on the window.
    fn std_dev(&self) -> f64 {
        self.inner.std_dev()
    }

    /// The sum of the masses at grid points at or below ``x``.
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

    /// The first grid point whose cumulative mass reaches ``p``.
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

    /// The masses as a ``Grid`` from 0, negative round-off set to 0 and the
    /// sum rescaled to 1, for the compound and risk-measure functions.
    ///
    /// Returns
    /// -------
    /// Grid
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the window starts below 0.
    fn to_grid(&self) -> PyResult<PyGrid> {
        Ok(PyGrid {
            inner: self.inner.to_grid().map_err(to_py)?,
        })
    }

    fn __len__(&self) -> usize {
        self.inner.masses().len()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChfDistribution(x_min={:?}, step={:?}, log2={})",
            self.inner.x_min(),
            self.inner.step(),
            self.inner.report().log2
        )
    }
}

/// An alpha-stable law ``S(alpha, beta, scale, loc)`` for ``invert_chf``.
///
/// With ``beta`` 1 or -1 its one power tail is inverted along a tilted
/// line; with ``|beta| < 1`` it is split into its two totally skewed
/// halves, each inverted along its own tilted line and convolved. At
/// ``alpha = 2`` it is the normal with variance ``2 scale ** 2``.
///
/// Parameters
/// ----------
/// alpha : float
///     The index of stability, in ``(0, 2]``.
/// beta : float
///     The skewness, in ``[-1, 1]``.
/// scale : float, default 1
///     Positive.
/// loc : float, default 0
///     The location, read in ``parameterization``.
/// parameterization : {"S1", "S0"}, default "S1"
///     Nolan's S1 (scipy's default) or S0.
///
/// Raises
/// ------
/// ValueError
///     If a parameter is out of range.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Stable
/// >>> s = Stable(1.5, 1.0, 2.0)
/// >>> round(s.tail_constant, 6), round(s.loc0, 12)
/// (1.128379, -2.0)
#[pyclass(name = "Stable", module = "prospicio.distributions", frozen)]
pub(crate) struct PyStable {
    inner: Stable,
}

#[pymethods]
impl PyStable {
    #[new]
    #[pyo3(signature = (alpha, beta, scale = 1.0, loc = 0.0, parameterization = "S1"))]
    fn new(alpha: f64, beta: f64, scale: f64, loc: f64, parameterization: &str) -> PyResult<Self> {
        let p = match parameterization.to_ascii_uppercase().as_str() {
            "S1" => StableParameterization::S1,
            "S0" => StableParameterization::S0,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "parameterization must be \"S1\" or \"S0\", got {parameterization:?}"
                )));
            }
        };
        Ok(Self {
            inner: Stable::new(alpha, beta, scale, loc, p).map_err(to_py)?,
        })
    }

    /// The index of stability.
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }

    /// The skewness.
    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta()
    }

    /// The scale.
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale()
    }

    /// The S1 location.
    #[getter]
    fn loc(&self) -> f64 {
        self.inner.loc()
    }

    /// The S0 location.
    #[getter]
    fn loc0(&self) -> f64 {
        self.inner.loc0()
    }

    /// ``A`` in ``P(X > x) ~ A x ** -alpha``.
    #[getter]
    fn tail_constant(&self) -> f64 {
        self.inner.tail_constant()
    }

    /// ``B`` in ``P(X < -x) ~ B x ** -alpha``.
    #[getter]
    fn left_tail_constant(&self) -> f64 {
        self.inner.left_tail_constant()
    }

    /// The mean: ``loc`` for ``alpha > 1``, else infinite or ``nan``.
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn __repr__(&self) -> String {
        format!(
            "Stable(alpha={:?}, beta={:?}, scale={:?}, loc={:?})",
            self.inner.alpha(),
            self.inner.beta(),
            self.inner.scale(),
            self.inner.loc()
        )
    }
}

/// The normal inverse Gaussian law ``NIG(alpha, beta, delta, mu)`` for
/// ``invert_chf`` (Barndorff-Nielsen, 1997).
///
/// With ``|beta| < alpha`` and ``gamma = sqrt(alpha**2 - beta**2)``, its
/// cumulant generating function is
/// ``mu s + delta (gamma - sqrt(alpha**2 - (beta + s)**2))``. SciPy's
/// ``norminvgauss(a, b, loc, scale)`` is ``alpha = a / scale``,
/// ``beta = b / scale``, ``delta = scale``, ``mu = loc``.
///
/// Parameters
/// ----------
/// alpha : float
///     Tail heaviness, positive.
/// beta : float
///     Skewness, ``|beta| < alpha``.
/// delta : float
///     Scale, positive.
/// mu : float, default 0
///     Location.
///
/// Raises
/// ------
/// ValueError
///     If a parameter is out of range.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import NormalInverseGaussian, invert_chf
/// >>> d = invert_chf(NormalInverseGaussian(2.0, 0.5, 1.5, 0.3), log2=12)
/// >>> round(d.mean(), 6)
/// 0.687298
#[pyclass(
    name = "NormalInverseGaussian",
    module = "prospicio.distributions",
    frozen
)]
pub(crate) struct PyNormalInverseGaussian {
    inner: NormalInverseGaussian,
}

#[pymethods]
impl PyNormalInverseGaussian {
    #[new]
    #[pyo3(signature = (alpha, beta, delta, mu = 0.0))]
    fn new(alpha: f64, beta: f64, delta: f64, mu: f64) -> PyResult<Self> {
        Ok(Self {
            inner: NormalInverseGaussian::new(alpha, beta, delta, mu).map_err(to_py)?,
        })
    }

    /// ``alpha``.
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }

    /// ``beta``.
    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta()
    }

    /// ``delta``.
    #[getter]
    fn delta(&self) -> f64 {
        self.inner.delta()
    }

    /// ``mu``.
    #[getter]
    fn mu(&self) -> f64 {
        self.inner.mu()
    }

    /// The mean.
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// The variance.
    #[getter]
    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// The skewness.
    #[getter]
    fn skewness(&self) -> f64 {
        self.inner.skewness()
    }

    fn __repr__(&self) -> String {
        format!(
            "NormalInverseGaussian(alpha={:?}, beta={:?}, delta={:?}, mu={:?})",
            self.inner.alpha(),
            self.inner.beta(),
            self.inner.delta(),
            self.inner.mu()
        )
    }
}

/// The variance gamma law ``VG(sigma, nu, theta)`` with a location ``mu``
/// for ``invert_chf`` (Madan, Carr and Chang, 1998).
///
/// Its characteristic function is
/// ``exp(i mu s) (1 - i theta nu s + sigma**2 nu s**2 / 2) ** (-1 / nu)``;
/// at ``theta = 0``, ``nu = 1`` it is the Laplace law with scale
/// ``sigma / sqrt(2)``. A large ``nu`` decays slowly in frequency and needs
/// a small step.
///
/// Parameters
/// ----------
/// sigma : float
///     Volatility, positive.
/// nu : float
///     Variance rate, positive.
/// theta : float, default 0
///     Drift.
/// mu : float, default 0
///     Location.
///
/// Raises
/// ------
/// ValueError
///     If a parameter is out of range.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import VarianceGamma
/// >>> round(VarianceGamma(0.2, 0.5, -0.1).variance, 12)
/// 0.045
#[pyclass(name = "VarianceGamma", module = "prospicio.distributions", frozen)]
pub(crate) struct PyVarianceGamma {
    inner: VarianceGamma,
}

#[pymethods]
impl PyVarianceGamma {
    #[new]
    #[pyo3(signature = (sigma, nu, theta = 0.0, mu = 0.0))]
    fn new(sigma: f64, nu: f64, theta: f64, mu: f64) -> PyResult<Self> {
        Ok(Self {
            inner: VarianceGamma::new(sigma, nu, theta, mu).map_err(to_py)?,
        })
    }

    /// ``sigma``.
    #[getter]
    fn sigma(&self) -> f64 {
        self.inner.sigma()
    }

    /// ``nu``.
    #[getter]
    fn nu(&self) -> f64 {
        self.inner.nu()
    }

    /// ``theta``.
    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta()
    }

    /// ``mu``.
    #[getter]
    fn mu(&self) -> f64 {
        self.inner.mu()
    }

    /// The mean.
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// The variance.
    #[getter]
    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// The skewness.
    #[getter]
    fn skewness(&self) -> f64 {
        self.inner.skewness()
    }

    fn __repr__(&self) -> String {
        format!(
            "VarianceGamma(sigma={:?}, nu={:?}, theta={:?}, mu={:?})",
            self.inner.sigma(),
            self.inner.nu(),
            self.inner.theta(),
            self.inner.mu()
        )
    }
}

/// The CGMY tempered stable law with a location ``mu`` for ``invert_chf``
/// (Carr, Geman, Madan and Yor, 2002).
///
/// Its cumulant generating function is
/// ``mu s + C Gamma(-Y) ((M - s)**Y - M**Y + (G + s)**Y - G**Y)`` for
/// ``-G < s < M``.
///
/// Parameters
/// ----------
/// C : float
///     Activity, positive.
/// G : float
///     Left tempering, positive.
/// M : float
///     Right tempering, positive.
/// Y : float
///     Index, below 2 and not 0 or 1.
/// mu : float, default 0
///     Location.
///
/// Raises
/// ------
/// ValueError
///     If a parameter is out of range.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import TemperedStable
/// >>> round(TemperedStable(1.0, 5.0, 10.0, 0.5).mean, 6)
/// -0.232166
#[pyclass(name = "TemperedStable", module = "prospicio.distributions", frozen)]
pub(crate) struct PyTemperedStable {
    inner: TemperedStable,
}

#[pymethods]
impl PyTemperedStable {
    #[new]
    #[pyo3(signature = (C, G, M, Y, mu = 0.0))]
    #[allow(non_snake_case)]
    fn new(C: f64, G: f64, M: f64, Y: f64, mu: f64) -> PyResult<Self> {
        Ok(Self {
            inner: TemperedStable::new(C, G, M, Y, mu).map_err(to_py)?,
        })
    }

    /// ``C``.
    #[getter(C)]
    fn c(&self) -> f64 {
        self.inner.c()
    }

    /// ``G``.
    #[getter(G)]
    fn g(&self) -> f64 {
        self.inner.g()
    }

    /// ``M``.
    #[getter(M)]
    fn m(&self) -> f64 {
        self.inner.m()
    }

    /// ``Y``.
    #[getter(Y)]
    fn y(&self) -> f64 {
        self.inner.y()
    }

    /// ``mu``.
    #[getter]
    fn mu(&self) -> f64 {
        self.inner.mu()
    }

    /// The mean.
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// The variance.
    #[getter]
    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// The skewness.
    #[getter]
    fn skewness(&self) -> f64 {
        self.inner.skewness()
    }

    fn __repr__(&self) -> String {
        format!(
            "TemperedStable(C={:?}, G={:?}, M={:?}, Y={:?}, mu={:?})",
            self.inner.c(),
            self.inner.g(),
            self.inner.m(),
            self.inner.y(),
            self.inner.mu()
        )
    }
}

/// The Tweedie law at any power ``p`` outside ``(0, 1]`` for
/// ``invert_chf``, in its reproductive form: mean ``mean``, variance
/// ``dispersion * mean ** p`` (Jorgensen's exponential dispersion model).
///
/// ``p = 0`` is the normal, ``p < 0`` the extreme stable laws on the whole
/// line, ``1 < p < 2`` the compound Poisson-gamma (``Tweedie``) with its
/// mass at 0, ``p = 2`` the gamma, ``p = 3`` the inverse Gaussian and
/// ``p > 2`` the positive extreme stable laws, all with light tails.
///
/// Parameters
/// ----------
/// power : float
///     At most 0, or above 1 and finite.
/// mean : float
///     Positive.
/// dispersion : float
///     Positive.
///
/// Raises
/// ------
/// ValueError
///     If a parameter is out of range.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import TweediePower, invert_chf
/// >>> d = invert_chf(TweediePower(2.5, 3.0, 0.4), log2=12)
/// >>> round(d.mean(), 6)
/// 3.0
#[pyclass(name = "TweediePower", module = "prospicio.distributions", frozen)]
pub(crate) struct PyTweediePower {
    inner: TweediePower,
}

#[pymethods]
impl PyTweediePower {
    #[new]
    fn new(power: f64, mean: f64, dispersion: f64) -> PyResult<Self> {
        Ok(Self {
            inner: TweediePower::new(power, mean, dispersion).map_err(to_py)?,
        })
    }

    /// The law with power ``power``, mean ``mean`` and coefficient of
    /// variation ``cv``: dispersion ``cv**2 * mean ** (2 - power)``.
    ///
    /// Parameters
    /// ----------
    /// power : float
    /// mean : float
    /// cv : float
    ///
    /// Returns
    /// -------
    /// TweediePower
    #[staticmethod]
    fn from_mean_cv(power: f64, mean: f64, cv: f64) -> PyResult<Self> {
        Ok(Self {
            inner: TweediePower::from_mean_cv(power, mean, cv).map_err(to_py)?,
        })
    }

    /// The power ``p``.
    #[getter]
    fn power(&self) -> f64 {
        self.inner.power()
    }

    /// The mean.
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// The dispersion.
    #[getter]
    fn dispersion(&self) -> f64 {
        self.inner.dispersion()
    }

    /// The variance, ``dispersion * mean ** power``.
    #[getter]
    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// The skewness, ``power`` times the coefficient of variation.
    #[getter]
    fn skewness(&self) -> f64 {
        self.inner.skewness()
    }

    fn __repr__(&self) -> String {
        format!(
            "TweediePower(power={:?}, mean={:?}, dispersion={:?})",
            self.inner.power(),
            self.inner.mean(),
            self.inner.dispersion()
        )
    }
}

/// A Python callable of a list of frequencies returning their complex
/// values. A Python exception becomes the error message.
fn py_chf(f: Py<PyAny>) -> Arc<ChfFn> {
    Arc::new(move |t: &[f64]| {
        Python::attach(|py| {
            let out = f
                .call1(py, (t.to_vec(),))
                .map_err(|e| prospicio_core::Error::Data(e.to_string()))?;
            let mut values = Vec::with_capacity(t.len());
            for item in out
                .bind(py)
                .try_iter()
                .map_err(|e| prospicio_core::Error::Data(e.to_string()))?
            {
                let z = item
                    .and_then(|z| {
                        let re = z.getattr("real")?.extract::<f64>()?;
                        let im = z.getattr("imag")?.extract::<f64>()?;
                        Ok(Complex64::new(re, im))
                    })
                    .map_err(|e| prospicio_core::Error::Data(e.to_string()))?;
                values.push(z);
            }
            Ok(values)
        })
    })
}

/// A law's distribution on a grid, by inverting its characteristic
/// function: ``aggregate``'s ``AggregateCHF`` (``chfagg``).
///
/// The characteristic function is sampled on the FFT's frequency grid,
/// times ``sinc(t step / 2)``, and inverted once, so bucket ``k`` holds the
/// probability within half a step of ``x_min + k * step``. With ``step``
/// left out, the window is sized from the law's tails (Chernoff bounds) to
/// leave ``1 - bucket_sizing_p`` of the mass outside, half on each side,
/// and the step is rounded up to 1, 2, 4, 5 or 8 times a power of ten
/// (a power of two below 1); when the dropped frequencies are above 1e-6
/// the step moves down that ladder while that helps. A Tweedie's mass at 0
/// is inverted apart, so it does not ring.
///
/// A stable law with one power tail (``beta`` 1 or -1) is inverted along
/// the line ``Im s = theta`` (``tilt``), where the integrand decays like
/// ``exp(-theta x) f(x)``: what wraps is the damped tail, and the mass
/// beyond the window is left out and reported in ``outside_above`` (or
/// added to the first bucket and reported in ``outside_below`` for a left
/// tail). The automatic tilt balances the damped wrap against the
/// round-off the tilt amplifies, and caps the step.
///
/// Parameters
/// ----------
/// law : Normal, Gamma, Exponential, InverseGaussian, Tweedie, Affine, Stable, NormalInverseGaussian, VarianceGamma, TemperedStable, TweediePower or callable
///     A law with a closed-form characteristic function (an ``Affine`` of
///     one of them), or your own characteristic function: a callable of a
///     list of real frequencies ``t`` returning ``E exp(i t X)`` at each,
///     complex. Your own function needs ``step`` and ``x_min``, since it
///     says nothing of the law's tails.
/// log2 : int, default 16
///     The grid has ``2 ** log2`` buckets, 1 to 28.
/// step : float, optional
///     The bucket; sized from the tails when left out.
/// x_min : float, optional
///     The grid's first point, with ``step``; placed from the tails when
///     left out.
/// bucket_sizing_p : float, default 0.99999
///     The mass the window covers when it is sized from the tails.
/// tilt : bool or float, default True
///     ``True``: a law with one power tail at the automatic tilt, a
///     stable law with two split into its halves and each tilted, every
///     other law untilted. ``False``: never tilt (a heavy tail wraps). A
///     number ``tau >= 0``: exactly ``tau`` e-folds of damping across the
///     window, for a stable law only.
///
/// Returns
/// -------
/// ChfDistribution
///
/// Raises
/// ------
/// ValueError
///     If an option is out of range, ``x_min`` is given without ``step``,
///     your own function is given without both, or it returns a value that
///     is not finite; a number ``tilt`` on a law with no power tail.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Gamma, invert_chf
/// >>> d = invert_chf(Gamma(4.0, 25.0), log2=12)
/// >>> d.step, d.x_min, round(d.mean(), 6)
/// (0.25, 0.0, 100.0)
#[pyfunction]
#[pyo3(signature = (law, *, log2 = 16, step = None, x_min = None, bucket_sizing_p = BUCKET_SIZING_P, tilt = None))]
pub(crate) fn invert_chf(
    py: Python<'_>,
    law: &Bound<'_, PyAny>,
    log2: u32,
    step: Option<f64>,
    x_min: Option<f64>,
    bucket_sizing_p: f64,
    tilt: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyChfDistribution> {
    // bool before float, so tilt=1 is one e-fold and tilt=True automatic.
    let tilt = match tilt {
        None => Tilt::Auto,
        Some(t) if t.is_instance_of::<pyo3::types::PyBool>() => {
            if t.extract::<bool>()? {
                Tilt::Auto
            } else {
                Tilt::Off
            }
        }
        Some(t) => Tilt::Fixed(t.extract::<f64>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("tilt must be True, False or a number")
        })?),
    };
    let law = if let Ok(n) = law.extract::<PyRef<'_, PyNormal>>() {
        ChfLaw::Normal(n.inner)
    } else if let Ok(st) = law.extract::<PyRef<'_, PyStable>>() {
        ChfLaw::Stable(st.inner)
    } else if let Ok(d) = law.extract::<PyRef<'_, PyNormalInverseGaussian>>() {
        ChfLaw::NormalInverseGaussian(d.inner)
    } else if let Ok(d) = law.extract::<PyRef<'_, PyVarianceGamma>>() {
        ChfLaw::VarianceGamma(d.inner)
    } else if let Ok(d) = law.extract::<PyRef<'_, PyTemperedStable>>() {
        ChfLaw::TemperedStable(d.inner)
    } else if let Ok(d) = law.extract::<PyRef<'_, PyTweediePower>>() {
        ChfLaw::TweediePower(d.inner)
    } else if let Ok(d) = extract_dist(law) {
        ChfLaw::from_dist(&d).map_err(to_py)?
    } else if law.is_callable() {
        ChfLaw::User(UserChf::new(py_chf(law.clone().unbind()), "chf", false))
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "law must be a Normal, Gamma, Exponential, InverseGaussian, Tweedie, Affine, \
             Stable, NormalInverseGaussian, VarianceGamma, TemperedStable or TweediePower, or a callable \
             characteristic function",
        ));
    };
    let options = ChfOptions {
        log2,
        step,
        x_min,
        bucket_sizing_p,
        tilt,
    };
    let inner = py.detach(|| invert(&law, &options)).map_err(to_py)?;
    Ok(PyChfDistribution { inner })
}
