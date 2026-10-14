//! `prospicio.credibility`: classical credibility, the limited fluctuation
//! standards and the Bühlmann–Straub model (Aggregate lane,
//! `docs/design/credibility.md`).

use prospicio_pricing::credibility::{
    BuhlmannStraub, BuhlmannStraubFit, LimitedFluctuation, StructureEstimator,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::to_py;

/// Limited fluctuation (classical) credibility: the expected number of
/// claims for the observed value to lie within ``k`` of its mean with
/// probability ``p``, under a Poisson count and the normal approximation.
///
/// With ``z = Phi^-1((1 + p) / 2)`` and ``n_F = (z / k)**2``, the full
/// standards in claims are ``n_F`` for the frequency, ``n_F * cv**2`` for
/// the severity and ``n_F * (1 + cv**2)`` for the pure premium; partial
/// credibility is ``min(1, sqrt(n / standard))``.
///
/// Parameters
/// ----------
/// p : float, default 0.90
/// k : float, default 0.05
///
/// Raises
/// ------
/// ValueError
///     If ``p`` is not in ``(0, 1)`` or ``k`` is not positive.
///
/// Examples
/// --------
/// >>> from prospicio.credibility import LimitedFluctuation
/// >>> lf = LimitedFluctuation(0.90, 0.05)
/// >>> round(lf.frequency_standard())
/// 1082
/// >>> lf.p, lf.k
/// (0.9, 0.05)
#[pyclass(name = "LimitedFluctuation", module = "prospicio.credibility", frozen)]
pub(crate) struct PyLimitedFluctuation {
    inner: LimitedFluctuation,
}

#[pymethods]
impl PyLimitedFluctuation {
    #[new]
    #[pyo3(signature = (p = 0.90, k = 0.05))]
    fn new(p: f64, k: f64) -> PyResult<Self> {
        Ok(Self {
            inner: LimitedFluctuation::new(p, k).map_err(to_py)?,
        })
    }

    /// The probability ``p`` of lying within ``k`` of the mean.
    #[getter]
    fn p(&self) -> f64 {
        self.inner.p()
    }

    /// The tolerance ``k``, relative to the mean.
    #[getter]
    fn k(&self) -> f64 {
        self.inner.k()
    }

    /// Full standard for the frequency, in expected claims.
    ///
    /// Returns
    /// -------
    /// float
    fn frequency_standard(&self) -> f64 {
        self.inner.frequency_standard()
    }

    /// Full standard for the severity, in claims.
    ///
    /// Parameters
    /// ----------
    /// cv : float
    ///     The severity's coefficient of variation.
    ///
    /// Returns
    /// -------
    /// float
    fn severity_standard(&self, cv: f64) -> f64 {
        self.inner.severity_standard(cv)
    }

    /// Full standard for the pure premium, in expected claims.
    ///
    /// Parameters
    /// ----------
    /// cv : float
    ///     The severity's coefficient of variation.
    ///
    /// Returns
    /// -------
    /// float
    fn pure_premium_standard(&self, cv: f64) -> f64 {
        self.inner.pure_premium_standard(cv)
    }

    /// Credibility ``min(1, sqrt(n / standard))`` of ``n`` claims.
    ///
    /// Parameters
    /// ----------
    /// n : float
    /// standard : float
    ///
    /// Returns
    /// -------
    /// float
    fn credibility(&self, n: f64, standard: f64) -> f64 {
        self.inner.credibility(n, standard)
    }

    fn __repr__(&self) -> String {
        format!(
            "LimitedFluctuation(p={}, k={}, frequency_standard={:.2})",
            self.inner.p(),
            self.inner.k(),
            self.inner.frequency_standard()
        )
    }
}

/// A fitted Bühlmann–Straub model, from ``buhlmann_straub``.
#[pyclass(name = "BuhlmannStraubFit", module = "prospicio.credibility", frozen)]
pub(crate) struct PyBuhlmannStraubFit {
    inner: BuhlmannStraubFit,
}

#[pymethods]
impl PyBuhlmannStraubFit {
    /// The credibility-weighted collective mean.
    #[getter]
    fn collective_mean(&self) -> f64 {
        self.inner.collective_mean
    }

    /// The between-group variance ``a`` (may be negative; the factors are
    /// then 0).
    #[getter]
    fn between_variance(&self) -> f64 {
        self.inner.between_variance
    }

    /// The within-group variance ``s2``.
    #[getter]
    fn within_variance(&self) -> f64 {
        self.inner.within_variance
    }

    /// Bühlmann's ``k = s2 / a`` (``inf`` when ``a <= 0``).
    #[getter]
    fn k(&self) -> f64 {
        self.inner.k()
    }

    /// Each group's weighted mean.
    #[getter]
    fn group_means(&self) -> Vec<f64> {
        self.inner.group_means.clone()
    }

    /// Each group's total weight.
    #[getter]
    fn group_weights(&self) -> Vec<f64> {
        self.inner.group_weights.clone()
    }

    /// Each group's credibility factor.
    #[getter]
    fn credibility(&self) -> Vec<f64> {
        self.inner.credibility.clone()
    }

    /// Each group's credibility premium.
    #[getter]
    fn premiums(&self) -> Vec<f64> {
        self.inner.premiums.clone()
    }

    /// Iterations of the iterative estimator (0 for the unbiased one).
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    fn __repr__(&self) -> String {
        format!(
            "BuhlmannStraubFit(collective_mean={:?}, between_variance={:?}, within_variance={:?})",
            self.inner.collective_mean, self.inner.between_variance, self.inner.within_variance
        )
    }
}

/// The Bühlmann–Straub credibility model, as R's ``actuar::cm(~group)``:
/// group ``i`` in period ``j`` has ratio ``X_ij`` (loss per unit of
/// exposure) with weight ``w_ij``. Without weights it is Bühlmann's model.
///
/// Parameters
/// ----------
/// ratios : list of list of float
///     One row per group, one column per period; NaN for a missing cell.
/// weights : list of list of float, optional
///     The same shape, NaN in the same places; equal weights if omitted.
/// estimator : {"unbiased", "iterative"}, default "unbiased"
///     The between variance: Bühlmann and Straub's unbiased estimator
///     (``actuar``'s ``"Buhlmann-Gisler"`` and ``"Ohlsson"``), or Bichsel
///     and Straub's iterative one.
/// tol : float, default 1.49e-8
///     Relative tolerance of the iterative estimator.
/// max_iter : int, default 100
///
/// Returns
/// -------
/// BuhlmannStraubFit
///
/// Raises
/// ------
/// ValueError
///     For fewer than two groups, too few observations, mismatched shapes
///     or missing cells, or missing ratios without weights.
///
/// Examples
/// --------
/// >>> from prospicio.credibility import buhlmann_straub
/// >>> fit = buhlmann_straub([[1.0, 2.0, 3.0], [4.0, 5.0, 9.0]])
/// >>> fit.within_variance
/// 4.0
#[pyfunction]
#[pyo3(signature = (ratios, weights = None, estimator = "unbiased", tol = 1.4901161193847656e-8, max_iter = 100))]
pub(crate) fn buhlmann_straub(
    ratios: Vec<Vec<f64>>,
    weights: Option<Vec<Vec<f64>>>,
    estimator: &str,
    tol: f64,
    max_iter: usize,
) -> PyResult<PyBuhlmannStraubFit> {
    let estimator = match estimator {
        "unbiased" => StructureEstimator::Unbiased,
        "iterative" => StructureEstimator::Iterative { tol, max_iter },
        other => {
            return Err(PyValueError::new_err(format!(
                "estimator must be \"unbiased\" or \"iterative\", got {other:?}"
            )));
        }
    };
    let inner = BuhlmannStraub::new(estimator)
        .fit(&ratios, weights.as_deref())
        .map_err(to_py)?;
    Ok(PyBuhlmannStraubFit { inner })
}
