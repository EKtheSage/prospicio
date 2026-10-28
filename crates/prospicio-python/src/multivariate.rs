//! `prospicio.risk.MultivariateNormal` and `MultivariateLognormal`
//! (Probability lane), over `prospicio_prob::multivariate`, and the joint
//! behind `PredictiveDistribution.from_gaussian`.

use prospicio_prob::{
    ComponentKey, GaussianJoint, KeyValue, MultivariateLognormal, MultivariateNormal,
    PredictiveDistribution,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::distributions::{KeyArg, PyLognormal, PyPredictiveDistribution, key_from_py};
use crate::normal::PyNormal;
use crate::to_py;

/// A square matrix from nested lists, row-major.
fn flatten_matrix(m: Vec<Vec<f64>>, name: &str) -> PyResult<Vec<f64>> {
    let d = m.len();
    if m.iter().any(|row| row.len() != d) {
        return Err(PyValueError::new_err(format!(
            "{name} must be a square matrix"
        )));
    }
    Ok(m.into_iter().flatten().collect())
}

/// A row-major `d × d` matrix as nested lists.
fn nest(m: &[f64], d: usize) -> Vec<Vec<f64>> {
    m.chunks(d).map(<[f64]>::to_vec).collect()
}

/// The multivariate normal with mean ``mean`` and covariance ``cov``.
///
/// Inputs such as a reserve by origin with its covariance across origins,
/// or lines joined by a correlation matrix, arrive as a mean and a
/// covariance; this holds the joint with its exact marginals and total,
/// and ``PredictiveDistribution.from_gaussian`` draws it by seed.
///
/// Parameters
/// ----------
/// mean : list of float
/// cov : list of list of float
///     Symmetric and positive definite, one row and column per component.
///
/// Raises
/// ------
/// ValueError
///     If a value is not finite, ``cov`` is not square with one row per
///     component, not symmetric or not positive definite (a zero variance
///     or a perfect correlation is singular and refused).
///
/// Examples
/// --------
/// >>> from prospicio.risk import MultivariateNormal
/// >>> mvn = MultivariateNormal([100.0, 50.0], [[400.0, 120.0], [120.0, 100.0]])
/// >>> t = mvn.total()
/// >>> t.mean(), round(t.variance(), 9)
/// (150.0, 740.0)
/// >>> mvn.marginal(1).sigma
/// 10.0
#[pyclass(name = "MultivariateNormal", module = "prospicio.risk", frozen)]
pub(crate) struct PyMultivariateNormal {
    inner: MultivariateNormal,
}

#[pymethods]
impl PyMultivariateNormal {
    #[new]
    fn new(mean: Vec<f64>, cov: Vec<Vec<f64>>) -> PyResult<Self> {
        let cov = flatten_matrix(cov, "cov")?;
        Ok(Self {
            inner: MultivariateNormal::new(mean, cov).map_err(to_py)?,
        })
    }

    /// Number of components.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// The mean vector.
    #[getter]
    fn mean(&self) -> Vec<f64> {
        self.inner.mean().to_vec()
    }

    /// The covariance matrix.
    #[getter]
    fn cov(&self) -> Vec<Vec<f64>> {
        nest(self.inner.cov(), self.inner.dim())
    }

    /// The correlation matrix.
    ///
    /// Returns
    /// -------
    /// list of list of float
    fn correlation(&self) -> Vec<Vec<f64>> {
        nest(&self.inner.correlation(), self.inner.dim())
    }

    /// Component ``j``'s marginal, exact.
    ///
    /// Parameters
    /// ----------
    /// j : int
    ///
    /// Returns
    /// -------
    /// Normal
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``j`` is out of range.
    fn marginal(&self, j: usize) -> PyResult<PyNormal> {
        Ok(PyNormal {
            inner: self.inner.marginal(j).map_err(to_py)?,
        })
    }

    /// The total over all components, exact: normal with mean the sum of
    /// the means and variance the sum of the covariance's entries.
    ///
    /// Returns
    /// -------
    /// Normal
    fn total(&self) -> PyNormal {
        PyNormal {
            inner: self.inner.total(),
        }
    }

    fn __repr__(&self) -> String {
        format!("MultivariateNormal(dim={})", self.inner.dim())
    }
}

/// The multivariate lognormal: ``exp(Y)`` for ``Y`` multivariate normal.
///
/// Built from a mean and covariance on the original scale, matched
/// exactly (``from_moments``: ``log_cov[i][j] = ln(1 + cov[i][j] / (m_i
/// m_j))``, ``mu_i = ln m_i - log_cov[i][i] / 2``), or from the log scale
/// (``from_log_scale``).
///
/// Examples
/// --------
/// >>> from prospicio.risk import MultivariateLognormal
/// >>> mvl = MultivariateLognormal.from_moments([100.0, 50.0], [[400.0, 120.0], [120.0, 100.0]])
/// >>> [round(m, 9) for m in mvl.mean]
/// [100.0, 50.0]
/// >>> round(mvl.cov[0][1], 9), round(mvl.total_variance(), 6)
/// (120.0, 740.0)
#[pyclass(name = "MultivariateLognormal", module = "prospicio.risk", frozen)]
pub(crate) struct PyMultivariateLognormal {
    inner: MultivariateLognormal,
}

#[pymethods]
impl PyMultivariateLognormal {
    /// The lognormal with mean ``mean`` and covariance ``cov`` on the
    /// original scale.
    ///
    /// Parameters
    /// ----------
    /// mean : list of float
    ///     Positive.
    /// cov : list of list of float
    ///
    /// Returns
    /// -------
    /// MultivariateLognormal
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a mean is not positive, ``cov`` is not a symmetric square
    ///     matrix, a covariance is at or below ``-m_i m_j`` (no lognormal
    ///     has it), or the covariance on the log scale is not positive
    ///     definite (possible even when ``cov`` is).
    #[staticmethod]
    fn from_moments(mean: Vec<f64>, cov: Vec<Vec<f64>>) -> PyResult<Self> {
        let cov = flatten_matrix(cov, "cov")?;
        Ok(Self {
            inner: MultivariateLognormal::from_moments(mean, cov).map_err(to_py)?,
        })
    }

    /// The lognormal ``exp(Y)``, ``Y`` normal with mean ``mu`` and
    /// covariance ``log_cov``.
    ///
    /// Parameters
    /// ----------
    /// mu : list of float
    /// log_cov : list of list of float
    ///
    /// Returns
    /// -------
    /// MultivariateLognormal
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``MultivariateNormal``.
    #[staticmethod]
    fn from_log_scale(mu: Vec<f64>, log_cov: Vec<Vec<f64>>) -> PyResult<Self> {
        let log_cov = flatten_matrix(log_cov, "log_cov")?;
        Ok(Self {
            inner: MultivariateLognormal::from_log_scale(mu, log_cov).map_err(to_py)?,
        })
    }

    /// Number of components.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// The mean on the original scale, exact.
    #[getter]
    fn mean(&self) -> Vec<f64> {
        self.inner.mean()
    }

    /// The covariance on the original scale, exact.
    #[getter]
    fn cov(&self) -> Vec<Vec<f64>> {
        nest(&self.inner.cov(), self.inner.dim())
    }

    /// The mean on the log scale.
    #[getter]
    fn mu(&self) -> Vec<f64> {
        self.inner.log_normal().mean().to_vec()
    }

    /// The covariance on the log scale.
    #[getter]
    fn log_cov(&self) -> Vec<Vec<f64>> {
        nest(self.inner.log_normal().cov(), self.inner.dim())
    }

    /// The correlation matrix on the original scale.
    ///
    /// Returns
    /// -------
    /// list of list of float
    fn correlation(&self) -> Vec<Vec<f64>> {
        nest(&self.inner.correlation(), self.inner.dim())
    }

    /// Component ``j``'s marginal, exact.
    ///
    /// Parameters
    /// ----------
    /// j : int
    ///
    /// Returns
    /// -------
    /// Lognormal
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``j`` is out of range.
    fn marginal(&self, j: usize) -> PyResult<PyLognormal> {
        Ok(PyLognormal {
            inner: self.inner.marginal(j).map_err(to_py)?,
        })
    }

    /// The total's mean, exact. A sum of lognormals is not lognormal; draw
    /// the total with ``PredictiveDistribution.from_gaussian``.
    ///
    /// Returns
    /// -------
    /// float
    fn total_mean(&self) -> f64 {
        self.inner.total_mean()
    }

    /// The total's variance, exact.
    ///
    /// Returns
    /// -------
    /// float
    fn total_variance(&self) -> f64 {
        self.inner.total_variance()
    }

    fn __repr__(&self) -> String {
        format!("MultivariateLognormal(dim={})", self.inner.dim())
    }
}

/// `PredictiveDistribution.from_gaussian`'s body: either joint, keys
/// defaulting to `(0,), (1,), ...` under `"component"`.
pub(crate) fn from_gaussian(
    py: Python<'_>,
    joint: &Bound<'_, PyAny>,
    n_sims: usize,
    seed: u64,
    keys: Option<Vec<Vec<KeyArg>>>,
    dims: Option<Vec<String>>,
) -> PyResult<PyPredictiveDistribution> {
    let normal;
    let lognormal;
    let joint: &dyn GaussianJoint =
        if let Ok(j) = joint.extract::<PyRef<'_, PyMultivariateNormal>>() {
            normal = j.inner.clone();
            &normal
        } else if let Ok(j) = joint.extract::<PyRef<'_, PyMultivariateLognormal>>() {
            lognormal = j.inner.clone();
            &lognormal
        } else {
            return Err(PyTypeError::new_err(
                "expected a MultivariateNormal or MultivariateLognormal",
            ));
        };
    let components: Vec<ComponentKey> = match keys {
        Some(keys) => keys.into_iter().map(key_from_py).collect(),
        None => (0..joint.dim())
            .map(|j| vec![KeyValue::Int(j as i64)])
            .collect(),
    };
    let dims = dims.unwrap_or_else(|| vec!["component".into()]);
    let inner = py
        .detach(|| PredictiveDistribution::from_gaussian(dims, components, joint, n_sims, seed))
        .map_err(to_py)?;
    Ok(PyPredictiveDistribution { inner })
}
