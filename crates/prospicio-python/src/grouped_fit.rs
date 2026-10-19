//! `prospicio.distributions.fit_grouped`, `fit_lev_distance` and
//! `grouped_goodness_of_fit` (Probability lane), over
//! `prospicio_prob::grouped_fit`.

use std::collections::BTreeMap;

use prospicio_prob::grouped_fit::{
    GroupedFit, GroupedLosses, GroupedMethod, LevDistanceFit, fit_grouped as fit_grouped_rs,
    fit_lev_distance as fit_lev_distance_rs,
};
use prospicio_prob::severity_fit::Family;
use pyo3::prelude::*;

use crate::to_py;

fn by_name(family: Family, values: impl IntoIterator<Item = f64>) -> BTreeMap<&'static str, f64> {
    family
        .parameter_names()
        .iter()
        .copied()
        .zip(values)
        .collect()
}

/// A severity fitted to grouped losses, from ``fit_grouped``: the
/// distribution, its parameters with standard errors and covariance (from
/// the grouped data's Fisher information), the expected count in each band
/// and the chi-square test.
#[pyclass(name = "GroupedFit", module = "prospicio.distributions", frozen)]
pub(crate) struct PyGroupedFit {
    inner: GroupedFit,
}

#[pymethods]
impl PyGroupedFit {
    /// The family's name.
    #[getter]
    fn family(&self) -> &'static str {
        self.inner.family().name()
    }

    /// The objective it was fitted by, ``"ml"`` or ``"chi_square"``.
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method().name()
    }

    /// The fitted distribution.
    #[getter]
    fn distribution(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::distributions::dist_to_py(py, self.inner.dist().clone())
    }

    /// The parameters by name.
    #[getter]
    fn parameters(&self) -> BTreeMap<&'static str, f64> {
        by_name(self.inner.family(), self.inner.parameters().iter().copied())
    }

    /// Each parameter's standard error, by name; NaN where the information
    /// is singular.
    #[getter]
    fn standard_errors(&self) -> BTreeMap<&'static str, f64> {
        by_name(self.inner.family(), self.inner.standard_errors())
    }

    /// The parameters' covariance, one row per parameter in the family's
    /// order: the inverse of the grouped data's Fisher information, which
    /// both methods attain asymptotically.
    #[getter]
    fn covariance(&self) -> Vec<Vec<f64>> {
        let n = self.inner.parameters().len();
        self.inner
            .covariance()
            .chunks(n)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// Pearson's chi-square at the fit.
    #[getter]
    fn chi_square(&self) -> f64 {
        self.inner.chi_square()
    }

    /// The multinomial log-likelihood at the fit.
    #[getter]
    fn log_likelihood(&self) -> f64 {
        self.inner.log_likelihood()
    }

    /// The expected count in each band at the fit.
    #[getter]
    fn expected(&self) -> Vec<f64> {
        self.inner.expected().to_vec()
    }

    /// The chi-square test's degrees of freedom: bands with a positive
    /// expected count, less one, less the number of parameters.
    #[getter]
    fn degrees_of_freedom(&self) -> usize {
        self.inner.degrees_of_freedom()
    }

    /// The chi-square test's p-value; NaN with no degrees of freedom.
    #[getter]
    fn p_value(&self) -> f64 {
        self.inner.p_value()
    }

    /// Whether the simplex met its tolerance.
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged()
    }

    fn __repr__(&self) -> String {
        format!(
            "GroupedFit({}, {}, {:?}, chi_square={:.6})",
            self.inner.family().name(),
            self.inner.method().name(),
            self.inner.parameters(),
            self.inner.chi_square()
        )
    }
}

/// Fit of a lognormal, gamma, Weibull, Burr or Lomax severity to grouped
/// losses, a count of losses in each size band, by maximum likelihood on
/// the bands or by minimum chi-square (Bahnemann, *Distributions for
/// Actuaries*, section 2.4). The losses are conditioned on lying within the
/// bands: a first bound above 0 is a deductible's truncation and a finite
/// last bound a truncation from above.
///
/// Parameters
/// ----------
/// family : {"lognormal", "gamma", "weibull", "burr", "lomax"}
/// bounds : list of float
///     The band bounds, one more than ``counts``, non-negative and strictly
///     increasing; only the last may be ``math.inf``.
/// counts : list of float
///     The count (or any weight) in each band ``(bounds[k], bounds[k + 1]]``.
/// method : {"ml", "chi_square"}
///     Maximum likelihood (the default) or minimum chi-square.
///
/// Returns
/// -------
/// GroupedFit
///
/// Raises
/// ------
/// ValueError
///     If the bands or counts are invalid or the method is unknown.
///
/// Examples
/// --------
/// Bahnemann's P2.16, a gamma by minimum chi-square:
///
/// >>> import math
/// >>> from prospicio.distributions import fit_grouped
/// >>> bounds = [0, 1000, 1500, 2000, 2500, 3000, math.inf]
/// >>> fit = fit_grouped("gamma", bounds, [42, 61, 47, 26, 14, 10], "chi_square")
/// >>> round(fit.parameters["shape"], 4), round(fit.parameters["scale"], 2)
/// (4.7432, 337.31)
#[pyfunction]
#[pyo3(signature = (family, bounds, counts, method = "ml"))]
pub(crate) fn fit_grouped(
    py: Python<'_>,
    family: &str,
    bounds: Vec<f64>,
    counts: Vec<f64>,
    method: &str,
) -> PyResult<PyGroupedFit> {
    let family = Family::from_name(family).map_err(to_py)?;
    let method = GroupedMethod::from_name(method).map_err(to_py)?;
    let data = GroupedLosses::new(bounds, counts).map_err(to_py)?;
    let inner = py
        .detach(|| fit_grouped_rs(family, &data, method))
        .map_err(to_py)?;
    Ok(PyGroupedFit { inner })
}

/// A severity against grouped losses: each band's probability conditional
/// on the range of the bands, the expected counts, Pearson's chi-square and
/// the multinomial log-likelihood.
///
/// Parameters
/// ----------
/// severity : a distribution
/// bounds : list of float
/// counts : list of float
///
/// Returns
/// -------
/// dict
///     ``probabilities`` and ``expected`` (lists), ``chi_square`` and
///     ``log_likelihood`` (floats).
///
/// Examples
/// --------
/// Bahnemann's Ex 2.7:
///
/// >>> import math
/// >>> from prospicio.distributions import Gamma, grouped_goodness_of_fit
/// >>> bounds = [0, 1000, 1500, 2000, 2500, 3000, math.inf]
/// >>> g = grouped_goodness_of_fit(Gamma(3.907288, 397.931), bounds, [42, 61, 47, 26, 14, 10])
/// >>> round(g["chi_square"], 3)
/// 3.095
#[pyfunction]
pub(crate) fn grouped_goodness_of_fit<'py>(
    py: Python<'py>,
    severity: &Bound<'py, PyAny>,
    bounds: Vec<f64>,
    counts: Vec<f64>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use pyo3::types::PyDictMethods;
    let dist = crate::distributions::extract_dist(severity)?;
    let data = GroupedLosses::new(bounds, counts).map_err(to_py)?;
    let p = data.probabilities(&dist).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("the severity gives the bands no probability")
    })?;
    let n = data.total();
    let d = pyo3::types::PyDict::new(py);
    d.set_item("expected", p.iter().map(|&p| n * p).collect::<Vec<_>>())?;
    d.set_item("probabilities", p)?;
    d.set_item("chi_square", data.chi_square(&dist))?;
    d.set_item("log_likelihood", data.log_likelihood(&dist))?;
    Ok(d)
}

/// A severity fitted by the minimum distance between its limited expected
/// values and empirical ones, from ``fit_lev_distance``.
#[pyclass(name = "LevDistanceFit", module = "prospicio.distributions", frozen)]
pub(crate) struct PyLevDistanceFit {
    inner: LevDistanceFit,
}

#[pymethods]
impl PyLevDistanceFit {
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
    fn parameters(&self) -> BTreeMap<&'static str, f64> {
        by_name(self.inner.family(), self.inner.parameters().iter().copied())
    }

    /// The minimized weighted squared distance.
    #[getter]
    fn distance(&self) -> f64 {
        self.inner.distance()
    }

    /// The fitted limited expected value at each size.
    #[getter]
    fn fitted(&self) -> Vec<f64> {
        self.inner.fitted().to_vec()
    }

    /// Whether the simplex met its tolerance.
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged()
    }

    fn __repr__(&self) -> String {
        format!(
            "LevDistanceFit({}, {:?}, distance={:.6})",
            self.inner.family().name(),
            self.inner.parameters(),
            self.inner.distance()
        )
    }
}

/// Fit of a lognormal, gamma, Weibull, Burr or Lomax severity by minimizing
/// ``sum(w * (E[X; x] - e) ** 2)``, the weighted squared distance between
/// its limited expected values and the empirical ones ``e`` at the sizes
/// ``x`` (Bahnemann's minimum distance, section 2.5). Empirical limited
/// means are exact from grouped data with each band's total, and stay exact
/// under a policy limit at or above the last size.
///
/// Parameters
/// ----------
/// family : {"lognormal", "gamma", "weibull", "burr", "lomax"}
/// sizes : list of float
///     Positive, strictly increasing limits.
/// empirical : list of float
///     The empirical limited expected value at each size.
/// weights : list of float, optional
///     Non-negative weights, 1 by default.
/// start : list of float, optional
///     Starting parameters in the family's order; by default, from losses
///     spread to match the empirical limited means.
///
/// Returns
/// -------
/// LevDistanceFit
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Lognormal, fit_lev_distance
/// >>> x = [500.0, 1000.0, 2000.0, 5000.0, 10000.0]
/// >>> e = [Lognormal(7.0, 1.2).lev(v) for v in x]
/// >>> round(fit_lev_distance("lognormal", x, e).parameters["meanlog"], 5)
/// 7.0
#[pyfunction]
#[pyo3(signature = (family, sizes, empirical, weights = None, start = None))]
pub(crate) fn fit_lev_distance(
    py: Python<'_>,
    family: &str,
    sizes: Vec<f64>,
    empirical: Vec<f64>,
    weights: Option<Vec<f64>>,
    start: Option<Vec<f64>>,
) -> PyResult<PyLevDistanceFit> {
    let family = Family::from_name(family).map_err(to_py)?;
    let inner = py
        .detach(|| {
            fit_lev_distance_rs(
                family,
                &sizes,
                &empirical,
                weights.as_deref(),
                start.as_deref(),
            )
        })
        .map_err(to_py)?;
    Ok(PyLevDistanceFit { inner })
}
