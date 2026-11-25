//! Python bindings for `prospicio_models::boosting` and
//! `metrics::mean_deviance`: the arithmetic the Python and R boosters
//! share (#977).

use prospicio_models::{boosting, metrics};
use pyo3::prelude::*;

use crate::models::{family, link};
use crate::to_py;

/// Weighted mean deviance ``sum(w d(y, mu)) / sum(w)`` of a family (the
/// number of rows when there are no weights): the score to compare models
/// on held-out data.
///
/// Parameters
/// ----------
/// family : str
/// y : list of float
/// mu : list of float
/// weights : list of float, optional
/// theta : float, optional
/// power : float, optional
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from prospicio.models import mean_deviance
/// >>> mean_deviance("gaussian", [1.0, 3.0], [2.0, 2.0])
/// 1.0
#[pyfunction]
#[pyo3(signature = (family, y, mu, weights = None, theta = None, power = None))]
pub(crate) fn mean_deviance(
    family: &str,
    y: Vec<f64>,
    mu: Vec<f64>,
    weights: Option<Vec<f64>>,
    theta: Option<f64>,
    power: Option<f64>,
) -> PyResult<f64> {
    let f = self::family(family, theta, power)?;
    metrics::mean_deviance(f, &y, &mu, weights.as_deref()).map_err(to_py)
}

/// The smallest ``y`` whose cumulative weight, in increasing ``y``,
/// reaches ``alpha`` of the total weight.
///
/// Parameters
/// ----------
/// y : list of float
/// weights : list of float
/// alpha : float
///     In ``[0, 1]``.
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from prospicio.models import weighted_quantile
/// >>> weighted_quantile([3.0, 1.0, 2.0], [1.0, 1.0, 2.0], 0.5)
/// 2.0
#[pyfunction]
pub(crate) fn weighted_quantile(y: Vec<f64>, weights: Vec<f64>, alpha: f64) -> PyResult<f64> {
    boosting::weighted_quantile(&y, &weights, alpha).map_err(to_py)
}

/// The constant link-scale score a booster starts from:
/// ``log(sum(w y) / sum(w exp(offset)))`` for a log link (the weighted
/// total must be positive), ``sum(w (y - offset)) / sum(w)`` for an
/// identity link.
///
/// Parameters
/// ----------
/// link : {"log", "identity"}
/// y, weights, offset : list of float
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from prospicio.models import base_score
/// >>> base_score("identity", [1.0, 3.0], [1.0, 1.0], [0.0, 0.0])
/// 2.0
#[pyfunction]
pub(crate) fn base_score(
    link: &str,
    y: Vec<f64>,
    weights: Vec<f64>,
    offset: Vec<f64>,
) -> PyResult<f64> {
    let l = self::link(Some(link), prospicio_models::Family::Gaussian, None)?;
    boosting::base_score(l, &y, &weights, &offset).map_err(to_py)
}

/// Each row's Pearson term ``w (y - mu)^2 / V(mu)``, ``V`` the family's
/// variance function.
///
/// Parameters
/// ----------
/// family : str
/// y, mu, weights : list of float
/// theta : float, optional
/// power : float, optional
///
/// Returns
/// -------
/// list of float
#[pyfunction]
#[pyo3(signature = (family, y, mu, weights, theta = None, power = None))]
pub(crate) fn pearson_terms(
    family: &str,
    y: Vec<f64>,
    mu: Vec<f64>,
    weights: Vec<f64>,
    theta: Option<f64>,
    power: Option<f64>,
) -> PyResult<Vec<f64>> {
    let f = self::family(family, theta, power)?;
    boosting::pearson_terms(f, &y, &mu, &weights).map_err(to_py)
}

/// The Pearson dispersion ``sum(w (y - mu)^2 / V(mu)) / n``.
///
/// Parameters
/// ----------
/// family : str
/// y, mu, weights : list of float
/// theta : float, optional
/// power : float, optional
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from prospicio.models import pearson_dispersion
/// >>> pearson_dispersion("poisson", [1.0, 4.0], [2.0, 2.0], [1.0, 1.0])
/// 1.25
#[pyfunction]
#[pyo3(signature = (family, y, mu, weights, theta = None, power = None))]
pub(crate) fn pearson_dispersion(
    family: &str,
    y: Vec<f64>,
    mu: Vec<f64>,
    weights: Vec<f64>,
    theta: Option<f64>,
    power: Option<f64>,
) -> PyResult<f64> {
    let f = self::family(family, theta, power)?;
    boosting::pearson_dispersion(f, &y, &mu, &weights).map_err(to_py)
}

/// The indices ``i`` (from 1) whose step from ``means[i - 1]`` goes
/// against ``direction`` by more than ``tol`` times the curve's largest
/// absolute value.
///
/// Parameters
/// ----------
/// means : list of float
/// direction : {1, -1}
/// tol : float, default 1e-9
///
/// Returns
/// -------
/// list of int
///
/// Examples
/// --------
/// >>> from prospicio.models import monotone_violations
/// >>> monotone_violations([1.0, 2.0, 1.5, 3.0], 1)
/// [2]
#[pyfunction]
#[pyo3(signature = (means, direction, tol = 1e-9))]
pub(crate) fn monotone_violations(
    means: Vec<f64>,
    direction: i32,
    tol: f64,
) -> PyResult<Vec<usize>> {
    boosting::monotone_violations(&means, direction, tol).map_err(to_py)
}
