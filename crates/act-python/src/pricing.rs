//! `actuarialrs.pricing` (Aggregate lane): the collective model, layer
//! rating and reinsurance tower matching over `act_aggregate` and
//! `act_pricing` (`docs/design/pareto.md`).

use act_aggregate::CollectiveModel;
use act_pricing::layer::XsLayer;
use act_pricing::tower::{Reference, SelectionRule, TowerModel};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::aggregate::{AnyCount, PyEventSet};
use crate::distributions::AnySeverity;
use crate::pareto::PyPiecewisePareto;
use crate::to_py;

/// The collective risk model: a claim count and a severity, with layer
/// moments in closed form.
///
/// For the layer ``limit`` xs ``attachment`` applied to each loss, with
/// ``Y`` the loss to the layer from one claim, the aggregate has mean
/// ``E[N] E[Y]`` and variance ``E[N] Var[Y] + Var[N] E[Y]**2``.
///
/// Parameters
/// ----------
/// frequency : Poisson, NegativeBinomial or Binomial
/// severity : Lognormal, Grid, Pareto, PiecewisePareto, LogAffinePareto or GeneralizedPareto
///
/// Examples
/// --------
/// >>> from actuarialrs.distributions import Pareto, claim_count
/// >>> from actuarialrs.pricing import CollectiveModel
/// >>> m = CollectiveModel(claim_count(2.0, 1.5), Pareto(1e6, 2.0))
/// >>> round(m.layer_mean(4e6, 1e6))
/// 1600000
/// >>> m.excess_frequency(2e6)
/// 0.5
#[pyclass(name = "CollectiveModel", module = "actuarialrs.pricing", frozen)]
pub(crate) struct PyCollectiveModel {
    inner: CollectiveModel<AnyCount, AnySeverity>,
}

#[pymethods]
impl PyCollectiveModel {
    #[new]
    fn new(frequency: &Bound<'_, PyAny>, severity: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = CollectiveModel::new(
            AnyCount::extract(frequency)?,
            AnySeverity::extract(severity)?,
        );
        Ok(Self { inner })
    }

    /// Expected number of losses above ``x``.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///
    /// Returns
    /// -------
    /// float
    fn excess_frequency(&self, x: f64) -> f64 {
        self.inner.excess_frequency(x)
    }

    /// Expected aggregate loss to the layer ``limit`` xs ``attachment``.
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
    fn layer_mean(&self, limit: f64, attachment: f64) -> f64 {
        self.inner.layer_mean(limit, attachment)
    }

    /// Variance of the aggregate loss to the layer.
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

    /// Standard deviation of the aggregate loss to the layer.
    ///
    /// Parameters
    /// ----------
    /// limit : float
    /// attachment : float
    ///
    /// Returns
    /// -------
    /// float
    fn layer_std(&self, limit: f64, attachment: f64) -> f64 {
        self.inner.layer_std_dev(limit, attachment)
    }

    /// Expected aggregate loss.
    ///
    /// Returns
    /// -------
    /// float
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// Variance of the aggregate loss.
    ///
    /// Returns
    /// -------
    /// float
    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// ``n_sims`` simulated years of individual losses.
    ///
    /// Parameters
    /// ----------
    /// n_sims : int
    /// seed : int
    ///
    /// Returns
    /// -------
    /// EventSet
    fn simulate(&self, py: Python<'_>, n_sims: usize, seed: u64) -> PyResult<PyEventSet> {
        let inner = py
            .detach(|| self.inner.simulate(n_sims, seed))
            .map_err(to_py)?;
        Ok(PyEventSet::from(inner))
    }
}

/// A frequency and a piecewise Pareto severity that reproduce a tower,
/// a PML curve or a set of references.
#[pyclass(name = "TowerModel", module = "actuarialrs.pricing", frozen)]
pub(crate) struct PyTowerModel {
    inner: TowerModel,
}

#[pymethods]
impl PyTowerModel {
    /// Expected number of losses a year above the lowest threshold.
    #[getter]
    fn frequency(&self) -> f64 {
        self.inner.frequency
    }

    /// The fitted severity.
    #[getter]
    fn severity(&self) -> PyPiecewisePareto {
        PyPiecewisePareto {
            inner: self.inner.severity.clone(),
        }
    }

    /// Expected number of losses a year above ``x``.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///
    /// Returns
    /// -------
    /// float
    fn excess_frequency(&self, x: f64) -> f64 {
        self.inner.excess_frequency(x)
    }

    /// Expected loss a year to ``limit`` xs ``attachment``.
    ///
    /// Parameters
    /// ----------
    /// limit : float
    /// attachment : float
    ///
    /// Returns
    /// -------
    /// float
    fn layer_loss(&self, limit: f64, attachment: f64) -> f64 {
        self.inner.layer_loss(limit, attachment)
    }

    fn __repr__(&self) -> String {
        format!(
            "TowerModel(frequency={:?}, t={:?}, alpha={:?})",
            self.inner.frequency,
            self.inner.severity.thresholds(),
            self.inner.severity.alphas()
        )
    }
}

fn xs(limit: f64, attachment: f64) -> PyResult<XsLayer> {
    XsLayer::new(limit, attachment).map_err(to_py)
}

fn rule(name: &str) -> PyResult<SelectionRule> {
    match name {
        "minimize" => Ok(SelectionRule::MinimizeAlphaRatio),
        "midpoint" => Ok(SelectionRule::Midpoint),
        _ => Err(PyValueError::new_err(
            "rule must be \"minimize\" or \"midpoint\"",
        )),
    }
}

/// Increased limit factor ``LEV(limit) / LEV(basic_limit)``.
///
/// Parameters
/// ----------
/// severity : a severity
/// limit : float
/// basic_limit : float
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from actuarialrs.distributions import Pareto
/// >>> from actuarialrs.pricing import ilf
/// >>> round(ilf(Pareto(100.0, 2.0), 1000.0, 200.0), 12)
/// 1.266666666667
#[pyfunction]
pub(crate) fn ilf(severity: &Bound<'_, PyAny>, limit: f64, basic_limit: f64) -> PyResult<f64> {
    let sev = AnySeverity::extract(severity)?;
    act_pricing::layer::ilf(&sev, limit, basic_limit).map_err(to_py)
}

/// Loss elimination ratio of a deductible, ``LEV(deductible) / E[X]``.
///
/// Parameters
/// ----------
/// severity : a severity
/// deductible : float
///
/// Returns
/// -------
/// float
#[pyfunction]
pub(crate) fn loss_elimination_ratio(
    severity: &Bound<'_, PyAny>,
    deductible: f64,
) -> PyResult<f64> {
    let sev = AnySeverity::extract(severity)?;
    act_pricing::layer::loss_elimination_ratio(&sev, deductible).map_err(to_py)
}

/// Expected loss of layer ``to`` per unit of expected loss of layer
/// ``from_``, under a Pareto with this alpha (and truncation).
///
/// Parameters
/// ----------
/// from_ : tuple of float
///     ``(limit, attachment)``.
/// to : tuple of float
///     ``(limit, attachment)``.
/// alpha : float
/// truncation : float, optional
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from actuarialrs.pricing import pareto_extrapolation
/// >>> round(pareto_extrapolation((1e6, 1e6), (2e6, 2e6), 2.0), 12)
/// 0.5
#[pyfunction]
#[pyo3(signature = (from_, to, alpha, truncation = None))]
pub(crate) fn pareto_extrapolation(
    from_: (f64, f64),
    to: (f64, f64),
    alpha: f64,
    truncation: Option<f64>,
) -> PyResult<f64> {
    act_pricing::layer::pareto_extrapolation(
        xs(from_.0, from_.1)?,
        xs(to.0, to.1)?,
        alpha,
        truncation,
    )
    .map_err(to_py)
}

/// The Pareto alpha at which two layers have the given expected losses.
///
/// Parameters
/// ----------
/// a : tuple of float
///     ``(limit, attachment, expected_loss)``.
/// b : tuple of float
///     ``(limit, attachment, expected_loss)``; one layer must lie above
///     the other.
/// truncation : float, optional
///
/// Returns
/// -------
/// float
#[pyfunction]
#[pyo3(signature = (a, b, truncation = None))]
pub(crate) fn alpha_between_layers(
    a: (f64, f64, f64),
    b: (f64, f64, f64),
    truncation: Option<f64>,
) -> PyResult<f64> {
    act_pricing::layer::alpha_between_layers((xs(a.0, a.1)?, a.2), (xs(b.0, b.1)?, b.2), truncation)
        .map_err(to_py)
}

/// The Pareto alpha at which ``frequency`` losses a year above
/// ``threshold`` give the layer an expected loss of ``expected_loss``.
///
/// Parameters
/// ----------
/// threshold : float
/// frequency : float
/// limit : float
/// attachment : float
/// expected_loss : float
/// truncation : float, optional
///
/// Returns
/// -------
/// float
#[pyfunction]
#[pyo3(signature = (threshold, frequency, limit, attachment, expected_loss, truncation = None))]
pub(crate) fn alpha_between_frequency_and_layer(
    threshold: f64,
    frequency: f64,
    limit: f64,
    attachment: f64,
    expected_loss: f64,
    truncation: Option<f64>,
) -> PyResult<f64> {
    act_pricing::layer::alpha_between_frequency_and_layer(
        threshold,
        frequency,
        xs(limit, attachment)?,
        expected_loss,
        truncation,
    )
    .map_err(to_py)
}

/// The Pareto alpha between two excess frequencies.
///
/// Parameters
/// ----------
/// threshold_1 : float
/// frequency_1 : float
/// threshold_2 : float
/// frequency_2 : float
/// truncation : float, optional
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from actuarialrs.pricing import alpha_between_frequencies
/// >>> round(alpha_between_frequencies(1e6, 4.0, 2e6, 1.0), 12)
/// 2.0
#[pyfunction]
#[pyo3(signature = (threshold_1, frequency_1, threshold_2, frequency_2, truncation = None))]
pub(crate) fn alpha_between_frequencies(
    threshold_1: f64,
    frequency_1: f64,
    threshold_2: f64,
    frequency_2: f64,
    truncation: Option<f64>,
) -> PyResult<f64> {
    act_pricing::layer::alpha_between_frequencies(
        threshold_1,
        frequency_1,
        threshold_2,
        frequency_2,
        truncation,
    )
    .map_err(to_py)
}

/// Matches a tower of contiguous layers, the last unlimited, with one
/// frequency and a piecewise Pareto severity (Riegel 2018).
///
/// Parameters
/// ----------
/// attachments : list of float
///     Increasing attachment points; layer ``i`` runs to the next one, the
///     last is unlimited.
/// layer_losses : list of float
///     Expected loss a year of each layer.
/// frequencies : list of float or None, optional
///     Expected losses a year above each attachment point; ``None`` (or a
///     ``None`` entry) to derive them.
/// rule : {"minimize", "midpoint"}, default "minimize"
///     How the free threshold inside each layer is chosen.
///
/// Returns
/// -------
/// TowerModel
///
/// Examples
/// --------
/// >>> from actuarialrs.pricing import match_tower
/// >>> m = match_tower([1000.0, 1500.0, 2000.0], [100.0, 90.0, 120.0], [0.25, None, None])
/// >>> round(m.layer_loss(500.0, 1500.0), 9)
/// 90.0
#[pyfunction]
#[pyo3(signature = (attachments, layer_losses, frequencies = None, rule = "minimize"))]
pub(crate) fn match_tower(
    py: Python<'_>,
    attachments: Vec<f64>,
    layer_losses: Vec<f64>,
    frequencies: Option<Vec<Option<f64>>>,
    rule: &str,
) -> PyResult<PyTowerModel> {
    let rule = self::rule(rule)?;
    let frequencies = frequencies.unwrap_or_default();
    let inner = py
        .detach(|| act_pricing::tower::match_tower(&attachments, &layer_losses, &frequencies, rule))
        .map_err(to_py)?;
    Ok(PyTowerModel { inner })
}

/// The model through the points of a PML curve: ``amounts[j]`` is
/// exceeded once in ``return_periods[j]`` years.
///
/// Parameters
/// ----------
/// return_periods : list of float
/// amounts : list of float
/// tail_alpha : float, default 2.0
///     Alpha above the largest amount.
/// truncation : float, optional
///     Truncation of the last piece.
///
/// Returns
/// -------
/// TowerModel
#[pyfunction]
#[pyo3(signature = (return_periods, amounts, tail_alpha = 2.0, truncation = None))]
pub(crate) fn fit_pml_curve(
    return_periods: Vec<f64>,
    amounts: Vec<f64>,
    tail_alpha: f64,
    truncation: Option<f64>,
) -> PyResult<PyTowerModel> {
    let inner =
        act_pricing::tower::fit_pml_curve(&return_periods, &amounts, tail_alpha, truncation)
            .map_err(to_py)?;
    Ok(PyTowerModel { inner })
}

/// A model that reproduces every reference: expected layer losses (which
/// may overlap or leave gaps) and excess frequencies.
///
/// Parameters
/// ----------
/// layers : list of tuple of float, optional
///     ``(limit, attachment, expected_loss)`` per layer.
/// frequencies : list of tuple of float, optional
///     ``(threshold, frequency)`` per excess frequency.
/// default_alpha : float, default 2.0
///     Alpha above the highest point, unless an unlimited layer sets it.
/// rule : {"minimize", "midpoint"}, default "minimize"
///
/// Returns
/// -------
/// TowerModel
///
/// Examples
/// --------
/// >>> from actuarialrs.pricing import fit_references
/// >>> m = fit_references([(1000.0, 1000.0, 150.0), (3000.0, 1500.0, 160.0)], [(1000.0, 0.3)])
/// >>> round(m.layer_loss(3000.0, 1500.0), 6)
/// 160.0
#[pyfunction]
#[pyo3(signature = (layers = Vec::new(), frequencies = Vec::new(), default_alpha = 2.0, rule = "minimize"))]
pub(crate) fn fit_references(
    py: Python<'_>,
    layers: Vec<(f64, f64, f64)>,
    frequencies: Vec<(f64, f64)>,
    default_alpha: f64,
    rule: &str,
) -> PyResult<PyTowerModel> {
    let rule = self::rule(rule)?;
    let mut refs = Vec::with_capacity(layers.len() + frequencies.len());
    for (limit, attachment, expected_loss) in layers {
        refs.push(Reference::Layer {
            layer: xs(limit, attachment)?,
            expected_loss,
        });
    }
    for (threshold, frequency) in frequencies {
        refs.push(Reference::Frequency {
            threshold,
            frequency,
        });
    }
    let inner = py
        .detach(|| act_pricing::tower::fit_references(&refs, default_alpha, rule))
        .map_err(to_py)?;
    Ok(PyTowerModel { inner })
}
