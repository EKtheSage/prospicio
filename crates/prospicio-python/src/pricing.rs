//! `prospicio.pricing` (Aggregate lane): the collective model, layer
//! rating and reinsurance tower matching over `prospicio_aggregate` and
//! `prospicio_pricing` (`docs/design/pareto.md`).

use prospicio_aggregate::CollectiveModel;
use prospicio_pricing::exposure::{
    ExposureCurve, Interpolation, Mbbefd, SeverityCurve, TabulatedCurve,
};
use prospicio_pricing::layer::XsLayer;
use prospicio_pricing::quote::{Loads, PremiumTerms, Quote, TowerQuote};
use prospicio_pricing::risk_load::{self, PortfolioPrice, PremiumRule, Price, ReinstatementPrice};
use prospicio_pricing::tower::{Reference, SelectionRule, TowerModel};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::aggregate::{AnyCount, PyEventSet};
use crate::distributions::{
    PyGrid, PyPredictiveDistribution, PySampled, extract_severity, key_to_py,
};
use crate::pareto::PyPiecewisePareto;
use crate::risk::PyDistortion;
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
/// >>> from prospicio.distributions import Pareto, claim_count
/// >>> from prospicio.pricing import CollectiveModel
/// >>> m = CollectiveModel(claim_count(2.0, 1.5), Pareto(1e6, 2.0))
/// >>> round(m.layer_mean(4e6, 1e6))
/// 1600000
/// >>> m.excess_frequency(2e6)
/// 0.5
#[pyclass(name = "CollectiveModel", module = "prospicio.pricing", frozen)]
pub(crate) struct PyCollectiveModel {
    inner: CollectiveModel<AnyCount, prospicio_prob::SeverityDist>,
}

#[pymethods]
impl PyCollectiveModel {
    #[new]
    fn new(frequency: &Bound<'_, PyAny>, severity: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner =
            CollectiveModel::new(AnyCount::extract(frequency)?, extract_severity(severity)?);
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
#[pyclass(name = "TowerModel", module = "prospicio.pricing", frozen)]
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

    /// Checks on the matched severity's shape. A match reproduces its
    /// references exactly, but between and beyond them a piece can be
    /// implausible: an alpha below ``min_alpha`` (below 1 on the last,
    /// untruncated piece the mean is infinite), or neighbouring alphas more
    /// than ``max_ratio`` apart.
    ///
    /// Parameters
    /// ----------
    /// min_alpha : float, default 1.0
    /// max_ratio : float, default 2.0
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``min_alpha``, ``max_alpha``, ``max_ratio``, ``infinite_mean`` and
    ///     ``warnings`` (a list of str, empty when the shape looks plausible).
    #[pyo3(signature = (min_alpha = 1.0, max_ratio = 2.0))]
    fn diagnostics<'py>(
        &self,
        py: Python<'py>,
        min_alpha: f64,
        max_ratio: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let d = self.inner.diagnostics(min_alpha, max_ratio);
        let out = PyDict::new(py);
        out.set_item("min_alpha", d.min_alpha)?;
        out.set_item("max_alpha", d.max_alpha)?;
        out.set_item("max_ratio", d.max_ratio)?;
        out.set_item("infinite_mean", d.infinite_mean)?;
        out.set_item("warnings", d.warnings)?;
        Ok(out)
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
/// >>> from prospicio.distributions import Pareto
/// >>> from prospicio.pricing import ilf
/// >>> round(ilf(Pareto(100.0, 2.0), 1000.0, 200.0), 12)
/// 1.266666666667
#[pyfunction]
pub(crate) fn ilf(severity: &Bound<'_, PyAny>, limit: f64, basic_limit: f64) -> PyResult<f64> {
    let sev = extract_severity(severity)?;
    prospicio_pricing::layer::ilf(&sev, limit, basic_limit).map_err(to_py)
}

/// Coefficient of variation of the loss to ``limit`` xs ``attachment``
/// per loss that reaches the layer: ``sd / mean`` of
/// ``min(max(X - a, 0), l)`` given ``X > a``. It is the CV that
/// ``LimitedFluctuation``'s pure premium standard needs for a layer.
///
/// Parameters
/// ----------
/// severity : a severity
/// limit : float
///     Positive; ``inf`` for an unlimited layer.
/// attachment : float
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Gamma
/// >>> from prospicio.pricing import layer_cv
/// >>> round(layer_cv(Gamma(1.0, 100.0), float("inf"), 50.0), 9)
/// 1.0
#[pyfunction]
pub(crate) fn layer_cv(severity: &Bound<'_, PyAny>, limit: f64, attachment: f64) -> PyResult<f64> {
    let sev = extract_severity(severity)?;
    prospicio_pricing::layer::layer_cv(&sev, limit, attachment).map_err(to_py)
}

/// The limited average severity at each policy limit from claims censored
/// at their policies' limits, layer by layer (Werner and Modlin, *Basic
/// Ratemaking*, Table 11.6): each layer between consecutive limits uses only
/// the policies whose limit is at least its top.
///
/// Parameters
/// ----------
/// tables : list of SizeOfLoss
///     One table per policy limit, with the claims capped at the limit, so
///     each table's last bound is its limit. Each layer's bounds must be
///     band edges of the tables it uses.
///
/// Returns
/// -------
/// list of dict
///     One per limit, ascending: ``bottom``, ``top``, ``claims`` (of the
///     policies used), ``severity`` (their average loss in the layer) and
///     ``las`` (the layers up to ``top`` added).
///
/// Examples
/// --------
/// >>> from prospicio.distributions import SizeOfLoss
/// >>> from prospicio.pricing import censored_las
/// >>> tables = [
/// ...     SizeOfLoss([0, 100e3], [2019], [156_657_898]),
/// ...     SizeOfLoss([0, 100e3, 250e3], [690, 773], [34_903_214, 142_767_479]),
/// ...     SizeOfLoss([0, 100e3, 250e3, 500e3], [712, 574, 232],
/// ...                [35_768_111, 90_009_422, 81_092_725]),
/// ... ]
/// >>> [round(layer["severity"]) for layer in censored_las(tables)]
/// [77046, 44575, 15213]
#[pyfunction]
pub(crate) fn censored_las<'py>(
    py: Python<'py>,
    tables: Vec<PyRef<'py, crate::pareto::PySizeOfLoss>>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let tables: Vec<_> = tables.iter().map(|t| t.inner.clone()).collect();
    prospicio_pricing::layer::censored_las(&tables)
        .map_err(to_py)?
        .into_iter()
        .map(|l| {
            let d = PyDict::new(py);
            d.set_item("bottom", l.bottom)?;
            d.set_item("top", l.top)?;
            d.set_item("claims", l.claims)?;
            d.set_item("severity", l.severity)?;
            d.set_item("las", l.las)?;
            Ok(d)
        })
        .collect()
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
    let sev = extract_severity(severity)?;
    prospicio_pricing::layer::loss_elimination_ratio(&sev, deductible).map_err(to_py)
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
/// >>> from prospicio.pricing import pareto_extrapolation
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
    prospicio_pricing::layer::pareto_extrapolation(
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
    prospicio_pricing::layer::alpha_between_layers(
        (xs(a.0, a.1)?, a.2),
        (xs(b.0, b.1)?, b.2),
        truncation,
    )
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
    prospicio_pricing::layer::alpha_between_frequency_and_layer(
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
/// >>> from prospicio.pricing import alpha_between_frequencies
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
    prospicio_pricing::layer::alpha_between_frequencies(
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
/// >>> from prospicio.pricing import match_tower
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
        .detach(|| {
            prospicio_pricing::tower::match_tower(&attachments, &layer_losses, &frequencies, rule)
        })
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
        prospicio_pricing::tower::fit_pml_curve(&return_periods, &amounts, tail_alpha, truncation)
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
/// alpha_bounds : tuple of (float, float), optional
///     ``(lo, hi)``: keep the Pareto alpha between consecutive reference
///     points within these bounds. The two pieces fitted inside a limited
///     layer can still fall either side; check with
///     :meth:`TowerModel.diagnostics`.
///
/// Returns
/// -------
/// TowerModel
///
/// Raises
/// ------
/// ValueError
///     Inconsistent references, or bounds that leave no tower meeting them.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import fit_references
/// >>> m = fit_references([(1000.0, 1000.0, 150.0), (3000.0, 1500.0, 160.0)], [(1000.0, 0.3)])
/// >>> round(m.layer_loss(3000.0, 1500.0), 6)
/// 160.0
#[pyfunction]
#[pyo3(signature = (layers = Vec::new(), frequencies = Vec::new(), default_alpha = 2.0, rule = "minimize", alpha_bounds = None))]
pub(crate) fn fit_references(
    py: Python<'_>,
    layers: Vec<(f64, f64, f64)>,
    frequencies: Vec<(f64, f64)>,
    default_alpha: f64,
    rule: &str,
    alpha_bounds: Option<(f64, f64)>,
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
        .detach(|| {
            prospicio_pricing::tower::fit_references_bounded(
                &refs,
                default_alpha,
                rule,
                alpha_bounds,
            )
        })
        .map_err(to_py)?;
    Ok(PyTowerModel { inner })
}

/// The premium rule from the keyword arguments: exactly one of a cost of
/// capital and a pricing distortion.
pub(crate) fn premium_rule(
    cost_of_capital: Option<f64>,
    distortion: Option<PyRef<'_, PyDistortion>>,
) -> PyResult<PremiumRule> {
    match (cost_of_capital, distortion) {
        (Some(r), None) => PremiumRule::cost_of_capital(r).map_err(to_py),
        (None, Some(d)) => Ok(PremiumRule::Distortion(d.inner.clone())),
        _ => Err(PyValueError::new_err(
            "give exactly one of cost_of_capital and distortion",
        )),
    }
}

/// The risk-loaded price of a cover, or of one component's share of a
/// portfolio: expected loss, premium and the assets backing the loss.
///
/// Returned by ``price`` and ``price_portfolio``.
#[pyclass(name = "Price", module = "prospicio.pricing", frozen)]
pub(crate) struct PyPrice {
    pub(crate) inner: Price,
}

#[pymethods]
impl PyPrice {
    /// Expected loss ``E[X]``.
    #[getter]
    fn expected_loss(&self) -> f64 {
        self.inner.expected_loss
    }

    /// Premium ``P``.
    #[getter]
    fn premium(&self) -> f64 {
        self.inner.premium
    }

    /// Assets ``a`` backing the loss.
    #[getter]
    fn assets(&self) -> f64 {
        self.inner.assets
    }

    /// Margin ``P - E[X]``.
    #[getter]
    fn margin(&self) -> f64 {
        self.inner.margin()
    }

    /// Capital ``a - P``: the assets the premium does not fund.
    #[getter]
    fn capital(&self) -> f64 {
        self.inner.capital()
    }

    /// Loss ratio ``E[X] / P``.
    #[getter]
    fn loss_ratio(&self) -> f64 {
        self.inner.loss_ratio()
    }

    /// Return on capital after income tax, the margin less the tax over
    /// capital.
    #[getter]
    fn return_on_capital(&self) -> f64 {
        self.inner.return_on_capital()
    }

    fn __repr__(&self) -> String {
        format!(
            "Price(expected_loss={}, premium={}, assets={})",
            self.inner.expected_loss, self.inner.premium, self.inner.assets
        )
    }
}

fn prices(v: &[Price]) -> Vec<PyPrice> {
    v.iter().map(|&inner| PyPrice { inner }).collect()
}

/// Prices of a portfolio's components and of the portfolio as a whole.
///
/// Returned by ``price_portfolio``.
#[pyclass(name = "PortfolioPrice", module = "prospicio.pricing", frozen)]
pub(crate) struct PyPortfolioPrice {
    inner: PortfolioPrice,
}

#[pymethods]
impl PyPortfolioPrice {
    /// Component keys, one tuple per component.
    ///
    /// Returns
    /// -------
    /// list of tuple
    fn components<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.inner
            .components
            .iter()
            .map(|k| key_to_py(py, k))
            .collect()
    }

    /// Each component's share of the portfolio price; these add up to
    /// ``total``.
    #[getter]
    fn allocated(&self) -> Vec<PyPrice> {
        prices(&self.inner.allocated)
    }

    /// Each component priced on its own.
    #[getter]
    fn standalone(&self) -> Vec<PyPrice> {
        prices(&self.inner.standalone)
    }

    /// The portfolio, priced on the total of its components.
    #[getter]
    fn total(&self) -> PyPrice {
        PyPrice {
            inner: self.inner.total,
        }
    }

    /// Premium saved by writing the components together: the sum of the
    /// standalone premiums less the portfolio premium.
    ///
    /// Returns
    /// -------
    /// float
    fn diversification(&self) -> f64 {
        self.inner.diversification()
    }

    fn __repr__(&self) -> String {
        format!(
            "PortfolioPrice(components={}, premium={})",
            self.inner.components.len(),
            self.inner.total.premium
        )
    }
}

/// Risk-loaded price of a cover from its simulated losses.
///
/// The assets backing the loss are a distortion risk measure of it. The
/// premium is either a pricing distortion of the loss, or set by a
/// constant cost of capital ``r`` on the capital ``a - P``, which gives
/// ``P = (E[X] + r a) / (1 + r)``.
///
/// Parameters
/// ----------
/// losses : Sampled, PredictiveDistribution or Grid
///     Loss draws (for a ``PredictiveDistribution``, its total), or a loss
///     distribution on a grid, such as a layer's exact annual loss from
///     ``Tower.on_grid``.
/// assets : Distortion
///     The measure that sets the assets, for example ``Distortion.tvar(0.99)``.
/// cost_of_capital : float, optional
///     Positive rate. Give this or ``distortion``.
/// distortion : Distortion, optional
///     Pricing distortion; it must load less than ``assets``.
///
/// Returns
/// -------
/// Price
///
/// Raises
/// ------
/// ValueError
///     Unless exactly one rule is given, or if the premium exceeds the
///     assets.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Sampled
/// >>> from prospicio.pricing import price
/// >>> from prospicio.risk import Distortion
/// >>> p = price(Sampled([0.0, 0.0, 2.0, 6.0]), Distortion.tvar(0.5), cost_of_capital=0.25)
/// >>> p.premium, p.capital
/// (2.4, 1.6)
/// >>> from prospicio.distributions import Grid
/// >>> g = price(Grid(2.0, [0.5, 0.25, 0.0, 0.25]), Distortion.tvar(0.5), cost_of_capital=0.25)
/// >>> round(g.premium, 12)
/// 2.4
#[pyfunction]
#[pyo3(signature = (losses, assets, *, cost_of_capital = None, distortion = None))]
pub(crate) fn price(
    losses: &Bound<'_, PyAny>,
    assets: PyRef<'_, PyDistortion>,
    cost_of_capital: Option<f64>,
    distortion: Option<PyRef<'_, PyDistortion>>,
) -> PyResult<PyPrice> {
    let rule = premium_rule(cost_of_capital, distortion)?;
    let inner = if let Ok(s) = losses.extract::<PyRef<'_, PySampled>>() {
        risk_load::price(&s.inner, &rule, &assets.inner)
    } else if let Ok(pd) = losses.extract::<PyRef<'_, PyPredictiveDistribution>>() {
        risk_load::price(pd.inner.total(), &rule, &assets.inner)
    } else if let Ok(g) = losses.extract::<PyRef<'_, PyGrid>>() {
        risk_load::price_grid(&g.inner, &rule, &assets.inner)
    } else {
        return Err(PyTypeError::new_err(
            "losses must be a Sampled, a PredictiveDistribution or a Grid",
        ));
    }
    .map_err(to_py)?;
    Ok(PyPrice { inner })
}

/// The upfront premium of a cover whose reinstatement premium counts in
/// its own price, from ``price_with_reinstatements``.
#[pyclass(name = "ReinstatementPrice", module = "prospicio.pricing", frozen)]
pub(crate) struct PyReinstatementPrice {
    inner: ReinstatementPrice,
}

#[pymethods]
impl PyReinstatementPrice {
    /// The upfront (deposit) premium ``P``.
    #[getter]
    fn upfront(&self) -> f64 {
        self.inner.upfront
    }

    /// The expected reinstatement premium ``P E[g]``.
    #[getter]
    fn expected_reinstatement_premium(&self) -> f64 {
        self.inner.expected_reinstatement_premium
    }

    /// The expected premium in all, ``P (1 + E[g])``.
    #[getter]
    fn expected_premium(&self) -> f64 {
        self.inner.expected_premium()
    }

    /// The expected ceded loss ``E[X]``.
    #[getter]
    fn expected_loss(&self) -> f64 {
        self.inner.expected_loss
    }

    /// The price of the net outcome ``X - P g`` at the solution.
    #[getter]
    fn net(&self) -> PyPrice {
        PyPrice {
            inner: self.inner.net,
        }
    }

    /// The margin on all the premium, ``P (1 + E[g]) - E[X]``.
    #[getter]
    fn margin(&self) -> f64 {
        self.inner.margin()
    }

    /// The capital: the net outcome's assets less ``P``.
    #[getter]
    fn capital(&self) -> f64 {
        self.inner.capital()
    }

    /// The return on capital.
    #[getter]
    fn return_on_capital(&self) -> f64 {
        self.inner.return_on_capital()
    }

    /// The bisection steps taken.
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    fn __repr__(&self) -> String {
        format!(
            "ReinstatementPrice(upfront={}, expected_reinstatement_premium={}, expected_loss={})",
            self.inner.upfront, self.inner.expected_reinstatement_premium, self.inner.expected_loss
        )
    }
}

/// The upfront premium of a cover whose reinstatement premium counts in
/// its own price (#587).
///
/// The reinsurer's net outcome is ``X - P g``, with ``X`` the year's ceded
/// loss and ``g`` its reinstatement premium per unit of upfront premium
/// (a tower's reinstatement premium at a unit deposit, for example), so
/// its assets and price depend on ``P``; the upfront premium solves
/// ``P = price(X - P g)``. With ``g = 0`` this is ``price``, with a
/// constant ``g`` it is ``price(X) / (1 + g)``, and under the identity
/// distortion it is Sundt's ``E[X] / (1 + E[g])``.
///
/// Parameters
/// ----------
/// losses : array_like
///     The ceded loss in each simulation.
/// g : array_like
///     The reinstatement premium per unit of upfront premium in each
///     simulation, finite and non-negative.
/// assets : Distortion
/// cost_of_capital : float, optional
/// distortion : Distortion, optional
///     Exactly one of ``cost_of_capital`` and ``distortion``, as in
///     ``price``.
///
/// Returns
/// -------
/// ReinstatementPrice
///
/// Examples
/// --------
/// >>> from prospicio.pricing import price_with_reinstatements
/// >>> from prospicio.risk import Distortion
/// >>> x = [0.0, 0.0, 2.0, 6.0]
/// >>> r = price_with_reinstatements(x, [l / 6 for l in x], Distortion.tvar(0.5),
/// ...                               cost_of_capital=0.25)
/// >>> r.upfront < 2.4, round(r.return_on_capital, 9)
/// (True, 0.25)
#[pyfunction]
#[pyo3(signature = (losses, g, assets, *, cost_of_capital = None, distortion = None))]
pub(crate) fn price_with_reinstatements(
    losses: Vec<f64>,
    g: Vec<f64>,
    assets: PyRef<'_, PyDistortion>,
    cost_of_capital: Option<f64>,
    distortion: Option<PyRef<'_, PyDistortion>>,
) -> PyResult<PyReinstatementPrice> {
    let rule = premium_rule(cost_of_capital, distortion)?;
    let inner =
        risk_load::price_with_reinstatements(&losses, &g, &rule, &assets.inner).map_err(to_py)?;
    Ok(PyReinstatementPrice { inner })
}

/// Prices a portfolio and allocates the price to its components.
///
/// Premium and assets are each allocated by co-measure (the natural
/// allocation): component prices add up to the portfolio's, and a
/// component that diversifies the portfolio is priced below its
/// standalone price. With a cost of capital, every component earns the
/// rate on its allocated capital.
///
/// Parameters
/// ----------
/// pd : PredictiveDistribution
///     Components that add up to the portfolio: segments or covers, not
///     gross, ceded and net side by side.
/// assets : Distortion
/// cost_of_capital : float, optional
/// distortion : Distortion, optional
///     Exactly one of ``cost_of_capital`` and ``distortion``, as in
///     ``price``.
///
/// Returns
/// -------
/// PortfolioPrice
///
/// Examples
/// --------
/// >>> from prospicio.distributions import PredictiveDistribution
/// >>> from prospicio.pricing import price_portfolio
/// >>> from prospicio.risk import Distortion
/// >>> pd = PredictiveDistribution(["cover"], [("a",), ("b",)],
/// ...                             [[0.0, 2.0], [1.0, 1.0], [4.0, 0.0], [8.0, 0.0]])
/// >>> p = price_portfolio(pd, Distortion.tvar(0.5), cost_of_capital=0.1)
/// >>> [round(c.premium, 6) for c in p.allocated]
/// [3.5, 0.681818]
/// >>> p.allocated[1].margin < 0  # the second cover hedges the first
/// True
#[pyfunction]
#[pyo3(signature = (pd, assets, *, cost_of_capital = None, distortion = None))]
pub(crate) fn price_portfolio(
    py: Python<'_>,
    pd: PyRef<'_, PyPredictiveDistribution>,
    assets: PyRef<'_, PyDistortion>,
    cost_of_capital: Option<f64>,
    distortion: Option<PyRef<'_, PyDistortion>>,
) -> PyResult<PyPortfolioPrice> {
    let rule = premium_rule(cost_of_capital, distortion)?;
    let (pd, a) = (&pd.inner, assets.inner.clone());
    let inner = py
        .detach(|| risk_load::price_portfolio(pd, &rule, &a))
        .map_err(to_py)?;
    Ok(PyPortfolioPrice { inner })
}

/// The MBBEFD exposure curve and destruction-rate distribution (Bernegger,
/// 1997), with ``b >= 0`` and ``g >= 1``; ``1/g`` is the probability of a
/// total loss.
///
/// ``G(x)`` is the share of a risk's expected loss below the fraction
/// ``x`` of its maximum possible loss (MPL). ``Mbbefd.swiss_re(c)`` gives
/// Bernegger's one-parameter family: ``c = 1.5, 2, 3, 4`` are the Swiss Re
/// curves and ``c = 5`` the Lloyd's curve.
///
/// Parameters
/// ----------
/// b : float
/// g : float
///
/// Examples
/// --------
/// >>> from prospicio.pricing import Mbbefd
/// >>> c3 = Mbbefd.swiss_re(3.0)
/// >>> top = c3.layer_share(5e6, 5e6, 10e6)
/// >>> bottom = c3.layer_share(5e6, 0.0, 10e6)
/// >>> round(top + bottom, 12), top < bottom
/// (1.0, True)
#[pyclass(name = "Mbbefd", module = "prospicio.pricing", frozen)]
pub(crate) struct PyMbbefd {
    pub(crate) inner: Mbbefd,
}

#[pymethods]
impl PyMbbefd {
    #[new]
    fn new(b: f64, g: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Mbbefd::new(b, g).map_err(to_py)?,
        })
    }

    /// Bernegger's curve ``c``: ``b = exp(3.1 - 0.15 (1 + c) c)``,
    /// ``g = exp((0.78 + 0.12 c) c)``.
    ///
    /// Parameters
    /// ----------
    /// c : float
    ///     Non-negative; 0 is the straight line.
    ///
    /// Returns
    /// -------
    /// Mbbefd
    #[staticmethod]
    fn swiss_re(c: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Mbbefd::swiss_re(c).map_err(to_py)?,
        })
    }

    /// The Swiss Re curve whose mean destruction rate is ``rate``, and its
    /// ``c``.
    ///
    /// Parameters
    /// ----------
    /// rate : float
    ///     Between the curves at ``c = 10`` and ``c = 0`` (0.5).
    ///
    /// Returns
    /// -------
    /// tuple of (Mbbefd, float)
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.pricing import Mbbefd
    /// >>> curve, c = Mbbefd.swiss_re_for_mean_rate(Mbbefd.swiss_re(3.0).mean())
    /// >>> round(c, 9)
    /// 3.0
    #[staticmethod]
    fn swiss_re_for_mean_rate(rate: f64) -> PyResult<(Self, f64)> {
        let (inner, c) = Mbbefd::swiss_re_for_mean_rate(rate).map_err(to_py)?;
        Ok((Self { inner }, c))
    }

    /// The Swiss Re curve that gives ``limit`` xs ``attachment`` the share
    /// ``share`` of a risk's loss with maximum possible loss ``mpl``
    /// (``G((a + l)/mpl) - G(a/mpl)``), and its ``c``.
    ///
    /// Parameters
    /// ----------
    /// limit, attachment, mpl : float
    /// share : float
    ///
    /// Returns
    /// -------
    /// tuple of (Mbbefd, float)
    #[staticmethod]
    fn swiss_re_for_layer_share(
        limit: f64,
        attachment: f64,
        mpl: f64,
        share: f64,
    ) -> PyResult<(Self, f64)> {
        let (inner, c) =
            Mbbefd::swiss_re_for_layer_share(limit, attachment, mpl, share).map_err(to_py)?;
        Ok((Self { inner }, c))
    }

    /// Parameter ``b``.
    #[getter]
    fn b(&self) -> f64 {
        self.inner.b()
    }

    /// Parameter ``g``.
    #[getter]
    fn g(&self) -> f64 {
        self.inner.g_parameter()
    }

    /// The exposure curve ``G(x)`` at each ``x`` (clamped to [0, 1]).
    ///
    /// Parameters
    /// ----------
    /// x : list of float
    ///
    /// Returns
    /// -------
    /// list of float
    fn curve(&self, x: Vec<f64>) -> Vec<f64> {
        x.iter().map(|&v| self.inner.g(v)).collect()
    }

    /// Distribution function of the destruction rate at each ``x``.
    ///
    /// Parameters
    /// ----------
    /// x : list of float
    ///
    /// Returns
    /// -------
    /// list of float
    fn cdf(&self, x: Vec<f64>) -> Vec<f64> {
        x.iter().map(|&v| self.inner.cdf(v)).collect()
    }

    /// Mean destruction rate, ``1 / G'(0)``.
    ///
    /// Returns
    /// -------
    /// float
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// Probability of a total loss, ``1/g``.
    ///
    /// Returns
    /// -------
    /// float
    fn total_loss_probability(&self) -> f64 {
        self.inner.total_loss_probability()
    }

    /// Destruction rate (loss over MPL) at each probability ``u`` in
    /// ``(0, 1)``: draws with this curve as their exposure curve.
    ///
    /// Parameters
    /// ----------
    /// u : list of float
    ///
    /// Returns
    /// -------
    /// list of float
    fn rate_quantile(&self, u: Vec<f64>) -> Vec<f64> {
        u.iter().map(|&v| self.inner.rate_quantile(v)).collect()
    }

    /// Share of a risk's expected loss in the layer ``limit`` xs
    /// ``attachment``, for a risk with maximum possible loss ``mpl``.
    ///
    /// Parameters
    /// ----------
    /// limit : float
    /// attachment : float
    /// mpl : float
    ///
    /// Returns
    /// -------
    /// float
    fn layer_share(&self, limit: f64, attachment: f64, mpl: f64) -> PyResult<f64> {
        self.inner
            .layer_share(limit, attachment, mpl)
            .map_err(to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "Mbbefd(b={}, g={})",
            self.inner.b(),
            self.inner.g_parameter()
        )
    }
}

/// A tabulated exposure curve: points ``(x, G(x))`` from ``(0, 0)`` to
/// ``(1, 1)``, interpolated linearly or log-log, as published curves are given
/// (Salzmann's homeowners scale, Ludwig's curves, ISO PSOLD tables, a
/// reinsurer's own).
///
/// The table must be concave (its slopes never increase). Its destruction
/// rate is discrete: the points' ``x`` with probabilities from the drops in
/// slope, and a total loss with probability last slope over first. Its
/// mean rate is the first chord's, ``x1 / G(x1)``, so a table needs fine
/// first points for the expected loss to be right.
///
/// With ``interpolation="log-log"`` the curve runs as a power law between
/// points (straight on log-log axes), ``G(x) = G_k (x / x_k) ** p_k``, which
/// follows a curve's bend between sparse points better than a chord; the
/// first piece, from ``(0, 0)``, stays straight. The interpolated curve must
/// be concave, so the elasticities ``p_k`` may not rise from piece to piece:
/// a table with a total-loss mass at the top (the Swiss Re curves with ``c``
/// up to 3, read at a few points) fails and keeps linear interpolation. The
/// destruction rate is then continuous within each piece past the first.
///
/// Parameters
/// ----------
/// x : list of float
///     Increasing from 0 to 1.
/// g : list of float
///     ``G(x)``, from 0 to 1.
/// interpolation : {"linear", "log-log"}, default "linear"
///
/// Raises
/// ------
/// ValueError
///     If the points do not run from ``(0, 0)`` to ``(1, 1)``, or are not
///     increasing and concave.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import TabulatedCurve
/// >>> t = TabulatedCurve([0.0, 0.1, 0.5, 1.0], [0.0, 0.4, 0.8, 1.0])
/// >>> round(t.curve([0.3])[0], 12), t.mean_rate()
/// (0.6, 0.25)
#[pyclass(name = "TabulatedCurve", module = "prospicio.pricing", frozen)]
pub(crate) struct PyTabulatedCurve {
    pub(crate) inner: TabulatedCurve,
}

#[pymethods]
impl PyTabulatedCurve {
    #[new]
    #[pyo3(signature = (x, g, interpolation = "linear"))]
    fn new(x: Vec<f64>, g: Vec<f64>, interpolation: &str) -> PyResult<Self> {
        let how = match interpolation {
            "linear" => Interpolation::Linear,
            "log-log" => Interpolation::LogLog,
            other => {
                return Err(PyValueError::new_err(format!(
                    "interpolation must be \"linear\" or \"log-log\", not {other:?}"
                )));
            }
        };
        Ok(Self {
            inner: TabulatedCurve::with_interpolation(&x, &g, how).map_err(to_py)?,
        })
    }

    /// How the curve runs between its points: ``"linear"`` or
    /// ``"log-log"``.
    #[getter]
    fn interpolation(&self) -> &'static str {
        match self.inner.interpolation() {
            Interpolation::Linear => "linear",
            Interpolation::LogLog => "log-log",
        }
    }

    /// The exposure curve of ``severity`` capped at ``mpl``, tabulated at
    /// ``points`` points, for a risk profile's band (which takes an
    /// ``Mbbefd`` or a ``TabulatedCurve``).
    ///
    /// Parameters
    /// ----------
    /// severity : a severity
    /// mpl : float
    ///     The maximum possible loss.
    /// points : int, default 200
    ///     At least 4.
    ///
    /// Returns
    /// -------
    /// TabulatedCurve
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import Lognormal
    /// >>> from prospicio.pricing import TabulatedCurve
    /// >>> t = TabulatedCurve.from_severity(Lognormal.from_mean_cv(1e5, 2.0), 1e6)
    /// >>> t.x[0], t.x[-1], t.g[-1]
    /// (0.0, 1.0, 1.0)
    #[staticmethod]
    #[pyo3(signature = (severity, mpl, points = 200))]
    fn from_severity(severity: &Bound<'_, PyAny>, mpl: f64, points: usize) -> PyResult<Self> {
        let sev = extract_severity(severity)?;
        Ok(Self {
            inner: TabulatedCurve::from_severity(&sev, mpl, points).map_err(to_py)?,
        })
    }

    /// The table's ``x``.
    #[getter]
    fn x(&self) -> Vec<f64> {
        self.inner.x().to_vec()
    }

    /// The table's ``G(x)``.
    #[getter]
    fn g(&self) -> Vec<f64> {
        self.inner.g_values().to_vec()
    }

    /// The exposure curve ``G(x)`` at each ``x`` (clamped to [0, 1]).
    ///
    /// Parameters
    /// ----------
    /// x : list of float
    ///
    /// Returns
    /// -------
    /// list of float
    fn curve(&self, x: Vec<f64>) -> Vec<f64> {
        x.iter().map(|&v| self.inner.g(v)).collect()
    }

    /// Mean destruction rate, ``x1 / G(x1)``.
    ///
    /// Returns
    /// -------
    /// float
    fn mean_rate(&self) -> f64 {
        self.inner.mean_rate()
    }

    /// Destruction rate at each probability ``u`` in ``(0, 1)``.
    ///
    /// Parameters
    /// ----------
    /// u : list of float
    ///
    /// Returns
    /// -------
    /// list of float
    fn rate_quantile(&self, u: Vec<f64>) -> Vec<f64> {
        u.iter().map(|&v| self.inner.rate_quantile(v)).collect()
    }

    /// Share of a risk's expected loss in the layer ``limit`` xs
    /// ``attachment``, for a risk with maximum possible loss ``mpl``.
    ///
    /// Parameters
    /// ----------
    /// limit : float
    /// attachment : float
    /// mpl : float
    ///
    /// Returns
    /// -------
    /// float
    fn layer_share(&self, limit: f64, attachment: f64, mpl: f64) -> PyResult<f64> {
        self.inner
            .layer_share(limit, attachment, mpl)
            .map_err(to_py)
    }

    fn __getnewargs__(&self) -> (Vec<f64>, Vec<f64>, &'static str) {
        (
            self.inner.x().to_vec(),
            self.inner.g_values().to_vec(),
            self.interpolation(),
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "TabulatedCurve({} points, {})",
            self.inner.x().len(),
            self.interpolation()
        )
    }
}

/// A band's exposure curve from a Python ``Mbbefd`` or ``TabulatedCurve``.
fn band_curve(obj: &Bound<'_, PyAny>) -> PyResult<prospicio_pricing::profile::BandCurve> {
    if let Ok(c) = obj.extract::<PyRef<'_, PyMbbefd>>() {
        return Ok(c.inner.into());
    }
    if let Ok(c) = obj.extract::<PyRef<'_, PyTabulatedCurve>>() {
        return Ok(c.inner.clone().into());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "each band's curve must be an Mbbefd or a TabulatedCurve",
    ))
}

/// A risk profile for property per-risk business: bands of sum insured,
/// each with an expected loss (given, or premium times a loss ratio) and
/// its own exposure curve.
///
/// Each band's representative risk has sum insured ``SI`` (its total sum
/// insured over its number of risks, say), taken as its MPL. The band
/// expects ``EL / (SI * curve.mean_rate)`` losses a year; each simulated
/// loss is the band's ``SI`` times a destruction rate from the band's
/// curve, and carries that ``SI``, so a surplus treaty (``Layer.surplus``)
/// and the per-risk excess of loss it inures to apply to the events. The
/// exposure-rated expectations (``expected_layer_loss``,
/// ``expected_surplus_loss``) check the simulation.
///
/// Parameters
/// ----------
/// sums_insured : list of float
///     One per band.
/// risks : list of float
///     Number of risks per band (for reference).
/// curves : Mbbefd or TabulatedCurve, or a list of them
///     One curve for every band, or one per band.
/// expected_losses : list of float, optional
///     Expected annual loss per band. Give this, or ``premiums``.
/// premiums : list of float, optional
///     Premium per band, with ``loss_ratio``.
/// loss_ratio : float or list of float, optional
///     Expected loss ratio, one for all bands or one per band.
/// lower, upper : list of float or None, optional
///     Bounds of each band's sums insured (``None`` for a band without).
///     A band with bounds spreads its risks' sums insured uniformly between
///     them: its mean ``SI`` is ``(lower + upper) / 2`` (in place of
///     ``sums_insured``), each simulated loss draws its own ``SI`` between
///     the bounds, and the exposure-rated expectations average over the
///     band, weighted by sum insured.
/// spread : {"uniform", "tilted", "auto"}, optional
///     How a band with bounds spreads its sums insured. ``"tilted"`` keeps
///     ``sums_insured`` as the mean, so the spread matches both the bounds
///     and the band's total sum insured: the density ``∝ exp(θ s)`` on the
///     bounds with ``θ`` solved for that mean (``sums_insured`` must lie
///     strictly between the bounds). ``"uniform"`` moves the mean to
///     ``(lower + upper) / 2``. ``"auto"`` tilts a band whose
///     ``sums_insured`` lies strictly between its bounds and spreads any
///     other uniformly. Not given, the spread is uniform, with a
///     ``UserWarning`` for each band whose mean moves (as with ``"auto"``).
///     A band may start at ``lower = 0``.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import Mbbefd, RiskProfile
/// >>> p = RiskProfile([1e6, 10e6], [800, 50], Mbbefd.swiss_re(3.0),
/// ...                 premiums=[2e6, 1e6], loss_ratio=0.6)
/// >>> round(p.expected_loss())
/// 1800000
/// >>> events = p.simulate(1000, 7)
/// >>> events.has_sums_insured
/// True
#[pyclass(name = "RiskProfile", module = "prospicio.pricing", frozen)]
pub(crate) struct PyRiskProfile {
    inner: prospicio_pricing::profile::RiskProfile,
}

#[pymethods]
impl PyRiskProfile {
    #[new]
    #[pyo3(signature = (sums_insured, risks, curves, expected_losses = None, premiums = None, loss_ratio = None, lower = None, upper = None, spread = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        sums_insured: Vec<f64>,
        risks: Vec<f64>,
        curves: &Bound<'_, PyAny>,
        expected_losses: Option<Vec<f64>>,
        premiums: Option<Vec<f64>>,
        loss_ratio: Option<&Bound<'_, PyAny>>,
        lower: Option<Vec<Option<f64>>>,
        upper: Option<Vec<Option<f64>>>,
        spread: Option<&str>,
    ) -> PyResult<Self> {
        use prospicio_pricing::profile::{Band, RiskProfile};
        use pyo3::exceptions::PyValueError;
        let curves_py = curves.py();
        let n = sums_insured.len();
        if risks.len() != n {
            return Err(PyValueError::new_err("give one number of risks per band"));
        }
        let curves: Vec<_> = match curves.extract::<Vec<Bound<'_, PyAny>>>() {
            Ok(list) => list.iter().map(band_curve).collect::<PyResult<_>>()?,
            Err(_) => vec![band_curve(curves)?; n],
        };
        if curves.len() != n {
            return Err(PyValueError::new_err("give one curve, or one per band"));
        }
        let bands = match (expected_losses, premiums) {
            (Some(el), None) => {
                if loss_ratio.is_some() {
                    return Err(PyValueError::new_err("loss_ratio goes with premiums"));
                }
                if el.len() != n {
                    return Err(PyValueError::new_err("give one expected loss per band"));
                }
                (0..n)
                    .map(|i| {
                        Band::from_expected_loss(
                            sums_insured[i],
                            risks[i],
                            el[i],
                            curves[i].clone(),
                        )
                    })
                    .collect::<prospicio_core::Result<Vec<_>>>()
            }
            (None, Some(pr)) => {
                let lr: Vec<f64> = match loss_ratio {
                    None => return Err(PyValueError::new_err("premiums need a loss_ratio")),
                    Some(l) => match l.extract::<f64>() {
                        Ok(v) => vec![v; n],
                        Err(_) => l.extract::<Vec<f64>>()?,
                    },
                };
                if pr.len() != n || lr.len() != n {
                    return Err(PyValueError::new_err(
                        "give one premium per band, and one loss ratio or one per band",
                    ));
                }
                (0..n)
                    .map(|i| {
                        Band::from_premium(
                            sums_insured[i],
                            risks[i],
                            pr[i],
                            lr[i],
                            curves[i].clone(),
                        )
                    })
                    .collect::<prospicio_core::Result<Vec<_>>>()
            }
            _ => {
                return Err(PyValueError::new_err(
                    "give expected_losses, or premiums with a loss_ratio",
                ));
            }
        }
        .map_err(to_py)?;
        #[derive(Clone, Copy, PartialEq)]
        enum Spread {
            Auto,
            Uniform,
            Tilted,
        }
        // Not given: uniform, with a warning for each band whose mean moves.
        let warn = matches!(spread, None | Some("auto"));
        let spread = match spread.unwrap_or("uniform") {
            "auto" => Spread::Auto,
            "uniform" => Spread::Uniform,
            "tilted" => Spread::Tilted,
            other => {
                return Err(PyValueError::new_err(format!(
                    "spread must be \"auto\", \"uniform\" or \"tilted\", not {other:?}"
                )));
            }
        };
        let bands = match (lower, upper) {
            (None, None) => bands,
            (Some(lo), Some(up)) => {
                if lo.len() != n || up.len() != n {
                    return Err(PyValueError::new_err(
                        "give one lower and one upper per band",
                    ));
                }
                let bands: Vec<Band> = bands
                    .into_iter()
                    .zip(lo.into_iter().zip(up))
                    .map(|(b, bounds)| match (bounds, spread) {
                        ((None, None), _) => Ok(b),
                        ((Some(l), Some(u)), Spread::Tilted) => {
                            b.with_tilted_bounds(l, u).map_err(to_py)
                        }
                        ((Some(l), Some(u)), Spread::Uniform) => b.with_bounds(l, u).map_err(to_py),
                        ((Some(l), Some(u)), Spread::Auto) => {
                            b.with_spread_bounds(l, u).map_err(to_py)
                        }
                        _ => Err(PyValueError::new_err("a band has both bounds or neither")),
                    })
                    .collect::<PyResult<_>>()?;
                if warn {
                    let moved: Vec<String> = bands
                        .iter()
                        .enumerate()
                        .filter(|(_, b)| {
                            (b.mean_sum_insured() - b.sum_insured).abs()
                                > 1e-9 * b.sum_insured.abs().max(1.0)
                        })
                        .map(|(i, b)| {
                            format!("band {i}: {} -> {}", b.sum_insured, b.mean_sum_insured())
                        })
                        .collect();
                    if !moved.is_empty() {
                        let py = curves_py;
                        let msg = format!(
                            "a band spread uniformly between its bounds no longer has \
                             sums_insured as its mean ({}); expected claims and layer losses use \
                             the moved mean. Give spread=\"tilted\" or \"auto\" to keep it, or \
                             spread=\"uniform\" to silence this",
                            moved.join(", ")
                        );
                        let cat = py.get_type::<pyo3::exceptions::PyUserWarning>();
                        PyErr::warn(py, &cat, &std::ffi::CString::new(msg).unwrap(), 1)?;
                    }
                }
                bands
            }
            _ => return Err(PyValueError::new_err("give lower and upper together")),
        };
        Ok(Self {
            inner: RiskProfile::new(bands).map_err(to_py)?,
        })
    }

    /// Expected annual loss, all bands.
    ///
    /// Returns
    /// -------
    /// float
    fn expected_loss(&self) -> f64 {
        self.inner.expected_loss()
    }

    /// Expected number of losses a year, per band: losses of every size,
    /// not only those that reach a layer.
    ///
    /// Returns
    /// -------
    /// list of float
    fn expected_claims(&self) -> Vec<f64> {
        self.inner
            .bands()
            .iter()
            .map(|b| b.expected_claims())
            .collect()
    }

    /// Exposure-rated expected loss to a per-risk layer ``limit`` xs
    /// ``attachment``, optionally on each risk net of a surplus treaty.
    ///
    /// Parameters
    /// ----------
    /// limit : float
    ///     ``inf`` for unlimited.
    /// attachment : float
    /// surplus_retention, surplus_lines : float, optional
    ///     A surplus treaty the layer inures to.
    ///
    /// Returns
    /// -------
    /// float
    #[pyo3(signature = (limit, attachment, surplus_retention = None, surplus_lines = None))]
    fn expected_layer_loss(
        &self,
        limit: f64,
        attachment: f64,
        surplus_retention: Option<f64>,
        surplus_lines: Option<f64>,
    ) -> PyResult<f64> {
        let surplus = match (surplus_retention, surplus_lines) {
            (Some(r), Some(k)) => Some((r, k)),
            (None, None) => None,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "give both surplus_retention and surplus_lines, or neither",
                ));
            }
        };
        self.inner
            .expected_layer_loss(limit, attachment, surplus)
            .map_err(to_py)
    }

    /// Expected annual loss ceded to a surplus treaty.
    ///
    /// Parameters
    /// ----------
    /// retention : float
    /// lines : float
    ///
    /// Returns
    /// -------
    /// float
    fn expected_surplus_loss(&self, retention: f64, lines: f64) -> f64 {
        self.inner.expected_surplus_loss(retention, lines)
    }

    /// ``n_sims`` years of losses, each with its risk's sum insured.
    ///
    /// Parameters
    /// ----------
    /// n_sims : int
    /// seed : int
    ///
    /// Returns
    /// -------
    /// EventSet
    fn simulate(
        &self,
        py: Python<'_>,
        n_sims: usize,
        seed: u64,
    ) -> PyResult<crate::aggregate::PyEventSet> {
        let inner = py
            .detach(|| self.inner.simulate(n_sims, seed))
            .map_err(to_py)?;
        Ok(crate::aggregate::PyEventSet { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "RiskProfile({} bands, expected loss {})",
            self.inner.bands().len(),
            self.inner.expected_loss()
        )
    }
}

/// The exposure curve of a severity capped at the maximum possible loss
/// ``mpl``: ``G(x) = LEV(x mpl) / LEV(mpl)`` at each ``x``.
///
/// Parameters
/// ----------
/// severity : a severity
/// mpl : float
/// x : list of float
///
/// Returns
/// -------
/// list of float
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Pareto
/// >>> from prospicio.pricing import severity_exposure_curve
/// >>> g = severity_exposure_curve(Pareto(1e5, 1.5), 1e7, [0.0, 0.5, 1.0])
/// >>> g[0], round(g[2], 12), g[1] > 0.5
/// (0.0, 1.0, True)
#[pyfunction]
pub(crate) fn severity_exposure_curve(
    severity: &Bound<'_, PyAny>,
    mpl: f64,
    x: Vec<f64>,
) -> PyResult<Vec<f64>> {
    let sev = extract_severity(severity)?;
    let curve = SeverityCurve::new(&sev, mpl).map_err(to_py)?;
    Ok(x.iter().map(|&v| curve.g(v)).collect())
}

/// A cover's quote: the premium the cedant pays, built up from a
/// risk-loaded price, with the ratios the market quotes.
///
/// ``Quote.technical(price, ...)`` grosses the risk premium ``R`` (the
/// expected loss plus the margin for capital, from ``price``) up for
/// internal expenses ``e`` and brokerage ``b``, both fractions of the
/// premium, and a fixed expense ``f``: ``P = (R + f) / (1 - e - b)``.
/// ``quote.at(premium)`` reads a quoted premium the other way: the
/// reinsurer keeps ``P (1 - e - b) - f`` as its risk premium, and the margin
/// and return on capital follow with the same expected loss and assets.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Sampled
/// >>> from prospicio.pricing import Quote, price
/// >>> from prospicio.risk import Distortion
/// >>> p = price(Sampled([0.0, 0.0, 2.0, 6.0]), Distortion.tvar(0.5), cost_of_capital=0.1)
/// >>> q = Quote.technical(p, expenses=0.03, brokerage=0.10, limit=10.0)
/// >>> round(q.premium, 6), round(q.rate_on_line, 6), round(q.return_on_capital, 6)
/// (2.507837, 0.250784, 0.1)
/// >>> round(q.at(3.0).return_on_capital, 4)
/// 0.4388
#[pyclass(name = "Quote", module = "prospicio.pricing", frozen)]
pub(crate) struct PyQuote {
    inner: Quote,
}

#[pymethods]
impl PyQuote {
    /// The technical premium of a risk-loaded price.
    ///
    /// Parameters
    /// ----------
    /// price : Price
    ///     The risk-loaded price, from ``price``; its premium is the risk
    ///     premium.
    /// expenses : float, default 0.0
    ///     Internal expenses, a fraction of the premium.
    /// brokerage : float, default 0.0
    ///     Brokerage, a fraction of the premium.
    /// fixed : float, default 0.0
    ///     A fixed expense per contract, an amount.
    /// limit : float, optional
    ///     The cover's limit at the placed share, for the rate on line;
    ///     omit it for a cover without one, such as a quota share.
    /// premium_tax : float, default 0.0
    ///     Premium tax, a fraction of all the premium (deposit and
    ///     reinstatement premium), grossed up as brokerage is.
    /// income_tax : float, default 0.0
    ///     Income tax on the margin. The price's return on capital is then
    ///     the return after tax, and the risk premium is raised to give it.
    /// reinstatement_rate : float, default 0.0
    ///     Expected reinstatement premium per unit of deposit: a tower
    ///     grid's ``expected_reinstatement_premium`` for the layer over its
    ///     deposit premium (reinstatement premium is proportional to it).
    /// commission : float, default 0.0
    ///     A flat ceding commission, a fraction of the deposit.
    /// brokerage_on_reinstatements : bool, default True
    ///     Whether brokerage is charged on reinstatement premium too.
    ///
    /// Returns
    /// -------
    /// Quote
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a load or rate is negative or not below 1, the loads and
    ///     commission leave no premium, or the limit is not positive.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import Sampled
    /// >>> from prospicio.pricing import Quote, price
    /// >>> from prospicio.risk import Distortion
    /// >>> p = price(Sampled([0.0, 0.0, 2.0, 6.0]), Distortion.tvar(0.5), cost_of_capital=0.1)
    /// >>> q = Quote.technical(p, brokerage=0.10, income_tax=0.21, reinstatement_rate=0.5)
    /// >>> round(q.return_on_capital, 6), q.pre_tax_return_on_capital > 0.1
    /// (0.1, True)
    /// >>> round(q.premium_in_all / q.premium, 6)
    /// 1.5
    #[staticmethod]
    #[pyo3(signature = (
        price, *, expenses = 0.0, brokerage = 0.0, fixed = 0.0, limit = None,
        premium_tax = 0.0, income_tax = 0.0, reinstatement_rate = 0.0, commission = 0.0,
        brokerage_on_reinstatements = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn technical(
        price: PyRef<'_, PyPrice>,
        expenses: f64,
        brokerage: f64,
        fixed: f64,
        limit: Option<f64>,
        premium_tax: f64,
        income_tax: f64,
        reinstatement_rate: f64,
        commission: f64,
        brokerage_on_reinstatements: bool,
    ) -> PyResult<Self> {
        let loads = Loads::new(expenses, brokerage, fixed)
            .and_then(|l| l.with_premium_tax(premium_tax))
            .and_then(|l| l.with_income_tax(income_tax))
            .map_err(to_py)?;
        let mut terms = PremiumTerms::new(reinstatement_rate, commission).map_err(to_py)?;
        if !brokerage_on_reinstatements {
            terms = terms.without_brokerage_on_reinstatements();
        }
        Ok(Self {
            inner: Quote::technical_with(&price.inner, loads, terms, limit).map_err(to_py)?,
        })
    }

    /// The same cover at a quoted deposit: what the quote leaves after the
    /// loads and commissions is the risk premium.
    ///
    /// Parameters
    /// ----------
    /// premium : float
    ///     The deposit; for a swing-rated layer, the expected premium from
    ///     its tower grid (``expected_premium``).
    /// commission : float, default 0.0
    ///     Expected commissions beyond the flat one, an amount: a sliding
    ///     scale or profit commission read off the layer's tower grid at
    ///     this deposit (``expected_ceding_commission``,
    ///     ``expected_profit_commission``).
    ///
    /// Returns
    /// -------
    /// Quote
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the risk premium the quote leaves reaches the assets.
    #[pyo3(signature = (premium, *, commission = 0.0))]
    fn at(&self, premium: f64, commission: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .at_with_commission(premium, commission)
                .map_err(to_py)?,
        })
    }

    /// The deposit premium the cedant pays upfront.
    #[getter]
    fn premium(&self) -> f64 {
        self.inner.premium
    }

    /// Expected reinstatement premium.
    #[getter]
    fn reinstatement_premium(&self) -> f64 {
        self.inner.reinstatement_premium()
    }

    /// The deposit plus the expected reinstatement premium.
    #[getter]
    fn premium_in_all(&self) -> f64 {
        self.inner.premium_in_all()
    }

    /// Premium tax, an amount.
    #[getter]
    fn premium_tax(&self) -> f64 {
        self.inner.premium_tax()
    }

    /// Commissions paid back to the cedant, an amount.
    #[getter]
    fn commission(&self) -> f64 {
        self.inner.commission()
    }

    /// Income tax on the margin, an amount.
    #[getter]
    fn income_tax(&self) -> f64 {
        self.inner.income_tax()
    }

    /// Return on capital before income tax, margin over capital.
    #[getter]
    fn pre_tax_return_on_capital(&self) -> f64 {
        self.inner.pre_tax_return_on_capital()
    }

    /// Expected loss.
    #[getter]
    fn expected_loss(&self) -> f64 {
        self.inner.expected_loss
    }

    /// The risk premium: what the reinsurer keeps for the loss and capital.
    #[getter]
    fn risk_premium(&self) -> f64 {
        self.inner.risk_premium
    }

    /// Assets backing the loss.
    #[getter]
    fn assets(&self) -> f64 {
        self.inner.assets
    }

    /// Margin, risk premium less expected loss.
    #[getter]
    fn margin(&self) -> f64 {
        self.inner.margin()
    }

    /// Capital, assets less the risk premium.
    #[getter]
    fn capital(&self) -> f64 {
        self.inner.capital()
    }

    /// Return on capital after income tax, the margin less the tax over
    /// capital.
    #[getter]
    fn return_on_capital(&self) -> f64 {
        self.inner.return_on_capital()
    }

    /// Internal expenses, an amount.
    #[getter]
    fn expenses(&self) -> f64 {
        self.inner.expenses()
    }

    /// Brokerage, an amount.
    #[getter]
    fn brokerage(&self) -> f64 {
        self.inner.brokerage()
    }

    /// The premium net of brokerage.
    #[getter]
    fn net_premium(&self) -> f64 {
        self.inner.net_premium()
    }

    /// Expected loss over all the premium.
    #[getter]
    fn loss_ratio(&self) -> f64 {
        self.inner.loss_ratio()
    }

    /// All the premium over expected loss.
    #[getter]
    fn multiple(&self) -> f64 {
        self.inner.multiple()
    }

    /// The deposit over the limit, or ``None`` without a limit.
    #[getter]
    fn rate_on_line(&self) -> Option<f64> {
        self.inner.rate_on_line()
    }

    /// Limit over the deposit, or ``None`` without a limit.
    #[getter]
    fn payback(&self) -> Option<f64> {
        self.inner.payback()
    }

    /// The limit, or ``None``.
    #[getter]
    fn limit(&self) -> Option<f64> {
        self.inner.limit
    }

    /// The build-up and ratios as a dict, for a table.
    ///
    /// Returns
    /// -------
    /// dict of str to float
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let q = &self.inner;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("expected_loss", q.expected_loss)?;
        d.set_item("margin", q.margin())?;
        d.set_item("risk_premium", q.risk_premium)?;
        d.set_item("expenses", q.expenses())?;
        d.set_item("brokerage", q.brokerage())?;
        d.set_item("premium_tax", q.premium_tax())?;
        d.set_item("commission", q.commission())?;
        d.set_item("premium", q.premium)?;
        d.set_item("reinstatement_premium", q.reinstatement_premium())?;
        d.set_item("premium_in_all", q.premium_in_all())?;
        d.set_item("net_premium", q.net_premium())?;
        d.set_item("assets", q.assets)?;
        d.set_item("capital", q.capital())?;
        d.set_item("income_tax", q.income_tax())?;
        d.set_item("pre_tax_return_on_capital", q.pre_tax_return_on_capital())?;
        d.set_item("return_on_capital", q.return_on_capital())?;
        d.set_item("loss_ratio", q.loss_ratio())?;
        d.set_item("multiple", q.multiple())?;
        d.set_item("rate_on_line", q.rate_on_line())?;
        d.set_item("payback", q.payback())?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "Quote(premium={}, expected_loss={}, return_on_capital={})",
            self.inner.premium,
            self.inner.expected_loss,
            self.inner.return_on_capital()
        )
    }
}

/// A whole tower quoted at once: each layer standalone from its own ceded
/// grid, and the total cession, every layer together, as one cover.
///
/// Each standalone price holds capital for its layer alone, so the layers'
/// premiums add up to more than the total's; ``diversification`` is the
/// difference. No share of the total is allocated to the layers.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Grid, Poisson
/// >>> from prospicio.pricing import TowerQuote
/// >>> from prospicio.reinsurance import Layer, Tower
/// >>> from prospicio.risk import Distortion
/// >>> sev = Grid(1.0, [0.0, 0.4, 0.3, 0.2, 0.1])
/// >>> tower = Tower([Layer("1x1", 1.0, 1.0), Layer("2x2", 2.0, 2.0)])
/// >>> grids = tower.on_grid(Poisson(3.0), sev, 256)
/// >>> q = TowerQuote.technical(tower, grids, Distortion.tvar(0.99), cost_of_capital=0.1,
/// ...                          expenses=0.03, brokerage=0.10)
/// >>> len(q.layers), q.diversification > 0
/// (2, True)
#[pyclass(name = "TowerQuote", module = "prospicio.pricing", frozen)]
pub(crate) struct PyTowerQuote {
    inner: TowerQuote,
}

#[pymethods]
impl PyTowerQuote {
    /// The technical quote of each layer and of the total cession.
    ///
    /// Each layer is priced on its ceded grid with the reinstatement rate
    /// and flat ceding commission its grid and terms give, and its limit at
    /// the placed share for the rate on line. The total is priced on
    /// ``grids.total_ceded``, with the layers' reinstatement rates and
    /// commissions weighted by their technical deposits and no limit.
    ///
    /// Parameters
    /// ----------
    /// tower : Tower
    /// grids : TowerGrids
    ///     ``tower.on_grid(...)``.
    /// assets : Distortion
    /// cost_of_capital : float, optional
    /// distortion : Distortion, optional
    ///     Exactly one of ``cost_of_capital`` and ``distortion``, as in
    ///     ``price``.
    /// expenses, brokerage, fixed, premium_tax, income_tax : float
    ///     The loads, as in ``Quote.technical``.
    /// brokerage_on_reinstatements : bool, default True
    ///
    /// Returns
    /// -------
    /// TowerQuote
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``grids.total_ceded`` is ``None``: several layers with annual
    ///     terms (reinstatements, aggregate limits or deductibles), or an
    ///     aggregate cover after per-occurrence stages, cede a sum of
    ///     dependent annual totals. Quote the layers one by one with
    ///     ``Quote.technical`` instead.
    #[staticmethod]
    #[pyo3(signature = (
        tower, grids, assets, *, cost_of_capital = None, distortion = None, expenses = 0.0,
        brokerage = 0.0, fixed = 0.0, premium_tax = 0.0, income_tax = 0.0,
        brokerage_on_reinstatements = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn technical(
        tower: PyRef<'_, crate::reinsurance::PyTower>,
        grids: PyRef<'_, crate::reinsurance::PyTowerGrids>,
        assets: PyRef<'_, PyDistortion>,
        cost_of_capital: Option<f64>,
        distortion: Option<PyRef<'_, PyDistortion>>,
        expenses: f64,
        brokerage: f64,
        fixed: f64,
        premium_tax: f64,
        income_tax: f64,
        brokerage_on_reinstatements: bool,
    ) -> PyResult<Self> {
        let rule = premium_rule(cost_of_capital, distortion)?;
        let loads = Loads::new(expenses, brokerage, fixed)
            .and_then(|l| l.with_premium_tax(premium_tax))
            .and_then(|l| l.with_income_tax(income_tax))
            .map_err(to_py)?;
        let inner = TowerQuote::technical(
            &tower.inner,
            &grids.inner,
            &rule,
            &assets.inner,
            loads,
            brokerage_on_reinstatements,
        )
        .map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Each layer's standalone quote, in tower order.
    #[getter]
    fn layers(&self) -> Vec<PyQuote> {
        self.inner
            .layers
            .iter()
            .map(|&inner| PyQuote { inner })
            .collect()
    }

    /// The quote of the total cession.
    #[getter]
    fn total(&self) -> PyQuote {
        PyQuote {
            inner: self.inner.total,
        }
    }

    /// The layers' technical deposits added up less the total's.
    #[getter]
    fn diversification(&self) -> f64 {
        self.inner.diversification()
    }

    fn __repr__(&self) -> String {
        format!(
            "TowerQuote(layers={}, total={}, diversification={})",
            self.inner.layers.len(),
            self.inner.total.premium,
            self.inner.diversification()
        )
    }
}
