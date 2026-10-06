//! `actuarialrs.pricing` (Aggregate lane): the collective model, layer
//! rating and reinsurance tower matching over `act_aggregate` and
//! `act_pricing` (`docs/design/pareto.md`).

use act_aggregate::CollectiveModel;
use act_pricing::exposure::{ExposureCurve, Mbbefd, SeverityCurve, TabulatedCurve};
use act_pricing::layer::XsLayer;
use act_pricing::risk_load::{self, PortfolioPrice, PremiumRule, Price};
use act_pricing::tower::{Reference, SelectionRule, TowerModel};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::aggregate::{AnyCount, PyEventSet};
use crate::distributions::{PyPredictiveDistribution, PySampled, extract_severity, key_to_py};
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
/// >>> from actuarialrs.distributions import Pareto, claim_count
/// >>> from actuarialrs.pricing import CollectiveModel
/// >>> m = CollectiveModel(claim_count(2.0, 1.5), Pareto(1e6, 2.0))
/// >>> round(m.layer_mean(4e6, 1e6))
/// 1600000
/// >>> m.excess_frequency(2e6)
/// 0.5
#[pyclass(name = "CollectiveModel", module = "actuarialrs.pricing", frozen)]
pub(crate) struct PyCollectiveModel {
    inner: CollectiveModel<AnyCount, act_prob::SeverityDist>,
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
    let sev = extract_severity(severity)?;
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
    let sev = extract_severity(severity)?;
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

/// The premium rule from the keyword arguments: exactly one of a cost of
/// capital and a pricing distortion.
fn premium_rule(
    cost_of_capital: Option<f64>,
    distortion: Option<PyRef<'_, PyDistortion>>,
) -> PyResult<PremiumRule> {
    match (cost_of_capital, distortion) {
        (Some(r), None) => PremiumRule::cost_of_capital(r).map_err(to_py),
        (None, Some(d)) => Ok(PremiumRule::Distortion(d.inner)),
        _ => Err(PyValueError::new_err(
            "give exactly one of cost_of_capital and distortion",
        )),
    }
}

/// The risk-loaded price of a cover, or of one component's share of a
/// portfolio: expected loss, premium and the assets backing the loss.
///
/// Returned by ``price`` and ``price_portfolio``.
#[pyclass(name = "Price", module = "actuarialrs.pricing", frozen)]
pub(crate) struct PyPrice {
    inner: Price,
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

    /// Return on capital, margin over capital.
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
#[pyclass(name = "PortfolioPrice", module = "actuarialrs.pricing", frozen)]
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
/// losses : Sampled or PredictiveDistribution
///     Loss draws; for a ``PredictiveDistribution``, its total.
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
/// >>> from actuarialrs.distributions import Sampled
/// >>> from actuarialrs.pricing import price
/// >>> from actuarialrs.risk import Distortion
/// >>> p = price(Sampled([0.0, 0.0, 2.0, 6.0]), Distortion.tvar(0.5), cost_of_capital=0.25)
/// >>> p.premium, p.capital
/// (2.4, 1.6)
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
    } else {
        return Err(PyTypeError::new_err(
            "losses must be a Sampled or a PredictiveDistribution",
        ));
    }
    .map_err(to_py)?;
    Ok(PyPrice { inner })
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
/// >>> from actuarialrs.distributions import PredictiveDistribution
/// >>> from actuarialrs.pricing import price_portfolio
/// >>> from actuarialrs.risk import Distortion
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
    let (pd, a) = (&pd.inner, assets.inner);
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
/// >>> from actuarialrs.pricing import Mbbefd
/// >>> c3 = Mbbefd.swiss_re(3.0)
/// >>> top = c3.layer_share(5e6, 5e6, 10e6)
/// >>> bottom = c3.layer_share(5e6, 0.0, 10e6)
/// >>> round(top + bottom, 12), top < bottom
/// (1.0, True)
#[pyclass(name = "Mbbefd", module = "actuarialrs.pricing", frozen)]
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
/// ``(1, 1)``, interpolated linearly, as published curves are given
/// (Salzmann's homeowners scale, Ludwig's curves, ISO PSOLD tables, a
/// reinsurer's own).
///
/// The table must be concave (its slopes never increase). Its destruction
/// rate is discrete: the points' ``x`` with probabilities from the drops in
/// slope, and a total loss with probability last slope over first. Its
/// mean rate is the first chord's, ``x1 / G(x1)``, so a table needs fine
/// first points for the expected loss to be right.
///
/// Parameters
/// ----------
/// x : list of float
///     Increasing from 0 to 1.
/// g : list of float
///     ``G(x)``, from 0 to 1.
///
/// Raises
/// ------
/// ValueError
///     If the points do not run from ``(0, 0)`` to ``(1, 1)``, or are not
///     increasing and concave.
///
/// Examples
/// --------
/// >>> from actuarialrs.pricing import TabulatedCurve
/// >>> t = TabulatedCurve([0.0, 0.1, 0.5, 1.0], [0.0, 0.4, 0.8, 1.0])
/// >>> round(t.curve([0.3])[0], 12), t.mean_rate()
/// (0.6, 0.25)
#[pyclass(name = "TabulatedCurve", module = "actuarialrs.pricing", frozen)]
pub(crate) struct PyTabulatedCurve {
    pub(crate) inner: TabulatedCurve,
}

#[pymethods]
impl PyTabulatedCurve {
    #[new]
    fn new(x: Vec<f64>, g: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: TabulatedCurve::new(&x, &g).map_err(to_py)?,
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

    fn __getnewargs__(&self) -> (Vec<f64>, Vec<f64>) {
        (self.inner.x().to_vec(), self.inner.g_values().to_vec())
    }

    fn __repr__(&self) -> String {
        format!("TabulatedCurve({} points)", self.inner.x().len())
    }
}

/// A band's exposure curve from a Python ``Mbbefd`` or ``TabulatedCurve``.
fn band_curve(obj: &Bound<'_, PyAny>) -> PyResult<act_pricing::profile::BandCurve> {
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
///
/// Examples
/// --------
/// >>> from actuarialrs.pricing import Mbbefd, RiskProfile
/// >>> p = RiskProfile([1e6, 10e6], [800, 50], Mbbefd.swiss_re(3.0),
/// ...                 premiums=[2e6, 1e6], loss_ratio=0.6)
/// >>> round(p.expected_loss())
/// 1800000
/// >>> events = p.simulate(1000, 7)
/// >>> events.has_sums_insured
/// True
#[pyclass(name = "RiskProfile", module = "actuarialrs.pricing", frozen)]
pub(crate) struct PyRiskProfile {
    inner: act_pricing::profile::RiskProfile,
}

#[pymethods]
impl PyRiskProfile {
    #[new]
    #[pyo3(signature = (sums_insured, risks, curves, expected_losses = None, premiums = None, loss_ratio = None))]
    fn new(
        sums_insured: Vec<f64>,
        risks: Vec<f64>,
        curves: &Bound<'_, PyAny>,
        expected_losses: Option<Vec<f64>>,
        premiums: Option<Vec<f64>>,
        loss_ratio: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        use act_pricing::profile::{Band, RiskProfile};
        use pyo3::exceptions::PyValueError;
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
                    .collect::<act_core::Result<Vec<_>>>()
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
                    .collect::<act_core::Result<Vec<_>>>()
            }
            _ => {
                return Err(PyValueError::new_err(
                    "give expected_losses, or premiums with a loss_ratio",
                ));
            }
        }
        .map_err(to_py)?;
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

    /// Expected number of losses a year, per band.
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
/// >>> from actuarialrs.distributions import Pareto
/// >>> from actuarialrs.pricing import severity_exposure_curve
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
