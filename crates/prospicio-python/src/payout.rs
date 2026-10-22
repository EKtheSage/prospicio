//! `prospicio.pricing` (Aggregate lane): the time value of money in a price
//! over `prospicio_pricing::payout`, and return metrics on simulated cash
//! flows over `prospicio_pricing::returns` (`docs/design/time-value.md`).

use prospicio_pricing::payout::{self, PaymentPattern, PayoutPrice};
use prospicio_pricing::returns::{self, IrrOptions, TieredFee, TimeUnit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::pricing::{PyPrice, premium_rule};
use crate::risk::PyDistortion;
use crate::to_py;

/// The share of the ultimate loss paid at each time, in years from the
/// treaty's inception.
///
/// With a pattern that does not depend on the size of the loss, every
/// simulated year's present value is its loss times ``discount_factor``, so
/// a loss distribution is discounted by scaling it.
///
/// Parameters
/// ----------
/// times : list of float
///     Non-negative, strictly increasing times, in years from inception.
/// shares : list of float
///     The share paid at each time; they sum to 1.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import PaymentPattern
/// >>> p = PaymentPattern.from_development([1.0, 2.0, 3.0], [4.0, 1.25, 1.0])
/// >>> [round(s, 12) for s in p.shares]
/// [0.25, 0.55, 0.2]
/// >>> round(p.duration, 12), round(p.discount_factor(0.05), 6)
/// (1.95, 0.909729)
#[pyclass(name = "PaymentPattern", module = "prospicio.pricing", frozen)]
pub(crate) struct PyPaymentPattern {
    inner: PaymentPattern,
}

#[pymethods]
impl PyPaymentPattern {
    #[new]
    fn new(times: Vec<f64>, shares: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: PaymentPattern::new(times, shares).map_err(to_py)?,
        })
    }

    /// From development factors to ultimate by age, such as a paid chain
    /// ladder's ``cdf``: the share paid by each age is one over its factor.
    ///
    /// The last factor must be 1 (add any tail as a last age). Ages are read
    /// as times from inception; ``shifted`` moves the pattern when losses
    /// start developing later.
    ///
    /// Parameters
    /// ----------
    /// ages : list of float
    /// factors : list of float
    ///
    /// Returns
    /// -------
    /// PaymentPattern
    #[staticmethod]
    fn from_development(ages: Vec<f64>, factors: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: PaymentPattern::from_development(&ages, &factors).map_err(to_py)?,
        })
    }

    /// The pattern ``by`` years later.
    ///
    /// Parameters
    /// ----------
    /// by : float
    ///
    /// Returns
    /// -------
    /// PaymentPattern
    fn shifted(&self, by: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.shifted(by).map_err(to_py)?,
        })
    }

    /// The times.
    #[getter]
    fn times(&self) -> Vec<f64> {
        self.inner.times().to_vec()
    }

    /// The share paid at each time.
    #[getter]
    fn shares(&self) -> Vec<f64> {
        self.inner.shares().to_vec()
    }

    /// The mean time to payment, undiscounted.
    #[getter]
    fn duration(&self) -> f64 {
        self.inner.duration()
    }

    /// The share not yet paid at time ``t``.
    ///
    /// Parameters
    /// ----------
    /// t : float
    ///
    /// Returns
    /// -------
    /// float
    fn outstanding(&self, t: f64) -> f64 {
        self.inner.outstanding(t)
    }

    /// The present value at inception of a unit of ultimate loss, at an
    /// annual effective ``rate``.
    ///
    /// Parameters
    /// ----------
    /// rate : float
    ///
    /// Returns
    /// -------
    /// float
    fn discount_factor(&self, rate: f64) -> f64 {
        self.inner.discount_factor(rate)
    }

    /// The discounted capital years per unit of initial capital: capital
    /// held in each year in proportion to the share outstanding at its
    /// start, charged at the start of the year.
    ///
    /// Parameters
    /// ----------
    /// rate : float
    ///
    /// Returns
    /// -------
    /// float
    fn capital_years(&self, rate: f64) -> f64 {
        self.inner.capital_years(rate)
    }

    fn __repr__(&self) -> String {
        format!(
            "PaymentPattern(times={:?}, shares={:?})",
            self.inner.times(),
            self.inner.shares()
        )
    }
}

/// A price restated over a payout, from ``price_over_payout``.
#[pyclass(name = "PayoutPrice", module = "prospicio.pricing", frozen)]
pub(crate) struct PyPayoutPrice {
    inner: PayoutPrice,
}

#[pymethods]
impl PyPayoutPrice {
    /// Expected loss, undiscounted.
    #[getter]
    fn expected_loss(&self) -> f64 {
        self.inner.expected_loss
    }

    /// Expected loss discounted to inception.
    #[getter]
    fn discounted_loss(&self) -> f64 {
        self.inner.discounted_loss
    }

    /// Premium at inception.
    #[getter]
    fn premium(&self) -> f64 {
        self.inner.premium
    }

    /// Assets, the one-year risk measure.
    #[getter]
    fn assets(&self) -> f64 {
        self.inner.assets
    }

    /// Initial capital, assets less premium.
    #[getter]
    fn capital(&self) -> f64 {
        self.inner.capital()
    }

    /// Margin over the discounted loss.
    #[getter]
    fn margin(&self) -> f64 {
        self.inner.margin()
    }

    /// The losses' discount factor.
    #[getter]
    fn discount_factor(&self) -> f64 {
        self.inner.discount_factor
    }

    /// The discounted capital years (1 for a distortion price).
    #[getter]
    fn capital_years(&self) -> f64 {
        self.inner.capital_years
    }

    fn __repr__(&self) -> String {
        format!(
            "PayoutPrice(premium={}, discounted_loss={}, capital_years={})",
            self.inner.premium, self.inner.discounted_loss, self.inner.capital_years
        )
    }
}

/// A one-period price restated over a payout.
///
/// With a cost of capital ``r``, the premium solves
/// ``P = E[X] d + r (a - P) A``: the loss discounted by the pattern's
/// ``discount_factor`` ``d``, plus the cost of the initial capital held while
/// losses are outstanding over ``A = capital_years``. With a pricing
/// distortion, the premium is discounted with the losses. Give the same rule
/// the price was made with. A pattern that pays everything within the first
/// year, at a zero rate, gives back the one-period price.
///
/// Parameters
/// ----------
/// price : Price
///     The one-period price, from ``price``.
/// pattern : PaymentPattern
/// rate : float, default 0.0
///     Annual effective discount rate.
/// cost_of_capital : float, optional
/// distortion : Distortion, optional
///
/// Returns
/// -------
/// PayoutPrice
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Sampled
/// >>> from prospicio.pricing import PaymentPattern, price, price_over_payout
/// >>> from prospicio.risk import Distortion
/// >>> p = price(Sampled([0.0, 0.0, 2.0, 6.0]), Distortion.tvar(0.5), cost_of_capital=0.25)
/// >>> slow = PaymentPattern([1.0, 2.0, 3.0], [0.2, 0.3, 0.5])
/// >>> q = price_over_payout(p, slow, rate=0.0, cost_of_capital=0.25)
/// >>> round(q.capital_years, 12), round(q.premium, 6)
/// (2.3, 2.730159)
#[pyfunction]
#[pyo3(signature = (price, pattern, *, rate = 0.0, cost_of_capital = None, distortion = None))]
pub(crate) fn price_over_payout(
    price: PyRef<'_, PyPrice>,
    pattern: PyRef<'_, PyPaymentPattern>,
    rate: f64,
    cost_of_capital: Option<f64>,
    distortion: Option<PyRef<'_, PyDistortion>>,
) -> PyResult<PyPayoutPrice> {
    let rule = premium_rule(cost_of_capital, distortion)?;
    Ok(PyPayoutPrice {
        inner: payout::price_over_payout(&price.inner, &rule, &pattern.inner, rate)
            .map_err(to_py)?,
    })
}

fn time_unit(unit: &str) -> PyResult<TimeUnit> {
    match unit {
        "years" => Ok(TimeUnit::Years),
        "months" => Ok(TimeUnit::Months),
        "days" => Ok(TimeUnit::Days),
        other => Err(PyValueError::new_err(format!(
            "unknown unit {other:?}; use \"years\", \"months\" or \"days\""
        ))),
    }
}

/// One simulation's flows a row, all rows as long as the times.
fn flow_matrix(flows: Vec<Vec<f64>>, times: usize) -> PyResult<Vec<f64>> {
    if let Some((k, row)) = flows.iter().enumerate().find(|(_, r)| r.len() != times) {
        return Err(PyValueError::new_err(format!(
            "row {k} has {} flows for {times} times",
            row.len()
        )));
    }
    Ok(flows.concat())
}

/// The internal rates of return of simulated cash flows, from ``irr``.
#[pyclass(name = "IrrResult", module = "prospicio.pricing", frozen)]
pub(crate) struct PyIrrResult {
    pub(crate) inner: Vec<returns::Irr>,
}

#[pymethods]
impl PyIrrResult {
    /// Each simulation's annual rate, NaN where no root was found.
    #[getter]
    fn rate(&self) -> Vec<f64> {
        self.inner.iter().map(|r| r.rate).collect()
    }

    /// Whether each simulation's search found a root.
    #[getter]
    fn found(&self) -> Vec<bool> {
        self.inner.iter().map(|r| r.found).collect()
    }

    /// The net present value at each rate, NaN where no root was found.
    #[getter]
    fn npv(&self) -> Vec<f64> {
        self.inner.iter().map(|r| r.npv).collect()
    }

    /// The roots each search saw on its grid: more than one means several
    /// rates of return, and ``rate`` is the one nearest 0.
    #[getter]
    fn sign_changes(&self) -> Vec<usize> {
        self.inner.iter().map(|r| r.sign_changes).collect()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        let found = self.inner.iter().filter(|r| r.found).count();
        format!("IrrResult(n={}, found={found})", self.inner.len())
    }
}

/// The net present value of each simulation's cash flows at an annual
/// rate, ``sum_k x_k (1 + rate)^-t_k``.
///
/// Parameters
/// ----------
/// flows : list of list of float
///     One simulation's flows a row, one column a time.
/// times : list of float
///     The time of each column, in ``unit`` from the start.
/// rate : float
///     Annual effective rate, above -1.
/// unit : str, default "years"
///     ``"years"``, ``"months"`` (a twelfth of a year) or ``"days"``
///     (actual/365).
///
/// Returns
/// -------
/// list of float
///
/// Examples
/// --------
/// >>> from prospicio.pricing import npv
/// >>> [abs(round(v, 12)) for v in npv([[-100.0, 121.0]], [0.0, 2.0], 0.1)]
/// [0.0]
#[pyfunction]
#[pyo3(signature = (flows, times, rate, *, unit = "years"))]
pub(crate) fn npv(
    py: Python<'_>,
    flows: Vec<Vec<f64>>,
    times: Vec<f64>,
    rate: f64,
    unit: &str,
) -> PyResult<Vec<f64>> {
    let times = time_unit(unit)?.years(&times);
    let flows = flow_matrix(flows, times.len())?;
    py.detach(|| returns::npv_rows(&flows, &times, rate))
        .map_err(to_py)
}

/// The internal rate of return of each simulation's cash flows, an annual
/// rate discounting by ``(1 + r)^-t``.
///
/// The flows' value is computed on a grid of ``points`` rates from
/// ``lower`` to ``upper``; of its sign changes (and exact zeros) the one
/// whose bracket's midpoint is nearest 0 is halved ``bisections`` times and
/// refined by ``newton_steps`` Newton steps clipped to the bracket. A
/// simulation with no root in the range (flows of one sign, say) has
/// ``found`` false and NaN; one with several has ``sign_changes`` above 1.
/// The defaults are a reference simulation model's.
///
/// Parameters
/// ----------
/// flows : list of list of float
///     One simulation's flows a row, one column a time.
/// times : list of float
///     The time of each column, in ``unit`` from the start.
/// unit : str, default "years"
///     ``"years"``, ``"months"`` (a twelfth of a year) or ``"days"``
///     (actual/365).
/// lower, upper : float, optional
///     The range searched, by default -0.95 to 1.
/// points : int, default 400
/// bisections : int, default 8
/// newton_steps : int, default 3
///
/// Returns
/// -------
/// IrrResult
///
/// Examples
/// --------
/// >>> from prospicio.pricing import irr
/// >>> r = irr([[-100.0, 230.0, -132.0], [10.0, 5.0, 1.0]], [0.0, 1.0, 2.0])
/// >>> [round(x, 12) for x in r.rate[:1]], r.found, r.sign_changes
/// ([0.1], [True, False], [2, 0])
#[pyfunction]
#[pyo3(signature = (flows, times, *, unit = "years", lower = None, upper = None, points = 400, bisections = 8, newton_steps = 3))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn irr(
    py: Python<'_>,
    flows: Vec<Vec<f64>>,
    times: Vec<f64>,
    unit: &str,
    lower: Option<f64>,
    upper: Option<f64>,
    points: usize,
    bisections: usize,
    newton_steps: usize,
) -> PyResult<PyIrrResult> {
    let times = time_unit(unit)?.years(&times);
    let flows = flow_matrix(flows, times.len())?;
    let d = IrrOptions::default();
    let options = IrrOptions {
        lower: lower.unwrap_or(d.lower),
        upper: upper.unwrap_or(d.upper),
        points,
        bisections,
        newton_steps,
    };
    let inner = py
        .detach(|| returns::irr_rows(&flows, &times, &options))
        .map_err(to_py)?;
    Ok(PyIrrResult { inner })
}

/// The multiple on invested capital, ``1 + released / (funded + opening)``.
///
/// Parameters
/// ----------
/// released : float
///     Equity released over the term.
/// funded : float
///     Capital funded during the term.
/// opening : float
///     Capital held at the start.
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from prospicio.pricing import moic
/// >>> moic(30.0, 40.0, 60.0)
/// 1.3
#[pyfunction]
pub(crate) fn moic(released: f64, funded: f64, opening: f64) -> PyResult<f64> {
    returns::moic(released, funded, opening).map_err(to_py)
}

/// The leverage of written premium on capital,
/// ``rate * premium / (funded + opening)``. The rate is the caller's.
///
/// Parameters
/// ----------
/// rate : float
/// premium : float
///     Written premium.
/// funded : float
///     Capital funded during the term.
/// opening : float
///     Capital held at the start.
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from prospicio.pricing import leverage
/// >>> leverage(0.5, 300.0, 50.0, 100.0)
/// 1.0
#[pyfunction]
pub(crate) fn leverage(rate: f64, premium: f64, funded: f64, opening: f64) -> PyResult<f64> {
    returns::leverage(rate, premium, funded, opening).map_err(to_py)
}

/// A fee tiered on the margin: slope ``slopes[i]`` on the part of each
/// margin between ``breakpoints[i]`` and ``breakpoints[i + 1]``, the last
/// tier unbounded, nothing below the first breakpoint.
///
/// Parameters
/// ----------
/// margins : list of float
/// breakpoints : list of float
///     Strictly increasing; the first is the floor.
/// slopes : list of float
///     One slope a breakpoint, the slope of the tier that starts there.
///
/// Returns
/// -------
/// list of float
///
/// Examples
/// --------
/// >>> from prospicio.pricing import tiered_fee
/// >>> [round(f, 12) for f in tiered_fee([3.0, 10.0, 25.0], [5.0, 15.0], [0.1, 0.2])]
/// [0.0, 0.5, 3.0]
#[pyfunction]
pub(crate) fn tiered_fee(
    margins: Vec<f64>,
    breakpoints: Vec<f64>,
    slopes: Vec<f64>,
) -> PyResult<Vec<f64>> {
    let fee = TieredFee::new(breakpoints, slopes).map_err(to_py)?;
    Ok(margins.iter().map(|&m| fee.fee(m)).collect())
}
