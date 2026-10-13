//! `prospicio.pricing` (Aggregate lane): the time value of money in a price
//! over `prospicio_pricing::payout` (`docs/design/time-value.md`).

use prospicio_pricing::payout::{self, PaymentPattern, PayoutPrice};
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
