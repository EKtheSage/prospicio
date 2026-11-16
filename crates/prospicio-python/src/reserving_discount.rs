//! `prospicio.reserving` discounted reserves and the IFRS 17 risk
//! adjustment (issue #553, `docs/design/reserving-ifrs17.md`, PR 2), over
//! `prospicio_reserving::discount`.
//!
//! Each supported fit class has a `discounted` method (in `reserving.rs`,
//! since a class has one `#[pymethods]` block) that calls [`discounted`]
//! here; the curve, the result and the risk adjustment live in this module.

use prospicio_reserving::discount::{
    Capital, CapitalCharge, DiscountCurve, Discountable, DiscountedReserves, Discounting,
    PaymentBasis, PaymentTiming, RiskAdjustment, RiskAdjustmentResult, SegmentRiskAdjustment,
    discounted_reserves,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::distributions::PyPredictiveDistribution;
use crate::reserving::{err, py_repr};
use crate::unknown_name;

/// The payment timings by name: the middle of each period (decision 3a,
/// the default) or its end.
const TIMINGS: [(&str, PaymentTiming); 2] = [
    ("mid", PaymentTiming::MidPeriod),
    ("end", PaymentTiming::EndOfPeriod),
];

/// When the cost of capital is charged: the end of each year (Solvency II,
/// the default) or its start.
const CHARGES: [(&str, CapitalCharge); 2] = [
    ("end", CapitalCharge::EndOfYear),
    ("start", CapitalCharge::StartOfYear),
];

const BASES: [(&str, bool); 2] = [("paid", false), ("incurred", true)];

const METHODS: [(&str, u8); 3] = [("confidence", 0), ("tvar", 1), ("cost_of_capital", 2)];

const CAPITALS: [(&str, u8); 3] = [("var", 0), ("tvar", 1), ("amount", 2)];

fn named<T: Copy>(arg: &str, got: &str, names: &[(&str, T)]) -> PyResult<T> {
    names
        .iter()
        .find(|(n, _)| *n == got)
        .map(|(_, v)| *v)
        .ok_or_else(|| unknown_name(arg, got, names))
}

fn timing_name(t: PaymentTiming) -> &'static str {
    match t {
        PaymentTiming::MidPeriod => "mid",
        PaymentTiming::EndOfPeriod => "end",
    }
}

/// Annual effective spot rates by maturity in years, annually compounded:
/// linear in the rate between maturities and flat before the first and
/// after the last. The discount factor of a payment at ``t`` years is
/// ``(1 + r(t)) ** -t``.
///
/// Parameters
/// ----------
/// maturities : list of float
///     Maturities in years: finite, positive and strictly increasing.
/// rates : list of float
///     The spot rate at each maturity: finite and above -1.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import DiscountCurve
/// >>> curve = DiscountCurve([1.0, 5.0], [0.02, 0.04])
/// >>> curve.rate(3.0)
/// 0.03
/// >>> round(curve.factor(2.0), 6)
/// 0.951814
/// >>> DiscountCurve.flat(0.03)
/// DiscountCurve(maturities=[1.0], rates=[0.03])
#[pyclass(name = "DiscountCurve", module = "prospicio.reserving", frozen)]
pub(crate) struct PyDiscountCurve {
    pub(crate) inner: DiscountCurve,
}

#[pymethods]
impl PyDiscountCurve {
    #[new]
    fn new(maturities: Vec<f64>, rates: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: DiscountCurve::spot(&maturities, &rates).map_err(err)?,
        })
    }

    /// One annual effective rate at every maturity.
    ///
    /// Parameters
    /// ----------
    /// rate : float
    ///     Finite and above -1.
    ///
    /// Returns
    /// -------
    /// DiscountCurve
    #[staticmethod]
    fn flat(rate: f64) -> PyResult<Self> {
        Ok(Self {
            inner: DiscountCurve::flat(rate).map_err(err)?,
        })
    }

    /// The maturities in years (``[1.0]`` for a flat curve).
    #[getter]
    fn maturities(&self) -> Vec<f64> {
        self.inner.maturities().to_vec()
    }

    /// The spot rate at each maturity.
    #[getter]
    fn rates(&self) -> Vec<f64> {
        self.inner.rates().to_vec()
    }

    /// The spot rate at ``t`` years.
    ///
    /// Parameters
    /// ----------
    /// t : float
    ///
    /// Returns
    /// -------
    /// float
    fn rate(&self, t: f64) -> f64 {
        self.inner.rate(t)
    }

    /// The discount factor of a payment at ``t`` years, ``(1 + r(t)) ** -t``.
    ///
    /// Parameters
    /// ----------
    /// t : float
    ///
    /// Returns
    /// -------
    /// float
    fn factor(&self, t: f64) -> f64 {
        self.inner.factor(t)
    }

    fn __getnewargs__(&self) -> (Vec<f64>, Vec<f64>) {
        (self.maturities(), self.rates())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyDiscountCurve>>()
            .is_ok_and(|o| o.inner == self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "DiscountCurve(maturities={:?}, rates={:?})",
            self.inner.maturities(),
            self.inner.rates()
        )
    }
}

/// The settings of a fit's `discounted` method, as Python gives them.
pub(crate) struct DiscountArgs<'py> {
    pub(crate) basis: &'py str,
    pub(crate) rate: Option<f64>,
    pub(crate) curve: Option<PyRef<'py, PyDiscountCurve>>,
    pub(crate) timing: &'py str,
    pub(crate) paid_pattern: Option<Vec<f64>>,
    pub(crate) latest_paid: Option<Vec<f64>>,
}

impl DiscountArgs<'_> {
    fn resolve(self) -> PyResult<(Discounting, PaymentBasis)> {
        let curve = match (self.rate, self.curve) {
            (Some(r), None) => DiscountCurve::flat(r).map_err(err)?,
            (None, Some(c)) => c.inner.clone(),
            _ => {
                return Err(PyValueError::new_err(
                    "give exactly one of rate (a flat annual rate) and curve (a DiscountCurve)",
                ));
            }
        };
        let timing = named("timing", self.timing, &TIMINGS)?;
        let incurred = named("basis", self.basis, &BASES)?;
        let basis = match (incurred, self.paid_pattern, self.latest_paid) {
            (false, None, None) => PaymentBasis::Paid,
            (false, _, _) => {
                return Err(PyValueError::new_err(
                    "paid_pattern and latest_paid are for basis=\"incurred\"; a fit on paid \
                     claims is paid along its own patterns",
                ));
            }
            (true, Some(paid_pattern), Some(latest_paid)) => PaymentBasis::Incurred {
                paid_pattern,
                latest_paid,
            },
            (true, _, _) => {
                return Err(PyValueError::new_err(
                    "basis=\"incurred\" needs paid_pattern (the cumulative paid share at each \
                     of the fit's pattern ages) and latest_paid (one per component of the \
                     reserves)",
                ));
            }
        };
        Ok((Discounting { curve, timing }, basis))
    }
}

/// [`discounted_reserves`] of any supported fit, for its `discounted`
/// method.
pub(crate) fn discounted<F: Discountable + Sync + ?Sized>(
    py: Python<'_>,
    fit: &F,
    args: DiscountArgs<'_>,
) -> PyResult<PyDiscountedReserves> {
    let (discounting, basis) = args.resolve()?;
    let inner = py
        .detach(|| discounted_reserves(fit, &discounting, &basis))
        .map_err(err)?;
    Ok(PyDiscountedReserves { inner })
}

/// A fit's unpaid claims discounted simulation by simulation, from a fit's
/// ``discounted`` method (``docs/design/reserving-ifrs17.md``).
///
/// Each simulation's unpaid (the reserves on the paid basis; the reserves
/// plus each origin's case reserve on the incurred basis) is spread over
/// that simulation's development pattern (the rises of its running
/// maximum after each origin's latest age), timed at the middle or end of
/// each development period and discounted on the curve; ``Compartmental``
/// is paid by its simulated paid cells instead. Quantiles of the
/// discounted total are read from the discounted draws, never by scaling
/// undiscounted quantiles.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import MackBootstrap, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2019] * 4 + [2020] * 3 + [2021] * 2 + [2022],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     {"paid": [100.0, 150, 170, 175, 110, 160, 185, 120, 185, 130]},
/// ... )
/// >>> fit = MackBootstrap(n_sims=1000, seed=1).fit(tri, "paid")
/// >>> d = fit.discounted(basis="paid", rate=0.03)
/// >>> d.best_estimate < d.unpaid.mean()
/// True
/// >>> ra = d.risk_adjustment("confidence", 0.75)
/// >>> ra.implied_confidence >= 0.75
/// True
#[pyclass(name = "DiscountedReserves", module = "prospicio.reserving", frozen)]
pub(crate) struct PyDiscountedReserves {
    inner: DiscountedReserves,
}

fn capital_arg(obj: &Bound<'_, PyAny>) -> PyResult<Capital> {
    if let Ok(k) = obj.extract::<f64>() {
        return Ok(Capital::Amount(k));
    }
    if let Ok((name, x)) = obj.extract::<(String, f64)>() {
        return Ok(match named("capital", &name, &CAPITALS)? {
            0 => Capital::Var(x),
            1 => Capital::Tvar(x),
            _ => Capital::Amount(x),
        });
    }
    Err(PyTypeError::new_err(format!(
        "capital must be an amount or a pair (\"var\" | \"tvar\" | \"amount\", value), got {}",
        py_repr(obj)
    )))
}

#[pymethods]
impl PyDiscountedReserves {
    /// The unpaid claims, undiscounted: the fit's reserves on the paid
    /// basis, the reserves plus each origin's case reserve on the incurred
    /// basis. The same components and simulations as the fit's
    /// ``reserves``.
    #[getter]
    fn unpaid(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.unpaid.clone(),
        }
    }

    /// The present value of each draw of ``unpaid``: the same components
    /// and simulations.
    #[getter]
    fn discounted(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.discounted.clone(),
        }
    }

    /// The best estimate: the mean discounted total.
    #[getter]
    fn best_estimate(&self) -> f64 {
        self.inner.best_estimate
    }

    /// The best estimate outstanding at ``t = 0, 1, 2, ...`` whole years from
    /// the valuation date, valued at ``t``, ending with 0 at the first whole
    /// year after the last payment. ``runoff[0]`` is the best estimate.
    #[getter]
    fn runoff(&self) -> Vec<f64> {
        self.inner.runoff.clone()
    }

    /// Each component's mean present value over its mean unpaid, in the
    /// order of the components; 1 where the mean unpaid is 0.
    #[getter]
    fn discount_factors(&self) -> Vec<f64> {
        self.inner.discount_factors.clone()
    }

    /// The curve the unpaid claims were discounted on.
    #[getter]
    fn curve(&self) -> PyDiscountCurve {
        PyDiscountCurve {
            inner: self.inner.discounting().curve.clone(),
        }
    }

    /// When in each development period the payments fall: ``"mid"`` or
    /// ``"end"``.
    #[getter]
    fn timing(&self) -> &'static str {
        timing_name(self.inner.discounting().timing)
    }

    /// Each segment's label as a list of key values, for a fit over
    /// segments; empty for a fit of one segment.
    #[getter]
    fn segment_labels(&self) -> Vec<Vec<String>> {
        self.inner.segment_labels()
    }

    /// Segment ``i``'s run-off, as ``runoff``, in the order of
    /// ``segment_labels``.
    ///
    /// Parameters
    /// ----------
    /// i : int
    ///
    /// Returns
    /// -------
    /// list of float
    fn segment_runoff(&self, i: usize) -> PyResult<Vec<f64>> {
        self.inner
            .segment_runoff(i)
            .map(<[f64]>::to_vec)
            .ok_or_else(|| PyValueError::new_err(format!("no segment {i}")))
    }

    /// The risk adjustment on the discounted total ``X``, whose mean is the
    /// best estimate ``BE``, with the confidence level it corresponds to
    /// (IFRS 17 para. 119), and for a fit over segments each segment's
    /// stand-alone one (not an allocation of the total).
    ///
    /// Parameters
    /// ----------
    /// method : {"confidence", "tvar", "cost_of_capital"}
    ///     ``"confidence"``: ``Q_p(X) - BE``, the inverted-cdf quantile.
    ///     ``"tvar"``: ``TVaR_p(X) - BE``. ``"cost_of_capital"``: ``rate``
    ///     times the capital held in each future year, discounted, the
    ///     capital running off with the best estimate:
    ///     ``K_t = K_0 runoff[t] / runoff[0]``.
    /// p : float, optional
    ///     The level of ``"confidence"`` and ``"tvar"``, in ``[0, 1]``.
    /// rate : float, optional
    ///     The cost-of-capital rate, for ``"cost_of_capital"``.
    /// capital : float or tuple, optional
    ///     ``K_0`` for ``"cost_of_capital"`` (no default): a pair
    ///     ``("var", q)`` (``Q_q(X) - BE``) or ``("tvar", q)``, or an amount
    ///     (a float or ``("amount", k)``), such as the one-year view's
    ///     99.5% loss. An amount is the total's, so no segment results are
    ///     reported for it.
    /// charge : {"end", "start"}, default "end"
    ///     Charge each year's capital at its end, ``rate sum K_t v(t + 1)``
    ///     (Solvency II's risk margin), or its start,
    ///     ``rate sum K_t v(t)`` (``price_over_payout``).
    ///
    /// Returns
    /// -------
    /// RiskAdjustmentResult
    #[pyo3(signature = (method, p = None, *, rate = None, capital = None, charge = "end"))]
    fn risk_adjustment(
        &self,
        method: &str,
        p: Option<f64>,
        rate: Option<f64>,
        capital: Option<&Bound<'_, PyAny>>,
        charge: &str,
    ) -> PyResult<PyRiskAdjustmentResult> {
        let kind = named("method", method, &METHODS)?;
        let charge = named("charge", charge, &CHARGES)?;
        let level = |p: Option<f64>| {
            p.ok_or_else(|| PyValueError::new_err(format!("method={method:?} needs p")))
        };
        let method = match kind {
            0 | 1 => {
                if rate.is_some() || capital.is_some() {
                    return Err(PyValueError::new_err(
                        "rate and capital are for method=\"cost_of_capital\"",
                    ));
                }
                let p = level(p)?;
                if kind == 0 {
                    RiskAdjustment::ConfidenceLevel(p)
                } else {
                    RiskAdjustment::Tvar(p)
                }
            }
            _ => {
                if p.is_some() {
                    return Err(PyValueError::new_err(
                        "method=\"cost_of_capital\" takes capital, not p: capital=(\"var\", q)",
                    ));
                }
                let rate = rate.ok_or_else(|| {
                    PyValueError::new_err("method=\"cost_of_capital\" needs rate")
                })?;
                let capital = capital.ok_or_else(|| {
                    PyValueError::new_err(
                        "method=\"cost_of_capital\" needs capital: (\"var\", q), (\"tvar\", q) \
                         or an amount",
                    )
                })?;
                RiskAdjustment::CostOfCapital {
                    rate,
                    capital: capital_arg(capital)?,
                    charge,
                }
            }
        };
        Ok(PyRiskAdjustmentResult {
            inner: self.inner.risk_adjustment(&method).map_err(err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "DiscountedReserves(best_estimate={:.6}, timing={:?})",
            self.inner.best_estimate,
            self.timing()
        )
    }
}

/// A risk adjustment and the confidence level it corresponds to, from
/// ``DiscountedReserves.risk_adjustment``.
#[pyclass(name = "RiskAdjustmentResult", module = "prospicio.reserving", frozen)]
pub(crate) struct PyRiskAdjustmentResult {
    inner: RiskAdjustmentResult,
}

#[pymethods]
impl PyRiskAdjustmentResult {
    /// The risk adjustment.
    #[getter]
    fn amount(&self) -> f64 {
        self.inner.amount
    }

    /// The mean discounted total.
    #[getter]
    fn best_estimate(&self) -> f64 {
        self.inner.best_estimate
    }

    /// The share of the discounted draws at or below ``best_estimate +
    /// amount`` (IFRS 17 para. 119): under ``"confidence"`` at level ``p``
    /// at least ``p``.
    #[getter]
    fn implied_confidence(&self) -> f64 {
        self.inner.implied_confidence
    }

    /// The capital at the valuation date, ``K_0``, of a cost-of-capital
    /// method; ``None`` otherwise.
    #[getter]
    fn capital(&self) -> Option<f64> {
        self.inner.capital
    }

    /// Each segment's stand-alone result, for a fit over segments; empty
    /// for a fit of one segment and under a capital amount.
    #[getter]
    fn segments(&self) -> Vec<PySegmentRiskAdjustment> {
        self.inner
            .segments
            .iter()
            .map(|s| PySegmentRiskAdjustment { inner: s.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "RiskAdjustmentResult(amount={:.6}, best_estimate={:.6}, implied_confidence={:.6})",
            self.inner.amount, self.inner.best_estimate, self.inner.implied_confidence
        )
    }
}

/// One segment's stand-alone risk adjustment, in
/// ``RiskAdjustmentResult.segments``.
#[pyclass(name = "SegmentRiskAdjustment", module = "prospicio.reserving", frozen)]
pub(crate) struct PySegmentRiskAdjustment {
    inner: SegmentRiskAdjustment,
}

#[pymethods]
impl PySegmentRiskAdjustment {
    /// The segment's label, one value per key.
    #[getter]
    fn label(&self) -> Vec<String> {
        self.inner.label.clone()
    }

    /// Its risk adjustment on its own discounted total and run-off.
    #[getter]
    fn result(&self) -> PyRiskAdjustmentResult {
        PyRiskAdjustmentResult {
            inner: self.inner.result.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SegmentRiskAdjustment(label={:?}, amount={:.6})",
            self.inner.label, self.inner.result.amount
        )
    }
}
