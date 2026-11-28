//! Python bindings for `prospicio_reserving::ifrs17`: the IFRS 17
//! measurement of a group of contracts under the general model, the
//! premium allocation approach, and reinsurance contracts held
//! (`reserving-ifrs17-measurement.md`, PR 5). Each group is rolled forward
//! in place, a period at a time, and saved with `to_json` or `pickle`.

use prospicio_reserving::ifrs17::paa::{
    Lic, PaaBalances, PaaGroup, PaaInputs, PaaOptions, PaaPeriod,
};
use prospicio_reserving::ifrs17::reinsurance::{
    Allocation, CededActual, CededCashFlows, CededLine, CededRow, LossRecoveryMovement,
    NonPerformance, Recovered, ReinsuranceBalances, ReinsuranceHeld, ReinsuranceInputs,
    ReinsuranceOptions, ReinsurancePeriod,
};
use prospicio_reserving::ifrs17::{
    ActualCashFlows, Balances, CashFlowTiming, ComponentLine, ComponentRow, CoverageLine,
    CoverageRow, ExpectedCashFlows, Group, GroupOptions, GroupPeriod, PeriodInputs, Revenue,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::reserving_discount::PyDiscountCurve;

fn err(e: prospicio_reserving::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

const TIMINGS: &[(&str, CashFlowTiming)] = &[
    ("start", CashFlowTiming::StartOfPeriod),
    ("mid", CashFlowTiming::MidPeriod),
    ("end", CashFlowTiming::EndOfPeriod),
];

fn timing(arg: &str, name: &str) -> PyResult<CashFlowTiming> {
    TIMINGS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|&(_, t)| t)
        .ok_or_else(|| crate::unknown_name(arg, name, TIMINGS))
}

/// A dict's float entry, `default` when absent.
fn get(d: &Bound<'_, PyDict>, key: &str, default: f64) -> PyResult<f64> {
    match d.get_item(key)? {
        Some(v) => v.extract(),
        None => Ok(default),
    }
}

fn get_list(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Vec<f64>>> {
    match d.get_item(key)? {
        Some(v) => Ok(Some(v.extract()?)),
        None => Ok(None),
    }
}

fn check_keys(d: &Bound<'_, PyDict>, what: &str, allowed: &[&str]) -> PyResult<()> {
    for k in d.keys() {
        let k: String = k.extract()?;
        if !allowed.contains(&k.as_str()) {
            return Err(PyValueError::new_err(format!(
                "{what} has an unknown key {k:?}; the keys are {}",
                allowed.join(", ")
            )));
        }
    }
    Ok(())
}

/// Expected cash flows from a dict of lists by future period.
fn expected(d: &Bound<'_, PyDict>) -> PyResult<ExpectedCashFlows> {
    check_keys(
        d,
        "the expected cash flows",
        &[
            "premiums",
            "claims",
            "expenses",
            "acquisition",
            "investment_component",
            "premium_timing",
            "claim_timing",
            "acquisition_timing",
        ],
    )?;
    let streams = [
        "premiums",
        "claims",
        "expenses",
        "acquisition",
        "investment_component",
    ];
    let mut lists = Vec::with_capacity(streams.len());
    for s in streams {
        lists.push(get_list(d, s)?);
    }
    let n = lists.iter().flatten().map(Vec::len).max().unwrap_or(0);
    let mut it = lists.into_iter().map(|l| l.unwrap_or_else(|| vec![0.0; n]));
    let mut cash = ExpectedCashFlows::new(n)
        .with_premiums(it.next().unwrap())
        .with_claims(it.next().unwrap())
        .with_expenses(it.next().unwrap())
        .with_acquisition(it.next().unwrap())
        .with_investment_component(it.next().unwrap());
    let t = |key: &str, default: CashFlowTiming| -> PyResult<CashFlowTiming> {
        match d.get_item(key)? {
            Some(v) => timing(key, &v.extract::<String>()?),
            None => Ok(default),
        }
    };
    let (p, c, a) = (
        t("premium_timing", cash.premium_timing)?,
        t("claim_timing", cash.claim_timing)?,
        t("acquisition_timing", cash.acquisition_timing)?,
    );
    cash = cash.with_timing(p, c, a);
    Ok(cash)
}

fn actual(d: &Bound<'_, PyDict>) -> PyResult<ActualCashFlows> {
    check_keys(
        d,
        "the actual cash flows",
        &[
            "premiums",
            "premiums_for_future_service",
            "claims",
            "expenses",
            "acquisition",
            "investment_component",
        ],
    )?;
    Ok(ActualCashFlows {
        premiums: get(d, "premiums", 0.0)?,
        premiums_for_future_service: get(d, "premiums_for_future_service", 0.0)?,
        claims: get(d, "claims", 0.0)?,
        expenses: get(d, "expenses", 0.0)?,
        acquisition: get(d, "acquisition", 0.0)?,
        investment_component: get(d, "investment_component", 0.0)?,
    })
}

fn actual_dict<'py>(py: Python<'py>, a: ActualCashFlows) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("premiums", a.premiums)?;
    d.set_item("premiums_for_future_service", a.premiums_for_future_service)?;
    d.set_item("claims", a.claims)?;
    d.set_item("expenses", a.expenses)?;
    d.set_item("acquisition", a.acquisition)?;
    d.set_item("investment_component", a.investment_component)?;
    Ok(d)
}

fn balances_dict<'py>(py: Python<'py>, b: Balances) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("present_value", b.present_value)?;
    d.set_item("present_value_locked_in", b.present_value_locked_in)?;
    d.set_item("risk_adjustment", b.risk_adjustment)?;
    d.set_item("csm", b.csm)?;
    d.set_item("loss_component", b.loss_component)?;
    d.set_item("liability", b.liability())?;
    d.set_item("accumulated_oci", b.accumulated_oci())?;
    Ok(d)
}

fn components<'py>(py: Python<'py>, rows: &[ComponentRow]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let line: fn(&ComponentRow) -> ComponentLine = |r| r.line;
    d.set_item(
        "line",
        rows.iter().map(|r| line(r).name()).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "present_value",
        rows.iter().map(|r| r.present_value).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "risk_adjustment",
        rows.iter().map(|r| r.risk_adjustment).collect::<Vec<_>>(),
    )?;
    d.set_item("csm", rows.iter().map(|r| r.csm).collect::<Vec<_>>())?;
    Ok(d)
}

fn coverage<'py>(py: Python<'py>, rows: &[CoverageRow]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let line: fn(&CoverageRow) -> CoverageLine = |r| r.line;
    d.set_item(
        "line",
        rows.iter().map(|r| line(r).name()).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "excluding_loss_component",
        rows.iter()
            .map(|r| r.excluding_loss_component)
            .collect::<Vec<_>>(),
    )?;
    d.set_item(
        "loss_component",
        rows.iter().map(|r| r.loss_component).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "incurred_claims",
        rows.iter().map(|r| r.incurred_claims).collect::<Vec<_>>(),
    )?;
    Ok(d)
}

fn group_period_dict<'py>(py: Python<'py>, p: &GroupPeriod) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let r: Revenue = p.revenue;
    d.set_item("period", p.period)?;
    d.set_item("insurance_revenue", p.insurance_revenue())?;
    d.set_item(
        "expected_claims_and_expenses",
        r.expected_claims_and_expenses,
    )?;
    d.set_item("risk_adjustment_released", r.risk_adjustment_released)?;
    d.set_item("csm_released", r.csm_released)?;
    d.set_item("acquisition_recovered", r.acquisition_recovered)?;
    d.set_item("premium_experience", r.premium_experience)?;
    d.set_item("insurance_service_expenses", p.insurance_service_expenses())?;
    d.set_item("incurred_claims", p.incurred_claims)?;
    d.set_item("acquisition_amortisation", p.acquisition_amortisation)?;
    d.set_item("onerous_losses", p.onerous_losses)?;
    d.set_item("loss_reversals", p.loss_reversals)?;
    d.set_item("insurance_service_result", p.insurance_service_result())?;
    d.set_item("finance_expenses", p.finance_expenses)?;
    d.set_item("finance_expenses_in_profit", p.finance_expenses_in_profit())?;
    d.set_item("finance_expenses_in_oci", p.finance_expenses_in_oci)?;
    d.set_item("profit", p.profit())?;
    d.set_item("other_comprehensive_income", p.other_comprehensive_income())?;
    d.set_item("loss_component_share", p.loss_component_share)?;
    d.set_item("by_component", components(py, &p.by_component)?)?;
    d.set_item("by_coverage", coverage(py, &p.by_coverage)?)?;
    d.set_item("closing", balances_dict(py, p.closing)?)?;
    Ok(d)
}

/// A group of insurance contracts under IFRS 17's general measurement
/// model, recognised from its expected cash flows and rolled forward a
/// period at a time (``docs/design/reserving-ifrs17-measurement.md``).
///
/// The group holds its state: :meth:`roll_forward` changes it, and
/// :meth:`to_json` (or ``pickle``) saves it to roll forward next period.
///
/// Parameters
/// ----------
/// expected : dict
///     The expected cash flows by future period, the next first: lists
///     under ``premiums``, ``claims``, ``expenses``, ``acquisition`` and
///     ``investment_component`` (absent ones nil), and optionally
///     ``premium_timing``, ``claim_timing`` and ``acquisition_timing``
///     (``"start"``, ``"mid"`` or ``"end"``; premiums and acquisition at
///     the start and claims at the end by default).
/// risk_adjustment : float
///     The RA at recognition.
/// curve : DiscountCurve
///     The locked-in curve.
/// coverage_units : list of float, optional
///     One per period of coverage; equal by default.
/// discount_coverage_units : bool, default False
/// period_years : float, default 1.0
/// oci : bool, default False
///     Insurance finance income or expense at the locked-in curve in
///     profit or loss, the rest in OCI.
/// accretion_curve : DiscountCurve, optional
///     The curve the CSM accretes at, when it differs.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import DiscountCurve, Ifrs17Group
/// >>> # IFRS 17 Example 1A: premium 900 at once, claims of 200 a year, RA 120.
/// >>> g = Ifrs17Group({"premiums": [900.0, 0.0, 0.0], "claims": [200.0] * 3}, 120.0,
/// ...                 DiscountCurve.flat(0.05))
/// >>> round(g.balances["csm"])
/// 235
/// >>> year = g.roll_forward(g.as_expected(), {"claims": [200.0] * 2}, 80.0)
/// >>> round(year["closing"]["csm"]), round(year["closing"]["liability"])
/// (165, 617)
#[pyclass(name = "Ifrs17Group", module = "prospicio.reserving")]
pub(crate) struct PyIfrs17Group {
    inner: Group,
    last: Option<GroupPeriod>,
}

#[pymethods]
impl PyIfrs17Group {
    #[new]
    #[pyo3(signature = (expected, risk_adjustment, curve, coverage_units = None, discount_coverage_units = false, period_years = 1.0, oci = false, accretion_curve = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        expected: &Bound<'_, PyDict>,
        risk_adjustment: f64,
        curve: &PyDiscountCurve,
        coverage_units: Option<Vec<f64>>,
        discount_coverage_units: bool,
        period_years: f64,
        oci: bool,
        accretion_curve: Option<PyRef<'_, PyDiscountCurve>>,
    ) -> PyResult<Self> {
        let options = GroupOptions {
            coverage_units,
            discount_coverage_units,
            period_years,
            oci,
            accretion_curve: accretion_curve.map(|c| c.inner.clone()),
        };
        Ok(Self {
            inner: Group::recognise(
                self::expected(expected)?,
                risk_adjustment,
                curve.inner.clone(),
                options,
            )
            .map_err(err)?,
            last: None,
        })
    }

    /// The balances after the last period (at recognition before the
    /// first): ``present_value``, ``present_value_locked_in``,
    /// ``risk_adjustment``, ``csm``, ``loss_component``, ``liability`` and
    /// ``accumulated_oci``.
    #[getter]
    fn balances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        balances_dict(py, self.inner.balances())
    }

    /// The periods rolled forward.
    #[getter]
    fn period(&self) -> usize {
        self.inner.period()
    }

    /// The next period's cash flows as expected, a dict for
    /// :meth:`roll_forward`'s ``actual``.
    fn as_expected<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        actual_dict(py, self.inner.as_expected())
    }

    /// Roll the group forward over one period.
    ///
    /// Parameters
    /// ----------
    /// actual : dict
    ///     What happened in the period: ``premiums``,
    ///     ``premiums_for_future_service``, ``claims``, ``expenses``,
    ///     ``acquisition`` and ``investment_component`` (absent ones nil).
    /// expected : dict
    ///     The expected cash flows as at the period end, as for the
    ///     constructor.
    /// risk_adjustment : float
    ///     The RA at the period end.
    /// risk_adjustment_released, acquisition_recovered : float, optional
    ///     By coverage units when not given.
    /// future_coverage_units : list of float, optional
    /// current_curve : DiscountCurve, optional
    ///     The curve at the period end, maturities from then.
    /// financial_revision : dict, optional
    ///     The opening view revised for financial assumptions.
    ///
    /// Returns
    /// -------
    /// dict
    ///     The profit or loss lines, ``by_component`` and ``by_coverage``
    ///     (the two reconciliations, as columns with a ``line`` name), and
    ///     ``closing``.
    #[pyo3(signature = (actual, expected, risk_adjustment, risk_adjustment_released = None, acquisition_recovered = None, future_coverage_units = None, current_curve = None, financial_revision = None))]
    #[allow(clippy::too_many_arguments)]
    fn roll_forward<'py>(
        &mut self,
        py: Python<'py>,
        actual: &Bound<'py, PyDict>,
        expected: &Bound<'py, PyDict>,
        risk_adjustment: f64,
        risk_adjustment_released: Option<f64>,
        acquisition_recovered: Option<f64>,
        future_coverage_units: Option<Vec<f64>>,
        current_curve: Option<PyRef<'py, PyDiscountCurve>>,
        financial_revision: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut inputs = PeriodInputs::new(
            self::actual(actual)?,
            self::expected(expected)?,
            risk_adjustment,
        );
        if let Some(r) = risk_adjustment_released {
            inputs = inputs.with_risk_adjustment_released(r);
        }
        if let Some(a) = acquisition_recovered {
            inputs = inputs.with_acquisition_recovered(a);
        }
        if let Some(u) = future_coverage_units {
            inputs = inputs.with_future_coverage_units(u);
        }
        if let Some(c) = current_curve {
            inputs = inputs.with_current_curve(c.inner.clone());
        }
        if let Some(f) = financial_revision {
            inputs = inputs.with_financial_revision(self::expected(f)?);
        }
        let p = self.inner.roll_forward(&inputs).map_err(err)?;
        let d = group_period_dict(py, &p)?;
        self.last = Some(p);
        Ok(d)
    }

    /// The state as a JSON document (format ``risk_rs.ifrs17_group``) that
    /// :meth:`from_json` reads, to roll forward from where it stopped.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import DiscountCurve, Ifrs17Group
    /// >>> g = Ifrs17Group({"premiums": [900.0, 0.0], "claims": [400.0] * 2}, 30.0,
    /// ...                 DiscountCurve.flat(0.0))
    /// >>> Ifrs17Group.from_json(g.to_json()).balances == g.balances
    /// True
    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    /// The group a :meth:`to_json` document describes.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Group::from_json(text).map_err(err)?,
            last: None,
        })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let from_json = slf.get_type().getattr("from_json")?;
        Ok((from_json, (slf.borrow().inner.to_json(),)))
    }
}

fn paa_balances_dict<'py>(py: Python<'py>, b: PaaBalances) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item(
        "lrc_excluding_loss_component",
        b.lrc_excluding_loss_component,
    )?;
    d.set_item("loss_component", b.loss_component)?;
    d.set_item("lrc", b.lrc())?;
    d.set_item("lic_present_value", b.lic.present_value)?;
    d.set_item("lic_risk_adjustment", b.lic.risk_adjustment)?;
    d.set_item("liability", b.liability())?;
    Ok(d)
}

fn paa_period_dict<'py>(py: Python<'py>, p: &PaaPeriod) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("period", p.period)?;
    d.set_item("insurance_revenue", p.insurance_revenue)?;
    d.set_item("incurred_claims", p.incurred_claims)?;
    d.set_item("past_service", p.past_service)?;
    d.set_item("acquisition_expense", p.acquisition_expense)?;
    d.set_item("onerous_losses", p.onerous_losses)?;
    d.set_item("loss_reversals", p.loss_reversals)?;
    d.set_item("insurance_service_expenses", p.insurance_service_expenses())?;
    d.set_item("insurance_service_result", p.insurance_service_result())?;
    d.set_item("finance_expenses", p.finance_expenses)?;
    d.set_item("profit", p.profit())?;
    d.set_item("by_coverage", coverage(py, &p.by_coverage)?)?;
    d.set_item("closing", paa_balances_dict(py, p.closing)?)?;
    Ok(d)
}

/// A group of insurance contracts under IFRS 17's premium allocation
/// approach: the LRC from premiums less revenue on a pattern, and the LIC
/// at its fulfilment cash flows.
///
/// Parameters
/// ----------
/// expected_premiums : float
///     The premium receipts expected over the coverage.
/// acquisition : float
///     The acquisition cash flows expected.
/// revenue_pattern : list of float
///     The weights of the revenue by period of coverage.
/// expense_acquisition : bool, default False
///     Expense acquisition cash flows when paid instead of amortising them.
/// lic_curve : DiscountCurve, optional
///     The curve that discounts the LIC; undiscounted by default.
/// period_years : float, default 1.0
///
/// Examples
/// --------
/// >>> from prospicio.reserving import PaaGroup
/// >>> g = PaaGroup(1_000.0, 100.0, [1.0, 1.0])
/// >>> year = g.roll_forward(1_000.0, 100.0, 300.0, 200.0, 100.0, 10.0)
/// >>> year["insurance_revenue"], year["insurance_service_expenses"]
/// (500.0, 360.0)
#[pyclass(name = "PaaGroup", module = "prospicio.reserving")]
pub(crate) struct PyPaaGroup {
    inner: PaaGroup,
}

#[pymethods]
impl PyPaaGroup {
    #[new]
    #[pyo3(signature = (expected_premiums, acquisition, revenue_pattern, expense_acquisition = false, lic_curve = None, period_years = 1.0))]
    fn new(
        expected_premiums: f64,
        acquisition: f64,
        revenue_pattern: Vec<f64>,
        expense_acquisition: bool,
        lic_curve: Option<PyRef<'_, PyDiscountCurve>>,
        period_years: f64,
    ) -> PyResult<Self> {
        let options = PaaOptions {
            revenue_pattern,
            expense_acquisition,
            lic_curve: lic_curve.map(|c| c.inner.clone()),
            period_years,
        };
        Ok(Self {
            inner: PaaGroup::recognise(expected_premiums, acquisition, options).map_err(err)?,
        })
    }

    /// The balances after the last period: the LRC excluding and including
    /// the loss component, the LIC's present value and RA, the liability.
    #[getter]
    fn balances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        paa_balances_dict(py, self.inner.balances())
    }

    /// Roll the group forward over one period.
    ///
    /// Parameters
    /// ----------
    /// premiums_received, acquisition_paid, claims_incurred, claims_paid : float
    /// lic_present_value, lic_risk_adjustment : float
    ///     The LIC at the period end.
    /// expected_premiums : float, optional
    ///     The premium receipts expected over the whole coverage, revised.
    /// remaining_fulfilment : float, optional
    ///     The GMM fulfilment cash flows of the remaining coverage, for the
    ///     onerous test.
    ///
    /// Returns
    /// -------
    /// dict
    ///     The profit or loss lines, ``by_coverage`` and ``closing``.
    #[pyo3(signature = (premiums_received, acquisition_paid, claims_incurred, claims_paid, lic_present_value, lic_risk_adjustment, expected_premiums = None, remaining_fulfilment = None))]
    #[allow(clippy::too_many_arguments)]
    fn roll_forward<'py>(
        &mut self,
        py: Python<'py>,
        premiums_received: f64,
        acquisition_paid: f64,
        claims_incurred: f64,
        claims_paid: f64,
        lic_present_value: f64,
        lic_risk_adjustment: f64,
        expected_premiums: Option<f64>,
        remaining_fulfilment: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut inputs = PaaInputs::new(
            premiums_received,
            acquisition_paid,
            claims_incurred,
            claims_paid,
            Lic::new(lic_present_value, lic_risk_adjustment),
        );
        if let Some(p) = expected_premiums {
            inputs = inputs.with_expected_premiums(p);
        }
        if let Some(f) = remaining_fulfilment {
            inputs = inputs.with_remaining_fulfilment(f);
        }
        let p = self.inner.roll_forward(&inputs).map_err(err)?;
        paa_period_dict(py, &p)
    }

    /// The state as a JSON document (format ``risk_rs.ifrs17_paa_group``).
    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    /// The group a :meth:`to_json` document describes.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PaaGroup::from_json(text).map_err(err)?,
        })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let from_json = slf.get_type().getattr("from_json")?;
        Ok((from_json, (slf.borrow().inner.to_json(),)))
    }
}

fn ceded(d: &Bound<'_, PyDict>) -> PyResult<CededCashFlows> {
    check_keys(
        d,
        "the expected ceded cash flows",
        &[
            "premiums",
            "recoveries",
            "premium_timing",
            "recovery_timing",
        ],
    )?;
    let premiums = get_list(d, "premiums")?;
    let recoveries = get_list(d, "recoveries")?;
    let n = premiums
        .iter()
        .chain(&recoveries)
        .map(Vec::len)
        .max()
        .unwrap_or(0);
    let mut c = CededCashFlows::new(
        premiums.unwrap_or_else(|| vec![0.0; n]),
        recoveries.unwrap_or_else(|| vec![0.0; n]),
    );
    let t = |key: &str, default: CashFlowTiming| -> PyResult<CashFlowTiming> {
        match d.get_item(key)? {
            Some(v) => timing(key, &v.extract::<String>()?),
            None => Ok(default),
        }
    };
    let (p, r) = (
        t("premium_timing", c.premium_timing)?,
        t("recovery_timing", c.recovery_timing)?,
    );
    c = c.with_timing(p, r);
    Ok(c)
}

fn ceded_balances_dict<'py>(
    py: Python<'py>,
    b: ReinsuranceBalances,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("present_value", b.present_value)?;
    d.set_item("present_value_locked_in", b.present_value_locked_in)?;
    d.set_item("non_performance", b.non_performance)?;
    d.set_item("risk_adjustment", b.risk_adjustment)?;
    d.set_item("csm", b.csm)?;
    d.set_item("loss_recovery", b.loss_recovery)?;
    d.set_item("asset", b.asset())?;
    Ok(d)
}

fn ceded_period_dict<'py>(py: Python<'py>, p: &ReinsurancePeriod) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let a: Allocation = p.allocation;
    let r: Recovered = p.recovered;
    let lr: LossRecoveryMovement = p.loss_recovery;
    d.set_item("period", p.period)?;
    d.set_item("allocation", a.total())?;
    d.set_item("expected_recoveries", a.expected_recoveries)?;
    d.set_item("risk_adjustment_released", a.risk_adjustment_released)?;
    d.set_item("csm_released", a.csm_released)?;
    d.set_item("premium_experience", a.premium_experience)?;
    d.set_item("recovered", r.total())?;
    d.set_item("claims_recovered", r.claims)?;
    d.set_item("loss_recoveries", r.losses)?;
    d.set_item("loss_recovery_reversals", r.loss_reversals)?;
    d.set_item("non_performance", r.non_performance)?;
    d.set_item("past_events_cost", p.past_events_cost)?;
    d.set_item("net_expense", p.net_expense())?;
    d.set_item("finance_income", p.finance_income)?;
    d.set_item("profit", p.profit())?;
    let m = PyDict::new(py);
    m.set_item("opening", lr.opening)?;
    m.set_item("losses", lr.losses)?;
    m.set_item("reversals", lr.reversals)?;
    m.set_item("finance", lr.finance)?;
    m.set_item("released", lr.released)?;
    m.set_item("closing", lr.closing)?;
    d.set_item("loss_recovery", m)?;
    let rows: &[CededRow] = &p.by_component;
    let c = PyDict::new(py);
    let line: fn(&CededRow) -> CededLine = |r| r.line;
    c.set_item(
        "line",
        rows.iter().map(|r| line(r).name()).collect::<Vec<_>>(),
    )?;
    c.set_item(
        "present_value",
        rows.iter().map(|r| r.present_value).collect::<Vec<_>>(),
    )?;
    c.set_item(
        "non_performance",
        rows.iter().map(|r| r.non_performance).collect::<Vec<_>>(),
    )?;
    c.set_item(
        "risk_adjustment",
        rows.iter().map(|r| r.risk_adjustment).collect::<Vec<_>>(),
    )?;
    c.set_item("csm", rows.iter().map(|r| r.csm).collect::<Vec<_>>())?;
    d.set_item("by_component", c)?;
    d.set_item("closing", ceded_balances_dict(py, p.closing)?)?;
    Ok(d)
}

/// A group of reinsurance contracts held under IFRS 17, measured as an
/// asset: the CSM a net cost or a net gain, the loss-recovery component on
/// onerous underlying groups, and the reinsurer's non-performance.
///
/// Parameters
/// ----------
/// expected : dict
///     The expected ceded cash flows by future period: lists under
///     ``premiums`` (paid to the reinsurer) and ``recoveries``, and
///     optionally ``premium_timing`` and ``recovery_timing``.
/// risk_adjustment : float
///     The RA for the risk transferred.
/// curve : DiscountCurve
///     The locked-in curve.
/// underlying : list of (Ifrs17Group, float), optional
///     The underlying groups and the share of their claims recovered; a
///     group not yet rolled forward that is onerous gives income at
///     recognition.
/// coverage_units : list of float, optional
/// period_years : float, default 1.0
/// retroactive : bool, default False
///     Expense a net cost at recognition (paragraph 65A).
/// default_probability, loss_given_default : float, optional
///     The reinsurer's annual probability of default and the share of a
///     recovery lost on default; no allowance by default.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import DiscountCurve, ReinsuranceHeld
/// >>> held = ReinsuranceHeld({"premiums": [300.0, 0.0, 0.0], "recoveries": [90.0] * 3}, 15.0,
/// ...                        DiscountCurve.flat(0.0))
/// >>> held.balances["csm"]
/// 15.0
/// >>> year = held.roll_forward(held.as_expected(), {"recoveries": [90.0] * 2}, 10.0)
/// >>> year["allocation"], year["net_expense"]
/// (100.0, 10.0)
#[pyclass(name = "ReinsuranceHeld", module = "prospicio.reserving")]
pub(crate) struct PyReinsuranceHeld {
    inner: ReinsuranceHeld,
}

#[pymethods]
impl PyReinsuranceHeld {
    #[new]
    #[pyo3(signature = (expected, risk_adjustment, curve, underlying = None, coverage_units = None, period_years = 1.0, retroactive = false, default_probability = None, loss_given_default = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        expected: &Bound<'_, PyDict>,
        risk_adjustment: f64,
        curve: &PyDiscountCurve,
        underlying: Option<Vec<(PyRef<'_, PyIfrs17Group>, f64)>>,
        coverage_units: Option<Vec<f64>>,
        period_years: f64,
        retroactive: bool,
        default_probability: Option<f64>,
        loss_given_default: Option<f64>,
    ) -> PyResult<Self> {
        let non_performance = match (default_probability, loss_given_default) {
            (None, None) => None,
            (p, lgd) => {
                Some(NonPerformance::new(p.unwrap_or(0.0), lgd.unwrap_or(1.0)).map_err(err)?)
            }
        };
        let options = ReinsuranceOptions {
            coverage_units,
            period_years,
            retroactive,
            non_performance,
        };
        let underlying = underlying.unwrap_or_default();
        let pairs: Vec<(&Group, f64)> = underlying.iter().map(|(g, s)| (&g.inner, *s)).collect();
        Ok(Self {
            inner: ReinsuranceHeld::recognise(
                ceded(expected)?,
                risk_adjustment,
                curve.inner.clone(),
                options,
                &pairs,
            )
            .map_err(err)?,
        })
    }

    /// The balances after the last period: ``present_value``,
    /// ``present_value_locked_in``, ``non_performance``,
    /// ``risk_adjustment``, ``csm``, ``loss_recovery`` and ``asset``.
    #[getter]
    fn balances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        ceded_balances_dict(py, self.inner.balances())
    }

    /// The periods rolled forward.
    #[getter]
    fn period(&self) -> usize {
        self.inner.period()
    }

    /// The next period's ceded cash flows as expected, a dict for
    /// :meth:`roll_forward`'s ``actual``.
    fn as_expected<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let a = self.inner.as_expected();
        let d = PyDict::new(py);
        d.set_item("premiums", a.premiums)?;
        d.set_item("premiums_for_future_service", a.premiums_for_future_service)?;
        d.set_item("recoveries", a.recoveries)?;
        Ok(d)
    }

    /// Roll the group forward over one period.
    ///
    /// Parameters
    /// ----------
    /// actual : dict
    ///     ``premiums``, ``premiums_for_future_service`` and ``recoveries``
    ///     of the period (absent ones nil).
    /// expected : dict
    ///     The expected ceded cash flows as at the period end.
    /// risk_adjustment : float
    ///     The RA at the period end.
    /// underlying : list of Ifrs17Group, optional
    ///     The underlying groups given at recognition, in their order,
    ///     each already rolled forward over this period.
    /// risk_adjustment_released : float, optional
    /// future_coverage_units : list of float, optional
    /// current_curve : DiscountCurve, optional
    /// default_probability, loss_given_default : float, optional
    ///     The reinsurer's risk of non-performance revised at the period end.
    ///
    /// Returns
    /// -------
    /// dict
    ///     The allocation of premiums paid and its parts, the amounts
    ///     recovered and their parts, ``net_expense``, ``finance_income``,
    ///     ``profit``, the ``loss_recovery`` movement, ``by_component`` and
    ///     ``closing``.
    #[pyo3(signature = (actual, expected, risk_adjustment, underlying = None, risk_adjustment_released = None, future_coverage_units = None, current_curve = None, default_probability = None, loss_given_default = None))]
    #[allow(clippy::too_many_arguments)]
    fn roll_forward<'py>(
        &mut self,
        py: Python<'py>,
        actual: &Bound<'py, PyDict>,
        expected: &Bound<'py, PyDict>,
        risk_adjustment: f64,
        underlying: Option<Vec<PyRef<'py, PyIfrs17Group>>>,
        risk_adjustment_released: Option<f64>,
        future_coverage_units: Option<Vec<f64>>,
        current_curve: Option<PyRef<'py, PyDiscountCurve>>,
        default_probability: Option<f64>,
        loss_given_default: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        check_keys(
            actual,
            "the actual ceded cash flows",
            &["premiums", "premiums_for_future_service", "recoveries"],
        )?;
        let a = CededActual {
            premiums: get(actual, "premiums", 0.0)?,
            premiums_for_future_service: get(actual, "premiums_for_future_service", 0.0)?,
            recoveries: get(actual, "recoveries", 0.0)?,
        };
        let mut inputs = ReinsuranceInputs::new(a, ceded(expected)?, risk_adjustment);
        if let Some(r) = risk_adjustment_released {
            inputs = inputs.with_risk_adjustment_released(r);
        }
        if let Some(u) = future_coverage_units {
            inputs = inputs.with_future_coverage_units(u);
        }
        if let Some(c) = current_curve {
            inputs = inputs.with_current_curve(c.inner.clone());
        }
        if default_probability.is_some() || loss_given_default.is_some() {
            inputs = inputs.with_non_performance(
                NonPerformance::new(
                    default_probability.unwrap_or(0.0),
                    loss_given_default.unwrap_or(1.0),
                )
                .map_err(err)?,
            );
        }
        let underlying = underlying.unwrap_or_default();
        let next = self.inner.period() + 1;
        let mut periods = Vec::with_capacity(underlying.len());
        for (k, g) in underlying.iter().enumerate() {
            match &g.last {
                Some(p) if p.period == next => periods.push(p),
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "underlying group {k} has not been rolled forward over period {next}"
                    )));
                }
            }
        }
        let p = self.inner.roll_forward(&inputs, &periods).map_err(err)?;
        ceded_period_dict(py, &p)
    }

    /// The state as a JSON document (format
    /// ``risk_rs.ifrs17_reinsurance_held``).
    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    /// The group a :meth:`to_json` document describes.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: ReinsuranceHeld::from_json(text).map_err(err)?,
        })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let from_json = slf.get_type().getattr("from_json")?;
        Ok((from_json, (slf.borrow().inner.to_json(),)))
    }
}
