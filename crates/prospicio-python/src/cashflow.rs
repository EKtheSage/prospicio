//! `prospicio.pricing` (Aggregate lane): a collateralised deal's monthly
//! cash flows over `prospicio_pricing::cashflow` (`docs/design/time-value.md`,
//! decisions 13 to 20), and the deal valued after binding (decision 34).

use prospicio_pricing::cashflow::{
    ActualState, Adjustment, AdjustmentBase, Base, Buffer, CashflowTerms, Cashflows, Conditioning,
    Dates, FeeBasis, Forward, Funding, FundingRule, FundingSource, Interest, InterestBasis,
    MonthlyPatterns, ObligationReference, Operating, PatternDraws, PerformanceFee, PremiumFlow,
    Requirement, Term, Timing, Trace, Valuation, condition_on_paid as condition, deal_ave as ave,
};
use prospicio_pricing::returns::TieredFee;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::payout::{PyIrrResult, irr_options};
use crate::to_py;

fn base(name: &str) -> PyResult<Base> {
    Ok(match name {
        "unearned-premium" => Base::UnearnedPremium,
        "reserve" => Base::Reserve,
        "outstanding" => Base::Outstanding,
        "written" => Base::Written,
        "earned" => Base::Earned,
        "fixed" => Base::Fixed,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown base {other:?}; use \"unearned-premium\", \"reserve\", \
                 \"outstanding\", \"written\", \"earned\" or \"fixed\""
            )));
        }
    })
}

/// A set of months: ``"monthly"``, ``"quarterly"``, an interval ``n``
/// (every ``n`` months from month 0), a list of months, or a dict
/// ``{"every": n, "first": m}``.
fn dates(spec: &Bound<'_, PyAny>, name: &str) -> PyResult<Dates> {
    if let Ok(s) = spec.extract::<String>() {
        return match s.as_str() {
            "monthly" => Ok(Dates::monthly()),
            "quarterly" => Ok(Dates::quarterly()),
            other => Err(PyValueError::new_err(format!(
                "{name}: unknown dates {other:?}; use \"monthly\", \"quarterly\", an interval, \
                 a list of months or {{\"every\": n, \"first\": m}}"
            ))),
        };
    }
    if let Ok(months) = spec.extract::<usize>() {
        return Ok(Dates::Every { months, first: 0 });
    }
    if let Ok(d) = spec.cast::<PyDict>() {
        let get = |key: &str| -> PyResult<Option<usize>> {
            d.get_item(key)?.map(|v| v.extract::<usize>()).transpose()
        };
        if let Some(bad) = d
            .keys()
            .iter()
            .filter_map(|k| k.extract::<String>().ok())
            .find(|k| k != "every" && k != "first")
        {
            return Err(PyValueError::new_err(format!(
                "{name}: unknown key {bad:?}; use \"every\" and \"first\""
            )));
        }
        let months = get("every")?
            .ok_or_else(|| PyValueError::new_err(format!("{name}: \"every\" is missing")))?;
        return Ok(Dates::Every {
            months,
            first: get("first")?.unwrap_or(0),
        });
    }
    if let Ok(months) = spec.extract::<Vec<usize>>() {
        return Ok(Dates::Months(months));
    }
    Err(PyValueError::new_err(format!(
        "{name}: give \"monthly\", \"quarterly\", an interval, a list of months or \
         {{\"every\": n, \"first\": m}}"
    )))
}

/// When an account releases: the dates above, or ``"at-commutation"``.
fn timing(spec: &Bound<'_, PyAny>, name: &str) -> PyResult<Timing> {
    if let Ok(s) = spec.extract::<String>() {
        match s.as_str() {
            "monthly" => return Ok(Timing::Monthly),
            "at-commutation" => return Ok(Timing::AtCommutation),
            _ => {}
        }
    }
    Ok(Timing::At(dates(spec, name)?))
}

/// What a collateral account must hold at the end of each month: a sum of
/// loads on bases, floored, with an optional switch to a limit.
///
/// With ``W``, ``E`` and ``C`` the premium written, the premium earned and
/// the loss paid to date, ``u`` the ultimate loss ratio and ``P`` the
/// premium, the bases are ``"unearned-premium"`` (``W - E``),
/// ``"reserve"`` (``u E - C``), ``"outstanding"`` (``u P - C``),
/// ``"written"`` (``W``), ``"earned"`` (``E``) and ``"fixed"`` (1, so the
/// load is an amount).
///
/// The reserve on earned premium is small early in the term: a requirement
/// on it alone releases the premium collected ahead of earning to the
/// investor at once, who then pays the claims later, so the flows start
/// positive (``Cashflows.paid_first``), can have several internal rates of
/// return (``irr``'s ``sign_changes``) and the one found is usually
/// meaningless. A collateralised fund usually wants ``"outstanding"``,
/// with ``"unearned-premium"`` beside it.
///
/// Parameters
/// ----------
/// terms : dict of str to float, or list of (str, float)
///     The load on each base.
/// floor : float, default 0
///     The least the requirement can be.
/// limit : (float, float), optional
///     ``(trigger, cap)``: when ``trigger * u >= cap`` the requirement is
///     ``cap * W - C`` in place of the terms.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import CollateralRequirement
/// >>> CollateralRequirement({"unearned-premium": 0.3, "reserve": 1.2}, limit=(1.2, 1.5))
/// CollateralRequirement(terms=[('unearned-premium', 0.3), ('reserve', 1.2)], floor=0, limit=(1.2, 1.5))
#[pyclass(
    name = "CollateralRequirement",
    module = "prospicio.pricing",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyCollateralRequirement {
    inner: Requirement,
    names: Vec<String>,
}

#[pymethods]
impl PyCollateralRequirement {
    #[new]
    #[pyo3(signature = (terms, *, floor = 0.0, limit = None))]
    fn new(terms: &Bound<'_, PyAny>, floor: f64, limit: Option<(f64, f64)>) -> PyResult<Self> {
        let pairs: Vec<(String, f64)> = match terms.cast::<PyDict>() {
            Ok(d) => d
                .iter()
                .map(|(k, v)| Ok((k.extract()?, v.extract()?)))
                .collect::<PyResult<_>>()?,
            Err(_) => terms.extract()?,
        };
        let mut inner = Requirement::new(
            pairs
                .iter()
                .map(|(b, l)| Ok(Term::new(base(b)?, *l)))
                .collect::<PyResult<_>>()?,
        )
        .with_floor(floor);
        if let Some((trigger, cap)) = limit {
            inner = inner.with_limit(trigger, cap);
        }
        Ok(PyCollateralRequirement {
            inner,
            names: pairs.into_iter().map(|(b, _)| b).collect(),
        })
    }

    /// The load on each base, in the order given.
    #[getter]
    fn terms(&self) -> Vec<(String, f64)> {
        self.names
            .iter()
            .cloned()
            .zip(self.inner.terms.iter().map(|t| t.load))
            .collect()
    }

    /// The least the requirement can be.
    #[getter]
    fn floor(&self) -> f64 {
        self.inner.floor
    }

    /// ``(trigger, cap)``, or None.
    #[getter]
    fn limit(&self) -> Option<(f64, f64)> {
        self.inner.limit.map(|l| (l.trigger, l.cap))
    }

    fn __repr__(&self) -> String {
        let terms = self
            .terms()
            .iter()
            .map(|(b, l)| format!("('{b}', {l})"))
            .collect::<Vec<_>>()
            .join(", ");
        let limit = match self.limit() {
            Some((t, c)) => format!("({t}, {c})"),
            None => "None".into(),
        };
        format!(
            "CollateralRequirement(terms=[{terms}], floor={}, limit={limit})",
            self.inner.floor
        )
    }
}

/// A premium adjustment paid into the trust at the end of a month: the
/// rate on written premium less ``u`` (the ultimate loss ratio, or the loss
/// paid to date over the premium written to date), and less the
/// commission rate when asked, times the premium written to date.
///
/// Parameters
/// ----------
/// month : int
///     The month, before commutation.
/// written_rate : float
///     The rate on written premium the loss ratio is set against.
/// less_commission : bool, default False
/// base : {"ultimate", "to-date"}, default "ultimate"
///
/// Examples
/// --------
/// >>> from prospicio.pricing import PremiumAdjustment
/// >>> PremiumAdjustment(18, 0.3, less_commission=True)
/// PremiumAdjustment(month=18, written_rate=0.3, less_commission=True, base='ultimate')
#[pyclass(
    name = "PremiumAdjustment",
    module = "prospicio.pricing",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyPremiumAdjustment {
    inner: Adjustment,
}

#[pymethods]
impl PyPremiumAdjustment {
    #[new]
    #[pyo3(signature = (month, written_rate, *, less_commission = false, base = "ultimate"))]
    fn new(month: usize, written_rate: f64, less_commission: bool, base: &str) -> PyResult<Self> {
        let base = match base {
            "ultimate" => AdjustmentBase::Ultimate,
            "to-date" => AdjustmentBase::ToDate,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown base {other:?}; use \"ultimate\" or \"to-date\""
                )));
            }
        };
        Ok(PyPremiumAdjustment {
            inner: Adjustment {
                month,
                written_rate,
                less_commission,
                base,
            },
        })
    }

    /// The month, before commutation.
    #[getter]
    fn month(&self) -> usize {
        self.inner.month
    }

    /// The rate on written premium the loss ratio is set against.
    #[getter]
    fn written_rate(&self) -> f64 {
        self.inner.written_rate
    }

    /// Whether the commission rate comes off too.
    #[getter]
    fn less_commission(&self) -> bool {
        self.inner.less_commission
    }

    /// ``"ultimate"`` or ``"to-date"``.
    #[getter]
    fn base(&self) -> &'static str {
        match self.inner.base {
            AdjustmentBase::Ultimate => "ultimate",
            AdjustmentBase::ToDate => "to-date",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PremiumAdjustment(month={}, written_rate={}, less_commission={}, base='{}')",
            self.inner.month,
            self.inner.written_rate,
            if self.inner.less_commission {
                "True"
            } else {
                "False"
            },
            self.base()
        )
    }
}

/// A collateralised deal's monthly cash flows: the premium, the monthly
/// patterns, the collateral requirement and one rule for each account.
///
/// Month ``t`` runs from 0 to the commutation month ``T - 1``. Each month
/// the trust takes the capital called, the premium collected and any
/// adjustment, pays the month's claims, earns interest and releases what
/// it holds above the requirement to the operating account, which pays
/// the expenses and releases to the investor what it holds above its
/// buffer. At commutation the trust pays the commutation requirement to
/// the cedant, the operating account takes the rest (or makes good a
/// shortfall), pays the fee and distributes what is left.
///
/// Parameters
/// ----------
/// premium : float
/// commutation_month : int
///     ``T``: the deal commutes at the end of month ``T - 1``.
/// written, earned, paid : list of float
///     Cumulative monthly shares of the premium written and earned and the
///     ultimate loss paid, all of one length, at least ``T`` plus the
///     buffer's months.
/// requirement : CollateralRequirement
/// commutation_requirement : CollateralRequirement, optional
///     What the trust pays at commutation; by default ``requirement``.
/// collected_rate : float, default 1
///     The share of written premium paid into the trust.
/// deduct_commission : bool, default False
///     Take each simulation's commission rate (``run``'s ``commission``)
///     off ``collected_rate``. Without it the commission does not reduce
///     the premium paid in and is read only by adjustments with
///     ``less_commission`` and by the fee: a flat commission ``c`` is then
///     ``collected_rate = 1 - c``, and a sliding scale needs this option.
/// lag : int, default 0
///     The months between writing and collection.
/// adjustments : list of PremiumAdjustment, optional
/// funding_dates : default "monthly"
///     The months capital may be called: ``"monthly"``, ``"quarterly"``,
///     an interval, a list of months or ``{"every": n, "first": m}``.
/// funding_rule : {"shortfall", "obligation-increase"}, default "shortfall"
///     Call what the trust lacks, or the rise in the obligation
///     ``S_t - premium_offset * W_{t - lag}`` over ``reference``.
/// premium_offset : float, default 1
/// reference : {"previous-date", "running-max"}, default "previous-date"
/// funding_source : {"investor", "operating-first"}, default "investor"
/// collateral_release : default "monthly"
///     When the trust releases its excess: the dates above or
///     ``"at-commutation"``.
/// annual_yield : float, default 0
///     The trust's yield, compounded monthly.
/// interest_basis : {"average", "opening", "closing"}, default "average"
/// opening_fixed, opening_share : float, default 0
///     The operating account's opening capital, ``opening_fixed +
///     opening_share * premium``.
/// expense_rate : float, default 0
///     The annual expense rate on the requirement, a twelfth a month.
/// buffer_months : int, default 0
///     The operating account keeps the requirement's rise over the next
///     ``buffer_months`` months, with the claims paid (``buffer_paid``)
///     and the expenses (``buffer_expenses``) over them when asked.
/// buffer_paid, buffer_expenses : bool, default False
/// equity_release : default "monthly"
///     When the operating account releases its excess to the investor.
/// fee : (list of float, list of float), optional
///     A performance fee's ``(breakpoints, slopes)``, as ``tiered_fee``.
/// fee_basis : {"profit", "profit-over-premium", "profit-over-capital"}, default "profit"
///
/// Examples
/// --------
/// >>> from prospicio.pricing import CashflowTerms, CollateralRequirement
/// >>> terms = CashflowTerms(
/// ...     100.0, 3, [1.0] * 3, [1 / 3, 2 / 3, 1.0], [1 / 3, 2 / 3, 1.0],
/// ...     CollateralRequirement({"reserve": 1.0}),
/// ...     commutation_requirement=CollateralRequirement({"outstanding": 1.0}),
/// ... )
/// >>> out = terms.run([0.6, 1.5])
/// >>> [[round(x, 9) for x in row] for row in out.flows]
/// [[80.0, -20.0, -20.0], [50.0, -50.0, -50.0]]
/// >>> [round(x, 9) for x in out.funded]
/// [40.0, 100.0]
#[pyclass(name = "CashflowTerms", module = "prospicio.pricing", frozen)]
pub(crate) struct PyCashflowTerms {
    pub(crate) inner: CashflowTerms,
}

#[pymethods]
impl PyCashflowTerms {
    #[new]
    #[pyo3(signature = (
        premium, commutation_month, written, earned, paid, requirement, *,
        commutation_requirement = None, collected_rate = 1.0, deduct_commission = false, lag = 0,
        adjustments = None,
        funding_dates = None, funding_rule = "shortfall", premium_offset = 1.0,
        reference = "previous-date", funding_source = "investor", collateral_release = None,
        annual_yield = 0.0, interest_basis = "average", opening_fixed = 0.0,
        opening_share = 0.0, expense_rate = 0.0, buffer_months = 0, buffer_paid = false,
        buffer_expenses = false, equity_release = None, fee = None, fee_basis = "profit"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        premium: f64,
        commutation_month: usize,
        written: Vec<f64>,
        earned: Vec<f64>,
        paid: Vec<f64>,
        requirement: PyCollateralRequirement,
        commutation_requirement: Option<PyCollateralRequirement>,
        collected_rate: f64,
        deduct_commission: bool,
        lag: usize,
        adjustments: Option<Vec<PyPremiumAdjustment>>,
        funding_dates: Option<&Bound<'_, PyAny>>,
        funding_rule: &str,
        premium_offset: f64,
        reference: &str,
        funding_source: &str,
        collateral_release: Option<&Bound<'_, PyAny>>,
        annual_yield: f64,
        interest_basis: &str,
        opening_fixed: f64,
        opening_share: f64,
        expense_rate: f64,
        buffer_months: usize,
        buffer_paid: bool,
        buffer_expenses: bool,
        equity_release: Option<&Bound<'_, PyAny>>,
        fee: Option<(Vec<f64>, Vec<f64>)>,
        fee_basis: &str,
    ) -> PyResult<Self> {
        let patterns = MonthlyPatterns::new(written, earned, paid).map_err(to_py)?;
        let mut t = CashflowTerms::new(premium, commutation_month, patterns, requirement.inner)
            .map_err(to_py)?;
        if let Some(c) = commutation_requirement {
            t.commutation_requirement = c.inner;
        }
        t.premium_flow = PremiumFlow {
            collected_rate,
            deduct_commission,
            lag,
            adjustments: adjustments
                .unwrap_or_default()
                .into_iter()
                .map(|a| a.inner)
                .collect(),
        };
        let rule = match funding_rule {
            "shortfall" => FundingRule::Shortfall,
            "obligation-increase" => FundingRule::ObligationIncrease {
                premium_offset,
                reference: match reference {
                    "previous-date" => ObligationReference::PreviousDate,
                    "running-max" => ObligationReference::RunningMax,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unknown reference {other:?}; use \"previous-date\" or \
                             \"running-max\""
                        )));
                    }
                },
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown funding rule {other:?}; use \"shortfall\" or \
                     \"obligation-increase\""
                )));
            }
        };
        let source = match funding_source {
            "investor" => FundingSource::Investor,
            "operating-first" => FundingSource::OperatingFirst,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown funding source {other:?}; use \"investor\" or \"operating-first\""
                )));
            }
        };
        t.funding = Funding {
            dates: match funding_dates {
                Some(d) => dates(d, "funding_dates")?,
                None => Dates::monthly(),
            },
            rule,
            source,
        };
        if let Some(r) = collateral_release {
            t.collateral_release = timing(r, "collateral_release")?;
        }
        t.interest = Interest {
            annual_yield,
            basis: match interest_basis {
                "average" => InterestBasis::Average,
                "opening" => InterestBasis::Opening,
                "closing" => InterestBasis::Closing,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown interest basis {other:?}; use \"average\", \"opening\" or \
                         \"closing\""
                    )));
                }
            },
        };
        let fee = fee
            .map(|(breakpoints, slopes)| -> PyResult<PerformanceFee> {
                Ok(PerformanceFee {
                    tiers: TieredFee::new(breakpoints, slopes).map_err(to_py)?,
                    basis: match fee_basis {
                        "profit" => FeeBasis::Profit,
                        "profit-over-premium" => FeeBasis::ProfitOverPremium,
                        "profit-over-capital" => FeeBasis::ProfitOverCapital,
                        other => {
                            return Err(PyValueError::new_err(format!(
                                "unknown fee basis {other:?}; use \"profit\", \
                                 \"profit-over-premium\" or \"profit-over-capital\""
                            )));
                        }
                    },
                })
            })
            .transpose()?;
        t.operating = Operating {
            opening_fixed,
            opening_share,
            expense_rate,
            buffer: Buffer {
                months: buffer_months,
                paid: buffer_paid,
                expenses: buffer_expenses,
            },
            release: match equity_release {
                Some(r) => timing(r, "equity_release")?,
                None => Timing::Monthly,
            },
            fee,
        };
        // `run` checks the terms again; checking here reports a bad rule
        // when the terms are built.
        t.trace(0.0, 0.0).map_err(to_py)?;
        Ok(PyCashflowTerms { inner: t })
    }

    /// The premium.
    #[getter]
    fn premium(&self) -> f64 {
        self.inner.premium
    }

    /// The month count ``T``: the deal commutes at the end of month ``T - 1``.
    #[getter]
    fn commutation_month(&self) -> usize {
        self.inner.commutation_month
    }

    /// Each simulation's cash flows to the investor, from its ultimate
    /// loss ratio and commission rate (0 when not given).
    ///
    /// With ``paid_patterns`` each simulation pays its ultimate loss by its
    /// own monthly cumulative paid pattern (#600), the written and earned
    /// patterns staying the terms': one pattern a simulation, or with
    /// ``pattern_index`` a few patterns and each simulation's index into
    /// them. Every pattern has the terms' length, does not decrease and
    /// ends at the terms' paid share.
    ///
    /// Parameters
    /// ----------
    /// u : list of float
    /// commission : list of float, optional
    /// paid_patterns : list of list of float, optional
    /// pattern_index : list of int, optional
    ///
    /// Returns
    /// -------
    /// Cashflows
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.pricing import CashflowTerms, CollateralRequirement
    /// >>> ramp = lambda m: [min((t + 1) / m, 1.0) for t in range(48)]
    /// >>> req = CollateralRequirement({"outstanding": 1.0})
    /// >>> terms = CashflowTerms(100.0, 36, ramp(12), ramp(24), ramp(36), req)
    /// >>> # The same year paid fast and paid as the terms say.
    /// >>> cf = terms.run([0.8, 0.8], paid_patterns=[ramp(12), ramp(36)], pattern_index=[0, 1])
    /// >>> cf.flows[0] == cf.flows[1]
    /// False
    /// >>> cf.flows[1] == terms.run([0.8]).flows[0]
    /// True
    #[pyo3(signature = (u, commission = None, *, paid_patterns = None, pattern_index = None))]
    fn run(
        &self,
        py: Python<'_>,
        u: Vec<f64>,
        commission: Option<Vec<f64>>,
        paid_patterns: Option<Vec<Vec<f64>>>,
        pattern_index: Option<Vec<usize>>,
    ) -> PyResult<PyCashflows> {
        let c = commission.unwrap_or_else(|| vec![0.0; u.len()]);
        let draws = match (paid_patterns, pattern_index) {
            (None, None) => None,
            (Some(p), None) => Some(PatternDraws::new(p).map_err(to_py)?),
            (Some(p), Some(i)) => Some(PatternDraws::indexed(p, i).map_err(to_py)?),
            (None, Some(_)) => {
                return Err(PyValueError::new_err("pattern_index needs paid_patterns"));
            }
        };
        let terms = &self.inner;
        let inner = py
            .detach(|| match &draws {
                None => terms.run(&u, &c),
                Some(d) => terms.run_with_patterns(&u, &c, d),
            })
            .map_err(to_py)?;
        Ok(PyCashflows { inner })
    }

    /// ``run`` with each simulation's requirement given month by month,
    /// one row a simulation of ``commutation_month + buffer_months``
    /// values, in place of ``requirement``; the commutation requirement is
    /// still read from its terms.
    ///
    /// Parameters
    /// ----------
    /// u : list of float
    /// requirement : list of list of float
    /// commission : list of float, optional
    ///
    /// Returns
    /// -------
    /// Cashflows
    #[pyo3(signature = (u, requirement, commission = None))]
    fn run_with_requirement(
        &self,
        py: Python<'_>,
        u: Vec<f64>,
        requirement: Vec<Vec<f64>>,
        commission: Option<Vec<f64>>,
    ) -> PyResult<PyCashflows> {
        let width = self.inner.commutation_month + self.inner.operating.buffer.months;
        if let Some((k, row)) = requirement
            .iter()
            .enumerate()
            .find(|(_, r)| r.len() != width)
        {
            return Err(PyValueError::new_err(format!(
                "requirement row {k} has {} months; it needs {width}",
                row.len()
            )));
        }
        let c = commission.unwrap_or_else(|| vec![0.0; u.len()]);
        let rows = requirement.concat();
        let inner = py
            .detach(|| self.inner.run_with_requirement(&u, &c, &rows))
            .map_err(to_py)?;
        Ok(PyCashflows { inner })
    }

    /// Every account line of one simulation, month by month: a dict of
    /// columns, each a list of ``commutation_month`` values, with the
    /// scalars ``commutation_paid``, ``fee`` and ``conservation_error``
    /// (the cash created or lost over the deal, 0 up to rounding).
    ///
    /// Parameters
    /// ----------
    /// u : float
    /// commission : float, default 0
    /// paid_pattern : list of float, optional
    ///     The simulation's own paid pattern, as one of ``run``'s
    ///     ``paid_patterns``.
    ///
    /// Returns
    /// -------
    /// dict
    #[pyo3(signature = (u, commission = 0.0, *, paid_pattern = None))]
    fn trace<'py>(
        &self,
        py: Python<'py>,
        u: f64,
        commission: f64,
        paid_pattern: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let t = match paid_pattern {
            None => self.inner.trace(u, commission),
            Some(p) => self.inner.trace_with_pattern(u, commission, &p),
        }
        .map_err(to_py)?;
        trace_dict(py, &t, 0)
    }

    /// The deal valued at the start of ``month`` from its actual accounts
    /// to the end of the month before (#603): ``Valuation.run`` simulates
    /// the months from there to commutation, each simulation conditioned
    /// on the actual paid. Valuation is at a month end.
    ///
    /// ``premium_collected`` is what the trust has received; premium the
    /// schedule would have collected by then and the trust has not is
    /// collected in ``month``. ``flows``, the investor's actual flows in
    /// months ``0`` to ``month - 1``, give the return since inception.
    /// Under ``funding_rule="obligation-increase"`` the funding reference
    /// (the trace's ``funding_reference``) is required after month 0. A
    /// valuation after a premium adjustment month is refused.
    ///
    /// Parameters
    /// ----------
    /// month : int
    /// premium_collected, paid, collateral, funded, released, operating : float
    /// flows : list of float, optional
    /// funding_reference : float, optional
    ///
    /// Returns
    /// -------
    /// Valuation
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.pricing import CashflowTerms, CollateralRequirement
    /// >>> ramp = lambda m: [min((t + 1) / m, 1.0) for t in range(48)]
    /// >>> req = CollateralRequirement({"outstanding": 1.0})
    /// >>> terms = CashflowTerms(100.0, 36, ramp(12), ramp(24), ramp(36), req)
    /// >>> # A path at a loss ratio of 0.8, valued after 12 months.
    /// >>> val = terms.valuation(12, **terms.actual_state(0.8, 12))
    /// >>> f = val.run([0.5, 0.8, 1.2])
    /// >>> [round(x, 9) for x in f.since_inception_rows()[1]] == [
    /// ...     round(x, 9) for x in terms.run([0.8]).flows[0]]
    /// True
    #[pyo3(signature = (
        month, *, premium_collected = 0.0, paid = 0.0, collateral = 0.0, funded = 0.0,
        released = 0.0, operating = 0.0, flows = None, funding_reference = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn valuation(
        &self,
        month: usize,
        premium_collected: f64,
        paid: f64,
        collateral: f64,
        funded: f64,
        released: f64,
        operating: f64,
        flows: Option<Vec<f64>>,
        funding_reference: Option<f64>,
    ) -> PyResult<PyValuation> {
        let state = ActualState {
            month,
            premium_collected,
            paid,
            collateral,
            funded,
            released,
            operating,
            flows,
            funding_reference,
        };
        let inner = self.inner.valuation(state).map_err(to_py)?;
        Ok(PyValuation { inner })
    }

    /// The actual state of one simulated path at the start of ``month``,
    /// as keywords for ``valuation``: to value a path of a
    /// simulation as if it had happened.
    ///
    /// Parameters
    /// ----------
    /// u : float
    /// month : int
    /// commission : float, default 0
    /// paid_pattern : list of float, optional
    ///
    /// Returns
    /// -------
    /// dict
    #[pyo3(signature = (u, month, commission = 0.0, *, paid_pattern = None))]
    fn actual_state<'py>(
        &self,
        py: Python<'py>,
        u: f64,
        month: usize,
        commission: f64,
        paid_pattern: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let trace = match paid_pattern {
            None => self.inner.trace(u, commission),
            Some(p) => self.inner.trace_with_pattern(u, commission, &p),
        }
        .map_err(to_py)?;
        let s = ActualState::from_trace(&trace, month).map_err(to_py)?;
        let d = PyDict::new(py);
        d.set_item("premium_collected", s.premium_collected)?;
        d.set_item("paid", s.paid)?;
        d.set_item("collateral", s.collateral)?;
        d.set_item("funded", s.funded)?;
        d.set_item("released", s.released)?;
        d.set_item("operating", s.operating)?;
        d.set_item("flows", s.flows)?;
        d.set_item("funding_reference", s.funding_reference)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "CashflowTerms(premium={}, commutation_month={})",
            self.inner.premium, self.inner.commutation_month
        )
    }
}

fn conditioning(name: &str) -> PyResult<Conditioning> {
    match name {
        "bornhuetter-ferguson" => Ok(Conditioning::BornhuetterFerguson),
        "truncation" => Ok(Conditioning::Truncation),
        other => Err(PyValueError::new_err(format!(
            "unknown conditioning {other:?}; use \"bornhuetter-ferguson\" or \"truncation\""
        ))),
    }
}

fn rows(flat: Vec<f64>, width: usize) -> Vec<Vec<f64>> {
    flat.chunks(width.max(1)).map(<[f64]>::to_vec).collect()
}

/// A bound deal's terms with its actual state at a valuation month, from
/// ``CashflowTerms.valuation`` (#603).
///
/// ``conditioning`` says how each simulation's ultimate loss ratio is
/// conditioned on the actual paid ``A`` when the pattern is ``S`` paid:
/// ``"bornhuetter-ferguson"`` (the default) keeps every simulation at
/// ``A + u P (1 - S)``; ``"truncation"`` keeps those with ``u P >= A``,
/// unchanged.
#[pyclass(name = "Valuation", module = "prospicio.pricing", frozen)]
pub(crate) struct PyValuation {
    inner: Valuation,
}

#[pymethods]
impl PyValuation {
    /// The valuation month, the first month the forward run simulates.
    #[getter]
    fn month(&self) -> usize {
        self.inner.month()
    }

    /// The share of the paid pattern reached before the valuation month.
    #[getter]
    fn share_paid(&self) -> f64 {
        self.inner.share_paid()
    }

    /// The trust's and operating account's balances, the investment the
    /// forward return starts from.
    #[getter]
    fn nav(&self) -> f64 {
        self.inner.nav()
    }

    /// Each simulation's cash flows from the valuation month to
    /// commutation, conditioned on the actual paid.
    ///
    /// Parameters
    /// ----------
    /// u : list of float
    /// commission : list of float, optional
    /// conditioning : str, default "bornhuetter-ferguson"
    ///
    /// Returns
    /// -------
    /// Forward
    #[pyo3(signature = (u, commission = None, *, conditioning = "bornhuetter-ferguson"))]
    fn run(
        &self,
        py: Python<'_>,
        u: Vec<f64>,
        commission: Option<Vec<f64>>,
        conditioning: &str,
    ) -> PyResult<PyForward> {
        let how = self::conditioning(conditioning)?;
        let c = commission.unwrap_or_else(|| vec![0.0; u.len()]);
        let inner = py.detach(|| self.inner.run(&u, &c, how)).map_err(to_py)?;
        Ok(PyForward { inner })
    }

    /// Every account line of one simulation from the valuation month, as
    /// ``CashflowTerms.trace``, with ``opening_balance`` the actual
    /// balances; ``conservation_error`` counts them.
    ///
    /// Parameters
    /// ----------
    /// u : float
    /// commission : float, default 0
    /// conditioning : str, default "bornhuetter-ferguson"
    ///
    /// Returns
    /// -------
    /// dict
    #[pyo3(signature = (u, commission = 0.0, *, conditioning = "bornhuetter-ferguson"))]
    fn trace<'py>(
        &self,
        py: Python<'py>,
        u: f64,
        commission: f64,
        conditioning: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let how = self::conditioning(conditioning)?;
        let t = self.inner.trace(u, commission, how).map_err(to_py)?;
        trace_dict(py, &t, self.inner.month())
    }

    fn __repr__(&self) -> String {
        format!(
            "Valuation(month={}, nav={})",
            self.inner.month(),
            self.inner.nav()
        )
    }
}

/// The forward cash flows of a ``Valuation``: one row a simulation kept,
/// months from the valuation month to commutation, with the returns from
/// the valuation month (the net asset value as the investment) and since
/// inception (the actual flows to date first).
#[pyclass(name = "Forward", module = "prospicio.pricing", frozen)]
pub(crate) struct PyForward {
    inner: Forward,
}

#[pymethods]
impl PyForward {
    /// The valuation month.
    #[getter]
    fn month(&self) -> usize {
        self.inner.month()
    }

    /// The simulations kept, by their index in the loss ratios given.
    #[getter]
    fn kept(&self) -> Vec<usize> {
        self.inner.kept().to_vec()
    }

    /// Each kept simulation's ultimate loss ratio given the paid to date.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        self.inner.ultimate().to_vec()
    }

    /// The flows in months from the valuation month, with totals since
    /// inception (``funded``, ``released``, ``fee``).
    #[getter]
    fn flows(&self) -> PyCashflows {
        PyCashflows {
            inner: self.inner.flows().clone(),
        }
    }

    /// The net asset value at the valuation.
    #[getter]
    fn nav(&self) -> f64 {
        self.inner.nav()
    }

    /// The forward flows with the net asset value paid in at their first
    /// month: one list a kept simulation.
    fn forward_rows(&self) -> Vec<Vec<f64>> {
        rows(self.inner.forward_rows(), self.inner.flows().months())
    }

    /// The actual flows to date followed by each kept simulation's forward
    /// flows; needs the actual ``flows`` in the state.
    fn since_inception_rows(&self) -> PyResult<Vec<Vec<f64>>> {
        let width = self.inner.month() + self.inner.flows().months();
        Ok(rows(
            self.inner.since_inception_rows().map_err(to_py)?,
            width,
        ))
    }

    /// Each kept simulation's net present value from the valuation month.
    fn forward_npv(&self, rate: f64) -> PyResult<Vec<f64>> {
        self.inner.forward_npv(rate).map_err(to_py)
    }

    /// Each kept simulation's net present value since inception.
    fn since_inception_npv(&self, rate: f64) -> PyResult<Vec<f64>> {
        self.inner.since_inception_npv(rate).map_err(to_py)
    }

    /// Each kept simulation's multiple from the valuation month,
    /// ``1 + net / (nav + funded after it)``.
    fn forward_moic(&self) -> PyResult<Vec<f64>> {
        self.inner.forward_moic().map_err(to_py)
    }

    /// Each kept simulation's multiple on invested capital since inception.
    fn since_inception_moic(&self) -> PyResult<Vec<f64>> {
        self.inner.since_inception_moic().map_err(to_py)
    }

    /// Each kept simulation's internal rate of return from the valuation
    /// month, as ``Cashflows.irr``.
    #[pyo3(signature = (*, lower = None, upper = None, points = 400, bisections = 8, newton_steps = 3, extend = false))]
    #[allow(clippy::too_many_arguments)]
    fn forward_irr(
        &self,
        py: Python<'_>,
        lower: Option<f64>,
        upper: Option<f64>,
        points: usize,
        bisections: usize,
        newton_steps: usize,
        extend: bool,
    ) -> PyResult<PyIrrResult> {
        let o = irr_options(lower, upper, points, bisections, newton_steps, extend);
        let inner = py.detach(|| self.inner.forward_irr(&o)).map_err(to_py)?;
        Ok(PyIrrResult { inner })
    }

    /// Each kept simulation's internal rate of return since inception.
    #[pyo3(signature = (*, lower = None, upper = None, points = 400, bisections = 8, newton_steps = 3, extend = false))]
    #[allow(clippy::too_many_arguments)]
    fn since_inception_irr(
        &self,
        py: Python<'_>,
        lower: Option<f64>,
        upper: Option<f64>,
        points: usize,
        bisections: usize,
        newton_steps: usize,
        extend: bool,
    ) -> PyResult<PyIrrResult> {
        let o = irr_options(lower, upper, points, bisections, newton_steps, extend);
        let inner = py
            .detach(|| self.inner.since_inception_irr(&o))
            .map_err(to_py)?;
        Ok(PyIrrResult { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.kept().len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Forward(month={}, n={}, months={})",
            self.inner.month(),
            self.inner.kept().len(),
            self.inner.flows().months()
        )
    }
}

/// Each simulation's ultimate loss ratio conditioned on the loss ``paid``
/// to date when the pattern is ``share_paid`` paid (#603): a dict with
/// ``kept``, the simulations kept, and ``ultimate``, their ultimate loss
/// ratios. ``"bornhuetter-ferguson"`` keeps every simulation at
/// ``paid / premium + u (1 - share_paid)``; ``"truncation"`` keeps those
/// with ``u premium >= paid``, unchanged.
///
/// Parameters
/// ----------
/// u : list of float
/// premium, paid, share_paid : float
/// conditioning : str, default "bornhuetter-ferguson"
///
/// Returns
/// -------
/// dict
///
/// Examples
/// --------
/// >>> from prospicio.pricing import condition_on_paid
/// >>> c = condition_on_paid([0.2, 0.8], 100.0, 30.0, 0.4, conditioning="truncation")
/// >>> c["kept"], c["ultimate"]
/// ([1], [0.8])
#[pyfunction]
#[pyo3(signature = (u, premium, paid, share_paid, *, conditioning = "bornhuetter-ferguson"))]
pub(crate) fn condition_on_paid<'py>(
    py: Python<'py>,
    u: Vec<f64>,
    premium: f64,
    paid: f64,
    share_paid: f64,
    conditioning: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let how = self::conditioning(conditioning)?;
    let c = condition(&u, premium, paid, share_paid, how).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("kept", c.kept)?;
    d.set_item("ultimate", c.ultimate)?;
    Ok(d)
}

/// The actual cumulative paid at the end of each month ``0, 1, ...``
/// against what the priced simulations expected (#603): each simulation's
/// ``u P S(t)`` with ``S`` the terms' paid pattern, or the simulation's own
/// from ``paid_patterns`` (and ``pattern_index``). A dict of columns
/// ``month``, ``actual``, ``expected`` (the mean), ``ratio``,
/// ``percentile`` (the actual's percentile among the simulations, ties
/// counted half), with ``levels`` and ``quantiles`` (one list a month, the
/// expected paid at each level, numpy's linear quantile). A reported loss
/// works the same way with the reported pattern as ``paid_patterns``.
///
/// Parameters
/// ----------
/// actual : list of float
/// terms : CashflowTerms
/// u : list of float
/// paid_patterns : list of list of float, optional
/// pattern_index : list of int, optional
/// levels : list of float, default [0.05, 0.5, 0.95]
///
/// Returns
/// -------
/// dict
///
/// Examples
/// --------
/// >>> from prospicio.pricing import CashflowTerms, CollateralRequirement, deal_ave
/// >>> req = CollateralRequirement({"outstanding": 1.0})
/// >>> terms = CashflowTerms(100.0, 4, [1.0] * 4, [1.0] * 4, [0.25, 0.5, 0.75, 1.0], req)
/// >>> r = deal_ave([15.0, 45.0], terms, [0.4, 0.6, 0.8])
/// >>> r["expected"], r["percentile"]
/// ([15.0, 30.0], [0.5, 1.0])
#[pyfunction]
#[pyo3(signature = (actual, terms, u, *, paid_patterns = None, pattern_index = None, levels = None))]
pub(crate) fn deal_ave<'py>(
    py: Python<'py>,
    actual: Vec<f64>,
    terms: PyRef<'py, PyCashflowTerms>,
    u: Vec<f64>,
    paid_patterns: Option<Vec<Vec<f64>>>,
    pattern_index: Option<Vec<usize>>,
    levels: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let draws = match (paid_patterns, pattern_index) {
        (None, None) => None,
        (Some(p), None) => Some(PatternDraws::new(p).map_err(to_py)?),
        (Some(p), Some(i)) => Some(PatternDraws::indexed(p, i).map_err(to_py)?),
        (None, Some(_)) => {
            return Err(PyValueError::new_err("pattern_index needs paid_patterns"));
        }
    };
    let levels = levels.unwrap_or_else(|| vec![0.05, 0.5, 0.95]);
    let r = ave(&actual, &terms.inner, &u, draws.as_ref(), &levels).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("month", r.month)?;
    d.set_item("actual", r.actual)?;
    d.set_item("expected", r.expected)?;
    d.set_item("ratio", r.ratio)?;
    d.set_item("percentile", r.percentile)?;
    d.set_item("quantiles", rows(r.quantiles, r.levels.len()))?;
    d.set_item("levels", r.levels)?;
    Ok(d)
}

fn trace_dict<'py>(py: Python<'py>, t: &Trace, first: usize) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item(
        "month",
        (first..first + t.requirement.len()).collect::<Vec<_>>(),
    )?;
    let columns: [(&str, &Vec<f64>); 14] = [
        ("requirement", &t.requirement),
        ("premium", &t.premium),
        ("adjustment", &t.adjustment),
        ("paid", &t.paid),
        ("funding_investor", &t.funding_investor),
        ("funding_operating", &t.funding_operating),
        ("interest", &t.interest),
        ("collateral_release", &t.collateral_release),
        ("collateral_balance", &t.collateral_balance),
        ("expenses", &t.expenses),
        ("equity_release", &t.equity_release),
        ("operating_balance", &t.operating_balance),
        ("investor", &t.investor),
        ("funding_reference", &t.funding_reference),
    ];
    for (k, v) in columns {
        d.set_item(k, v)?;
    }
    d.set_item("opening_balance", t.opening_balance)?;
    d.set_item("commutation_paid", t.commutation_paid)?;
    d.set_item("fee", t.fee)?;
    d.set_item("conservation_error", t.conservation_error())?;
    Ok(d)
}

/// Simulated monthly cash flows to the investor, from
/// ``CashflowTerms.run``: positive when paid to the investor.
#[pyclass(name = "Cashflows", module = "prospicio.pricing", frozen)]
pub(crate) struct PyCashflows {
    pub(crate) inner: Cashflows,
}

#[pymethods]
impl PyCashflows {
    /// The months a simulation runs, the commutation month.
    #[getter]
    fn months(&self) -> usize {
        self.inner.months()
    }

    /// Each simulation's flows, one list of ``months`` values a simulation.
    #[getter]
    fn flows(&self) -> Vec<Vec<f64>> {
        let m = self.inner.months().max(1);
        self.inner.flows().chunks(m).map(<[f64]>::to_vec).collect()
    }

    /// The months 0 to ``months - 1``, the times of ``flows`` for ``npv``
    /// and ``irr`` with ``unit="months"``.
    #[getter]
    fn month_offsets(&self) -> Vec<f64> {
        self.inner.month_offsets()
    }

    /// The capital each simulation's investor paid into the trust.
    #[getter]
    fn funded(&self) -> Vec<f64> {
        self.inner.funded().to_vec()
    }

    /// The opening capital each investor paid in.
    #[getter]
    fn opening(&self) -> Vec<f64> {
        self.inner.opening().to_vec()
    }

    /// The equity released to each investor before commutation.
    #[getter]
    fn released(&self) -> Vec<f64> {
        self.inner.released().to_vec()
    }

    /// What each investor received at commutation, after the fee.
    #[getter]
    fn distributed(&self) -> Vec<f64> {
        self.inner.distributed().to_vec()
    }

    /// The performance fee in each simulation.
    #[getter]
    fn fee(&self) -> Vec<f64> {
        self.inner.fee().to_vec()
    }

    /// What each trust paid the cedant at commutation.
    #[getter]
    fn commutation(&self) -> Vec<f64> {
        self.inner.commutation().to_vec()
    }

    /// Whether each simulation's first nonzero investor flow is positive:
    /// the investor is paid before putting money in, so its flows can have
    /// several internal rates of return (``irr``'s ``sign_changes``) and the
    /// one found is usually meaningless. A requirement on the reserve alone
    /// does this; a collateralised fund usually wants ``"outstanding"``.
    #[getter]
    fn paid_first(&self) -> Vec<bool> {
        self.inner.paid_first()
    }

    /// Each simulation's net present value at an annual rate.
    fn npv(&self, rate: f64) -> PyResult<Vec<f64>> {
        self.inner.npv(rate).map_err(to_py)
    }

    /// Each simulation's annual internal rate of return, as ``irr``.
    #[pyo3(signature = (*, lower = None, upper = None, points = 400, bisections = 8, newton_steps = 3, extend = false))]
    #[allow(clippy::too_many_arguments)]
    fn irr(
        &self,
        py: Python<'_>,
        lower: Option<f64>,
        upper: Option<f64>,
        points: usize,
        bisections: usize,
        newton_steps: usize,
        extend: bool,
    ) -> PyResult<PyIrrResult> {
        let options = irr_options(lower, upper, points, bisections, newton_steps, extend);
        let inner = py.detach(|| self.inner.irr(&options)).map_err(to_py)?;
        Ok(PyIrrResult { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Cashflows(n={}, months={})",
            self.inner.len(),
            self.inner.months()
        )
    }
}
