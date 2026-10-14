//! `prospicio.pricing`: ratemaking, premium at the current rate level by
//! the parallelogram method, trend and the overall indication (Aggregate lane,
//! `docs/design/ratemaking.md`).

use prospicio_pricing::ratemaking::{
    Applies, ExpenseCategory, LossRatioExperience, LossRatioYear, Period, Provisions,
    PurePremiumExperience, PurePremiumYear, RateChange, RateHistory, RateLevel, TrendFit,
    TrendKind,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::to_py;

/// The year fraction of a date on Werner and Modlin's half-month grid: the
/// first of a month is its start, a day near the middle is half way, and
/// the last day is the next month's start (6/30 is 7/1).
///
/// It is a ratemaking date, not a calendar: do not assign accident years
/// with it, since ``year_fraction(2016, 12, 31)`` is 2017.0. Take the
/// accident year from the date itself.
///
/// Parameters
/// ----------
/// year : int
/// month : int
/// day : int
///
/// Returns
/// -------
/// float
///
/// Raises
/// ------
/// ValueError
///     If the month or day does not exist.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import year_fraction
/// >>> year_fraction(2013, 6, 30) - year_fraction(2011, 1, 1)
/// 2.5
#[pyfunction]
pub(crate) fn year_fraction(year: i32, month: u8, day: u8) -> PyResult<f64> {
    prospicio_pricing::ratemaking::year_fraction(year, month, day).map_err(to_py)
}

/// ``(1 + rate) ** years``, the factor of an exponential trend.
///
/// Parameters
/// ----------
/// rate : float
///     The annual trend, ``0.02`` for +2%.
/// years : float
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from prospicio.pricing import trend_factor
/// >>> round(trend_factor(0.02, 2.5), 4)
/// 1.0508
#[pyfunction]
pub(crate) fn trend_factor(rate: f64, years: f64) -> f64 {
    prospicio_pricing::ratemaking::trend_factor(rate, years)
}

/// An experience or forecast period ``[start, end)`` in year fractions:
/// premium earned (losses occurring) in a calendar window, or policies
/// written in a policy window. A forecast period is the policy period of
/// the policies written while new rates are in effect.
///
/// Build one with ``Period.calendar`` or ``Period.policy``.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import Period
/// >>> cy = Period.calendar(2011.0, 2012.0)
/// >>> future = Period.policy(2013.0, 2014.0)
/// >>> future.average_written_date(1.0) - cy.average_written_date(1.0)
/// 2.5
#[pyclass(name = "Period", module = "prospicio.pricing", frozen)]
pub(crate) struct PyPeriod {
    inner: Period,
}

#[pymethods]
impl PyPeriod {
    /// Premium earned (losses occurring) in ``[start, end)``.
    ///
    /// Parameters
    /// ----------
    /// start : float
    /// end : float
    ///
    /// Returns
    /// -------
    /// Period
    #[staticmethod]
    fn calendar(start: f64, end: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Period::calendar(start, end).map_err(to_py)?,
        })
    }

    /// Policies written in ``[start, end)``, over their whole term.
    ///
    /// Parameters
    /// ----------
    /// start : float
    /// end : float
    ///
    /// Returns
    /// -------
    /// Period
    #[staticmethod]
    fn policy(start: f64, end: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Period::policy(start, end).map_err(to_py)?,
        })
    }

    /// ``"calendar"`` or ``"policy"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            Period::Calendar { .. } => "calendar",
            Period::Policy { .. } => "policy",
        }
    }

    /// The average date the period's premium was written, for policies of
    /// ``term`` years: premium is trended from and to these dates.
    ///
    /// Parameters
    /// ----------
    /// term : float
    ///
    /// Returns
    /// -------
    /// float
    fn average_written_date(&self, term: f64) -> f64 {
        self.inner.average_written_date(term)
    }

    /// The average date the period's premium is earned, the average
    /// accident date of its losses: losses are trended from and to these
    /// dates.
    ///
    /// Parameters
    /// ----------
    /// term : float
    ///
    /// Returns
    /// -------
    /// float
    fn average_earned_date(&self, term: f64) -> f64 {
        self.inner.average_earned_date(term)
    }

    fn __repr__(&self) -> String {
        match self.inner {
            Period::Calendar { from, to } => format!("Period.calendar({from:?}, {to:?})"),
            Period::Policy { from, to } => format!("Period.policy({from:?}, {to:?})"),
        }
    }
}

fn applies(values: Option<Vec<String>>, n: usize) -> PyResult<Vec<Applies>> {
    let Some(values) = values else {
        return Ok(vec![Applies::Written; n]);
    };
    if values.len() != n {
        return Err(PyValueError::new_err(
            "applies must have one entry per rate change",
        ));
    }
    values
        .iter()
        .map(|a| match a.as_str() {
            "written" => Ok(Applies::Written),
            "in_force" => Ok(Applies::InForce),
            other => Err(PyValueError::new_err(format!(
                "applies must be \"written\" or \"in_force\", got {other:?}"
            ))),
        })
        .collect()
}

/// A history of rate changes from an initial level of 1, and the
/// parallelogram method: the portion of a period's premium at each rate
/// level under uniform writing and earning, and the factor to the current
/// level (Werner and Modlin, ch. 5).
///
/// Parameters
/// ----------
/// effective : list of float
///     Effective dates as year fractions, in order.
/// changes : list of float
///     Overall average changes, ``0.05`` for +5%.
/// applies : list of str, optional
///     ``"written"`` (policies written on or after the date; the default)
///     or ``"in_force"`` (all premium earned on or after it, a law change).
///
/// Raises
/// ------
/// ValueError
///     For dates out of order, a change of -100% or less, or mismatched
///     lengths.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import Period, RateHistory
/// >>> h = RateHistory([2010.5, 2011.0, 2012.25], [0.05, 0.10, -0.01])
/// >>> level = h.rate_level(Period.calendar(2011.0, 2012.0), term=1.0)
/// >>> level.shares
/// [0.125, 0.375, 0.5]
/// >>> round(level.on_level_factor, 4)
/// 1.0431
#[pyclass(name = "RateHistory", module = "prospicio.pricing", frozen)]
pub(crate) struct PyRateHistory {
    inner: RateHistory,
}

#[pymethods]
impl PyRateHistory {
    #[new]
    #[pyo3(signature = (effective, changes, applies = None))]
    fn new(effective: Vec<f64>, changes: Vec<f64>, applies: Option<Vec<String>>) -> PyResult<Self> {
        if effective.len() != changes.len() {
            return Err(PyValueError::new_err(
                "effective and changes must have the same length",
            ));
        }
        let applies = self::applies(applies, changes.len())?;
        let changes: Vec<RateChange> = effective
            .iter()
            .zip(&changes)
            .zip(&applies)
            .map(|((&effective, &change), &applies)| RateChange {
                effective,
                change,
                applies,
            })
            .collect();
        Ok(Self {
            inner: RateHistory::new(&changes).map_err(to_py)?,
        })
    }

    /// A history from the cumulative level after each change, as an
    /// exhibit prints them; rounded printed levels carry into the factors.
    ///
    /// Parameters
    /// ----------
    /// effective : list of float
    /// levels : list of float
    /// applies : list of str, optional
    ///
    /// Returns
    /// -------
    /// RateHistory
    #[staticmethod]
    #[pyo3(signature = (effective, levels, applies = None))]
    fn from_levels(
        effective: Vec<f64>,
        levels: Vec<f64>,
        applies: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let applies = self::applies(applies, levels.len())?;
        Ok(Self {
            inner: RateHistory::from_levels(&effective, &levels, &applies).map_err(to_py)?,
        })
    }

    /// The level after every change.
    #[getter]
    fn current_level(&self) -> f64 {
        self.inner.current_level()
    }

    /// The portion of ``period``'s premium at each rate level, for
    /// policies of ``term`` years.
    ///
    /// Parameters
    /// ----------
    /// period : Period
    /// term : float
    ///
    /// Returns
    /// -------
    /// RateLevel
    fn rate_level(&self, period: &PyPeriod, term: f64) -> PyResult<PyRateLevel> {
        Ok(PyRateLevel {
            inner: self.inner.rate_level(&period.inner, term).map_err(to_py)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RateHistory(changes={}, current_level={:?})",
            self.inner.changes().len(),
            self.inner.current_level()
        )
    }
}

/// A period's premium by rate level, from ``RateHistory.rate_level``. Each
/// list has one entry per cell of constant level with a positive share.
#[pyclass(name = "RateLevel", module = "prospicio.pricing", frozen)]
pub(crate) struct PyRateLevel {
    inner: RateLevel,
}

#[pymethods]
impl PyRateLevel {
    /// Each cell's share of the period's premium.
    #[getter]
    fn shares(&self) -> Vec<f64> {
        self.inner.portions.iter().map(|p| p.share).collect()
    }

    /// Each cell's cumulative rate level.
    #[getter]
    fn levels(&self) -> Vec<f64> {
        self.inner.portions.iter().map(|p| p.level).collect()
    }

    /// The number of written-policy changes in effect for each cell.
    #[getter]
    fn written_changes(&self) -> Vec<usize> {
        self.inner
            .portions
            .iter()
            .map(|p| p.written_changes)
            .collect()
    }

    /// The number of in-force changes in effect for each cell.
    #[getter]
    fn in_force_changes(&self) -> Vec<usize> {
        self.inner
            .portions
            .iter()
            .map(|p| p.in_force_changes)
            .collect()
    }

    /// ``sum(shares * levels)``.
    #[getter]
    fn average_level(&self) -> f64 {
        self.inner.average_level
    }

    /// The level after every change in the history.
    #[getter]
    fn current_level(&self) -> f64 {
        self.inner.current_level
    }

    /// ``current_level / average_level``.
    #[getter]
    fn on_level_factor(&self) -> f64 {
        self.inner.on_level_factor()
    }

    fn __repr__(&self) -> String {
        format!(
            "RateLevel(average_level={:?}, on_level_factor={:?})",
            self.inner.average_level,
            self.inner.on_level_factor()
        )
    }
}

/// A trend fitted by least squares to values at times in years:
/// ``log(y)`` on time for an exponential trend, ``y`` for a linear one. For
/// quarterly points, pass times a quarter apart.
///
/// Build one with ``TrendFit.exponential`` or ``TrendFit.linear``.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import TrendFit
/// >>> t = [0.0, 0.25, 0.5, 0.75]
/// >>> fit = TrendFit.exponential(t, [100 * 1.03 ** x for x in t])
/// >>> round(fit.annual_change, 6)
/// 0.03
#[pyclass(name = "TrendFit", module = "prospicio.pricing", frozen)]
pub(crate) struct PyTrendFit {
    inner: TrendFit,
}

#[pymethods]
impl PyTrendFit {
    /// Fit ``log(y) = a + b t``; every value must be positive.
    ///
    /// Parameters
    /// ----------
    /// times : list of float
    /// values : list of float
    ///
    /// Returns
    /// -------
    /// TrendFit
    #[staticmethod]
    fn exponential(times: Vec<f64>, values: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: TrendFit::exponential(&times, &values).map_err(to_py)?,
        })
    }

    /// Fit ``y = a + b t``.
    ///
    /// Parameters
    /// ----------
    /// times : list of float
    /// values : list of float
    ///
    /// Returns
    /// -------
    /// TrendFit
    #[staticmethod]
    fn linear(times: Vec<f64>, values: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: TrendFit::linear(&times, &values).map_err(to_py)?,
        })
    }

    /// ``"exponential"`` or ``"linear"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind {
            TrendKind::Exponential => "exponential",
            TrendKind::Linear => "linear",
        }
    }

    /// The intercept, on the log scale for an exponential fit.
    #[getter]
    fn intercept(&self) -> f64 {
        self.inner.intercept
    }

    /// The slope per year, on the log scale for an exponential fit.
    #[getter]
    fn slope(&self) -> f64 {
        self.inner.slope
    }

    /// The coefficient of determination, on the fitted scale.
    #[getter]
    fn r_squared(&self) -> f64 {
        self.inner.r_squared
    }

    /// The number of points.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }

    /// The annual trend: ``exp(slope) - 1`` for an exponential fit, the
    /// slope (an amount a year) for a linear one.
    #[getter]
    fn annual_change(&self) -> f64 {
        self.inner.annual_change()
    }

    /// The fitted value at ``time``.
    ///
    /// Parameters
    /// ----------
    /// time : float
    ///
    /// Returns
    /// -------
    /// float
    fn fitted(&self, time: f64) -> f64 {
        self.inner.fitted(time)
    }

    fn __repr__(&self) -> String {
        format!(
            "TrendFit(kind={:?}, annual_change={:?}, n={})",
            self.kind(),
            self.inner.annual_change(),
            self.inner.n
        )
    }
}

/// A column of per-year values: a list of ``n`` numbers, a single number
/// for every year, or ``None`` for 1.
fn column(value: Option<&Bound<'_, PyAny>>, n: usize, name: &str) -> PyResult<Vec<f64>> {
    let Some(value) = value else {
        return Ok(vec![1.0; n]);
    };
    if let Ok(x) = value.extract::<f64>() {
        return Ok(vec![x; n]);
    }
    let v: Vec<f64> = value.extract()?;
    if v.len() != n {
        return Err(PyValueError::new_err(format!(
            "{name} has {} values for {n} years",
            v.len()
        )));
    }
    Ok(v)
}

/// Experience years for the loss ratio method: earned premium brought to
/// the current rate level and trended, and losses developed, trended and
/// loaded for LAE, combined as the ratio of their totals (Werner and
/// Modlin, ch. 8).
///
/// Every factor is a list with one value per year, one number for all
/// years, or ``None`` for 1.
///
/// Parameters
/// ----------
/// earned_premium : list of float
/// losses : list of float
///     Reported or paid losses, with whatever LAE they include.
/// on_level_factor, premium_trend, development, loss_trend, lae_factor : list of float or float, optional
///     ``lae_factor`` is ``1 +`` the LAE ratio not in ``losses``; a net
///     trend goes in ``loss_trend``.
///
/// Raises
/// ------
/// ValueError
///     For no years, mismatched lengths, or a premium or factor that is not
///     positive.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import LossRatioExperience
/// >>> e = LossRatioExperience([100.0, 300.0], [50.0, 250.0])
/// >>> e.loss_ratio
/// 0.75
#[pyclass(name = "LossRatioExperience", module = "prospicio.pricing", frozen)]
pub(crate) struct PyLossRatioExperience {
    inner: LossRatioExperience,
}

#[pymethods]
impl PyLossRatioExperience {
    #[new]
    #[pyo3(signature = (earned_premium, losses, on_level_factor = None, premium_trend = None, development = None, loss_trend = None, lae_factor = None))]
    fn new(
        earned_premium: Vec<f64>,
        losses: Vec<f64>,
        on_level_factor: Option<&Bound<'_, PyAny>>,
        premium_trend: Option<&Bound<'_, PyAny>>,
        development: Option<&Bound<'_, PyAny>>,
        loss_trend: Option<&Bound<'_, PyAny>>,
        lae_factor: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let n = earned_premium.len();
        if losses.len() != n {
            return Err(PyValueError::new_err(
                "earned_premium and losses must have the same length",
            ));
        }
        let olf = column(on_level_factor, n, "on_level_factor")?;
        let pt = column(premium_trend, n, "premium_trend")?;
        let dev = column(development, n, "development")?;
        let lt = column(loss_trend, n, "loss_trend")?;
        let lae = column(lae_factor, n, "lae_factor")?;
        let years: Vec<LossRatioYear> = (0..n)
            .map(|i| LossRatioYear {
                earned_premium: earned_premium[i],
                on_level_factor: olf[i],
                premium_trend: pt[i],
                losses: losses[i],
                development: dev[i],
                loss_trend: lt[i],
                lae_factor: lae[i],
            })
            .collect();
        Ok(Self {
            inner: LossRatioExperience::new(&years).map_err(to_py)?,
        })
    }

    /// Total projected premium at the current rate level.
    #[getter]
    fn projected_premium(&self) -> f64 {
        self.inner.projected_premium()
    }

    /// Total projected ultimate losses and LAE.
    #[getter]
    fn projected_losses(&self) -> f64 {
        self.inner.projected_losses()
    }

    /// The projected loss and LAE ratio, the ratio of the totals.
    #[getter]
    fn loss_ratio(&self) -> f64 {
        self.inner.loss_ratio()
    }

    /// Each year's projected premium.
    #[getter]
    fn projected_premium_by_year(&self) -> Vec<f64> {
        self.inner
            .years()
            .iter()
            .map(|y| y.projected_premium())
            .collect()
    }

    /// Each year's projected losses and LAE.
    #[getter]
    fn projected_losses_by_year(&self) -> Vec<f64> {
        self.inner
            .years()
            .iter()
            .map(|y| y.projected_losses())
            .collect()
    }

    /// Each year's projected loss ratio.
    #[getter]
    fn loss_ratio_by_year(&self) -> Vec<f64> {
        self.inner.years().iter().map(|y| y.loss_ratio()).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "LossRatioExperience(years={}, loss_ratio={:?})",
            self.inner.years().len(),
            self.inner.loss_ratio()
        )
    }
}

/// Experience years for the pure premium method: losses developed, trended
/// and loaded for LAE over earned exposure, combined as the ratio of their
/// totals.
///
/// Parameters
/// ----------
/// exposure : list of float
/// losses : list of float
/// development, loss_trend, lae_factor : list of float or float, optional
///
/// Raises
/// ------
/// ValueError
///     For no years, mismatched lengths, or an exposure or factor that is
///     not positive.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import PurePremiumExperience
/// >>> e = PurePremiumExperience([10.0, 30.0], [1000.0, 2000.0], loss_trend=1.1)
/// >>> round(e.pure_premium, 6)
/// 82.5
#[pyclass(name = "PurePremiumExperience", module = "prospicio.pricing", frozen)]
pub(crate) struct PyPurePremiumExperience {
    inner: PurePremiumExperience,
}

#[pymethods]
impl PyPurePremiumExperience {
    #[new]
    #[pyo3(signature = (exposure, losses, development = None, loss_trend = None, lae_factor = None))]
    fn new(
        exposure: Vec<f64>,
        losses: Vec<f64>,
        development: Option<&Bound<'_, PyAny>>,
        loss_trend: Option<&Bound<'_, PyAny>>,
        lae_factor: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let n = exposure.len();
        if losses.len() != n {
            return Err(PyValueError::new_err(
                "exposure and losses must have the same length",
            ));
        }
        let dev = column(development, n, "development")?;
        let lt = column(loss_trend, n, "loss_trend")?;
        let lae = column(lae_factor, n, "lae_factor")?;
        let years: Vec<PurePremiumYear> = (0..n)
            .map(|i| PurePremiumYear {
                exposure: exposure[i],
                losses: losses[i],
                development: dev[i],
                loss_trend: lt[i],
                lae_factor: lae[i],
            })
            .collect();
        Ok(Self {
            inner: PurePremiumExperience::new(&years).map_err(to_py)?,
        })
    }

    /// Total earned exposure.
    #[getter]
    fn exposure(&self) -> f64 {
        self.inner.exposure()
    }

    /// Total projected ultimate losses and LAE.
    #[getter]
    fn projected_losses(&self) -> f64 {
        self.inner.projected_losses()
    }

    /// The projected pure premium, the ratio of the totals.
    #[getter]
    fn pure_premium(&self) -> f64 {
        self.inner.pure_premium()
    }

    /// Each year's projected losses and LAE.
    #[getter]
    fn projected_losses_by_year(&self) -> Vec<f64> {
        self.inner
            .years()
            .iter()
            .map(|y| y.projected_losses())
            .collect()
    }

    /// Each year's projected pure premium.
    #[getter]
    fn pure_premium_by_year(&self) -> Vec<f64> {
        self.inner
            .years()
            .iter()
            .map(|y| y.pure_premium())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "PurePremiumExperience(years={}, pure_premium={:?})",
            self.inner.years().len(),
            self.inner.pure_premium()
        )
    }
}

/// The provisions proportional to premium: variable expenses ``V`` and the
/// target underwriting profit ``Q`` (Werner and Modlin, ch. 7 and 8).
///
/// Parameters
/// ----------
/// variable : float
///     In ``[0, 1)``.
/// profit : float
///     May be negative (investment income); ``variable + profit < 1``.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import Provisions
/// >>> p = Provisions(0.25, 0.10)
/// >>> round(p.pure_premium_rate(300.0, 25.0), 6)
/// 500.0
/// >>> round(p.loss_ratio_change(0.65, 0.065), 6)
/// 0.1
#[pyclass(name = "Provisions", module = "prospicio.pricing", frozen)]
pub(crate) struct PyProvisions {
    inner: Provisions,
}

#[pymethods]
impl PyProvisions {
    #[new]
    fn new(variable: f64, profit: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Provisions::new(variable, profit).map_err(to_py)?,
        })
    }

    /// The variable expense provision.
    #[getter]
    fn variable(&self) -> f64 {
        self.inner.variable
    }

    /// The target profit provision.
    #[getter]
    fn profit(&self) -> f64 {
        self.inner.profit
    }

    /// ``1 - V - Q``.
    #[getter]
    fn variable_permissible_loss_ratio(&self) -> f64 {
        self.inner.variable_permissible_loss_ratio()
    }

    /// ``1 - F - V - Q``.
    ///
    /// Parameters
    /// ----------
    /// fixed_ratio : float, default 0.0
    ///
    /// Returns
    /// -------
    /// float
    #[pyo3(signature = (fixed_ratio = 0.0))]
    fn permissible_loss_ratio(&self, fixed_ratio: f64) -> f64 {
        self.inner.permissible_loss_ratio(fixed_ratio)
    }

    /// The loss ratio method's indicated change, ``(LR + F) / (1 - V - Q) - 1``.
    ///
    /// Parameters
    /// ----------
    /// loss_ratio : float
    ///     The projected loss and LAE ratio at current rates.
    /// fixed_ratio : float, default 0.0
    ///
    /// Returns
    /// -------
    /// float
    #[pyo3(signature = (loss_ratio, fixed_ratio = 0.0))]
    fn loss_ratio_change(&self, loss_ratio: f64, fixed_ratio: f64) -> f64 {
        self.inner.loss_ratio_change(loss_ratio, fixed_ratio)
    }

    /// The pure premium method's indicated rate, ``(PP + fixed) / (1 - V - Q)``.
    /// Other costs per exposure, such as net reinsurance, add to either.
    ///
    /// Parameters
    /// ----------
    /// pure_premium : float
    /// fixed_per_exposure : float, default 0.0
    ///
    /// Returns
    /// -------
    /// float
    #[pyo3(signature = (pure_premium, fixed_per_exposure = 0.0))]
    fn pure_premium_rate(&self, pure_premium: f64, fixed_per_exposure: f64) -> f64 {
        self.inner
            .pure_premium_rate(pure_premium, fixed_per_exposure)
    }

    fn __repr__(&self) -> String {
        format!(
            "Provisions(variable={:?}, profit={:?})",
            self.inner.variable, self.inner.profit
        )
    }
}

/// The premium-based projection of expenses: each expense ratio split by
/// its fixed share and summed, as ``(fixed, variable)``. With every share 0
/// it is the all-variable method.
///
/// Parameters
/// ----------
/// ratios : list of float
///     Expense ratios to premium.
/// fixed_shares : list of float
///     The share of each that is fixed, in ``[0, 1]``.
///
/// Returns
/// -------
/// tuple of float
///
/// Examples
/// --------
/// >>> from prospicio.pricing import premium_based_expenses
/// >>> fixed, variable = premium_based_expenses([0.06, 0.10], [0.5, 0.0])
/// >>> round(fixed, 6), round(variable, 6)
/// (0.03, 0.13)
#[pyfunction]
pub(crate) fn premium_based_expenses(
    ratios: Vec<f64>,
    fixed_shares: Vec<f64>,
) -> PyResult<(f64, f64)> {
    if ratios.len() != fixed_shares.len() {
        return Err(PyValueError::new_err(
            "ratios and fixed_shares must have the same length",
        ));
    }
    let categories: Vec<ExpenseCategory> = ratios
        .iter()
        .zip(&fixed_shares)
        .map(|(&ratio, &fixed_share)| ExpenseCategory { ratio, fixed_share })
        .collect();
    prospicio_pricing::ratemaking::premium_based_expenses(&categories).map_err(to_py)
}

/// ``Z * estimate + (1 - Z) * complement``.
///
/// Parameters
/// ----------
/// estimate : float
/// credibility : float
/// complement : float
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from prospicio.pricing import credibility_weighted
/// >>> round(credibility_weighted(200.0, 0.3, 300.0), 6)
/// 270.0
#[pyfunction]
pub(crate) fn credibility_weighted(estimate: f64, credibility: f64, complement: f64) -> f64 {
    prospicio_pricing::ratemaking::credibility_weighted(estimate, credibility, complement)
}

/// The trended present rates complement as an indicated change:
/// ``(1 + prior_indicated) / (1 + prior_implemented) * trend - 1``, with
/// ``trend`` the factor from the last review's target effective date to the
/// new one (the loss trend over the premium trend for a loss ratio).
///
/// Parameters
/// ----------
/// prior_indicated : float
/// prior_implemented : float
/// trend : float
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from prospicio.pricing import trend_factor, trended_present_rates
/// >>> c = trended_present_rates(0.10, 0.06, trend_factor(0.05, 2.0))
/// >>> round(200 * (1 + c), 2)
/// 228.82
#[pyfunction]
pub(crate) fn trended_present_rates(
    prior_indicated: f64,
    prior_implemented: f64,
    trend: f64,
) -> f64 {
    prospicio_pricing::ratemaking::trended_present_rates(prior_indicated, prior_implemented, trend)
}
