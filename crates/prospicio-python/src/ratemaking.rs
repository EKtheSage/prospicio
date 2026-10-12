//! `prospicio.pricing`: ratemaking, premium at the current rate level by
//! the parallelogram method and trend (Aggregate lane,
//! `docs/design/ratemaking.md`).

use prospicio_pricing::ratemaking::{
    Applies, Period, RateChange, RateHistory, RateLevel, TrendFit, TrendKind,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::to_py;

/// The year fraction of a date on Werner and Modlin's half-month grid: the
/// first of a month is its start, a day near the middle is half way, and
/// the last day is the next month's start (6/30 is 7/1).
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
#[derive(Clone)]
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
    fn new(
        effective: Vec<f64>,
        changes: Vec<f64>,
        applies: Option<Vec<String>>,
    ) -> PyResult<Self> {
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
