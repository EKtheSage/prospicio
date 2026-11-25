//! Python bindings for `prospicio_pricing::earned`: written, earned,
//! unearned and in-force amounts from policy dates.

use prospicio_pricing::earned::{
    Date, EarningMethod, Windows, earned_exposure as core_earned_exposure,
    in_force as core_in_force, policies_from_columns,
};
use prospicio_pricing::ratemaking::PeriodKind;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{to_py, unknown_name};

/// A date from a `datetime.date`, a `datetime.datetime` (its date) or a
/// `YYYY-MM-DD` string.
fn to_date(obj: &Bound<'_, PyAny>) -> PyResult<Date> {
    let text: String = match obj.extract::<String>() {
        Ok(text) => text,
        Err(_) => obj
            .call_method0("isoformat")
            .and_then(|iso| iso.extract::<String>())
            .map_err(|_| {
                PyValueError::new_err("a date must be a datetime.date or a YYYY-MM-DD string")
            })?,
    };
    // A datetime's isoformat carries the time after the date.
    let day = text.get(..10).unwrap_or(&text);
    Date::parse(day).map_err(to_py)
}

fn to_dates(objs: &[Bound<'_, PyAny>]) -> PyResult<Vec<Date>> {
    objs.iter().map(to_date).collect()
}

type Change<'py> = (usize, Bound<'py, PyAny>, f64);

fn policies(
    effective: &[Bound<'_, PyAny>],
    expiration: &[Bound<'_, PyAny>],
    amount: &[f64],
    label: Option<Vec<String>>,
    changes: Option<Vec<Change<'_>>>,
    cancellation: Option<Vec<Option<Bound<'_, PyAny>>>>,
) -> PyResult<Vec<prospicio_pricing::earned::Policy>> {
    let changes = changes
        .unwrap_or_default()
        .iter()
        .map(|(k, date, amount)| Ok((*k, to_date(date)?, *amount)))
        .collect::<PyResult<Vec<_>>>()?;
    let cancellation = cancellation
        .unwrap_or_default()
        .iter()
        .map(|c| c.as_ref().map(to_date).transpose())
        .collect::<PyResult<Vec<_>>>()?;
    policies_from_columns(
        &to_dates(effective)?,
        &to_dates(expiration)?,
        amount,
        &label.unwrap_or_default(),
        &changes,
        &cancellation,
    )
    .map_err(to_py)
}

/// Written, earned, unearned and in-force amounts of a table of policies,
/// by window and label.
///
/// Each policy earns its amount (an exposure count, or a premium) evenly
/// over its term, from its effective date up to, and not including, its
/// expiration date. With ``kind="calendar"`` a window holds what was
/// written and earned in it, and the unearned and in-force amounts at its
/// end. With ``kind="policy"`` a window holds the policies written in it,
/// with all they earn; give ``valuation`` to see them as of a date.
///
/// "At a window's end" and "as of a valuation date" mean the close of the
/// day before: a policy that expires on that date is still in force, and
/// one effective on it is not yet written.
///
/// Parameters
/// ----------
/// effective, expiration : list of datetime.date or str
///     One entry a policy; strings are ``YYYY-MM-DD``. The expiration is
///     the first day the policy is not in force.
/// amount : list of float
///     The full-term amount of each policy.
/// windows : list of datetime.date or str
///     The dates the windows lie between, increasing; see
///     ``calendar_windows``.
/// kind : {"calendar", "policy"}, default "calendar"
/// method : {"daily", "half_month", "twenty_fourths"}, default "daily"
///     Exact days; the half-month grid of the ratemaking exhibits; or the
///     "15th of the month" rule, every policy date at mid-month.
/// valuation : datetime.date or str, optional
///     See the table as of this date.
/// label : list of str, optional
///     A label a policy; the table has a set of rows for each.
/// changes : list of (int, date, float), optional
///     Mid-term changes: the position of the policy (from 0), the date,
///     and the full-term amount from that date.
/// cancellation : list of (date or None), optional
///     A cancellation date a policy, the first day not in force.
///
/// Returns
/// -------
/// dict
///     Columns ``label``, ``from``, ``to`` (ISO dates), ``written``,
///     ``earned``, ``unearned`` and ``in_force``, one row a label and
///     window.
///
/// Raises
/// ------
/// ValueError
///     For columns of different lengths, a date that does not exist, an
///     unknown ``kind`` or ``method``, or a policy, change or
///     cancellation that is not valid.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import calendar_windows, earned_exposure
/// >>> t = earned_exposure(["2010-10-01"], ["2011-10-01"], [1.0],
/// ...                     calendar_windows(2010, 2011), method="half_month")
/// >>> t["written"], t["earned"]
/// ([1.0, 0.0], [0.25, 0.75])
#[pyfunction]
#[pyo3(signature = (effective, expiration, amount, windows, kind = "calendar", method = "daily", valuation = None, label = None, changes = None, cancellation = None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn earned_exposure<'py>(
    py: Python<'py>,
    effective: Vec<Bound<'py, PyAny>>,
    expiration: Vec<Bound<'py, PyAny>>,
    amount: Vec<f64>,
    windows: Vec<Bound<'py, PyAny>>,
    kind: &str,
    method: &str,
    valuation: Option<Bound<'py, PyAny>>,
    label: Option<Vec<String>>,
    changes: Option<Vec<Change<'py>>>,
    cancellation: Option<Vec<Option<Bound<'py, PyAny>>>>,
) -> PyResult<Bound<'py, PyDict>> {
    let kind =
        PeriodKind::from_name(kind).ok_or_else(|| unknown_name("kind", kind, PeriodKind::NAMES))?;
    let method = EarningMethod::from_name(method)
        .ok_or_else(|| unknown_name("method", method, EarningMethod::NAMES))?;
    let policies = policies(
        &effective,
        &expiration,
        &amount,
        label,
        changes,
        cancellation,
    )?;
    let windows = Windows::new(to_dates(&windows)?).map_err(to_py)?;
    let valuation = valuation.as_ref().map(to_date).transpose()?;
    let table =
        core_earned_exposure(&policies, &windows, kind, method, valuation).map_err(to_py)?;
    let rows = table.rows();
    let column = |f: fn(&prospicio_pricing::earned::EarnedRow) -> f64| -> Vec<f64> {
        rows.iter().map(f).collect()
    };
    let d = PyDict::new(py);
    d.set_item(
        "label",
        rows.iter().map(|r| r.label.clone()).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "from",
        rows.iter().map(|r| r.from.to_string()).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "to",
        rows.iter().map(|r| r.to.to_string()).collect::<Vec<_>>(),
    )?;
    d.set_item("written", column(|r| r.written))?;
    d.set_item("earned", column(|r| r.earned))?;
    d.set_item("unearned", column(|r| r.unearned))?;
    d.set_item("in_force", column(|r| r.in_force))?;
    Ok(d)
}

/// The dates between calendar windows: years, quarters or months from 1
/// January of ``first_year`` through 31 December of ``last_year``.
///
/// Parameters
/// ----------
/// first_year, last_year : int
/// by : {"year", "quarter", "month"}, default "year"
///
/// Returns
/// -------
/// list of str
///     ISO dates, one more than there are windows.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import calendar_windows
/// >>> calendar_windows(2024, 2024, by="quarter")
/// ['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01', '2025-01-01']
#[pyfunction]
#[pyo3(signature = (first_year, last_year, by = "year"))]
pub(crate) fn calendar_windows(first_year: i32, last_year: i32, by: &str) -> PyResult<Vec<String>> {
    let windows = Windows::calendar(by, first_year, last_year).map_err(to_py)?;
    Ok(windows.bounds().iter().map(Date::to_string).collect())
}

/// The amount in force on a date: the sum of the full-term amounts, as
/// they stand that day, of the policies effective on or before it and not
/// yet expired or cancelled. A snapshot, with no regard to how long each
/// policy runs.
///
/// Parameters
/// ----------
/// effective, expiration : list of datetime.date or str
/// amount : list of float
/// date : datetime.date or str
/// changes : list of (int, date, float), optional
/// cancellation : list of (date or None), optional
///     As in ``earned_exposure``.
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from prospicio.pricing import in_force
/// >>> in_force(["2011-07-01"], ["2012-07-01"], [400.0], "2012-01-01",
/// ...          changes=[(0, "2012-01-01", 800.0)])
/// 800.0
#[pyfunction]
#[pyo3(signature = (effective, expiration, amount, date, changes = None, cancellation = None))]
pub(crate) fn in_force<'py>(
    effective: Vec<Bound<'py, PyAny>>,
    expiration: Vec<Bound<'py, PyAny>>,
    amount: Vec<f64>,
    date: Bound<'py, PyAny>,
    changes: Option<Vec<Change<'py>>>,
    cancellation: Option<Vec<Option<Bound<'py, PyAny>>>>,
) -> PyResult<f64> {
    let policies = policies(
        &effective,
        &expiration,
        &amount,
        None,
        changes,
        cancellation,
    )?;
    Ok(core_in_force(&policies, to_date(&date)?))
}
