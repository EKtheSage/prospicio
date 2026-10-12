//! Aggregate lane: ratemaking for the R `ratemaking.R` API, premium at the
//! current rate level by the parallelogram method and trend
//! (`docs/design/ratemaking.md`).

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_pricing::ratemaking::{Applies, Period, RateChange, RateHistory, TrendFit};

use crate::to_r;

/// Year fractions of dates on the half-month grid.
#[extendr]
fn year_fractions(year: &[i32], month: &[i32], day: &[i32]) -> Result<Vec<f64>> {
    if month.len() != year.len() || day.len() != year.len() {
        return Err(Error::Other(
            "year, month and day must have the same length".into(),
        ));
    }
    year.iter()
        .zip(month)
        .zip(day)
        .map(|((&y, &m), &d)| {
            let (m, d) = (
                u8::try_from(m).map_err(|_| Error::Other(format!("no month {m}")))?,
                u8::try_from(d).map_err(|_| Error::Other(format!("no day {d}")))?,
            );
            prospicio_pricing::ratemaking::year_fraction(y, m, d).map_err(to_r)
        })
        .collect()
}

fn period(kind: &str, start: f64, end: f64) -> Result<Period> {
    match kind {
        "calendar" => Period::calendar(start, end).map_err(to_r),
        "policy" => Period::policy(start, end).map_err(to_r),
        other => Err(Error::Other(format!(
            "period must be \"calendar\" or \"policy\", got {other}"
        ))),
    }
}

/// `c(average written date, average earned date)` of a period.
#[extendr]
fn period_dates(kind: &str, start: f64, end: f64, term: f64) -> Result<Vec<f64>> {
    let p = period(kind, start, end)?;
    Ok(vec![p.average_written_date(term), p.average_earned_date(term)])
}

fn applies(values: &[String]) -> Result<Vec<Applies>> {
    values
        .iter()
        .map(|a| match a.as_str() {
            "written" => Ok(Applies::Written),
            "in_force" => Ok(Applies::InForce),
            other => Err(Error::Other(format!(
                "applies must be \"written\" or \"in_force\", got {other}"
            ))),
        })
        .collect()
}

/// The parallelogram method for one period. `values` are changes, or
/// cumulative levels when `levels` is true.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rate_level_rust(
    effective: &[f64],
    values: &[f64],
    applies_to: Vec<String>,
    levels: bool,
    kind: &str,
    start: f64,
    end: f64,
    term: f64,
) -> Result<List> {
    if effective.len() != values.len() || applies_to.len() != values.len() {
        return Err(Error::Other(
            "effective, the changes and applies must have the same length".into(),
        ));
    }
    let applies = applies(&applies_to)?;
    let history = if levels {
        RateHistory::from_levels(effective, values, &applies)
    } else {
        let changes: Vec<RateChange> = effective
            .iter()
            .zip(values)
            .zip(&applies)
            .map(|((&effective, &change), &applies)| RateChange {
                effective,
                change,
                applies,
            })
            .collect();
        RateHistory::new(&changes)
    }
    .map_err(to_r)?;
    let level = history
        .rate_level(&period(kind, start, end)?, term)
        .map_err(to_r)?;
    let p = &level.portions;
    Ok(list!(
        written_changes = p.iter().map(|x| x.written_changes as f64).collect::<Vec<_>>(),
        in_force_changes = p.iter().map(|x| x.in_force_changes as f64).collect::<Vec<_>>(),
        level = p.iter().map(|x| x.level).collect::<Vec<_>>(),
        share = p.iter().map(|x| x.share).collect::<Vec<_>>(),
        average_level = level.average_level,
        current_level = level.current_level,
        on_level_factor = level.on_level_factor()
    ))
}

/// A least-squares trend: `c(intercept, slope, r_squared, annual_change)`.
#[extendr]
fn trend_fit_rust(times: &[f64], values: &[f64], kind: &str) -> Result<Vec<f64>> {
    let fit = match kind {
        "exponential" => TrendFit::exponential(times, values),
        "linear" => TrendFit::linear(times, values),
        other => {
            return Err(Error::Other(format!(
                "kind must be \"exponential\" or \"linear\", got {other}"
            )));
        }
    }
    .map_err(to_r)?;
    Ok(vec![
        fit.intercept,
        fit.slope,
        fit.r_squared,
        fit.annual_change(),
    ])
}

extendr_module! {
    mod ratemaking;
    fn year_fractions;
    fn period_dates;
    fn rate_level_rust;
    fn trend_fit_rust;
}
