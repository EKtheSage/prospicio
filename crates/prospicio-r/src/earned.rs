//! Aggregate lane: earned exposure from policy dates for the R `earned.R`
//! API (`docs/design/primary-rating.md`). Dates cross as `YYYY-MM-DD`
//! strings, an empty string for none.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_pricing::earned::{
    Date, EarningMethod, Policy, Windows, earned_exposure, in_force, policies_from_columns,
};
use prospicio_pricing::ratemaking::PeriodKind;

use crate::{to_r, unknown_name, whole};

fn dates(text: &[String]) -> Result<Vec<Date>> {
    text.iter().map(|t| Date::parse(t).map_err(to_r)).collect()
}

fn optional(text: &str) -> Result<Option<Date>> {
    if text.is_empty() {
        Ok(None)
    } else {
        Date::parse(text).map(Some).map_err(to_r)
    }
}

/// The policies of the columns R passes; `change_policy` counts from 1.
#[allow(clippy::too_many_arguments)]
fn policies(
    effective: &[String],
    expiration: &[String],
    amount: &[f64],
    label: &[String],
    change_policy: &[f64],
    change_date: &[String],
    change_amount: &[f64],
    cancellation: &[String],
) -> Result<Vec<Policy>> {
    if change_policy.len() != change_date.len() || change_policy.len() != change_amount.len() {
        return Err(Error::Other(
            "the changes' policy, date and amount columns differ in length".into(),
        ));
    }
    let change_dates = dates(change_date)?;
    let changes = change_policy
        .iter()
        .zip(change_dates)
        .zip(change_amount)
        .map(|((&k, date), &amount)| {
            let k = whole(k, "the changes' policy")? as usize;
            if k == 0 {
                return Err(Error::Other("the changes' policy counts from 1".into()));
            }
            Ok((k - 1, date, amount))
        })
        .collect::<Result<Vec<_>>>()?;
    let cancellation = cancellation
        .iter()
        .map(|c| optional(c))
        .collect::<Result<Vec<_>>>()?;
    policies_from_columns(
        &dates(effective)?,
        &dates(expiration)?,
        amount,
        label,
        &changes,
        &cancellation,
    )
    .map_err(to_r)
}

/// The table for R's `earned_exposure()`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn earned_exposure_rust(
    effective: Vec<String>,
    expiration: Vec<String>,
    amount: &[f64],
    bounds: Vec<String>,
    kind: &str,
    method: &str,
    valuation: &str,
    label: Vec<String>,
    change_policy: &[f64],
    change_date: Vec<String>,
    change_amount: &[f64],
    cancellation: Vec<String>,
) -> Result<List> {
    let kind =
        PeriodKind::from_name(kind).ok_or_else(|| unknown_name("kind", kind, PeriodKind::NAMES))?;
    let method = EarningMethod::from_name(method)
        .ok_or_else(|| unknown_name("method", method, EarningMethod::NAMES))?;
    let policies = policies(
        &effective,
        &expiration,
        amount,
        &label,
        change_policy,
        &change_date,
        change_amount,
        &cancellation,
    )?;
    let windows = Windows::new(dates(&bounds)?).map_err(to_r)?;
    let table =
        earned_exposure(&policies, &windows, kind, method, optional(valuation)?).map_err(to_r)?;
    let rows = table.rows();
    Ok(list!(
        label = rows
            .iter()
            .map(|r| r.label.clone())
            .collect::<Vec<String>>(),
        from = rows
            .iter()
            .map(|r| r.from.to_string())
            .collect::<Vec<String>>(),
        to = rows
            .iter()
            .map(|r| r.to.to_string())
            .collect::<Vec<String>>(),
        written = rows.iter().map(|r| r.written).collect::<Vec<f64>>(),
        earned = rows.iter().map(|r| r.earned).collect::<Vec<f64>>(),
        unearned = rows.iter().map(|r| r.unearned).collect::<Vec<f64>>(),
        in_force = rows.iter().map(|r| r.in_force).collect::<Vec<f64>>()
    ))
}

/// The window bounds for R's `calendar_windows()`, as ISO dates.
#[extendr]
fn calendar_windows_rust(first_year: f64, last_year: f64, by: &str) -> Result<Vec<String>> {
    let (first, last) = (
        whole(first_year, "first_year")?,
        whole(last_year, "last_year")?,
    );
    let windows = Windows::calendar(by, first as i32, last as i32).map_err(to_r)?;
    Ok(windows.bounds().iter().map(Date::to_string).collect())
}

/// The in-force amount on each of `at` for R's `in_force_amount()`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn in_force_amount_rust(
    effective: Vec<String>,
    expiration: Vec<String>,
    amount: &[f64],
    at: Vec<String>,
    change_policy: &[f64],
    change_date: Vec<String>,
    change_amount: &[f64],
    cancellation: Vec<String>,
) -> Result<Vec<f64>> {
    let policies = policies(
        &effective,
        &expiration,
        amount,
        &[],
        change_policy,
        &change_date,
        change_amount,
        &cancellation,
    )?;
    Ok(dates(&at)?
        .into_iter()
        .map(|date| in_force(&policies, date))
        .collect())
}

extendr_module! {
    mod earned;
    fn earned_exposure_rust;
    fn calendar_windows_rust;
    fn in_force_amount_rust;
}
