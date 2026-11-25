//! `calendar_index()`, `indexation()`, `restate()`, `inflate()`,
//! `reinflate()` and `future_cells()` (issue #979,
//! `docs/design/reserving-inflation.md`, PR 2): indexation of a triangle by
//! a calendar index and re-inflation of a projection, over
//! `prospicio_reserving::inflation`.
//!
//! The R side holds the index, the base and the cells as plain vectors in
//! S7 objects; every function here takes them and rebuilds the Rust values.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_reserving::{
    CalendarIndex, FutureCell, FutureCells, Grain, Indexation, Month, Period, Reinflated,
};

use crate::odp_glm::OdpGlmFit;
use crate::reserving::{ChainLadderFit, Triangle, to_r};

/// A period as a label: `"2020"`, `"2020H2"`, `"2020Q3"` or `"2020-07"`.
fn parse_period(text: &str) -> Option<Period> {
    let text = text.trim();
    let num = |s: &str| s.parse::<i32>().ok();
    let (grain, year, within) = if let Some((y, q)) = text.split_once('Q') {
        (
            Grain::Quarter,
            num(y)?,
            (num(q).filter(|q| (1..=4).contains(q))? - 1) * 3 + 1,
        )
    } else if let Some((y, h)) = text.split_once('H') {
        (
            Grain::Semester,
            num(y)?,
            (num(h).filter(|h| (1..=2).contains(h))? - 1) * 6 + 1,
        )
    } else if let Some((y, m)) = text.split_once('-') {
        (
            Grain::Month,
            num(y)?,
            num(m).filter(|m| (1..=12).contains(m))?,
        )
    } else {
        (Grain::Year, num(text)?, 1)
    };
    Some(Period::containing(
        Month::new(year, within as u8).ok()?,
        grain,
    ))
}

fn period(text: &str, what: &str) -> Result<Period> {
    parse_period(text).ok_or_else(|| {
        Error::Other(format!(
            "{what} must be a period such as \"2020\", \"2020H2\", \"2020Q3\" or \"2020-07\", \
             got {text:?}"
        ))
    })
}

fn month(text: &str, what: &str) -> Result<Month> {
    let parsed = text
        .trim()
        .split_once('-')
        .and_then(|(y, m)| Month::new(y.parse().ok()?, m.parse().ok()?).ok());
    parsed.ok_or_else(|| Error::Other(format!("{what} must be a month \"YYYY-MM\", got {text:?}")))
}

pub(crate) fn index(first: &str, values: Vec<f64>, rate: Nullable<f64>) -> Result<CalendarIndex> {
    let mut index = CalendarIndex::new(period(first, "first")?, values).map_err(to_r)?;
    if let Nullable::NotNull(rate) = rate {
        index = index.with_future_rate(rate).map_err(to_r)?;
    }
    Ok(index)
}

fn indexation(
    first: &str,
    values: Vec<f64>,
    rate: Nullable<f64>,
    base: &str,
) -> Result<Indexation> {
    let base = month(base, "base")?;
    let index = index(first, values, rate)?;
    // Fail now if the index does not cover the base period.
    index.value(base).map_err(to_r)?;
    Ok(Indexation { index, base })
}

/// The index's given periods, checking the index.
#[extendr]
fn calendar_index_periods_rust(
    first: &str,
    values: Vec<f64>,
    rate: Nullable<f64>,
) -> Result<Vec<String>> {
    Ok(index(first, values, rate)?
        .periods()
        .iter()
        .map(ToString::to_string)
        .collect())
}

/// The index at each of `months` ("YYYY-MM").
#[extendr]
fn calendar_index_value_rust(
    first: &str,
    values: Vec<f64>,
    rate: Nullable<f64>,
    months: Vec<String>,
) -> Result<Vec<f64>> {
    let index = index(first, values, rate)?;
    months
        .iter()
        .map(|m| index.value(month(m, "month")?).map_err(to_r))
        .collect()
}

/// Checks the indexation (the index covers the base period).
#[extendr]
fn indexation_check_rust(
    first: &str,
    values: Vec<f64>,
    rate: Nullable<f64>,
    base: &str,
) -> Result<String> {
    Ok(indexation(first, values, rate, base)?.base.to_string())
}

/// `I(p) / I(base)` at each of `months`.
#[extendr]
fn indexation_factor_rust(
    first: &str,
    values: Vec<f64>,
    rate: Nullable<f64>,
    base: &str,
    months: Vec<String>,
) -> Result<Vec<f64>> {
    let ix = indexation(first, values, rate, base)?;
    months
        .iter()
        .map(|m| ix.factor(month(m, "month")?).map_err(to_r))
        .collect()
}

/// `column` of the triangle restated at the base level, or (`inflate`)
/// taken from it to the money of the payment periods.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn indexation_apply_rust(
    triangle: &Triangle,
    column: &str,
    first: &str,
    values: Vec<f64>,
    rate: Nullable<f64>,
    base: &str,
    inflate: bool,
) -> Result<Triangle> {
    let ix = indexation(first, values, rate, base)?;
    let out = if inflate {
        ix.inflate(&triangle.inner, column)
    } else {
        ix.restate(&triangle.inner, column)
    };
    Ok(out.map_err(to_r)?.into())
}

/// Cells as the R side holds them: the origins and one entry per cell.
fn cells_from(
    origins: &[String],
    origin: &[String],
    age: &[i32],
    valuation: &[String],
    value: &[f64],
) -> Result<FutureCells> {
    let n = origin.len();
    if age.len() != n || valuation.len() != n || value.len() != n {
        return Err(Error::Other(
            "origin, age, valuation and value must have one entry per cell".into(),
        ));
    }
    let periods = origins
        .iter()
        .map(|s| period(s, "origins"))
        .collect::<Result<Vec<_>>>()?;
    let mut cells = Vec::with_capacity(n);
    for k in 0..n {
        let position = parse_period(&origin[k])
            .and_then(|p| periods.iter().position(|&q| q == p))
            .ok_or_else(|| {
                Error::Other(format!("origin {:?} is not one of the origins", origin[k]))
            })?;
        let age = u32::try_from(age[k])
            .map_err(|_| Error::Other(format!("age {} is not an age in months", age[k])))?;
        cells.push(FutureCell {
            origin: position,
            age,
            valuation: month(&valuation[k], "valuation")?,
            value: value[k],
        });
    }
    Ok(FutureCells {
        origins: periods,
        cells,
    })
}

/// The cells as a list of columns.
pub(crate) fn cells_list(cells: &FutureCells) -> List {
    let labels: Vec<String> = cells.origins.iter().map(ToString::to_string).collect();
    let c = &cells.cells;
    list!(
        origins = labels.clone(),
        origin = c
            .iter()
            .map(|x| labels[x.origin].clone())
            .collect::<Vec<_>>(),
        age = c.iter().map(|x| x.age as i32).collect::<Vec<_>>(),
        valuation = c
            .iter()
            .map(|x| x.valuation.to_string())
            .collect::<Vec<_>>(),
        value = c.iter().map(|x| x.value).collect::<Vec<_>>(),
        reserves = cells.reserves()
    )
}

/// A chain-ladder fit's future cells: the one segment chosen by `keys` =
/// `values` (none needed for a single segment).
#[extendr]
fn future_cells_chain_ladder_rust(
    fit: &ChainLadderFit,
    keys: Vec<String>,
    values: Vec<String>,
) -> Result<List> {
    if keys.len() != values.len() {
        return Err(Error::Other("one value is needed per key".into()));
    }
    let choice: Vec<(&str, &str)> = keys
        .iter()
        .zip(&values)
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let at = fit.inner.position(&choice).map_err(to_r)?;
    Ok(cells_list(&FutureCells::from_chain_ladder(
        &fit.inner.fits[at],
    )))
}

/// An ODP GLM fit's future cells and their means.
#[extendr]
fn future_cells_odp_glm_rust(fit: &OdpGlmFit) -> Result<List> {
    Ok(cells_list(
        &FutureCells::from_odp_glm(&fit.inner).map_err(to_r)?,
    ))
}

/// Checks cells and returns them as a list of columns, with the sum by
/// origin as `reserves`.
#[extendr]
fn future_cells_check_rust(
    origins: Vec<String>,
    origin: Vec<String>,
    age: Vec<i32>,
    valuation: Vec<String>,
    value: Vec<f64>,
) -> Result<List> {
    Ok(cells_list(&cells_from(
        &origins, &origin, &age, &valuation, &value,
    )?))
}

/// The cells re-inflated: the cells' columns, `reserves` and
/// `base_reserves` by origin.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn reinflate_rust(
    first: &str,
    values: Vec<f64>,
    rate: Nullable<f64>,
    base: &str,
    origins: Vec<String>,
    origin: Vec<String>,
    age: Vec<i32>,
    valuation: Vec<String>,
    value: Vec<f64>,
) -> Result<List> {
    let ix = indexation(first, values, rate, base)?;
    let cells = cells_from(&origins, &origin, &age, &valuation, &value)?;
    let out: Reinflated = ix.reinflate(&cells).map_err(to_r)?;
    let cols = cells_list(&out.cells);
    Ok(list!(
        origins = cols.elt(0)?,
        origin = cols.elt(1)?,
        age = cols.elt(2)?,
        valuation = cols.elt(3)?,
        value = cols.elt(4)?,
        reserves = cols.elt(5)?,
        base_reserves = out.base_reserves.clone()
    ))
}

extendr_module! {
    mod reserving_inflation;
    fn calendar_index_periods_rust;
    fn calendar_index_value_rust;
    fn indexation_check_rust;
    fn indexation_factor_rust;
    fn indexation_apply_rust;
    fn future_cells_chain_ladder_rust;
    fn future_cells_odp_glm_rust;
    fn future_cells_check_rust;
    fn reinflate_rust;
}
