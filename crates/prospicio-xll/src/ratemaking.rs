//! Ratemaking, part one (`docs/design/xll.md`): dates on Werner and
//! Modlin's half-month grid, premium at the current rate level by the
//! parallelogram method, the average written and earned dates that set
//! trend lengths, and fitted trends, over
//! `prospicio_pricing::ratemaking`.
//!
//! A date argument is a year fraction (`2011.5` is 1 July 2011) or an
//! Excel date; a number of 10,000 or more is an Excel date (after 1927)
//! and is put on the half-month grid as `PRO.YEARFRACTION` does (decision
//! 19).

use prospicio_pricing::ratemaking::{
    Applies, Period, RateChange, RateHistory, TrendFit, year_fraction,
};

use crate::exposure::broadcast;
use crate::pricing::{Result, numbers, scalar};
use crate::sheet::{Grid, Value};

/// Serial numbers from here up are Excel dates, below it year fractions.
const FIRST_SERIAL: f64 = 10_000.0;

/// The civil date of an Excel serial in the 1900 date system (serial 1
/// is 1 January 1900; Excel's phantom 29 February 1900, serial 60, is not
/// a date).
fn civil(serial: f64) -> Result<(i32, u8, u8)> {
    if !(61.0..2_958_466.0).contains(&serial) {
        return Err(format!("{serial} is not an Excel date after February 1900"));
    }
    // Days since 1970-01-01: serial 25569 is that day.
    let z = serial.floor() as i64 - 25_569 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u8;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u8;
    let year = (yoe + era * 400 + i64::from(month <= 2)) as i32;
    Ok((year, month, day))
}

/// A date argument as a year fraction: an Excel date (10,000 or more) on
/// the half-month grid, anything else as it is.
pub(crate) fn date(x: f64) -> Result<f64> {
    if x >= FIRST_SERIAL {
        let (y, m, d) = civil(x)?;
        year_fraction(y, m, d).map_err(|e| e.to_string())
    } else {
        Ok(x)
    }
}

/// Each Excel date on Werner and Modlin's half-month grid as a year
/// fraction: `PRO.YEARFRACTION(date)`. 30 June is 1 July (2013.5), 15
/// November is half way through it (2011.875), 31 December is the next 1
/// January.
pub fn year_fraction_of(args: &[Grid]) -> Result<Grid> {
    broadcast(&[&args[0]], &["date"], &[false], |v| {
        if v[0] < FIRST_SERIAL {
            return Err(format!(
                "date must be an Excel date (a serial from {FIRST_SERIAL}), got {}",
                v[0]
            ));
        }
        date(v[0])
    })
}

/// The rate changes on the sheet: effective dates, the changes (or the
/// cumulative levels when `given` is `"levels"`) and how each applies.
fn history(effective: &Grid, changes: &Grid, applies: &Grid, given: &Grid) -> Result<RateHistory> {
    let effective: Vec<f64> = numbers(effective, "effective")?
        .into_iter()
        .map(date)
        .collect::<Result<_>>()?;
    let values = numbers(changes, "changes")?;
    let n = effective.len();
    if values.len() != n {
        return Err(format!(
            "changes must have one value per effective date ({n}), got {}",
            values.len()
        ));
    }
    let applies = if applies.is_blank() {
        vec![Applies::Written; n]
    } else {
        let cells: Vec<&Value> = applies.cells.iter().collect();
        if cells.len() != 1 && cells.len() != n {
            return Err(format!(
                "applies must be one value or one per change ({n}), got {}",
                cells.len()
            ));
        }
        (0..n)
            .map(|k| match cells[if cells.len() == 1 { 0 } else { k }] {
                v if v.is_blank() => Ok(Applies::Written),
                Value::Text(s) => match s
                    .trim()
                    .to_ascii_lowercase()
                    .replace([' ', '-'], "_")
                    .as_str()
                {
                    "written" => Ok(Applies::Written),
                    "in_force" | "inforce" => Ok(Applies::InForce),
                    other => Err(format!(
                        "applies must be \"written\" or \"in_force\", got {other:?}"
                    )),
                },
                other => Err(format!(
                    "applies must be \"written\" or \"in_force\", got {other:?}"
                )),
            })
            .collect::<Result<_>>()?
    };
    let levels = match given.as_scalar() {
        Some(v) if v.is_blank() => false,
        Some(Value::Text(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "changes" => false,
            "levels" => true,
            other => {
                return Err(format!(
                    "given must be \"changes\" or \"levels\", got {other:?}"
                ));
            }
        },
        _ => return Err("given must be \"changes\" or \"levels\"".into()),
    };
    let e = |e: prospicio_core::Error| e.to_string();
    if levels {
        RateHistory::from_levels(&effective, &values, &applies).map_err(e)
    } else {
        let changes: Vec<RateChange> = effective
            .iter()
            .zip(&values)
            .zip(&applies)
            .map(|((&effective, &change), &applies)| RateChange {
                effective,
                change,
                applies,
            })
            .collect();
        RateHistory::new(&changes).map_err(e)
    }
}

/// `"calendar"` (the default: premium earned in the window) or `"policy"`
/// (policies written in it).
fn period_kind(g: &Grid) -> Result<bool> {
    match g.as_scalar() {
        Some(v) if v.is_blank() => Ok(false),
        Some(Value::Text(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "calendar" | "accident" => Ok(false),
            "policy" => Ok(true),
            other => Err(format!(
                "period must be \"calendar\" or \"policy\", got {other:?}"
            )),
        },
        _ => Err("period must be \"calendar\" or \"policy\"".into()),
    }
}

fn period(policy: bool, from: f64, to: f64) -> Result<Period> {
    let (from, to) = (date(from)?, date(to)?);
    if policy {
        Period::policy(from, to)
    } else {
        Period::calendar(from, to)
    }
    .map_err(|e| e.to_string())
}

/// The policy term in years, 1 if omitted.
fn term(g: &Grid) -> Result<f64> {
    if g.is_blank() {
        return Ok(1.0);
    }
    let t = scalar(g, "term")?;
    if t > 0.0 && t.is_finite() {
        Ok(t)
    } else {
        Err(format!("term must be positive, in years, got {t}"))
    }
}

/// The factor that brings each period's premium to the current rate level
/// (current level over the period's average level, by the parallelogram
/// method): `PRO.ONLEVEL(effective, changes, from, to, [period], [term],
/// [applies], [given])`. `from` and `to` broadcast, so columns of period
/// starts and ends give a column of factors.
pub fn on_level(args: &[Grid]) -> Result<Grid> {
    let h = history(&args[0], &args[1], &args[6], &args[7])?;
    let policy = period_kind(&args[4])?;
    let term = term(&args[5])?;
    broadcast(
        &[&args[2], &args[3]],
        &["from", "to"],
        &[false, false],
        |v| {
            let p = period(policy, v[0], v[1])?;
            h.rate_level(&p, term)
                .map(|l| l.on_level_factor())
                .map_err(|e| e.to_string())
        },
    )
}

/// One period's premium by rate level, the parallelogram's cells: each
/// cell's level and share, then the average and current levels and the
/// on-level factor: `PRO.RATELEVEL(effective, changes, from, to, [period],
/// [term], [applies], [given])`.
pub fn rate_level(args: &[Grid]) -> Result<Grid> {
    let h = history(&args[0], &args[1], &args[6], &args[7])?;
    let p = period(
        period_kind(&args[4])?,
        scalar(&args[2], "from")?,
        scalar(&args[3], "to")?,
    )?;
    let l = h
        .rate_level(&p, term(&args[5])?)
        .map_err(|e| e.to_string())?;
    let mut rows = vec![
        ["Written changes", "In-force changes", "Level", "Share"]
            .map(Value::from)
            .to_vec(),
    ];
    for c in &l.portions {
        rows.push(vec![
            Value::Number(c.written_changes as f64),
            Value::Number(c.in_force_changes as f64),
            Value::number(c.level),
            Value::number(c.share),
        ]);
    }
    for (label, x) in [
        ("Average level", l.average_level),
        ("Current level", l.current_level),
        ("On-level factor", l.on_level_factor()),
    ] {
        rows.push(vec![
            Value::from(label),
            Value::Empty,
            Value::number(x),
            Value::Empty,
        ]);
    }
    Ok(Grid::from_rows(rows))
}

/// The average date on which each period's premium is written or earned
/// (the average accident date of its losses), which sets trend lengths:
/// `PRO.AVGDATE(from, to, [period], [term], [basis])`, `basis`
/// `"earned"` (default) or `"written"`. A trend factor is then
/// `(1 + trend) ^ (to_date - from_date)`.
pub fn average_date(args: &[Grid]) -> Result<Grid> {
    let policy = period_kind(&args[2])?;
    let term = term(&args[3])?;
    let written = match args[4].as_scalar() {
        Some(v) if v.is_blank() => false,
        Some(Value::Text(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "earned" | "accident" => false,
            "written" => true,
            other => {
                return Err(format!(
                    "basis must be \"earned\" or \"written\", got {other:?}"
                ));
            }
        },
        _ => return Err("basis must be \"earned\" or \"written\"".into()),
    };
    broadcast(
        &[&args[0], &args[1]],
        &["from", "to"],
        &[false, false],
        |v| {
            let p = period(policy, v[0], v[1])?;
            Ok(if written {
                p.average_written_date(term)
            } else {
                p.average_earned_date(term)
            })
        },
    )
}

/// A trend fitted by least squares to `values` at `times` (in years, or
/// Excel dates): `PRO.TREND.FIT(times, values, [kind])`, `kind`
/// `"exponential"` (default) or `"linear"`. Returns the annual change
/// (`exp(slope) - 1`, or the slope for a linear fit), intercept, slope,
/// R-squared and the number of points; blank pairs are skipped.
pub fn trend_fit(args: &[Grid]) -> Result<Grid> {
    let (t, y) = (&args[0], &args[1]);
    if t.cells.len() != y.cells.len() {
        return Err(format!(
            "times and values must have the same number of cells, got {} and {}",
            t.cells.len(),
            y.cells.len()
        ));
    }
    let (mut times, mut values) = (Vec::new(), Vec::new());
    for (k, (a, b)) in t.cells.iter().zip(&y.cells).enumerate() {
        match (a, b) {
            (Value::Number(a), Value::Number(b)) => {
                times.push(date(*a)?);
                values.push(*b);
            }
            (a, b) if a.is_blank() || b.is_blank() => {}
            _ => return Err(format!("times and values must be numbers (pair {})", k + 1)),
        }
    }
    let exponential = match args[2].as_scalar() {
        Some(v) if v.is_blank() => true,
        Some(Value::Text(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "exponential" => true,
            "linear" => false,
            other => {
                return Err(format!(
                    "kind must be \"exponential\" or \"linear\", got {other:?}"
                ));
            }
        },
        _ => return Err("kind must be \"exponential\" or \"linear\"".into()),
    };
    let fit = if exponential {
        TrendFit::exponential(&times, &values)
    } else {
        TrendFit::linear(&times, &values)
    }
    .map_err(|e| e.to_string())?;
    Ok(Grid::from_rows(vec![
        vec![
            Value::from("Kind"),
            Value::from(if exponential { "exponential" } else { "linear" }),
        ],
        vec![
            Value::from("Annual change"),
            Value::number(fit.annual_change()),
        ],
        vec![Value::from("Intercept"), Value::number(fit.intercept)],
        vec![Value::from("Slope"), Value::number(fit.slope)],
        vec![Value::from("R-squared"), Value::number(fit.r_squared)],
        vec![Value::from("Points"), Value::Number(fit.n as f64)],
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn excel_serials_are_civil_dates() {
        assert_eq!(civil(61.0).unwrap(), (1900, 3, 1));
        assert_eq!(civil(25_569.0).unwrap(), (1970, 1, 1));
        assert_eq!(civil(41_455.0).unwrap(), (2013, 6, 30));
        assert_eq!(civil(40_862.0).unwrap(), (2011, 11, 15));
        assert_eq!(civil(42_369.0).unwrap(), (2015, 12, 31));
        assert_eq!(civil(43_890.0).unwrap(), (2020, 2, 29));
        assert!(civil(60.0).is_err());
        assert_eq!(date(41_455.0).unwrap(), 2013.5);
        assert_eq!(date(2011.25).unwrap(), 2011.25);
    }
}
