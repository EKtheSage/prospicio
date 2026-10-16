//! Ratemaking (`docs/design/xll.md`): dates on Werner and Modlin's
//! half-month grid, premium at the current rate level by the
//! parallelogram method, the average written and earned dates that set
//! trend lengths, fitted trends, the loss ratio and pure premium
//! methods' experience and the indicated rate with its complements, over
//! `prospicio_pricing::ratemaking`.
//!
//! A date argument is a year fraction (`2011.5` is 1 July 2011) or an
//! Excel date; a number of 10,000 or more is an Excel date (after 1927)
//! and is put on the half-month grid as `PRO.YEARFRACTION` does (decision
//! 19).

use prospicio_pricing::ratemaking::{
    Applies, ExpenseCategory, LossRatioExperience, LossRatioYear, Period, Provisions,
    PurePremiumExperience, PurePremiumYear, RateChange, RateHistory, TrendFit,
    credibility_weighted, premium_based_expenses, trended_present_rates, year_fraction,
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

/// A factor per year: omitted for 1, one value for every year, or one
/// per year.
fn factors(g: &Grid, name: &str, n: usize) -> Result<Vec<f64>> {
    if g.is_blank() {
        return Ok(vec![1.0; n]);
    }
    each(g, name, n, "year")
}

/// One value for all `n`, or one per `unit`.
fn each(g: &Grid, name: &str, n: usize, unit: &str) -> Result<Vec<f64>> {
    if g.cells.len() == 1 {
        return Ok(vec![scalar(g, name)?; n]);
    }
    let v = numbers(g, name)?;
    if v.len() == n && g.cells.len() == n {
        Ok(v)
    } else {
        Err(format!(
            "{name} must be one value, or one per {unit} ({n}), got {} cells",
            g.cells.len()
        ))
    }
}

/// Per-year values, one per year, all present.
fn per_year(g: &Grid, name: &str, n: Option<usize>) -> Result<Vec<f64>> {
    let v = numbers(g, name)?;
    if v.len() != g.cells.len() {
        return Err(format!("{name} must have a number in every cell"));
    }
    match n {
        Some(n) if v.len() != n => Err(format!(
            "{name} must have one value per year ({n}), got {}",
            v.len()
        )),
        _ => Ok(v),
    }
}

/// The loss ratio method's experience (Werner and Modlin, ch. 8): each
/// year's premium at the current rate level, trended, and its losses
/// developed, trended and loaded for LAE, its loss ratio, and the total
/// (the ratio of the totals): `PRO.LR.EXPERIENCE(earned_premium, losses,
/// [on_level], [premium_trend], [development], [loss_trend], [lae])`. A
/// factor is omitted (1), one for every year, or one per year.
pub fn lr_experience(args: &[Grid]) -> Result<Grid> {
    let premium = per_year(&args[0], "earned_premium", None)?;
    let n = premium.len();
    let losses = per_year(&args[1], "losses", Some(n))?;
    let olf = factors(&args[2], "on_level", n)?;
    let pt = factors(&args[3], "premium_trend", n)?;
    let dev = factors(&args[4], "development", n)?;
    let lt = factors(&args[5], "loss_trend", n)?;
    let lae = factors(&args[6], "lae", n)?;
    let years: Vec<LossRatioYear> = (0..n)
        .map(|k| LossRatioYear {
            earned_premium: premium[k],
            on_level_factor: olf[k],
            premium_trend: pt[k],
            losses: losses[k],
            development: dev[k],
            loss_trend: lt[k],
            lae_factor: lae[k],
        })
        .collect();
    let e = LossRatioExperience::new(&years).map_err(|e| e.to_string())?;
    let mut rows = vec![
        ["Projected premium", "Projected losses", "Loss ratio"]
            .map(Value::from)
            .to_vec(),
    ];
    for y in e.years() {
        rows.push(vec![
            Value::number(y.projected_premium()),
            Value::number(y.projected_losses()),
            Value::number(y.loss_ratio()),
        ]);
    }
    rows.push(vec![
        Value::number(e.projected_premium()),
        Value::number(e.projected_losses()),
        Value::number(e.loss_ratio()),
    ]);
    Ok(Grid::from_rows(rows))
}

/// The pure premium method's experience: each year's exposure, losses
/// developed, trended and loaded for LAE, and pure premium, then the total:
/// `PRO.PP.EXPERIENCE(exposure, losses, [development], [loss_trend],
/// [lae])`.
pub fn pp_experience(args: &[Grid]) -> Result<Grid> {
    let exposure = per_year(&args[0], "exposure", None)?;
    let n = exposure.len();
    let losses = per_year(&args[1], "losses", Some(n))?;
    let dev = factors(&args[2], "development", n)?;
    let lt = factors(&args[3], "loss_trend", n)?;
    let lae = factors(&args[4], "lae", n)?;
    let years: Vec<PurePremiumYear> = (0..n)
        .map(|k| PurePremiumYear {
            exposure: exposure[k],
            losses: losses[k],
            development: dev[k],
            loss_trend: lt[k],
            lae_factor: lae[k],
        })
        .collect();
    let e = PurePremiumExperience::new(&years).map_err(|e| e.to_string())?;
    let mut rows = vec![
        ["Exposure", "Projected losses", "Pure premium"]
            .map(Value::from)
            .to_vec(),
    ];
    for y in e.years() {
        rows.push(vec![
            Value::number(y.exposure),
            Value::number(y.projected_losses()),
            Value::number(y.pure_premium()),
        ]);
    }
    rows.push(vec![
        Value::number(e.exposure()),
        Value::number(e.projected_losses()),
        Value::number(e.pure_premium()),
    ]);
    Ok(Grid::from_rows(rows))
}

fn provisions(variable: &Grid, profit: &Grid) -> Result<Provisions> {
    Provisions::new(scalar(variable, "variable")?, scalar(profit, "profit")?)
        .map_err(|e| e.to_string())
}

/// An optional ratio or amount, 0 if omitted.
fn or_zero(g: &Grid, name: &str) -> Result<f64> {
    if g.is_blank() {
        Ok(0.0)
    } else {
        scalar(g, name)
    }
}

/// The loss ratio method's indicated change, `(LR + F) / (1 - V - Q) - 1`:
/// `PRO.INDICATED.CHANGE(loss_ratio, variable, profit, [fixed_ratio])`.
pub fn indicated_change(args: &[Grid]) -> Result<Grid> {
    let p = provisions(&args[1], &args[2])?;
    let fixed = or_zero(&args[3], "fixed_ratio")?;
    broadcast(&[&args[0]], &["loss_ratio"], &[false], |v| {
        Ok(p.loss_ratio_change(v[0], fixed))
    })
}

/// The pure premium method's indicated rate, `(PP + fixed) / (1 - V -
/// Q)`: `PRO.INDICATED.RATE(pure_premium, variable, profit,
/// [fixed_per_exposure])`.
pub fn indicated_rate(args: &[Grid]) -> Result<Grid> {
    let p = provisions(&args[1], &args[2])?;
    let fixed = or_zero(&args[3], "fixed_per_exposure")?;
    broadcast(&[&args[0]], &["pure_premium"], &[false], |v| {
        Ok(p.pure_premium_rate(v[0], fixed))
    })
}

/// The permissible loss ratio, `1 - F - V - Q`:
/// `PRO.PERMISSIBLE.LR(variable, profit, [fixed_ratio])`.
pub fn permissible_lr(args: &[Grid]) -> Result<Grid> {
    let p = provisions(&args[0], &args[1])?;
    let fixed = or_zero(&args[2], "fixed_ratio")?;
    Ok(Grid::scalar(Value::number(p.permissible_loss_ratio(fixed))))
}

/// The trended present rates complement as an indicated change,
/// `(1 + prior indicated) / (1 + prior implemented) * trend - 1`:
/// `PRO.TRENDED.PRESENT(prior_indicated, prior_implemented, trend)`.
pub fn trended_present(args: &[Grid]) -> Result<Grid> {
    broadcast(
        &[&args[0], &args[1], &args[2]],
        &["prior_indicated", "prior_implemented", "trend"],
        &[false; 3],
        |v| {
            if v[1] <= -1.0 {
                return Err("prior_implemented must be above -1".into());
            }
            Ok(trended_present_rates(v[0], v[1], v[2]))
        },
    )
}

/// `Z * estimate + (1 - Z) * complement`:
/// `PRO.CRED.WEIGHTED(estimate, credibility, complement)`.
pub fn cred_weighted(args: &[Grid]) -> Result<Grid> {
    broadcast(
        &[&args[0], &args[1], &args[2]],
        &["estimate", "credibility", "complement"],
        &[false; 3],
        |v| {
            if (0.0..=1.0).contains(&v[1]) {
                Ok(credibility_weighted(v[0], v[1], v[2]))
            } else {
                Err(format!("credibility must be in [0, 1], got {}", v[1]))
            }
        },
    )
}

/// The premium-based projection's fixed and variable expense ratios: each
/// category's ratio split by its fixed share, summed:
/// `PRO.EXPENSES(ratios, fixed_shares)`.
pub fn expenses(args: &[Grid]) -> Result<Grid> {
    let ratios = per_year(&args[0], "ratios", None)?;
    let shares = factors(&args[1], "fixed_shares", ratios.len())
        .map_err(|e| e.replace("per year", "per ratio"))?;
    let shares = if args[1].is_blank() {
        vec![0.0; ratios.len()]
    } else {
        shares
    };
    let categories: Vec<ExpenseCategory> = ratios
        .iter()
        .zip(&shares)
        .map(|(&ratio, &fixed_share)| ExpenseCategory { ratio, fixed_share })
        .collect();
    let (fixed, variable) = premium_based_expenses(&categories).map_err(|e| e.to_string())?;
    Ok(Grid::from_rows(vec![
        vec![Value::from("Fixed"), Value::number(fixed)],
        vec![Value::from("Variable"), Value::number(variable)],
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
