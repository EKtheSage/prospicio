//! A latest diagonal (a bordereau) projected with a selected pattern: the
//! chain ladder and the expected-loss methods on one value per origin at
//! its own age, over `prospicio_reserving::LatestDiagonal`
//! (`docs/design/xll.md`, decision 39; `reserving-v02.md` decisions 15 to
//! 18).
//!
//! Every function takes the origins, their ages or a valuation date and
//! their latest values, the pattern as values with their ages, and the
//! same seven optional arguments in one order: `pattern_ages`, `kind`,
//! `tail`, `grain`, `valuation`, `past_end` and `pattern_origin`.

use prospicio_core::{Grain, Lag, Month, Period};
use prospicio_reserving::{
    Benktander, BornhuetterFerguson, CapeCod, ChainLadder, ChainLadderFit, Development,
    ExpectedLoss, ExpectedLossFit, LatestDiagonal, PastPatternEnd, PatternKind, SelectedPattern,
    Tail,
};

use crate::ratemaking::civil;
use crate::reserving::{count, number, number_or, pattern_kind, tail, text_option};
use crate::sheet::{Grid, Value};

type Result<T> = std::result::Result<T, String>;

/// Serial numbers from here up are Excel dates, below it years (decision
/// 19's rule).
const FIRST_SERIAL: f64 = 10_000.0;

/// Where the crate's error for an origin past the pattern's end starts to
/// name its own settings; the add-in names the sheet's instead.
const PAST_END_CRATE: &str = ": an origin older than the pattern's end needs";

/// The add-in's ending of that error, in the sheet's arguments.
const PAST_END_SHEET: &str = ": an origin older than the pattern's end needs past_end \"none\" \
     (a factor of 1) or \"tail\" with a tail (a factor, or \"exponential\", \"inverse_power\" or \
     \"bondy\" fitted on the pattern's own factors)";

/// The cells of a row or column argument; `None` when it is omitted.
fn vector<'a>(grid: &'a Grid, name: &str) -> Result<Option<&'a [Value]>> {
    if grid.is_blank() {
        return Ok(None);
    }
    if grid.rows != 1 && grid.columns != 1 {
        return Err(format!("{name} must be a row or a column"));
    }
    Ok(Some(&grid.cells))
}

/// A grain argument: "year", "half" (or "semester"), "quarter" or
/// "month"; `None` when omitted.
fn grain_option(grid: &Grid, name: &str) -> Result<Option<Grain>> {
    match text_option(grid, name)?.as_deref() {
        None => Ok(None),
        Some("year") => Ok(Some(Grain::Year)),
        Some("half" | "semester") => Ok(Some(Grain::Semester)),
        Some("quarter") => Ok(Some(Grain::Quarter)),
        Some("month") => Ok(Some(Grain::Month)),
        Some(other) => Err(format!(
            "{name} must be \"year\", \"half\", \"quarter\" or \"month\", got {other:?}"
        )),
    }
}

/// The month of an Excel date (10,000 or more).
fn date_month(x: f64, what: &str) -> Result<Month> {
    let (y, m, _) = civil(x).map_err(|e| format!("{what}: {e}"))?;
    Month::new(y, m).map_err(|e| e.to_string())
}

/// A cell as an error message shows it.
fn shown(v: &Value) -> String {
    match v {
        Value::Number(x) => x.to_string(),
        Value::Text(s) => format!("{s:?}"),
        Value::Empty => "a blank".into(),
        other => format!("{other:?}"),
    }
}

/// The `origins` argument: a year (a whole number below 10,000) or an
/// Excel date per origin.
fn origins(grid: &Grid) -> Result<Vec<Month>> {
    let Some(cells) = vector(grid, "origins")? else {
        return Err("origins is required: a year or an Excel date per origin".into());
    };
    cells
        .iter()
        .enumerate()
        .map(|(i, v)| match v {
            Value::Number(x) if *x >= FIRST_SERIAL => date_month(*x, &format!("origins {}", i + 1)),
            Value::Number(x) if x.fract() == 0.0 && *x >= 1.0 => Ok(Month::january(*x as i32)),
            other => Err(format!(
                "origins {} must be a year or an Excel date, got {}",
                i + 1,
                shown(other)
            )),
        })
        .collect()
}

/// A row or column argument with one cell per origin; `None` when omitted.
fn per_origin<'a>(grid: &'a Grid, name: &str, n: usize) -> Result<Option<&'a [Value]>> {
    let Some(cells) = vector(grid, name)? else {
        return Ok(None);
    };
    if cells.len() != n {
        return Err(format!(
            "{name} must have {n} values, one per origin, got {}",
            cells.len()
        ));
    }
    Ok(Some(cells))
}

/// A whole number of months, at least 1, as an age.
fn months(v: &Value) -> Option<Lag> {
    match v {
        Value::Number(x) if x.fract() == 0.0 && *x >= 1.0 && *x <= 1e6 => Some(*x as Lag),
        _ => None,
    }
}

/// The `ages` argument: a whole number of months, at least 1, per origin.
fn ages(cells: &[Value]) -> Result<Vec<Lag>> {
    cells
        .iter()
        .enumerate()
        .map(|(i, v)| {
            months(v).ok_or_else(|| {
                format!(
                    "ages {} must be a whole number of months from 1, got {}",
                    i + 1,
                    shown(v)
                )
            })
        })
        .collect()
}

/// The `latest` argument: a finite number per origin, 0 for one with
/// nothing reported.
fn latest(grid: &Grid, n: usize) -> Result<Vec<f64>> {
    let Some(cells) = per_origin(grid, "latest", n)? else {
        return Err("latest is required: the latest cumulative value per origin".into());
    };
    cells
        .iter()
        .enumerate()
        .map(|(i, v)| match v {
            Value::Number(x) if x.is_finite() => Ok(*x),
            v if v.is_blank() => Err(format!(
                "latest {} is blank: give 0 for an origin with nothing reported",
                i + 1
            )),
            other => Err(format!(
                "latest {} must be a number, got {}",
                i + 1,
                shown(other)
            )),
        })
        .collect()
}

/// The `exposure` argument: a positive number per origin.
fn exposure(grid: &Grid, n: usize) -> Result<Vec<f64>> {
    let Some(cells) = per_origin(grid, "exposure", n)? else {
        return Err(format!(
            "exposure is required: a row or column of {n} values, one per origin"
        ));
    };
    cells
        .iter()
        .enumerate()
        .map(|(i, v)| match v {
            Value::Number(x) if x.is_finite() && *x > 0.0 => Ok(*x),
            other => Err(format!(
                "exposure {} must be a positive number, got {}",
                i + 1,
                shown(other)
            )),
        })
        .collect()
}

/// The `pattern`, `pattern_ages` and `kind` arguments as a fully selected
/// pattern: each value at its age in months (12, 24, ... by cell when
/// `pattern_ages` is omitted). A pair with a blank value is skipped; a
/// value needs an age. An age-to-age factor runs from its age to the next
/// age given (`reserving-v02.md` decisions 16 and 18 item 4), so a skipped
/// value widens the factor before it: 1.5, blank, 1.2 at 12, 24, 36 is 1.5
/// from 12 to 36 months.
fn pattern(values: &Grid, ages_arg: &Grid, kind: &Grid) -> Result<SelectedPattern> {
    let kind = pattern_kind(kind)?;
    let Some(cells) = vector(values, "pattern")? else {
        return Err(
            "pattern is required: the selected pattern's values, such as a row of PRO.CL.LDF"
                .into(),
        );
    };
    let ages = match vector(ages_arg, "pattern_ages")? {
        Some(a) if a.len() != cells.len() => {
            return Err(format!(
                "pattern_ages must have {} cells, one per pattern value, got {}",
                cells.len(),
                a.len()
            ));
        }
        a => a,
    };
    let mut by_age = Vec::with_capacity(cells.len());
    for (i, v) in cells.iter().enumerate() {
        let x = match v {
            v if v.is_blank() => continue,
            Value::Number(x) => *x,
            other => {
                return Err(format!(
                    "pattern {} must be a number, got {}",
                    i + 1,
                    shown(other)
                ));
            }
        };
        let age = match ages.map(|a| &a[i]) {
            None => 12 * (i as Lag + 1),
            Some(a) => months(a).ok_or_else(|| {
                format!(
                    "pattern_ages {} must be a whole number of months from 1, got {}",
                    i + 1,
                    shown(a)
                )
            })?,
        };
        by_age.push((age, x));
    }
    Ok(SelectedPattern::new(kind, by_age))
}

/// The `past_end` argument: "reject" (default), "none" or "tail".
fn past_end(grid: &Grid) -> Result<PastPatternEnd> {
    match text_option(grid, "past_end")?.as_deref() {
        None | Some("reject") => Ok(PastPatternEnd::Reject),
        Some("none" | "no_development") => Ok(PastPatternEnd::NoDevelopment),
        Some("tail") => Ok(PastPatternEnd::Tail),
        Some(other) => Err(format!(
            "past_end must be \"reject\", \"none\" or \"tail\", got {other:?}"
        )),
    }
}

/// A tail that is a factor of 1, which the crate reads as no tail.
fn is_no_tail(t: &Tail) -> bool {
    matches!(t, Tail::Constant(c) if c.factor == 1.0 && c.attachment_age.is_none())
}

/// A crate error as the sheet shows it: the error for an origin past the
/// pattern's end names the sheet's `past_end` and `tail`, not the crate's
/// settings.
fn sheet_error(e: prospicio_reserving::Error) -> String {
    let s = e.to_string();
    match s.find(PAST_END_CRATE) {
        Some(i) => format!("{}{PAST_END_SHEET}", &s[..i]),
        None => s,
    }
}

/// What a diagonal function fits: the diagonal and the chain ladder that
/// holds its pattern and tail.
struct Prepared {
    diagonal: LatestDiagonal,
    chain_ladder: ChainLadder,
}

/// The arguments every diagonal function takes: `origins`, `ages` and
/// `latest` at `args[0..3]`, `exposure` when given, `pattern` at
/// `args[pattern_at]` and the seven shared optional arguments from
/// `args[shared]`.
fn prepare(
    args: &[Grid],
    exposure_arg: Option<&Grid>,
    pattern_at: usize,
    shared: usize,
) -> Result<Prepared> {
    let [
        pattern_ages,
        kind,
        tail_arg,
        grain_arg,
        valuation,
        past_end_arg,
        pattern_origin,
    ]: [&Grid; 7] = std::array::from_fn(|k| &args[shared + k]);
    let grain = grain_option(grain_arg, "grain")?.unwrap_or(Grain::Year);
    let origins = origins(&args[0])?;
    let n = origins.len();
    let ages_cells = per_origin(&args[1], "ages", n)?;
    let latest = latest(&args[2], n)?;
    let diagonal = match (ages_cells, valuation.is_blank()) {
        (Some(_), false) => return Err("give ages or valuation, not both".into()),
        (None, true) => {
            return Err(
                "ages or valuation is required: each origin's age in months, or the date all are valued at"
                    .into(),
            );
        }
        (Some(cells), true) => LatestDiagonal::new(grain, &origins, &ages(cells)?, &latest),
        (None, false) => {
            let at = match valuation.as_scalar() {
                Some(Value::Number(x)) if *x >= FIRST_SERIAL => date_month(*x, "valuation")?,
                _ => {
                    return Err(
                        "valuation must be one Excel date, such as 45657 for 31 December 2024"
                            .into(),
                    );
                }
            };
            if let Some(o) = origins
                .iter()
                .position(|m| at.months_since(m.floor(grain)) < 0)
            {
                return Err(format!("origin {} starts after the valuation", o + 1));
            }
            LatestDiagonal::at_valuation(grain, &origins, at, &latest)
        }
    }
    .map_err(sheet_error)?;
    let diagonal = match exposure_arg {
        Some(e) => diagonal
            .with_exposure(&exposure(e, n)?)
            .map_err(sheet_error)?,
        None => diagonal,
    };
    let rule = past_end(past_end_arg)?;
    let tail = tail(tail_arg)?;
    match rule {
        PastPatternEnd::NoDevelopment if !is_no_tail(&tail) => {
            return Err(
                "past_end \"none\" is a factor of 1 past the pattern's end: leave tail empty, or use past_end \"tail\""
                    .into(),
            );
        }
        PastPatternEnd::Tail if is_no_tail(&tail) => {
            return Err(
                "past_end \"tail\" needs a tail: for a factor of 1 use past_end \"none\"".into(),
            );
        }
        _ => {}
    }
    let mut pattern = pattern(&args[pattern_at], pattern_ages, kind)?;
    if let Some(basis) = grain_option(pattern_origin, "pattern_origin")? {
        pattern = pattern
            .averaged_over_origins(basis, grain)
            .map_err(sheet_error)?;
    }
    Ok(Prepared {
        diagonal: diagonal.with_past_pattern_end(rule),
        chain_ladder: ChainLadder {
            development: Development {
                pattern: Some(pattern),
                ..Default::default()
            },
            tail,
        },
    })
}

/// An origin's label: the year for accident years, so a lookup against a
/// column of years finds it, else the period as text (`2024Q3`).
fn label(grain: Grain, p: &Period) -> Value {
    match grain {
        Grain::Year => Value::Number(f64::from(p.start().year())),
        _ => Value::Text(p.to_string()),
    }
}

/// The chain ladder's diagonal exhibit: Origin, Age, Latest, CDF, Ultimate,
/// Reserve per origin, oldest first, then a total row.
fn chain_ladder_exhibit(fit: &ChainLadderFit, grain: Grain) -> Grid {
    let reserves = fit.reserves();
    let mut rows = vec![
        ["Origin", "Age", "Latest", "CDF", "Ultimate", "Reserve"]
            .map(Value::from)
            .to_vec(),
    ];
    for (i, p) in fit.origins.iter().enumerate() {
        let d = fit.latest_position[i];
        rows.push(vec![
            label(grain, p),
            Value::Number(f64::from(fit.development.development[d])),
            Value::number(fit.latest[i]),
            Value::number(fit.cdf[d]),
            Value::number(fit.ultimate[i]),
            Value::number(reserves[i]),
        ]);
    }
    rows.push(vec![
        Value::from("Total"),
        Value::Empty,
        Value::number(fit.latest.iter().sum()),
        Value::Empty,
        Value::number(fit.total_ultimate()),
        Value::number(fit.total_reserve()),
    ]);
    Grid::from_rows(rows)
}

/// The expected-loss methods' diagonal exhibit: Origin, Age, Exposure,
/// Latest, CDF, A priori, Ultimate and Reserve per origin, then a total
/// row.
fn expected_loss_exhibit(fit: &ExpectedLossFit, grain: Grain) -> Grid {
    let cl = &fit.chain_ladder;
    let reserves = fit.reserves();
    let mut rows = vec![
        [
            "Origin", "Age", "Exposure", "Latest", "CDF", "A priori", "Ultimate", "Reserve",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for (i, p) in cl.origins.iter().enumerate() {
        let d = cl.latest_position[i];
        rows.push(vec![
            label(grain, p),
            Value::Number(f64::from(cl.development.development[d])),
            Value::number(fit.exposure[i]),
            Value::number(cl.latest[i]),
            Value::number(cl.cdf[d]),
            Value::number(fit.apriori[i]),
            Value::number(fit.ultimate[i]),
            Value::number(reserves[i]),
        ]);
    }
    rows.push(vec![
        Value::from("Total"),
        Value::Empty,
        Value::number(fit.exposure.iter().sum()),
        Value::number(cl.latest.iter().sum()),
        Value::Empty,
        Value::Empty,
        Value::number(fit.total_ultimate()),
        Value::number(fit.total_reserve()),
    ]);
    Grid::from_rows(rows)
}

/// The chain ladder on a latest diagonal:
/// `PRO.DIAG.CL(origins, ages, latest, pattern, [pattern_ages], [kind], [tail], [grain], [valuation], [past_end], [pattern_origin])`.
pub fn chain_ladder(args: &[Grid]) -> Result<Grid> {
    let p = prepare(args, None, 3, 4)?;
    let fit = p
        .chain_ladder
        .fit_diagonal(&p.diagonal)
        .map_err(sheet_error)?;
    Ok(chain_ladder_exhibit(&fit, p.diagonal.origin_grain()))
}

/// The expected loss method on a latest diagonal:
/// `PRO.DIAG.EL(origins, ages, latest, exposure, apriori, pattern, ...)`,
/// the seven shared arguments after `pattern`.
pub fn expected_loss(args: &[Grid]) -> Result<Grid> {
    let p = prepare(args, Some(&args[3]), 5, 6)?;
    let fit = ExpectedLoss {
        apriori: number(&args[4], "apriori")?,
        chain_ladder: p.chain_ladder,
    }
    .fit_diagonal(&p.diagonal)
    .map_err(sheet_error)?;
    Ok(expected_loss_exhibit(&fit, p.diagonal.origin_grain()))
}

/// Bornhuetter-Ferguson on a latest diagonal:
/// `PRO.DIAG.BF(origins, ages, latest, exposure, apriori, pattern, ...)`.
pub fn bornhuetter_ferguson(args: &[Grid]) -> Result<Grid> {
    let p = prepare(args, Some(&args[3]), 5, 6)?;
    let fit = BornhuetterFerguson {
        apriori: number(&args[4], "apriori")?,
        chain_ladder: p.chain_ladder,
    }
    .fit_diagonal(&p.diagonal)
    .map_err(sheet_error)?;
    Ok(expected_loss_exhibit(&fit, p.diagonal.origin_grain()))
}

/// Benktander on a latest diagonal:
/// `PRO.DIAG.BENKTANDER(origins, ages, latest, exposure, apriori, pattern, [iterations], ...)`.
pub fn benktander(args: &[Grid]) -> Result<Grid> {
    let p = prepare(args, Some(&args[3]), 5, 7)?;
    let fit = Benktander {
        apriori: number(&args[4], "apriori")?,
        n_iters: count(&args[6], "iterations", 1, 10_000)?,
        chain_ladder: p.chain_ladder,
    }
    .fit_diagonal(&p.diagonal)
    .map_err(sheet_error)?;
    Ok(expected_loss_exhibit(&fit, p.diagonal.origin_grain()))
}

/// Cape Cod on a latest diagonal, trended to the diagonal's valuation (the
/// latest month any origin is valued at):
/// `PRO.DIAG.CAPECOD(origins, ages, latest, exposure, pattern, [trend], [decay], ...)`.
pub fn cape_cod(args: &[Grid]) -> Result<Grid> {
    let p = prepare(args, Some(&args[3]), 4, 7)?;
    let fit = CapeCod {
        trend: number_or(&args[5], "trend", 0.0)?,
        decay: number_or(&args[6], "decay", 1.0)?,
        chain_ladder: p.chain_ladder,
    }
    .fit_diagonal(&p.diagonal)
    .map_err(sheet_error)?;
    Ok(expected_loss_exhibit(
        &fit.expected_loss,
        p.diagonal.origin_grain(),
    ))
}

/// A pattern averaged from the origin basis it was measured on to a
/// coarser one (`SelectedPattern::averaged_over_origins`): Age, Percent
/// developed and CDF for every month of the result. Fed back to the
/// diagonal functions as `pattern_ages` and `pattern` with kind "percent",
/// it gives what their `pattern_origin` gives:
/// `PRO.PATTERN.AVERAGE(pattern_ages, pattern, pattern_origin, [grain], [kind])`.
pub fn pattern_average(args: &[Grid]) -> Result<Grid> {
    let given = pattern(&args[1], &args[0], &args[4])?;
    let Some(basis) = grain_option(&args[2], "pattern_origin")? else {
        return Err(
            "pattern_origin is required: the origin basis the pattern was measured on, such as \"month\""
                .into(),
        );
    };
    let grain = grain_option(&args[3], "grain")?.unwrap_or(Grain::Year);
    if basis == grain {
        return Err(
            "pattern_origin and grain are the same: the pattern is used as given, with nothing to average"
                .into(),
        );
    }
    let averaged = given
        .averaged_over_origins(basis, grain)
        .map_err(sheet_error)?;
    if averaged.kind != PatternKind::PercentDeveloped {
        return Err("internal error: an averaged pattern is percents developed".into());
    }
    let mut rows = vec![
        ["Age", "Percent developed", "CDF"]
            .map(Value::from)
            .to_vec(),
    ];
    for &(age, p) in &averaged.values {
        rows.push(vec![
            Value::Number(f64::from(age)),
            Value::number(p),
            Value::number(1.0 / p),
        ]);
    }
    Ok(Grid::from_rows(rows))
}
