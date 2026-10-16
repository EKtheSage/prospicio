//! The reserving functions: the chain ladder, Mack's model and the ODP
//! bootstrap on a triangle laid out on the sheet (`docs/design/xll.md`).
//!
//! A triangle argument is a range of cumulative values, one row per origin
//! (oldest first) and one column per development age (youngest first).
//! Blank cells, empty text and `#N/A` are unobserved; each row is observed
//! from its first column up to its latest value and blank after it. Ages
//! are read as development periods, so the same range serves a yearly,
//! quarterly or monthly triangle; only the labels a user puts around it
//! say which.

use prospicio_core::{Grain, Lag, Month};
use prospicio_prob::{Distribution, Sampled};
use prospicio_reserving::{
    Average, ChainLadder, ChainLadderFit, CurveShape, Development, DevelopmentColumn, Long, Mack,
    OdpBootstrap, OdpBootstrapFit, ProcessDistribution, Tail, TailBondy, TailCurve, Triangle,
};

use crate::sheet::{CellError, Grid, Value, position};

/// The measure name inside the triangle; never shown on the sheet.
const COLUMN: &str = "values";

type Result<T> = std::result::Result<T, String>;

/// The triangle laid out in `grid`.
pub fn triangle(grid: &Grid) -> Result<Triangle> {
    if grid.rows < 1 || grid.columns < 2 {
        return Err("the triangle needs at least two development columns".into());
    }
    let mut origin = Vec::new();
    let mut age: Vec<Lag> = Vec::new();
    let mut values = Vec::new();
    for r in 0..grid.rows {
        let mut ended = false;
        let mut observed = 0;
        for c in 0..grid.columns {
            let x = match grid.get(r, c) {
                Value::Number(x) => *x,
                v if v.is_blank() => {
                    ended = true;
                    continue;
                }
                Value::Error(CellError::NA) => {
                    ended = true;
                    continue;
                }
                Value::Error(e) => {
                    return Err(format!("triangle {} is {e}", position(r, c)));
                }
                Value::Text(s) => {
                    return Err(format!(
                        "triangle {} is text ({s:?}); select the values only, without labels",
                        position(r, c)
                    ));
                }
                Value::Bool(_) => {
                    return Err(format!("triangle {} is TRUE or FALSE", position(r, c)));
                }
                Value::Empty => unreachable!("blank handled above"),
            };
            if ended {
                return Err(format!(
                    "triangle {} has a value after a blank; each origin must be observed \
                     from the first column up to its latest value",
                    position(r, c)
                ));
            }
            origin.push(Month::january(2001 + r as i32));
            age.push(12 * (c as Lag + 1));
            values.push(x);
            observed += 1;
        }
        if observed == 0 {
            return Err(format!("triangle row {} has no values", r + 1));
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[(COLUMN, &values)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .map_err(|e| e.to_string())
}

/// The text of an optional argument, lower-cased; `None` when omitted.
fn text_option(grid: &Grid, name: &str) -> Result<Option<String>> {
    match grid.as_scalar() {
        Some(v) if v.is_blank() => Ok(None),
        Some(Value::Text(s)) => Ok(Some(s.trim().to_ascii_lowercase())),
        _ => Err(format!("{name} must be a single text value")),
    }
}

/// The `average` argument: "volume" (the default), "simple" or
/// "regression".
pub fn average(grid: &Grid) -> Result<Average> {
    match text_option(grid, "average")?.as_deref() {
        None | Some("volume") => Ok(Average::Volume),
        Some("simple") => Ok(Average::Simple),
        Some("regression") => Ok(Average::Regression),
        Some(other) => Err(format!(
            "average must be \"volume\", \"simple\" or \"regression\", got {other:?}"
        )),
    }
}

/// The `tail` argument: omitted for none, a number for a constant factor
/// to ultimate at the oldest age, or "exponential", "inverse_power" or
/// "bondy" for a fitted tail.
pub fn tail(grid: &Grid) -> Result<Tail> {
    match grid.as_scalar() {
        Some(v) if v.is_blank() => Ok(Tail::default()),
        Some(Value::Number(f)) if f.is_finite() && *f > 0.0 => Ok(Tail::from(*f)),
        Some(Value::Number(f)) => Err(format!("a tail factor must be positive, got {f}")),
        Some(Value::Text(s)) => match s
            .trim()
            .to_ascii_lowercase()
            .replace([' ', '-'], "_")
            .as_str()
        {
            "none" => Ok(Tail::default()),
            "exponential" => Ok(Tail::from(TailCurve::default())),
            "inverse_power" => Ok(Tail::from(TailCurve {
                curve: CurveShape::InversePower,
                ..TailCurve::default()
            })),
            "bondy" => Ok(Tail::from(TailBondy::default())),
            other => Err(format!(
                "tail must be a factor or \"exponential\", \"inverse_power\" or \"bondy\", \
                 got {other:?}"
            )),
        },
        _ => Err("tail must be a single number or text value".into()),
    }
}

/// The origin labels for a result table: the `origins` argument when
/// given, one cell per origin as a row or a column, else 1, 2, ...
fn origin_labels(grid: &Grid, n: usize) -> Result<Vec<Value>> {
    if grid.is_blank() {
        return Ok((1..=n).map(|i| Value::Number(i as f64)).collect());
    }
    if grid.cells.len() != n || (grid.rows != 1 && grid.columns != 1) {
        return Err(format!(
            "origins must be a row or column of {n} labels, one per triangle row"
        ));
    }
    Ok(grid.cells.clone())
}

/// Labels of each development link, "12-24" and on, then the tail's
/// "120-Ult", in development periods of 12.
fn link_labels(ages: usize) -> Vec<Value> {
    (1..=ages)
        .map(|k| {
            let to = if k == ages {
                "Ult".to_owned()
            } else {
                (12 * (k + 1)).to_string()
            };
            Value::Text(format!("{}-{to}", 12 * k))
        })
        .collect()
}

fn development(average: Average) -> Development {
    Development {
        average,
        ..Development::default()
    }
}

/// The chain-ladder fit of a triangle argument and its options.
fn chain_ladder(
    triangle_arg: &Grid,
    average_arg: &Grid,
    tail_arg: &Grid,
) -> Result<ChainLadderFit> {
    let tri = triangle(triangle_arg)?;
    ChainLadder {
        development: development(average(average_arg)?),
        tail: tail(tail_arg)?,
    }
    .fit(&tri, COLUMN)
    .map_err(|e| e.to_string())
}

/// Selected age-to-age factors, one per development column: the factors
/// within the triangle, then the tail factor to ultimate (1 without a
/// tail), so the row sits under the triangle's columns.
pub fn cl_ldf(args: &[Grid]) -> Result<Grid> {
    let fit = chain_ladder(&args[0], &args[1], &args[2])?;
    let mut ldf = fit.ldf().to_vec();
    ldf.push(fit.tail.factor);
    Ok(Grid::row(&ldf))
}

/// Factors from each development column's age to ultimate.
pub fn cl_cdf(args: &[Grid]) -> Result<Grid> {
    let fit = chain_ladder(&args[0], &args[1], &args[2])?;
    Ok(Grid::row(&fit.cdf))
}

/// Projected ultimate per origin, a column beside the triangle.
pub fn cl_ultimate(args: &[Grid]) -> Result<Grid> {
    let fit = chain_ladder(&args[0], &args[1], &args[2])?;
    Ok(Grid::column(&fit.ultimate))
}

/// Reserve (ultimate less latest, IBNR) per origin.
pub fn cl_reserve(args: &[Grid]) -> Result<Grid> {
    let fit = chain_ladder(&args[0], &args[1], &args[2])?;
    Ok(Grid::column(&fit.reserves()))
}

/// The chain-ladder exhibit: Origin, Latest, CDF, Ultimate, Reserve per
/// origin, then a total row.
pub fn cl_summary(args: &[Grid]) -> Result<Grid> {
    let fit = chain_ladder(&args[0], &args[1], &args[2])?;
    let labels = origin_labels(&args[3], fit.origins.len())?;
    let reserves = fit.reserves();
    let mut rows = vec![
        ["Origin", "Latest", "CDF", "Ultimate", "Reserve"]
            .map(Value::from)
            .to_vec(),
    ];
    for (i, label) in labels.into_iter().enumerate() {
        rows.push(vec![
            label,
            Value::number(fit.latest[i]),
            Value::number(fit.cdf[fit.latest_position[i]]),
            Value::number(fit.ultimate[i]),
            Value::number(reserves[i]),
        ]);
    }
    rows.push(vec![
        Value::from("Total"),
        Value::number(fit.latest.iter().sum()),
        Value::Empty,
        Value::number(fit.total_ultimate()),
        Value::number(fit.total_reserve()),
    ]);
    Ok(Grid::from_rows(rows))
}

/// The development exhibit: per link, the selected factor, the factor to
/// ultimate, Mack's sigma and the factor's standard error, then the tail.
pub fn cl_factors(args: &[Grid]) -> Result<Grid> {
    let fit = chain_ladder(&args[0], &args[1], &args[2])?;
    let ldf = fit.ldf();
    let n = ldf.len();
    let mut rows = vec![
        ["Link", "LDF", "CDF", "Sigma", "Std err"]
            .map(Value::from)
            .to_vec(),
    ];
    for (k, label) in link_labels(n + 1).into_iter().enumerate() {
        let (f, sigma, se) = if k < n {
            (ldf[k], fit.development.sigma[k], fit.development.std_err[k])
        } else {
            (fit.tail.factor, fit.tail.sigma, fit.tail.std_err)
        };
        rows.push(vec![
            label,
            Value::number(f),
            Value::number(fit.cdf[k]),
            Value::number(sigma),
            Value::number(se),
        ]);
    }
    Ok(Grid::from_rows(rows))
}

/// Mack's exhibit: Origin, Latest, Ultimate, Reserve, Process SE,
/// Parameter SE, Mack SE and CV per origin, then a total row.
pub fn mack_summary(args: &[Grid]) -> Result<Grid> {
    let tri = triangle(&args[0])?;
    let fit = Mack {
        development: development(average(&args[1])?),
        tail: tail(&args[2])?,
        ..Mack::default()
    }
    .fit(&tri, COLUMN)
    .map_err(|e| e.to_string())?;
    let cl = &fit.chain_ladder;
    let labels = origin_labels(&args[3], cl.origins.len())?;
    let reserves = cl.reserves();
    let cv = |se: f64, reserve: f64| {
        if reserve > 0.0 {
            Value::number(se / reserve)
        } else {
            Value::Empty
        }
    };
    let mut rows = vec![
        [
            "Origin",
            "Latest",
            "Ultimate",
            "Reserve",
            "Process SE",
            "Parameter SE",
            "Mack SE",
            "CV",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for (i, label) in labels.into_iter().enumerate() {
        rows.push(vec![
            label,
            Value::number(cl.latest[i]),
            Value::number(cl.ultimate[i]),
            Value::number(reserves[i]),
            Value::number(fit.process_risk[i]),
            Value::number(fit.parameter_risk[i]),
            Value::number(fit.standard_error[i]),
            cv(fit.standard_error[i], reserves[i]),
        ]);
    }
    rows.push(vec![
        Value::from("Total"),
        Value::number(cl.latest.iter().sum()),
        Value::number(cl.total_ultimate()),
        Value::number(cl.total_reserve()),
        Value::number(fit.total_process_risk),
        Value::number(fit.total_parameter_risk),
        Value::number(fit.total_standard_error),
        cv(fit.total_standard_error, cl.total_reserve()),
    ]);
    Ok(Grid::from_rows(rows))
}

/// The most simulations `PRO.ODP` runs: a recalculation should not hang
/// Excel for minutes.
pub const MAX_SIMS: usize = 1_000_000;

/// The percentiles `PRO.ODP` reports when none are given.
pub const PERCENTILES: [f64; 4] = [0.75, 0.9, 0.95, 0.995];

/// An optional whole-number argument within `1..=max`.
fn count(grid: &Grid, name: &str, default: usize, max: usize) -> Result<usize> {
    match grid.as_scalar() {
        Some(v) if v.is_blank() => Ok(default),
        Some(Value::Number(x)) if x.fract() == 0.0 && *x >= 1.0 && *x <= max as f64 => {
            Ok(*x as usize)
        }
        _ => Err(format!("{name} must be a whole number from 1 to {max}")),
    }
}

/// The `seed` argument: a whole number from 0, default 0.
fn seed(grid: &Grid) -> Result<u64> {
    match grid.as_scalar() {
        Some(v) if v.is_blank() => Ok(0),
        Some(Value::Number(x)) if x.fract() == 0.0 && *x >= 0.0 && *x < 2f64.powi(53) => {
            Ok(*x as u64)
        }
        _ => Err("seed must be a whole number from 0".into()),
    }
}

/// The `percentiles` argument: probabilities strictly between 0 and 1, as
/// a row or a column; blanks are skipped.
fn percentiles(grid: &Grid) -> Result<Vec<f64>> {
    if grid.is_blank() {
        return Ok(PERCENTILES.to_vec());
    }
    if grid.rows != 1 && grid.columns != 1 {
        return Err("percentiles must be a row or a column".into());
    }
    grid.cells
        .iter()
        .filter(|v| !v.is_blank())
        .map(|v| match v {
            Value::Number(p) if *p > 0.0 && *p < 1.0 => Ok(*p),
            other => Err(format!(
                "percentiles must be probabilities between 0 and 1, such as 0.995, got {other:?}"
            )),
        })
        .collect()
}

/// The `process` argument: "gamma" (default) or "none" (parameter error
/// only).
fn process(grid: &Grid) -> Result<ProcessDistribution> {
    match text_option(grid, "process")?.as_deref() {
        None | Some("gamma") => Ok(ProcessDistribution::Gamma),
        Some("none") => Ok(ProcessDistribution::None),
        Some(other) => Err(format!(
            "process must be \"gamma\" or \"none\", got {other:?}"
        )),
    }
}

/// A percentile's column heading: "P99.5" for 0.995.
fn percentile_label(p: f64) -> String {
    let pct = format!("{:.4}", p * 100.0);
    format!("P{}", pct.trim_end_matches('0').trim_end_matches('.'))
}

/// The ODP bootstrap of a triangle argument with its options; `sims` at
/// most `max`.
fn odp_fit(
    triangle_arg: &Grid,
    sims: &Grid,
    seed_arg: &Grid,
    tail_arg: &Grid,
    process_arg: &Grid,
    max: usize,
) -> Result<OdpBootstrapFit> {
    let tri = triangle(triangle_arg)?;
    OdpBootstrap {
        n_sims: count(sims, "sims", 10_000, max)?,
        seed: seed(seed_arg)?,
        tail: tail(tail_arg)?,
        process: process(process_arg)?,
        ..OdpBootstrap::default()
    }
    .fit(&tri, COLUMN)
    .map_err(|e| e.to_string())
}

/// The ODP bootstrap exhibit: per origin, the chain-ladder reserve and the
/// simulated reserve's mean, standard deviation, CV and percentiles, then
/// the total, whose percentiles are of the simulated total (not the sum of
/// the origins'):
/// `PRO.ODP(triangle, [sims], [seed], [percentiles], [tail], [process], [origins])`.
pub fn odp_summary(args: &[Grid]) -> Result<Grid> {
    let probabilities = percentiles(&args[3])?;
    let fit = odp_fit(&args[0], &args[1], &args[2], &args[4], &args[5], MAX_SIMS)?;
    let cl = &fit.chain_ladder;
    let labels = origin_labels(&args[6], cl.origins.len())?;

    let n = fit.reserves.n_components();
    let draws = fit.reserves.draw_matrix();
    let column = |j: usize| -> Vec<f64> { draws.iter().skip(j).step_by(n).copied().collect() };
    let stats = |s: &Sampled, reserve: f64| -> Result<Vec<Value>> {
        let (mean, sd) = (s.mean(), s.std_dev());
        let mut row = vec![
            Value::number(reserve),
            Value::number(mean),
            Value::number(sd),
            if mean > 0.0 {
                Value::number(sd / mean)
            } else {
                Value::Empty
            },
        ];
        for &p in &probabilities {
            row.push(Value::number(s.quantile(p).map_err(|e| e.to_string())?));
        }
        Ok(row)
    };

    let mut header: Vec<Value> = ["Origin", "CL reserve", "Mean", "SD", "CV"]
        .map(Value::from)
        .to_vec();
    header.extend(
        probabilities
            .iter()
            .map(|&p| Value::Text(percentile_label(p))),
    );
    let mut rows = vec![header];
    let reserves = cl.reserves();
    for (j, label) in labels.into_iter().enumerate() {
        let s = Sampled::new(column(j)).map_err(|e| e.to_string())?;
        let mut row = vec![label];
        row.extend(stats(&s, reserves[j])?);
        rows.push(row);
    }
    let mut total = vec![Value::from("Total")];
    total.extend(stats(fit.reserves.total(), cl.total_reserve())?);
    rows.push(total);
    Ok(Grid::from_rows(rows))
}

/// The most simulations `PRO.ODP.SIMS` returns: each is a row of cells
/// Excel holds in memory (100,000 rows of 11 cells is about 35 MB), well
/// under its 1,048,576 rows.
pub const MAX_SIM_ROWS: usize = 100_000;

/// The ODP bootstrap's simulated reserves, one row per simulation: a
/// column per origin, then the total, under a header row of the origin
/// labels and "Total". The same `sims`, `seed`, `tail` and `process` give
/// the draws `PRO.ODP` summarizes:
/// `PRO.ODP.SIMS(triangle, [sims], [seed], [tail], [process], [origins])`.
pub fn odp_sims(args: &[Grid]) -> Result<Grid> {
    let fit = odp_fit(
        &args[0],
        &args[1],
        &args[2],
        &args[3],
        &args[4],
        MAX_SIM_ROWS,
    )?;
    let mut header = origin_labels(&args[5], fit.chain_ladder.origins.len())?;
    header.push(Value::from("Total"));
    let n = fit.reserves.n_components();
    let mut rows = vec![header];
    for row in fit.reserves.draw_matrix().chunks_exact(n) {
        let mut cells: Vec<Value> = row.iter().map(|&x| Value::number(x)).collect();
        cells.push(Value::number(row.iter().sum()));
        rows.push(cells);
    }
    Ok(Grid::from_rows(rows))
}
