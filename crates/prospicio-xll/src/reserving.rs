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
use prospicio_prob::{Distribution, PredictiveDistribution, Sampled};
use prospicio_reserving::{
    Average, Benktander, BornhuetterFerguson, CapeCod, ChainLadder, ChainLadderFit, ClarkCapeCod,
    ClarkFit, ClarkLdf, CurveShape, Development, DevelopmentColumn, ExpectedLoss, ExpectedLossFit,
    GrowthCurve, Long, Mack, MackBootstrap, MackBootstrapFit, MackFit, MackProcess, OdpBootstrap,
    OdpBootstrapFit, OneYearMethod, PatternKind, PatternWeight, ProcessDistribution,
    SelectedPattern, SigmaInterpolation, Tail, TailBondy, TailCurve, Triangle,
};

use crate::sheet::{CellError, Grid, Value, position};

/// The measure name inside the triangle; never shown on the sheet.
const COLUMN: &str = "values";

/// Its position among the triangle's measures: the first.
pub(crate) const COLUMN_POSITION: usize = 0;

type Result<T> = std::result::Result<T, String>;

/// The triangle laid out in `grid`.
pub fn triangle(grid: &Grid) -> Result<Triangle> {
    triangle_with(grid, None)
}

/// The measure name of an exposure column inside the triangle.
const EXPOSURE: &str = "exposure";

/// The triangle laid out in `grid`, with `exposure` (one value per origin)
/// as a second measure on every observed cell of its origin when given.
fn triangle_with(grid: &Grid, exposure: Option<&[f64]>) -> Result<Triangle> {
    triangle_of(grid, exposure, true)
}

/// The triangle laid out in `grid`, of cumulative values or, with
/// `cumulative` false, of incremental ones.
pub(crate) fn triangle_of(
    grid: &Grid,
    exposure: Option<&[f64]>,
    cumulative: bool,
) -> Result<Triangle> {
    if grid.rows < 1 || grid.columns < 2 {
        return Err("the triangle needs at least two development columns".into());
    }
    let mut origin = Vec::new();
    let mut age: Vec<Lag> = Vec::new();
    let mut values = Vec::new();
    let mut exposures = Vec::new();
    if let Some(e) = exposure {
        if e.len() != grid.rows {
            return Err(format!(
                "exposure must have {} values, one per triangle row, got {}",
                grid.rows,
                e.len()
            ));
        }
    }
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
            if let Some(e) = exposure {
                exposures.push(e[r]);
            }
            observed += 1;
        }
        if observed == 0 {
            return Err(format!("triangle row {} has no values", r + 1));
        }
    }
    let mut measures: Vec<(&str, &[f64])> = vec![(COLUMN, &values)];
    if exposure.is_some() {
        measures.push((EXPOSURE, &exposures));
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &measures,
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative,
    })
    .map_err(|e| e.to_string())
}

/// The text of an optional argument, lower-cased; `None` when omitted.
pub(crate) fn text_option(grid: &Grid, name: &str) -> Result<Option<String>> {
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
pub(crate) fn origin_labels(grid: &Grid, n: usize) -> Result<Vec<Value>> {
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

/// The `sigma` argument: how Mack's last sigma is filled in,
/// "log-linear" (default, R's and chainladder-python's) or "mack" (Mack
/// 1993).
fn sigma(grid: &Grid) -> Result<SigmaInterpolation> {
    match text_option(grid, "sigma")?.as_deref() {
        None | Some("log-linear" | "loglinear" | "log_linear") => Ok(SigmaInterpolation::LogLinear),
        Some("mack") => Ok(SigmaInterpolation::Mack),
        Some(other) => Err(format!(
            "sigma must be \"log-linear\" or \"mack\", got {other:?}"
        )),
    }
}

/// The `selected`, `kind` and `weight` arguments: factors chosen by the
/// user, one cell per triangle column (blank keeps the estimate), as
/// age-to-age factors ("ldf", the default), factors to ultimate ("cdf") or
/// percents developed ("percent"), blended `weight * selected + (1 -
/// weight) * estimated` (1, fully selected, if omitted). `None` when no
/// factor is selected.
fn selected(selected: &Grid, kind: &Grid, weight: &Grid) -> Result<Option<SelectedPattern>> {
    if selected.is_blank() {
        if !kind.is_blank() || !weight.is_blank() {
            return Err("kind and weight need selected factors".into());
        }
        return Ok(None);
    }
    if selected.rows != 1 && selected.columns != 1 {
        return Err("selected must be a row or a column, one cell per triangle column".into());
    }
    let kind = match text_option(kind, "kind")?.as_deref() {
        None | Some("ldf" | "age-to-age") => PatternKind::AgeToAge,
        Some("cdf" | "age-to-ultimate") => PatternKind::AgeToUltimate,
        Some("percent" | "percent developed" | "percent_developed") => {
            PatternKind::PercentDeveloped
        }
        Some(other) => {
            return Err(format!(
                "kind must be \"ldf\", \"cdf\" or \"percent\", got {other:?}"
            ));
        }
    };
    let mut values = Vec::new();
    for (c, v) in selected.cells.iter().enumerate() {
        match v {
            Value::Number(x) => values.push((12 * (c as Lag + 1), *x)),
            v if v.is_blank() => {}
            other => {
                return Err(format!(
                    "selected {} must be a number, got {other:?}",
                    c + 1
                ));
            }
        }
    }
    let mut pattern = SelectedPattern::new(kind, values);
    if !weight.is_blank() {
        pattern.weight = PatternWeight::All(number(weight, "weight")?);
    }
    Ok(Some(pattern))
}

/// The development options of a chain-ladder function.
fn development_with(
    average_arg: &Grid,
    selected_args: [&Grid; 3],
    sigma_arg: &Grid,
) -> Result<Development> {
    let [s, kind, weight] = selected_args;
    Ok(Development {
        average: average(average_arg)?,
        sigma_interpolation: sigma(sigma_arg)?,
        pattern: selected(s, kind, weight)?,
    })
}

/// The chain-ladder fit of a triangle argument and its options.
fn chain_ladder(
    triangle_arg: &Grid,
    development: Development,
    tail_arg: &Grid,
) -> Result<ChainLadderFit> {
    let tri = triangle(triangle_arg)?;
    ChainLadder {
        development,
        tail: tail(tail_arg)?,
    }
    .fit(&tri, COLUMN)
    .map_err(|e| e.to_string())
}

/// The chain-ladder fit of a vector function's arguments: `triangle,
/// [average], [tail], [selected], [kind], [weight]`.
fn chain_ladder_args(args: &[Grid]) -> Result<ChainLadderFit> {
    let dev = development_with(&args[1], [&args[3], &args[4], &args[5]], &Grid::missing())?;
    chain_ladder(&args[0], dev, &args[2])
}

/// Selected age-to-age factors, one per development column: the factors
/// within the triangle, then the tail factor to ultimate (1 without a
/// tail), so the row sits under the triangle's columns.
pub fn cl_ldf(args: &[Grid]) -> Result<Grid> {
    let fit = chain_ladder_args(args)?;
    let mut ldf = fit.ldf().to_vec();
    ldf.push(fit.tail.factor);
    Ok(Grid::row(&ldf))
}

/// Factors from each development column's age to ultimate.
pub fn cl_cdf(args: &[Grid]) -> Result<Grid> {
    let fit = chain_ladder_args(args)?;
    Ok(Grid::row(&fit.cdf))
}

/// Projected ultimate per origin, a column beside the triangle.
pub fn cl_ultimate(args: &[Grid]) -> Result<Grid> {
    let fit = chain_ladder_args(args)?;
    Ok(Grid::column(&fit.ultimate))
}

/// Reserve (ultimate less latest, IBNR) per origin.
pub fn cl_reserve(args: &[Grid]) -> Result<Grid> {
    let fit = chain_ladder_args(args)?;
    Ok(Grid::column(&fit.reserves()))
}

/// The chain-ladder exhibit: Origin, Latest, CDF, Ultimate, Reserve per
/// origin, then a total row.
pub fn cl_summary(args: &[Grid]) -> Result<Grid> {
    let dev = development_with(&args[1], [&args[4], &args[5], &args[6]], &Grid::missing())?;
    let fit = chain_ladder(&args[0], dev, &args[2])?;
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
    let dev = development_with(&args[1], [&args[3], &args[4], &args[5]], &args[6])?;
    let fit = chain_ladder(&args[0], dev, &args[2])?;
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
    let fit = mack_fit(args)?;
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
pub(crate) fn count(grid: &Grid, name: &str, default: usize, max: usize) -> Result<usize> {
    match grid.as_scalar() {
        Some(v) if v.is_blank() => Ok(default),
        Some(Value::Number(x)) if x.fract() == 0.0 && *x >= 1.0 && *x <= max as f64 => {
            Ok(*x as usize)
        }
        _ => Err(format!("{name} must be a whole number from 1 to {max}")),
    }
}

/// The `seed` argument: a whole number from 0, default 0.
pub(crate) fn seed(grid: &Grid) -> Result<u64> {
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
pub(crate) fn percentiles(grid: &Grid) -> Result<Vec<f64>> {
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
pub(crate) fn process(grid: &Grid) -> Result<ProcessDistribution> {
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

/// A bootstrap's exhibit: per origin, the chain-ladder reserve and the
/// simulated reserve's mean, SD, CV and `probabilities`' percentiles, then
/// the total, whose percentiles are of the simulated total.
pub(crate) fn bootstrap_exhibit(
    reserves: &PredictiveDistribution,
    cl: &ChainLadderFit,
    labels: Vec<Value>,
    probabilities: &[f64],
) -> Result<Grid> {
    let n = reserves.n_components();
    let draws = reserves.draw_matrix();
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
        for &p in probabilities {
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
    let cl_reserves = cl.reserves();
    for (j, label) in labels.into_iter().enumerate() {
        let s = Sampled::new(column(j)).map_err(|e| e.to_string())?;
        let mut row = vec![label];
        row.extend(stats(&s, cl_reserves[j])?);
        rows.push(row);
    }
    let mut total = vec![Value::from("Total")];
    total.extend(stats(reserves.total(), cl.total_reserve())?);
    rows.push(total);
    Ok(Grid::from_rows(rows))
}

/// A bootstrap's draws: a header of the origin labels and "Total", then
/// one row per simulation.
fn draws_table(reserves: &PredictiveDistribution, mut header: Vec<Value>) -> Grid {
    header.push(Value::from("Total"));
    let n = reserves.n_components();
    let mut rows = vec![header];
    for row in reserves.draw_matrix().chunks_exact(n) {
        let mut cells: Vec<Value> = row.iter().map(|&x| Value::number(x)).collect();
        cells.push(Value::number(row.iter().sum()));
        rows.push(cells);
    }
    Grid::from_rows(rows)
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
    let labels = origin_labels(&args[6], fit.chain_ladder.origins.len())?;
    bootstrap_exhibit(&fit.reserves, &fit.chain_ladder, labels, &probabilities)
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
    let labels = origin_labels(&args[5], fit.chain_ladder.origins.len())?;
    Ok(draws_table(&fit.reserves, labels))
}

/// The `process` argument of Mack's bootstrap: "gamma" (default),
/// "lognormal", "normal", "residuals" or "none" (parameter error only).
fn mack_process(grid: &Grid) -> Result<MackProcess> {
    match text_option(grid, "process")?.as_deref() {
        None | Some("gamma") => Ok(MackProcess::Gamma),
        Some("lognormal") => Ok(MackProcess::Lognormal),
        Some("normal") => Ok(MackProcess::Normal),
        Some("residuals") => Ok(MackProcess::Residuals),
        Some("none") => Ok(MackProcess::None),
        Some(other) => Err(format!(
            "process must be \"gamma\", \"lognormal\", \"normal\", \"residuals\" or \"none\", \
             got {other:?}"
        )),
    }
}

/// Mack's bootstrap (England, Verrall and Wuthrich) of a triangle argument
/// with its options, centred residuals; `sims` at most `max`.
#[allow(clippy::too_many_arguments)]
fn mack_bootstrap_fit(
    triangle_arg: &Grid,
    sims: &Grid,
    seed_arg: &Grid,
    average_arg: &Grid,
    tail_arg: &Grid,
    process_arg: &Grid,
    sigma_arg: &Grid,
    max: usize,
) -> Result<MackBootstrapFit> {
    let tri = triangle(triangle_arg)?;
    MackBootstrap {
        n_sims: count(sims, "sims", 10_000, max)?,
        seed: seed(seed_arg)?,
        process: mack_process(process_arg)?,
        development: Development {
            sigma_interpolation: sigma(sigma_arg)?,
            ..development(average(average_arg)?)
        },
        tail: tail(tail_arg)?,
        ..MackBootstrap::default()
    }
    .fit(&tri, COLUMN)
    .map_err(|e| e.to_string())
}

/// Mack's bootstrap exhibit, laid out as `PRO.ODP`'s:
/// `PRO.MACK.BOOT(triangle, [sims], [seed], [percentiles], [average], [tail], [process], [origins], [sigma])`.
pub fn mack_bootstrap_summary(args: &[Grid]) -> Result<Grid> {
    let probabilities = percentiles(&args[3])?;
    let fit = mack_bootstrap_fit(
        &args[0], &args[1], &args[2], &args[4], &args[5], &args[6], &args[8], MAX_SIMS,
    )?;
    let cl = &fit.mack.chain_ladder;
    let labels = origin_labels(&args[7], cl.origins.len())?;
    bootstrap_exhibit(&fit.reserves, cl, labels, &probabilities)
}

/// Mack's bootstrap's simulated reserves, laid out as `PRO.ODP.SIMS`':
/// `PRO.MACK.BOOT.SIMS(triangle, [sims], [seed], [average], [tail], [process], [origins], [sigma])`.
pub fn mack_bootstrap_sims(args: &[Grid]) -> Result<Grid> {
    let fit = mack_bootstrap_fit(
        &args[0],
        &args[1],
        &args[2],
        &args[3],
        &args[4],
        &args[5],
        &args[7],
        MAX_SIM_ROWS,
    )?;
    let labels = origin_labels(&args[6], fit.mack.chain_ladder.origins.len())?;
    Ok(draws_table(&fit.reserves, labels))
}

/// The `exposure` argument: one positive number per origin, as a row or a
/// column.
fn exposure(grid: &Grid, origins: usize) -> Result<Vec<f64>> {
    if grid.cells.len() != origins || (grid.rows != 1 && grid.columns != 1) {
        return Err(format!(
            "exposure must be a row or column of {origins} values, one per triangle row"
        ));
    }
    grid.cells
        .iter()
        .enumerate()
        .map(|(i, v)| match v {
            Value::Number(x) if *x > 0.0 => Ok(*x),
            other => Err(format!(
                "exposure {} must be a positive number, got {other:?}",
                i + 1
            )),
        })
        .collect()
}

/// A required number argument.
fn number(grid: &Grid, name: &str) -> Result<f64> {
    match grid.as_scalar() {
        Some(Value::Number(x)) => Ok(*x),
        Some(v) if v.is_blank() => Err(format!("{name} is required")),
        _ => Err(format!("{name} must be a single number")),
    }
}

/// An optional number argument with a default.
fn number_or(grid: &Grid, name: &str, default: f64) -> Result<f64> {
    match grid.as_scalar() {
        Some(v) if v.is_blank() => Ok(default),
        _ => number(grid, name),
    }
}

/// The triangle and exposure arguments of an expected-loss method.
fn exposed_triangle(triangle_arg: &Grid, exposure_arg: &Grid) -> Result<Triangle> {
    let e = exposure(exposure_arg, triangle_arg.rows)?;
    triangle_with(triangle_arg, Some(&e))
}

/// The expected-loss methods' exhibit: Origin, Exposure, Latest, CDF,
/// A priori (the loss ratio applied), Ultimate and Reserve per origin, then
/// a total row.
fn expected_loss_exhibit(fit: &ExpectedLossFit, labels: Vec<Value>) -> Grid {
    let cl = &fit.chain_ladder;
    let reserves = fit.reserves();
    let mut rows = vec![
        [
            "Origin", "Exposure", "Latest", "CDF", "A priori", "Ultimate", "Reserve",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for (i, label) in labels.into_iter().enumerate() {
        rows.push(vec![
            label,
            Value::number(fit.exposure[i]),
            Value::number(cl.latest[i]),
            Value::number(cl.cdf[cl.latest_position[i]]),
            Value::number(fit.apriori[i]),
            Value::number(fit.ultimate[i]),
            Value::number(reserves[i]),
        ]);
    }
    rows.push(vec![
        Value::from("Total"),
        Value::number(fit.exposure.iter().sum()),
        Value::number(cl.latest.iter().sum()),
        Value::Empty,
        Value::Empty,
        Value::number(fit.total_ultimate()),
        Value::number(fit.total_reserve()),
    ]);
    Grid::from_rows(rows)
}

/// The chain ladder an expected-loss method takes its pattern from.
fn pattern(average_arg: &Grid, tail_arg: &Grid) -> Result<ChainLadder> {
    Ok(ChainLadder {
        development: development(average(average_arg)?),
        tail: tail(tail_arg)?,
    })
}

/// The expected loss method: exposure times the a priori loss ratio:
/// `PRO.EL(triangle, exposure, apriori, [average], [tail], [origins])`.
pub fn expected_loss(args: &[Grid]) -> Result<Grid> {
    let tri = exposed_triangle(&args[0], &args[1])?;
    let fit = ExpectedLoss {
        apriori: number(&args[2], "apriori")?,
        chain_ladder: pattern(&args[3], &args[4])?,
    }
    .fit(&tri, COLUMN, EXPOSURE)
    .map_err(|e| e.to_string())?;
    let labels = origin_labels(&args[5], fit.chain_ladder.origins.len())?;
    Ok(expected_loss_exhibit(&fit, labels))
}

/// Bornhuetter-Ferguson: the latest plus the a priori ultimate's
/// unreported share:
/// `PRO.BF(triangle, exposure, apriori, [average], [tail], [origins])`.
pub fn bornhuetter_ferguson(args: &[Grid]) -> Result<Grid> {
    let tri = exposed_triangle(&args[0], &args[1])?;
    let fit = BornhuetterFerguson {
        apriori: number(&args[2], "apriori")?,
        chain_ladder: pattern(&args[3], &args[4])?,
    }
    .fit(&tri, COLUMN, EXPOSURE)
    .map_err(|e| e.to_string())?;
    let labels = origin_labels(&args[5], fit.chain_ladder.origins.len())?;
    Ok(expected_loss_exhibit(&fit, labels))
}

/// Benktander: Bornhuetter-Ferguson iterated, each pass taking the last
/// ultimate as its a priori (1 is Bornhuetter-Ferguson):
/// `PRO.BENKTANDER(triangle, exposure, apriori, [iterations], [average], [tail], [origins])`.
pub fn benktander(args: &[Grid]) -> Result<Grid> {
    let tri = exposed_triangle(&args[0], &args[1])?;
    let fit = Benktander {
        apriori: number(&args[2], "apriori")?,
        n_iters: count(&args[3], "iterations", 1, 10_000)?,
        chain_ladder: pattern(&args[4], &args[5])?,
    }
    .fit(&tri, COLUMN, EXPOSURE)
    .map_err(|e| e.to_string())?;
    let labels = origin_labels(&args[6], fit.chain_ladder.origins.len())?;
    Ok(expected_loss_exhibit(&fit, labels))
}

/// Cape Cod (Stanard-Buhlmann): the a priori loss ratio estimated from the
/// triangle, trended and decayed by distance between origins:
/// `PRO.CAPECOD(triangle, exposure, [trend], [decay], [average], [tail], [origins])`.
pub fn cape_cod(args: &[Grid]) -> Result<Grid> {
    let tri = exposed_triangle(&args[0], &args[1])?;
    let fit = CapeCod {
        trend: number_or(&args[2], "trend", 0.0)?,
        decay: number_or(&args[3], "decay", 1.0)?,
        chain_ladder: pattern(&args[4], &args[5])?,
    }
    .fit(&tri, COLUMN, EXPOSURE)
    .map_err(|e| e.to_string())?;
    let labels = origin_labels(&args[6], fit.expected_loss.chain_ladder.origins.len())?;
    Ok(expected_loss_exhibit(&fit.expected_loss, labels))
}

/// Mack's model of a triangle argument with the chain-ladder options at
/// `args[1]` (average), `args[2]` (tail) and `args[4..8]` (selected, kind,
/// weight, sigma).
fn mack_fit(args: &[Grid]) -> Result<MackFit> {
    let tri = triangle(&args[0])?;
    Mack {
        development: development_with(&args[1], [&args[4], &args[5], &args[6]], &args[7])?,
        tail: tail(&args[2])?,
        ..Mack::default()
    }
    .fit(&tri, COLUMN)
    .map_err(|e| e.to_string())
}

/// Merz and Wuthrich's one-year view of Mack's model: per origin the
/// reserve, the standard error of the claims development result in the
/// coming year (R's `CDR(1)S.E.`) and of the full run-off (Mack's), then
/// the total:
/// `PRO.MACK.CDR(triangle, [average], [tail], [origins], [selected], [kind], [weight], [sigma])`.
pub fn mack_cdr(args: &[Grid]) -> Result<Grid> {
    let fit = mack_fit(args)?;
    let cdr = fit.claims_development_result().map_err(|e| e.to_string())?;
    let cl = &fit.chain_ladder;
    let labels = origin_labels(&args[3], cl.origins.len())?;
    let reserves = cl.reserves();
    let run_off = cdr.run_off_standard_error();
    let mut rows = vec![
        ["Origin", "Reserve", "CDR(1) SE", "Run-off SE"]
            .map(Value::from)
            .to_vec(),
    ];
    for (i, label) in labels.into_iter().enumerate() {
        rows.push(vec![
            label,
            Value::number(reserves[i]),
            Value::number(cdr.one_year_standard_error[i]),
            Value::number(run_off[i]),
        ]);
    }
    rows.push(vec![
        Value::from("Total"),
        Value::number(cl.total_reserve()),
        Value::number(cdr.total_one_year_standard_error),
        Value::number(fit.total_standard_error),
    ]);
    Ok(Grid::from_rows(rows))
}

/// The standard error of each origin's claims development result in each
/// future calendar year (R's `CDR(k)S.E.`), a column per year, then the
/// total's:
/// `PRO.MACK.CDR.YEARS(triangle, [average], [tail], [origins], [selected], [kind], [weight], [sigma])`.
pub fn mack_cdr_years(args: &[Grid]) -> Result<Grid> {
    let fit = mack_fit(args)?;
    let cdr = fit.claims_development_result().map_err(|e| e.to_string())?;
    let labels = origin_labels(&args[3], cdr.origins.len())?;
    let years = cdr.by_calendar_year.len();
    let mut header = vec![Value::from("Origin")];
    header.extend((1..=years).map(|t| Value::Text(format!("Year {t}"))));
    let mut rows = vec![header];
    for (o, label) in labels.into_iter().enumerate() {
        let mut row = vec![label];
        row.extend(
            cdr.by_calendar_year
                .iter()
                .map(|year| Value::number(year[o])),
        );
        rows.push(row);
    }
    let mut total = vec![Value::from("Total")];
    total.extend(cdr.total_by_calendar_year.iter().map(|&x| Value::number(x)));
    rows.push(total);
    Ok(Grid::from_rows(rows))
}

/// The percentiles a simulated CDR exhibit reports when none are given:
/// the adverse tail (a negative CDR is adverse) and its mirror.
pub const CDR_PERCENTILES: [f64; 4] = [0.005, 0.05, 0.95, 0.995];

/// A simulated one-year exhibit: per origin the opening reserve and the
/// simulated claims development result's mean, SD and percentiles, then
/// the total, whose percentiles are of the simulated total CDR.
fn cdr_exhibit(
    cdr: &PredictiveDistribution,
    opening_reserve: &[f64],
    labels: Vec<Value>,
    probabilities: &[f64],
) -> Result<Grid> {
    let n = cdr.n_components();
    let draws = cdr.draw_matrix();
    let stats = |s: &Sampled, reserve: f64| -> Result<Vec<Value>> {
        let mut row = vec![
            Value::number(reserve),
            Value::number(s.mean()),
            Value::number(s.std_dev()),
        ];
        for &p in probabilities {
            row.push(Value::number(s.quantile(p).map_err(|e| e.to_string())?));
        }
        Ok(row)
    };
    let mut header: Vec<Value> = ["Origin", "Opening reserve", "Mean CDR", "SD"]
        .map(Value::from)
        .to_vec();
    header.extend(
        probabilities
            .iter()
            .map(|&p| Value::Text(percentile_label(p))),
    );
    let mut rows = vec![header];
    for (j, label) in labels.into_iter().enumerate() {
        let column: Vec<f64> = draws.iter().skip(j).step_by(n).copied().collect();
        let s = Sampled::new(column).map_err(|e| e.to_string())?;
        let mut row = vec![label];
        row.extend(stats(&s, opening_reserve[j])?);
        rows.push(row);
    }
    let mut total = vec![Value::from("Total")];
    total.extend(stats(cdr.total(), opening_reserve.iter().sum())?);
    rows.push(total);
    Ok(Grid::from_rows(rows))
}

/// The `percentiles` argument of a CDR exhibit, [`CDR_PERCENTILES`] when
/// omitted.
fn cdr_percentiles(grid: &Grid) -> Result<Vec<f64>> {
    if grid.is_blank() {
        Ok(CDR_PERCENTILES.to_vec())
    } else {
        percentiles(grid)
    }
}

/// The ODP bootstrap's one-year view, re-reserving with the chain ladder
/// on each simulated year:
/// `PRO.ODP.CDR(triangle, [sims], [seed], [percentiles], [tail], [process], [origins])`.
pub fn odp_cdr(args: &[Grid]) -> Result<Grid> {
    let tri = triangle(&args[0])?;
    let probabilities = cdr_percentiles(&args[3])?;
    let tail = tail(&args[4])?;
    let fit = OdpBootstrap {
        n_sims: count(&args[1], "sims", 10_000, MAX_SIMS)?,
        seed: seed(&args[2])?,
        tail,
        process: process(&args[5])?,
        ..OdpBootstrap::default()
    }
    .one_year(
        &tri,
        COLUMN,
        &OneYearMethod::ChainLadder(ChainLadder {
            tail,
            ..ChainLadder::default()
        }),
    )
    .map_err(|e| e.to_string())?;
    let labels = origin_labels(&args[6], fit.opening_reserve.len())?;
    cdr_exhibit(&fit.cdr, &fit.opening_reserve, labels, &probabilities)
}

/// Mack's bootstrap's one-year view, re-reserving with the chain ladder
/// (the same average and sigma rule) on each simulated year:
/// `PRO.MACK.BOOT.CDR(triangle, [sims], [seed], [percentiles], [average], [tail], [process], [origins], [sigma])`.
pub fn mack_bootstrap_cdr(args: &[Grid]) -> Result<Grid> {
    let tri = triangle(&args[0])?;
    let probabilities = cdr_percentiles(&args[3])?;
    let dev = Development {
        sigma_interpolation: sigma(&args[8])?,
        ..development(average(&args[4])?)
    };
    let tail = tail(&args[5])?;
    let fit = MackBootstrap {
        n_sims: count(&args[1], "sims", 10_000, MAX_SIMS)?,
        seed: seed(&args[2])?,
        process: mack_process(&args[6])?,
        development: dev.clone(),
        tail,
        ..MackBootstrap::default()
    }
    .one_year(
        &tri,
        COLUMN,
        &OneYearMethod::ChainLadder(ChainLadder {
            development: dev,
            tail,
        }),
    )
    .map_err(|e| e.to_string())?;
    let labels = origin_labels(&args[7], fit.opening_reserve.len())?;
    cdr_exhibit(&fit.cdr, &fit.opening_reserve, labels, &probabilities)
}

/// The `curve` argument of Clark's methods: "loglogistic" (default) or
/// "weibull".
fn growth_curve(grid: &Grid) -> Result<GrowthCurve> {
    match text_option(grid, "curve")?.as_deref() {
        None | Some("loglogistic" | "log-logistic" | "log_logistic") => {
            Ok(GrowthCurve::LogLogistic)
        }
        Some("weibull") => Ok(GrowthCurve::Weibull),
        Some(other) => Err(format!(
            "curve must be \"loglogistic\" or \"weibull\", got {other:?}"
        )),
    }
}

/// The `max_age` argument: the age in months development stops at;
/// omitted (or "inf") develops to infinity, as R's `maxage = Inf`.
fn max_age(grid: &Grid) -> Result<Option<f64>> {
    match grid.as_scalar() {
        Some(v) if v.is_blank() => Ok(None),
        Some(Value::Text(s))
            if matches!(s.trim().to_ascii_lowercase().as_str(), "inf" | "unlimited") =>
        {
            Ok(None)
        }
        Some(Value::Number(x)) if *x > 0.0 => Ok(Some(*x)),
        _ => Err("max_age must be a positive age in months, or \"inf\"".into()),
    }
}

/// Clark's growth-curve fit: the LDF method without `exposure`, Cape Cod
/// with it.
fn clark_fit(
    triangle_arg: &Grid,
    exposure_arg: &Grid,
    curve: &Grid,
    max: &Grid,
) -> Result<ClarkFit> {
    let (curve, max_age) = (growth_curve(curve)?, max_age(max)?);
    if exposure_arg.is_blank() {
        let tri = triangle(triangle_arg)?;
        ClarkLdf { curve, max_age }.fit(&tri, COLUMN)
    } else {
        let tri = exposed_triangle(triangle_arg, exposure_arg)?;
        ClarkCapeCod { curve, max_age }.fit(&tri, COLUMN, EXPOSURE)
    }
    .map_err(|e| e.to_string())
}

/// Clark's exhibit (R's `ClarkLDF` and `ClarkCapeCod`): per origin the
/// latest, the ultimate, the reserve and its process, parameter and total
/// standard errors, then the total:
/// `PRO.CLARK(triangle, [exposure], [curve], [max_age], [origins])`.
pub fn clark(args: &[Grid]) -> Result<Grid> {
    let fit = clark_fit(&args[0], &args[1], &args[2], &args[3])?;
    let cl = &fit.chain_ladder;
    let labels = origin_labels(&args[4], cl.origins.len())?;
    let reserves = fit.reserves();
    let mut rows = vec![
        [
            "Origin",
            "Latest",
            "Ultimate",
            "Reserve",
            "Process SE",
            "Parameter SE",
            "Total SE",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for (i, label) in labels.into_iter().enumerate() {
        rows.push(vec![
            label,
            Value::number(cl.latest[i]),
            Value::number(fit.ultimate[i]),
            Value::number(reserves[i]),
            Value::number(fit.process_risk[i]),
            Value::number(fit.parameter_risk[i]),
            Value::number(fit.standard_error[i]),
        ]);
    }
    rows.push(vec![
        Value::from("Total"),
        Value::number(cl.latest.iter().sum()),
        Value::number(fit.total_ultimate()),
        Value::number(fit.total_reserve()),
        Value::number(fit.total_process_risk),
        Value::number(fit.total_parameter_risk),
        Value::number(fit.total_standard_error),
    ]);
    Ok(Grid::from_rows(rows))
}

/// Clark's fitted parameters, a name and a value per row: the curve,
/// omega, theta, the expected loss ratio (Cape Cod only), the scale
/// (R's `sigma2`) and the number of incremental observations:
/// `PRO.CLARK.PARAMS(triangle, [exposure], [curve], [max_age])`.
pub fn clark_params(args: &[Grid]) -> Result<Grid> {
    let fit = clark_fit(&args[0], &args[1], &args[2], &args[3])?;
    let curve = match fit.curve {
        GrowthCurve::LogLogistic => "loglogistic",
        GrowthCurve::Weibull => "weibull",
    };
    let mut rows = vec![
        vec![Value::from("Curve"), Value::from(curve)],
        vec![Value::from("Omega"), Value::number(fit.omega)],
        vec![Value::from("Theta"), Value::number(fit.theta)],
    ];
    if let Some(elr) = fit.elr {
        rows.push(vec![Value::from("ELR"), Value::number(elr)]);
    }
    rows.push(vec![Value::from("Scale"), Value::number(fit.scale)]);
    rows.push(vec![
        Value::from("Observations"),
        Value::Number(fit.n_observations as f64),
    ]);
    Ok(Grid::from_rows(rows))
}
