//! Catastrophe model output on a sheet (`docs/design/xll.md`, decision
//! 44): a year loss table's annual totals and largest occurrences, and an
//! event loss table's occurrence exceedance curve, over
//! `prospicio_aggregate::ylt` and `prospicio_aggregate::cat`.
//!
//! A year loss table is columns of one row per loss: its year, from 1,
//! and its loss, with optionally its time, the fraction of its year
//! elapsed, for an hours clause that joins the losses within a window
//! into one occurrence. Years without a row lose nothing; `n_years` gives
//! the table's years when its last ones are empty. Quantiles and TVaRs
//! on the simulated years follow `quantile`: `"inverted_cdf"` (the
//! default, R `type = 1`, with the exact TVaR), `"linear"` (R `type = 7`,
//! with the TVaR of the interpolated quantile) or `"kth_largest"` (the
//! `k`-th worst year, with the mean of the `k` worst), as
//! `prospicio.risk.statistics` reads them.

use prospicio_aggregate::EventSet;
use prospicio_aggregate::cat::EventLossTable;
use prospicio_aggregate::ylt::{Expansion, YearLossTable};
use prospicio_prob::statistics::{QuantileMethod, TvarMethod};

use crate::aggregate::percent;
use crate::pricing::{Result, numbers};
use crate::reserving::{count, number, text_option};
use crate::sheet::{Grid, Value, position};

fn err(e: prospicio_core::Error) -> String {
    e.to_string()
}

/// The most rows a table may have.
const MAX_ROWS: usize = 1 << 22;

/// A range's cells in reading order, `None` for a blank one; it must be a
/// row or a column.
fn cells(g: &Grid, name: &str) -> Result<Vec<Option<f64>>> {
    if g.rows != 1 && g.columns != 1 {
        return Err(format!("{name} must be a row or a column"));
    }
    if g.cells.len() > MAX_ROWS {
        return Err(format!("{name} has more than {MAX_ROWS} rows"));
    }
    (0..g.rows)
        .flat_map(|r| (0..g.columns).map(move |c| (r, c)))
        .map(|(r, c)| match g.get(r, c) {
            Value::Number(x) => Ok(Some(*x)),
            v if v.is_blank() => Ok(None),
            other => Err(format!(
                "{name} {} must be a number, got {other:?}",
                position(r, c)
            )),
        })
        .collect()
}

/// Columns of equal length read row by row: a row blank in every column
/// is skipped (the empty end of a selected range), a row blank in some is
/// an error.
fn table(columns: &[(&Grid, &str)]) -> Result<Vec<Vec<f64>>> {
    let read = columns
        .iter()
        .map(|(g, name)| cells(g, name))
        .collect::<Result<Vec<_>>>()?;
    let n = read[0].len();
    for (k, (_, name)) in columns.iter().enumerate().skip(1) {
        if read[k].len() != n {
            return Err(format!(
                "{name} has {} cells for {n} {}: give one per row",
                read[k].len(),
                columns[0].1
            ));
        }
    }
    let mut out = vec![Vec::with_capacity(n); columns.len()];
    for i in 0..n {
        let blank = read.iter().filter(|c| c[i].is_none()).count();
        if blank == columns.len() {
            continue;
        }
        if blank > 0 {
            let (_, name) = columns[read.iter().position(|c| c[i].is_none()).unwrap()];
            return Err(format!("row {}: {name} is blank", i + 1));
        }
        for (k, c) in read.iter().enumerate() {
            out[k].push(c[i].unwrap());
        }
    }
    if out[0].is_empty() {
        return Err(format!("{} is required", columns[0].1));
    }
    Ok(out)
}

/// The year loss table's simulated years, one per year of the table, with
/// the hours clause applied when `hours` is given:
/// `years, losses, [n_years], [times], [hours], [negative]`.
fn years(
    years: &Grid,
    losses: &Grid,
    n_years: &Grid,
    times: &Grid,
    hours: &Grid,
    negative: &Grid,
) -> Result<(EventSet, usize)> {
    let mut columns = vec![(years, "years"), (losses, "losses")];
    if !times.is_blank() {
        columns.push((times, "times"));
    }
    let read = table(&columns)?;
    let ys = read[0]
        .iter()
        .enumerate()
        .map(|(k, &y)| {
            if y >= 1.0 && y.fract() == 0.0 && y <= MAX_ROWS as f64 {
                Ok(y as usize)
            } else {
                Err(format!("row {}: years run 1, 2, ..., got {y}", k + 1))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let negative = match negative.as_scalar() {
        Some(v) if v.is_blank() => false,
        Some(Value::Bool(b)) => *b,
        _ => return Err("negative must be TRUE or FALSE".into()),
    };
    let mut ylt = if negative {
        YearLossTable::from_negative_losses(&ys, &read[1])
    } else {
        YearLossTable::new(&ys, &read[1])
    }
    .map_err(err)?;
    if let Some(t) = read.get(2) {
        ylt = ylt.with_times(t.clone()).map_err(err)?;
    }
    if !n_years.is_blank() {
        let n = count(n_years, "n_years", 1, MAX_ROWS)?;
        ylt = ylt.with_n_years(n).map_err(err)?;
    }
    let events = ylt.events(None, Expansion::Tile).map_err(err)?;
    let n_losses = ys.len();
    if hours.is_blank() {
        return Ok((events, n_losses));
    }
    if times.is_blank() {
        return Err("an hours clause needs each loss's time: give times".into());
    }
    let h = number(hours, "hours")?;
    Ok((events.grouped(h).map_err(err)?, n_losses))
}

/// The `quantile` argument and the TVaR that goes with it.
fn methods(g: &Grid) -> Result<(QuantileMethod, TvarMethod)> {
    match text_option(g, "quantile")?.as_deref() {
        None | Some("inverted_cdf") => Ok((QuantileMethod::InvertedCdf, TvarMethod::Exact)),
        Some("linear") => Ok((QuantileMethod::Linear, TvarMethod::Linear)),
        Some("kth_largest") => Ok((QuantileMethod::KthLargest, TvarMethod::WholeYears)),
        Some(other) => Err(format!(
            "quantile must be \"inverted_cdf\", \"linear\" or \"kth_largest\", got {other:?}"
        )),
    }
}

fn row(label: &str, x: f64) -> Vec<Value> {
    vec![Value::from(label), Value::number(x)]
}

fn sorted(mut x: Vec<f64>) -> Vec<f64> {
    x.sort_by(f64::total_cmp);
    x
}

/// Each year's total and largest occurrence.
fn totals_and_maxima(events: &EventSet) -> (Vec<f64>, Vec<f64>) {
    (0..events.n_sims())
        .map(|y| {
            let e = events.events(y);
            (e.iter().sum::<f64>(), e.iter().copied().fold(0.0, f64::max))
        })
        .unzip()
}

/// A year loss table's annual totals: `PRO.YLT.STATS(years, losses,
/// [n_years], [probabilities], [quantile], [times], [hours], [negative])`.
/// The rows are the years, losses and occurrences (fewer than the losses
/// under an hours clause), the years with a loss, the occurrences a year,
/// the annual total's mean (the average annual loss), SD and CV
/// (population, dividing by the years), skewness, smallest and largest
/// year, the largest occurrence, then the total's VaR and TVaR at each
/// probability (0.99, 0.995, 0.999 if omitted).
pub fn ylt_stats(args: &[Grid]) -> Result<Grid> {
    let (events, n_losses) = years(&args[0], &args[1], &args[2], &args[5], &args[6], &args[7])?;
    let ps = probabilities(&args[3])?;
    let (q, t) = methods(&args[4])?;
    let (totals, maxima) = totals_and_maxima(&events);
    let n = totals.len() as f64;
    let counts = events.counts();
    let mean = totals.iter().sum::<f64>() / n;
    let m2 = totals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let m3 = totals.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
    let sorted = sorted(totals);
    let mut rows = vec![
        row("Years", n),
        row("Losses", n_losses as f64),
        row("Occurrences", counts.iter().sum::<usize>() as f64),
        row(
            "Years with a loss",
            counts.iter().filter(|&&c| c > 0).count() as f64,
        ),
        row(
            "Occurrences a year",
            counts.iter().sum::<usize>() as f64 / n,
        ),
        row("Mean", mean),
        row("SD", m2.sqrt()),
        row(
            "CV",
            if mean == 0.0 {
                f64::NAN
            } else {
                m2.sqrt() / mean
            },
        ),
        row("Skewness", m3 / m2.powf(1.5)),
        row("Min", sorted[0]),
        row("Max", sorted[sorted.len() - 1]),
        row(
            "Largest occurrence",
            maxima.iter().copied().fold(0.0, f64::max),
        ),
    ];
    for &p in &ps {
        rows.push(vec![
            Value::Text(format!("VaR {}", percent(p))),
            Value::number(q.quantile(&sorted, p).map_err(err)?),
        ]);
    }
    for &p in &ps {
        rows.push(vec![
            Value::Text(format!("TVaR {}", percent(p))),
            Value::number(t.tvar(&sorted, p).map_err(err)?),
        ]);
    }
    Ok(Grid::from_rows(rows))
}

/// Probabilities in (0, 1), 0.99, 0.995 and 0.999 if omitted.
fn probabilities(g: &Grid) -> Result<Vec<f64>> {
    let ps = if g.is_blank() {
        vec![0.99, 0.995, 0.999]
    } else {
        numbers(g, "probabilities")?
    };
    if let Some(p) = ps.iter().find(|p| !(**p > 0.0 && **p < 1.0)) {
        return Err(format!("probabilities must be in (0, 1), got {p}"));
    }
    Ok(ps)
}

/// The return periods asked for, as annual probabilities (a value below
/// 1 is a probability), and whether they are the defaults.
fn exceedance_probabilities(g: &Grid) -> Result<(Vec<f64>, bool)> {
    if g.is_blank() {
        return Ok((
            [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
                .iter()
                .map(|t| 1.0 / t)
                .collect(),
            true,
        ));
    }
    let ps = numbers(g, "return_periods")?
        .into_iter()
        .map(|t| match t {
            t if t > 1.0 && t.is_finite() => Ok(1.0 / t),
            t if t > 0.0 && t < 1.0 => Ok(t),
            t => Err(format!(
                "return periods must be above 1 (or probabilities in (0, 1)), got {t}"
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((ps, false))
}

/// `1 / p`, blank for a probability of 0.
fn period(p: f64) -> Value {
    Value::number(1.0 / p)
}

/// A year loss table's exceedance curves: `PRO.YLT.OEP(years, losses,
/// [return_periods], [amounts], [n_years], [quantile], [times], [hours],
/// [negative])`. By return period (2 to 1000 years if omitted, those
/// past the table's years left out), a row each: the return period, its
/// annual probability `p`, the occurrence PML (the quantile at `1 - p` of
/// each year's largest occurrence), the aggregate PML (of each year's
/// total) and the TVaRs of both at `1 - p`. With `amounts` instead, a row
/// each: the amount, the OEP and AEP (the share of years whose largest
/// occurrence, or total, is above it) and their return periods.
pub fn ylt_oep(args: &[Grid]) -> Result<Grid> {
    let (events, _) = years(&args[0], &args[1], &args[4], &args[6], &args[7], &args[8])?;
    let (q, t) = methods(&args[5])?;
    let (totals, maxima) = totals_and_maxima(&events);
    let n = totals.len();
    if !args[3].is_blank() {
        if !args[2].is_blank() {
            return Err("give return_periods or amounts, not both".into());
        }
        let share =
            |draws: &[f64], x: f64| draws.iter().filter(|&&d| d > x).count() as f64 / n as f64;
        let mut rows = vec![
            [
                "Amount",
                "OEP",
                "AEP",
                "OEP return period",
                "AEP return period",
            ]
            .map(Value::from)
            .to_vec(),
        ];
        for x in numbers(&args[3], "amounts")? {
            let (o, a) = (share(&maxima, x), share(&totals, x));
            rows.push(vec![
                Value::number(x),
                Value::number(o),
                Value::number(a),
                period(o),
                period(a),
            ]);
        }
        return Ok(Grid::from_rows(rows));
    }
    let (mut ps, defaults) = exceedance_probabilities(&args[2])?;
    let floor = 1.0 / n as f64;
    if defaults {
        ps.retain(|&p| p >= floor);
    } else if let Some(p) = ps.iter().find(|&&p| p < floor) {
        return Err(format!(
            "the table has {n} years: return period {} reads past them",
            1.0 / p
        ));
    }
    let (totals, maxima) = (sorted(totals), sorted(maxima));
    let mut rows = vec![
        [
            "Return period",
            "Probability",
            "Occurrence PML",
            "Aggregate PML",
            "Occurrence TVaR",
            "Aggregate TVaR",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for p in ps {
        rows.push(vec![
            period(p),
            Value::number(p),
            Value::number(q.quantile(&maxima, 1.0 - p).map_err(err)?),
            Value::number(q.quantile(&totals, 1.0 - p).map_err(err)?),
            Value::number(t.tvar(&maxima, 1.0 - p).map_err(err)?),
            Value::number(t.tvar(&totals, 1.0 - p).map_err(err)?),
        ]);
    }
    Ok(Grid::from_rows(rows))
}

/// The event loss table from its columns, with its rates and means:
/// `rates, means, [sd_independent], [sd_correlated], exposures`, a
/// standard deviation omitted read as 0.
fn elt(args: &[Grid]) -> Result<(EventLossTable, Vec<f64>, Vec<f64>)> {
    let names = [
        "rates",
        "means",
        "sd_independent",
        "sd_correlated",
        "exposures",
    ];
    let given: Vec<usize> = (0..5)
        .filter(|&k| k % 4 == 0 || k == 1 || !args[k].is_blank())
        .collect();
    let columns: Vec<(&Grid, &str)> = given.iter().map(|&k| (&args[k], names[k])).collect();
    let read = table(&columns)?;
    let column = |k: usize| match given.iter().position(|&g| g == k) {
        Some(i) => read[i].clone(),
        None => vec![0.0; read[0].len()],
    };
    let (rates, means) = (column(0), column(1));
    let elt =
        EventLossTable::new(&rates, &means, &column(2), &column(3), &column(4)).map_err(err)?;
    Ok((elt, rates, means))
}

/// An event loss table's summary: `PRO.ELT.STATS(rates, means,
/// [sd_independent], [sd_correlated], exposures)`. The rows are the
/// events, the mean number of events a year `Λ = Σ rate`, the chance of a
/// year with an event `1 - exp(-Λ)`, the expected annual loss
/// `Σ rate × mean`, the annual loss's SD and CV (a compound Poisson's,
/// `Var = Σ rate (mean² + σ²)` with `σ` the two standard deviations
/// summed) and the largest exposure.
pub fn elt_stats(args: &[Grid]) -> Result<Grid> {
    let (elt, rates, means) = elt(args)?;
    let variance: f64 = rates
        .iter()
        .zip(&means)
        .zip(elt.standard_deviations())
        .map(|((r, m), s)| r * (m * m + s * s))
        .sum();
    let eal = elt.expected_annual_loss();
    Ok(Grid::from_rows(vec![
        row("Events", elt.len() as f64),
        row("Events a year", elt.rate()),
        row("Chance of an event", -(-elt.rate()).exp_m1()),
        row("Expected annual loss", eal),
        row("SD", variance.sqrt()),
        row("CV", variance.sqrt() / eal),
        row(
            "Largest exposure",
            elt.exposures().iter().copied().fold(0.0, f64::max),
        ),
    ]))
}

/// An event loss table's occurrence exceedance curve:
/// `PRO.ELT.OEP(rates, means, [sd_independent], [sd_correlated],
/// exposures, [return_periods], [amounts])`, events independent Poisson
/// processes and each one's loss a beta on `[0, exposure]` with its mean
/// and standard deviation. By return period (2 to 1000 years if omitted,
/// those past the chance of any event left out), a row each: the return
/// period, its annual probability `p`, the occurrence PML (the loss whose
/// OEP is `p`), the annual frequency of events above it and its
/// occurrence return period (`1 /` that frequency). With `amounts`
/// instead, a row each: the amount, the frequency of events above it, the
/// OEP `1 - exp(-frequency)` and its return period.
pub fn elt_oep(args: &[Grid]) -> Result<Grid> {
    let (elt, _, _) = elt(args)?;
    if !args[6].is_blank() {
        if !args[5].is_blank() {
            return Err("give return_periods or amounts, not both".into());
        }
        let mut rows = vec![
            ["Amount", "Exceedance frequency", "OEP", "Return period"]
                .map(Value::from)
                .to_vec(),
        ];
        for x in numbers(&args[6], "amounts")? {
            let p = elt.oep(x);
            rows.push(vec![
                Value::number(x),
                Value::number(elt.exceedance_frequency(x)),
                Value::number(p),
                period(p),
            ]);
        }
        return Ok(Grid::from_rows(rows));
    }
    let (mut ps, defaults) = exceedance_probabilities(&args[5])?;
    let ceiling = -(-elt.rate()).exp_m1();
    if defaults {
        ps.retain(|&p| p < ceiling);
    }
    let mut rows = vec![
        [
            "Return period",
            "Probability",
            "Occurrence PML",
            "Exceedance frequency",
            "Occurrence return period",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for p in ps {
        if p >= ceiling {
            return Err(format!(
                "the chance of a year with any event is {ceiling}; probability {p} is past it"
            ));
        }
        let x = elt.occurrence_pml(p).map_err(err)?;
        let f = elt.exceedance_frequency(x);
        rows.push(vec![
            period(p),
            Value::number(p),
            Value::number(x),
            Value::number(f),
            period(f),
        ]);
    }
    Ok(Grid::from_rows(rows))
}
