//! Aggregate (compound) distributions (`docs/design/xll.md`): the annual
//! total of a claim count and a severity, ground up or to a
//! per-occurrence layer, its moments, VaR and TVaR on an FFT grid, and the
//! statistics of per-occurrence layers, and the occurrence and aggregate
//! PML curves, over `prospicio_aggregate`.
//!
//! The count is in the Panjer class by its mean and dispersion
//! `Var N / E N` (1 Poisson, above 1 negative binomial, below 1 binomial);
//! `aggregate`'s mixing CV `c` is the dispersion `1 + c² E N`. The grid is
//! sized automatically unless `step` and `points` are given, in which case
//! the severity is rounded onto `points` buckets of `step` (its mass above
//! the last lumped there, `aggregate`'s cap) and the total is on `points`
//! too.

use prospicio_aggregate::pml::occurrence_pml;
use prospicio_aggregate::{
    CompoundMoments, LayerStatistics, Sizing, compound_moments, fft, fft_auto,
    occurrence_layer_statistics, round_bucket,
};
use prospicio_prob::{Counting, Distribution, Grid, PanjerClass, SeverityDist};

use crate::pricing::{Result, amount, numbers, scalar, severity};
use crate::sheet::{Grid as Range, Value};

/// The claim count with `mean` and `dispersion` (1 if omitted), as the
/// concrete family `PanjerClass` picks: `PanjerClass` itself does not
/// forward the closed-form third cumulant, and the default it falls back
/// to cancels at large means.
fn count(mean: &Range, dispersion: &Range) -> Result<Box<dyn Counting>> {
    let d = if dispersion.is_blank() {
        1.0
    } else {
        scalar(dispersion, "dispersion")?
    };
    let n = PanjerClass::from_mean_dispersion(scalar(mean, "count_mean")?, d)
        .map_err(|e| e.to_string())?;
    Ok(match n {
        PanjerClass::Binomial(b) => Box::new(b),
        PanjerClass::Poisson(p) => Box::new(p),
        PanjerClass::NegativeBinomial(nb) => Box::new(nb),
    })
}

/// `step` and `points`, both or neither.
fn grid_control(step: &Range, points: &Range) -> Result<Option<(f64, usize)>> {
    match (step.is_blank(), points.is_blank()) {
        (true, true) => Ok(None),
        (false, false) => {
            let h = scalar(step, "step")?;
            let n = scalar(points, "points")?;
            if !(h > 0.0 && h.is_finite()) {
                return Err(format!("step must be positive, got {h}"));
            }
            if !(n >= 2.0 && n <= MAX_POINTS as f64 && n.fract() == 0.0) {
                return Err(format!(
                    "points must be a whole number from 2 to {MAX_POINTS}, got {n}"
                ));
            }
            Ok(Some((h, n as usize)))
        }
        _ => Err("give both step and points, or neither".into()),
    }
}

/// The most points a grid may have: 2^20 (about 8 MB a buffer).
const MAX_POINTS: usize = 1 << 20;

/// A per-occurrence layer, `None` for ground up.
fn layer(limit: &Range, attachment: &Range) -> Result<Option<(f64, f64)>> {
    if limit.is_blank() && attachment.is_blank() {
        return Ok(None);
    }
    let one = |g: &Range, name: &str, unlimited: bool, default: f64| -> Result<f64> {
        if g.is_blank() {
            return Ok(default);
        }
        match g.as_scalar() {
            Some(v) => Ok(amount(v, name, 0, 0, unlimited)?.unwrap_or(default)),
            None => Err(format!("{name} must be a single value")),
        }
    };
    let l = one(limit, "limit", true, f64::INFINITY)?;
    let a = one(attachment, "attachment", false, 0.0)?;
    if l.is_nan() || l <= 0.0 || !(a >= 0.0 && a.is_finite()) {
        return Err(format!(
            "the layer needs a positive limit and a non-negative attachment, got {l} xs {a}"
        ));
    }
    Ok(Some((l, a)))
}

/// The annual total's grid, its moments and how it was built.
struct Total {
    grid: Grid,
    moments: CompoundMoments,
    aliasing_error: f64,
    tail_mass: f64,
}

fn total(
    n: &dyn Counting,
    sev: &SeverityDist,
    layer: Option<(f64, f64)>,
    control: Option<(f64, usize)>,
) -> Result<Total> {
    let e = |e: prospicio_core::Error| e.to_string();
    let image = |g: &Grid| -> Result<Grid> {
        match layer {
            None => Ok(g.clone()),
            Some((l, a)) => g.map(|x| (x - a).clamp(0.0, l)).map(|(g, _)| g).map_err(e),
        }
    };
    let (layer_sev, points) = if let Some((step, points)) = control {
        (
            image(&Grid::rounding(sev, step, points).map_err(e)?.0)?,
            points,
        )
    } else {
        match layer {
            Some((l, a)) if l.is_finite() => {
                // The severity to the layer's top on about 4,096 buckets,
                // then as many points as the total's mean plus 12 SDs needs.
                let mut step = round_bucket((a + l) / 4096.0).map_err(e)?;
                loop {
                    let top = ((a + l) / step).ceil() as usize + 1;
                    let g = image(&Grid::rounding(sev, step, top + 1).map_err(e)?.0)?;
                    let m = compound_moments(n, &g).map_err(e)?;
                    let extent = m.mean + 12.0 * m.variance.sqrt() + l;
                    let points = ((extent / step).ceil() as usize + 1)
                        .next_power_of_two()
                        .max(1 << 10);
                    if points <= MAX_POINTS {
                        break (g, points);
                    }
                    step = round_bucket(2.0 * step).map_err(e)?;
                }
            }
            _ => {
                let (_, _, size) = fft_auto(n, sev, &Sizing::default()).map_err(e)?;
                let (g, _) = Grid::rounding(sev, size.step, size.points).map_err(e)?;
                (image(&g)?, size.points)
            }
        }
    };
    // Ground up, the exact moments when they are finite: the grid's lose
    // what its top lumps of a heavy tail. A layer's are exact on the grid.
    let on_grid = compound_moments(n, &layer_sev).map_err(e)?;
    let moments = match layer {
        None => compound_moments(n, sev)
            .ok()
            .filter(|m| m.third_cumulant.is_finite())
            .unwrap_or(on_grid),
        Some(_) => on_grid,
    };
    let (grid, report) = fft(n, &layer_sev, points).map_err(e)?;
    Ok(Total {
        grid,
        moments,
        aliasing_error: report.aliasing_error,
        tail_mass: report.tail_mass,
    })
}

/// The annual total's mean, SD, CV and skewness, then its VaR and TVaR at
/// each probability, and the grid it was computed on:
/// `PRO.AGG.STATS(count_mean, [dispersion], family, p1, p2, [p3],
/// [limit], [attachment], [probabilities], [step], [points])`. With a
/// limit or attachment the total is of the per-occurrence layer's losses.
/// Ground up the moments are exact when finite (the grid's mean is shown
/// below them), to a layer they are the grid's; VaR and TVaR are the
/// grid's. Probabilities default to 0.99, 0.995 and 0.999.
pub fn stats(args: &[Range]) -> Result<Range> {
    let n = count(&args[0], &args[1])?;
    let sev = severity(&args[2], &args[3], &args[4], &args[5])?;
    let layer = layer(&args[6], &args[7])?;
    let ps = if args[8].is_blank() {
        vec![0.99, 0.995, 0.999]
    } else {
        numbers(&args[8], "probabilities")?
    };
    if let Some(p) = ps.iter().find(|p| !(**p > 0.0 && **p < 1.0)) {
        return Err(format!("probabilities must be in (0, 1), got {p}"));
    }
    let t = total(&*n, &sev, layer, grid_control(&args[9], &args[10])?)?;
    let m = &t.moments;
    let mut rows = vec![
        row("Mean", m.mean),
        row("SD", m.variance.sqrt()),
        row("CV", m.cv()),
        row("Skewness", m.skew()),
    ];
    let e = |e: prospicio_core::Error| e.to_string();
    for &p in &ps {
        rows.push(vec![
            Value::Text(format!("VaR {}", percent(p))),
            Value::number(t.grid.quantile(p).map_err(e)?),
        ]);
    }
    for &p in &ps {
        rows.push(vec![
            Value::Text(format!("TVaR {}", percent(p))),
            Value::number(t.grid.tvar(p).map_err(e)?),
        ]);
    }
    rows.extend([
        row("Grid mean", t.grid.mean()),
        row("Step", t.grid.step()),
        row("Points", t.grid.len() as f64),
        row("Tail mass", t.tail_mass),
        row("Aliasing error", t.aliasing_error),
    ]);
    Ok(Range::from_rows(rows))
}

fn row(label: &str, x: f64) -> Vec<Value> {
    vec![Value::from(label), Value::number(x)]
}

/// `0.995` as `"99.5%"`.
fn percent(p: f64) -> String {
    let s = format!("{:.6}", 100.0 * p);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    format!("{s}%")
}

/// Statistics of per-occurrence layers, one column per layer, as
/// `aggregate`'s `reins_stats_df`: the chances a claim reaches and
/// exhausts the layer and that the year has a loss to it, the excess
/// count's and the layer severity's (given a loss to it) mean, CV and
/// skewness, the
/// annual ceded loss's mean, SD, CV and skewness, and the loss on line:
/// `PRO.AGG.LAYERS(count_mean, [dispersion], family, p1, p2, [p3], limits,
/// attachments, [shares], [step], [points])`.
pub fn layers(args: &[Range]) -> Result<Range> {
    let n = count(&args[0], &args[1])?;
    let sev = severity(&args[2], &args[3], &args[4], &args[5])?;
    let limits: Vec<f64> = cells(&args[6], "limits", true)?;
    let k = limits.len();
    let each = |g: &Range, name: &str, default: Option<f64>| -> Result<Vec<f64>> {
        if g.is_blank() {
            return default
                .map(|d| vec![d; k])
                .ok_or(format!("{name} is required"));
        }
        let v = cells(g, name, false)?;
        match v.len() {
            1 => Ok(vec![v[0]; k]),
            len if len == k => Ok(v),
            len => Err(format!(
                "{name} must be one value, or one per limit ({k}), got {len}"
            )),
        }
    };
    let attachments = each(&args[7], "attachments", None)?;
    let shares = each(&args[8], "shares", Some(1.0))?;
    let e = |e: prospicio_core::Error| e.to_string();
    let grid = match grid_control(&args[9], &args[10])? {
        Some((step, points)) => Grid::rounding(&sev, step, points).map_err(e)?.0,
        None => {
            // Fine enough for the lowest layer and long enough for the top.
            let top = limits
                .iter()
                .zip(&attachments)
                .map(|(l, a)| if l.is_finite() { a + l } else { *a })
                .fold(0.0, f64::max);
            let lowest = limits.iter().cloned().fold(f64::INFINITY, f64::min);
            let reach = if limits.iter().any(|l| l.is_infinite()) {
                sev.quantile(1.0 - 1e-12).map_err(e)?.max(top)
            } else {
                top
            };
            let step = round_bucket((lowest.min(reach) / 1024.0).max(reach / MAX_POINTS as f64))
                .map_err(e)?;
            let points = ((reach / step).ceil() as usize + 2).min(MAX_POINTS);
            Grid::rounding(&sev, step, points).map_err(e)?.0
        }
    };
    let stats = (0..k)
        .map(|j| {
            occurrence_layer_statistics(&*n, &grid, shares[j], limits[j], attachments[j]).map_err(e)
        })
        .collect::<Result<Vec<_>>>()?;
    let label = |name: &str, f: &dyn Fn(&LayerStatistics) -> f64| {
        let mut r = vec![Value::from(name)];
        r.extend(stats.iter().map(|s| Value::number(f(s))));
        r
    };
    let mut header = vec![Value::from("Layer")];
    header.extend((0..k).map(|j| {
        let l = if limits[j].is_finite() {
            format!("{}", limits[j])
        } else {
            "unlimited".into()
        };
        let share = if shares[j] == 1.0 {
            String::new()
        } else {
            format!("{}% of ", 100.0 * shares[j])
        };
        Value::Text(format!("{share}{l} xs {}", attachments[j]))
    }));
    Ok(Range::from_rows(vec![
        header,
        label("Pr attach", &|s| s.pr_attach),
        label("Pr detach", &|s| s.pr_detach),
        label("Pr loss", &|s| s.pr_loss),
        label("Count mean", &|s| s.count.mean),
        label("Count CV", &|s| s.count.cv()),
        label("Count skewness", &|s| s.count.skew()),
        label("Severity mean", &|s| s.severity.mean),
        label("Severity CV", &|s| s.severity.cv()),
        label("Severity skewness", &|s| s.severity.skew()),
        label("Annual mean", &|s| s.aggregate.mean),
        label("Annual SD", &|s| s.aggregate.variance.sqrt()),
        label("Annual CV", &|s| s.aggregate.cv()),
        label("Annual skewness", &|s| s.aggregate.skew()),
        label("Loss on line", &|s| s.loss_on_line),
    ]))
}

/// The occurrence and aggregate PML curves, one row per return period:
/// the annual probability `p`, the occurrence PML (the loss the year's
/// largest occurrence exceeds with probability `p`, exact for any count:
/// `1 − G_N(F(x)) = p`), the aggregate PML (the annual total's VaR at
/// `1 − p`, on the FFT grid), the severity's exceedance at the occurrence
/// PML and the average years between occurrences above it:
/// `PRO.AGG.PML(count_mean, [dispersion], family, p1, p2, [p3],
/// [return_periods], [step], [points])`. A return period below 1 is read
/// as the probability itself.
pub fn pml(args: &[Range]) -> Result<Range> {
    let n = count(&args[0], &args[1])?;
    let sev = severity(&args[2], &args[3], &args[4], &args[5])?;
    let periods = if args[6].is_blank() {
        vec![2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    } else {
        cells(&args[6], "return_periods", false)?
    };
    let ps = periods
        .iter()
        .map(|&t| match t {
            t if t > 1.0 && t.is_finite() => Ok(1.0 / t),
            t if t > 0.0 && t < 1.0 => Ok(t),
            t => Err(format!(
                "return periods must be above 1 (or probabilities in (0, 1)), got {t}"
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    let e = |e: prospicio_core::Error| e.to_string();
    let none = 1.0 - n.pgf(0.0);
    let t = total(&*n, &sev, None, grid_control(&args[7], &args[8])?)?;
    let mut rows = vec![
        [
            "Return period",
            "Probability",
            "Occurrence PML",
            "Aggregate PML",
            "Severity survival",
            "Occurrence return period",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for p in ps {
        if p >= none {
            return Err(format!(
                "the chance of a year with any occurrence is {none}; probability {p} is past it"
            ));
        }
        let o = occurrence_pml(&*n, &sev, p).map_err(e)?;
        rows.push(vec![
            Value::number(o.annual_return_period),
            Value::number(p),
            Value::number(o.loss),
            Value::number(t.grid.quantile(1.0 - p).map_err(e)?),
            Value::number(o.severity_survival),
            Value::number(o.occurrence_return_period),
        ]);
    }
    Ok(Range::from_rows(rows))
}

/// Every cell of a range as a number, `"unlimited"` allowed when asked.
fn cells(g: &Range, name: &str, unlimited: bool) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(g.cells.len());
    for r in 0..g.rows {
        for c in 0..g.columns {
            match amount(g.get(r, c), name, r, c, unlimited)? {
                Some(x) => out.push(x),
                None => return Err(format!("{name} must have a value in every cell")),
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn percents() {
        assert_eq!(super::percent(0.995), "99.5%");
        assert_eq!(super::percent(0.99), "99%");
        assert_eq!(super::percent(0.9999), "99.99%");
    }
}
