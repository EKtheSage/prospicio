//! Aggregate (compound) distributions (`docs/design/xll.md`): the annual
//! total of a claim count and a severity, ground up or to a
//! per-occurrence layer, its moments, VaR and TVaR on an FFT grid, and the
//! statistics of per-occurrence layers and of aggregate covers (a stop
//! loss) on the annual total, a tower's views priced by a distortion,
//! and the occurrence and aggregate PML curves, over
//! `prospicio_aggregate`.
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
    AggregateLayerStatistics, CompoundMoments, Layer, LayerStatistics, Sizing, Tower, ViewPrice,
    aggregate_layer_statistics, compound_moments, fft, fft_auto, occurrence_layer_statistics,
    round_bucket,
};
use prospicio_prob::{Counting, Distortion, Distribution, Grid, PanjerClass, SeverityDist};

use crate::pricing::{Result, amount, numbers, scalar, severity};
use crate::reserving::text_option;
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

/// The severity on the grid (to the per-occurrence layer when given) and
/// the number of points the annual total needs: sized automatically, or
/// rounded onto `points` buckets of `step`.
fn severity_grid(
    n: &dyn Counting,
    sev: &SeverityDist,
    layer: Option<(f64, f64)>,
    control: Option<(f64, usize)>,
) -> Result<(Grid, usize)> {
    let e = |e: prospicio_core::Error| e.to_string();
    let image = |g: &Grid| -> Result<Grid> {
        match layer {
            None => Ok(g.clone()),
            Some((l, a)) => g.map(|x| (x - a).clamp(0.0, l)).map(|(g, _)| g).map_err(e),
        }
    };
    if let Some((step, points)) = control {
        return Ok((
            image(&Grid::rounding(sev, step, points).map_err(e)?.0)?,
            points,
        ));
    }
    match layer {
        Some((l, a)) if l.is_finite() => {
            // The severity to the layer's top on about 4,096 buckets, then
            // as many points as the total's mean plus 12 SDs needs.
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
                    return Ok((g, points));
                }
                step = round_bucket(2.0 * step).map_err(e)?;
            }
        }
        _ => {
            let (_, _, size) = fft_auto(n, sev, &Sizing::default()).map_err(e)?;
            let (g, _) = Grid::rounding(sev, size.step, size.points).map_err(e)?;
            Ok((image(&g)?, size.points))
        }
    }
}

fn total(
    n: &dyn Counting,
    sev: &SeverityDist,
    layer: Option<(f64, f64)>,
    control: Option<(f64, usize)>,
) -> Result<Total> {
    let e = |e: prospicio_core::Error| e.to_string();
    let (layer_sev, points) = severity_grid(n, sev, layer, control)?;
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
pub(crate) fn percent(p: f64) -> String {
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
    let attachments = per_limit(&args[7], "attachments", k, None)?;
    let shares = per_limit(&args[8], "shares", k, Some(1.0))?;
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
    Ok(Range::from_rows(vec![
        header(&limits, &attachments, &shares),
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

/// The header row of a table with a column per layer: `"Layer"`, then
/// each layer's [`layer_name`].
fn header(limits: &[f64], attachments: &[f64], shares: &[f64]) -> Vec<Value> {
    let mut header = vec![Value::from("Layer")];
    header.extend(
        limits
            .iter()
            .zip(attachments)
            .zip(shares)
            .map(|((&l, &a), &c)| Value::Text(layer_name(l, a, c))),
    );
    header
}

/// `"[share% of ]limit xs attachment"`.
fn layer_name(limit: f64, attachment: f64, share: f64) -> String {
    let l = if limit.is_finite() {
        format!("{limit}")
    } else {
        "unlimited".into()
    };
    let share = if share == 1.0 {
        String::new()
    } else {
        format!("{}% of ", 100.0 * share)
    };
    format!("{share}{l} xs {attachment}")
}

/// One value per layer from `g`: one value repeated, or one per limit;
/// `default` when blank, or an error naming `name` as required.
fn per_limit(g: &Range, name: &str, k: usize, default: Option<f64>) -> Result<Vec<f64>> {
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
}

/// Statistics of aggregate covers on the annual total, one column per
/// cover, as `aggregate`'s `reins_stats_df` for its `agg` layers: the
/// subject total's mean, the chances the year reaches and exhausts the
/// cover, the annual ceded loss's mean, SD, CV and skewness, and the loss
/// on line: `PRO.AGG.COVER(count_mean, [dispersion], family, p1, p2,
/// [p3], limits, retentions, [shares], [occ_limits], [occ_attachments],
/// [occ_shares], [subject], [step], [points])`. A stop loss is `limits`
/// xs `retentions` of the ground-up total; with per-occurrence layers the
/// covers sit on the total `net` of them (the default, `aggregate`'s
/// `occurrence net of ... aggregate ceded to`) or on their `ceded` total
/// (an annual deductible and limit on an excess of loss treaty). Every
/// statistic is the grid's, sized as for `PRO.AGG.STATS` ground up.
pub fn cover(args: &[Range]) -> Result<Range> {
    let n = count(&args[0], &args[1])?;
    let sev = severity(&args[2], &args[3], &args[4], &args[5])?;
    let limits = cells(&args[6], "limits", true)?;
    let k = limits.len();
    let retentions = per_limit(&args[7], "retentions", k, None)?;
    let shares = per_limit(&args[8], "shares", k, Some(1.0))?;
    let occ = if args[9].is_blank() && args[10].is_blank() {
        Vec::new()
    } else {
        let limits = cells(&args[9], "occ_limits", true)?;
        let m = limits.len();
        let attachments = per_limit(&args[10], "occ_attachments", m, None)?;
        let shares = per_limit(&args[11], "occ_shares", m, Some(1.0))?;
        (0..m)
            .map(|j| {
                Layer::xol(format!("occurrence {}", j + 1), limits[j], attachments[j])
                    .and_then(|l| l.share(shares[j]))
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>>>()?
    };
    let ceded = match text_option(&args[12], "subject")?.as_deref() {
        None | Some("net") => false,
        Some("ceded") => true,
        Some(other) => {
            return Err(format!(
                "subject must be \"net\" (default) or \"ceded\", got {other:?}"
            ));
        }
    };
    if ceded && occ.is_empty() {
        return Err("subject \"ceded\" needs occurrence layers (occ_limits)".into());
    }
    let e = |e: prospicio_core::Error| e.to_string();
    let subject = if occ.is_empty() {
        total(&*n, &sev, None, grid_control(&args[13], &args[14])?)?.grid
    } else {
        let (grid, points) = severity_grid(&*n, &sev, None, grid_control(&args[13], &args[14])?)?;
        let grids = Tower::new(occ)
            .map_err(e)?
            .on_grid(&*n, &grid, points)
            .map_err(e)?;
        if ceded {
            grids.stage_ceded[0]
                .clone()
                .ok_or("the occurrence layers' total cession is not on one grid")?
        } else {
            grids
                .net
                .ok_or("the net of the occurrence layers is not on one grid")?
        }
    };
    let stats = (0..k)
        .map(|j| {
            aggregate_layer_statistics(&subject, shares[j], limits[j], retentions[j]).map_err(e)
        })
        .collect::<Result<Vec<_>>>()?;
    let label = |name: &str, f: &dyn Fn(&AggregateLayerStatistics) -> f64| {
        let mut r = vec![Value::from(name)];
        r.extend(stats.iter().map(|s| Value::number(f(s))));
        r
    };
    let mean = subject.mean();
    Ok(Range::from_rows(vec![
        header(&limits, &retentions, &shares),
        label("Subject mean", &|_| mean),
        label("Pr attach", &|s| s.pr_attach),
        label("Pr detach", &|s| s.pr_detach),
        label("Annual mean", &|s| s.aggregate.mean),
        label("Annual SD", &|s| s.aggregate.variance.sqrt()),
        label("Annual CV", &|s| s.aggregate.cv()),
        label("Annual skewness", &|s| s.aggregate.skew()),
        label("Loss on line", &|s| s.loss_on_line),
    ]))
}

/// A distortion by `aggregate`'s short name (or its long one) and its
/// parameter.
/// A distortion by name, any of `PRO.DISTORTION`'s families; a capped
/// family takes `r0` and its parameter together in `parameter`.
fn distortion(name: &Range, parameter: &Range) -> Result<Distortion> {
    if name.is_blank() {
        return Err("distortion is required".into());
    }
    crate::distortions::parse(name, parameter, &Range::missing())
}

/// A tower on the annual grid, each view priced on its own by a
/// distortion, as `aggregate`'s `reins_price_df`: one row each for the
/// gross, every per-occurrence layer, the layers' total when there are
/// several, the aggregate cover and the net, with the expected loss, SD,
/// CV, the premium `ρ_g(X)`, margin, loss ratio and, for a layer or the
/// cover, the rate on line (premium over share times limit):
/// `PRO.TOWER.PRICE(count_mean, [dispersion], family, p1, p2, [p3],
/// [limit], [attachment], limits, attachments, [shares],
/// [agg_deductibles], [agg_limits], [cover_limit], [cover_retention],
/// distortion, parameter, [step], [points])`. `limit` and `attachment`
/// are the gross per-claim terms (a policy limit, `PRO.AGG.STATS`'s);
/// the layers' annual deductibles and limits apply to each layer's annual
/// recoveries; the cover is a stop loss on the annual total net of the
/// layers. Every view is exact on the grid when the boundaries fall on
/// its points; otherwise means stay exact and shapes are smeared by up to
/// a step.
pub fn tower_price(args: &[Range]) -> Result<Range> {
    let n = count(&args[0], &args[1])?;
    let sev = severity(&args[2], &args[3], &args[4], &args[5])?;
    let gross = layer(&args[6], &args[7])?;
    let limits = cells(&args[8], "limits", true)?;
    let k = limits.len();
    let attachments = per_limit(&args[9], "attachments", k, None)?;
    let shares = per_limit(&args[10], "shares", k, Some(1.0))?;
    let deductibles = per_limit(&args[11], "agg_deductibles", k, Some(0.0))?;
    let agg_limits = if args[12].is_blank() {
        vec![f64::INFINITY; k]
    } else {
        let v = cells(&args[12], "agg_limits", true)?;
        match v.len() {
            1 => vec![v[0]; k],
            len if len == k => v,
            len => {
                return Err(format!(
                    "agg_limits must be one value, or one per limit ({k}), got {len}"
                ));
            }
        }
    };
    let cover = layer(&args[13], &args[14])?;
    if !args[13].is_blank() && args[14].is_blank() {
        return Err("give the cover's retention with its limit".into());
    }
    let g = distortion(&args[15], &args[16])?;
    let e = |e: prospicio_core::Error| e.to_string();
    let occurrence = (0..k)
        .map(|j| {
            let mut l = Layer::xol(format!("layer {}", j + 1), limits[j], attachments[j])?
                .share(shares[j])?;
            if deductibles[j] > 0.0 {
                l = l.aggregate_deductible(deductibles[j])?;
            }
            if agg_limits[j].is_finite() {
                l = l.aggregate_limit(agg_limits[j])?;
            }
            Ok(l)
        })
        .collect::<prospicio_core::Result<Vec<_>>>()
        .map_err(e)?;
    let tower = match cover {
        None => Tower::new(occurrence),
        Some((l, r)) => Tower::inuring(vec![
            occurrence,
            vec![Layer::stop_loss("cover", l, r).map_err(e)?],
        ]),
    }
    .map_err(e)?;
    let (grid, points) = severity_grid(&*n, &sev, gross, grid_control(&args[17], &args[18])?)?;
    let grids = tower.on_grid(&*n, &grid, points).map_err(e)?;
    let price = grids.price(&g);
    let mut rows = vec![
        [
            "View",
            "Loss",
            "SD",
            "CV",
            "Premium",
            "Margin",
            "Loss ratio",
            "Rate on line",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    view(&mut rows, "Gross", &grids.gross, &price.gross, None);
    for j in 0..k {
        let line = shares[j] * limits[j].min(agg_limits[j]);
        let name = layer_name(limits[j], attachments[j], shares[j]);
        view(
            &mut rows,
            &name,
            &grids.ceded[j],
            &price.ceded[j],
            Some(line),
        );
    }
    let not_one = "not one compound: a layer has annual terms";
    if k > 1 {
        match (&grids.stage_ceded[0], &price.stages[0]) {
            (Some(grid), Some(p)) => view(&mut rows, "Occurrence total", grid, p, None),
            _ => rows.push(note("Occurrence total", not_one)),
        }
    }
    if let Some((l, r)) = cover {
        let name = format!("Aggregate cover {}", layer_name(l, r, 1.0));
        view(&mut rows, &name, &grids.ceded[k], &price.ceded[k], Some(l));
    }
    match (&grids.net, &price.net) {
        (Some(grid), Some(p)) => view(&mut rows, "Net", grid, p, None),
        _ => rows.push(note("Net", not_one)),
    }
    Ok(Range::from_rows(rows))
}

/// One view's row of `PRO.TOWER.PRICE`; `line` is the limit its rate on
/// line is over, if any.
fn view(rows: &mut Vec<Vec<Value>>, label: &str, grid: &Grid, p: &ViewPrice, line: Option<f64>) {
    let sd = grid.variance().sqrt();
    rows.push(vec![
        Value::from(label),
        Value::number(p.loss),
        Value::number(sd),
        Value::number(sd / p.loss),
        Value::number(p.premium),
        Value::number(p.margin),
        Value::number(p.loss_ratio),
        line.filter(|l| l.is_finite())
            .map_or(Value::Empty, |l| Value::number(p.premium / l)),
    ]);
}

/// A row with a label and a note in place of its numbers.
fn note(label: &str, text: &str) -> Vec<Value> {
    let mut r = vec![Value::from(label), Value::from(text)];
    r.extend(std::iter::repeat_n(Value::Empty, 6));
    r
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

/// A layer's contract terms in expectation over its annual loss on the
/// grid: `PRO.LAYER.TERMS(count_mean, [dispersion], family, p1, p2, [p3],
/// layer_limit, layer_attachment, premium, [reinstatements],
/// [commissions], [loss_ratios], [profit_share], [profit_allowance],
/// [share], [policy_limit], [step], [points])`. `premium` is the deposit
/// at 100%; `reinstatements` a row of rates (each a share of the deposit,
/// pro rata as to amount), which also sets the annual limit to the limit
/// times one more than their number; `commissions` one flat rate, or with
/// `loss_ratios` a sliding scale; a profit commission of `profit_share`
/// after `profit_allowance`. The rows are the expected ceded loss, the
/// premium, the reinstatement premium, the ceding and profit commissions,
/// the premium net of them, the margin over the loss and the loss ratio
/// on the premium with reinstatements.
pub fn layer_terms(args: &[Range]) -> Result<Range> {
    let n = count(&args[0], &args[1])?;
    let sev = severity(&args[2], &args[3], &args[4], &args[5])?;
    let (limit, attachment) =
        layer(&args[6], &args[7])?.ok_or("the layer needs layer_limit and layer_attachment")?;
    let premium = scalar(&args[8], "premium")?;
    let share = if args[14].is_blank() {
        1.0
    } else {
        scalar(&args[14], "share")?
    };
    let gross = layer(&args[15], &Range::missing())?;
    let e = |e: prospicio_core::Error| e.to_string();
    let mut l = Layer::xol("layer", limit, attachment)
        .and_then(|l| l.share(share))
        .map_err(e)?;
    l = if args[9].is_blank() {
        l.deposit_premium(premium).map_err(e)?
    } else {
        let rates = numbers(&args[9], "reinstatements")?;
        l.paid_reinstatements(share * premium, rates).map_err(e)?
    };
    if !args[10].is_blank() {
        l = if args[11].is_blank() {
            l.ceding_commission(scalar(&args[10], "commissions")?)
        } else {
            l.sliding_scale(crate::terms::anchors(&args[10], &args[11])?)
        }
        .map_err(e)?;
    } else if !args[11].is_blank() {
        return Err("loss_ratios go with commissions, for a sliding scale".into());
    }
    match (args[12].is_blank(), args[13].is_blank()) {
        (true, true) => {}
        (false, false) => {
            l = l
                .profit_commission(
                    scalar(&args[12], "profit_share")?,
                    scalar(&args[13], "profit_allowance")?,
                )
                .map_err(e)?;
        }
        _ => return Err("give profit_share and profit_allowance together".into()),
    }
    let tower = Tower::new(vec![l]).map_err(e)?;
    let (grid, points) = severity_grid(&*n, &sev, gross, grid_control(&args[16], &args[17])?)?;
    let grids = tower.on_grid(&*n, &grid, points).map_err(e)?;
    let loss = grids.ceded[0].mean();
    let p = grids.expected_premium[0];
    let rip = grids.expected_reinstatement_premium[0];
    let cc = grids.expected_ceding_commission[0];
    let pc = grids.expected_profit_commission[0];
    let net = p + rip - cc - pc;
    Ok(Range::from_rows(vec![
        row("Expected loss", loss),
        row("Premium", p),
        row("Reinstatement premium", rip),
        row("Ceding commission", cc),
        row("Profit commission", pc),
        row("Net premium", net),
        row("Margin", net - loss),
        row("Loss ratio", loss / (p + rip)),
    ]))
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
