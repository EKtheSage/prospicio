//! The severity functions: a severity's cdf, survival, quantile and
//! moments, and its limited expected values, layer losses, a layer's CV,
//! increased limit factors and loss elimination ratios, for the families
//! in [`FAMILIES`] (`docs/design/xll.md`).
//!
//! Limits and attachments may be single values or ranges. Two ranges
//! combine cell by cell, and a row against a column makes a table, as
//! Excel's own arithmetic does (`=A1:A5 + B1:D1`): a column of
//! attachments and a row of limits give every layer in one formula. A
//! blank cell gives a blank result; a limit of `"unlimited"` is no limit.

use prospicio_pricing::layer::{
    XsLayer, alpha_between_frequencies, alpha_between_frequency_and_layer, alpha_between_layers,
    loss_elimination_ratio, pareto_extrapolation,
};
use prospicio_pricing::layer::{ilf, layer_cv};
use prospicio_prob::evt::Gpd;
use prospicio_prob::{
    Burr, Dist, Distribution, Exponential, Gamma, InverseGamma, InverseGaussian, Loglogistic,
    Lognormal, Pareto, PiecewisePareto, Severity, SeverityDist, Truncation, Weibull,
};

use crate::sheet::{Grid, Value, position};

pub(crate) type Result<T> = std::result::Result<T, String>;

/// One number argument, which must be a single value.
pub(crate) fn scalar(grid: &Grid, name: &str) -> Result<f64> {
    match grid.as_scalar() {
        Some(Value::Number(x)) => Ok(*x),
        Some(v) if v.is_blank() => Err(format!("{name} is required")),
        _ => Err(format!("{name} must be a single number")),
    }
}

/// The families `family` names, and what `p1`, `p2` and `p3` are for each.
pub const FAMILIES: &str = "pareto (t, alpha, [truncation]), lognormal (meanlog, sdlog), \
lognormal_cv (mean, cv), gamma (shape, scale), gamma_cv (mean, cv), weibull (shape, scale), \
loglogistic (shape, scale), exponential (mean), inverse_gamma (shape, scale), inverse_gaussian (mean, shape), \
burr (alpha, gamma, scale), lomax (alpha, scale), gpd (xi, beta), piecewise_pareto (thresholds, alphas, \
[truncation, [\"lp\" or \"wd\"]]), gen_pareto (t, alpha_ini, alpha_tail)";

/// The numbers of a range, in reading order, skipping blank cells.
pub(crate) fn numbers(grid: &Grid, name: &str) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(grid.cells.len());
    for r in 0..grid.rows {
        for c in 0..grid.columns {
            match grid.get(r, c) {
                Value::Number(x) => out.push(*x),
                v if v.is_blank() => {}
                other => {
                    return Err(format!(
                        "{name} {} must be a number, got {other:?}",
                        position(r, c)
                    ));
                }
            }
        }
    }
    if out.is_empty() {
        return Err(format!("{name} is required"));
    }
    Ok(out)
}

/// Parallel ranges that line up row by row, read together, one `Vec` per
/// range. A position blank in every range is dropped (the empty end of a
/// selected range); a position blank in some ranges and not in others is
/// an error naming the range and the cell, since dropping each range's
/// blanks on its own would pair the values with the wrong partners. A
/// range shorter than another counts as blank past its end, so a tail
/// blank in every range is dropped as before, and a number past the end of
/// a shorter range is an error.
pub(crate) fn paired(ranges: &[(&Grid, &str)]) -> Result<Vec<Vec<f64>>> {
    let first_name = ranges[0].1;
    let longest = ranges.iter().map(|(g, _)| g.cells.len()).max().unwrap_or(0);
    let mut out = vec![Vec::with_capacity(longest); ranges.len()];
    for i in 0..longest {
        let mut row = Vec::with_capacity(ranges.len());
        for (g, name) in ranges {
            row.push(match g.cells.get(i) {
                None => None,
                Some(Value::Number(x)) => Some(*x),
                Some(v) if v.is_blank() => None,
                Some(other) => {
                    return Err(format!(
                        "{name} {} must be a number, got {other:?}",
                        position(i / g.columns, i % g.columns)
                    ));
                }
            });
        }
        if row.iter().all(Option::is_none) {
            continue;
        }
        if let Some(k) = row.iter().position(Option::is_none) {
            let given = row.iter().position(Option::is_some).unwrap();
            let (gk, name_k) = ranges[k];
            let (gg, name_g) = ranges[given];
            let at = position(i / gg.columns, i % gg.columns);
            return Err(if i < gk.cells.len() {
                format!(
                    "{name_k} {} is blank but {name_g} is not: give a value in both or leave both blank",
                    position(i / gk.columns, i % gk.columns)
                )
            } else {
                format!(
                    "{name_k} has {} cells, too few for {name_g} {at}: give one value per row",
                    gk.cells.len()
                )
            });
        }
        for (o, x) in out.iter_mut().zip(row) {
            o.push(x.unwrap());
        }
    }
    if out[0].is_empty() {
        return Err(format!("{first_name} is required"));
    }
    Ok(out)
}

/// A piecewise Pareto's truncation: a point, or a point and its kind
/// (`"lp"`, the last piece only, R's default, or `"wd"`, the whole
/// distribution) in two cells.
pub(crate) fn piecewise_truncation(p3: &Grid) -> Result<(f64, Truncation)> {
    let cells: Vec<&Value> = p3.cells.iter().filter(|v| !v.is_blank()).collect();
    let point = match cells.first() {
        Some(Value::Number(x)) => *x,
        _ => {
            return Err(
                "p3 (truncation) must be a number, then optionally \"lp\" or \"wd\"".into(),
            );
        }
    };
    let kind = match cells.get(1) {
        None => Truncation::LastPiece,
        Some(Value::Text(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "lp" | "last_piece" => Truncation::LastPiece,
            "wd" | "whole_distribution" => Truncation::WholeDistribution,
            other => {
                return Err(format!(
                    "the truncation kind must be \"lp\" or \"wd\", got {other:?}"
                ));
            }
        },
        Some(other) => {
            return Err(format!(
                "the truncation kind must be \"lp\" or \"wd\", got {other:?}"
            ));
        }
    };
    if cells.len() > 2 {
        return Err("p3 takes a truncation point and at most its kind".into());
    }
    Ok((point, kind))
}

/// The severity named by `family` with parameters `p1`, `p2` and `p3`
/// (see [`FAMILIES`]): a `prospicio_prob` distribution with exact limited
/// moments.
pub(crate) fn severity(family: &Grid, p1: &Grid, p2: &Grid, p3: &Grid) -> Result<SeverityDist> {
    let name = match family.as_scalar() {
        Some(Value::Text(s)) => s.trim().to_ascii_lowercase().replace([' ', '-'], "_"),
        _ => return Err(format!("family must be one of: {FAMILIES}")),
    };
    let e = |e: prospicio_core::Error| e.to_string();
    let two = || -> Result<(f64, f64)> { Ok((scalar(p1, "p1")?, scalar(p2, "p2")?)) };
    let no_p3 = |dist: Dist| -> Result<Dist> {
        if p3.is_blank() {
            Ok(dist)
        } else {
            Err(format!(
                "p3 is taken only by \"pareto\", \"burr\", \"piecewise_pareto\" and \"gen_pareto\", not {name:?}"
            ))
        }
    };
    let dist: Dist = match name.as_str() {
        "pareto" => {
            let (t, alpha) = two()?;
            let p = Pareto::new(t, alpha).map_err(e)?;
            if p3.is_blank() {
                p.into()
            } else {
                p.truncated(scalar(p3, "p3 (truncation)")?)
                    .map_err(e)?
                    .into()
            }
        }
        "lognormal" => no_p3(
            two()
                .and_then(|(m, s)| Lognormal::new(m, s).map_err(e))?
                .into(),
        )?,
        "lognormal_cv" => no_p3(
            two()
                .and_then(|(m, c)| Lognormal::from_mean_cv(m, c).map_err(e))?
                .into(),
        )?,
        "gamma" => no_p3(two().and_then(|(k, t)| Gamma::new(k, t).map_err(e))?.into())?,
        "gamma_cv" => no_p3(
            two()
                .and_then(|(m, c)| Gamma::from_mean_cv(m, c).map_err(e))?
                .into(),
        )?,
        "weibull" => no_p3(
            two()
                .and_then(|(k, l)| Weibull::new(k, l).map_err(e))?
                .into(),
        )?,
        "loglogistic" | "log_logistic" => no_p3(
            two()
                .and_then(|(k, t)| Loglogistic::new(k, t).map_err(e))?
                .into(),
        )?,
        "exponential" => {
            if !p2.is_blank() {
                return Err("\"exponential\" takes only p1, the mean: leave p2 blank".into());
            }
            no_p3(
                Exponential::new(scalar(p1, "p1 (mean)")?)
                    .map_err(e)?
                    .into(),
            )?
        }
        "inverse_gamma" => no_p3(
            two()
                .and_then(|(k, t)| InverseGamma::new(k, t).map_err(e))?
                .into(),
        )?,
        "inverse_gaussian" => no_p3(
            two()
                .and_then(|(m, l)| InverseGaussian::new(m, l).map_err(e))?
                .into(),
        )?,
        "lomax" => no_p3(
            two()
                .and_then(|(alpha, scale)| Burr::new(alpha, 1.0, scale).map_err(e))?
                .into(),
        )?,
        "burr" => {
            let (alpha, gamma) = two()?;
            Burr::new(alpha, gamma, scalar(p3, "p3 (scale)")?)
                .map_err(e)?
                .into()
        }
        "gpd" | "generalized_pareto" => no_p3(
            two()
                .and_then(|(xi, beta)| Gpd::new(xi, beta).map_err(e))?
                .into(),
        )?,
        "piecewise_pareto" => {
            let [thresholds, alphas] =
                <[_; 2]>::try_from(paired(&[(p1, "p1 (thresholds)"), (p2, "p2 (alphas)")])?)
                    .unwrap();
            let p = PiecewisePareto::new(thresholds, alphas).map_err(e)?;
            if p3.is_blank() {
                p.into()
            } else {
                let (point, kind) = piecewise_truncation(p3)?;
                p.truncated(point, kind).map_err(e)?.into()
            }
        }
        "gen_pareto" => {
            let (t, alpha_ini) = two()?;
            Gpd::riegel(t, alpha_ini, scalar(p3, "p3 (alpha_tail)")?)
                .map_err(e)?
                .into()
        }
        other => return Err(format!("family must be one of: {FAMILIES}; got {other:?}")),
    };
    SeverityDist::try_from(dist).map_err(|_| "not a severity".into())
}

/// One cell of a limit or attachment range: `None` for a blank.
pub(crate) fn amount(
    v: &Value,
    name: &str,
    r: usize,
    c: usize,
    unlimited: bool,
) -> Result<Option<f64>> {
    match v {
        Value::Number(x) => Ok(Some(*x)),
        v if v.is_blank() => Ok(None),
        Value::Text(s)
            if unlimited
                && matches!(s.trim().to_ascii_lowercase().as_str(), "unlimited" | "inf") =>
        {
            Ok(Some(f64::INFINITY))
        }
        other => Err(format!(
            "{name} {} must be a number{}, got {other:?}",
            position(r, c),
            if unlimited { " or \"unlimited\"" } else { "" }
        )),
    }
}

/// The size two ranges combine to: equal sizes, or 1 along a side
/// stretched to the other's, as Excel's arithmetic does.
fn combined(a: &Grid, b: &Grid, names: (&str, &str)) -> Result<(usize, usize)> {
    let side = |x: usize, y: usize| match (x, y) {
        _ if x == y => Some(x),
        (1, y) => Some(y),
        (x, 1) => Some(x),
        _ => None,
    };
    match (side(a.rows, b.rows), side(a.columns, b.columns)) {
        (Some(r), Some(c)) => Ok((r, c)),
        _ => Err(format!(
            "{} ({} x {}) and {} ({} x {}) do not combine: give ranges of the same size, \
             or a row and a column",
            names.0, a.rows, a.columns, names.1, b.rows, b.columns
        )),
    }
}

/// Cell `(r, c)` of `g` stretched along a side of length 1.
pub(crate) fn at(g: &Grid, r: usize, c: usize) -> &Value {
    g.get(
        if g.rows == 1 { 0 } else { r },
        if g.columns == 1 { 0 } else { c },
    )
}

/// `f(x, y)` over two broadcast ranges; blank where either cell is.
fn map2(
    xs: &Grid,
    ys: &Grid,
    names: (&str, &str),
    unlimited: (bool, bool),
    f: impl Fn(f64, f64) -> Result<f64>,
) -> Result<Grid> {
    let (rows, columns) = combined(xs, ys, names)?;
    let mut cells = Vec::with_capacity(rows * columns);
    for r in 0..rows {
        for c in 0..columns {
            let x = amount(at(xs, r, c), names.0, r, c, unlimited.0)?;
            let y = amount(at(ys, r, c), names.1, r, c, unlimited.1)?;
            cells.push(match (x, y) {
                (Some(x), Some(y)) => Value::number(f(x, y)?),
                _ => Value::Empty,
            });
        }
    }
    Ok(Grid {
        rows,
        columns,
        cells,
    })
}

fn check_layer(limit: f64, attachment: f64) -> Result<()> {
    if limit.is_nan() || limit <= 0.0 || !(attachment.is_finite() && attachment >= 0.0) {
        return Err(format!(
            "the layer {limit} xs {attachment} needs a positive limit and a finite, \
             non-negative attachment"
        ));
    }
    Ok(())
}

/// Limited expected value `E[min(X, limit)]` per limit:
/// `PRO.PRC.LEV(limit, family, p1, p2, [truncation])`.
pub fn lev(args: &[Grid]) -> Result<Grid> {
    let sev = severity(&args[1], &args[2], &args[3], &args[4])?;
    let limits = &args[0];
    let cells = (0..limits.cells.len())
        .map(|i| {
            let (r, c) = (i / limits.columns, i % limits.columns);
            Ok(match amount(limits.get(r, c), "limit", r, c, true)? {
                Some(l) if l.is_nan() || l <= 0.0 => {
                    return Err(format!("a limit must be positive, got {l}"));
                }
                Some(l) => Value::number(sev.lev(l)),
                None => Value::Empty,
            })
        })
        .collect::<Result<_>>()?;
    Ok(Grid {
        rows: limits.rows,
        columns: limits.columns,
        cells,
    })
}

/// Expected loss to the layer `limit` xs `attachment` per ground-up loss:
/// `PRO.PRC.LAYER(limit, attachment, family, p1, p2, [truncation])`.
pub fn layer(args: &[Grid]) -> Result<Grid> {
    let sev = severity(&args[2], &args[3], &args[4], &args[5])?;
    map2(
        &args[0],
        &args[1],
        ("limit", "attachment"),
        (true, false),
        |l, a| {
            check_layer(l, a)?;
            Ok(sev.layer(l, a))
        },
    )
}

/// Variance of the loss to the layer `limit` xs `attachment` per loss,
/// `Var[min((X - attachment)+, limit)]`:
/// `PRO.PRC.LAYER.VAR(limit, attachment, family, p1, p2, [p3])`. A limit of
/// `"unlimited"` gives the stop loss's variance, infinite when the second
/// moment is.
pub fn layer_variance(args: &[Grid]) -> Result<Grid> {
    let sev = severity(&args[2], &args[3], &args[4], &args[5])?;
    map2(
        &args[0],
        &args[1],
        ("limit", "attachment"),
        (true, false),
        |l, a| {
            check_layer(l, a)?;
            Ok(sev.layer_variance(l, a))
        },
    )
}

/// CV of the loss to the layer given a loss above its attachment:
/// `PRO.PRC.LAYER.CV(limit, attachment, family, p1, p2, [truncation])`.
pub fn layer_coefficient_of_variation(args: &[Grid]) -> Result<Grid> {
    let sev = severity(&args[2], &args[3], &args[4], &args[5])?;
    map2(
        &args[0],
        &args[1],
        ("limit", "attachment"),
        (true, false),
        |l, a| layer_cv(&sev, l, a).map_err(|e| e.to_string()),
    )
}

/// Increased limit factor `LEV(limit) / LEV(basic_limit)`:
/// `PRO.PRC.ILF(limit, basic_limit, family, p1, p2, [truncation])`.
pub fn increased_limit_factor(args: &[Grid]) -> Result<Grid> {
    let sev = severity(&args[2], &args[3], &args[4], &args[5])?;
    map2(
        &args[0],
        &args[1],
        ("limit", "basic_limit"),
        (true, false),
        |l, b| ilf(&sev, l, b).map_err(|e| e.to_string()),
    )
}

/// `f(x)` over one range, blank where a cell is; `name` for errors.
pub(crate) fn map1(
    xs: &Grid,
    name: &str,
    unlimited: bool,
    f: impl Fn(f64) -> Result<f64>,
) -> Result<Grid> {
    let mut cells = Vec::with_capacity(xs.cells.len());
    for r in 0..xs.rows {
        for c in 0..xs.columns {
            cells.push(match amount(xs.get(r, c), name, r, c, unlimited)? {
                Some(x) => Value::number(f(x)?),
                None => Value::Empty,
            });
        }
    }
    Ok(Grid {
        rows: xs.rows,
        columns: xs.columns,
        cells,
    })
}

/// The distribution function `P(X <= x)` at each `x`:
/// `PRO.SEV.CDF(x, family, p1, p2, [p3])`.
pub fn cdf(args: &[Grid]) -> Result<Grid> {
    let sev = severity(&args[1], &args[2], &args[3], &args[4])?;
    map1(&args[0], "x", false, |x| Ok(sev.cdf(x)))
}

/// The survival function `P(X > x)` at each `x`:
/// `PRO.SEV.SURVIVAL(x, family, p1, p2, [p3])`.
pub fn survival(args: &[Grid]) -> Result<Grid> {
    let sev = severity(&args[1], &args[2], &args[3], &args[4])?;
    map1(&args[0], "x", false, |x| Ok(sev.survival(x)))
}

/// The quantile (VaR) at each probability `p`:
/// `PRO.SEV.QUANTILE(p, family, p1, p2, [p3])`.
pub fn quantile(args: &[Grid]) -> Result<Grid> {
    let sev = severity(&args[1], &args[2], &args[3], &args[4])?;
    map1(&args[0], "p", false, |p| {
        sev.quantile(p).map_err(|e| e.to_string())
    })
}

/// A severity's mean, standard deviation and CV, a name and a value per
/// row (blank where a moment is infinite):
/// `PRO.SEV.STATS(family, p1, p2, [p3])`.
pub fn stats(args: &[Grid]) -> Result<Grid> {
    let sev = severity(&args[0], &args[1], &args[2], &args[3])?;
    let (mean, sd) = (sev.mean(), sev.std_dev());
    Ok(Grid::from_rows(vec![
        vec![Value::from("Family"), Value::from(sev.dist().family())],
        vec![Value::from("Mean"), Value::number(mean)],
        vec![Value::from("SD"), Value::number(sd)],
        vec![Value::from("CV"), Value::number(sd / mean)],
    ]))
}

/// The loss elimination ratio of a deductible, `LEV(d) / E[X]`, the
/// credit for a straight deductible, at each `deductible`:
/// `PRO.PRC.LER(deductible, family, p1, p2, [p3])`.
pub fn ler(args: &[Grid]) -> Result<Grid> {
    let sev = severity(&args[1], &args[2], &args[3], &args[4])?;
    map1(&args[0], "deductible", false, |d| {
        loss_elimination_ratio(&sev, d).map_err(|e| e.to_string())
    })
}

/// An optional `truncation` argument of the Pareto helpers: omitted (or
/// "unlimited") for none.
fn truncation(grid: &Grid) -> Result<Option<f64>> {
    match grid.as_scalar() {
        Some(v) if v.is_blank() => Ok(None),
        Some(v) => match amount(v, "truncation", 0, 0, true)? {
            Some(x) if x.is_infinite() => Ok(None),
            x => Ok(x),
        },
        None => Err("truncation must be a single number".into()),
    }
}

/// A layer `limit` xs `attachment` for the Pareto helpers; `limit` may be
/// "unlimited".
fn xs_layer(limit: &Grid, attachment: &Grid, which: &str) -> Result<XsLayer> {
    let l = match limit.as_scalar() {
        Some(v) => amount(v, &format!("{which} limit"), 0, 0, true)?,
        None => None,
    }
    .ok_or_else(|| format!("{which} limit must be a single number or \"unlimited\""))?;
    XsLayer::new(l, scalar(attachment, &format!("{which} attachment"))?).map_err(|e| e.to_string())
}

/// The expected loss of each `to` layer per unit of the `from` layer's,
/// under a Pareto with this `alpha` (R's `Pareto_Extrapolation`); the `to`
/// limits and attachments broadcast as `PRO.PRC.LAYER`'s:
/// `PRO.PRC.PARETO.EXTRAPOLATE(from_limit, from_attachment, to_limit, to_attachment, alpha, [truncation])`.
pub fn pareto_extrapolate(args: &[Grid]) -> Result<Grid> {
    let from = xs_layer(&args[0], &args[1], "from")?;
    let alpha = scalar(&args[4], "alpha")?;
    let t = truncation(&args[5])?;
    map2(
        &args[2],
        &args[3],
        ("to_limit", "to_attachment"),
        (true, false),
        |l, a| {
            let to = XsLayer::new(l, a).map_err(|e| e.to_string())?;
            pareto_extrapolation(from, to, alpha, t).map_err(|e| e.to_string())
        },
    )
}

/// The Pareto alpha at which two layers have the given expected losses
/// (R's `Pareto_Find_Alpha_btw_Layers`):
/// `PRO.PRC.PARETO.ALPHA(limit1, attachment1, loss1, limit2, attachment2, loss2, [truncation])`.
pub fn pareto_alpha_layers(args: &[Grid]) -> Result<Grid> {
    let a = (
        xs_layer(&args[0], &args[1], "first")?,
        scalar(&args[2], "loss1")?,
    );
    let b = (
        xs_layer(&args[3], &args[4], "second")?,
        scalar(&args[5], "loss2")?,
    );
    let alpha = alpha_between_layers(a, b, truncation(&args[6])?).map_err(|e| e.to_string())?;
    Ok(Grid::scalar(Value::number(alpha)))
}

/// The Pareto alpha at which `frequency` losses a year above `threshold`
/// give the layer an expected loss of `loss` (R's
/// `Pareto_Find_Alpha_btw_FQ_Layer`):
/// `PRO.PRC.PARETO.ALPHA.FREQLAYER(threshold, frequency, limit, attachment, loss, [truncation])`.
pub fn pareto_alpha_frequency_layer(args: &[Grid]) -> Result<Grid> {
    let alpha = alpha_between_frequency_and_layer(
        scalar(&args[0], "threshold")?,
        scalar(&args[1], "frequency")?,
        xs_layer(&args[2], &args[3], "the")?,
        scalar(&args[4], "loss")?,
        truncation(&args[5])?,
    )
    .map_err(|e| e.to_string())?;
    Ok(Grid::scalar(Value::number(alpha)))
}

/// The Pareto alpha between two frequencies above two thresholds (R's
/// `Pareto_Find_Alpha_btw_FQs`):
/// `PRO.PRC.PARETO.ALPHA.FREQS(threshold1, frequency1, threshold2, frequency2, [truncation])`.
pub fn pareto_alpha_frequencies(args: &[Grid]) -> Result<Grid> {
    let alpha = alpha_between_frequencies(
        scalar(&args[0], "threshold1")?,
        scalar(&args[1], "frequency1")?,
        scalar(&args[2], "threshold2")?,
        scalar(&args[3], "frequency2")?,
        truncation(&args[4])?,
    )
    .map_err(|e| e.to_string())?;
    Ok(Grid::scalar(Value::number(alpha)))
}

#[cfg(test)]
mod tests {
    use super::paired;
    use crate::sheet::{Grid, Value};

    fn col(v: &[Option<f64>]) -> Grid {
        Grid::from_rows(
            v.iter()
                .map(|x| vec![x.map_or(Value::Empty, Value::Number)])
                .collect(),
        )
    }

    #[test]
    fn paired_drops_rows_blank_in_every_range() {
        let a = col(&[Some(1.0), None, Some(3.0), None]);
        let b = col(&[Some(10.0), None, Some(30.0), None]);
        let out = paired(&[(&a, "a"), (&b, "b")]).unwrap();
        assert_eq!(out, vec![vec![1.0, 3.0], vec![10.0, 30.0]]);
    }

    #[test]
    fn paired_refuses_a_row_blank_in_some_ranges() {
        // One blank in each range, in different rows: the lengths agree
        // after dropping them, the pairing does not.
        let a = col(&[Some(1.0), None, Some(3.0), Some(4.0)]);
        let b = col(&[Some(10.0), Some(20.0), Some(30.0), None]);
        let e = paired(&[(&a, "a"), (&b, "b")]).unwrap_err();
        assert_eq!(
            e,
            "a row 2, column 1 is blank but b is not: give a value in both or leave both blank"
        );
    }

    #[test]
    fn paired_refuses_ranges_of_different_lengths_and_empty_ones() {
        let a = col(&[Some(1.0), Some(2.0)]);
        let b = col(&[Some(1.0)]);
        let e = paired(&[(&a, "a"), (&b, "b")]).unwrap_err();
        assert_eq!(
            e,
            "b has 1 cells, too few for a row 2, column 1: give one value per row"
        );
        // Extra cells that are blank in the longer range are a blank tail.
        let long = col(&[Some(1.0), Some(2.0), None, None]);
        let short = col(&[Some(5.0), Some(6.0)]);
        let equal = col(&[Some(5.0), Some(6.0), None, None]);
        assert_eq!(
            paired(&[(&long, "a"), (&short, "b")]).unwrap(),
            paired(&[(&long, "a"), (&equal, "b")]).unwrap()
        );
        assert_eq!(
            paired(&[(&short, "b"), (&long, "a")]).unwrap(),
            vec![vec![5.0, 6.0], vec![1.0, 2.0]]
        );
        let none = col(&[None, None]);
        assert_eq!(
            paired(&[(&none, "a"), (&none, "b")]).unwrap_err(),
            "a is required"
        );
        let text = Grid::scalar(Value::from("x"));
        let e = paired(&[(&text, "a"), (&b, "b")]).unwrap_err();
        assert!(e.contains("a row 1, column 1 must be a number"), "{e}");
    }

    #[test]
    fn paired_reads_three_ranges() {
        let a = Grid::row(&[1.0, 2.0]);
        let b = Grid::row(&[3.0, 4.0]);
        let c = Grid::row(&[5.0, 6.0]);
        let out = paired(&[(&a, "a"), (&b, "b"), (&c, "c")]).unwrap();
        assert_eq!(out, vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
    }
}
