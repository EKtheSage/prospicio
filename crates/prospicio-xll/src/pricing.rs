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

use prospicio_pricing::layer::loss_elimination_ratio;
use prospicio_pricing::layer::{ilf, layer_cv};
use prospicio_prob::evt::Gpd;
use prospicio_prob::{
    Burr, Dist, Distribution, Gamma, InverseGamma, InverseGaussian, Loglogistic, Lognormal, Pareto,
    Severity, SeverityDist, Weibull,
};

use crate::sheet::{Grid, Value, position};

type Result<T> = std::result::Result<T, String>;

/// One number argument, which must be a single value.
fn scalar(grid: &Grid, name: &str) -> Result<f64> {
    match grid.as_scalar() {
        Some(Value::Number(x)) => Ok(*x),
        Some(v) if v.is_blank() => Err(format!("{name} is required")),
        _ => Err(format!("{name} must be a single number")),
    }
}

/// The families `family` names, and what `p1`, `p2` and `p3` are for each.
pub const FAMILIES: &str = "pareto (t, alpha, [truncation]), lognormal (meanlog, sdlog), \
lognormal_cv (mean, cv), gamma (shape, scale), gamma_cv (mean, cv), weibull (shape, scale), \
loglogistic (shape, scale), inverse_gamma (shape, scale), inverse_gaussian (mean, shape), \
burr (alpha, gamma, scale), gpd (xi, beta)";

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
                "p3 is taken only by \"pareto\" (truncation) and \"burr\", not {name:?}"
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
        other => return Err(format!("family must be one of: {FAMILIES}; got {other:?}")),
    };
    SeverityDist::try_from(dist).map_err(|_| "not a severity".into())
}

/// One cell of a limit or attachment range: `None` for a blank.
fn amount(v: &Value, name: &str, r: usize, c: usize, unlimited: bool) -> Result<Option<f64>> {
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
fn at(g: &Grid, r: usize, c: usize) -> &Value {
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
/// `PRO.LEV(limit, family, p1, p2, [truncation])`.
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
/// `PRO.LAYER(limit, attachment, family, p1, p2, [truncation])`.
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

/// CV of the loss to the layer given a loss above its attachment:
/// `PRO.LAYER.CV(limit, attachment, family, p1, p2, [truncation])`.
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
/// `PRO.ILF(limit, basic_limit, family, p1, p2, [truncation])`.
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
fn map1(xs: &Grid, name: &str, unlimited: bool, f: impl Fn(f64) -> Result<f64>) -> Result<Grid> {
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
/// `PRO.LER(deductible, family, p1, p2, [p3])`.
pub fn ler(args: &[Grid]) -> Result<Grid> {
    let sev = severity(&args[1], &args[2], &args[3], &args[4])?;
    map1(&args[0], "deductible", false, |d| {
        loss_elimination_ratio(&sev, d).map_err(|e| e.to_string())
    })
}
