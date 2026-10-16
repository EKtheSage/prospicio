//! The layer pricing functions: limited expected values, layer losses, a
//! layer's CV and increased limit factors from a Pareto or lognormal
//! severity (`docs/design/xll.md`).
//!
//! Limits and attachments may be single values or ranges. Two ranges
//! combine cell by cell, and a row against a column makes a table, as
//! Excel's own arithmetic does (`=A1:A5 + B1:D1`): a column of
//! attachments and a row of limits give every layer in one formula. A
//! blank cell gives a blank result; a limit of `"unlimited"` is no limit.

use prospicio_pricing::layer::{ilf, layer_cv};
use prospicio_prob::{Lognormal, Pareto, Severity};

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

/// The severity named by `family` with parameters `p1`, `p2` and, for a
/// Pareto, an optional `truncation`.
fn severity(family: &Grid, p1: &Grid, p2: &Grid, truncation: &Grid) -> Result<Box<dyn Severity>> {
    let name = match family.as_scalar() {
        Some(Value::Text(s)) => s.trim().to_ascii_lowercase().replace([' ', '-'], "_"),
        _ => return Err("family must be \"pareto\", \"lognormal\" or \"lognormal_cv\"".into()),
    };
    let truncated = !truncation.is_blank();
    let e = |e: prospicio_core::Error| e.to_string();
    match name.as_str() {
        "pareto" => {
            let p = Pareto::new(
                scalar(p1, "p1 (the Pareto threshold t)")?,
                scalar(p2, "p2 (alpha)")?,
            )
            .map_err(e)?;
            if truncated {
                Ok(Box::new(
                    p.truncated(scalar(truncation, "truncation")?).map_err(e)?,
                ))
            } else {
                Ok(Box::new(p))
            }
        }
        "lognormal" | "lognormal_cv" if truncated => {
            Err("truncation is taken only with family \"pareto\"".into())
        }
        "lognormal" => Ok(Box::new(
            Lognormal::new(scalar(p1, "p1 (meanlog)")?, scalar(p2, "p2 (sdlog)")?).map_err(e)?,
        )),
        "lognormal_cv" => Ok(Box::new(
            Lognormal::from_mean_cv(scalar(p1, "p1 (the mean)")?, scalar(p2, "p2 (the CV)")?)
                .map_err(e)?,
        )),
        other => Err(format!(
            "family must be \"pareto\", \"lognormal\" or \"lognormal_cv\", got {other:?}"
        )),
    }
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
