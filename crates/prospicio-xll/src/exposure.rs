//! Exposure curves for property per-risk rating (`docs/design/xll.md`):
//! a curve's `G(x)`, the share of a risk's expected loss in a layer, its
//! destruction-rate quantile and statistics, and the Swiss Re `c` that
//! matches a mean rate or a layer's share, over
//! `prospicio_pricing::exposure`.
//!
//! A curve is named as `curve, p1, [p2], [p3]`, see [`CURVES`]. Limits,
//! attachments and MPLs broadcast against each other as Excel's own
//! arithmetic does: a column of MPLs against a row of attachments gives a
//! table.

use prospicio_pricing::exposure::{ExposureCurve, Interpolation, Mbbefd, TabulatedCurve};

use crate::pricing::{Result, amount, at, map1, numbers, scalar};
use crate::sheet::{Grid, Value};

/// The curves `curve` names, and what `p1`, `p2` and `p3` are for each.
pub const CURVES: &str = "swiss_re (c), mbbefd (b, g), tabulated (x, G(x), \
[\"linear\" or \"log-log\"])";

/// An exposure curve built from the sheet.
enum Curve {
    Mbbefd(Mbbefd),
    Tabulated(TabulatedCurve),
}

impl Curve {
    fn get(&self) -> &dyn ExposureCurve {
        match self {
            Self::Mbbefd(m) => m,
            Self::Tabulated(t) => t,
        }
    }
}

fn curve(name: &Grid, p1: &Grid, p2: &Grid, p3: &Grid) -> Result<Curve> {
    let name = match name.as_scalar() {
        Some(Value::Text(s)) => s.trim().to_ascii_lowercase().replace([' ', '-'], "_"),
        _ => return Err(format!("curve must be one of: {CURVES}")),
    };
    let e = |e: prospicio_core::Error| e.to_string();
    let unused = |g: &Grid, what: &str| -> Result<()> {
        if g.is_blank() {
            Ok(())
        } else {
            Err(format!("{what} is not taken by {name:?}"))
        }
    };
    match name.as_str() {
        "swiss_re" | "swissre" => {
            unused(p2, "p2")?;
            unused(p3, "p3")?;
            Ok(Curve::Mbbefd(
                Mbbefd::swiss_re(scalar(p1, "p1 (c)")?).map_err(e)?,
            ))
        }
        "mbbefd" => {
            unused(p3, "p3")?;
            Ok(Curve::Mbbefd(
                Mbbefd::new(scalar(p1, "p1 (b)")?, scalar(p2, "p2 (g)")?).map_err(e)?,
            ))
        }
        "tabulated" => {
            let x = numbers(p1, "p1 (x)")?;
            let g = numbers(p2, "p2 (G(x))")?;
            let how = match p3.as_scalar() {
                Some(v) if v.is_blank() => Interpolation::Linear,
                Some(Value::Text(s)) => match s.trim().to_ascii_lowercase().as_str() {
                    "linear" => Interpolation::Linear,
                    "log-log" | "log_log" | "loglog" => Interpolation::LogLog,
                    other => {
                        return Err(format!(
                            "p3 (interpolation) must be \"linear\" or \"log-log\", got {other:?}"
                        ));
                    }
                },
                _ => return Err("p3 (interpolation) must be \"linear\" or \"log-log\"".into()),
            };
            Ok(Curve::Tabulated(
                TabulatedCurve::with_interpolation(&x, &g, how).map_err(e)?,
            ))
        }
        other => Err(format!("curve must be one of: {CURVES}; got {other:?}")),
    }
}

/// `f` over ranges broadcast against each other, blank where any cell is;
/// `names` for errors, `unlimited` where `"unlimited"` is allowed.
fn broadcast(
    grids: &[&Grid],
    names: &[&str],
    unlimited: &[bool],
    f: impl Fn(&[f64]) -> Result<f64>,
) -> Result<Grid> {
    let side = |sizes: &mut dyn Iterator<Item = usize>| -> Option<usize> {
        let mut n = 1;
        for s in sizes {
            match (n, s) {
                (_, 1) => {}
                (1, s) => n = s,
                (n0, s) if n0 == s => {}
                _ => return None,
            }
        }
        Some(n)
    };
    let shape = (
        side(&mut grids.iter().map(|g| g.rows)),
        side(&mut grids.iter().map(|g| g.columns)),
    );
    let (rows, columns) = match shape {
        (Some(r), Some(c)) => (r, c),
        _ => {
            let sizes: Vec<String> = grids
                .iter()
                .zip(names)
                .map(|(g, n)| format!("{n} ({} x {})", g.rows, g.columns))
                .collect();
            return Err(format!(
                "{} do not combine: give ranges of the same size, or rows and columns",
                sizes.join(", ")
            ));
        }
    };
    let mut cells = Vec::with_capacity(rows * columns);
    let mut xs = vec![0.0; grids.len()];
    for r in 0..rows {
        for c in 0..columns {
            let mut blank = false;
            for (k, g) in grids.iter().enumerate() {
                match amount(at(g, r, c), names[k], r, c, unlimited[k])? {
                    Some(x) => xs[k] = x,
                    None => blank = true,
                }
            }
            cells.push(if blank {
                Value::Empty
            } else {
                Value::number(f(&xs)?)
            });
        }
    }
    Ok(Grid {
        rows,
        columns,
        cells,
    })
}

/// The exposure curve `G(x)` at each `x` in `[0, 1]`:
/// `PRO.EXPOSURE.CURVE(x, curve, p1, [p2], [p3])`.
pub fn exposure_curve(args: &[Grid]) -> Result<Grid> {
    let c = curve(&args[1], &args[2], &args[3], &args[4])?;
    map1(&args[0], "x", false, |x| {
        if (0.0..=1.0).contains(&x) {
            Ok(c.get().g(x))
        } else {
            Err(format!("x must be in [0, 1], got {x}"))
        }
    })
}

/// Share of a risk's expected loss in `limit` xs `attachment`, for a risk
/// of maximum possible loss `mpl`:
/// `PRO.EXPOSURE.SHARE(limit, attachment, mpl, curve, p1, [p2], [p3])`.
pub fn share(args: &[Grid]) -> Result<Grid> {
    let c = curve(&args[3], &args[4], &args[5], &args[6])?;
    broadcast(
        &[&args[0], &args[1], &args[2]],
        &["limit", "attachment", "mpl"],
        &[true, false, false],
        |v| {
            c.get()
                .layer_share(v[0], v[1], v[2])
                .map_err(|e| e.to_string())
        },
    )
}

/// The destruction rate (loss over MPL) at each probability `u`:
/// `PRO.EXPOSURE.QUANTILE(u, curve, p1, [p2], [p3])`.
pub fn quantile(args: &[Grid]) -> Result<Grid> {
    let c = curve(&args[1], &args[2], &args[3], &args[4])?;
    map1(&args[0], "u", false, |u| {
        if u > 0.0 && u < 1.0 {
            Ok(c.get().rate_quantile(u))
        } else {
            Err(format!("u must be in (0, 1), got {u}"))
        }
    })
}

/// A curve's parameters and destruction-rate statistics as a two-column
/// table: `PRO.EXPOSURE.STATS(curve, p1, [p2], [p3])`. The MBBEFD's `b`,
/// `g` and probability of a total loss are blank for a tabulated curve.
pub fn stats(args: &[Grid]) -> Result<Grid> {
    let c = curve(&args[0], &args[1], &args[2], &args[3])?;
    let (kind, b, g, total) = match &c {
        Curve::Mbbefd(m) => (
            "mbbefd",
            Value::number(m.b()),
            Value::number(m.g_parameter()),
            Value::number(m.total_loss_probability()),
        ),
        Curve::Tabulated(_) => ("tabulated", Value::Empty, Value::Empty, Value::Empty),
    };
    let rows = vec![
        vec![Value::from("Curve"), Value::from(kind)],
        vec![Value::from("b"), b],
        vec![Value::from("g"), g],
        vec![Value::from("Mean rate"), Value::number(c.get().mean_rate())],
        vec![Value::from("Total loss probability"), total],
    ];
    Ok(Grid::from_rows(rows))
}

/// The Swiss Re `c` whose mean destruction rate is each `rate`:
/// `PRO.SWISSRE.C(rate)`.
pub fn swiss_re_c(args: &[Grid]) -> Result<Grid> {
    map1(&args[0], "rate", false, |r| {
        Mbbefd::swiss_re_for_mean_rate(r)
            .map(|(_, c)| c)
            .map_err(|e| e.to_string())
    })
}

/// The Swiss Re `c` whose share of a risk's expected loss in `limit` xs
/// `attachment` is `share`: `PRO.SWISSRE.C.LAYER(limit, attachment, mpl,
/// share)`, each broadcast against the others.
pub fn swiss_re_c_layer(args: &[Grid]) -> Result<Grid> {
    broadcast(
        &[&args[0], &args[1], &args[2], &args[3]],
        &["limit", "attachment", "mpl", "share"],
        &[true, false, false, false],
        |v| {
            Mbbefd::swiss_re_for_layer_share(v[0], v[1], v[2], v[3])
                .map(|(_, c)| c)
                .map_err(|e| e.to_string())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_rejects_mismatched_shapes() {
        let row = Grid::from_rows(vec![vec![Value::Number(1.0), Value::Number(2.0)]]);
        let col = Grid::from_rows(vec![vec![Value::Number(1.0)], vec![Value::Number(2.0)]]);
        let three = Grid::from_rows(vec![vec![Value::Number(1.0); 3]]);
        let ok = broadcast(
            &[&row, &col, &Grid::scalar(Value::Number(1.0))],
            &["a", "b", "c"],
            &[false; 3],
            |v| Ok(v[0] + v[1] + v[2]),
        )
        .unwrap();
        assert_eq!((ok.rows, ok.columns), (2, 2));
        assert_eq!(ok.get(1, 0), &Value::Number(4.0));
        let err = broadcast(&[&row, &three], &["a", "b"], &[false; 2], |v| Ok(v[0])).unwrap_err();
        assert!(err.contains("do not combine"), "{err}");
    }
}
