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
use prospicio_pricing::profile::{Band, BandCurve, RiskProfile};

use crate::pricing::{Result, amount, at, map1, numbers, scalar};
use crate::sheet::{Grid, Value};

/// The curves `curve` names, and what `p1`, `p2` and `p3` are for each.
pub const CURVES: &str = "swiss_re (c), mbbefd (b, g), tabulated (x, G(x), \
[\"linear\" or \"log-log\"])";

/// The single curve named by `curve, p1, p2, p3`.
fn curve(name: &Grid, p1: &Grid, p2: &Grid, p3: &Grid) -> Result<BandCurve> {
    Ok(curves(name, p1, p2, p3, 1)?.remove(0))
}

/// `n` curves named by `curve, p1, p2, p3`: a Swiss Re `c` or an MBBEFD
/// `b` and `g` may be a range with one value per band, a tabulated curve
/// is the same for all.
pub(crate) fn curves(
    name: &Grid,
    p1: &Grid,
    p2: &Grid,
    p3: &Grid,
    n: usize,
) -> Result<Vec<BandCurve>> {
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
    // Parameter `k` of `n`: one value for all, or one per band.
    let each = |g: &Grid, what: &str| -> Result<Vec<f64>> {
        if g.cells.len() == 1 {
            return Ok(vec![scalar(g, what)?; n]);
        }
        let values = numbers(g, what)?;
        if values.len() == n && g.cells.len() == n {
            Ok(values)
        } else {
            Err(format!(
                "{what} must be one number, or one per band ({n}), got {} cells",
                g.cells.len()
            ))
        }
    };
    match name.as_str() {
        "swiss_re" | "swissre" => {
            unused(p2, "p2")?;
            unused(p3, "p3")?;
            each(p1, "p1 (c)")?
                .into_iter()
                .map(|c| Ok(Mbbefd::swiss_re(c).map_err(e)?.into()))
                .collect()
        }
        "mbbefd" => {
            unused(p3, "p3")?;
            let (bs, gs) = (each(p1, "p1 (b)")?, each(p2, "p2 (g)")?);
            bs.into_iter()
                .zip(gs)
                .map(|(b, g)| Ok(Mbbefd::new(b, g).map_err(e)?.into()))
                .collect()
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
            let t: BandCurve = TabulatedCurve::with_interpolation(&x, &g, how)
                .map_err(e)?
                .into();
            Ok(vec![t; n])
        }
        other => Err(format!("curve must be one of: {CURVES}; got {other:?}")),
    }
}

/// `f` over ranges broadcast against each other, blank where any cell is;
/// `names` for errors, `unlimited` where `"unlimited"` is allowed.
pub(crate) fn broadcast(
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
/// `PRO.EXPOSURE.CURVE(x, curve, p1, [p2], [p3])`. A row of Swiss Re `c`
/// (or MBBEFD `b`, `g`) against a column of `x` gives one column per
/// curve.
pub fn exposure_curve(args: &[Grid]) -> Result<Grid> {
    curve_table(args, "x", |c, x| {
        if (0.0..=1.0).contains(&x) {
            Ok(c.g(x))
        } else {
            Err(format!("x must be in [0, 1], got {x}"))
        }
    })
}

/// `f` at each value of `args[0]` on the curves named by `args[1..5]`.
/// One curve maps the values cell by cell. A Swiss Re `c` or an MBBEFD
/// `b` and `g` given as a row (or column) of several values names one
/// curve each, and the result is a table: a column of values against a
/// row of curves, one column per curve, or a row of values against a
/// column of curves, one row per curve.
fn curve_table(
    args: &[Grid],
    name: &str,
    f: impl Fn(&BandCurve, f64) -> Result<f64>,
) -> Result<Grid> {
    let xs = &args[0];
    let tabulated = matches!(args[1].as_scalar(), Some(Value::Text(s)) if s.trim().eq_ignore_ascii_case("tabulated"));
    let params = [&args[2], &args[3]]
        .into_iter()
        .find(|g| g.cells.len() > 1)
        .filter(|_| !tabulated);
    let Some(params) = params else {
        let c = curve(&args[1], &args[2], &args[3], &args[4])?;
        return map1(xs, name, false, |x| f(&c, x));
    };
    let n = params.cells.len();
    let across = params.rows == 1;
    if params.rows != 1 && params.columns != 1 {
        return Err("several curves' parameters must be a row or a column".into());
    }
    if (across && xs.columns != 1) || (!across && xs.rows != 1) {
        return Err(format!(
            "with several curves, {name} must be a column against a row of parameters, \
             or a row against a column"
        ));
    }
    let curves = curves(&args[1], &args[2], &args[3], &args[4], n)?;
    let m = xs.cells.len();
    let (rows, columns) = if across { (m, n) } else { (n, m) };
    let mut cells = Vec::with_capacity(m * n);
    for r in 0..rows {
        for c in 0..columns {
            let (i, k) = if across { (r, c) } else { (c, r) };
            let (xr, xc) = if across { (i, 0) } else { (0, i) };
            cells.push(match amount(xs.get(xr, xc), name, xr, xc, false)? {
                Some(x) => Value::number(f(&curves[k], x)?),
                None => Value::Empty,
            });
        }
    }
    Ok(Grid {
        rows,
        columns,
        cells,
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
        |v| c.layer_share(v[0], v[1], v[2]).map_err(|e| e.to_string()),
    )
}

/// The destruction rate (loss over MPL) at each probability `u`:
/// `PRO.EXPOSURE.QUANTILE(u, curve, p1, [p2], [p3])`.
pub fn quantile(args: &[Grid]) -> Result<Grid> {
    curve_table(args, "u", |c, u| {
        if u > 0.0 && u < 1.0 {
            Ok(c.rate_quantile(u))
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
        BandCurve::Mbbefd(m) => (
            "mbbefd",
            Value::number(m.b()),
            Value::number(m.g_parameter()),
            Value::number(m.total_loss_probability()),
        ),
        BandCurve::Tabulated(_) => ("tabulated", Value::Empty, Value::Empty, Value::Empty),
    };
    let rows = vec![
        vec![Value::from("Curve"), Value::from(kind)],
        vec![Value::from("b"), b],
        vec![Value::from("g"), g],
        vec![Value::from("Mean rate"), Value::number(c.mean_rate())],
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

/// Each band's optional bound: omitted for none, else one cell per band,
/// a blank for a band without.
fn bounds(g: &Grid, name: &str, n: usize) -> Result<Vec<Option<f64>>> {
    if g.is_blank() {
        return Ok(vec![None; n]);
    }
    if g.cells.len() != n {
        return Err(format!(
            "{name} must have one cell per band ({n}), got {}",
            g.cells.len()
        ));
    }
    g.cells
        .iter()
        .enumerate()
        .map(|(k, v)| match v {
            Value::Number(x) => Ok(Some(*x)),
            v if v.is_blank() => Ok(None),
            other => Err(format!(
                "{name} of band {} must be a number, got {other:?}",
                k + 1
            )),
        })
        .collect()
}

/// An exposure-rated risk profile: each band's mean sum insured, expected
/// loss and claims, its loss ceded to a surplus treaty and its expected
/// loss to the per-risk layer `limit` xs `attachment` (net of the surplus
/// when one is given), with a total row:
/// `PRO.PROFILE(sums_insured, expected_losses, limit, attachment, curve,
/// p1, [p2], [p3], [lower], [upper], [spread], [retention], [lines])`.
pub fn profile(args: &[Grid]) -> Result<Grid> {
    let si = numbers(&args[0], "sums_insured")?;
    let el = numbers(&args[1], "expected_losses")?;
    let n = si.len();
    if el.len() != n {
        return Err(format!(
            "expected_losses must have one value per band ({n}), got {}",
            el.len()
        ));
    }
    let one = |g: &Grid, name: &str, unlimited: bool| -> Result<f64> {
        match g.as_scalar() {
            Some(v) => amount(v, name, 0, 0, unlimited)?.ok_or(format!("{name} is required")),
            None => Err(format!("{name} must be a single value")),
        }
    };
    let limit = one(&args[2], "limit", true)?;
    let attachment = if args[3].is_blank() {
        0.0
    } else {
        one(&args[3], "attachment", false)?
    };
    let curves = curves(&args[4], &args[5], &args[6], &args[7], n)?;
    let lower = bounds(&args[8], "lower", n)?;
    let upper = bounds(&args[9], "upper", n)?;
    let spread = match args[10].as_scalar() {
        Some(v) if v.is_blank() => "uniform".to_owned(),
        Some(Value::Text(s)) => s.trim().to_ascii_lowercase(),
        _ => return Err("spread must be \"uniform\", \"tilted\" or \"auto\"".into()),
    };
    let surplus = match (args[11].is_blank(), args[12].is_blank()) {
        (true, true) => None,
        (false, false) => Some((scalar(&args[11], "retention")?, scalar(&args[12], "lines")?)),
        _ => return Err("a surplus treaty needs both retention and lines".into()),
    };
    let e = |e: prospicio_core::Error| e.to_string();
    let mut bands = Vec::with_capacity(n);
    for k in 0..n {
        let band = Band::from_expected_loss(si[k], 1.0, el[k], curves[k].clone())
            .map_err(|x| format!("band {}: {x}", k + 1))?;
        let band = match (lower[k], upper[k]) {
            (None, None) => band,
            (Some(l), Some(u)) => match spread.as_str() {
                "uniform" => band.with_bounds(l, u),
                "tilted" => band.with_tilted_bounds(l, u),
                "auto" => band.with_spread_bounds(l, u),
                other => {
                    return Err(format!(
                        "spread must be \"uniform\", \"tilted\" or \"auto\", got {other:?}"
                    ));
                }
            }
            .map_err(|x| format!("band {}: {x}", k + 1))?,
            _ => return Err(format!("band {} needs both bounds or neither", k + 1)),
        };
        bands.push(band);
    }
    let mut rows = vec![
        [
            "Sum insured",
            "Mean SI",
            "Expected loss",
            "Expected claims",
            "Surplus ceded",
            "Layer loss",
            "Layer share",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    let (mut claims, mut ceded, mut layer) = (0.0, 0.0, 0.0);
    for b in &bands {
        let one = RiskProfile::new(vec![b.clone()]).map_err(e)?;
        let c = one.expected_claims();
        let s = surplus.map(|(r, l)| one.expected_surplus_loss(r, l));
        let x = one
            .expected_layer_loss(limit, attachment, surplus)
            .map_err(e)?;
        claims += c;
        ceded += s.unwrap_or(0.0);
        layer += x;
        rows.push(vec![
            Value::number(b.sum_insured),
            Value::number(b.mean_sum_insured()),
            Value::number(b.expected_loss),
            Value::number(c),
            s.map_or(Value::Empty, Value::number),
            Value::number(x),
            ratio(x, b.expected_loss),
        ]);
    }
    let total: f64 = el.iter().sum();
    rows.push(vec![
        Value::from("Total"),
        Value::Empty,
        Value::number(total),
        Value::number(claims),
        surplus.map_or(Value::Empty, |_| Value::number(ceded)),
        Value::number(layer),
        ratio(layer, total),
    ]);
    Ok(Grid::from_rows(rows))
}

/// `x / of`, blank when `of` is 0.
fn ratio(x: f64, of: f64) -> Value {
    if of > 0.0 {
        Value::number(x / of)
    } else {
        Value::Empty
    }
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
