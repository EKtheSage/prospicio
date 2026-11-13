//! Development curves fitted over every age (`docs/design/xll.md`,
//! decision 50): `prospicio_reserving::DevelopmentCurve`, the exponential,
//! inverse power, Hoerl, Sherman and Clark's growth curves fitted to a
//! triangle's age-to-age factors, with fit statistics to choose between
//! them. Ages are in months, the triangle's columns 12, 24, ...; the
//! fitted pattern goes on to `PRO.RES.CL` and the other chain-ladder
//! functions as their `selected` factors and `tail`.

use prospicio_core::Lag;
use prospicio_reserving::{CurveShape, DevelopmentCurve, DevelopmentCurveFit};

use crate::diagonal::months;
use crate::reserving::{COLUMN, average, link_labels, text_option, triangle};
use crate::sheet::{Grid, Value};

type Result<T> = std::result::Result<T, String>;

/// Every shape, in the order the comparison lists them by default.
const SHAPES: [&str; 6] = [
    "exponential",
    "inverse_power",
    "hoerl",
    "sherman",
    "loglogistic",
    "weibull",
];

/// Periods past the oldest age the curve is extrapolated for its tail, as
/// chainladder-python's `TailCurve` and the Python binding default to.
const TAIL_PERIODS: usize = 100;

fn shape_named(name: &str) -> Result<CurveShape> {
    let key = name.trim().to_ascii_lowercase().replace([' ', '-'], "_");
    CurveShape::from_name(&key).ok_or_else(|| crate::unknown_name("shape", &key, CurveShape::NAMES))
}

/// The `fit_period` argument: blank for every age, or one or two cells,
/// the first age in months and the age the fit stops before, either blank
/// for open-ended.
fn fit_period(grid: &Grid) -> Result<(Option<Lag>, Option<Lag>)> {
    if grid.is_blank() {
        return Ok((None, None));
    }
    if grid.cells.len() > 2 || (grid.rows != 1 && grid.columns != 1) {
        return Err("fit_period must be one or two cells: the first age and the age it stops before, in months".into());
    }
    let age = |i: usize| -> Result<Option<Lag>> {
        match grid.cells.get(i) {
            None => Ok(None),
            Some(v) if v.is_blank() => Ok(None),
            Some(v) => months(v).map(Some).ok_or_else(|| {
                format!("fit_period's ages must be whole numbers of months from 1, got {v:?}")
            }),
        }
    };
    Ok((age(0)?, age(1)?))
}

/// An optional age in months.
fn age_option(grid: &Grid, name: &str) -> Result<Option<Lag>> {
    match grid.as_scalar() {
        Some(v) if v.is_blank() => Ok(None),
        Some(v) => months(v)
            .map(Some)
            .ok_or_else(|| format!("{name} must be a whole number of months from 1, got {v:?}")),
        None => Err(format!("{name} must be a single age in months")),
    }
}

/// The `tail_periods` argument: periods past the oldest age, 100 if
/// omitted, 0 for no tail.
fn tail_periods(grid: &Grid) -> Result<usize> {
    match grid.as_scalar() {
        Some(v) if v.is_blank() => Ok(TAIL_PERIODS),
        Some(Value::Number(x)) if x.fract() == 0.0 && (0.0..=10_000.0).contains(x) => {
            Ok(*x as usize)
        }
        _ => Err("tail_periods must be a whole number from 0 to 10000".into()),
    }
}

/// The triangle's development fit under `average`, and the curve settings.
fn fit_one(
    triangle_arg: &Grid,
    shape: CurveShape,
    average_arg: &Grid,
    period: &Grid,
    keep: Option<Lag>,
) -> Result<DevelopmentCurveFit> {
    let dev = average(average_arg)?
        .fit(&triangle(triangle_arg)?, COLUMN)
        .map_err(|e| e.to_string())?;
    DevelopmentCurve {
        shape,
        fit_period: fit_period(period)?,
        keep_observed_until: keep,
    }
    .fit(&dev)
    .map_err(|e| e.to_string())
}

fn number_or_blank(x: f64) -> Value {
    if x.is_finite() {
        Value::number(x)
    } else {
        Value::Empty
    }
}

/// `PRO.RES.CURVE(triangle, [shape], [average], [fit_period],
/// [keep_observed_until], [tail_periods])`: a row per development link,
/// its observed factor, the curve's, the residual, whether it entered the
/// fit and the selected factor (observed before `keep_observed_until`,
/// the curve's from there on), then the curve's tail to ultimate.
pub fn curve(args: &[Grid]) -> Result<Grid> {
    let shape = match text_option(&args[1], "shape")? {
        None => CurveShape::Exponential,
        Some(s) => shape_named(&s)?,
    };
    let keep = age_option(&args[4], "keep_observed_until")?;
    let fit = fit_one(&args[0], shape, &args[2], &args[3], keep)?;
    let periods = tail_periods(&args[5])?;
    let n = fit.fitted.len();
    let labels = link_labels(n + 1);
    let mut rows = vec![
        [
            "Development",
            "Observed",
            "Fitted",
            "Residual",
            "In fit",
            "Selected",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for (p, label) in labels.iter().take(n).enumerate() {
        rows.push(vec![
            label.clone(),
            number_or_blank(fit.observed[p]),
            Value::number(fit.fitted[p]),
            number_or_blank(fit.residuals[p]),
            Value::Bool(fit.in_fit[p]),
            Value::number(fit.ldf[p]),
        ]);
    }
    let tail = fit.tail(periods);
    rows.push(vec![
        labels[n].clone(),
        Value::Empty,
        Value::number(tail),
        Value::Empty,
        Value::Empty,
        Value::number(tail),
    ]);
    Ok(Grid::from_rows(rows))
}

/// `PRO.RES.CURVE.STATS(triangle, [shapes], [average], [fit_period],
/// [tail_periods])`: a row per shape (every shape if omitted), its
/// parameters, the number of points and parameters, the sum of squares,
/// R-squared and adjusted R-squared on the factor scale, and its tail; a
/// shape that cannot be fitted shows why in its row.
pub fn stats(args: &[Grid]) -> Result<Grid> {
    let names: Vec<String> = if args[1].is_blank() {
        SHAPES.iter().map(|s| (*s).to_owned()).collect()
    } else {
        args[1]
            .cells
            .iter()
            .filter(|v| !v.is_blank())
            .map(|v| match v {
                Value::Text(s) => Ok(s.trim().to_owned()),
                other => Err(format!("shapes must be text, got {other:?}")),
            })
            .collect::<Result<_>>()?
    };
    let periods = tail_periods(&args[4])?;
    // The triangle and settings are checked once, whatever the shapes.
    average(&args[2])?;
    triangle(&args[0])?;
    fit_period(&args[3])?;
    let mut rows = vec![
        [
            "Shape",
            "a / omega",
            "b / theta",
            "c",
            "Points",
            "Parameters",
            "SSE",
            "R-squared",
            "Adj R-squared",
            "Tail",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for name in names {
        let shape = shape_named(&name)?;
        let mut row = vec![Value::Text(name)];
        match fit_one(&args[0], shape, &args[2], &args[3], None) {
            Ok(fit) => {
                for i in 0..3 {
                    row.push(
                        fit.parameters
                            .get(i)
                            .map_or(Value::Empty, |x| Value::number(*x)),
                    );
                }
                row.push(Value::Number(fit.n_points as f64));
                row.push(Value::Number(fit.n_parameters as f64));
                row.push(Value::number(fit.sum_of_squares));
                row.push(number_or_blank(fit.r_squared));
                row.push(number_or_blank(fit.adjusted_r_squared));
                row.push(Value::number(fit.tail(periods)));
            }
            Err(e) => {
                row.push(Value::Text(e));
                row.extend(std::iter::repeat_n(Value::Empty, 8));
            }
        }
        rows.push(row);
    }
    Ok(Grid::from_rows(rows))
}
