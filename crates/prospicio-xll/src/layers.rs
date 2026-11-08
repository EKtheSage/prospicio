//! Excess-layer development and benchmark patterns on a sheet
//! (`docs/design/xll.md`, decision 58): a layer's development factors
//! derived from a ground-up triangle and a severity
//! (`prospicio_reserving::excess`), and a growth curve fitted to a
//! benchmark pattern (`prospicio_reserving::BenchmarkCurve`).

use prospicio_core::Lag;
use prospicio_reserving::excess::{Basis, ExcessPattern, Layer, SeverityByMaturity, ShapeBase};
use prospicio_reserving::{BenchmarkCurve, GrowthCurve, PatternKind};

use crate::counts::measures;
use crate::pricing::severity;
use crate::reserving::{COLUMN, number, number_or, pattern_kind, tail, text_option, triangle};
use crate::sheet::{Grid, Value};

type Result<T> = std::result::Result<T, String>;

fn err(e: impl ToString) -> String {
    e.to_string()
}

/// The counts measure inside a two-measure triangle.
const COUNTS: &str = "counts";

/// `attachment` and `limit` as a layer: a blank limit is unlimited.
fn layer(attachment: &Grid, limit: &Grid) -> Result<Layer> {
    let a = number(attachment, "attachment")?;
    let l = number_or(limit, "limit", f64::INFINITY)?;
    Layer::new(a, l).map_err(err)
}

/// The numbers of a row or column, blanks skipped.
fn numbers(grid: &Grid, name: &str) -> Result<Vec<f64>> {
    if grid.rows != 1 && grid.columns != 1 {
        return Err(format!("{name} must be a row or a column"));
    }
    grid.cells
        .iter()
        .filter(|v| !v.is_blank())
        .map(|v| match v {
            Value::Number(x) => Ok(*x),
            other => Err(format!("{name} must be numbers, got {other:?}")),
        })
        .collect()
}

/// An excess pattern as a sheet table or row.
fn pattern_output(p: &ExcessPattern, output: &Grid) -> Result<Grid> {
    match text_option(output, "output")?.as_deref() {
        None | Some("summary") => {
            let mut rows = vec![
                ["Age", "Ground-up LDF", "Layer LDF", "Layer CDF", "Shift"]
                    .map(Value::from)
                    .to_vec(),
            ];
            for (k, &age) in p.ages.iter().enumerate() {
                let link = |v: &[f64]| v.get(k).map_or(Value::Empty, |&x| Value::number(x));
                rows.push(vec![
                    Value::Number(f64::from(age)),
                    link(&p.ground_up_ldf),
                    link(&p.layer_ldf),
                    Value::number(p.layer_cdf[k]),
                    link(&p.shifts),
                ]);
            }
            rows.push(vec![
                Value::from("Tail"),
                Value::number(p.ground_up_tail),
                Value::number(p.layer_tail),
                Value::Empty,
                p.shifts
                    .get(p.ages.len())
                    .map_or(Value::Empty, |&x| Value::number(x)),
            ]);
            Ok(Grid::from_rows(rows))
        }
        Some("ldf") => {
            let mut row = p.layer_ldf.clone();
            if p.layer_tail != 1.0 {
                row.push(p.layer_tail);
            }
            Ok(Grid::row(&row))
        }
        Some("cdf") => Ok(Grid::row(&p.layer_cdf)),
        Some(other) => Err(format!(
            "output must be \"summary\", \"ldf\" or \"cdf\", got {other:?}"
        )),
    }
}

/// `PRO.RES.EXCESS.PATTERN(triangle, attachment, limit, family, p1, p2,
/// [p3], [counts], [tail], [cap], [output])`: the layer's development
/// derived from a ground-up incurred triangle by the scale shift, the
/// claims at each age the ultimate ones (the severity) scaled down.
pub fn scale_shift(args: &[Grid]) -> Result<Grid> {
    let tri = if args[7].is_blank() {
        triangle(&args[0])?
    } else {
        measures(&[(COLUMN, &args[0]), (COUNTS, &args[7])], &[])?
    };
    let layer = layer(&args[1], &args[2])?;
    let ultimate = Box::new(severity(&args[3], &args[4], &args[5], &args[6])?);
    let model = match args[9].as_scalar() {
        Some(v) if v.is_blank() => SeverityByMaturity::scale_shift(ultimate),
        _ => SeverityByMaturity::scale_shift_capped(ultimate, number(&args[9], "cap")?),
    };
    let counts = (!args[7].is_blank()).then_some(COUNTS);
    let p = ExcessPattern::from_ground_up(
        &tri,
        COLUMN,
        Basis::Incurred,
        counts,
        &tail(&args[8])?,
        layer,
        &model,
    )
    .map_err(err)?;
    pattern_output(&p, &args[10])
}

/// `PRO.RES.EXCESS.SHAPE(triangle, attachment, limit, threshold, alphas,
/// ultimate_alpha, [base], [base_limit], [tail], [output])`: Pinto and
/// Gogol's shape drift, the Pareto shape above `threshold` moving from
/// `alphas` (one per column) to `ultimate_alpha`; the triangle holds the
/// counts above the threshold (`base` blank or "counts") or the losses in
/// a base layer attaching at `base` (unlimited, or up to `base_limit`).
pub fn shape_drift(args: &[Grid]) -> Result<Grid> {
    let tri = triangle(&args[0])?;
    let layer = layer(&args[1], &args[2])?;
    let threshold = number(&args[3], "threshold")?;
    let alphas = numbers(&args[4], "alphas")?;
    if alphas.len() != args[0].columns {
        return Err(format!(
            "alphas must have {} values, one per triangle column, got {}",
            args[0].columns,
            alphas.len()
        ));
    }
    let alphas: Vec<(Lag, f64)> = alphas
        .into_iter()
        .enumerate()
        .map(|(c, a)| (12 * (c as Lag + 1), a))
        .collect();
    let ultimate_alpha = number(&args[5], "ultimate_alpha")?;
    let base = match args[6].as_scalar() {
        Some(v) if v.is_blank() => ShapeBase::Counts,
        Some(Value::Text(s)) if s.trim().eq_ignore_ascii_case("counts") => ShapeBase::Counts,
        Some(Value::Number(_)) => ShapeBase::Losses(layer_of(&args[6], &args[7])?),
        _ => {
            return Err(
                "base must be \"counts\" or the attachment of the base layer the triangle holds"
                    .into(),
            );
        }
    };
    if matches!(base, ShapeBase::Counts) && !args[7].is_blank() {
        return Err("base_limit is the base layer's; leave it empty with counts".into());
    }
    let model = SeverityByMaturity::pareto_shape(threshold, alphas, ultimate_alpha, base);
    let p = ExcessPattern::from_ground_up(
        &tri,
        COLUMN,
        Basis::Incurred,
        None,
        &tail(&args[8])?,
        layer,
        &model,
    )
    .map_err(err)?;
    pattern_output(&p, &args[9])
}

fn layer_of(attachment: &Grid, limit: &Grid) -> Result<Layer> {
    let a = number(attachment, "base")?;
    let l = number_or(limit, "base_limit", f64::INFINITY)?;
    Layer::new(a, l).map_err(err)
}

/// `PRO.RES.BENCHMARK.CURVE(ages, values, [kind], [curve], [weights],
/// [max_age], [origin_width], [at_ages], [output])`: a log-logistic or
/// Weibull growth curve fitted by weighted least squares to a benchmark
/// pattern (percents developed or factors to ultimate). `output` `"fit"`
/// (default) gives each benchmark age's observed, fitted and residual
/// values, then omega, theta and the sum of squares; `"pattern"` the
/// fitted percents developed at `at_ages` (the benchmark's ages if
/// omitted), as a row; `"cdf"` the factors to ultimate there.
pub fn benchmark_curve(args: &[Grid]) -> Result<Grid> {
    let ages = numbers(&args[0], "ages")?;
    let values = numbers(&args[1], "values")?;
    if ages.len() != values.len() {
        return Err(format!(
            "ages and values must have as many cells, got {} and {}",
            ages.len(),
            values.len()
        ));
    }
    let kind = if args[2].is_blank() {
        PatternKind::PercentDeveloped
    } else {
        pattern_kind(&args[2])?
    };
    let curve = match text_option(&args[3], "curve")?.as_deref() {
        None | Some("loglogistic" | "log-logistic" | "log_logistic") => GrowthCurve::LogLogistic,
        Some("weibull") => GrowthCurve::Weibull,
        Some(other) => {
            return Err(format!(
                "curve must be \"loglogistic\" or \"weibull\", got {other:?}"
            ));
        }
    };
    let weights = if args[4].is_blank() {
        None
    } else {
        Some(numbers(&args[4], "weights")?)
    };
    let optional = |g: &Grid, name: &str| -> Result<Option<f64>> {
        if g.is_blank() {
            Ok(None)
        } else {
            number(g, name).map(Some)
        }
    };
    let settings = BenchmarkCurve {
        curve,
        max_age: optional(&args[5], "max_age")?,
        origin_width: optional(&args[6], "origin_width")?,
    };
    let points: Vec<(f64, f64)> = ages.iter().copied().zip(values).collect();
    let fit = settings
        .fit(kind, &points, weights.as_deref())
        .map_err(err)?;
    let at = || -> Result<Vec<f64>> {
        if args[7].is_blank() {
            Ok(fit.ages.clone())
        } else {
            numbers(&args[7], "at_ages")
        }
    };
    match text_option(&args[8], "output")?.as_deref() {
        None | Some("fit") => {
            let mut rows = vec![
                ["Age", "Observed", "Fitted", "Residual", "Weight"]
                    .map(Value::from)
                    .to_vec(),
            ];
            for k in 0..fit.ages.len() {
                rows.push(vec![
                    Value::number(fit.ages[k]),
                    Value::number(fit.observed[k]),
                    Value::number(fit.fitted[k]),
                    Value::number(fit.residuals[k]),
                    Value::number(fit.weights[k]),
                ]);
            }
            rows.push(vec![Value::Empty; 5]);
            for (label, x) in [
                ("Omega", fit.omega),
                ("Theta", fit.theta),
                ("Sum of squares", fit.sum_of_squares),
            ] {
                let mut row = vec![Value::from(label), Value::number(x)];
                row.resize(5, Value::Empty);
                rows.push(row);
            }
            Ok(Grid::from_rows(rows))
        }
        Some("pattern") => Ok(Grid::row(
            &at()?
                .iter()
                .map(|&a| fit.percent_developed(a))
                .collect::<Vec<_>>(),
        )),
        Some("cdf") => Ok(Grid::row(
            &at()?.iter().map(|&a| fit.cdf(a)).collect::<Vec<_>>(),
        )),
        Some(other) => Err(format!(
            "output must be \"fit\", \"pattern\" or \"cdf\", got {other:?}"
        )),
    }
}
