//! The ODP model as a quasi-Poisson GLM on a sheet (`docs/design/xll.md`,
//! decision 35): the reserves' analytic prediction error, the GLM's
//! coefficients and its residuals by cell or grouped
//! (`prospicio_reserving::OdpGlm`). Origins and development and calendar
//! periods are numbered from 1, as the triangle's rows and columns are;
//! an `origins` argument labels the origins.

use prospicio_core::Lag;
use prospicio_reserving::{OdpGlm, OdpGlmFit, ResidualGroup, SumPredictionError};

use crate::reserving::{origin_labels, text_option, triangle};
use crate::sheet::{Grid, Value};

type Result<T> = std::result::Result<T, String>;

fn fit(triangle_arg: &Grid) -> Result<OdpGlmFit> {
    OdpGlm::default()
        .fit(&triangle(triangle_arg)?, crate::reserving::COLUMN)
        .map_err(|e| e.to_string())
}

/// The development period, from 1, of an age in months.
fn period(age: Lag) -> f64 {
    f64::from(age / 12)
}

/// The latest cumulative value of each row of the triangle.
fn latest(grid: &Grid) -> Vec<f64> {
    (0..grid.rows)
        .map(|r| {
            (0..grid.columns)
                .filter_map(|c| match grid.get(r, c) {
                    Value::Number(x) => Some(*x),
                    _ => None,
                })
                .next_back()
                .unwrap_or(0.0)
        })
        .collect()
}

/// `PRO.ODP.GLM(triangle, [origins])`: per origin and in total, the
/// latest value, ultimate, reserve (the chain ladder's), process and
/// parameter standard errors, RMSEP and CV, Taylor and McGuire's Table 5-4.
pub fn summary(args: &[Grid]) -> Result<Grid> {
    let fit = fit(&args[0])?;
    let error = fit.prediction_error().map_err(|e| e.to_string())?;
    let labels = origin_labels(&args[1], fit.origins.len())?;
    let latest = latest(&args[0]);
    let row = |label: Value, latest: f64, e: &SumPredictionError| {
        vec![
            label,
            Value::number(latest),
            Value::number(latest + e.total),
            Value::number(e.total),
            Value::number(e.process_variance.sqrt()),
            Value::number(e.parameter_variance.sqrt()),
            Value::number(e.rmsep()),
            if e.total > 0.0 {
                Value::number(e.rmsep() / e.total)
            } else {
                Value::Empty
            },
        ]
    };
    let mut rows = vec![
        [
            "Origin",
            "Latest",
            "Ultimate",
            "Reserve",
            "Process SE",
            "Parameter SE",
            "RMSEP",
            "CV",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for ((label, e), l) in labels.into_iter().zip(&error.by_origin).zip(&latest) {
        rows.push(row(label, *l, e));
    }
    rows.push(row(Value::from("Total"), latest.iter().sum(), &error.total));
    Ok(Grid::from_rows(rows))
}

/// `PRO.ODP.GLM.PARAMS(triangle, [origins])`: a row per coefficient (its
/// term, level, estimate and standard error), the intercept first, then
/// the origins from the second and the development periods from the
/// second (the first of each is the reference), then Pearson's dispersion.
pub fn params(args: &[Grid]) -> Result<Grid> {
    let fit = fit(&args[0])?;
    let labels = origin_labels(&args[1], fit.origins.len())?;
    let names = fit.coding.names();
    let (estimates, errors) = (fit.glm.coefficients(), fit.glm.std_errors());
    // Each coefficient's term, level and sort key.
    let mut coefficients: Vec<(usize, f64, Value, Value, usize)> = names
        .iter()
        .enumerate()
        .map(|(j, name)| {
            let level = |prefix: &str| {
                name.strip_prefix(prefix)
                    .and_then(|s| s.strip_suffix(']'))
                    .ok_or_else(|| format!("unexpected coefficient {name:?}"))
            };
            let (key, term, label) = if name == "(Intercept)" {
                (0, Value::from("Intercept"), Value::Empty)
            } else if name.starts_with("origin[") {
                let level = level("origin[")?;
                let o = fit
                    .origins
                    .iter()
                    .position(|p| p.to_string() == level)
                    .ok_or_else(|| format!("unexpected coefficient {name:?}"))?;
                (1 + o, Value::from("Origin"), labels[o].clone())
            } else {
                let age: Lag = level("development[")?
                    .parse()
                    .map_err(|_| format!("unexpected coefficient {name:?}"))?;
                (
                    1 + fit.origins.len() + age as usize,
                    Value::from("Development"),
                    Value::Number(period(age)),
                )
            };
            Ok((key, estimates[j], term, label, j))
        })
        .collect::<Result<_>>()?;
    coefficients.sort_by_key(|c| c.0);
    let mut rows = vec![
        ["Term", "Level", "Estimate", "SE"]
            .map(Value::from)
            .to_vec(),
    ];
    for (_, estimate, term, label, j) in coefficients {
        rows.push(vec![
            term,
            label,
            Value::number(estimate),
            Value::number(errors[j]),
        ]);
    }
    rows.push(vec![
        Value::from("Dispersion"),
        Value::Empty,
        Value::number(fit.glm.dispersion()),
        Value::Empty,
    ]);
    Ok(Grid::from_rows(rows))
}

/// `PRO.ODP.GLM.RESIDUALS(triangle, [by], [origins])`: with `by` blank or
/// "cell", a row per observed increment, origin-major: origin, development
/// and calendar period, actual, fitted, actual to fitted, Pearson and
/// deviance residuals, leverage and the standardized deviance and Pearson
/// residuals (blank for a cell fitted exactly, of leverage 1). With `by`
/// "origin", "development" or "calendar", the standardized deviance
/// residuals of each group: their count, mean and standard deviation.
pub fn residuals(args: &[Grid]) -> Result<Grid> {
    let fit = fit(&args[0])?;
    let labels = origin_labels(&args[2], fit.origins.len())?;
    let r = fit.residuals().map_err(|e| e.to_string())?;
    let origin = |p| {
        fit.origins
            .iter()
            .position(|o| *o == p)
            .expect("every observed cell's origin is an origin of the fit")
    };
    let group = match text_option(&args[1], "by")?.as_deref() {
        None | Some("cell" | "cells") => None,
        Some("origin") => Some(ResidualGroup::Origin),
        Some("development") => Some(ResidualGroup::Development),
        Some("calendar") => Some(ResidualGroup::Calendar),
        Some(other) => {
            return Err(format!(
                "by must be \"cell\", \"origin\", \"development\" or \"calendar\", got {other:?}"
            ));
        }
    };
    let Some(group) = group else {
        let mut rows = vec![
            [
                "Origin",
                "Development",
                "Calendar",
                "Actual",
                "Fitted",
                "Actual/Fitted",
                "Pearson",
                "Deviance",
                "Leverage",
                "Std deviance",
                "Std Pearson",
            ]
            .map(Value::from)
            .to_vec(),
        ];
        for i in 0..r.actual.len() {
            rows.push(vec![
                labels[origin(r.origin[i])].clone(),
                Value::Number(period(r.development[i])),
                Value::Number(r.calendar[i].0 as f64 + 1.0),
                Value::number(r.actual[i]),
                Value::number(r.fitted[i]),
                Value::number(r.actual_to_fitted[i]),
                Value::number(r.pearson[i]),
                Value::number(r.deviance[i]),
                Value::number(r.leverage[i]),
                Value::number(r.standardized_deviance[i]),
                Value::number(r.standardized_pearson[i]),
            ]);
        }
        return Ok(Grid::from_rows(rows));
    };
    let summary = r.summary(group);
    // The label of each group as the sheet numbers it: the origin's own
    // label, or the development or calendar period from 1. The summary's
    // groups are in the order of the first cell of each.
    let label = |k: usize, text: &str| -> Value {
        match group {
            ResidualGroup::Origin => labels[k].clone(),
            ResidualGroup::Development => text
                .parse::<Lag>()
                .map_or(Value::from(text), |age| Value::Number(period(age))),
            ResidualGroup::Calendar => Value::Number(k as f64 + 1.0),
        }
    };
    let mut rows = vec![
        [
            match group {
                ResidualGroup::Origin => "Origin",
                ResidualGroup::Development => "Development",
                ResidualGroup::Calendar => "Calendar",
            },
            "N",
            "Mean",
            "SD",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for (k, g) in summary.iter().enumerate() {
        rows.push(vec![
            label(k, &g.label),
            Value::Number(g.n as f64),
            Value::number(g.mean),
            Value::number(g.sd),
        ]);
    }
    Ok(Grid::from_rows(rows))
}
