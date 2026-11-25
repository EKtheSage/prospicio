//! `separation()`: Taylor's separation method, arithmetic and geometric
//! (issue #979, `docs/design/reserving-inflation.md`, PR 5), over
//! `prospicio_reserving::separation`.
//!
//! The R side passes the method's settings as plain values and holds the
//! fit's fields in an S7 object; the future cells come back as the columns
//! `reserving_inflation::cells_list` gives.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_reserving::{
    FutureInflation, Retransformation, Separation, SeparationFit, SeparationKind,
};

use crate::reserving::{Triangle, to_r};
use crate::reserving_inflation::{cells_list, index};

fn kind_of(text: &str) -> Result<SeparationKind> {
    match text {
        "arithmetic" => Ok(SeparationKind::Arithmetic),
        "geometric" => Ok(SeparationKind::Geometric(Retransformation::GeometricMean)),
        "geometric_lognormal" => Ok(SeparationKind::Geometric(Retransformation::LognormalMean)),
        other => Err(Error::Other(format!(
            "kind must be \"arithmetic\", \"geometric\" or \"geometric_lognormal\", got {other:?}"
        ))),
    }
}

/// Fits `column` of the triangle (the one segment chosen by `keys` =
/// `values`, none needed for a single segment). `future` is `"rate"`
/// (`rate`), `"index"` (the calendar index `first`, `index_values`,
/// `index_rate`) or `"trend"` (`trend_periods`); the settings of the other
/// kinds are ignored.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn separation_fit_rust(
    triangle: &Triangle,
    column: &str,
    keys: Vec<String>,
    values: Vec<String>,
    kind: &str,
    normaliser: Nullable<Vec<f64>>,
    future: &str,
    rate: f64,
    first: &str,
    index_values: Vec<f64>,
    index_rate: Nullable<f64>,
    trend_periods: i32,
) -> Result<List> {
    if keys.len() != values.len() {
        return Err(Error::Other("one value is needed per key".into()));
    }
    let future = match future {
        "rate" => FutureInflation::Rate(rate),
        "index" => FutureInflation::Index(index(first, index_values, index_rate)?),
        "trend" => FutureInflation::Trend {
            periods: usize::try_from(trend_periods)
                .ok()
                .filter(|&p| p >= 2)
                .ok_or_else(|| {
                    Error::Other(format!(
                        "trend_periods must be a whole number of at least 2, got {trend_periods}"
                    ))
                })?,
        },
        other => {
            return Err(Error::Other(format!(
                "future must be \"rate\", \"index\" or \"trend\", got {other:?}"
            )));
        }
    };
    let model = Separation {
        kind: kind_of(kind)?,
        normaliser: match normaliser {
            Nullable::NotNull(v) => Some(v),
            Nullable::Null => None,
        },
        future,
    };
    let fits = model.fit_segments(&triangle.inner, column).map_err(to_r)?;
    let choice: Vec<(&str, &str)> = keys
        .iter()
        .zip(&values)
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let at = fits.position(&choice).map_err(to_r)?;
    let fit: &SeparationFit = &fits.fits[at];
    let fitted: Vec<f64> = fit
        .fitted
        .iter()
        .flat_map(|row| row.iter().map(|c| c.unwrap_or(f64::NAN)))
        .collect();
    let log_variance = match fit.log_variance {
        Some(v) => Robj::from(v),
        None => Robj::from(()),
    };
    Ok(list!(
        origins = fit
            .origins
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ages = fit.ages.iter().map(|&a| a as i32).collect::<Vec<_>>(),
        normaliser = fit.normaliser.clone(),
        proportions = fit.proportions.clone(),
        calendar = fit
            .calendar
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        indices = fit.indices.clone(),
        future_period = fit
            .future_indices
            .iter()
            .map(|(p, _)| p.to_string())
            .collect::<Vec<_>>(),
        future_index = fit
            .future_indices
            .iter()
            .map(|(_, v)| *v)
            .collect::<Vec<_>>(),
        fitted = fitted,
        future = cells_list(&fit.future),
        reserves = fit.reserves.clone(),
        log_variance = log_variance
    ))
}

extendr_module! {
    mod reserving_separation;
    fn separation_fit_rust;
}
