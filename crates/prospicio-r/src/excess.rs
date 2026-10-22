//! `excess_pattern()` and its companions (issue #192,
//! `docs/design/reserving-excess.md`, PR 3): a layer's development pattern
//! derived from the ground-up triangle, over `prospicio_reserving::excess`.
//!
//! The pattern comes back as plain numbers; the R side holds them in an S7
//! object and gives them back for `credibility_weights`. Exposure curves
//! reach the reserving crate only through the shares the R side passes
//! (decision 2).

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_core::Lag;
use prospicio_reserving::PatternWeight;
use prospicio_reserving::excess::{
    Basis, ExcessPattern, Layer, SeverityByMaturity, ShapeBase, layer_at_shift, layer_exposure,
    layer_share,
};

use crate::distributions::severity_from_robj;
use crate::reserving::{Triangle, tail_arg};
use crate::whole;

fn to_r(e: prospicio_reserving::Error) -> Error {
    Error::Other(e.to_string())
}

fn layer(attachment: f64, limit: f64) -> Result<Layer> {
    Layer::new(attachment, limit).map_err(to_r)
}

fn lag(x: f64, what: &str) -> Result<Lag> {
    Lag::try_from(whole(x, what)?).map_err(|_| Error::Other(format!("{what} is too large")))
}

/// The pattern's numbers as a named list.
fn pattern_list(p: &ExcessPattern) -> Result<List> {
    let ages: Vec<f64> = p.ages.iter().map(|&a| f64::from(a)).collect();
    List::from_names_and_values(
        [
            "ages",
            "layer_ldf",
            "layer_cdf",
            "ground_up_ldf",
            "ground_up_tail",
            "shifts",
            "layer_tail",
        ],
        [
            Robj::from(ages),
            Robj::from(p.layer_ldf.clone()),
            Robj::from(p.layer_cdf.clone()),
            Robj::from(p.ground_up_ldf.clone()),
            Robj::from(p.ground_up_tail),
            Robj::from(p.shifts.clone()),
            Robj::from(p.layer_tail),
        ],
    )
}

/// The layer pattern of `excess_pattern()`. `kind` is "scale_shift"
/// (`ultimate` the severity, `ground_up_cap` NaN for none) or
/// "pareto_shape" (`threshold`, the shapes `alphas` at `alpha_ages`,
/// `ultimate_alpha`, and the base: NaN `base_attachment` for the counts, else
/// the base layer). `counts` is `NULL` or a column name; `tail` a tail
/// estimator.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn excess_pattern_rust(
    triangle: Robj,
    column: &str,
    attachment: f64,
    limit: f64,
    kind: &str,
    ultimate: Robj,
    ground_up_cap: f64,
    threshold: f64,
    alpha_ages: Vec<f64>,
    alphas: Vec<f64>,
    ultimate_alpha: f64,
    base_attachment: f64,
    base_limit: f64,
    counts: Robj,
    tail: Robj,
) -> Result<List> {
    let triangle = <&Triangle>::try_from(&triangle)
        .map_err(|_| Error::Other("triangle must be a triangle".into()))?;
    let layer = layer(attachment, limit)?;
    let model = match kind {
        "scale_shift" => {
            let sev = severity_from_robj(&ultimate)?;
            if ground_up_cap.is_nan() {
                SeverityByMaturity::scale_shift(Box::new(sev))
            } else {
                SeverityByMaturity::scale_shift_capped(Box::new(sev), ground_up_cap)
            }
        }
        "pareto_shape" => {
            if alpha_ages.len() != alphas.len() {
                return Err(Error::Other("alphas needs one shape for each age".into()));
            }
            let pairs = alpha_ages
                .iter()
                .zip(&alphas)
                .map(|(&a, &x)| Ok((lag(a, "alphas: an age")?, x)))
                .collect::<Result<Vec<_>>>()?;
            let base = if base_attachment.is_nan() {
                ShapeBase::Counts
            } else {
                ShapeBase::Losses(self::layer(base_attachment, base_limit)?)
            };
            SeverityByMaturity::pareto_shape(threshold, pairs, ultimate_alpha, base)
        }
        _ => {
            return Err(Error::Other(format!(
                "model must be a scale_shift() or a pareto_shape(), not \"{kind}\""
            )));
        }
    };
    let counts: Option<&str> = counts.as_str();
    let pattern = ExcessPattern::from_ground_up(
        &triangle.inner,
        column,
        Basis::Incurred,
        counts,
        &tail_arg(&tail)?,
        layer,
        &model,
    )
    .map_err(to_r)?;
    pattern_list(&pattern)
}

/// The weights of `excess_credibility_weights()`, from a pattern's numbers
/// (`credibility_weights` reads the ages and the layer factors): a list of
/// `ages` and `weights`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn excess_credibility_weights_rust(
    ages: Vec<f64>,
    layer_ldf: Vec<f64>,
    layer_cdf: Vec<f64>,
    ground_up_ldf: Vec<f64>,
    ground_up_tail: f64,
    shifts: Vec<f64>,
    layer_tail: f64,
    layer_triangle: Robj,
    column: &str,
    counts: Robj,
    k: f64,
    max_influence: f64,
) -> Result<List> {
    let triangle = <&Triangle>::try_from(&layer_triangle)
        .map_err(|_| Error::Other("layer_triangle must be a triangle".into()))?;
    let pattern = ExcessPattern {
        ages: ages
            .iter()
            .map(|&a| lag(a, "ages"))
            .collect::<Result<Vec<_>>>()?,
        layer_ldf,
        layer_cdf,
        ground_up_ldf,
        ground_up_tail,
        shifts,
        layer_tail,
    };
    let counts: Option<&str> = counts.as_str();
    let weight = pattern
        .credibility_weights(&triangle.inner, column, counts, k, max_influence)
        .map_err(to_r)?;
    let PatternWeight::ByAge(by_age) = weight else {
        return Err(Error::Other("expected a weight by age".into()));
    };
    let (w_ages, w): (Vec<f64>, Vec<f64>) =
        by_age.into_iter().map(|(a, x)| (f64::from(a), x)).unzip();
    List::from_names_and_values(["ages", "weights"], [Robj::from(w_ages), Robj::from(w)])
}

/// The share of `layer_share()`.
#[extendr]
fn excess_layer_share_rust(severity: Robj, attachment: f64, limit: f64) -> Result<f64> {
    let sev = severity_from_robj(&severity)?;
    layer_share(&sev, layer(attachment, limit)?).map_err(to_r)
}

/// The loss to the layer at a shift of `layer_at_shift()`.
#[extendr]
fn excess_layer_at_shift_rust(
    severity: Robj,
    attachment: f64,
    limit: f64,
    shift: f64,
) -> Result<f64> {
    if !(shift.is_finite() && shift > 0.0) {
        return Err(Error::Other("shift must be finite and positive".into()));
    }
    let sev = severity_from_robj(&severity)?;
    Ok(layer_at_shift(&sev, layer(attachment, limit)?, shift))
}

/// The triangle of `layer_exposure()`.
#[extendr]
fn excess_layer_exposure_rust(
    triangle: Robj,
    exposure: &str,
    shares: Vec<f64>,
    name: &str,
) -> Result<Triangle> {
    let triangle = <&Triangle>::try_from(&triangle)
        .map_err(|_| Error::Other("triangle must be a triangle".into()))?;
    let inner = layer_exposure(&triangle.inner, exposure, &shares, name).map_err(to_r)?;
    Ok(Triangle { inner })
}

extendr_module! {
    mod excess;
    fn excess_pattern_rust;
    fn excess_credibility_weights_rust;
    fn excess_layer_share_rust;
    fn excess_layer_at_shift_rust;
    fn excess_layer_exposure_rust;
}
