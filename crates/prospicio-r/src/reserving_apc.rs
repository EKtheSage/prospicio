//! `apc_glm()`: the age-period-cohort reserving GLM (issue #979,
//! `docs/design/reserving-inflation.md`, PR 7), over
//! `prospicio_reserving::apc`.
//!
//! The R side holds the fit's fields in an S7 object. The future cells come
//! back as the columns `reserving_inflation::cells_list` gives, and the
//! underlying ODP GLM fit as an `OdpGlmFit` pointer, which carries the
//! prediction error, the draws and the residuals.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_reserving::{ApcEffects, ApcGlm, ApcGlmFit, Extrapolation, FutureCells};

use crate::odp_glm::OdpGlmFit;
use crate::reserving::{Triangle, to_r};
use crate::reserving_inflation::cells_list;
use crate::whole;

fn effects_of(text: &str) -> Result<ApcEffects> {
    match text.to_ascii_uppercase().as_str() {
        "APC" => Ok(ApcEffects::Apc),
        "AP" => Ok(ApcEffects::AgePeriod),
        "AC" => Ok(ApcEffects::AccidentAge),
        "PC" => Ok(ApcEffects::AccidentPeriod),
        _ => Err(Error::Other(format!(
            "effects must be \"APC\", \"AP\", \"AC\" or \"PC\", got {text:?}"
        ))),
    }
}

fn effects_name(effects: ApcEffects) -> &'static str {
    match effects {
        ApcEffects::Apc => "APC",
        ApcEffects::AgePeriod => "AP",
        ApcEffects::AccidentAge => "AC",
        ApcEffects::AccidentPeriod => "PC",
    }
}

fn extrapolation_of(text: &str) -> Result<Extrapolation> {
    match text.to_ascii_uppercase().as_str() {
        "I0" => Ok(Extrapolation::I0),
        "I1" => Ok(Extrapolation::I1),
        "I2" => Ok(Extrapolation::I2),
        _ => Err(Error::Other(format!(
            "extrapolation must be \"I0\", \"I1\" or \"I2\", got {text:?}"
        ))),
    }
}

fn extrapolation_name(rule: Extrapolation) -> &'static str {
    match rule {
        Extrapolation::I0 => "I0",
        Extrapolation::I1 => "I1",
        Extrapolation::I2 => "I2",
    }
}

/// Fits the APC GLM of `apc_glm()` to `column` of a single-segment triangle,
/// with the `effects` and `extrapolation` named as R `apc` names them, the
/// error `family` ("poisson", "gamma" or "tweedie", whose `power` is NaN
/// otherwise), the IRLS `tolerance` and the iteration limit. The fitted cells come row-major, origins by ages, NaN
/// below the latest diagonal.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn apc_glm_fit_rust(
    triangle: &Triangle,
    column: &str,
    effects: &str,
    extrapolation: &str,
    family: &str,
    power: f64,
    tolerance: f64,
    max_iterations: f64,
) -> Result<List> {
    if !matches!(family, "poisson" | "gamma" | "tweedie") {
        return Err(Error::Other(format!(
            "family must be \"poisson\", \"gamma\" or \"tweedie\", not \"{family}\""
        )));
    }
    let model = ApcGlm {
        family: crate::models::family(family, f64::NAN, power)?,
        effects: effects_of(effects)?,
        extrapolation: extrapolation_of(extrapolation)?,
        tolerance,
        max_iterations: whole(max_iterations, "max_iterations")? as usize,
    };
    let fit: ApcGlmFit = model.fit(&triangle.inner, column).map_err(to_r)?;
    let future = cells_list(&FutureCells::from_apc_glm(&fit).map_err(to_r)?);
    let fitted: Vec<f64> = fit
        .fitted
        .iter()
        .flat_map(|row| row.iter().map(|c| c.unwrap_or(f64::NAN)))
        .collect();
    let origins: Vec<String> = fit.origins.iter().map(ToString::to_string).collect();
    let ages: Vec<i32> = fit.ages.iter().map(|&a| a as i32).collect();
    let out = list!(
        effects = effects_name(fit.effects),
        extrapolation = extrapolation_name(fit.extrapolation),
        origins = origins,
        ages = ages,
        level = fit.level,
        age_slope = fit.age_slope,
        accident_slope = fit.accident_slope,
        age_second_differences = fit.age_second_differences.clone(),
        accident_second_differences = fit.accident_second_differences.clone(),
        calendar_second_differences = fit.calendar_second_differences.clone(),
        future_calendar_second_differences = fit.future_calendar_second_differences.to_vec(),
        fitted = fitted,
        deviance = fit.deviance,
        reserves = fit.reserves.clone(),
        future = future,
        odp = OdpGlmFit { inner: fit.fit }
    );
    Ok(out)
}

extendr_module! {
    mod reserving_apc;
    fn apc_glm_fit_rust;
}
