//! The age-period-cohort reserving GLM (`reserving-inflation.md`, PR 6,
//! #979, validation target 4) against Python `apc` 1.0.2
//! (`scripts/reserving_apc_python.py`) on Taylor-Ashe (GenIns): the deviance,
//! fitted values and canonical level, slopes and second differences of the
//! Poisson APC, AC, AP and PC models, the reserves under the `I2`, `I1` and
//! `I0` rules (the generator applies the rules to `apc`'s fitted second
//! differences: Python `apc` has no `I1` or `I0`), and R `apc`'s documented
//! `I0` total of 19,061,428.

use std::collections::HashMap;

use prospicio_reserving::{ApcEffects, ApcGlm, ApcGlmFit, Extrapolation, Period};
use prospicio_validation::{check, reference, triangle};

fn fit(method: &str) -> ApcGlmFit {
    let (effects, rule) = match method {
        "poisson_apc" => (ApcEffects::Apc, Extrapolation::I0),
        "poisson_ac" => (ApcEffects::AccidentAge, Extrapolation::I0),
        "poisson_ap" => (ApcEffects::AgePeriod, Extrapolation::I0),
        "poisson_pc" => (ApcEffects::AccidentPeriod, Extrapolation::I0),
        other => {
            let (base, rule) = other.rsplit_once('_').unwrap();
            let rule = match rule {
                "i2" => Extrapolation::I2,
                "i1" => Extrapolation::I1,
                "i0" => Extrapolation::I0,
                _ => panic!("unknown method {other}"),
            };
            let (effects, _) = match base {
                "poisson_apc" => (ApcEffects::Apc, ()),
                "poisson_ap" => (ApcEffects::AgePeriod, ()),
                "poisson_pc" => (ApcEffects::AccidentPeriod, ()),
                _ => panic!("unknown method {other}"),
            };
            return ApcGlm {
                effects,
                extrapolation: rule,
                ..ApcGlm::default()
            }
            .fit(&triangle("genins"), "values")
            .unwrap();
        }
    };
    ApcGlm {
        effects,
        extrapolation: rule,
        ..ApcGlm::default()
    }
    .fit(&triangle("genins"), "values")
    .unwrap()
}

#[test]
fn apc_matches_python_apc() {
    let mut fits: HashMap<String, ApcGlmFit> = HashMap::new();
    check(&reference("reserving_apc_python.csv"), |case| {
        let method = case.get("method");
        let fit = fits
            .entry(method.into())
            .or_insert_with(|| fit(method));
        let origin = |year: f64| {
            fit.origins
                .iter()
                .position(|&p| p == Period::year(year as i32))
        };
        let age = |months: f64| fit.ages.iter().position(|&a| f64::from(a) == months);
        match case.get("quantity") {
            "deviance" => Some(fit.deviance),
            "level" => Some(fit.level),
            "slope_age" => Some(fit.age_slope),
            "slope_cohort" => Some(fit.accident_slope),
            "dd_age" => fit
                .age_second_differences
                .get((case.number("arg")? as usize).checked_sub(2)?)
                .copied(),
            "dd_cohort" => fit
                .accident_second_differences
                .get((case.number("arg")? as usize).checked_sub(2)?)
                .copied(),
            "dd_period" => fit
                .calendar_second_differences
                .get((case.number("arg")? as usize).checked_sub(2)?)
                .copied(),
            "fitted" => {
                let (o, a) = (
                    origin(case.param("arg", "origin"))?,
                    age(case.param("arg", "age"))?,
                );
                fit.fitted[o][a]
            }
            "future" => {
                let (o, a) = (origin(case.param("arg", "origin"))?, case.param("arg", "age"));
                let cells = prospicio_reserving::FutureCells::from_apc_glm(fit);
                cells
                    .cells
                    .iter()
                    .find(|c| c.origin == o && f64::from(c.age) == a)
                    .map(|c| c.value)
            }
            "reserve" => Some(fit.reserves[origin(case.number("arg")?)?]),
            "total_reserve" | "documented_total" => Some(fit.total_reserve()),
            _ => None,
        }
    });
}
