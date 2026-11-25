//! The age-period-cohort reserving GLM (`reserving-inflation.md`, PR 6,
//! #979, validation target 4) against Python `apc` 1.0.2
//! (`scripts/reserving_apc_python.py`) on Taylor-Ashe (GenIns): the deviance,
//! fitted values and canonical level, slopes and second differences of the
//! Poisson APC, AC, AP and PC models, the reserves under the `I2`, `I1` and
//! `I0` rules (the generator applies the rules to `apc`'s fitted second
//! differences: Python `apc` has no `I1` or `I0`), and R `apc`'s documented
//! `I0` total of 19,061,428.

use std::collections::{BTreeMap, HashMap};

use prospicio_reserving::{
    ApcEffects, ApcGlm, ApcGlmFit, Extrapolation, FutureInflation, OdpGlmFit, Period, Separation,
    SeparationKind,
};
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

/// The separation methods of the prediction error rows: the arithmetic
/// method as a Poisson GLM, with a 5% rate future and a trend over the last
/// 4 calendar effects.
fn separation(method: &str) -> OdpGlmFit {
    let future = match method {
        "separation_rate" => FutureInflation::Rate(0.05),
        "separation_trend" => FutureInflation::Trend { periods: 4 },
        other => panic!("unknown method {other}"),
    };
    Separation {
        kind: SeparationKind::Arithmetic,
        normaliser: None,
        future,
    }
    .glm(&triangle("genins"), "values")
    .unwrap()
}

/// A row of the prediction error: the process or parameter variance of an
/// origin's reserve (arg: the year) or of the total (no arg).
fn prediction_error(fit: &OdpGlmFit, quantity: &str, arg: &str) -> Option<f64> {
    let error = fit.prediction_error().unwrap();
    let sum = if arg.is_empty() {
        error.total
    } else {
        let year: i32 = arg.parse().ok()?;
        let o = fit.origins.iter().position(|&p| p == Period::year(year))?;
        error.by_origin[o]
    };
    match quantity {
        "process_variance" => Some(sum.process_variance),
        "parameter_variance" => Some(sum.parameter_variance),
        _ => None,
    }
}

#[test]
fn apc_matches_python_apc() {
    let mut fits: HashMap<String, ApcGlmFit> = HashMap::new();
    let mut glms: HashMap<String, OdpGlmFit> = HashMap::new();
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    let mut methods: BTreeMap<(String, String), usize> = BTreeMap::new();
    check(&reference("reserving_apc_python.csv"), |case| {
        let method = case.get("method");
        *seen.entry(case.get("quantity").into()).or_default() += 1;
        *methods
            .entry((method.into(), case.get("quantity").into()))
            .or_default() += 1;
        if method.starts_with("separation") {
            let glm = glms
                .entry(method.into())
                .or_insert_with(|| separation(method));
            return match case.get("quantity") {
                "total_reserve" => Some(glm.total_reserve()),
                quantity => prediction_error(glm, quantity, case.get("arg")),
            };
        }
        let fit = fits
            .entry(method.into())
            .or_insert_with(|| fit(method));
        if let q @ ("process_variance" | "parameter_variance") = case.get("quantity") {
            return prediction_error(&fit.fit, q, case.get("arg"));
        }
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
                let cells = prospicio_reserving::FutureCells::from_apc_glm(fit).ok()?;
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

    // A generator that drops a whole quantity or method must fail here.
    let counts: Vec<(&str, usize)> = seen.iter().map(|(q, n)| (q.as_str(), *n)).collect();
    assert_eq!(
        counts,
        [
            ("dd_age", 24),
            ("dd_cohort", 24),
            ("dd_period", 24),
            ("deviance", 4),
            ("documented_total", 1),
            ("fitted", 220),
            ("future", 450),
            ("level", 4),
            ("parameter_variance", 120),
            ("process_variance", 120),
            ("reserve", 100),
            ("slope_age", 4),
            ("slope_cohort", 4),
            ("total_reserve", 12),
        ]
    );
    for (method, quantity) in [
        ("poisson_apc", "dd_period"),
        ("poisson_ap", "dd_period"),
        ("poisson_pc", "dd_period"),
        ("poisson_apc_i1", "total_reserve"),
        ("poisson_apc_i0", "total_reserve"),
        ("poisson_ac", "total_reserve"),
        ("poisson_ac", "parameter_variance"),
        ("poisson_apc_i0", "process_variance"),
        ("separation_trend", "parameter_variance"),
        ("separation_rate", "total_reserve"),
    ] {
        assert!(
            methods.contains_key(&(method.to_string(), quantity.to_string())),
            "{method} {quantity}"
        );
    }
}
