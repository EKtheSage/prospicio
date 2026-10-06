//! Tail parity: tail factors and Mack with a tail against R ChainLadder
//! (`MackChainLadder(tail = TRUE / <number>, tail.se, tail.sigma)`) and
//! chainladder-python (`TailConstant`, `TailCurve`, `TailBondy` with
//! `MackChainladder`) on RAA, GenIns and ABC.
//!
//! Every row of both reference files is evaluated; see
//! `scripts/reserving_tails_r.R` and `scripts/reserving_tails_python.py`.

use std::collections::HashMap;

use act_reserving::{
    Average, CurveShape, Development, Mack, MackFit, Period, SigmaInterpolation, Tail, TailBondy,
    TailConstant, TailCurve,
};
use act_validation::{Case, check, reference, triangle};

/// The Mack model of a reference `method`, on `dataset` for the methods
/// whose inputs depend on it.
fn model(dataset: &str, method: &str) -> Mack {
    let constant = |factor: f64, decay: f64, attachment_age: Option<u32>| {
        Tail::Constant(TailConstant {
            factor,
            decay,
            attachment_age,
        })
    };
    let volume = Development::default();
    let (development, tail) = match method {
        // R ChainLadder.
        "mack_tail_loglinear" => (volume, Tail::LogLinear),
        "mack_tail_loglinear_sigma_mack" => (
            Development {
                sigma_interpolation: SigmaInterpolation::Mack,
                ..volume
            },
            Tail::LogLinear,
        ),
        "mack_tail_loglinear_alpha2" => (
            Development {
                average: Average::Regression,
                ..volume
            },
            Tail::LogLinear,
        ),
        "mack_tail_constant" => (volume, 1.05.into()),
        "mack_tail_constant_sigma_mack" => (
            Development {
                sigma_interpolation: SigmaInterpolation::Mack,
                ..volume
            },
            1.05.into(),
        ),
        "mack_tail_given" => {
            let (sigma, std_err) = given_tail(dataset);
            return Mack {
                tail: 1.05.into(),
                tail_sigma: Some(sigma),
                tail_std_err: Some(std_err),
                ..Default::default()
            };
        }
        // chainladder-python.
        "tail_constant" => (volume, constant(1.05, 0.5, None)),
        "tail_constant_decay" => (volume, constant(1.1, 0.75, None)),
        "tail_constant_attach" => (volume, constant(1.05, 0.5, Some(72))),
        "tail_constant_below_one" => (volume, constant(0.98, 0.5, None)),
        "tail_curve_exponential" => (volume, Tail::Curve(TailCurve::default())),
        "tail_curve_inverse_power" => (
            volume,
            Tail::Curve(TailCurve {
                curve: CurveShape::InversePower,
                ..Default::default()
            }),
        ),
        "tail_curve_fit_period" => (
            volume,
            Tail::Curve(TailCurve {
                fit_period: (Some(36), Some(108)),
                extrap_periods: 50,
                ..Default::default()
            }),
        ),
        "tail_curve_off_grid" => (
            volume,
            Tail::Curve(TailCurve {
                fit_period: (Some(30), Some(102)),
                ..Default::default()
            }),
        ),
        "tail_curve_attach" => (
            volume,
            Tail::Curve(TailCurve {
                attachment_age: Some(60),
                ..Default::default()
            }),
        ),
        "tail_bondy" => (volume, Tail::Bondy(TailBondy::default())),
        "tail_bondy_generalized" => (
            volume,
            Tail::Bondy(TailBondy {
                earliest_age: Some(36),
                attachment_age: None,
            }),
        ),
        "tail_bondy_off_grid" => (
            volume,
            Tail::Bondy(TailBondy {
                earliest_age: Some(30),
                attachment_age: None,
            }),
        ),
        "tail_bondy_attach" => (
            volume,
            Tail::Bondy(TailBondy {
                earliest_age: Some(36),
                attachment_age: Some(72),
            }),
        ),
        other => panic!("unknown method {other}"),
    };
    Mack {
        development,
        tail,
        ..Default::default()
    }
}

/// The tail sigma and standard error R's `mack_tail_given` passes in:
/// twice the last factor's sigma and half its standard error.
fn given_tail(dataset: &str) -> (f64, f64) {
    let dev = Development::default()
        .fit(&triangle(dataset), "values")
        .unwrap();
    let last = dev.ldf.len() - 1;
    (dev.sigma[last] * 2.0, dev.std_err[last] / 2.0)
}

/// Fits each (dataset, method) once and answers reference cases from it.
#[derive(Default)]
struct Fits {
    mack: HashMap<(String, String), MackFit>,
}

impl Fits {
    fn eval(&mut self, case: &Case) -> Option<f64> {
        let (dataset, method, quantity) = (
            case.get("dataset"),
            case.get("method"),
            case.get("quantity"),
        );
        match quantity {
            "given_tail_sigma" => return Some(given_tail(dataset).0),
            "given_tail_std_err" => return Some(given_tail(dataset).1),
            _ => {}
        }
        let fit = self
            .mack
            .entry((dataset.to_string(), method.to_string()))
            .or_insert_with(|| {
                model(dataset, method)
                    .fit(&triangle(dataset), "values")
                    .unwrap_or_else(|e| panic!("{dataset} {method}: {e}"))
            });
        let cl = &fit.chain_ladder;
        let tail = &cl.tail;
        let age = || case.number("arg").map(|k| k as usize);
        let origin = || {
            let year = case.number("arg")? as i32;
            cl.origins.iter().position(|&p| p == Period::year(year))
        };
        match quantity {
            "tail_factor" => Some(tail.factor),
            "tail_sigma" => Some(tail.sigma),
            "tail_std_err" => Some(tail.std_err),
            "ldf" => tail.ldf.get(age()?).copied(),
            // Past the oldest age, the cdf is the product of the tail's
            // factors from that age on.
            "cdf" => {
                let k = age()?;
                match cl.cdf.get(k) {
                    Some(&c) => Some(c),
                    None => (k < tail.ldf.len()).then(|| tail.ldf[k..].iter().product()),
                }
            }
            "ultimate" => Some(cl.ultimate[origin()?]),
            "reserve" => Some(cl.reserves()[origin()?]),
            "se" => Some(fit.standard_error[origin()?]),
            "process_risk" => Some(fit.process_risk[origin()?]),
            "parameter_risk" => Some(fit.parameter_risk[origin()?]),
            "total_ultimate" => Some(cl.total_ultimate()),
            "total_reserve" => Some(cl.total_reserve()),
            "total_standard_error" => Some(fit.total_standard_error),
            "total_process_risk" => Some(fit.total_process_risk),
            "total_parameter_risk" => Some(fit.total_parameter_risk),
            _ => None,
        }
    }
}

#[test]
fn matches_r_mack_with_tail() {
    let mut fits = Fits::default();
    check(&reference("reserving_tails_r.csv"), |c| fits.eval(c));
}

#[test]
fn matches_chainladder_python_tails() {
    let mut fits = Fits::default();
    check(&reference("reserving_tails_python.csv"), |c| fits.eval(c));
}

#[test]
fn exponential_curve_and_r_rule_agree() {
    // chainladder-python's default TailCurve and R's tail = TRUE fit the
    // same line to ln(f - 1) and multiply the same 100 extrapolated factors
    // whenever every factor is above 1, as on these three datasets.
    for dataset in ["raa", "genins", "abc"] {
        let tri = triangle(dataset);
        let fit = |tail: Tail| {
            Mack {
                tail,
                ..Default::default()
            }
            .fit(&tri, "values")
            .unwrap()
        };
        let (r, python) = (fit(Tail::LogLinear), fit(Tail::Curve(TailCurve::default())));
        let rel = (r.chain_ladder.tail.factor / python.chain_ladder.tail.factor - 1.0).abs();
        assert!(rel < 1e-12, "{dataset}: {rel}");
        let rel = (r.total_standard_error / python.total_standard_error - 1.0).abs();
        assert!(rel < 1e-9, "{dataset}: {rel}");
    }
}
