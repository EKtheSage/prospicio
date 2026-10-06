//! Clark parity: `ClarkLdf` and `ClarkCapeCod` against R ChainLadder's
//! `ClarkLDF` and `ClarkCapeCod` (GenIns, RAA, GenIns with premium) and
//! chainladder-python's `ClarkLDF` where it is comparable.
//!
//! Every row of `reference/reserving_clark_r.csv` and
//! `reference/reserving_clark_python.csv` is evaluated; see
//! `scripts/reserving_clark_r.R` and `scripts/reserving_clark_python.py`. A
//! reference `method` is `clark_ldf` or `clark_cape_cod` followed by
//! `curve=`, `max_age=` (months, `inf` for none) and, for R, `optim=`.
//!
//! Tolerances reflect the reference optimizers, since ours runs to the
//! maximum (Nelder–Mead on the profiled likelihood to 1e-10):
//!
//! * R as shipped (`optim=default`, rel_tol 1e-2): its L-BFGS-B stops at a
//!   relative log-likelihood change of 1.5e-8, which leaves parameters,
//!   reserves and standard errors up to 4e-3 short of the maximum (measured
//!   by the generator against `optim=tight`; our largest gap is 3.8e-3, a
//!   theta standard error).
//! * R run to convergence (`optim=tight`, rel_tol 1e-5): factr = 1 leaves
//!   about 1e-7 in the parameters; standard errors, which go through the
//!   inverse Hessian, carry more (our largest gap is 1.7e-6, a parameter
//!   standard error).
//! * chainladder-python (rel_tol 1e-3): scipy's L-BFGS-B with default
//!   tolerances on the unscaled likelihood stops short (our largest gap is
//!   1.1e-4, a theta).

use std::collections::HashMap;

use act_reserving::{ClarkCapeCod, ClarkFit, ClarkLdf, GrowthCurve, Period};
use act_validation::{Case, check, reference, triangle, triangle_columns};

fn fit(dataset: &str, method: &str) -> ClarkFit {
    let mut parts = method.split(';');
    let name = parts.next().unwrap();
    let setting = |key: &str| {
        method
            .split(';')
            .find_map(|p| p.strip_prefix(key)?.strip_prefix('='))
            .unwrap_or_else(|| panic!("{method}: no {key}"))
    };
    let curve = match setting("curve") {
        "loglogistic" => GrowthCurve::LogLogistic,
        "weibull" => GrowthCurve::Weibull,
        other => panic!("{method}: unknown curve {other}"),
    };
    let max_age = match setting("max_age") {
        "inf" => None,
        m => Some(m.parse().unwrap()),
    };
    let fail = |e: act_reserving::Error| -> ! { panic!("{dataset} {method}: {e}") };
    match name {
        "clark_ldf" => {
            let (tri, column) = if dataset == "genins_premium" {
                (triangle_columns(dataset, &["paid", "premium"]), "paid")
            } else {
                (triangle(dataset), "values")
            };
            ClarkLdf { curve, max_age }
                .fit(&tri, column)
                .unwrap_or_else(|e| fail(e))
        }
        "clark_cape_cod" => {
            let tri = triangle_columns(dataset, &["paid", "premium"]);
            ClarkCapeCod { curve, max_age }
                .fit(&tri, "paid", "premium")
                .unwrap_or_else(|e| fail(e))
        }
        other => panic!("unknown method {other}"),
    }
}

/// The parameter standard error of `name` (`elr`, `omega` or `theta`).
fn parameter_se(f: &ClarkFit, name: &str) -> Option<f64> {
    let p = f.covariance.len();
    let index = match name {
        "omega" => p - 2,
        "theta" => p - 1,
        "elr" if f.elr.is_some() => 0,
        _ => return None,
    };
    Some(f.covariance[index][index].sqrt())
}

/// Evaluates every case of `file`, fitting each (dataset, method) once.
/// The R `optim` setting does not change our fit.
fn evaluate(file: &str) {
    let mut fits: HashMap<(String, String), ClarkFit> = HashMap::new();
    check(&reference(file), |case: &Case| {
        let dataset = case.get("dataset");
        let method: String = case
            .get("method")
            .split(';')
            .filter(|p| !p.starts_with("optim="))
            .collect::<Vec<_>>()
            .join(";");
        let key = (dataset.to_string(), method.clone());
        let f = fits.entry(key).or_insert_with(|| fit(dataset, &method));
        let origin = || {
            let year = case.number("arg")? as i32;
            f.chain_ladder
                .origins
                .iter()
                .position(|&p| p == Period::year(year))
        };
        match case.get("quantity") {
            "omega" => Some(f.omega),
            "theta" => Some(f.theta),
            "elr" => f.elr,
            "sigma2" => Some(f.scale),
            "omega_se" => parameter_se(f, "omega"),
            "theta_se" => parameter_se(f, "theta"),
            "elr_se" => parameter_se(f, "elr"),
            "expected_ultimate" => Some(f.expected_ultimate[origin()?]),
            "ultimate" => Some(f.ultimate[origin()?]),
            "reserve" => Some(f.reserves()[origin()?]),
            "process_se" => Some(f.process_risk[origin()?]),
            "parameter_se" => Some(f.parameter_risk[origin()?]),
            "standard_error" => Some(f.standard_error[origin()?]),
            "total_ultimate" => Some(f.total_ultimate()),
            "total_reserve" => Some(f.total_reserve()),
            "total_process_se" => Some(f.total_process_risk),
            "total_parameter_se" => Some(f.total_parameter_risk),
            "total_standard_error" => Some(f.total_standard_error),
            // chainladder-python's ldf_: the link ratio from development
            // age `arg` (months) to the next, G(next) / G(arg).
            "ldf" => {
                let age = case.number("arg")?;
                Some(f.growth(age + 12.0) / f.growth(age))
            }
            _ => None,
        }
    });
}

#[test]
fn matches_r_chainladder() {
    evaluate("reserving_clark_r.csv");
}

#[test]
fn matches_chainladder_python() {
    evaluate("reserving_clark_python.csv");
}
