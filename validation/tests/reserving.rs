//! Reserving parity: Chain Ladder and Mack against R ChainLadder and
//! chainladder-python on RAA, GenIns and ABC.
//!
//! Every row of both reference files is evaluated; see
//! `scripts/reserving_r.R` and `scripts/reserving_chainladder_python.py`.

use std::collections::HashMap;

use act_reserving::{
    Average, ChainLadder, ChainLadderFit, Development, Mack, MackFit, Period, SigmaInterpolation,
};
use act_validation::{Case, check, reference, triangle};

/// Development estimator for a reference `method`.
fn development(method: &str) -> Development {
    let (average, sigma_interpolation) = match method {
        "chain_ladder" | "mack" => (Average::Volume, SigmaInterpolation::LogLinear),
        "chain_ladder_simple" | "mack_alpha0" => (Average::Simple, SigmaInterpolation::LogLinear),
        "mack_alpha2" => (Average::Regression, SigmaInterpolation::LogLinear),
        "mack_sigma_mack" => (Average::Volume, SigmaInterpolation::Mack),
        other => panic!("unknown method {other}"),
    };
    Development {
        average,
        sigma_interpolation,
    }
}

/// Fits each (dataset, method) once and answers reference cases from it.
#[derive(Default)]
struct Fits {
    mack: HashMap<(String, String), MackFit>,
}

impl Fits {
    fn mack(&mut self, dataset: &str, method: &str) -> &MackFit {
        self.mack
            .entry((dataset.to_string(), method.to_string()))
            .or_insert_with(|| {
                Mack {
                    development: development(method),
                }
                .fit(&triangle(dataset), "values")
                .unwrap_or_else(|e| panic!("{dataset} {method}: {e}"))
            })
    }

    fn eval(&mut self, case: &Case) -> Option<f64> {
        let (dataset, method, quantity) = (
            case.get("dataset"),
            case.get("method"),
            case.get("quantity"),
        );
        let fit = self.mack(dataset, method);
        let cl = &fit.chain_ladder;
        let age = || case.number("arg").map(|k| k as usize);
        let origin = || {
            let year = case.number("arg")? as i32;
            cl.origins.iter().position(|&p| p == Period::year(year))
        };
        match quantity {
            "ata_factor" => cl.development.ldf.get(age()?).copied(),
            "cdf" => cl.cdf.get(age()?).copied(),
            "sigma" => cl.development.sigma.get(age()?).copied(),
            "f_se" => cl.development.std_err.get(age()?).copied(),
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
fn matches_r_chainladder() {
    let mut fits = Fits::default();
    check(&reference("reserving_chainladder_r.csv"), |c| fits.eval(c));
}

#[test]
fn matches_chainladder_python() {
    let mut fits = Fits::default();
    check(&reference("reserving_chainladder_python.csv"), |c| {
        fits.eval(c)
    });
}

#[test]
fn chain_ladder_alone_matches_mack_projection() {
    // The deterministic method gives the same projection Mack builds on.
    for dataset in ["raa", "genins", "abc"] {
        let tri = triangle(dataset);
        let cl: ChainLadderFit = ChainLadder::default().fit(&tri, "values").unwrap();
        let mack = Mack::default().fit(&tri, "values").unwrap();
        assert_eq!(cl, mack.chain_ladder, "{dataset}");
    }
}

#[test]
fn dataset_shapes() {
    for (name, n, first, latest_total) in [
        ("raa", 10, 1981, 160_987.0),
        ("genins", 10, 2001, 34_358_090.0),
        ("abc", 11, 1977, 10_221_194.0),
    ] {
        let tri = triangle(name);
        assert_eq!(tri.shape(), [1, 1, n, n], "{name}");
        assert_eq!(tri.origins()[0], Period::year(first), "{name}");
        let latest: f64 = tri.latest_diagonal().values(0, 0).iter().sum();
        assert_eq!(latest, latest_total, "{name}");
    }
}
