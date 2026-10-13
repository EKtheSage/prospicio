//! Meyers' CRC and CSR (`Meyers`, `docs/design/reserving-bayes.md`,
//! decisions 7 and 10) on his illustrative insurer, group 353 Commercial
//! Auto (`data/clrd_comauto_353.csv`, the cells at calendar years up to
//! 1997), against his own Stan code and his printed numbers.
//!
//! * Stan parity: `reference/reserving_meyers_stan.csv`, from
//!   `scripts/reserving_meyers_stan.py` (Meyers' programs transcribed into
//!   current syntax, CmdStan 2.39.0, 4 chains of 25,000 draws, dense
//!   metric), rows `stan_crc` and `stan_csr`, paid and incurred. Our side
//!   runs the defaults (4 chains of 1,000 warm-up and 2,000 kept draws,
//!   target acceptance 0.95, 10,000 simulations, seed 0). Each tolerance
//!   is at least 4 times our Monte Carlo error, measured per row as the
//!   seed-to-seed sd over seeds 0 to 199 (the `seed_sweep` test below,
//!   ignored in CI), and every row passes at each of those seeds; the
//!   measured spreads are in the script and the design note. The 1997
//!   reserve's sd has no row: its error is heavy-tailed, so its
//!   percentiles check its spread.
//! * Meyers' printed numbers (rows `meyers_2019_crc` and `meyers_2019_csr`:
//!   the total ultimate and its sd, `logelr`, `gamma` and elpd_loo, from
//!   Tables 5.1 to 7.3): a looser second check.
//! * The data: the age-10 sums of the full square are Meyers' outcomes,
//!   paid 40,000 and incurred 40,061.

use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::Instant;

use prospicio_prob::Distribution;
use prospicio_reserving::bayes::{Meyers, MeyersFit, MeyersModel};
use prospicio_reserving::{DevelopmentColumn, Grain, Lag, Long, Month, Period, Triangle};
use prospicio_validation::{Case, check, reference};

/// The rows of `data/clrd_comauto_353.csv`: origin, age in months, paid,
/// incurred, premium.
fn rows() -> Vec<(i32, Lag, f64, f64, f64)> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/clrd_comauto_353.csv");
    let text = std::fs::read_to_string(path).expect("data/clrd_comauto_353.csv");
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    assert_eq!(
        lines.next(),
        Some("origin,development,paid,incurred,premium")
    );
    lines
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (
                f[0].parse().unwrap(),
                f[1].parse().unwrap(),
                f[2].parse().unwrap(),
                f[3].parse().unwrap(),
                f[4].parse().unwrap(),
            )
        })
        .collect()
}

/// The training triangle: the cells at calendar years up to 1997, columns
/// `paid`, `incurred` and `premium`.
fn training() -> &'static Triangle {
    static TRIANGLE: OnceLock<Triangle> = OnceLock::new();
    TRIANGLE.get_or_init(|| {
        let rows: Vec<_> = rows()
            .into_iter()
            .filter(|r| r.0 + (r.1 / 12) as i32 - 1 <= 1997)
            .collect();
        assert_eq!(rows.len(), 55);
        let origin: Vec<Month> = rows.iter().map(|r| Month::january(r.0)).collect();
        let ages: Vec<Lag> = rows.iter().map(|r| r.1).collect();
        let column = |k: usize| -> Vec<f64> {
            rows.iter()
                .map(|r| [r.2, r.3, r.4][k])
                .collect::<Vec<f64>>()
        };
        let (paid, incurred, premium) = (column(0), column(1), column(2));
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[
                ("paid", &paid),
                ("incurred", &incurred),
                ("premium", &premium),
            ],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    })
}

fn model_of(name: &str) -> MeyersModel {
    match name {
        "crc" => MeyersModel::Crc,
        "csr" => MeyersModel::Csr,
        _ => panic!("no model {name}"),
    }
}

fn run(model: &str, column: &str, seed: u64) -> MeyersFit {
    Meyers {
        model: model_of(model),
        seed,
        ..Meyers::default()
    }
    .fit(training(), column, "premium")
    .expect("fits")
}

/// The default fit (seed 0) of each case, once.
fn fit(model: &str, column: &str) -> &'static MeyersFit {
    static FITS: OnceLock<BTreeMap<(String, String), MeyersFit>> = OnceLock::new();
    let fits = FITS.get_or_init(|| {
        let mut fits = BTreeMap::new();
        for m in ["crc", "csr"] {
            for c in ["paid", "incurred"] {
                let start = Instant::now();
                let fit = run(m, c, 0);
                let total = fit.ultimates.total();
                eprintln!(
                    "{m} {c}: Meyers default fit in {:.2?}; total ultimate mean {:.0} sd {:.0}; diagnostics {:?}",
                    start.elapsed(),
                    total.mean(),
                    total.std_dev(),
                    fit.diagnostics
                );
                fits.insert((m.to_string(), c.to_string()), fit);
            }
        }
        fits
    });
    &fits[&(model.to_string(), column.to_string())]
}

/// Our value of a reference row, or `None` for an unknown quantity.
fn value(fit: &MeyersFit, case: &Case) -> Option<f64> {
    let arg = case.get("arg");
    let reserve = || {
        if arg == "total" {
            fit.reserves.total().clone()
        } else {
            let origin = Period::year(arg.parse().expect("origin year"));
            fit.reserves.marginal(&vec![origin.into()]).expect("origin")
        }
    };
    Some(match case.get("quantity") {
        "mean" | "sd" => {
            // The draws' moments directly: summary() also computes R-hat
            // and ESS, slow in a debug build for every case.
            let j = fit.posterior.names().iter().position(|n| n == arg)?;
            let draws = fit.posterior.parameter(j).concat();
            let n = draws.len() as f64;
            let mean = draws.iter().sum::<f64>() / n;
            let var = draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
            if case.get("quantity") == "mean" {
                mean
            } else {
                var.sqrt()
            }
        }
        "reserve_mean" => reserve().mean(),
        "reserve_sd" => reserve().std_dev(),
        "reserve_q05" => reserve().quantile(0.05).ok()?,
        "reserve_q50" => reserve().quantile(0.5).ok()?,
        "reserve_q95" => reserve().quantile(0.95).ok()?,
        "total_ultimate_mean" => fit.ultimates.total().mean(),
        "total_ultimate_sd" => fit.ultimates.total().std_dev(),
        "elpd_loo" => fit.loo().ok()?.estimate.elpd,
        _ => return None,
    })
}

/// The model of a reference row's method, `stan_crc` or `meyers_2019_csr`.
fn method_model(case: &Case) -> &str {
    let method = case.get("method");
    &method[method.len() - 3..]
}

fn cases(prefix: &str) -> Vec<Case> {
    reference("reserving_meyers_stan.csv")
        .into_iter()
        .filter(|c| c.get("method").starts_with(prefix))
        .collect()
}

#[test]
fn outcomes_and_training_cells() {
    let rows = rows();
    assert_eq!(rows.len(), 100);
    let at_120 = |k: usize| -> f64 {
        rows.iter()
            .filter(|r| r.1 == 120)
            .map(|r| [r.2, r.3][k])
            .sum()
    };
    assert_eq!((at_120(0), at_120(1)), (40_000.0, 40_061.0));
    assert_eq!(training().shape(), [1, 3, 10, 10]);
}

#[test]
fn stan_parity() {
    check(&cases("stan_"), |case| {
        value(fit(method_model(case), case.get("dataset")), case)
    });
}

#[test]
fn meyers_printed_numbers() {
    check(&cases("meyers_2019_"), |case| {
        value(fit(method_model(case), case.get("dataset")), case)
    });
}

#[test]
fn defaults_pass_the_diagnostics() {
    for model in ["crc", "csr"] {
        for column in ["paid", "incurred"] {
            let d = &fit(model, column).diagnostics;
            assert!(d.warnings.is_empty(), "{model} {column}: {d:?}");
        }
    }
}

#[test]
fn loo_is_finite_with_a_k_per_cell() {
    let start = Instant::now();
    let loo = fit("csr", "paid").loo().unwrap();
    eprintln!(
        "CSR paid loo in {:.2?}: elpd {:.2} (se {:.2}), p_loo {:.2}, max k {:.3}",
        start.elapsed(),
        loo.estimate.elpd,
        loo.estimate.se,
        loo.estimate.p,
        loo.pareto_k
            .iter()
            .fold(f64::NEG_INFINITY, |m, &k| m.max(k))
    );
    assert_eq!(loo.pareto_k.len(), 55);
    assert!(loo.estimate.elpd.is_finite() && loo.estimate.se.is_finite());
}

/// The seed sweep that sizes the Stan-parity tolerances (decision 10): for
/// seeds `0..MEYERS_SEEDS` (default 20), every reference row's value,
/// written to `target/meyers_sweep.csv`. Run in a release build:
/// `cargo test --release -p prospicio-validation --test reserving_meyers
/// seed_sweep -- --ignored --nocapture`.
#[test]
#[ignore]
fn seed_sweep() {
    let seeds: u64 = std::env::var("MEYERS_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let cases = reference("reserving_meyers_stan.csv");
    let mut out = String::from("seed,dataset,method,quantity,arg,got,passes\n");
    for seed in 0..seeds {
        for model in ["crc", "csr"] {
            for column in ["paid", "incurred"] {
                let start = Instant::now();
                let fit = run(model, column, seed);
                let elapsed = start.elapsed();
                let d = &fit.diagnostics;
                eprintln!(
                    "seed {seed} {model} {column}: {elapsed:.2?}, max R-hat {:.4}, min ESS {:.0}/{:.0}, \
                     {} divergences, {} warnings {:?}",
                    d.max_rhat,
                    d.min_ess_bulk,
                    d.min_ess_tail,
                    d.divergences,
                    d.warnings.len(),
                    d.warnings
                );
                for case in cases
                    .iter()
                    .filter(|c| method_model(c) == model && c.get("dataset") == column)
                {
                    let got = value(&fit, case).unwrap_or(f64::NAN);
                    let want = case.number("expected").unwrap();
                    let err = (got - want).abs();
                    let passes = got == want
                        || err <= case.number("abs_tol").unwrap_or(0.0)
                        || err <= case.number("rel_tol").unwrap_or(0.0) * want.abs();
                    out.push_str(&format!(
                        "{seed},{column},{},{},{},{got:e},{passes}\n",
                        case.get("method"),
                        case.get("quantity"),
                        case.get("arg")
                    ));
                }
            }
        }
    }
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../target/meyers_sweep.csv");
    std::fs::write(path, out).unwrap();
    eprintln!("wrote {path}");
}
