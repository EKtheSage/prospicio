//! Meyers' correlated accident year model (`Meyers` with `MeyersModel::Cay`,
//! `docs/design/reserving-bayes.md`, decisions 7 and 10) on the incurred
//! data of his illustrative insurer, group 353 Commercial Auto
//! (`data/clrd_comauto_353.csv`, the cells at calendar years up to 1997),
//! against his own Stan code and his printed numbers.
//!
//! * Stan parity: `reference/reserving_meyers_cay_stan.csv`, from
//!   `scripts/reserving_meyers_cay_stan.py` (Meyers' `CAY.R` program
//!   transcribed into current syntax, CmdStan 2.39.0, 4 chains of 25,000
//!   draws, dense metric), rows `stan_cay`. Our side runs the defaults (4
//!   chains of 1,000 warm-up and 2,000 kept draws, target acceptance 0.95,
//!   10,000 simulations, seed 0). Each tolerance is at least 4 times our
//!   Monte Carlo error, measured per row as the seed-to-seed sd over seeds 0
//!   to 199 (the `seed_sweep` test below, ignored in CI), and every row
//!   passes at each of those seeds; the measured spreads are in the script
//!   and the design note. The 1997 reserve's sd has no row: its error is
//!   heavy-tailed, so its percentiles check its spread. The joint
//!   prediction is checked by the sum of the 1988 to 1996 reserves (arg
//!   `1988-1996`) and by the correlation of neighbouring origins' reserves
//!   (args `a/b`); the total's sd is heavy-tailed for the same reason as
//!   1997's, so its tolerance is loose. On these data the recursion at the
//!   oldest age moves those correlations by at most 0.03 (`sigma[10]` is
//!   near 0.013), under 2 Monte Carlo errors, so a unit test in `meyers.rs`
//!   checks the recursion itself.
//! * Meyers' printed numbers (rows `meyers_2019_cay`: the total ultimate and
//!   its sd, `logelr`, `rho` and elpd_loo, from Tables 8.1 to 8.3): a looser
//!   second check.

use std::sync::OnceLock;
use std::time::Instant;

use prospicio_prob::{Distribution, Sampled};
use prospicio_reserving::bayes::{Meyers, MeyersFit, MeyersModel};
use prospicio_reserving::{DevelopmentColumn, Grain, Lag, Long, Month, Period, Triangle};
use prospicio_validation::{Case, check, reference};

/// The incurred training triangle: the cells at calendar years up to 1997
/// of `data/clrd_comauto_353.csv`, columns `incurred` and `premium`.
fn training() -> &'static Triangle {
    static TRIANGLE: OnceLock<Triangle> = OnceLock::new();
    TRIANGLE.get_or_init(|| {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/clrd_comauto_353.csv");
        let text = std::fs::read_to_string(path).expect("data/clrd_comauto_353.csv");
        let mut lines = text.lines().filter(|l| !l.starts_with('#'));
        assert_eq!(
            lines.next(),
            Some("origin,development,paid,incurred,premium")
        );
        let (mut origin, mut ages, mut incurred, mut premium) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for line in lines {
            let f: Vec<&str> = line.split(',').collect();
            let (year, age): (i32, Lag) = (f[0].parse().unwrap(), f[1].parse().unwrap());
            if year + (age / 12) as i32 - 1 > 1997 {
                continue;
            }
            origin.push(Month::january(year));
            ages.push(age);
            incurred.push(f[3].parse::<f64>().unwrap());
            premium.push(f[4].parse::<f64>().unwrap());
        }
        assert_eq!(origin.len(), 55);
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("incurred", &incurred), ("premium", &premium)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    })
}

fn run(seed: u64) -> MeyersFit {
    Meyers {
        model: MeyersModel::Cay,
        seed,
        ..Meyers::default()
    }
    .fit(training(), "incurred", "premium")
    .expect("fits")
}

/// The default fit (seed 0), once.
fn fit() -> &'static MeyersFit {
    static FIT: OnceLock<MeyersFit> = OnceLock::new();
    FIT.get_or_init(|| {
        let start = Instant::now();
        let fit = run(0);
        let total = fit.ultimates.total();
        eprintln!(
            "cay incurred: Meyers default fit in {:.2?}; total ultimate mean {:.0} sd {:.0}; diagnostics {:?}",
            start.elapsed(),
            total.mean(),
            total.std_dev(),
            fit.diagnostics
        );
        fit
    })
}

/// Our value of a reference row, or `None` for an unknown quantity.
fn value(fit: &MeyersFit, case: &Case) -> Option<f64> {
    let arg = case.get("arg");
    // Each simulation's reserves by origin, from 1988.
    let rows = || fit.reserves.draw_matrix().chunks_exact(10);
    let reserve = || {
        if arg == "total" {
            fit.reserves.total().clone()
        } else if let Some((from, to)) = arg.split_once('-') {
            // The sum of the reserves of origins `from` to `to`.
            let (from, to): (usize, usize) = (
                from.parse::<usize>().unwrap() - 1988,
                to.parse::<usize>().unwrap() - 1988,
            );
            Sampled::new(rows().map(|r| r[from..=to].iter().sum()).collect()).unwrap()
        } else {
            let origin = Period::year(arg.parse().expect("origin year"));
            fit.reserves.marginal(&vec![origin.into()]).expect("origin")
        }
    };
    Some(match case.get("quantity") {
        "mean" | "sd" => {
            // The draws' moments directly: summary() also computes R-hat
            // and ESS, slow in a debug build.
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
        "reserve_corr" => {
            // The correlation of two origins' reserves, `a/b`.
            let (a, b) = arg.split_once('/')?;
            let (a, b) = (
                a.parse::<usize>().ok()? - 1988,
                b.parse::<usize>().ok()? - 1988,
            );
            let pairs: Vec<(f64, f64)> = rows().map(|r| (r[a], r[b])).collect();
            correlation(&pairs)
        }
        "total_ultimate_mean" => fit.ultimates.total().mean(),
        "total_ultimate_sd" => fit.ultimates.total().std_dev(),
        "elpd_loo" => fit.loo().ok()?.estimate.elpd,
        _ => return None,
    })
}

/// The sample correlation of pairs.
fn correlation(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len() as f64;
    let mx = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in pairs {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx).powi(2);
        syy += (y - my).powi(2);
    }
    sxy / (sxx * syy).sqrt()
}

fn cases(method: &str) -> Vec<Case> {
    reference("reserving_meyers_cay_stan.csv")
        .into_iter()
        .filter(|c| c.get("method") == method)
        .collect()
}

#[test]
fn stan_parity() {
    check(&cases("stan_cay"), |case| value(fit(), case));
}

#[test]
fn meyers_printed_numbers() {
    check(&cases("meyers_2019_cay"), |case| value(fit(), case));
}

#[test]
fn defaults_pass_the_diagnostics() {
    let d = &fit().diagnostics;
    assert!(d.warnings.is_empty(), "{d:?}");
}

/// The seed sweep that sizes the Stan-parity tolerances (decision 10): for
/// seeds `0..MEYERS_SEEDS` (default 20), every reference row's value,
/// written to `target/meyers_cay_sweep.csv`. Run in a release build:
/// `cargo test --release -p prospicio-validation --test reserving_meyers_cay
/// seed_sweep -- --ignored --nocapture`.
#[test]
#[ignore]
fn seed_sweep() {
    let seeds: u64 = std::env::var("MEYERS_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let cases = reference("reserving_meyers_cay_stan.csv");
    let mut out = String::from("seed,dataset,method,quantity,arg,got,passes\n");
    for seed in 0..seeds {
        let start = Instant::now();
        let fit = run(seed);
        let elapsed = start.elapsed();
        let d = &fit.diagnostics;
        eprintln!(
            "seed {seed} cay incurred: {elapsed:.2?}, max R-hat {:.4}, min ESS {:.0}/{:.0}, \
             {} divergences, {} warnings {:?}",
            d.max_rhat,
            d.min_ess_bulk,
            d.min_ess_tail,
            d.divergences,
            d.warnings.len(),
            d.warnings
        );
        for case in &cases {
            let got = value(&fit, case).unwrap_or(f64::NAN);
            let want = case.number("expected").unwrap();
            let err = (got - want).abs();
            let passes = got == want
                || err <= case.number("abs_tol").unwrap_or(0.0)
                || err <= case.number("rel_tol").unwrap_or(0.0) * want.abs();
            out.push_str(&format!(
                "{seed},{},{},{},{},{got:e},{passes}\n",
                case.get("dataset"),
                case.get("method"),
                case.get("quantity"),
                case.get("arg")
            ));
        }
    }
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../target/meyers_cay_sweep.csv"
    );
    std::fs::write(path, out).unwrap();
    eprintln!("wrote {path}");
}
