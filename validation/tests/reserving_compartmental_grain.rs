//! `Compartmental` on a triangle whose origin and development grains differ
//! (`docs/design/reserving-compartmental.md`, decision 13, PR I): annual
//! accident years 2011 to 2020 with quarterly development
//! (`data/compartmental_sim_quarterly.csv`, 220 outstanding and 220 paid
//! cells), simulated from Gesmann and Morris' model 1 with known parameters
//! by `scripts/compartmental_grain_simulate.py`. No public dataset has
//! unequal grain.
//!
//! * Stan parity: `reference/reserving_compartmental_grain_stan.csv`, from
//!   `scripts/reserving_compartmental_grain_stan.py`
//!   (`scripts/stan/compartmental_gm1.stan` unchanged, each cell's time its
//!   age in months over 12; CmdStan 2.39.0, 4 chains of 25,000 draws), rows
//!   `stan_gm1_grain`. Our side runs the defaults with the paper's preset
//!   (4 chains of 1,000 warm-up and 2,000 kept draws, target acceptance
//!   0.99, 10,000 simulations, seed 0).
//! * Recovery: the true parameters of the simulation lie in the posterior.
//! * The square: 40 quarterly ages, so reserves run to 120 months.

use std::sync::OnceLock;
use std::time::Instant;

use prospicio_prob::{Distribution, KeyValue};
use prospicio_reserving::bayes::{
    Compartmental, CompartmentalFit, CompartmentalPriors, Diagnostics, Outstanding,
};
use prospicio_reserving::{DevelopmentColumn, Grain, Lag, Long, Month, Period, Triangle};
use prospicio_validation::{Case, check, reference};

/// The simulated triangle: columns `paid`, `outstanding` and `premium`,
/// annual origins and quarterly development.
fn triangle() -> &'static Triangle {
    static TRIANGLE: OnceLock<Triangle> = OnceLock::new();
    TRIANGLE.get_or_init(|| {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/compartmental_sim_quarterly.csv"
        );
        let text = std::fs::read_to_string(path).expect("data/compartmental_sim_quarterly.csv");
        let mut lines = text.lines().filter(|l| !l.starts_with('#'));
        assert_eq!(
            lines.next(),
            Some("origin,development,paid,outstanding,premium")
        );
        let (mut origin, mut ages) = (vec![], vec![]);
        let (mut paid, mut os, mut premium) = (vec![], vec![], vec![]);
        for l in lines {
            let f: Vec<&str> = l.split(',').collect();
            origin.push(Month::january(f[0].parse().unwrap()));
            ages.push(f[1].parse::<Lag>().unwrap());
            paid.push(f[2].parse::<f64>().unwrap());
            os.push(f[3].parse::<f64>().unwrap());
            premium.push(f[4].parse::<f64>().unwrap());
        }
        assert_eq!(paid.len(), 220);
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &paid), ("outstanding", &os), ("premium", &premium)],
            origin_grain: Grain::Year,
            development_grain: Grain::Quarter,
            cumulative: true,
        })
        .unwrap()
    })
}

fn model(seed: u64) -> Compartmental {
    Compartmental {
        seed,
        ..Compartmental::new(CompartmentalPriors::gesmann_morris())
    }
}

fn run(model: &Compartmental) -> CompartmentalFit {
    model
        .fit(
            triangle(),
            "paid",
            Outstanding::Column("outstanding"),
            "premium",
        )
        .expect("fits")
}

/// The posterior mean and sd of a named parameter.
fn moments(fit: &CompartmentalFit, name: &str) -> Option<(f64, f64)> {
    let draws = parameter(fit, name)?;
    let n = draws.len() as f64;
    let mean = draws.iter().sum::<f64>() / n;
    let var = draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Some((mean, var.sqrt()))
}

fn parameter(fit: &CompartmentalFit, name: &str) -> Option<Vec<f64>> {
    let j = fit.posterior.names().iter().position(|n| n == name)?;
    Some(fit.posterior.parameter(j).concat())
}

fn report(label: &str, fit: &CompartmentalFit, elapsed: std::time::Duration) {
    let total = fit.reserves.total();
    let mean = |name: &str| moments(fit, name).map_or(f64::NAN, |m| m.0);
    eprintln!(
        "{label}: {elapsed:.2?}; total reserve mean {:.0} sd {:.0}; k_er {:.4} k_p {:.4} \
         RLR {:.4} RRF {:.4} tau {:.3} {:.3} rho {:.3} sigma {:.0} {:.0}; diagnostics {:?}",
        total.mean(),
        total.std_dev(),
        mean("k_er"),
        mean("k_p"),
        mean("RLR"),
        mean("RRF"),
        mean("tau_RLR"),
        mean("tau_RRF"),
        mean("rho"),
        mean("sigma_os"),
        mean("sigma_pd"),
        fit.diagnostics
    );
}

/// The default fit (seed 0), once.
fn fit() -> &'static CompartmentalFit {
    static FIT: OnceLock<CompartmentalFit> = OnceLock::new();
    FIT.get_or_init(|| {
        let start = Instant::now();
        let fit = run(&model(0));
        report("default fit", &fit, start.elapsed());
        fit
    })
}

/// Our value of a reference row, or `None` for an unknown quantity.
fn value(fit: &CompartmentalFit, case: &Case) -> Option<f64> {
    let arg = case.get("arg");
    let origin = |year: &str| -> Option<usize> {
        let p = Period::year(year.parse().ok()?);
        fit.origins.iter().position(|&o| o == p)
    };
    let reserve = || {
        if arg == "total" {
            Some(fit.reserves.total().clone())
        } else {
            let p = Period::year(arg.parse().ok()?);
            fit.reserves.marginal(&vec![p.into()])
        }
    };
    Some(match case.get("quantity") {
        "mean" => moments(fit, arg)?.0,
        "sd" => moments(fit, arg)?.1,
        "ulr_mean" => fit.ultimate_loss_ratio[origin(arg)?],
        "reserve_mean" => reserve()?.mean(),
        "reserve_sd" => reserve()?.std_dev(),
        "reserve_q05" => reserve()?.quantile(0.05).ok()?,
        "reserve_q50" => reserve()?.quantile(0.5).ok()?,
        "reserve_q95" => reserve()?.quantile(0.95).ok()?,
        "elpd_loo" => fit.loo().ok()?.loo.estimate.elpd,
        _ => return None,
    })
}

fn cases() -> Vec<Case> {
    let cases: Vec<Case> = reference("reserving_compartmental_grain_stan.csv")
        .into_iter()
        .filter(|c| c.get("method") == "stan_gm1_grain")
        .collect();
    assert!(!cases.is_empty(), "no reference rows");
    cases
}

fn converged(d: &Diagnostics) {
    assert!(
        d.max_rhat < 1.01 && d.min_ess_bulk > 400.0 && d.min_ess_tail > 400.0,
        "{d:?}"
    );
    assert!(d.divergences <= 2, "{d:?}");
}

#[test]
fn the_simulated_triangle() {
    let tri = triangle();
    assert_eq!(tri.shape(), [1, 3, 10, 40]);
    assert_eq!(tri.origin_grain(), Grain::Year);
    assert_eq!(tri.development_grain(), Grain::Quarter);
}

#[test]
fn stan_parity() {
    check(&cases(), |case| value(fit(), case));
}

#[test]
fn defaults_converge() {
    let fit = fit();
    converged(&fit.diagnostics);
    assert_eq!(fit.rate_swaps, 0);
    assert!(fit.population_origins.is_empty());
}

/// The square has 40 quarterly ages (ten origin years), so each origin's
/// future paid runs to 120 months and 2011, observed there, has no
/// reserve.
#[test]
fn the_square_runs_to_ten_years() {
    let fit = fit();
    let ages = |d: &prospicio_prob::PredictiveDistribution| -> Vec<i64> {
        d.components()
            .iter()
            .map(|k| match k[1] {
                KeyValue::Int(a) => a,
                _ => unreachable!(),
            })
            .collect()
    };
    let paid = ages(&fit.paid);
    assert_eq!(paid.iter().max(), Some(&120));
    assert_eq!(paid.iter().min(), Some(&15));
    // 2012 has 4 future quarters, ..., 2020 has 36.
    assert_eq!(paid.len(), (1..=9).map(|k| 4 * k).sum::<usize>());
    let r2011 = fit
        .reserves
        .marginal(&vec![Period::year(2011).into()])
        .unwrap();
    assert_eq!(r2011.mean(), 0.0);
}

/// The truth of `data/compartmental_sim_quarterly.csv` (its header).
const TRUTH: [(&str, f64); 11] = [
    ("b_ker", 1.0),
    ("b_kp", -2.0),
    ("b_RLR", 1.0),
    ("b_RRF", -1.0),
    ("tau_RLR", 0.4),
    ("tau_RRF", 0.2),
    ("rho", 0.3),
    ("sigma_os", 1500.0),
    ("sigma_pd", 500.0),
    ("k_er", 3.315512754226943),
    ("k_p", 0.8187307530779818),
];

/// Each origin's true `RLR` and `RRF`.
const ORIGIN_TRUTH: [(i32, f64, f64); 10] = [
    (2011, 0.8398454883101578, 0.7224581189276418),
    (2012, 0.7731604282657731, 0.7054233382944525),
    (2013, 0.8320381918945053, 0.7226129061510849),
    (2014, 0.918013981968971, 0.735901862116191),
    (2015, 0.8289775471762064, 0.7105395484384287),
    (2016, 0.8201495274185593, 0.7326573950190186),
    (2017, 0.7825519437364227, 0.7385919512830426),
    (2018, 0.92017035682639, 0.7279949275395228),
    (2019, 0.9030360966215708, 0.7163839672924008),
    (2020, 0.8302747793318699, 0.722000749671166),
];

/// The share of draws of `name` below `truth`: the truth's posterior
/// percentile.
fn percentile(fit: &CompartmentalFit, name: &str, truth: f64) -> f64 {
    let draws = parameter(fit, name).unwrap_or_else(|| panic!("no parameter {name}"));
    draws.iter().filter(|&&v| v < truth).count() as f64 / draws.len() as f64
}

/// Posterior recovery: every true parameter of the simulation, the
/// population's and each origin's `RLR` and `RRF`, lies within the
/// central 99% posterior interval of the default fit (the percentiles are
/// recorded in `knowledge/findings/compartmental-grain.md`).
#[test]
fn the_posterior_recovers_the_truth() {
    let fit = fit();
    let mut all = Vec::new();
    for (name, truth) in TRUTH {
        all.push((name.to_string(), percentile(fit, name, truth)));
    }
    for (o, rlr, rrf) in ORIGIN_TRUTH {
        all.push((
            format!("RLR[{o}]"),
            percentile(fit, &format!("RLR[{o}]"), rlr),
        ));
        all.push((
            format!("RRF[{o}]"),
            percentile(fit, &format!("RRF[{o}]"), rrf),
        ));
    }
    let inside_95 = all
        .iter()
        .filter(|(_, p)| (0.025..=0.975).contains(p))
        .count();
    for (name, p) in &all {
        eprintln!("{name}: truth at the posterior's {:.1}%", 100.0 * p);
    }
    eprintln!("{inside_95} of {} inside the central 95%", all.len());
    for (name, p) in &all {
        assert!((0.005..=0.995).contains(p), "{name}: truth at {p}");
    }
}

/// The seed sweep that checks the Stan-parity tolerances on this triangle:
/// for seeds `0..COMPARTMENTAL_SEEDS` (default 20), every reference row's
/// value, written to `target/compartmental_grain_sweep.csv`, with each
/// fit's diagnostics. Run in a release build:
/// `cargo test --release -p prospicio-validation --features slow-bayes --test
/// reserving_compartmental_grain seed_sweep -- --ignored --nocapture`.
#[test]
#[ignore]
fn seed_sweep() {
    let seeds: u64 = std::env::var("COMPARTMENTAL_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let cases = cases();
    let mut out = String::from("seed,quantity,arg,got,expected,abs_tol,rel_tol,passes\n");
    let mut diagnostics =
        String::from("seed,seconds,max_rhat,min_ess_bulk,min_ess_tail,divergences,warnings\n");
    for seed in 0..seeds {
        let start = Instant::now();
        let fit = run(&model(seed));
        let elapsed = start.elapsed();
        let d = &fit.diagnostics;
        eprintln!(
            "seed {seed}: {elapsed:.2?}, max R-hat {:.4}, min ESS {:.0}/{:.0}, {} divergences, \
             {} warnings {:?}",
            d.max_rhat,
            d.min_ess_bulk,
            d.min_ess_tail,
            d.divergences,
            d.warnings.len(),
            d.warnings
        );
        diagnostics.push_str(&format!(
            "{seed},{:.3},{},{},{},{},{}\n",
            elapsed.as_secs_f64(),
            d.max_rhat,
            d.min_ess_bulk,
            d.min_ess_tail,
            d.divergences,
            d.warnings.len()
        ));
        for case in &cases {
            let got = value(&fit, case).unwrap_or(f64::NAN);
            let want = case.number("expected").unwrap();
            let (abs_tol, rel_tol) = (
                case.number("abs_tol").unwrap_or(0.0),
                case.number("rel_tol").unwrap_or(0.0),
            );
            let err = (got - want).abs();
            let passes = got == want || err <= abs_tol || err <= rel_tol * want.abs();
            out.push_str(&format!(
                "{seed},{},{},{got:e},{want:e},{abs_tol:e},{rel_tol:e},{passes}\n",
                case.get("quantity"),
                case.get("arg")
            ));
        }
    }
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../target");
    std::fs::write(format!("{dir}/compartmental_grain_sweep.csv"), out).unwrap();
    std::fs::write(
        format!("{dir}/compartmental_grain_sweep_diagnostics.csv"),
        diagnostics,
    )
    .unwrap();
    eprintln!("wrote {dir}/compartmental_grain_sweep.csv");
}
