//! `Compartmental` fitted to paid and incurred (`Fitted::PaidAndIncurred`,
//! `docs/design/reserving-compartmental.md`, decision 12) on the papers'
//! company 337 workers' compensation (`data/clrd_wkcomp_337.csv`), trained
//! as in `reserving_compartmental.rs` (calendar years up to 1996, direct
//! premium; 90 observations over 9 accident years, and 1997 with premium
//! only), with the incurred column as the second response.
//!
//! * Stan parity: `reference/reserving_compartmental_incurred_stan.csv`,
//!   from `scripts/reserving_compartmental_incurred_stan.py` (`scripts/stan/
//!   compartmental_gm1_incurred.stan`: model 1 with the mean of an incurred
//!   cell RLR (OS + RRF PD) times premium, CmdStan 2.39.0, 4 chains of
//!   25,000 draws), rows `stan_gm1_incurred`. Our side runs the defaults
//!   with the paper's preset (4 chains of 1,000 warm-up and 2,000 kept
//!   draws, target acceptance 0.99, 10,000 simulations, seed 0).
//! * Convergence and the loo split by type.

use std::sync::OnceLock;
use std::time::Instant;

use prospicio_prob::Distribution;
use prospicio_reserving::bayes::{
    Compartmental, CompartmentalFit, CompartmentalPriors, CompartmentalStart, Fitted, Outstanding,
    RateOrder,
};
use prospicio_reserving::{DevelopmentColumn, Grain, Lag, Long, Month, Period, Triangle};
use prospicio_validation::{Case, check, reference};

/// A row of `data/clrd_wkcomp_337.csv`.
#[derive(Debug, Clone, Copy)]
struct Row {
    origin: i32,
    age: Lag,
    paid: f64,
    incurred: f64,
    case_incurred: f64,
    premium: f64,
    premium_net: f64,
}

fn rows() -> Vec<Row> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/clrd_wkcomp_337.csv");
    let text = std::fs::read_to_string(path).expect("data/clrd_wkcomp_337.csv");
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    assert_eq!(
        lines.next(),
        Some("origin,development,paid,incurred,case_incurred,premium,premium_net")
    );
    lines
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let n = |k: usize| f[k].parse::<f64>().unwrap();
            Row {
                origin: f[0].parse().unwrap(),
                age: f[1].parse().unwrap(),
                paid: n(2),
                incurred: n(3),
                case_incurred: n(4),
                premium: n(5),
                premium_net: n(6),
            }
        })
        .collect()
}

/// The paper's training triangle: the cells at calendar years up to 1996,
/// columns `paid`, `incurred` and `premium` (direct), and accident year 1997
/// with its premium only (at age 12, every other column missing).
fn training() -> &'static Triangle {
    static TRIANGLE: OnceLock<Triangle> = OnceLock::new();
    TRIANGLE.get_or_init(|| {
        let mut rows: Vec<Row> = rows()
            .into_iter()
            .filter(|r| r.origin + (r.age / 12) as i32 - 1 <= 1996)
            .collect();
        assert_eq!(rows.len(), 45);
        let premium_1997 = rows_1997_premium();
        rows.push(Row {
            origin: 1997,
            age: 12,
            paid: f64::NAN,
            incurred: f64::NAN,
            case_incurred: f64::NAN,
            premium: premium_1997,
            premium_net: f64::NAN,
        });
        let origin: Vec<Month> = rows.iter().map(|r| Month::january(r.origin)).collect();
        let ages: Vec<Lag> = rows.iter().map(|r| r.age).collect();
        let paid: Vec<f64> = rows.iter().map(|r| r.paid).collect();
        let incurred: Vec<f64> = rows.iter().map(|r| r.incurred).collect();
        let premium: Vec<f64> = rows.iter().map(|r| r.premium).collect();
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

fn rows_1997_premium() -> f64 {
    rows()
        .iter()
        .find(|r| r.origin == 1997)
        .map(|r| r.premium)
        .unwrap()
}

fn model(seed: u64) -> Compartmental {
    Compartmental {
        seed,
        fitted: Fitted::PaidAndIncurred,
        ..Compartmental::new(CompartmentalPriors::gesmann_morris())
    }
}

fn run(model: &Compartmental) -> CompartmentalFit {
    model
        .fit(
            training(),
            "paid",
            Outstanding::FromIncurred("incurred"),
            "premium",
        )
        .expect("fits")
}

fn report(label: &str, fit: &CompartmentalFit, elapsed: std::time::Duration) {
    let total = fit.reserves.total();
    let mean = |name: &str| {
        let j = fit
            .posterior
            .names()
            .iter()
            .position(|n| n == name)
            .unwrap();
        let d = fit.posterior.parameter(j).concat();
        d.iter().sum::<f64>() / d.len() as f64
    };
    eprintln!(
        "{label}: {elapsed:.2?}; total reserve mean {:.0} sd {:.0}; b_ker {:.2} b_kp {:.2} \
         b_RLR {:.2} b_RRF {:.2} tau {:.2} {:.2} rho {:.2} log sigma {:.2} {:.2}; \
         diagnostics {:?}",
        total.mean(),
        total.std_dev(),
        mean("b_ker"),
        mean("b_kp"),
        mean("b_RLR"),
        mean("b_RRF"),
        mean("tau_RLR"),
        mean("tau_RRF"),
        mean("rho"),
        mean("log_sigma_os"),
        mean("log_sigma_pd"),
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

/// The posterior mean and sd of a named parameter.
fn moments(fit: &CompartmentalFit, name: &str) -> Option<(f64, f64)> {
    let j = fit.posterior.names().iter().position(|n| n == name)?;
    let draws = fit.posterior.parameter(j).concat();
    let n = draws.len() as f64;
    let mean = draws.iter().sum::<f64>() / n;
    let var = draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Some((mean, var.sqrt()))
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
        "test_mean" => {
            // `incurred_1990` or `paid_1990`: the cell on the 1997 diagonal.
            let (kind, year) = arg.split_once('_')?;
            let year: i32 = year.parse().ok()?;
            test_mean(fit, kind == "paid", year)?
        }
        "elpd_loo" => fit.loo().ok()?.loo.estimate.elpd,
        _ => return None,
    })
}

/// The posterior mean of the expected incurred or paid of `year`'s cell
/// on the 1997 diagonal (the reference's `test_mean`, which has no process
/// noise): over the posterior draws of the rates and of `year`'s `RLR` and
/// `RRF`, or, for 1997 with no training cell, of the population's mean
/// `E[RLR]` and `E[RLR RRF]` given the draw (log-normal moments of its
/// effect).
fn test_mean(fit: &CompartmentalFit, paid: bool, year: i32) -> Option<f64> {
    let names = fit.posterior.names();
    let at = |name: &str| names.iter().position(|n| n == name);
    let i = fit.origins.iter().position(|&o| o == Period::year(year))?;
    let t = f64::from(1997 - year + 1);
    let (k_er, k_p) = (at("k_er")?, at("k_p")?);
    let own = (at(&format!("RLR[{year}]")), at(&format!("RRF[{year}]")));
    let (b_rlr, b_rrf, tau1, tau2, rho) = (
        at("b_RLR")?,
        at("b_RRF")?,
        at("tau_RLR")?,
        at("tau_RRF")?,
        at("rho")?,
    );
    let mut sum = 0.0;
    for d in fit.posterior.values().chunks_exact(fit.posterior.dim()) {
        let (a, b) = (d[k_er], d[k_p]);
        let os = a / (a - b) * ((-b * t).exp() - (-a * t).exp());
        let pd = (a * (1.0 - (-b * t).exp()) - b * (1.0 - (-a * t).exp())) / (a - b);
        let (rlr, rlr_rrf) = match own {
            (Some(r), Some(f)) => (d[r], d[r] * d[f]),
            _ => {
                // RLR = 0.7 e^(0.2 (b + u1)), RRF = 0.8 e^(0.1 (b + u2)).
                let (s1, s2) = (0.2 * d[tau1], 0.1 * d[tau2]);
                let rlr = 0.7 * (0.2 * d[b_rlr] + 0.5 * s1 * s1).exp();
                let v = s1 * s1 + s2 * s2 + 2.0 * d[rho] * s1 * s2;
                (
                    rlr,
                    0.56 * (0.2 * d[b_rlr] + 0.1 * d[b_rrf] + 0.5 * v).exp(),
                )
            }
        };
        sum += fit.premium[i]
            * if paid {
                rlr_rrf * pd
            } else {
                rlr * os + rlr_rrf * pd
            };
    }
    Some(sum / fit.posterior.n_draws() as f64)
}

fn cases(prefix: &str) -> Vec<Case> {
    reference("reserving_compartmental_incurred_stan.csv")
        .into_iter()
        .filter(|c| c.get("method").starts_with(prefix))
        .collect()
}

#[test]
fn the_data_and_the_training_triangle() {
    let rows = rows();
    assert_eq!(rows.len(), 100);
    // The papers' 1988 cell at age 1: premium 104, outstanding 53, paid 10
    // (USD millions).
    let first = rows[0];
    assert_eq!(
        (first.origin, first.age, first.paid, first.incurred),
        (1988, 12, 9558.0, 62679.0)
    );
    assert_eq!(
        (first.premium, first.premium_net, first.case_incurred),
        (104_437.0, 99_779.0, 38_060.0)
    );
    assert!(rows.iter().all(|r| r.incurred >= r.paid && r.paid > 0.0));
    assert_eq!(training().shape(), [1, 3, 10, 9]);
}

#[test]
fn stan_parity() {
    check(&cases("stan_gm1_incurred"), |case| value(fit(), case));
}

#[test]
fn defaults_converge_from_the_data_start() {
    let fit = fit();
    let d = &fit.diagnostics;
    assert!(d.warnings.is_empty(), "{d:?}");
    assert!(d.max_rhat < 1.01 && d.divergences == 0, "{d:?}");
    assert_eq!(fit.rate_swaps, 0);
    assert_eq!(fit.population_origins, [Period::year(1997)]);
}

#[test]
fn defaults_converge_from_the_prior_centre() {
    let start = Instant::now();
    let fit = run(&Compartmental {
        start: CompartmentalStart::PriorCentre,
        ..model(0)
    });
    report("prior-centre start", &fit, start.elapsed());
    let d = &fit.diagnostics;
    assert!(d.warnings.is_empty(), "{d:?}");
    assert!(d.max_rhat < 1.01 && d.divergences == 0, "{d:?}");
    // The same posterior as from the data start, within the Stan rows'
    // tolerances.
    let means: Vec<Case> = cases("stan_gm1_incurred")
        .into_iter()
        .filter(|c| c.get("quantity") == "mean")
        .collect();
    check(&means, |case| value(&fit, case));
}

#[test]
fn the_ordered_rates_give_the_same_posterior() {
    let start = Instant::now();
    let fit = run(&Compartmental {
        rate_order: RateOrder::Ordered,
        ..model(0)
    });
    report("ordered rates", &fit, start.elapsed());
    assert!(fit.diagnostics.warnings.is_empty(), "{:?}", fit.diagnostics);
    let means: Vec<Case> = cases("stan_gm1_incurred")
        .into_iter()
        .filter(|c| c.get("quantity") == "mean")
        .collect();
    check(&means, |case| value(&fit, case));
}

#[test]
fn loo_splits_by_type() {
    // The second type is the incurred cells, listed first.
    let loo = fit().loo().unwrap();
    assert_eq!(loo.loo.pareto_k.len(), 90);
    assert_eq!(loo.n_outstanding, 45);
    let sum = loo.elpd_outstanding + loo.elpd_paid;
    assert!((sum - loo.loo.estimate.elpd).abs() < 1e-9 * sum.abs());
    eprintln!(
        "elpd_loo {:.2} (se {:.2}; incurred {:.2}, paid {:.2}), p_loo {:.2}, max k {:.3}",
        loo.loo.estimate.elpd,
        loo.loo.estimate.se,
        loo.elpd_outstanding,
        loo.elpd_paid,
        loo.loo.estimate.p,
        loo.loo
            .pareto_k
            .iter()
            .fold(f64::NEG_INFINITY, |m, &k| m.max(k))
    );
}

/// The seed sweep that sizes the Stan-parity tolerances (decision 7): for
/// seeds `0..COMPARTMENTAL_SEEDS` (default 20), every reference row's value,
/// written to `target/compartmental_incurred_sweep.csv`, with each fit's diagnostics
/// and time. Run in a release build:
/// `cargo test --release -p prospicio-validation --features slow-bayes --test
/// reserving_compartmental_incurred seed_sweep -- --ignored --nocapture`.
#[test]
#[ignore]
fn seed_sweep() {
    let seeds: u64 = std::env::var("COMPARTMENTAL_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let cases = reference("reserving_compartmental_incurred_stan.csv");
    let mut out = String::from("seed,method,quantity,arg,got,passes\n");
    let mut diagnostics = String::from(
        "seed,seconds,max_rhat,min_ess_bulk,min_ess_tail,divergences,rate_swaps,warnings\n",
    );
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
            "{seed},{:.3},{},{},{},{},{},{}\n",
            elapsed.as_secs_f64(),
            d.max_rhat,
            d.min_ess_bulk,
            d.min_ess_tail,
            d.divergences,
            fit.rate_swaps,
            d.warnings.len()
        ));
        for case in &cases {
            let got = value(&fit, case).unwrap_or(f64::NAN);
            let want = case.number("expected").unwrap();
            let err = (got - want).abs();
            let passes = got == want
                || err <= case.number("abs_tol").unwrap_or(0.0)
                || err <= case.number("rel_tol").unwrap_or(0.0) * want.abs();
            out.push_str(&format!(
                "{seed},{},{},{},{got:e},{passes}\n",
                case.get("method"),
                case.get("quantity"),
                case.get("arg")
            ));
        }
    }
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../target");
    std::fs::write(format!("{dir}/compartmental_incurred_sweep.csv"), out).unwrap();
    std::fs::write(
        format!("{dir}/compartmental_incurred_sweep_diagnostics.csv"),
        diagnostics,
    )
    .unwrap();
    eprintln!("wrote {dir}/compartmental_incurred_sweep.csv");
}
