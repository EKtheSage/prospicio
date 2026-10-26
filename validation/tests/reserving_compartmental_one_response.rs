//! `Compartmental` fitted to one response alone with ordered rates
//! (`Fitted::PaidOnly` and `Fitted::IncurredOnly`,
//! `docs/design/reserving-compartmental.md`, decision 12, PR H) on the
//! papers' company 337 workers' compensation (`data/clrd_wkcomp_337.csv`),
//! trained as in `reserving_compartmental.rs` (calendar years up to 1996,
//! direct premium; 45 cells of the one response over 9 accident years, and
//! 1997 with premium only). Under incurred alone the paid column gives each
//! origin's latest paid only.
//!
//! * Stan parity: `reference/reserving_compartmental_one_response_stan.csv`,
//!   from `scripts/reserving_compartmental_one_response_stan.py`
//!   (`scripts/stan/compartmental_gm1_paid_only.stan` and
//!   `compartmental_gm1_incurred_only.stan`: model 1 on the one response with
//!   `k_er = k_p + e^delta`, CmdStan 2.39.0, 4 chains of 25,000 draws), rows
//!   `stan_gm1_paid_only` and `stan_gm1_incurred_only`. Our side runs the
//!   defaults with the paper's preset and `RateOrder::Ordered` (4 chains of
//!   1,000 warm-up and 2,000 kept draws, target acceptance 0.99, 10,000
//!   simulations, seed 0).
//! * Convergence from both starts, and the loo of the one type.

use std::sync::OnceLock;
use std::time::Instant;

use prospicio_prob::Distribution;
use prospicio_reserving::bayes::{
    Compartmental, CompartmentalFit, CompartmentalPriors, CompartmentalStart, Diagnostics, Fitted,
    Outstanding, RateOrder,
};
use prospicio_reserving::{DevelopmentColumn, Grain, Lag, Long, Month, Period, Triangle};
use prospicio_validation::{Case, check, reference};

/// A row of `data/clrd_wkcomp_337.csv` (the columns this test reads).
#[derive(Debug, Clone, Copy)]
struct Row {
    origin: i32,
    age: Lag,
    paid: f64,
    incurred: f64,
    premium: f64,
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
                premium: n(5),
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
        let all = rows();
        let premium_1997 = all.iter().find(|r| r.origin == 1997).unwrap().premium;
        let mut rows: Vec<Row> = all
            .into_iter()
            .filter(|r| r.origin + (r.age / 12) as i32 - 1 <= 1996)
            .collect();
        assert_eq!(rows.len(), 45);
        rows.push(Row {
            origin: 1997,
            age: 12,
            paid: f64::NAN,
            incurred: f64::NAN,
            premium: premium_1997,
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

/// The two variants: the setting and the method of its reference rows.
const MODES: [(Fitted, &str); 2] = [
    (Fitted::PaidOnly, "stan_gm1_paid_only"),
    (Fitted::IncurredOnly, "stan_gm1_incurred_only"),
];

fn model(fitted: Fitted, seed: u64) -> Compartmental {
    Compartmental {
        seed,
        fitted,
        rate_order: RateOrder::Ordered,
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

/// The posterior mean and sd of a named parameter.
fn moments(fit: &CompartmentalFit, name: &str) -> Option<(f64, f64)> {
    let j = fit.posterior.names().iter().position(|n| n == name)?;
    let draws = fit.posterior.parameter(j).concat();
    let n = draws.len() as f64;
    let mean = draws.iter().sum::<f64>() / n;
    let var = draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Some((mean, var.sqrt()))
}

/// The posterior sd of `log k_p`.
fn log_kp_sd(fit: &CompartmentalFit) -> f64 {
    let j = fit
        .posterior
        .names()
        .iter()
        .position(|n| n == "k_p")
        .unwrap();
    let logs: Vec<f64> = fit
        .posterior
        .parameter(j)
        .concat()
        .iter()
        .map(|v| v.ln())
        .collect();
    let n = logs.len() as f64;
    let m = logs.iter().sum::<f64>() / n;
    (logs.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

fn report(label: &str, fit: &CompartmentalFit, elapsed: std::time::Duration) {
    eprintln!("sd(log k_p) {:.3}", log_kp_sd(fit));
    let total = fit.reserves.total();
    let mean = |name: &str| moments(fit, name).map_or(f64::NAN, |m| m.0);
    eprintln!(
        "{label}: {elapsed:.2?}; total reserve mean {:.0} sd {:.0}; k_er {:.3} k_p {:.3} \
         RLR {:.3} RRF {:.3} tau {:.2} {:.2} rho {:.2}; diagnostics {:?}",
        total.mean(),
        total.std_dev(),
        mean("k_er"),
        mean("k_p"),
        mean("RLR"),
        mean("RRF"),
        mean("tau_RLR"),
        mean("tau_RRF"),
        mean("rho"),
        fit.diagnostics
    );
}

/// The default fit (seed 0) of each variant, once.
fn fit(fitted: Fitted) -> &'static CompartmentalFit {
    static PAID: OnceLock<CompartmentalFit> = OnceLock::new();
    static INCURRED: OnceLock<CompartmentalFit> = OnceLock::new();
    let cell = if fitted == Fitted::PaidOnly {
        &PAID
    } else {
        &INCURRED
    };
    cell.get_or_init(|| {
        let start = Instant::now();
        let fit = run(&model(fitted, 0));
        report(&format!("{fitted:?} default fit"), &fit, start.elapsed());
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

fn cases(method: &str) -> Vec<Case> {
    let cases: Vec<Case> = reference("reserving_compartmental_one_response_stan.csv")
        .into_iter()
        .filter(|c| c.get("method") == method)
        .collect();
    assert!(!cases.is_empty(), "no reference rows for {method}");
    cases
}

#[test]
fn the_training_triangle() {
    assert_eq!(training().shape(), [1, 3, 10, 9]);
}

#[test]
fn stan_parity_paid_only() {
    let (fitted, method) = MODES[0];
    check(&cases(method), |case| value(fit(fitted), case));
}

#[test]
fn stan_parity_incurred_only() {
    let (fitted, method) = MODES[1];
    check(&cases(method), |case| value(fit(fitted), case));
}

/// The default fit's diagnostics. Under paid alone the data start is the
/// prior centre (its least squares is not identified, decision 12), and at
/// seed 0 that fit has one divergent transition: on the sweep 3 of 20
/// paid-only fits missed a threshold (two with one divergent transition,
/// one with an R-hat of 1.013 on tau_RRF, which the diffuse preset also
/// reached on CI's runner), so paid alone is allowed up to two divergences
/// and an R-hat below 1.02 with the effective sample sizes still above 400.
/// Incurred alone passed on all 20 and keeps R-hat below 1.01.
fn converged(fitted: Fitted, d: &Diagnostics) {
    let rhat_limit = if fitted == Fitted::PaidOnly {
        1.02
    } else {
        1.01
    };
    assert!(d.max_rhat < rhat_limit, "{fitted:?}: {d:?}");
    if fitted == Fitted::PaidOnly {
        assert!(
            d.divergences <= 2 && d.min_ess_bulk > 400.0 && d.min_ess_tail > 400.0,
            "{fitted:?}: {d:?}"
        );
    } else {
        assert!(
            d.warnings
                .iter()
                .all(|w| w.contains("does not identify the payment rate"))
                && d.divergences == 0,
            "{fitted:?}: {d:?}"
        );
    }
}

#[test]
fn defaults_converge_from_the_data_start() {
    for (fitted, _) in MODES {
        let fit = fit(fitted);
        converged(fitted, &fit.diagnostics);
        // Ordered rates: no draw has the swap.
        assert_eq!(fit.rate_swaps, 0);
        assert_eq!(fit.population_origins, [Period::year(1997)]);
    }
}

#[test]
fn defaults_converge_from_the_prior_centre() {
    for (fitted, method) in MODES {
        let start = Instant::now();
        let fit = run(&Compartmental {
            start: CompartmentalStart::PriorCentre,
            ..model(fitted, 0)
        });
        report(
            &format!("{fitted:?} prior-centre start"),
            &fit,
            start.elapsed(),
        );
        converged(fitted, &fit.diagnostics);
        let means: Vec<Case> = cases(method)
            .into_iter()
            .filter(|c| c.get("quantity") == "mean")
            .collect();
        check(&means, |case| value(&fit, case));
    }
}

#[test]
fn loo_has_the_one_type() {
    for (fitted, _) in MODES {
        let loo = fit(fitted).loo().unwrap();
        assert_eq!(loo.loo.pareto_k.len(), 45);
        if fitted == Fitted::PaidOnly {
            assert_eq!((loo.n_outstanding, loo.elpd_outstanding), (0, 0.0));
            assert_eq!(loo.elpd_paid, loo.loo.estimate.elpd);
        } else {
            assert_eq!((loo.n_outstanding, loo.elpd_paid), (45, 0.0));
            assert_eq!(loo.elpd_outstanding, loo.loo.estimate.elpd);
        }
        eprintln!(
            "{fitted:?}: elpd_loo {:.2} (se {:.2}), p_loo {:.2}, max k {:.3}",
            loo.loo.estimate.elpd,
            loo.loo.estimate.se,
            loo.loo.estimate.p,
            loo.loo
                .pareto_k
                .iter()
                .fold(f64::NEG_INFINITY, |m, &k| m.max(k))
        );
    }
}

/// The comparison recorded in `knowledge/findings/compartmental-one-response.md`:
/// the total reserve, its sd and the population parameters of each
/// variant beside paid and outstanding and paid and incurred (free rates,
/// as their own references run), at seed 0. Run in a release build:
/// `cargo test --release -p prospicio-validation --features slow-bayes --test
/// reserving_compartmental_one_response compare -- --ignored --nocapture`.
#[test]
#[ignore]
fn compare_with_two_responses() {
    for fitted in [
        Fitted::PaidAndOutstanding,
        Fitted::PaidAndIncurred,
        Fitted::PaidOnly,
        Fitted::IncurredOnly,
    ] {
        let one = matches!(fitted, Fitted::PaidOnly | Fitted::IncurredOnly);
        let m = Compartmental {
            rate_order: if one {
                RateOrder::Ordered
            } else {
                RateOrder::Free
            },
            ..model(fitted, 0)
        };
        let start = Instant::now();
        let fit = run(&m);
        report(&format!("{fitted:?}"), &fit, start.elapsed());
        let sd = |name: &str| moments(&fit, name).map_or(f64::NAN, |m| m.1);
        eprintln!(
            "  sd k_er {:.3} k_p {:.3} RLR {:.3} RRF {:.3}; reserves by origin {:?}",
            sd("k_er"),
            sd("k_p"),
            sd("RLR"),
            sd("RRF"),
            fit.to_long().values[2]
                .1
                .iter()
                .map(|v| v.round())
                .collect::<Vec<_>>()
        );
    }
}

/// The seed sweep that checks the Stan-parity tolerances (decision 7) on
/// these variants: for seeds `0..COMPARTMENTAL_SEEDS` (default 20), every
/// reference row's value, written to
/// `target/compartmental_one_response_sweep.csv`, with each fit's
/// diagnostics and time. Run in a release build:
/// `cargo test --release -p prospicio-validation --features slow-bayes --test
/// reserving_compartmental_one_response seed_sweep -- --ignored --nocapture`.
#[test]
#[ignore]
fn seed_sweep() {
    let seeds: u64 = std::env::var("COMPARTMENTAL_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let mut out = String::from("seed,method,quantity,arg,got,expected,abs_tol,rel_tol,passes\n");
    let mut diagnostics = String::from(
        "seed,method,seconds,max_rhat,min_ess_bulk,min_ess_tail,divergences,warnings\n",
    );
    for (fitted, method) in MODES {
        let cases = cases(method);
        for seed in 0..seeds {
            let start = Instant::now();
            let fit = run(&model(fitted, seed));
            let elapsed = start.elapsed();
            let d = &fit.diagnostics;
            eprintln!(
                "{method} seed {seed}: {elapsed:.2?}, max R-hat {:.4}, min ESS {:.0}/{:.0}, {} \
                 divergences, {} warnings {:?}",
                d.max_rhat,
                d.min_ess_bulk,
                d.min_ess_tail,
                d.divergences,
                d.warnings.len(),
                d.warnings
            );
            diagnostics.push_str(&format!(
                "{seed},{method},{:.3},{},{},{},{},{}\n",
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
                    "{seed},{method},{},{},{got:e},{want:e},{abs_tol:e},{rel_tol:e},{passes}\n",
                    case.get("quantity"),
                    case.get("arg")
                ));
            }
        }
    }
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../target");
    std::fs::write(format!("{dir}/compartmental_one_response_sweep.csv"), out).unwrap();
    std::fs::write(
        format!("{dir}/compartmental_one_response_sweep_diagnostics.csv"),
        diagnostics,
    )
    .unwrap();
    eprintln!("wrote {dir}/compartmental_one_response_sweep.csv");
}

fn diffuse_model(fitted: Fitted, seed: u64) -> Compartmental {
    Compartmental {
        seed,
        fitted,
        rate_order: RateOrder::Ordered,
        ..Compartmental::new(CompartmentalPriors::diffuse())
    }
}

/// The diffuse preset on paid alone over a few seeds (incurred alone refuses
/// it, `diffuse_preset_is_refused_on_incurred_alone`), recorded in
/// `knowledge/findings/compartmental-one-response.md`. Run in a release build:
/// `cargo test --release -p prospicio-validation --features slow-bayes --test
/// reserving_compartmental_one_response diffuse_sweep -- --ignored --nocapture`.
#[test]
#[ignore]
fn diffuse_sweep() {
    for fitted in [Fitted::PaidOnly] {
        for seed in 0..5 {
            let start = Instant::now();
            let fit = run(&diffuse_model(fitted, seed));
            report(
                &format!("{fitted:?} diffuse seed {seed}"),
                &fit,
                start.elapsed(),
            );
            let sd = |name: &str| moments(&fit, name).map_or(f64::NAN, |m| m.1);
            eprintln!(
                "  sd k_er {:.3} k_p {:.3} RLR {:.3} RRF {:.3}",
                sd("k_er"),
                sd("k_p"),
                sd("RLR"),
                sd("RRF")
            );
        }
    }
}

/// Incurred alone under log-scale sds between the paper's and the widest
/// the fit accepts (0.5), for `DIFFUSE_SDS` (comma separated, default 0.3,0.5),
/// on all four rate and ratio priors, seeds 0 to 2. A wider sd is refused
/// (`MAX_INCURRED_ONLY_KP_SD`); the 0.7 and 1.0 rows of the findings note were
/// measured before that decision.
#[test]
#[ignore]
fn incurred_sd_ladder() {
    let sds: Vec<f64> = std::env::var("DIFFUSE_SDS")
        .unwrap_or_else(|_| "0.3,0.5".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    for sd in sds {
        let wide =
            |median: f64| prospicio_reserving::bayes::PositivePrior::LogNormal { median, sd };
        let priors = CompartmentalPriors {
            k_er: wide(3.0),
            k_p: wide(1.0),
            rlr: wide(0.7),
            rrf: wide(0.8),
            ..CompartmentalPriors::gesmann_morris()
        };
        for seed in 0..3 {
            let start = Instant::now();
            let m = Compartmental {
                seed,
                fitted: Fitted::IncurredOnly,
                rate_order: RateOrder::Ordered,
                ..Compartmental::new(priors)
            };
            let fit = run(&m);
            report(
                &format!("IncurredOnly sd {sd} seed {seed}"),
                &fit,
                start.elapsed(),
            );
            let s = |name: &str| moments(&fit, name).map_or(f64::NAN, |m| m.1);
            eprintln!(
                "  sd k_er {:.3} k_p {:.3} RLR {:.3} RRF {:.3}",
                s("k_er"),
                s("k_p"),
                s("RLR"),
                s("RRF")
            );
        }
    }
}

/// The diffuse preset, paid alone (decision 12, the user's decision of
/// 2026-10-22): the data pin `k_p` (the posterior sd of its log is 5% of the
/// prior's), the fit converges, and the total reserve is 181,550 within 1%,
/// the range of 5 seeds being 181,425 to 181,690 (sd 12,100 to 12,700), 2.9%
/// above the paper's preset on paid alone (176,448) and 4.7% above paid and
/// outstanding (173,390).
#[test]
fn diffuse_priors_on_paid_alone() {
    let fit = run(&diffuse_model(Fitted::PaidOnly, 0));
    report("PaidOnly diffuse", &fit, std::time::Duration::ZERO);
    converged(Fitted::PaidOnly, &fit.diagnostics);
    let total = fit.reserves.total();
    assert!(
        (total.mean() / 181_550.0 - 1.0).abs() < 0.01,
        "{}",
        total.mean()
    );
    assert!(total.std_dev() > 11_000.0 && total.std_dev() < 14_000.0);
    assert!(log_kp_sd(&fit) < 0.1, "{}", log_kp_sd(&fit));
}

/// The diffuse preset, incurred alone (decision 12, the user's decision of
/// 2026-10-23): refused. The incurred does not identify `k_p`, so the reserve
/// follows the prior on `k_p` at every width (on company 337 -51,000 with an
/// sd of 152,000 at a log-scale sd of 1, 203,000 at 0.3), and a `k_p` prior
/// wider than 0.5 is an error before any data are read. The error comes from
/// `fit`; the fit is not run.
#[test]
fn diffuse_preset_is_refused_on_incurred_alone() {
    let err = diffuse_model(Fitted::IncurredOnly, 0)
        .fit(
            training(),
            "paid",
            Outstanding::FromIncurred("incurred"),
            "premium",
        )
        .expect_err("a k_p prior with a log-scale sd of 1 is refused");
    let message = err.to_string();
    assert!(
        message.contains("does not identify the payment rate") && message.contains("at most 0.5"),
        "{message}"
    );
}

/// Under the paper's preset the same warning fires on incurred alone (the
/// posterior sd of `log k_p` is 99% of its prior's) and not on paid alone.
#[test]
fn the_paper_preset_warns_on_incurred_alone_only() {
    let warns = |fit: &CompartmentalFit| {
        fit.diagnostics
            .warnings
            .iter()
            .any(|w| w.contains("does not identify the payment rate"))
    };
    assert!(warns(fit(Fitted::IncurredOnly)));
    assert!(!warns(fit(Fitted::PaidOnly)));
}
