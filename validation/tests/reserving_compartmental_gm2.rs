//! Gesmann and Morris' compartmental model 2 (`Compartmental` with
//! `CompartmentalPriors::gesmann_morris_2()`, the log-normal likelihood on
//! loss ratios and incremental paid; `docs/design/reserving-compartmental.md`,
//! decisions 2 and 7) on the papers' company 337 workers' compensation
//! (`data/clrd_wkcomp_337.csv`), trained as the paper trains it (calendar
//! years up to 1996, outstanding = incurred - paid with bulk and IBNR,
//! direct premium; 90 observations over 9 accident years and 9 development
//! years, and 1997 with premium only), against the paper's code
//! reconstructed in Stan and its printed posterior.
//!
//! * Stan parity: `reference/reserving_compartmental_gm2_stan.csv`, from
//!   `scripts/reserving_compartmental_gm2_stan.py` (`scripts/stan/
//!   compartmental_gm2.stan`, CmdStan 2.39.0, 4 chains of 25,000 draws),
//!   rows `stan_gm2`. Our side runs the defaults with the model 2 preset and
//!   settings (`model` below), seed 0. Each tolerance is at least 4 times
//!   our Monte Carlo error, measured per row as the seed-to-seed sd over a
//!   seed sweep (the `seed_sweep` test below, ignored in CI); the measured
//!   spreads are in the script and the design note.
//! * The paper's printed `m2fit` (rows `gesmann_morris_2020_m2`): a looser
//!   second check.
//! * Convergence from the data-based start and from the prior centre
//!   (decision 5).

use std::sync::OnceLock;
use std::time::Instant;

use prospicio_core::StreamRng;
use prospicio_math::special::norm_quantile;
use prospicio_prob::Distribution;
use prospicio_reserving::bayes::{
    Compartmental, CompartmentalFit, CompartmentalPriors, CompartmentalStart, Outstanding,
};
use prospicio_reserving::{DevelopmentColumn, Grain, Lag, Long, Month, Period, Triangle};
use prospicio_validation::{Case, check, reference};

const REFERENCE: &str = "reserving_compartmental_gm2_stan.csv";

/// A row of `data/clrd_wkcomp_337.csv`: origin, age, paid, incurred and
/// direct premium.
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
/// with its premium only.
fn training() -> &'static Triangle {
    static TRIANGLE: OnceLock<Triangle> = OnceLock::new();
    TRIANGLE.get_or_init(|| {
        let all = rows();
        let mut rows: Vec<Row> = all
            .iter()
            .copied()
            .filter(|r| r.origin + (r.age / 12) as i32 - 1 <= 1996)
            .collect();
        assert_eq!(rows.len(), 45);
        rows.push(Row {
            origin: 1997,
            age: 12,
            paid: f64::NAN,
            incurred: f64::NAN,
            premium: all.iter().find(|r| r.origin == 1997).unwrap().premium,
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

/// Gesmann and Morris' model 2 with its preset and settings (`model_2`).
fn model(seed: u64) -> Compartmental {
    Compartmental {
        seed,
        ..Compartmental::model_2(CompartmentalPriors::gesmann_morris_2())
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
    let mean = |name: &str| moments(fit, name).map_or(f64::NAN, |m| m.0);
    eprintln!(
        "{label}: {elapsed:.2?}; total reserve mean {:.0} sd {:.0}; b_ker {:.2} b_kp {:.2} \
         b_RLR {:.2} b_RRF {:.2} tau {:.2} {:.2} {:.2} {:.2} rho {:.2} tau_dev {:.2} {:.2} \
         {:.2} {:.2} log sigma {:.2} {:.2}; cells with k_er <= k_p {:?}; diagnostics {:?}",
        total.mean(),
        total.std_dev(),
        mean("b_ker"),
        mean("b_kp"),
        mean("b_RLR"),
        mean("b_RRF"),
        mean("tau_RLR"),
        mean("tau_RRF"),
        mean("tau_ker"),
        mean("tau_kp"),
        mean("rho"),
        mean("tau_dev_RLR"),
        mean("tau_dev_RRF"),
        mean("tau_dev_ker"),
        mean("tau_dev_kp"),
        mean("log_sigma_os"),
        mean("log_sigma_pd"),
        fit.cell_rate_swaps,
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
            // `os_1990` or `paid_1990`: the cell on the 1997 diagonal.
            let (kind, year) = arg.split_once('_')?;
            test_mean(fit, kind == "paid", year.parse().ok()?)?
        }
        "elpd_loo" => fit.loo().ok()?.loo.estimate.elpd,
        _ => return None,
    })
}

/// New effects drawn per posterior draw for a level with no training cell
/// in [`test_mean`]: the expectation over them is a Monte Carlo average.
const NEW_LEVEL_DRAWS: usize = 25;

/// The posterior mean of the expected outstanding or incremental paid
/// amount of `year`'s cell on the 1997 diagonal (the reference's
/// `test_mean`, which has no process noise), with the paper's closed form:
/// each draw's parameters of the cell, its origin's values times its
/// age's development multipliers; for 1997 (no training cell) and age 10
/// (no training development year) the expectation over new effects from
/// the population, averaged over `NEW_LEVEL_DRAWS` draws of them.
fn test_mean(fit: &CompartmentalFit, paid: bool, year: i32) -> Option<f64> {
    let names = fit.posterior.names();
    let at = |name: &str| names.iter().position(|n| n == name);
    let i = fit.origins.iter().position(|&o| o == Period::year(year))?;
    let t = f64::from(1997 - year + 1);
    let age = 12 * (1997 - year + 1);
    let params = ["ker", "kp", "RLR", "RRF"];
    let natural = ["k_er", "k_p", "RLR", "RRF"];
    // The preset's medians and o-scale multipliers.
    let (m, s) = ([3.0, 1.0, 0.7, 0.8], [0.1, 0.1, 0.2, 0.1]);
    let own: Vec<Option<usize>> = natural
        .iter()
        .map(|n| at(&format!("{n}[{year}]")))
        .collect();
    let dev: Vec<Option<usize>> = params
        .iter()
        .map(|p| at(&format!("dev_{p}[{age}]")))
        .collect();
    let b: Vec<usize> = ["b_ker", "b_kp", "b_RLR", "b_RRF"]
        .iter()
        .map(|n| at(n))
        .collect::<Option<_>>()?;
    let tau: Vec<usize> = ["tau_ker", "tau_kp", "tau_RLR", "tau_RRF"]
        .iter()
        .map(|n| at(n))
        .collect::<Option<_>>()?;
    let tau_dev: Vec<usize> = params
        .iter()
        .map(|p| at(&format!("tau_dev_{p}")))
        .collect::<Option<_>>()?;
    let rho = at("rho")?;
    let new_origin = own.iter().any(Option::is_none);
    let new_age = dev.iter().any(Option::is_none);
    let k = if new_origin || new_age {
        NEW_LEVEL_DRAWS
    } else {
        1
    };
    let mut rng = StreamRng::new(20261015, year as u64);
    let mut normal = || norm_quantile(rng.next_open01());
    let pd = |a: f64, b: f64, rlr: f64, rrf: f64, t: f64| {
        rlr * rrf / (a - b) * (a * (1.0 - (-b * t).exp()) - b * (1.0 - (-a * t).exp()))
    };
    let mut sum = 0.0;
    for d in fit.posterior.values().chunks_exact(fit.posterior.dim()) {
        for _ in 0..k {
            // The o-scale value of each parameter at the cell.
            let mut o = [0.0; 4];
            if new_origin {
                let (z1, z2) = (normal(), normal());
                let r = d[rho];
                o[2] = d[b[2]] + d[tau[2]] * z1;
                o[3] = d[b[3]] + d[tau[3]] * (r * z1 + (1.0 - r * r).sqrt() * z2);
                o[0] = d[b[0]] + d[tau[0]] * normal();
                o[1] = d[b[1]] + d[tau[1]] * normal();
            } else {
                for p in 0..4 {
                    o[p] = (d[own[p]?] / m[p]).ln() / s[p];
                }
            }
            for p in 0..4 {
                o[p] += match dev[p] {
                    Some(j) => d[j],
                    None => d[tau_dev[p]] * normal(),
                };
            }
            let th: Vec<f64> = (0..4).map(|p| m[p] * (s[p] * o[p]).exp()).collect();
            let (a, bb, rlr, rrf) = (th[0], th[1], th[2], th[3]);
            sum += fit.premium[i]
                * if paid {
                    if t > 1.0 {
                        pd(a, bb, rlr, rrf, t) - pd(a, bb, rlr, rrf, t - 1.0)
                    } else {
                        pd(a, bb, rlr, rrf, t)
                    }
                } else {
                    rlr * a / (a - bb) * ((-bb * t).exp() - (-a * t).exp())
                };
        }
    }
    Some(sum / (fit.posterior.n_draws() * k) as f64)
}

fn cases(prefix: &str) -> Vec<Case> {
    reference(REFERENCE)
        .into_iter()
        .filter(|c| c.get("method").starts_with(prefix))
        .collect()
}

#[test]
fn stan_parity() {
    check(&cases("stan_"), |case| value(fit(), case));
}

#[test]
fn the_papers_printed_posterior() {
    check(&cases("gesmann_morris_2020_"), |case| value(fit(), case));
}

#[test]
fn defaults_converge_from_the_data_start() {
    let fit = fit();
    let d = &fit.diagnostics;
    assert!(d.warnings.is_empty(), "{d:?}");
    assert!(d.max_rhat < 1.01 && d.divergences == 0, "{d:?}");
    assert_eq!(fit.rate_swaps, 0);
    assert_eq!(fit.population_origins, [Period::year(1997)]);
    let loo = fit.loo().unwrap();
    assert_eq!((loo.loo.pareto_k.len(), loo.n_outstanding), (90, 45));
    eprintln!(
        "elpd_loo {:.2} (se {:.2}; outstanding {:.2}, paid {:.2}), p_loo {:.2}, max k {:.3}",
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
    let means: Vec<Case> = cases("stan_gm2")
        .into_iter()
        .filter(|c| c.get("quantity") == "mean")
        .collect();
    check(&means, |case| value(&fit, case));
}

/// The seed sweep that sizes the Stan-parity tolerances (decision 7): for
/// seeds `COMPARTMENTAL_SEED_FROM..COMPARTMENTAL_SEEDS` (default `0..20`;
/// several processes can share a sweep), every reference row's value,
/// written to `target/compartmental_gm2_sweep_<from>.csv`, with each fit's
/// diagnostics and time. Run in a release build:
/// `cargo test --release -p prospicio-validation --features slow-bayes --test
/// reserving_compartmental_gm2 seed_sweep -- --ignored --nocapture`.
#[test]
#[ignore]
fn seed_sweep() {
    let env = |name: &str, default: u64| -> u64 {
        std::env::var(name)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    };
    let (from, seeds) = (
        env("COMPARTMENTAL_SEED_FROM", 0),
        env("COMPARTMENTAL_SEEDS", 20),
    );
    // Before the reference exists (the sweep can run beside the Stan run),
    // the rows it will have, with no expected values: written under
    // target/ and read back as cases.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/reference/reserving_compartmental_gm2_stan.csv"
    );
    let cases = if std::path::Path::new(path).exists() {
        reference(REFERENCE)
    } else {
        let names = run(&Compartmental {
            n_sims: 10,
            sampler: prospicio_bayes::nuts::Sampler {
                chains: 1,
                tune: 10,
                draws: 10,
                ..model(0).sampler
            },
            ..model(0)
        })
        .posterior
        .names()
        .to_vec();
        let mut rows =
            String::from("dataset,method,quantity,arg,expected,abs_tol,rel_tol,source\n");
        let mut row = |quantity: &str, arg: &str| {
            rows.push_str(&format!("wkcomp_337,stan_gm2,{quantity},{arg},0,0,0,\n"));
        };
        for name in &names {
            row("mean", name);
            row("sd", name);
        }
        let years: Vec<String> = (1988..=1997).map(|y| y.to_string()).collect();
        for y in &years {
            row("ulr_mean", y);
        }
        for arg in years.iter().map(String::as_str).chain(["total"]) {
            for q in ["mean", "sd", "q05", "q50", "q95"] {
                row(&format!("reserve_{q}"), arg);
            }
        }
        for kind in ["os", "paid"] {
            for y in &years {
                row("test_mean", &format!("{kind}_{y}"));
            }
        }
        row("elpd_loo", "");
        let skeleton = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../target/compartmental_gm2_rows.csv"
        );
        std::fs::write(skeleton, rows).unwrap();
        reference("../../target/compartmental_gm2_rows.csv")
    };
    let mut out = String::from("seed,method,quantity,arg,got,passes\n");
    let mut diagnostics = String::from(
        "seed,seconds,max_rhat,min_ess_bulk,min_ess_tail,divergences,rate_swaps,cell_rate_swaps,warnings\n",
    );
    // Sampler settings other than the defaults, to compare them.
    let base = model(0);
    let sampler = prospicio_bayes::nuts::Sampler {
        tune: env("COMPARTMENTAL_TUNE", base.sampler.tune as u64) as usize,
        draws: env("COMPARTMENTAL_DRAWS", base.sampler.draws as u64) as usize,
        target_accept: std::env::var("COMPARTMENTAL_TARGET_ACCEPT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(base.sampler.target_accept),
        ..base.sampler
    };
    for seed in from..seeds {
        let start = Instant::now();
        let fit = run(&Compartmental {
            sampler,
            ..model(seed)
        });
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
            "{seed},{:.3},{},{},{},{},{},{},{}\n",
            elapsed.as_secs_f64(),
            d.max_rhat,
            d.min_ess_bulk,
            d.min_ess_tail,
            d.divergences,
            fit.rate_swaps,
            fit.cell_rate_swaps.unwrap_or(f64::NAN),
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
        // After each seed, so a long sweep keeps what it has done.
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../target");
        std::fs::write(format!("{dir}/compartmental_gm2_sweep_{from}.csv"), &out).unwrap();
        std::fs::write(
            format!("{dir}/compartmental_gm2_sweep_diagnostics_{from}.csv"),
            &diagnostics,
        )
        .unwrap();
    }
}
