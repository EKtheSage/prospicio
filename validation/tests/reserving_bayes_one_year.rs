//! The one-year view of the Bayesian ODP chain ladder by Pareto smoothed
//! importance sampling (`BayesOdp::one_year`,
//! `docs/design/reserving-bayes.md`, decision 16), on RAA and GenIns.
//!
//! * Against the ODP bootstrap's one-year view (`OdpBootstrap::one_year`
//!   re-reserving with the chain ladder, 50,000 simulations, seed 1): the
//!   same over-dispersed Poisson process and, under vague priors, about
//!   the same parameter error, so each origin's and the total's standard
//!   deviation of the claims development result should agree. The
//!   Bayesian side reweights every simulation (`ReweightOnly`, 10,000
//!   simulations), which is what CI can afford; refitting the flagged
//!   simulations, the default, moves the total's sd by about 1% (the
//!   design note, decision 16), inside the tolerances.
//! * Merz and Wüthrich's standard errors are printed by `report`, not
//!   checked: they assume Mack's process, whose variance is proportional
//!   to the cumulative value, and differ from the ODP's as the bootstrap's
//!   do (`knowledge/findings/one-year-bootstrap-vs-merz-wuthrich.md`).
//! * The importance sampling error: on 100 simulations, the total claims
//!   development result of each reweighted simulation against the same
//!   simulation refitted with NUTS (`RefitAll`, the same years), as a
//!   root mean square in units of the total's sd, separately for the
//!   simulations below and above `k_threshold`; and the share flagged.
//! * The mean claims development result is zero within Monte Carlo error.
//!
//! Every tolerance is at least 4 times its row's seed-to-seed sd (plus
//! the row's mean offset from its target) over seeds 0 to 79 (the refit
//! rows 0 to 19), and every row passes at each of them, using at most 0.79
//! of its tolerance; from the ignored `seed_sweep` (`SWEEP_START` moves
//! the first seed):
//! `SWEEP_SEEDS=40 PSIS_SEEDS=10 cargo test --release -p
//! prospicio-validation --features slow-bayes --test
//! reserving_bayes_one_year -- --ignored seed_sweep --nocapture`.

use std::time::Instant;

use prospicio_reserving::bayes::{BayesOdp, BayesOneYear, BayesOneYearFit, OneYearRefit};
use prospicio_reserving::{ChainLadder, Mack, OdpBootstrap, OneYearMethod};
use prospicio_validation::triangle;

/// The Bayesian side's simulations; the default.
const SIMS: usize = 10_000;
/// The bootstrap's simulations, enough that its error is small beside ours.
const BOOT_SIMS: usize = 50_000;
/// Simulations refitted for the importance sampling error.
const REFIT_SIMS: usize = 100;

/// `(dataset, row, target, tolerance)`: the ratio rows' target is 1 (the
/// bootstrap), the gap rows' 0 (a ceiling), the flagged share's the
/// sweep's mean. Origins are numbered from the oldest (0, which has no
/// cell in the year, is not checked).
const ROWS: &[(&str, &str, f64, f64)] = &[
    ("raa", "sd_ratio[1]", 1.0, 0.30),
    ("raa", "sd_ratio[2]", 1.0, 0.18),
    ("raa", "sd_ratio[3]", 1.0, 0.12),
    ("raa", "sd_ratio[4]", 1.0, 0.12),
    ("raa", "sd_ratio[5]", 1.0, 0.09),
    ("raa", "sd_ratio[6]", 1.0, 0.10),
    ("raa", "sd_ratio[7]", 1.0, 0.11),
    ("raa", "sd_ratio[8]", 1.0, 0.12),
    ("raa", "sd_ratio[9]", 1.0, 0.15),
    ("raa", "sd_ratio[total]", 1.0, 0.12),
    ("raa", "mean_z[total]", 0.0, 5.0),
    ("raa", "flagged_share", 0.136, 0.03),
    ("raa", "rms_gap_unflagged", 0.0, 0.08),
    ("raa", "rms_gap_flagged", 0.0, 0.7),
    ("genins", "sd_ratio[1]", 1.0, 0.12),
    ("genins", "sd_ratio[2]", 1.0, 0.08),
    ("genins", "sd_ratio[3]", 1.0, 0.09),
    ("genins", "sd_ratio[4]", 1.0, 0.08),
    ("genins", "sd_ratio[5]", 1.0, 0.09),
    ("genins", "sd_ratio[6]", 1.0, 0.09),
    ("genins", "sd_ratio[7]", 1.0, 0.09),
    ("genins", "sd_ratio[8]", 1.0, 0.11),
    ("genins", "sd_ratio[9]", 1.0, 0.10),
    ("genins", "sd_ratio[total]", 1.0, 0.10),
    ("genins", "mean_z[total]", 0.0, 5.0),
    ("genins", "flagged_share", 0.264, 0.035),
    ("genins", "rms_gap_unflagged", 0.0, 0.10),
    ("genins", "rms_gap_flagged", 0.0, 0.5),
];

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn sd(x: &[f64]) -> f64 {
    let m = mean(x);
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
}

/// Each origin's draws, then the total's.
fn columns(draws: &[f64], n: usize) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = (0..n)
        .map(|j| draws.chunks(n).map(|r| r[j]).collect())
        .collect();
    out.push(draws.chunks(n).map(|r| r.iter().sum()).collect());
    out
}

fn model(seed: u64, n_sims: usize) -> BayesOdp {
    BayesOdp {
        n_sims,
        seed,
        ..BayesOdp::default()
    }
}

fn view(dataset: &str, model: &BayesOdp, refit: OneYearRefit) -> BayesOneYearFit {
    model
        .one_year(
            &triangle(dataset),
            "values",
            &BayesOneYear {
                refit: Some(refit),
                ..BayesOneYear::default()
            },
        )
        .unwrap_or_else(|e| panic!("{dataset}: {e}"))
}

/// Each origin's and the total's sd of the bootstrap's one-year view.
fn bootstrap_sds(dataset: &str) -> Vec<f64> {
    let fit = OdpBootstrap {
        n_sims: BOOT_SIMS,
        seed: 1,
        ..Default::default()
    }
    .one_year(
        &triangle(dataset),
        "values",
        &OneYearMethod::ChainLadder(ChainLadder::default()),
    )
    .unwrap_or_else(|e| panic!("{dataset}: {e}"));
    let n = fit.cdr.n_components();
    columns(fit.cdr.draw_matrix(), n)
        .iter()
        .map(|c| sd(c))
        .collect()
}

/// The rows of `dataset` at `seed`; with `refits`, the importance sampling
/// error rows too. `boot` are the bootstrap's sds.
fn rows(dataset: &str, seed: u64, boot: &[f64], refits: bool) -> Vec<(String, f64)> {
    let full = view(dataset, &model(seed, SIMS), OneYearRefit::ReweightOnly);
    let n = full.cdr.n_components();
    let cols = columns(full.cdr.draw_matrix(), n);
    let mut out = Vec::new();
    for (j, c) in cols.iter().enumerate().skip(1) {
        let label = if j == n {
            "total".to_string()
        } else {
            j.to_string()
        };
        out.push((format!("sd_ratio[{label}]"), sd(c) / boot[j]));
    }
    let total = &cols[n];
    out.push((
        "mean_z[total]".into(),
        mean(total) / (sd(total) / (SIMS as f64).sqrt()),
    ));
    out.push(("flagged_share".into(), full.flagged_share()));
    if refits {
        let small = model(seed, REFIT_SIMS);
        let a = view(dataset, &small, OneYearRefit::ReweightOnly);
        let b = view(dataset, &small, OneYearRefit::RefitAll);
        let ta = &columns(a.cdr.draw_matrix(), n)[n];
        let tb = &columns(b.cdr.draw_matrix(), n)[n];
        let s = sd(total);
        for (name, flagged) in [("rms_gap_unflagged", false), ("rms_gap_flagged", true)] {
            let squares: Vec<f64> = (0..REFIT_SIMS)
                .filter(|&i| (a.pareto_k[i].is_nan() || a.pareto_k[i] > 0.7) == flagged)
                .map(|i| ((ta[i] - tb[i]) / s).powi(2))
                .collect();
            out.push((name.into(), mean(&squares).sqrt()));
        }
    }
    out
}

fn expected(dataset: &str, row: &str) -> (f64, f64) {
    let found: Vec<_> = ROWS
        .iter()
        .filter(|r| r.0 == dataset && r.1 == row)
        .collect();
    assert_eq!(found.len(), 1, "{dataset} {row}");
    (found[0].2, found[0].3)
}

/// How much of its tolerance a row uses.
fn used(dataset: &str, row: &str, value: f64) -> f64 {
    let (target, tol) = expected(dataset, row);
    (value - target).abs() / tol
}

#[test]
fn the_one_year_view_agrees_with_the_odp_bootstrap_and_with_refits() {
    let mut failures = Vec::new();
    let mut checked = 0;
    for dataset in ["raa", "genins"] {
        let t = Instant::now();
        let boot = bootstrap_sds(dataset);
        for (row, value) in rows(dataset, 0, &boot, true) {
            checked += 1;
            let u = used(dataset, &row, value);
            println!("{dataset} {row}: {value:.4} uses {u:.2} of its tolerance");
            if u > 1.0 {
                failures.push(format!(
                    "{dataset} {row}: {value} uses {u:.2} of its tolerance"
                ));
            }
        }
        println!("{dataset}: {:?}", t.elapsed());
    }
    assert_eq!(checked, ROWS.len());
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Seed-to-seed spread of every row, and the largest share of a tolerance
/// any seed uses (ignored; release build).
#[test]
#[ignore]
fn seed_sweep() {
    let env = |name: &str, default: u64| {
        std::env::var(name)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    };
    let (seeds, psis_seeds, start) = (
        env("SWEEP_SEEDS", 40),
        env("PSIS_SEEDS", 10),
        env("SWEEP_START", 0),
    );
    for dataset in ["raa", "genins"] {
        let boot = bootstrap_sds(dataset);
        let mut by_row: Vec<(String, Vec<f64>)> = Vec::new();
        for seed in start..start + seeds {
            for (row, value) in rows(dataset, seed, &boot, seed < start + psis_seeds) {
                match by_row.iter_mut().find(|(r, _)| *r == row) {
                    Some((_, v)) => v.push(value),
                    None => by_row.push((row, vec![value])),
                }
            }
        }
        for (row, values) in &by_row {
            let (target, tol) = expected(dataset, row);
            let worst = values
                .iter()
                .map(|v| used(dataset, row, *v))
                .fold(0.0, f64::max);
            let (lo, hi) = values
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                });
            println!(
                "{dataset} {row:20} n {:3} mean {:9.4} sd {:8.4} min {:9.4} max {:9.4} | target {target} tol {tol} (|offset| + 4 sd = {:.4}) worst {worst:.2}",
                values.len(),
                mean(values),
                sd(values),
                lo,
                hi,
                (mean(values) - target).abs() + 4.0 * sd(values)
            );
        }
    }
}

/// The comparison table of the design note: each origin's sd under each
/// setting, the bootstrap's and Merz–Wüthrich's, and the runtimes
/// (ignored; `NSIMS` simulations, default 10,000).
#[test]
#[ignore]
fn report() {
    let n_sims = std::env::var("NSIMS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(SIMS);
    for dataset in ["raa", "genins"] {
        let tri = triangle(dataset);
        let boot = bootstrap_sds(dataset);
        let mw = Mack::default()
            .fit(&tri, "values")
            .unwrap()
            .claims_development_result()
            .unwrap();
        let mut sds = Vec::new();
        for refit in [OneYearRefit::ReweightOnly, OneYearRefit::RefitFlagged] {
            let t = Instant::now();
            let v = view(dataset, &model(0, n_sims), refit);
            let n = v.cdr.n_components();
            println!(
                "{dataset} {refit:?}: {:?}, flagged {:.4}, refitted {}, refit warnings {}",
                t.elapsed(),
                v.flagged_share(),
                v.refitted.len(),
                v.refit_warnings
            );
            let cols = columns(v.cdr.draw_matrix(), n);
            sds.push((
                cols.iter().map(|c| sd(c)).collect::<Vec<_>>(),
                cols.iter().map(|c| mean(c)).collect::<Vec<_>>(),
                v.opening_reserve.clone(),
            ));
        }
        let n = sds[0].0.len() - 1;
        #[allow(clippy::needless_range_loop)]
        for j in 0..=n {
            let mwj = if j < n {
                mw.one_year_standard_error[j]
            } else {
                mw.total_one_year_standard_error
            };
            let open: f64 = if j < n {
                sds[0].2[j]
            } else {
                sds[0].2.iter().sum()
            };
            println!(
                "  {j:2} opening {open:>11.0} | reweight sd {:>10.0} mean {:>8.0} | refit-flagged sd {:>10.0} mean {:>8.0} | bootstrap {:>10.0} | MW {:>10.0}",
                sds[0].0[j], sds[0].1[j], sds[1].0[j], sds[1].1[j], boot[j], mwj
            );
        }
    }
}
