//! Mack's bootstrap, the lifetime view, under the link ratio exclusions
//! `n_periods`, `drop_high` and `drop_low` (decision 14 of
//! `docs/design/reserving-v02.md`), against Mack's standard errors under
//! the same settings.
//!
//! The drops stay fixed on the observed triangle, so the bootstrap is the
//! model [`Mack`] computes with 0/1 weights (R's `MackChainLadder(weights =
//! W)`, whose standard errors Mack's are at parity with): the mean reserve
//! is the chain ladder's under the same settings, and each origin's and the
//! total standard deviation is Mack's, with the one adjustment of
//! `knowledge/findings/mack-bootstrap-lifetime-vs-mack.md`: the parameter
//! error is scaled by the variance `v = 1 - m^2` of the resampled
//! residuals, here the kept ones. The tolerance is that test's: five Monte
//! Carlo standard errors of the standard deviation (`sd * sqrt((kurtosis -
//! 1) / (4 n))`) and of the mean (`sd / sqrt(n)`), at 20,000 simulations.

use prospicio_reserving::{Development, Mack, MackBootstrap, MackBootstrapFit, MackProcess};
use prospicio_validation::triangle;

/// Simulations per run; the tolerances scale with `1 / sqrt` of it.
const SIMS: usize = 20_000;
const SEED: u64 = 20_261_007;

fn development(n_periods: Option<usize>, drop_high: usize, drop_low: usize) -> Development {
    Development {
        n_periods,
        drop_high,
        drop_low,
        ..Default::default()
    }
}

/// Mean, standard deviation and the Monte Carlo standard error of the
/// standard deviation.
fn moments(x: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let m2 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let m4 = x.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n;
    if m2 == 0.0 {
        return (mean, 0.0, 0.0);
    }
    let sd = m2.sqrt();
    (mean, sd, sd * ((m4 / (m2 * m2) - 1.0) / (4.0 * n)).sqrt())
}

/// The variance of the kept residuals: the pool's mean square less its
/// squared mean.
fn pool_variance(fit: &MackBootstrapFit) -> f64 {
    let pool: Vec<f64> = fit
        .residuals
        .iter()
        .copied()
        .filter(|r| !r.is_nan())
        .collect();
    let n = pool.len() as f64;
    let m = pool.iter().sum::<f64>() / n;
    pool.iter().map(|r| r * r).sum::<f64>() / n - m * m
}

fn check(dataset: &str, setting: &str, dev: Development) {
    let tri = triangle(dataset);
    let fit = MackBootstrap {
        n_sims: SIMS,
        seed: SEED,
        process: MackProcess::Gamma,
        development: dev.clone(),
        ..Default::default()
    }
    .fit(&tri, "values")
    .unwrap_or_else(|e| panic!("{dataset} {setting}: {e}"));
    // Mack's model under the same settings, fitted on its own.
    let mack = Mack {
        development: dev,
        ..Default::default()
    }
    .fit(&tri, "values")
    .unwrap();
    assert!(
        mack.chain_ladder.development.excluded,
        "{dataset} {setting}: the settings exclude nothing"
    );
    assert_eq!(mack.standard_error, fit.mack.standard_error);

    let v = pool_variance(&fit);
    let n = fit.reserves.n_components();
    let rows = || fit.reserves.draw_matrix().chunks(n);
    let cl = &mack.chain_ladder;
    let mut failures = Vec::new();
    let mut report = |label: String,
                      reserve: f64,
                      process: f64,
                      parameter: f64,
                      se: f64,
                      d: &[f64]| {
        let (mean, sd, error) = moments(d);
        let want = (process * process + v * parameter * parameter).sqrt();
        println!(
            "{dataset} {setting} {label}: sd {sd:.1} vs {want:.1} ({:+.2} SE, plain Mack {se:.1}), \
             mean {mean:.1} vs {reserve:.1}",
            (sd - want) / error
        );
        if want == 0.0 {
            if sd != 0.0 || mean != 0.0 {
                failures.push(format!("{dataset} {setting} {label}: sd {sd}, want 0"));
            }
            return;
        }
        if (sd - want).abs() > 5.0 * error {
            failures.push(format!(
                "{dataset} {setting} {label}: sd {sd:.1} vs Mack {want:.1} ({:+.2} SE)",
                (sd - want) / error
            ));
        }
        let mean_error = sd / (SIMS as f64).sqrt();
        if (mean - reserve).abs() > 5.0 * mean_error {
            failures.push(format!(
                "{dataset} {setting} {label}: mean {mean:.1} vs chain ladder {reserve:.1} \
                 ({:+.2} SE)",
                (mean - reserve) / mean_error
            ));
        }
    };
    for j in 0..n {
        let d: Vec<f64> = rows().map(|r| r[j]).collect();
        report(
            cl.origins[j].to_string(),
            cl.ultimate[j] - cl.latest[j],
            mack.process_risk[j],
            mack.parameter_risk[j],
            mack.standard_error[j],
            &d,
        );
    }
    let total: Vec<f64> = rows().map(|r| r.iter().sum()).collect();
    report(
        "total".into(),
        cl.total_reserve(),
        mack.total_process_risk,
        mack.total_parameter_risk,
        mack.total_standard_error,
        &total,
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn bootstrap_under_a_window_follows_mack_with_the_same_window() {
    for dataset in ["raa", "genins"] {
        check(dataset, "n_periods = 5", development(Some(5), 0, 0));
    }
}

#[test]
fn bootstrap_under_drops_follows_mack_with_the_same_drops() {
    for dataset in ["raa", "genins"] {
        check(dataset, "drop 1 and 1", development(None, 1, 1));
    }
}
