//! Mack's bootstrap, the lifetime view (`MackBootstrap::fit`, decision 8 of
//! `docs/design/reserving-v02.md`), against Mack's analytic standard errors
//! on RAA, GenIns and ABC and against England, Verrall and Wüthrich (2019),
//! Table 4.
//!
//! * The reconciliation: with centred residuals, each origin's and the
//!   total standard deviation of the simulated reserves match R
//!   ChainLadder's `MackChainLadder(tri, est.sigma = ...)`
//!   (`reference/reserving_chainladder_r.csv`), with either way of filling
//!   the last sigma, within five Monte Carlo standard errors of the simulated
//!   standard deviation, `sd * sqrt((kurtosis - 1) / (4 n))`, and the mean
//!   reserve matches the chain ladder's within five standard errors of the
//!   mean, `sd / sqrt(n)`. The reference is Mack's closed form, independent
//!   of the simulation, with one adjustment: the resampled residuals have
//!   the pool's variance `v = 1 - m^2` (`m` its mean; its mean square is
//!   1), not 1, so the pseudo factors' variance, and with it every
//!   parameter error, is `v` times Mack's (RAA `v` = 0.981, GenIns 0.9998,
//!   ABC 0.996). The reference standard deviation is therefore
//!   `sqrt(process^2 + v parameter^2)` from R's process and parameter
//!   risks. Parameter error alone (`MackProcess::None`) is checked the same
//!   way against `sqrt(v)` times R's parameter risk.
//! * Uncentred, as EVW's Appendix 1 reads, the pool's mean biases every
//!   pseudo factor and the mean reserve with it (in total about +17% on
//!   RAA, +0.7% on GenIns and -0.8% on ABC), and the standard deviation
//!   grows with the mean (RAA +9%), so only the centred bootstrap
//!   reconciles.
//! * England, Verrall and Wüthrich (2019), Table 4: their bootstrap of
//!   Mack's model, 500,000 simulations on Taylor–Ashe (GenIns) with Mack's
//!   rule for the last sigma, expected reserve and standard deviation per
//!   origin and in total, within five standard errors of the two
//!   simulations combined. Their expected reserves are matched with
//!   centred residuals; uncentred, the total is eleven combined standard
//!   errors above theirs, which the test also checks (more than five).
//!
//! The Gamma process (EVW's parametric example) draws by inverting its
//! cdf, which is slow at large shapes: ABC's later cells have shapes in the
//! thousands. ABC is therefore simulated with the lognormal of the same
//! mean and variance, which at those shapes is close to the Gamma; the
//! standard deviation depends on the process only through its mean and
//! variance as long as the values stay positive, which both keep. RAA's
//! young origins have shapes below one, where the lognormal's heavy tail
//! makes the standard deviation's own standard error unreliable, so RAA and
//! GenIns use the Gamma.

use prospicio_reserving::{
    Development, MackBootstrap, MackBootstrapFit, MackProcess, Period, SigmaInterpolation,
};
use prospicio_validation::{Case, reference, triangle};
use std::sync::OnceLock;

/// Simulations per run; the tolerances scale with `1 / sqrt` of it.
const SIMS: usize = 20_000;
const SEED: u64 = 20_261_007;

fn boot(
    dataset: &str,
    sigma_interpolation: SigmaInterpolation,
    process: MackProcess,
    centre_residuals: bool,
) -> MackBootstrapFit {
    MackBootstrap {
        n_sims: SIMS,
        seed: SEED,
        process,
        development: Development {
            sigma_interpolation,
            ..Default::default()
        },
        centre_residuals,
    }
    .fit(&triangle(dataset), "values")
    .unwrap_or_else(|e| panic!("{dataset}: {e}"))
}

/// The process of the reconciliation runs (see the module documentation).
fn process(dataset: &str) -> MackProcess {
    if dataset == "abc" {
        MackProcess::Lognormal
    } else {
        MackProcess::Gamma
    }
}

/// GenIns with Mack's rule for the last sigma and centred residuals, which
/// the reconciliation and Table 4 both check: simulated once.
fn genins_mack() -> &'static MackBootstrapFit {
    static FIT: OnceLock<MackBootstrapFit> = OnceLock::new();
    FIT.get_or_init(|| boot("genins", SigmaInterpolation::Mack, MackProcess::Gamma, true))
}

/// Mean, standard deviation and the Monte Carlo standard error of the
/// standard deviation, `sd * sqrt((kurtosis - 1) / (4 n))`.
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

/// Each origin's draws of the reserve, then the total's (labelled `""`),
/// with the chain ladder's reserve.
fn columns(fit: &MackBootstrapFit) -> Vec<(String, f64, Vec<f64>)> {
    let n = fit.reserves.n_components();
    let rows = || fit.reserves.draw_matrix().chunks(n);
    let cl = &fit.mack.chain_ladder;
    let mut out: Vec<(String, f64, Vec<f64>)> = cl
        .origins
        .iter()
        .enumerate()
        .map(|(j, p)| {
            (
                p.to_string(),
                cl.ultimate[j] - cl.latest[j],
                rows().map(|r| r[j]).collect(),
            )
        })
        .collect();
    out.push((
        "".into(),
        cl.total_reserve(),
        rows().map(|r| r.iter().sum()).collect(),
    ));
    out
}

/// The variance of the resampled residuals: the pool's mean square less
/// its squared mean.
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

/// R's Mack `quantity` of `dataset` under reference `method`, for an origin
/// or (`""`) the total.
fn mack(cases: &[Case], dataset: &str, method: &str, quantity: &str, origin: &str) -> f64 {
    let (quantity, arg) = match (quantity, origin.is_empty()) {
        ("se", true) => ("total_standard_error".to_string(), ""),
        (q, true) => (format!("total_{q}"), ""),
        (q, false) => (q.to_string(), origin),
    };
    let found: Vec<&Case> = cases
        .iter()
        .filter(|c| {
            c.get("dataset") == dataset
                && c.get("method") == method
                && c.get("quantity") == quantity
                && c.get("arg") == arg
        })
        .collect();
    assert_eq!(found.len(), 1, "{dataset} {method} {quantity} {arg}");
    found[0].number("expected").unwrap()
}

/// Checks every origin and the total of `fit` against `want` (given the
/// origin) for the standard deviation and the chain ladder for the mean;
/// failures go to `failures`, led by `label`.
fn check(
    fit: &MackBootstrapFit,
    label: &str,
    want: impl Fn(&str) -> f64,
    failures: &mut Vec<String>,
) {
    for (origin, reserve, draws) in columns(fit) {
        let (mean, sd, error) = moments(&draws);
        let want = want(&origin);
        if want == 0.0 {
            // The oldest origin has nothing left to develop.
            if sd != 0.0 || mean != 0.0 {
                failures.push(format!("{label} {origin}: mean {mean}, sd {sd}, want 0"));
            }
            continue;
        }
        if (sd - want).abs() > 5.0 * error {
            failures.push(format!(
                "{label} {origin}: sd {sd:.1} vs Mack {want:.1} ({:+.2} Monte Carlo SE)",
                (sd - want) / error
            ));
        }
        let mean_error = sd / (SIMS as f64).sqrt();
        if (mean - reserve).abs() > 5.0 * mean_error {
            failures.push(format!(
                "{label} {origin}: mean {mean:.1} vs chain ladder {reserve:.1} \
                 ({:+.2} Monte Carlo SE)",
                (mean - reserve) / mean_error
            ));
        }
    }
}

const RULES: [(&str, SigmaInterpolation); 2] = [
    ("mack", SigmaInterpolation::LogLinear),
    ("mack_sigma_mack", SigmaInterpolation::Mack),
];

#[test]
fn centred_bootstrap_reconciles_with_mack() {
    // The Gamma runs are the slow ones: one rule each for RAA and GenIns
    // (the rule changes only the last sigma; parameter error alone is
    // checked under both below), both for ABC.
    let runs = [
        ("raa", RULES[0]),
        ("genins", RULES[1]),
        ("abc", RULES[0]),
        ("abc", RULES[1]),
    ];
    let cases = reference("reserving_chainladder_r.csv");
    let mut failures = Vec::new();
    for (dataset, (method, sigma_interpolation)) in runs {
        {
            let owned;
            let fit = if (dataset, sigma_interpolation) == ("genins", SigmaInterpolation::Mack) {
                genins_mack()
            } else {
                owned = boot(dataset, sigma_interpolation, process(dataset), true);
                &owned
            };
            let v = pool_variance(fit);
            let want = |origin: &str| {
                let process = mack(&cases, dataset, method, "process_risk", origin);
                let parameter = mack(&cases, dataset, method, "parameter_risk", origin);
                (process * process + v * parameter * parameter).sqrt()
            };
            // The adjustment is small: the plain standard error is within
            // 1% of the adjusted one.
            let (se, adjusted) = (mack(&cases, dataset, method, "se", ""), want(""));
            assert!(
                (se / adjusted - 1.0).abs() < 0.01,
                "{dataset}: {se} vs {adjusted}"
            );
            check(fit, &format!("{method} {dataset}"), want, &mut failures);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn parameter_error_is_mack_parameter_risk() {
    let cases = reference("reserving_chainladder_r.csv");
    let mut failures = Vec::new();
    for (method, sigma_interpolation) in RULES {
        for dataset in ["raa", "genins", "abc"] {
            let fit = boot(dataset, sigma_interpolation, MackProcess::None, true);
            let v = pool_variance(&fit);
            let want =
                |origin: &str| v.sqrt() * mack(&cases, dataset, method, "parameter_risk", origin);
            check(&fit, &format!("{method} {dataset}"), want, &mut failures);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn evw_2019_table_4() {
    // England, Verrall and Wüthrich (2019), Table 4, "Expected Reserves"
    // and "Bootstrap Std Dev": their bootstrap of Mack's model on
    // Taylor-Ashe, 500,000 simulations, Mack's rule for the last sigma
    // (their Table 1: 21.13). Accident years 2 to 10 are origins 2002 to
    // 2010.
    let table_4 = [
        ("2002", 94_740.0, 75_502.0),
        ("2003", 469_419.0, 121_842.0),
        ("2004", 709_488.0, 133_525.0),
        ("2005", 984_602.0, 261_623.0),
        ("2006", 1_418_656.0, 410_932.0),
        ("2007", 2_178_489.0, 558_356.0),
        ("2008", 3_922_105.0, 875_881.0),
        ("2009", 4_277_964.0, 972_731.0),
        ("2010", 4_629_277.0, 1_365_691.0),
        ("", 18_684_738.0, 2_448_700.0),
    ];
    const THEIRS: f64 = 500_000.0;
    let fit = genins_mack();
    assert_eq!(fit.mack.chain_ladder.origins[1], Period::year(2002));
    let draws = columns(fit);
    let find = |draws: &[(String, f64, Vec<f64>)], origin: &str| {
        draws
            .iter()
            .find(|(o, _, _)| o == origin)
            .unwrap()
            .2
            .clone()
    };
    let mut failures = Vec::new();
    for (origin, mean_want, sd_want) in table_4 {
        let (mean, sd, error) = moments(&find(&draws, origin));
        // Their standard errors, from this kurtosis at their sample size.
        let combined = error * (1.0 + SIMS as f64 / THEIRS).sqrt();
        if (sd - sd_want).abs() > 5.0 * combined {
            failures.push(format!(
                "{origin}: sd {sd:.1} vs EVW {sd_want} ({:+.2} combined SE)",
                (sd - sd_want) / combined
            ));
        }
        let mean_combined = sd * (1.0 / SIMS as f64 + 1.0 / THEIRS).sqrt();
        if (mean - mean_want).abs() > 5.0 * mean_combined {
            failures.push(format!(
                "{origin}: mean {mean:.1} vs EVW {mean_want} ({:+.2} combined SE)",
                (mean - mean_want) / mean_combined
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    // Uncentred, the total expected reserve is far above theirs. The mean
    // does not depend on the process, so parameter error alone shows it,
    // with its own smaller standard error.
    let uncentred = boot("genins", SigmaInterpolation::Mack, MackProcess::None, false);
    let (mean, sd, _) = moments(&find(&columns(&uncentred), ""));
    let (want, theirs_sd) = (18_684_738.0, 2_448_700.0);
    let combined = (sd * sd / SIMS as f64 + theirs_sd * theirs_sd / THEIRS).sqrt();
    let z = (mean - want) / combined;
    assert!(
        z > 5.0,
        "uncentred total mean {mean:.0} vs EVW {want} ({z:+.2})"
    );
}
