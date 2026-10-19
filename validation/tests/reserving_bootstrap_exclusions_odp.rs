//! The ODP bootstrap under the link ratio exclusions `n_periods`,
//! `drop_high` and `drop_low` (decision 14 of `docs/design/reserving-v02.md`)
//! against chainladder-python 0.10.1's `BootstrapODPSample` on RAA and
//! GenIns, under `n_periods = 3`, under one high and one low dropped and
//! under both (`reference/reserving_bootstrap_exclusions_python.csv`, from
//! `scripts/reserving_bootstrap_exclusions_python.py`).
//!
//! * Deterministic pieces, compared to 1e-9 (an absolute 1e-8 on the
//!   residuals of the cells that fit exactly): the fitted incremental
//!   values, which cells after the first age have a residual, the unscaled
//!   residuals (the first age's included) and the scale `phi` under
//!   prospicio's rule. chainladder-python gives the first age the weight of
//!   link 0, so dropping link `(o, 0)` also drops that origin's first-age
//!   residual; prospicio keeps it, so its `phi` is the reference's `phi`
//!   (the sum over the same cells) and not chainladder-python's `scale_`.
//! * The simulated mean total reserve, within Monte Carlo error. Both sides
//!   are parameter error alone (no process error) at 20,000 simulations;
//!   the reference's samples are refitted with the observed exclusions as
//!   fixed positions, as prospicio keeps them. The tolerance is five
//!   standard errors of the difference, `sqrt(se_prospicio^2 +
//!   se_reference^2)`, each `sd / sqrt(n)`. The two pools differ by the
//!   first-age residuals of the origins whose link 0 is dropped, so the
//!   tolerance also has to cover that: the observed z-scores are printed.
//!   One setting (RAA under the window and both drops) has heavy tails and
//!   no usable standard error of its mean: there the medians are compared
//!   instead, within the sum in quadrature of the half-widths of their
//!   five-sigma intervals from the order statistics (all settings compare
//!   their medians so).
//!   The standard deviation is not compared: the conventions differ
//!   (centring, dropped zeros, the hat matrix).

use prospicio_prob::Distribution;
use prospicio_reserving::{OdpBootstrap, OdpBootstrapFit, ProcessDistribution};
use prospicio_validation::{Case, check, reference, triangle};

const SIMS: usize = 20_000;
const SEED: u64 = 20_261_019;
/// Standard errors of the difference of two simulated means.
const Z: f64 = 5.0;

/// Settings whose reserve has outliers of 1e8 on a median of 4e4, and so no
/// usable standard error of its mean (see the script's docstring): only the
/// median is compared.
const HEAVY_TAILED: [(&str, &str); 1] = [("raa", "n_periods=3;drop_high=1;drop_low=1")];

fn bootstrap(method: &str, n_sims: usize) -> OdpBootstrap {
    let get = |name: &str| {
        method
            .split(';')
            .find_map(|p| p.strip_prefix(name)?.strip_prefix('='))
    };
    OdpBootstrap {
        n_sims,
        seed: SEED,
        process: ProcessDistribution::None,
        n_periods: get("n_periods").map(|v| v.parse().unwrap()),
        drop_high: get("drop_high").map_or(0, |v| v.parse().unwrap()),
        drop_low: get("drop_low").map_or(0, |v| v.parse().unwrap()),
        ..Default::default()
    }
}

#[test]
fn matches_chainladder_python_deterministic_pieces() {
    let cases: Vec<Case> = reference("reserving_bootstrap_exclusions_python.csv")
        .into_iter()
        .filter(|c| {
            matches!(
                c.get("quantity"),
                "fitted" | "has_residual" | "residual" | "phi"
            )
        })
        .collect();
    let mut fits: std::collections::HashMap<(String, String), OdpBootstrapFit> = Default::default();
    check(&cases, |case| {
        let (dataset, method) = (case.get("dataset"), case.get("method"));
        let fit = fits
            .entry((dataset.to_string(), method.to_string()))
            .or_insert_with(|| {
                bootstrap(method, 10)
                    .fit(&triangle(dataset), "values")
                    .unwrap_or_else(|e| panic!("{dataset} {method}: {e}"))
            });
        let quantity = case.get("quantity");
        if quantity == "phi" {
            return Some(fit.scale);
        }
        let nd = fit.chain_ladder.development.development.len();
        let (origin, age) = case.get("arg").split_once(';').unwrap();
        let o = fit
            .chain_ladder
            .origins
            .iter()
            .position(|p| p.to_string() == origin)?;
        let d = fit
            .chain_ladder
            .development
            .development
            .iter()
            .position(|&a| a == age.parse::<u32>().unwrap())?;
        let i = o * nd + d;
        match quantity {
            "fitted" => Some(fit.fitted[i]),
            // A cell whose fitted value is zero has no residual either way.
            "has_residual" => Some(f64::from(
                !fit.residuals[i].is_nan() || fit.fitted[i] == 0.0,
            )),
            "residual" => {
                let n = fit.fitted.iter().filter(|m| !m.is_nan()).count() as f64;
                let p = (fit.chain_ladder.origins.len() + nd - 1) as f64;
                Some(fit.residuals[i] / (n / (n - p)).sqrt())
            }
            _ => None,
        }
    });
}

#[test]
fn simulated_mean_reserve_matches_chainladder_python() {
    let cases: Vec<Case> = reference("reserving_bootstrap_exclusions_python.csv")
        .into_iter()
        .filter(|c| c.get("quantity") == "mean_reserve")
        .collect();
    assert_eq!(cases.len(), 8);
    let rows = reference("reserving_bootstrap_exclusions_python.csv");
    let mut failures = Vec::new();
    for case in &cases {
        let (dataset, method) = (case.get("dataset"), case.get("method"));
        let find = |quantity: &str| {
            rows.iter()
                .find(|c| {
                    c.get("dataset") == dataset
                        && c.get("method") == method
                        && c.get("quantity") == quantity
                })
                .and_then(|c| c.number("expected"))
                .unwrap()
        };
        let (want, want_se) = (find("mean_reserve"), find("mean_reserve_se"));
        let fit = bootstrap(method, SIMS)
            .fit(&triangle(dataset), "values")
            .unwrap();
        let total = fit.reserves.total();
        let (mean, se) = (total.mean(), total.std_dev() / (SIMS as f64).sqrt());
        let error = (se * se + want_se * want_se).sqrt();
        let z = (mean - want) / error;
        println!(
            "{dataset} {method}: mean {mean:.1} vs {want:.1}, standard error of the difference \
             {error:.1} ({z:+.2})"
        );
        if HEAVY_TAILED.contains(&(dataset, method)) {
            println!("  heavy tails: the mean is not compared");
        } else if z.abs() > Z {
            failures.push(format!(
                "{dataset} {method}: mean {mean:.1} vs {want:.1} ({z:+.2} standard errors)"
            ));
        }

        // The median, with the half-width of its five-sigma interval from
        // the order statistics on each side, combined in quadrature.
        let n = fit.reserves.n_components();
        let mut sums: Vec<f64> = fit
            .reserves
            .draw_matrix()
            .chunks(n)
            .map(|r| r.iter().sum())
            .collect();
        sums.sort_by(f64::total_cmp);
        let q = |p: f64| sums[((p * SIMS as f64) as usize).min(SIMS - 1)];
        let band = 2.5 / (SIMS as f64).sqrt();
        let (median, half) = (q(0.5), (q(0.5 + band) - q(0.5 - band)) / 2.0);
        let (want_median, want_half) = (find("median_reserve"), find("median_halfwidth"));
        let tolerance = (half * half + want_half * want_half).sqrt();
        println!(
            "{dataset} {method}: median {median:.1} vs {want_median:.1}, tolerance {tolerance:.1}"
        );
        if (median - want_median).abs() > tolerance {
            failures.push(format!(
                "{dataset} {method}: median {median:.1} vs {want_median:.1} (tolerance \
                 {tolerance:.1})"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
