//! The simulated one-year view with Mack's process
//! (`MackBootstrap::one_year`, decision 8 of `docs/design/reserving-v02.md`)
//! reconciled with Merz and Wüthrich on RAA, GenIns and ABC.
//!
//! * The reconciliation: with the volume-weighted chain ladder and no tail,
//!   each origin's and the total standard deviation of the simulated claims
//!   development result match R ChainLadder's
//!   `CDR(MackChainLadder(tri, est.sigma = ...))` `CDR(1)S.E.`
//!   (`reference/reserving_cdr_r.csv`), for both ways of filling the last
//!   sigma, within five Monte Carlo standard errors of the simulated
//!   standard deviation, `sd * sqrt((kurtosis - 1) / (4 n))`. The reference
//!   is independent of the simulation: Merz and Wüthrich's closed form.
//!   Five standard errors is about 2.5% (GenIns) to 5% (RAA's young
//!   origins) at `SIMS`; with 200,000 simulations RAA's young origins and
//!   total come out 0.4% to 1.3% above R, beyond Monte Carlo error.
//! * Why: `exact_covariance` gives the CDR's covariance exactly under the
//!   model the bootstrap samples. Linearised it is R's closed form to
//!   rounding, and exact it is at most 0.09% above, so Merz and Wüthrich's
//!   approximation (their Appendix A) is not the gap. The uncentred
//!   residuals are: they bias RAA's pseudo factors upwards (the first by
//!   14%), and the exact standard deviations under the bootstrap's own
//!   pseudo factors are 0.48% to 1.29% above R where the 200,000
//!   simulations are, within Monte Carlo error of them
//!   (`simulation_matches_the_exact_moments`, ignored: run it with
//!   `--release --ignored`).
//!   The reconciliation is of the standard deviation only: the mean CDR is
//!   biased by the uncentred residuals, which `MEAN_BIAS` pins. The unit
//!   test `centred_residuals_remove_the_mean_bias` checks that centring
//!   removes it.
//! * England, Verrall and Wüthrich (2019), Table 4: their bootstrap of
//!   Mack's model, 500,000 simulations on Taylor–Ashe (GenIns) with Mack's
//!   rule for the last sigma, one-year CDR standard deviation per origin and
//!   in total, within five standard errors of the two simulations combined.

use prospicio_reserving::{
    ChainLadder, Development, Mack, MackBootstrap, MackBootstrapSegment, MackProcess, OneYearFit,
    OneYearMethod, Period, SigmaInterpolation,
};
use prospicio_validation::{Case, reference, triangle};
use std::sync::OnceLock;

/// Simulations per dataset; the tolerances scale with `1 / sqrt` of it.
const SIMS: usize = 20_000;
const SEED: u64 = 20_261_006;

/// The total CDR's mean over its standard deviation with the log-linear
/// last sigma, a seed-pinned regression, not a reference: EVW's uncentred
/// residuals bias the pseudo factors (`MackBootstrap::centre_residuals`),
/// so the mean is not Merz and Wuthrich's zero.
const MEAN_BIAS: [(&str, f64); 3] = [("raa", -0.2142), ("genins", -0.0380), ("abc", 0.1755)];

fn one_year(
    dataset: &str,
    sigma_interpolation: SigmaInterpolation,
) -> OneYearFit<MackBootstrapSegment> {
    MackBootstrap {
        n_sims: SIMS,
        seed: SEED,
        process: MackProcess::Gamma,
        development: Development {
            sigma_interpolation,
            ..Default::default()
        },
        centre_residuals: false,
    }
    .one_year(
        &triangle(dataset),
        "values",
        &OneYearMethod::ChainLadder(ChainLadder::default()),
    )
    .unwrap_or_else(|e| panic!("{dataset}: {e}"))
}

/// GenIns with Mack's rule for the last sigma, which both tests check:
/// simulated once.
fn genins_mack() -> &'static OneYearFit<MackBootstrapSegment> {
    static FIT: OnceLock<OneYearFit<MackBootstrapSegment>> = OnceLock::new();
    FIT.get_or_init(|| one_year("genins", SigmaInterpolation::Mack))
}

/// Standard deviation of `x` and the Monte Carlo standard error of that
/// estimate, `sd * sqrt((kurtosis - 1) / (4 n))`.
fn sd_and_error(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let m2 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let m4 = x.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n;
    if m2 == 0.0 {
        return (0.0, 0.0);
    }
    let sd = m2.sqrt();
    (sd, sd * ((m4 / (m2 * m2) - 1.0) / (4.0 * n)).sqrt())
}

/// Each origin's draws of the CDR, then the total's (labelled `""`).
fn columns(fit: &OneYearFit<MackBootstrapSegment>) -> Vec<(String, Vec<f64>)> {
    let n = fit.cdr.n_components();
    let rows = || fit.cdr.draw_matrix().chunks(n);
    let mut out: Vec<(String, Vec<f64>)> = fit
        .bootstrap
        .mack
        .chain_ladder
        .origins
        .iter()
        .enumerate()
        .map(|(j, p)| (p.to_string(), rows().map(|r| r[j]).collect()))
        .collect();
    out.push(("".into(), rows().map(|r| r.iter().sum()).collect()));
    out
}

/// R's `CDR(1)S.E.` of `dataset` under reference `method`, for an origin or
/// (`""`) the total.
fn merz_wuthrich(cases: &[Case], dataset: &str, method: &str, origin: &str) -> f64 {
    let (quantity, arg) = if origin.is_empty() {
        ("total_one_year_se", "")
    } else {
        ("one_year_se", origin)
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

#[test]
fn mack_process_reconciles_with_merz_wuthrich() {
    let cdr = reference("reserving_cdr_r.csv");
    let mut failures = Vec::new();
    for (method, sigma_interpolation) in [
        ("cdr", SigmaInterpolation::LogLinear),
        ("cdr_sigma_mack", SigmaInterpolation::Mack),
    ] {
        for dataset in ["raa", "genins", "abc"] {
            let owned;
            let fit = if (dataset, sigma_interpolation) == ("genins", SigmaInterpolation::Mack) {
                genins_mack()
            } else {
                owned = one_year(dataset, sigma_interpolation);
                &owned
            };
            for (origin, draws) in columns(fit) {
                let want = merz_wuthrich(&cdr, dataset, method, &origin);
                let (sd, error) = sd_and_error(&draws);
                if let (true, Some(&(_, bias))) = (
                    origin.is_empty() && method == "cdr",
                    MEAN_BIAS.iter().find(|(d, _)| *d == dataset),
                ) {
                    let ratio = draws.iter().sum::<f64>() / SIMS as f64 / sd;
                    if (ratio - bias).abs() > 0.001 {
                        failures.push(format!("{dataset}: mean / sd {ratio:.4}, was {bias}"));
                    }
                }
                if want == 0.0 {
                    // The oldest origin has nothing left to develop.
                    if sd != 0.0 {
                        failures.push(format!("{method} {dataset} {origin}: sd {sd}, want 0"));
                    }
                    continue;
                }
                if (sd - want).abs() > 5.0 * error {
                    failures.push(format!(
                        "{method} {dataset} {origin}: sd {sd:.1} vs Merz-Wuthrich {want:.1} \
                         ({:+.2} Monte Carlo SE)",
                        (sd - want) / error
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The pseudo factors' mean and variance: `(f_k, sigma_k^2 / S_k)` for
/// Merz and Wüthrich's conditional resampling, the bootstrap's otherwise.
#[derive(Clone, Copy, PartialEq)]
enum Factors {
    MerzWuthrich,
    Bootstrap { centred: bool },
}

/// The exact covariance of the closing ultimates (so of the CDR) one year
/// on, under the model the bootstrap samples, for an annual triangle with
/// the volume-weighted chain ladder and no tail. Origin `i` (latest age
/// `k`) has next value `Z_i` with mean `E[f*_k] C_i` and variance
/// `Var[f*_k] C_i^2 + sigma_k^2 C_i`, the `Z_i` independent (each origin
/// uses its own pseudo factor, drawn from its own residuals, and its own
/// process draw). The refitted factor at `k` is `(A_k + Z_i) / S1_k`, `A_k`
/// and `S_k` the sums over the older origins at `k + 1` and `k`, and
/// `S1_k = S_k + C_i`; so the closing ultimate of origin `l`,
/// `Z_l` times the refitted factors of the older origins, is a product of
/// independent factors each linear in one `Z`, and its second moments are
/// products of means. With `linear`, the first-order (delta method)
/// covariance instead: Merz and Wüthrich's closed form, whose Appendix A
/// (A.1) replaces each product `prod(1 + a_j) - 1` by `sum(a_j)`.
/// The pseudo factors of the bootstrap have mean
/// `f_k + m sigma_k sum(sqrt(C)) / S_k` (`m` the uncentred pool's mean;
/// `f_k` when centred) and variance `(1 - m^2) sigma_k^2 / S_k` either way
/// (the uncentred pool's mean square is 1, and centring shifts the pool
/// without rescaling it). The process shape does not enter: the second
/// moments need only the first two moments of each `Z`.
fn exact_covariance(dataset: &str, factors: Factors, linear: bool) -> Vec<Vec<f64>> {
    exact_moments(dataset, factors, linear).1
}

/// The mean closing ultimates and their covariance, as `exact_covariance`
/// describes. With `Factors::MerzWuthrich` the means are the opening
/// chain-ladder ultimates: each refitted factor's mean is `f_k`, since
/// `A_k = f_k S_k`. The oldest origin, which has nothing to develop, has
/// mean 1 under every `Factors`, so differences of means are its CDR's.
fn exact_moments(dataset: &str, factors: Factors, linear: bool) -> (Vec<f64>, Vec<Vec<f64>>) {
    let tri = triangle(dataset);
    let fit = MackBootstrap {
        n_sims: 1,
        ..Default::default()
    }
    .one_year(
        &tri,
        "values",
        &OneYearMethod::ChainLadder(ChainLadder::default()),
    )
    .unwrap();
    let pool: Vec<f64> = fit
        .bootstrap
        .residuals
        .iter()
        .copied()
        .filter(|r| !r.is_nan())
        .collect();
    let m = pool.iter().sum::<f64>() / pool.len() as f64;
    let dev = &fit.bootstrap.mack.chain_ladder.development;
    let (f, sigma) = (&dev.ldf, &dev.sigma);
    let n = fit.bootstrap.mack.chain_ladder.origins.len();
    let c = |o: usize, d: usize| tri.get(0, 0, o, d).unwrap();
    let latest = |o: usize| n - 1 - o;

    // Mean and variance of each origin's next value, and the refitted
    // factor at its latest age, `a + b Z`.
    let (mut ez, mut vz, mut ab) = (vec![0.0; n], vec![0.0; n], vec![(1.0, 0.0); n]);
    for o in 1..n {
        let k = latest(o);
        let s: f64 = (0..o).map(|i| c(i, k)).sum();
        let a: f64 = (0..o).map(|i| c(i, k + 1)).sum();
        let sqrt_sum: f64 = (0..o).map(|i| c(i, k).sqrt()).sum();
        let (mean, var) = match factors {
            Factors::MerzWuthrich => (f[k], sigma[k].powi(2) / s),
            Factors::Bootstrap { centred } => {
                // Centring removes the bias from the mean only: the
                // centred pool's mean square is still `1 - m^2`.
                let shift = if centred { 0.0 } else { m };
                (
                    f[k] + shift * sigma[k] * sqrt_sum / s,
                    (1.0 - m * m) * sigma[k].powi(2) / s,
                )
            }
        };
        let ci = c(o, k);
        ez[o] = mean * ci;
        vz[o] = var * ci * ci + sigma[k].powi(2) * ci;
        ab[o] = (a / (s + ci), 1.0 / (s + ci));
    }
    // Origin `l`'s closing ultimate as factors `(a, b)` in each `Z_o`.
    let factor = |l: usize, o: usize| match o.cmp(&l) {
        std::cmp::Ordering::Equal => (0.0, 1.0),
        std::cmp::Ordering::Less if o >= 1 => ab[o],
        _ => (1.0, 0.0),
    };
    let mean = |l: usize| {
        (1..n)
            .map(|o| {
                let (a, b) = factor(l, o);
                a + b * ez[o]
            })
            .product::<f64>()
    };
    // The oldest origin has no next value: every factor is 1, so its
    // covariances come out zero.
    let covariance = |l: usize, j: usize| -> f64 {
        if linear {
            (1..n)
                .map(|o| {
                    let ((al, bl), (aj, bj)) = (factor(l, o), factor(j, o));
                    mean(l) * bl / (al + bl * ez[o]) * mean(j) * bj / (aj + bj * ez[o]) * vz[o]
                })
                .sum()
        } else {
            (1..n)
                .map(|o| {
                    let ((al, bl), (aj, bj)) = (factor(l, o), factor(j, o));
                    al * aj + (al * bj + aj * bl) * ez[o] + bl * bj * (ez[o].powi(2) + vz[o])
                })
                .product::<f64>()
                - mean(l) * mean(j)
        }
    };
    (
        (0..n).map(mean).collect(),
        (0..n)
            .map(|l| (0..n).map(|j| covariance(l, j)).collect())
            .collect(),
    )
}

/// Each origin's standard deviation from a covariance, then the total's.
fn standard_deviations(cov: &[Vec<f64>]) -> Vec<f64> {
    let mut out: Vec<f64> = (0..cov.len()).map(|i| cov[i][i].sqrt()).collect();
    out.push(cov.iter().flatten().sum::<f64>().sqrt());
    out
}

/// The origins' labels as `columns` gives them, then `""` for the total.
fn labels(dataset: &str) -> Vec<String> {
    let mut out: Vec<String> = Mack::default()
        .fit(&triangle(dataset), "values")
        .unwrap()
        .chain_ladder
        .origins
        .iter()
        .map(|p| p.to_string())
        .collect();
    out.push(String::new());
    out
}

#[test]
fn merz_wuthrich_is_the_linearised_exact_msep() {
    // Merz and Wüthrich (2008), Appendix A, approximate the product terms
    // of the conditional MSEP by sums (their (A.1), a lower bound). The
    // first-order covariance of `exact_covariance` reproduces R's
    // `CDR(1)S.E.` to rounding, which checks the model it encodes; the
    // exact one is at most 0.09% above on RAA, GenIns and ABC, so the
    // neglected terms do not explain RAA's 0.4% to 1.3% gap at 200,000
    // simulations (`uncentred_residuals_raise_raa_young_origins`).
    let cdr = reference("reserving_cdr_r.csv");
    for dataset in ["raa", "genins", "abc"] {
        let linear = standard_deviations(&exact_covariance(dataset, Factors::MerzWuthrich, true));
        let exact = standard_deviations(&exact_covariance(dataset, Factors::MerzWuthrich, false));
        for (k, origin) in labels(dataset).iter().enumerate() {
            let want = merz_wuthrich(&cdr, dataset, "cdr", origin);
            if want == 0.0 {
                assert_eq!(exact[k], 0.0, "{dataset} {origin}");
                continue;
            }
            assert!(
                (linear[k] / want - 1.0).abs() < 1e-9,
                "{dataset} {origin}: linearised {} vs R {want}",
                linear[k]
            );
            let ratio = exact[k] / want;
            assert!(
                (1.0 - 1e-12..1.001).contains(&ratio),
                "{dataset} {origin}: exact / R {ratio}"
            );
        }
    }
    // RAA's youngest origin and total, pinned.
    let raa = standard_deviations(&exact_covariance("raa", Factors::MerzWuthrich, false));
    assert!((raa[9] - 23_630.22).abs() < 0.01, "{}", raa[9]);
    assert!((raa[10] - 25_185.83).abs() < 0.01, "{}", raa[10]);
}

#[test]
fn uncentred_residuals_raise_raa_young_origins() {
    // EVW's uncentred residuals bias RAA's pseudo factors upwards by
    // `m sigma_k sum(sqrt(C)) / S_k` (pool mean m = 0.14), the first by 14%
    // of itself, which raises the means of the next values and refitted
    // factors and so the spread of their products: exactly,
    // 1988 to 1990 are 0.48%, 0.93% and 1.20% above Merz and Wüthrich and
    // the total 1.29%, while centred residuals give 0.46% below to 0.04%
    // above (the pool's variance, 1 - m^2, shrinks the parameter error):
    // RAA 1982 0.46% below, as uncentred, and the total 0.01% below.
    // Merz and Wüthrich's exact values are never below R, so the centred
    // pins separate the two models. GenIns and ABC, whose pool means are
    // 0.01 and -0.06, move by 0.13% at most.
    let cdr = reference("reserving_cdr_r.csv");
    let ratios = |dataset: &str, centred| -> Vec<f64> {
        let sd = standard_deviations(&exact_covariance(
            dataset,
            Factors::Bootstrap { centred },
            false,
        ));
        labels(dataset)
            .iter()
            .zip(sd)
            .map(
                |(origin, sd)| match merz_wuthrich(&cdr, dataset, "cdr", origin) {
                    0.0 => 1.0,
                    want => sd / want,
                },
            )
            .collect()
    };
    let raa = ratios("raa", false);
    for (k, want) in [(7, 1.004826), (8, 1.009296), (9, 1.012032), (10, 1.012934)] {
        assert!((raa[k] - want).abs() < 1e-6, "raa {k}: {}", raa[k]);
    }
    let raa = ratios("raa", true);
    for (k, want) in [(1, 0.995392), (9, 0.999997), (10, 0.999866)] {
        assert!((raa[k] - want).abs() < 1e-6, "raa centred {k}: {}", raa[k]);
    }
    for dataset in ["raa", "genins", "abc"] {
        assert!(
            ratios(dataset, true)
                .iter()
                .all(|r| (0.9953..1.0004).contains(r)),
            "{dataset} centred"
        );
        if dataset != "raa" {
            assert!(
                ratios(dataset, false)
                    .iter()
                    .all(|r| (r - 1.0).abs() < 0.0014),
                "{dataset} uncentred"
            );
        }
    }
}

#[test]
fn exact_mean_cdr_is_the_pool_bias() {
    // The exact total mean CDR over its exact SD, uncentred, is the
    // bootstrap's bias that `MEAN_BIAS` pins from 20,000 simulations, within
    // three Monte Carlo standard errors (about `1 / sqrt(SIMS)`); centred it
    // is zero. Under Merz and Wüthrich's factors the mean closing ultimates
    // are the opening chain-ladder ones, which checks the means.
    for (dataset, want) in [("raa", -0.2041), ("genins", -0.0344), ("abc", 0.1677)] {
        let ultimate = Mack::default()
            .fit(&triangle(dataset), "values")
            .unwrap()
            .chain_ladder
            .ultimate;
        let (opening, _) = exact_moments(dataset, Factors::MerzWuthrich, false);
        for (o, (u, cl)) in opening.iter().zip(&ultimate).enumerate().skip(1) {
            assert!((u / cl - 1.0).abs() < 1e-12, "{dataset} {o}: {u} vs {cl}");
        }
        for centred in [false, true] {
            let (closing, cov) = exact_moments(dataset, Factors::Bootstrap { centred }, false);
            let sd = standard_deviations(&cov)[closing.len()];
            let mean: f64 = opening.iter().zip(&closing).map(|(u0, u1)| u0 - u1).sum();
            let ratio = mean / sd;
            if centred {
                assert!(ratio.abs() < 1e-9, "{dataset} centred: {ratio}");
            } else {
                assert!((ratio - want).abs() < 5e-5, "{dataset}: {ratio}");
                let (_, simulated) = MEAN_BIAS.iter().find(|(d, _)| *d == dataset).unwrap();
                assert!(
                    (ratio - simulated).abs() < 3.0 / (SIMS as f64).sqrt(),
                    "{dataset}: exact {ratio} vs simulated {simulated}"
                );
            }
        }
    }
}

#[test]
#[ignore = "200,000 simulations per configuration: minutes"]
fn simulation_matches_the_exact_moments() {
    // At 200,000 simulations the bootstrap's standard deviations are within
    // four Monte Carlo standard errors of `exact_covariance` under its own
    // pseudo factors, uncentred and centred, Gamma and normal process, on
    // RAA, GenIns and ABC: the RAA gap to Merz and Wüthrich is the
    // uncentred residuals, not Monte Carlo error or the process shape.
    const N: usize = 200_000;
    let mut failures = Vec::new();
    for dataset in ["raa", "genins", "abc"] {
        for (process, centred) in [
            (MackProcess::Gamma, false),
            (MackProcess::Normal, false),
            (MackProcess::Gamma, true),
        ] {
            let fit = MackBootstrap {
                n_sims: N,
                seed: SEED,
                process,
                development: Development::default(),
                centre_residuals: centred,
            }
            .one_year(
                &triangle(dataset),
                "values",
                &OneYearMethod::ChainLadder(ChainLadder::default()),
            )
            .unwrap();
            let exact = standard_deviations(&exact_covariance(
                dataset,
                Factors::Bootstrap { centred },
                false,
            ));
            for (k, (origin, draws)) in columns(&fit).into_iter().enumerate() {
                let (sd, error) = sd_and_error(&draws);
                if (sd - exact[k]).abs() > 4.0 * error {
                    failures.push(format!(
                        "{dataset} {process:?} centred {centred} {origin}: {sd:.1} vs exact \
                         {:.1} ({:+.2} SE)",
                        exact[k],
                        (sd - exact[k]) / error
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn evw_2019_table_4() {
    // England, Verrall and Wüthrich (2019), Table 4, "One-Year CDR Std
    // Dev": their bootstrap of Mack's model on Taylor-Ashe, 500,000
    // simulations, Mack's rule for the last sigma (their Table 1: 21.13).
    // Accident years 2 to 10 are origins 2002 to 2010.
    let table_4 = [
        ("2002", 75_502.0),
        ("2003", 105_505.0),
        ("2004", 79_900.0),
        ("2005", 235_182.0),
        ("2006", 318_385.0),
        ("2007", 360_974.0),
        ("2008", 629_558.0),
        ("2009", 588_355.0),
        ("2010", 1_030_505.0),
        ("", 1_778_428.0),
    ];
    const THEIRS: f64 = 500_000.0;
    let fit = genins_mack();
    let draws = columns(fit);
    assert_eq!(
        fit.bootstrap.mack.chain_ladder.origins[1],
        Period::year(2002)
    );
    for (origin, want) in table_4 {
        let x = &draws.iter().find(|(o, _)| o == origin).unwrap().1;
        let (sd, error) = sd_and_error(x);
        // Their standard error, from this kurtosis at their sample size.
        let combined = error * (1.0 + SIMS as f64 / THEIRS).sqrt();
        assert!(
            (sd - want).abs() < 5.0 * combined,
            "{origin}: {sd:.1} vs EVW {want} ({:+.2} combined SE)",
            (sd - want) / combined
        );
    }
}
