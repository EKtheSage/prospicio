//! The simulated one-year view with a bootstrap tail
//! (`MackBootstrap::one_year` with `tail`, decision 11 of
//! `docs/design/reserving-v02.md`) on RAA, GenIns and ABC.
//!
//! No implementation computes a one-year view with a tail: R ChainLadder's
//! `CDR(MackChainLadder(tri, tail = 1.05))` stops with "tail factors are
//! currently not supported" (0.2.21), and chainladder-python has no CDR.
//! The targets are built from R outputs already committed, under the
//! semantics of decision 11 (the tail step valued one development period
//! after the oldest age, so in the coming year for an annual origin already
//! at the oldest age, which it then closes):
//!
//! * A constant tail `T = 1.05`, its sigma and standard error extrapolated
//!   as R does (both rules for the last sigma). An origin that stays open
//!   has opening and closing ultimates both `T` times those without a
//!   tail, so its CDR is `T` times its CDR without one, simulation by
//!   simulation: its standard deviation is `T` times Merz and Wüthrich's
//!   (R's `CDR(1)S.E.`, `reference/reserving_cdr_r.csv`), which the
//!   untailed bootstrap reproduces (`reserving_one_year_mack.rs`). The
//!   oldest origin's CDR is `U0 - C*`, `C*` drawn from the observed latest
//!   value `C` with a lognormal factor of mean `T` and Mack's standard
//!   error `se`, and process variance `tail_sigma^2 C`: its variance is
//!   `C^2 se^2 + tail_sigma^2 C`, which is R's `Mack.S.E` of that origin
//!   under `MackChainLadder(tri, tail = 1.05)`
//!   (`reference/reserving_tails_r.csv`), and its mean zero. It is
//!   independent of the other origins (it starts from an observed value,
//!   and the factor's draw is its own), so the total's variance is
//!   `T^2` times Merz and Wüthrich's total plus the oldest origin's.
//!   Tolerance: five Monte Carlo standard errors of the simulated standard
//!   deviation, `sd * sqrt((kurtosis - 1) / (4 n))`.
//! * R's estimated tail (`Tail::LogLinear`), refitted on each simulation's
//!   pseudo factors. Run with the Gamma process and with none on one seed,
//!   the two share the pseudo factors and the refitted tail, so the oldest
//!   origin's difference is its tail step's process error alone, standard
//!   deviation `tail_sigma sqrt(C)` with R's `tail_sigma`. Its one-year
//!   standard deviation is its lifetime one (`MackBootstrap::fit`) on
//!   another seed, within five standard errors of the two runs combined.
//!   Its mean is not asserted: the refitted tail is convex in the pseudo
//!   factors (`knowledge/findings/bootstrap-tail-parameter-error.md`).

use prospicio_reserving::{
    ChainLadder, Development, MackBootstrap, MackBootstrapSegment, MackProcess, OneYearFit,
    OneYearMethod, SigmaInterpolation, Tail,
};
use prospicio_validation::{Case, reference, triangle};

/// Simulations per run; the tolerances scale with `1 / sqrt` of it.
const SIMS: usize = 20_000;
const SEED: u64 = 20_261_011;
const DATASETS: [&str; 3] = ["raa", "genins", "abc"];

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

/// R's `quantity` of `dataset` under reference `method`; `arg` is the
/// origin, or empty.
fn r(cases: &[Case], dataset: &str, method: &str, quantity: &str, arg: &str) -> f64 {
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

/// Mack's bootstrap of `dataset` with this process, rule for the last
/// sigma and tail, and the chain ladder with the same tail re-reserved.
fn one_year(
    dataset: &str,
    process: MackProcess,
    rule: SigmaInterpolation,
    tail: Tail,
    seed: u64,
) -> (MackBootstrap, OneYearFit<MackBootstrapSegment>) {
    let b = MackBootstrap {
        n_sims: SIMS,
        seed,
        process,
        development: Development {
            sigma_interpolation: rule,
            ..Default::default()
        },
        tail,
        ..Default::default()
    };
    let method = OneYearMethod::ChainLadder(ChainLadder {
        tail,
        ..Default::default()
    });
    let fit = b
        .one_year(&triangle(dataset), "values", &method)
        .unwrap_or_else(|e| panic!("{dataset}: {e}"));
    (b, fit)
}

/// Each origin's draws of the CDR, labelled by origin, then the total's
/// (labelled `""`).
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

#[test]
fn constant_tail_reconciles_with_merz_wuthrich_and_mack() {
    let cdr = reference("reserving_cdr_r.csv");
    let tails = reference("reserving_tails_r.csv");
    let mut failures = Vec::new();
    for (mw, mack, rule) in [
        ("cdr", "mack_tail_constant", SigmaInterpolation::LogLinear),
        (
            "cdr_sigma_mack",
            "mack_tail_constant_sigma_mack",
            SigmaInterpolation::Mack,
        ),
    ] {
        for dataset in DATASETS {
            let factor = r(&tails, dataset, mack, "tail_factor", "");
            let (_, fit) = one_year(dataset, MackProcess::Gamma, rule, factor.into(), SEED);
            let cl = &fit.bootstrap.mack.chain_ladder;
            for (got, quantity) in [
                (cl.tail.std_err, "tail_std_err"),
                (cl.tail.sigma, "tail_sigma"),
            ] {
                let want = r(&tails, dataset, mack, quantity, "");
                assert!(
                    (got / want - 1.0).abs() < 1e-8,
                    "{dataset} {mack} {quantity}: {got} vs R {want}"
                );
            }
            let oldest = cl.origins[0].to_string();
            let c = cl.latest[0];
            let oldest_se = r(&tails, dataset, mack, "se", &oldest);
            let own = (c * c * cl.tail.std_err.powi(2) + cl.tail.sigma.powi(2) * c).sqrt();
            assert!(
                (own / oldest_se - 1.0).abs() < 1e-8,
                "{dataset} {mack}: oldest {own} vs R {oldest_se}"
            );
            for (origin, draws) in columns(&fit) {
                let want = if origin == oldest {
                    oldest_se
                } else if origin.is_empty() {
                    let total = r(&cdr, dataset, mw, "total_one_year_se", "");
                    ((factor * total).powi(2) + oldest_se.powi(2)).sqrt()
                } else {
                    factor * r(&cdr, dataset, mw, "one_year_se", &origin)
                };
                let (mean, sd, error) = moments(&draws);
                let label = format!("{mack} {dataset} {origin}");
                eprintln!(
                    "{label}: sd {sd:.1} vs {want:.1} ({:+.2} SE), mean / sd {:+.4}",
                    (sd - want) / error,
                    mean / sd
                );
                if (sd - want).abs() > 5.0 * error {
                    failures.push(format!(
                        "{label}: sd {sd:.1} vs {want:.1} ({:+.2} Monte Carlo SE)",
                        (sd - want) / error
                    ));
                }
                // Merz and Wüthrich's mean CDR is zero, and so is the
                // closed origin's.
                if mean.abs() > 5.0 * sd / (SIMS as f64).sqrt() {
                    failures.push(format!("{label}: mean {mean:.1}, sd {sd:.1}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn refitted_tail_step_has_mack_process_error() {
    let tails = reference("reserving_tails_r.csv");
    let mut failures = Vec::new();
    for (method, rule) in [
        ("mack_tail_loglinear", SigmaInterpolation::LogLinear),
        ("mack_tail_loglinear_sigma_mack", SigmaInterpolation::Mack),
    ] {
        for dataset in DATASETS {
            let run = |process| one_year(dataset, process, rule, Tail::LogLinear, SEED);
            let (b, full) = run(MackProcess::Gamma);
            let (_, parameter) = run(MackProcess::None);
            let cl = &full.bootstrap.mack.chain_ladder;
            let sigma = r(&tails, dataset, method, "tail_sigma", "");
            assert!(
                (cl.tail.sigma / sigma - 1.0).abs() < 1e-8,
                "{dataset} {method}: tail sigma {} vs R {sigma}",
                cl.tail.sigma
            );
            let label = format!("{method} {dataset}");
            let (with, without) = (&columns(&full)[0].1, &columns(&parameter)[0].1);
            let process: Vec<f64> = with.iter().zip(without).map(|(a, b)| a - b).collect();
            let (mean, sd, error) = moments(&process);
            let want = sigma * cl.latest[0].sqrt();
            eprintln!(
                "{label}: process sd {sd:.1} vs {want:.1} ({:+.2} SE)",
                (sd - want) / error
            );
            if (sd - want).abs() > 5.0 * error {
                failures.push(format!(
                    "{label}: process sd {sd:.1} vs R {want:.1} ({:+.2} Monte Carlo SE)",
                    (sd - want) / error
                ));
            }
            if mean.abs() > 5.0 * sd / (SIMS as f64).sqrt() {
                failures.push(format!("{label}: mean process error {mean:.1}"));
            }

            // The oldest origin's remaining development is the tail step,
            // in the year: its one-year view is its lifetime view.
            let lifetime = MackBootstrap {
                seed: SEED + 1,
                ..b
            }
            .fit(&triangle(dataset), "values")
            .unwrap();
            let n = lifetime.reserves.n_components();
            let reserve: Vec<f64> = lifetime
                .reserves
                .draw_matrix()
                .chunks(n)
                .map(|r| r[0])
                .collect();
            let (_, life, life_error) = moments(&reserve);
            let (_, one, one_error) = moments(with);
            let combined = (life_error.powi(2) + one_error.powi(2)).sqrt();
            eprintln!(
                "{label}: oldest one-year sd {one:.1} vs lifetime {life:.1} ({:+.2} SE)",
                (one - life) / combined
            );
            if (one - life).abs() > 5.0 * combined {
                failures.push(format!(
                    "{label}: oldest one-year sd {one:.1} vs lifetime {life:.1} ({:+.2} SE)",
                    (one - life) / combined
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
