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
//!   total come out 0.4% to 1.2% above R, beyond Monte Carlo error, which
//!   the finding `one-year-bootstrap-vs-merz-wuthrich.md` puts down to the
//!   closed form's linear approximation.
//! * England, Verrall and Wüthrich (2019), Table 4: their bootstrap of
//!   Mack's model, 500,000 simulations on Taylor–Ashe (GenIns) with Mack's
//!   rule for the last sigma, one-year CDR standard deviation per origin and
//!   in total, within five standard errors of the two simulations combined.

use act_reserving::{
    ChainLadder, Development, MackBootstrap, MackBootstrapSegment, MackProcess, OneYearFit,
    OneYearMethod, Period, SigmaInterpolation,
};
use act_validation::{Case, reference, triangle};

/// Simulations per dataset; the tolerances scale with `1 / sqrt` of it.
const SIMS: usize = 20_000;
const SEED: u64 = 20_261_006;

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
    }
    .one_year(
        &triangle(dataset),
        "values",
        &OneYearMethod::ChainLadder(ChainLadder::default()),
    )
    .unwrap_or_else(|e| panic!("{dataset}: {e}"))
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
            let fit = one_year(dataset, sigma_interpolation);
            for (origin, draws) in columns(&fit) {
                let want = merz_wuthrich(&cdr, dataset, method, &origin);
                let (sd, error) = sd_and_error(&draws);
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
    let fit = one_year("genins", SigmaInterpolation::Mack);
    let draws = columns(&fit);
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
