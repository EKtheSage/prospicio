//! The simulated one-year view (`OdpBootstrap::one_year`, decision 8 of
//! `docs/design/reserving-v02.md`) on RAA, GenIns and ABC.
//!
//! * Against Merz and Wüthrich: the standard deviation of each origin's and
//!   the total claims development result of the volume-weighted chain
//!   ladder, against R ChainLadder's `CDR(MackChainLadder(tri))`
//!   `CDR(1)S.E.` (`reference/reserving_cdr_r.csv`). The two differ by
//!   their process model, not by Monte Carlo error: the ODP bootstrap's
//!   variance is a constant `phi` times the mean, Mack's is `sigma_k^2`
//!   times the cumulative value. Re-reserving with Mack's own process
//!   (England, Verrall and Wüthrich 2019, Appendix 1) reproduces
//!   Merz–Wüthrich within 0.2%. So each check is against R's value times
//!   the measured ratio, `GAP`, with a tolerance of five Monte Carlo
//!   standard errors of the simulated standard deviation, and
//!   `knowledge/findings/one-year-bootstrap-vs-merz-wuthrich.md` records
//!   and explains the ratios.
//! * Against the lifetime ODP bootstrap: an origin with one cell left to
//!   develop has a one-year CDR that is its whole run-off, so its standard
//!   deviation is R `BootChainLadder`'s for that origin
//!   (`reference/reserving_bootstrap_r.csv`); every other origin's is below
//!   it.
//! * England, Verrall and Wüthrich (2019), Table 2: their Merz–Wüthrich and
//!   Mack numbers on Taylor–Ashe (GenIns). Their simulated one-year view
//!   (Table 4) bootstraps Mack's model, not the ODP, so it is not compared.

use act_reserving::{
    ChainLadder, Development, Mack, OdpBootstrap, OneYearFit, OneYearMethod, Period,
    ProcessDistribution, SigmaInterpolation,
};
use act_validation::{Case, reference, triangle};

/// Simulations per dataset; the tolerances scale with `1 / sqrt` of it.
const SIMS: usize = 20_000;
const SEED: u64 = 20_261_006;

/// Measured ratio of the simulated one-year standard deviation to R's
/// Merz–Wüthrich `CDR(1)S.E.`, per dataset and origin (`""` for the total),
/// with `SIMS` simulations from `SEED`.
const GAP: &[(&str, &str, f64)] = &[
    ("raa", "1982", 4.8961),
    ("raa", "1983", 1.7909),
    ("raa", "1984", 3.7417),
    ("raa", "1985", 1.1629),
    ("raa", "1986", 1.0217),
    ("raa", "1987", 1.6753),
    ("raa", "1988", 0.6539),
    ("raa", "1989", 0.7822),
    ("raa", "1990", 0.3032),
    ("raa", "", 0.4579),
    ("genins", "2002", 1.5931),
    ("genins", "2003", 1.8667),
    ("genins", "2004", 2.2162),
    ("genins", "2005", 0.8206),
    ("genins", "2006", 0.7015),
    ("genins", "2007", 0.8225),
    ("genins", "2008", 0.8051),
    ("genins", "2009", 1.0657),
    ("genins", "2010", 1.0153),
    ("genins", "", 1.0234),
    ("abc", "1978", 5.2849),
    ("abc", "1979", 5.9505),
    ("abc", "1980", 2.5659),
    ("abc", "1981", 1.5402),
    ("abc", "1982", 1.6324),
    ("abc", "1983", 0.8158),
    ("abc", "1984", 0.9978),
    ("abc", "1985", 0.9030),
    ("abc", "1986", 0.8033),
    ("abc", "1987", 0.7962),
    ("abc", "", 0.9700),
];

fn one_year(dataset: &str) -> OneYearFit {
    OdpBootstrap {
        n_sims: SIMS,
        seed: SEED,
        process: ProcessDistribution::Gamma,
    }
    .one_year(
        &triangle(dataset),
        "values",
        &OneYearMethod::ChainLadder(ChainLadder::default()),
    )
    .unwrap_or_else(|e| panic!("{dataset}: {e}"))
}

/// The single reference case of `dataset` with this method, quantity and
/// argument.
fn expected(cases: &[Case], dataset: &str, method: &str, quantity: &str, arg: &str) -> Case {
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
    found[0].clone()
}

/// Standard deviation of `x` and the Monte Carlo standard error of that
/// estimate, `sd * sqrt((kurtosis - 1) / (4 n))`.
fn sd_and_error(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let m2 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let m4 = x.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n;
    let sd = m2.sqrt();
    if m2 == 0.0 {
        return (0.0, 0.0);
    }
    (sd, sd * ((m4 / (m2 * m2) - 1.0) / (4.0 * n)).sqrt())
}

/// Each origin's draws of the CDR, then the total's, with their labels.
fn columns(fit: &OneYearFit) -> Vec<(String, Vec<f64>)> {
    let n = fit.cdr.n_components();
    let rows = || fit.cdr.draw_matrix().chunks(n);
    let mut out: Vec<(String, Vec<f64>)> = fit
        .bootstrap
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
fn one_year_sd_against_merz_wuthrich() {
    let cdr = reference("reserving_cdr_r.csv");
    let mut failures = Vec::new();
    for dataset in ["raa", "genins", "abc"] {
        let fit = one_year(dataset);
        for (origin, draws) in columns(&fit) {
            let (quantity, arg) = if origin.is_empty() {
                ("total_one_year_se", "")
            } else {
                ("one_year_se", origin.as_str())
            };
            let mw = expected(&cdr, dataset, "cdr", quantity, arg)
                .number("expected")
                .unwrap();
            let (sd, error) = sd_and_error(&draws);
            if mw == 0.0 {
                // The oldest origin has nothing left to develop.
                if sd != 0.0 {
                    failures.push(format!("{dataset} {origin}: sd {sd}, want 0"));
                }
                continue;
            }
            let gap = GAP
                .iter()
                .find(|(d, o, _)| *d == dataset && *o == origin)
                .unwrap_or_else(|| panic!("no measured gap for {dataset} {origin}"))
                .2;
            if (sd - gap * mw).abs() > 5.0 * error + 5e-5 * mw {
                failures.push(format!(
                    "{dataset} {origin}: sd / mw = {:.4}, measured {gap}",
                    sd / mw
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn one_year_sd_against_lifetime_bootstrap() {
    // R's BootChainLadder projects from the resampled latest value, the
    // one-year view from the observed one, which takes out its variance:
    // up to 2% of the standard deviation on these triangles.
    let boot = reference("reserving_bootstrap_r.csv");
    for dataset in ["raa", "genins", "abc"] {
        let fit = one_year(dataset);
        let origins = &fit.bootstrap.chain_ladder.origins;
        for (o, (origin, draws)) in columns(&fit).iter().enumerate().take(origins.len()) {
            let case = expected(&boot, dataset, "odp_bootstrap_gamma", "sd_reserve", origin);
            let (want, tol) = (
                case.number("expected").unwrap(),
                case.number("abs_tol").unwrap(),
            );
            let (sd, _) = sd_and_error(draws);
            assert!(sd <= want + tol, "{dataset} {origin}: {sd} above {want}");
            if o == 1 {
                // One cell left: the one-year view is the run-off.
                assert!(
                    (sd - want).abs() <= tol + 0.02 * want,
                    "{dataset} {origin}: {sd} vs {want}"
                );
            }
        }
    }
}

#[test]
fn evw_2019_table_2() {
    // England, Verrall and Wüthrich (2019), Table 2: chain-ladder reserves,
    // Mack RMSEP and Merz–Wüthrich RMSEP on Taylor–Ashe, with Mack's rule
    // for the last sigma (their Table 1: 21.13), rounded to units.
    let table_2 = [
        (2002, 94_634.0, 75_535.0, 75_535.0),
        (2003, 469_511.0, 121_699.0, 105_309.0),
        (2004, 709_638.0, 133_549.0, 79_846.0),
        (2005, 984_889.0, 261_406.0, 235_115.0),
        (2006, 1_419_459.0, 411_010.0, 318_427.0),
        (2007, 2_177_641.0, 558_317.0, 361_089.0),
        (2008, 3_920_301.0, 875_328.0, 629_681.0),
        (2009, 4_278_972.0, 971_258.0, 588_662.0),
        (2010, 4_625_811.0, 1_363_155.0, 1_029_925.0),
    ];
    let mack = Mack {
        development: Development {
            sigma_interpolation: SigmaInterpolation::Mack,
            ..Default::default()
        },
        ..Default::default()
    }
    .fit(&triangle("genins"), "values")
    .unwrap();
    let cdr = mack.claims_development_result().unwrap();
    let reserves = mack.chain_ladder.reserves();
    let close = |got: f64, want: f64| (got - want).abs() <= 0.5;
    for (year, reserve, mack_se, mw_se) in table_2 {
        let o = cdr
            .origins
            .iter()
            .position(|&p| p == Period::year(year))
            .unwrap();
        assert!(close(reserves[o], reserve), "{year}: {}", reserves[o]);
        assert!(close(mack.standard_error[o], mack_se), "{year}");
        assert!(close(cdr.one_year_standard_error[o], mw_se), "{year}");
    }
    assert!(close(mack.chain_ladder.total_reserve(), 18_680_856.0));
    assert!(close(mack.total_standard_error, 2_447_095.0));
    assert!(close(cdr.total_one_year_standard_error, 1_778_968.0));
}
