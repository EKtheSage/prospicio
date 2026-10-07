//! The simulated one-year view (`OdpBootstrap::one_year`, decision 8 of
//! `docs/design/reserving-v02.md`) on RAA, GenIns and ABC.
//!
//! * Against the lifetime ODP bootstrap, the independent check: an origin
//!   with one cell left to develop has a one-year CDR that is its whole
//!   run-off, so its standard deviation is R `BootChainLadder`'s for that
//!   origin (`reference/reserving_bootstrap_r.csv`) within that case's
//!   Monte Carlo tolerance; every other origin's is below it. The
//!   re-reserving itself is checked independently in `prospicio-reserving`'s unit
//!   test `mack_bootstrap_rereserving_reproduces_merz_wuthrich`: with Mack's
//!   process (England, Verrall and Wüthrich 2019, Appendix 1) it reproduces
//!   Merz–Wüthrich on GenIns within Monte Carlo error.
//! * A seed-pinned regression of the ODP one-year standard deviation, not
//!   a check against Merz and Wüthrich: each origin's and the total's
//!   standard deviation, as a ratio `GAP` to R ChainLadder's
//!   `CDR(MackChainLadder(tri))` `CDR(1)S.E.`
//!   (`reference/reserving_cdr_r.csv`), measured by this implementation
//!   with the same `SIMS` and `SEED`. The two differ by their process
//!   model: the ODP bootstrap's variance is a constant `phi` times the mean,
//!   Mack's is `sigma_k^2` times the cumulative value.
//!   `knowledge/findings/one-year-bootstrap-vs-merz-wuthrich.md` records and
//!   explains the ratios.
//! * England, Verrall and Wüthrich (2019), Table 2: their Merz–Wüthrich and
//!   Mack numbers on Taylor–Ashe (GenIns). Their simulated one-year view
//!   (Table 4) bootstraps Mack's model, not the ODP, so it is not compared.
//! * A quarterly development grain: each dataset split into quarters, each
//!   year's increment in four equal parts, has the annual opening reserve
//!   and, under both process models, about half the annual one-year
//!   standard deviation: the ODP's through its scale, which falls with the
//!   degrees of freedom, Mack's through its quarterly sigmas.

use prospicio_prob::PredictiveDistribution;
use prospicio_reserving::{
    ChainLadder, Development, DevelopmentColumn, Grain, Long, Mack, MackBootstrap, OdpBootstrap,
    OneYearFit, OneYearMethod, Period, ProcessDistribution, SigmaInterpolation, Triangle,
};
use prospicio_validation::{Case, reference, triangle};

/// Simulations per dataset; the tolerances scale with `1 / sqrt` of it.
const SIMS: usize = 20_000;
const SEED: u64 = 20_261_006;

/// Ratio of the simulated one-year standard deviation to R's Merz–Wüthrich
/// `CDR(1)S.E.`, per dataset and origin (`""` for the total), as this
/// implementation measured it with `SIMS` simulations from `SEED`: a
/// regression pin, not a reference value.
const GAP: &[(&str, &str, f64)] = &[
    ("raa", "1982", 4.9705),
    ("raa", "1983", 1.8219),
    ("raa", "1984", 3.8029),
    ("raa", "1985", 1.1873),
    ("raa", "1986", 1.0720),
    ("raa", "1987", 1.8135),
    ("raa", "1988", 0.7305),
    ("raa", "1989", 0.9679),
    ("raa", "1990", 0.5015),
    ("raa", "", 0.6087),
    ("genins", "2002", 1.6015),
    ("genins", "2003", 1.9168),
    ("genins", "2004", 2.2594),
    ("genins", "2005", 0.8457),
    ("genins", "2006", 0.7279),
    ("genins", "2007", 0.8749),
    ("genins", "2008", 0.9332),
    ("genins", "2009", 1.3504),
    ("genins", "2010", 1.7187),
    ("genins", "", 1.3620),
    ("abc", "1978", 5.3052),
    ("abc", "1979", 5.9808),
    ("abc", "1980", 2.5855),
    ("abc", "1981", 1.5543),
    ("abc", "1982", 1.6533),
    ("abc", "1983", 0.8376),
    ("abc", "1984", 1.0352),
    ("abc", "1985", 0.9606),
    ("abc", "1986", 0.9103),
    ("abc", "1987", 1.0783),
    ("abc", "", 1.1329),
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
fn one_year_sd_regression_against_merz_wuthrich() {
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
            // Five Monte Carlo standard errors, and the rounding of `GAP`.
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
    // Both project from the resampled latest value, so an origin with one
    // cell left has the same distribution in both views.
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
                    (sd - want).abs() <= tol,
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

/// `annual` with a quarterly development grain: each year's increment
/// split into four equal quarters, so the value at every 12 months is
/// unchanged and the latest diagonal is the same.
fn quarterly(annual: &Triangle) -> Triangle {
    let (mut origin, mut ages, mut values) = (vec![], vec![], vec![]);
    for (o, period) in annual.origins().iter().enumerate() {
        let mut previous = 0.0;
        for (d, &age) in annual.development().iter().enumerate() {
            let Some(v) = annual.get(0, 0, o, d) else {
                break;
            };
            for q in 1..=4 {
                origin.push(period.start());
                ages.push(age - 12 + 3 * q);
                values.push(previous + (v - previous) * f64::from(q) / 4.0);
            }
            previous = v;
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("values", &values)],
        origin_grain: Grain::Year,
        development_grain: Grain::Quarter,
        cumulative: true,
    })
    .unwrap()
}

/// Each component's draws, then the total's.
fn draws_by_origin(cdr: &PredictiveDistribution) -> Vec<Vec<f64>> {
    let n = cdr.n_components();
    let rows = || cdr.draw_matrix().chunks(n);
    let mut out: Vec<Vec<f64>> = (0..n).map(|j| rows().map(|r| r[j]).collect()).collect();
    out.push(rows().map(|r| r.iter().sum()).collect());
    out
}

#[test]
fn quarterly_split_halves_the_one_year_sd() {
    // Splitting a year's increment into four equal quarters leaves the
    // year's standard deviation about half the annual one under both
    // models (measured at 5,000 simulations: ODP 0.43 to 0.47 per origin,
    // total 0.44 to 0.47; Mack 0.48 to 0.72, total 0.53 to 0.55, the
    // highest for the origins with one year left, whose last sigma is
    // extrapolated). The ODP's variance is linear in the mean, so it is the
    // scale that falls: the Pearson chi-square is unchanged and the
    // degrees of freedom grow (36 to 171 on RAA), so the process SD falls
    // by sqrt(36 / 171) = 0.46. Mack's model takes the quarterly link
    // ratios as independent, each deviating about a quarter as much as the
    // annual one, so a sixteenth of the variance each, a quarter in all. The
    // opening reserve is the annual one: a year's quarterly volume-weighted
    // factors telescope to its annual factor. The oldest origin, at the
    // last age, has no cell in the coming year.
    let n_sims = 2_000;
    let cl = OneYearMethod::ChainLadder(ChainLadder::default());
    for dataset in ["raa", "genins", "abc"] {
        let annual = triangle(dataset);
        let split = quarterly(&annual);
        let odp = |tri: &Triangle| {
            OdpBootstrap {
                n_sims,
                seed: SEED,
                process: ProcessDistribution::Gamma,
            }
            .one_year(tri, "values", &cl)
            .unwrap_or_else(|e| panic!("{dataset}: {e}"))
        };
        let mack = |tri: &Triangle| {
            MackBootstrap {
                n_sims,
                seed: SEED,
                ..Default::default()
            }
            .one_year(tri, "values", &cl)
            .unwrap_or_else(|e| panic!("{dataset}: {e}"))
        };
        let (odp_a, odp_q) = (odp(&annual), odp(&split));
        for (q, a) in odp_q.opening_reserve.iter().zip(&odp_a.opening_reserve) {
            assert!(
                (q - a).abs() <= 1e-9 * a.abs().max(1.0),
                "{dataset}: {q} vs {a}"
            );
        }
        let (mack_a, mack_q) = (mack(&annual), mack(&split));
        for (model, a, q) in [
            ("odp", &odp_a.cdr, &odp_q.cdr),
            ("mack", &mack_a.cdr, &mack_q.cdr),
        ] {
            let (a, q) = (draws_by_origin(a), draws_by_origin(q));
            let total = a.len() - 1;
            assert!(q[0].iter().all(|&x| x == 0.0), "{dataset} {model}");
            for j in 1..=total {
                let ratio = sd_and_error(&q[j]).0 / sd_and_error(&a[j]).0;
                let (low, high) = if j == total { (0.4, 0.6) } else { (0.35, 0.75) };
                assert!(
                    (low..=high).contains(&ratio),
                    "{dataset} {model} component {j}: quarterly / annual sd {ratio}"
                );
            }
        }
    }
}
