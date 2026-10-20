//! Reserving parity: Chain Ladder and Mack against R ChainLadder and
//! chainladder-python on RAA, GenIns and ABC.
//!
//! Every row of both reference files is evaluated; see
//! `scripts/reserving_r.R` and `scripts/reserving_chainladder_python.py`.
//! The ODP bootstrap is checked against R's `BootChainLadder`: exactly for
//! the scale and residuals, within Monte Carlo tolerances for the reserve
//! distribution (`scripts/reserving_bootstrap_r.R`). The ODP GLM is checked
//! against Chain Ladder, the bootstrap's scale and R's quasi-Poisson `glm`
//! (`scripts/reserving_glm_r.R`). The calendar-diagonal backtest is checked
//! against Chain Ladder on the truncated triangle.

use std::collections::HashMap;

use prospicio_glm::Glm;
use prospicio_models::stack::stacking_weights;
use prospicio_models::{Column, Design, Terms};
use prospicio_prob::Distribution;
use prospicio_reserving::{
    Average, ChainLadder, ChainLadderFit, Development, DevelopmentColumn, GlmCandidate, Grain,
    Label, Long, Mack, MackFit, Month, OdpBootstrap, OdpBootstrapFit, OdpBootstrapFits, OdpGlm,
    OdpGlmFit, Period, ProcessDistribution, ResidualGroup, SigmaInterpolation, Triangle,
    TriangleFrame, TriangleModel, diagonal_backtest,
};
use prospicio_validation::{Case, check, reference, triangle};

/// Development estimator for a reference `method`.
fn development(method: &str) -> Development {
    let (average, sigma_interpolation) = match method {
        "chain_ladder" | "mack" => (Average::Volume, SigmaInterpolation::LogLinear),
        "chain_ladder_simple" | "mack_alpha0" => (Average::Simple, SigmaInterpolation::LogLinear),
        "mack_alpha2" => (Average::Regression, SigmaInterpolation::LogLinear),
        "mack_sigma_mack" => (Average::Volume, SigmaInterpolation::Mack),
        other => panic!("unknown method {other}"),
    };
    Development {
        average,
        sigma_interpolation,
        pattern: None,
        ..Default::default()
    }
}

/// Fits each (dataset, method) once and answers reference cases from it.
#[derive(Default)]
struct Fits {
    mack: HashMap<(String, String), MackFit>,
}

impl Fits {
    fn mack(&mut self, dataset: &str, method: &str) -> &MackFit {
        self.mack
            .entry((dataset.to_string(), method.to_string()))
            .or_insert_with(|| {
                Mack {
                    development: development(method),
                    ..Default::default()
                }
                .fit(&triangle(dataset), "values")
                .unwrap_or_else(|e| panic!("{dataset} {method}: {e}"))
            })
    }

    fn eval(&mut self, case: &Case) -> Option<f64> {
        let (dataset, method, quantity) = (
            case.get("dataset"),
            case.get("method"),
            case.get("quantity"),
        );
        let fit = self.mack(dataset, method);
        let cl = &fit.chain_ladder;
        let age = || case.number("arg").map(|k| k as usize);
        let origin = || {
            let year = case.number("arg")? as i32;
            cl.origins.iter().position(|&p| p == Period::year(year))
        };
        match quantity {
            "ata_factor" => cl.development.ldf.get(age()?).copied(),
            "cdf" => cl.cdf.get(age()?).copied(),
            "sigma" => cl.development.sigma.get(age()?).copied(),
            "f_se" => cl.development.std_err.get(age()?).copied(),
            "ultimate" => Some(cl.ultimate[origin()?]),
            "reserve" => Some(cl.reserves()[origin()?]),
            "se" => Some(fit.standard_error[origin()?]),
            "process_risk" => Some(fit.process_risk[origin()?]),
            "parameter_risk" => Some(fit.parameter_risk[origin()?]),
            "total_ultimate" => Some(cl.total_ultimate()),
            "total_reserve" => Some(cl.total_reserve()),
            "total_standard_error" => Some(fit.total_standard_error),
            "total_process_risk" => Some(fit.total_process_risk),
            "total_parameter_risk" => Some(fit.total_parameter_risk),
            _ => None,
        }
    }
}

#[test]
fn matches_r_chainladder() {
    let mut fits = Fits::default();
    check(&reference("reserving_chainladder_r.csv"), |c| fits.eval(c));
}

#[test]
fn matches_chainladder_python() {
    let mut fits = Fits::default();
    check(&reference("reserving_chainladder_python.csv"), |c| {
        fits.eval(c)
    });
}

#[test]
fn chain_ladder_alone_matches_mack_projection() {
    // The deterministic method gives the same projection Mack builds on.
    for dataset in ["raa", "genins", "abc"] {
        let tri = triangle(dataset);
        let cl: ChainLadderFit = ChainLadder::default().fit(&tri, "values").unwrap();
        let mack = Mack::default().fit(&tri, "values").unwrap();
        assert_eq!(cl, mack.chain_ladder, "{dataset}");
    }
}

#[test]
fn dataset_shapes() {
    for (name, n, first, latest_total) in [
        ("raa", 10, 1981, 160_987.0),
        ("genins", 10, 2001, 34_358_090.0),
        ("abc", 11, 1977, 10_221_194.0),
    ] {
        let tri = triangle(name);
        assert_eq!(tri.shape(), [1, 1, n, n], "{name}");
        assert_eq!(tri.origins()[0], Period::year(first), "{name}");
        let latest: f64 = tri.latest_diagonal().values(0, 0).iter().sum();
        assert_eq!(latest, latest_total, "{name}");
    }
}

/// Simulations per bootstrap; the reference tolerances assume this many.
const BOOTSTRAP_SIMS: usize = 20_000;

#[test]
fn bootstrap_matches_r_bootchainladder() {
    let mut fits: HashMap<(String, String), OdpBootstrapFit> = HashMap::new();
    check(&reference("reserving_bootstrap_r.csv"), |case| {
        let (dataset, method) = (case.get("dataset"), case.get("method"));
        let process = match method {
            "odp_bootstrap" | "odp_bootstrap_gamma" => ProcessDistribution::Gamma,
            "odp_bootstrap_param" => ProcessDistribution::None,
            other => panic!("unknown method {other}"),
        };
        let fit = fits
            .entry((dataset.to_string(), method.to_string()))
            .or_insert_with(|| {
                OdpBootstrap {
                    n_sims: BOOTSTRAP_SIMS,
                    seed: 20_261_004,
                    process,
                    ..Default::default()
                }
                .fit(&triangle(dataset), "values")
                .unwrap_or_else(|e| panic!("{dataset} {method}: {e}"))
            });
        let origins = &fit.chain_ladder.origins;
        let origin = |year: &str| {
            let year: i32 = year.parse().ok()?;
            origins.iter().position(|&p| p == Period::year(year))
        };
        let reserves = &fit.reserves;
        let marginal = |year: &str| reserves.marginal(&vec![origins[origin(year)?].into()]);
        match case.get("quantity") {
            "scale" => Some(fit.scale),
            "residual" => {
                let (year, k) = case.get("arg").split_once(':')?;
                let n_dev = fit.residuals.len() / origins.len();
                Some(fit.residuals[origin(year)? * n_dev + k.parse::<usize>().ok()?])
            }
            "mean_reserve" => Some(marginal(case.get("arg"))?.mean()),
            "sd_reserve" => Some(marginal(case.get("arg"))?.std_dev()),
            "mean_total" => Some(reserves.mean()),
            "sd_total" => Some(reserves.std_dev()),
            "quantile_total" => reserves.quantile(case.number("arg")?).ok(),
            _ => None,
        }
    });
}

/// The ODP model as a GLM on the cells: quasi-Poisson, intercept, origin
/// and development factors.
fn odp_candidate() -> GlmCandidate {
    GlmCandidate {
        name: "odp".into(),
        terms: Terms::new()
            .intercept()
            .factor("origin")
            .factor("development"),
        glm: Glm::over_dispersed_poisson(),
    }
}

/// A deliberately worse model: development only, every origin alike.
fn development_only() -> GlmCandidate {
    GlmCandidate {
        name: "development only".into(),
        terms: Terms::new().intercept().factor("development"),
        glm: Glm::over_dispersed_poisson(),
    }
}

#[test]
fn diagonal_backtest_on_raa() {
    let cells = TriangleFrame::new(&triangle("raa"), "values", None).unwrap();
    let (odp, worse) = (odp_candidate(), development_only());
    let bt = diagonal_backtest(&cells, &[&odp, &worse], 2, 4000, 2024, 0.9).unwrap();
    assert_eq!(bt.models(), ["odp", "development only"]);
    assert_eq!(
        bt.metrics(),
        [
            "crps",
            "coverage",
            "actual_vs_expected",
            "total_crps",
            "mean_log_density",
            "total_percentile"
        ]
    );
    assert_eq!(bt.n_splits(), 2);
    // 1989 and 1990: the newest origin (at 12) and 1981 (at 108, then
    // 120) have no training level, which leaves 7 and 8 scored cells.
    assert_eq!(bt.excluded(), [2, 2]);
    assert_eq!(bt.scored_rows()[0].len(), 7);
    assert_eq!(bt.scored_rows()[1].len(), 8);
    for m in 0..2 {
        assert!(
            bt.split_scores(m, 1)
                .iter()
                .all(|c| (0.0..=1.0).contains(c))
        );
        assert!(bt.split_scores(m, 2).iter().all(|ae| ae.is_finite()));
    }
    // The ODP model over-predicts the low 1990 diagonal: A/E 1.20, 0.67
    // (fitted with the negative 1982 increment, which the quasi-Poisson
    // accepts).
    let ae = bt.split_scores(0, 2);
    assert!(
        (ae[0] - 1.1999).abs() < 1e-3 && (ae[1] - 0.6735).abs() < 1e-3,
        "{ae:?}"
    );
    // On RAA's latest diagonals, development alone scores better than the
    // ODP model (cell CRPS about 1000 against 1400): its origin effects for
    // the young origins rest on one or two noisy cells.
    assert!(bt.mean(1, 0) < bt.mean(0, 0));

    // Aligned held-out log densities: one per scored cell, both splits.
    let lpd: Vec<Vec<f64>> = bt
        .log_densities()
        .iter()
        .map(|l| l.clone().expect("a GLM gives log densities"))
        .collect();
    assert!(lpd.iter().all(|l| l.len() == 15));
    let w = stacking_weights(&lpd).unwrap();
    assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);

    // Deterministic for a seed.
    let again = diagonal_backtest(&cells, &[&odp, &worse], 2, 4000, 2024, 0.9).unwrap();
    assert_eq!(again, bt);
}

/// The ODP GLM's backtest on RAA, pinned for the Python and R bindings,
/// which assert the same numbers (`test_reserving_backtest.py`,
/// `test-reserving-backtest.R`): a metric or split permutation fails them.
#[test]
fn diagonal_backtest_odp_scores_are_pinned() {
    let cells = TriangleFrame::new(&triangle("raa"), "values", None).unwrap();
    let odp = odp_candidate();
    let bt = diagonal_backtest(&cells, &[&odp], 2, 500, 3, 0.9).unwrap();
    // (metric, 1989, 1990, mean over the two diagonals).
    let pinned: [(&str, f64, f64, f64); 6] = [
        (
            "crps",
            1373.447669154837,
            1121.1766328225226,
            1247.3121509886798,
        ),
        ("coverage", 0.8571428571428571, 0.75, 0.8035714285714286),
        (
            "actual_vs_expected",
            1.1999148354486822,
            0.6734699434756398,
            0.936692389462161,
        ),
        (
            "total_crps",
            2357.940676273035,
            4366.701349320293,
            3362.321012796664,
        ),
        (
            "mean_log_density",
            -8.960644355528585,
            -8.720382749971636,
            -8.84051355275011,
        ),
        ("total_percentile", 0.75, 0.13, 0.44),
    ];
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1.0);
    for (k, (metric, first, second, mean)) in pinned.into_iter().enumerate() {
        assert_eq!(bt.metrics()[k], metric);
        let split = bt.split_scores(0, k);
        assert!(close(split[0], first), "{metric} 1989: {split:?}");
        assert!(close(split[1], second), "{metric} 1990: {split:?}");
        assert!(
            close(bt.mean(0, k), mean),
            "{metric} mean: {}",
            bt.mean(0, k)
        );
    }
}

#[test]
fn diagonal_backtest_prefers_the_odp_model_on_abc() {
    let cells = TriangleFrame::new(&triangle("abc"), "values", None).unwrap();
    let (odp, worse) = (odp_candidate(), development_only());
    let bt = diagonal_backtest(&cells, &[&odp, &worse], 3, 4000, 7, 0.9).unwrap();
    assert_eq!(bt.excluded(), [2, 2, 2]);
    assert!(bt.mean(0, 0) < bt.mean(1, 0), "ODP wins on cell CRPS");
    assert!(bt.mean(0, 3) < bt.mean(1, 3), "ODP wins on total CRPS");
    for &ae in bt.split_scores(0, 2) {
        assert!((ae - 1.0).abs() < 0.2, "ODP actual vs expected {ae}");
    }
    for m in 0..2 {
        assert!(
            bt.split_scores(m, 1)
                .iter()
                .all(|c| (0.0..=1.0).contains(c))
        );
    }
    let lpd: Vec<Vec<f64>> = bt.log_densities().iter().flatten().cloned().collect();
    assert_eq!(lpd.len(), 2);
    let w = stacking_weights(&lpd).unwrap();
    assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(w[0] > w[1], "stacking favours the ODP model: {w:?}");
}

/// On a held-out diagonal the ODP model's means are Chain Ladder's
/// one-period forecasts from the triangle valued before it.
#[test]
fn backtest_odp_means_match_chain_ladder_on_genins() {
    let tri = triangle("genins");
    let cells = TriangleFrame::new(&tri, "values", None).unwrap();
    let odp = odp_candidate();
    let splits = cells.diagonal_splits(3).unwrap();
    let bt = diagonal_backtest(&cells, &[&odp], 3, 10, 1, 0.9).unwrap();
    let long = tri.to_long();
    for (split, scored) in splits.iter().zip(bt.scored_rows()) {
        let forecast = odp.forecast(&cells, &split.train, scored, 10, 1).unwrap();
        // The triangle as it stood before the held-out diagonal.
        let cutoff = cells.calendar_of(scored[0]);
        let keep: Vec<usize> = (0..long.origin.len())
            .filter(|&i| {
                let o = long.origin[i].year() - 2001;
                let d = long.development[i] / 12 - 1;
                !long.values[0].1[i].is_nan() && i64::from(o) + i64::from(d) < cutoff
            })
            .collect();
        let origin: Vec<Month> = keep.iter().map(|&i| long.origin[i]).collect();
        let ages: Vec<u32> = keep.iter().map(|&i| long.development[i]).collect();
        let values: Vec<f64> = keep.iter().map(|&i| long.values[0].1[i]).collect();
        let before = Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("values", &values)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let cl = ChainLadder::default().fit(&before, "values").unwrap();
        for (&row, &mean) in scored.iter().zip(&forecast.mean) {
            let (o, d) = (cells.origin_of(row), cells.development_of(row));
            assert_eq!(cl.latest_position[o], d - 1);
            let expected = cl.latest[o] * (cl.development.ldf[d - 1] - 1.0);
            assert!(
                (mean - expected).abs() <= 1e-7 * expected.abs(),
                "origin {o} dev {d}: {mean} vs {expected}"
            );
        }
    }
}

/// Relative difference, with an absolute floor of 1 for values near zero.
fn rel_diff(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1.0)
}

#[test]
fn odp_glm_matches_r_glm() {
    let mut fits: HashMap<String, OdpGlmFit> = HashMap::new();
    check(&reference("reserving_glm_r.csv"), |case| {
        let dataset = case.get("dataset");
        assert_eq!(case.get("method"), "odp_glm");
        let fit = fits.entry(dataset.to_string()).or_insert_with(|| {
            OdpGlm::default()
                .fit(&triangle(dataset), "values")
                .unwrap_or_else(|e| panic!("{dataset}: {e}"))
        });
        let arg = case.get("arg");
        let coefficient = || fit.glm.names().iter().position(|n| n == arg);
        let origin = || {
            let year: i32 = arg.parse().ok()?;
            fit.origins.iter().position(|&p| p == Period::year(year))
        };
        match case.get("quantity") {
            "coefficient" => Some(fit.glm.coefficients()[coefficient()?]),
            "std_error" => Some(fit.glm.std_errors()[coefficient()?]),
            "dispersion" => Some(fit.glm.dispersion()),
            "reserve" => Some(fit.reserves[origin()?]),
            "total_reserve" => Some(fit.total_reserve()),
            _ => None,
        }
    });
}

#[test]
fn odp_glm_reproduces_chain_ladder_and_bootstrap_scale() {
    // RAA included: its negative 1982 increment at 84 months (-103) is
    // fitted by the quasi-likelihood, which R's quasipoisson refuses.
    for dataset in ["raa", "genins", "abc"] {
        let tri = triangle(dataset);
        let fit = OdpGlm::default().fit(&tri, "values").unwrap();
        // Renshaw and Verrall: the fitted future cells summed by origin are
        // the volume-weighted Chain Ladder reserves.
        let cl = ChainLadder::default().fit(&tri, "values").unwrap();
        assert_eq!(fit.origins, cl.origins);
        for (k, (a, b)) in fit.reserves.iter().zip(cl.reserves()).enumerate() {
            assert!(rel_diff(*a, b) < 1e-8, "{dataset} origin {k}: {a} vs {b}");
        }
        assert!(rel_diff(fit.total_reserve(), cl.total_reserve()) < 1e-8);
        // Pearson's dispersion is the ODP bootstrap's scale.
        let boot = OdpBootstrap {
            n_sims: 1,
            ..Default::default()
        }
        .fit(&tri, "values")
        .unwrap();
        let phi = fit.glm.dispersion();
        assert!(
            rel_diff(phi, boot.scale) < 1e-8,
            "{dataset}: {phi} vs {}",
            boot.scale
        );
    }
}

#[test]
fn odp_glm_predictive_distribution_matches_its_mean() {
    const SIMS: usize = 20_000;
    for dataset in ["raa", "genins", "abc"] {
        let tri = triangle(dataset);
        let fit = OdpGlm::default().fit(&tri, "values").unwrap();
        let cl = ChainLadder::default().fit(&tri, "values").unwrap();
        let pd = fit.predict_distribution(SIMS, 20_261_005).unwrap();
        assert_eq!(pd.dims(), ["origin", "development"]);
        assert_eq!(pd.n_components(), fit.future.len());
        let mc_se = pd.std_dev() / (SIMS as f64).sqrt();
        // Mean-preserving parameter draws: the draws average the Chain
        // Ladder reserve.
        assert!(
            (pd.mean() - cl.total_reserve()).abs() < 4.0 * mc_se,
            "{dataset}: {} vs {}",
            pd.mean(),
            cl.total_reserve()
        );
        // Reserves by origin: one component per origin with a future cell.
        let by_origin = pd.aggregate(&["origin"]).unwrap();
        assert_eq!(by_origin.n_components(), cl.origins.len() - 1);
        for (k, key) in by_origin.components().iter().enumerate() {
            let o = cl
                .origins
                .iter()
                .position(|p| key[0] == (*p).into())
                .unwrap();
            assert!(o > 0, "{dataset}: the oldest origin is fully developed");
            let m = by_origin.marginal(key).unwrap();
            let se = m.std_dev() / (SIMS as f64).sqrt();
            let want: f64 = fit
                .future
                .iter()
                .zip(fit.future_means.iter().copied())
                .filter(|((origin, _), _)| *origin == cl.origins[o])
                .map(|(_, mean)| mean)
                .sum();
            assert!(
                (m.mean() - want).abs() < 4.0 * se,
                "{dataset} component {k}"
            );
        }
    }
}

/// Taylor and McGuire, *Stochastic Loss Reserving Using Generalized Linear
/// Models* (CAS Monograph 3), Table 1-1: the incremental paid losses of
/// the NJ Manufacturers workers compensation triangle, accident years 1988
/// to 1997.
const TAYLOR_MCGUIRE: [&[f64]; 10] = [
    &[
        41821., 34729., 20147., 15965., 11285., 5924., 4775., 3742., 3435., 2958.,
    ],
    &[
        48167., 39495., 24444., 18178., 10840., 7379., 5683., 4758., 3959.,
    ],
    &[52058., 47459., 27359., 17916., 11448., 8846., 5869., 5391.],
    &[57251., 49510., 27036., 20871., 14304., 10552., 7742.],
    &[59213., 54129., 29566., 22484., 14114., 10000.],
    &[59475., 52076., 26836., 22332., 14756.],
    &[65607., 44648., 27062., 22655.],
    &[56748., 39315., 26748.],
    &[52212., 40030.],
    &[43962.],
];

/// Taylor and McGuire's triangle, cumulated.
fn taylor_mcguire() -> Triangle {
    let (mut origin, mut age, mut paid) = (Vec::new(), Vec::new(), Vec::new());
    for (k, row) in TAYLOR_MCGUIRE.iter().enumerate() {
        let mut cumulative = 0.0;
        for (j, v) in row.iter().enumerate() {
            cumulative += v;
            origin.push(Month::january(1988 + k as i32));
            age.push(12 * (j as u32 + 1));
            paid.push(cumulative);
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[("paid", &paid)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

#[test]
fn odp_glm_prediction_error_matches_taylor_mcguire_table_5_4() {
    let tri = taylor_mcguire();
    let fit = OdpGlm::default().fit(&tri, "paid").unwrap();
    let error = fit.prediction_error().unwrap();
    // Table 5-4: forecast and RMSEP by accident year 1989 to 1997, then
    // the total, as printed (to the unit).
    let table = [
        (3_398.0, 924.0),
        (8_155.0, 1_363.0),
        (14_579.0, 1_775.0),
        (22_645.0, 2_169.0),
        (31_865.0, 2_523.0),
        (45_753.0, 3_036.0),
        (60_093.0, 3_577.0),
        (80_983.0, 4_538.0),
        (105_874.0, 6_786.0),
    ];
    assert_eq!(error.by_origin[0].rmsep(), 0.0);
    for (e, (forecast, rmsep)) in error.by_origin[1..].iter().zip(table) {
        assert!(
            (e.total - forecast).abs() <= 0.5,
            "{} vs {forecast}",
            e.total
        );
        assert!((e.rmsep() - rmsep).abs() <= 0.5, "{} vs {rmsep}", e.rmsep());
    }
    assert!((error.total.total - 373_346.0).abs() <= 0.5);
    assert!((error.total.rmsep() - 14_076.0).abs() <= 0.5);
    // The split the guide prints, and the monograph's 10,275 for the
    // total if the accident years were independent.
    assert!((error.total.process_variance.sqrt() - 6_539.0).abs() <= 0.5);
    assert!((error.total.parameter_variance.sqrt() - 12_465.0).abs() <= 0.5);
    let independent: f64 = error.by_origin.iter().map(|e| e.rmsep().powi(2)).sum();
    assert!((independent.sqrt() - 10_275.0).abs() <= 0.5);
}

#[test]
fn odp_glm_residuals_match_taylor_mcguire_chapter_6() {
    let fit = OdpGlm::default().fit(&taylor_mcguire(), "paid").unwrap();
    let r = fit.residuals().unwrap();
    assert_eq!(r.actual.len(), 55);
    // The two cells with leverage 1: 1988's last year and 1997's first.
    let exact: Vec<usize> = (0..55)
        .filter(|&i| r.standardized_deviance[i].is_nan())
        .collect();
    assert_eq!(exact, [9, 54]);
    for i in 0..55 {
        let h = r.leverage[i];
        assert!((0.0..=1.0 + 1e-9).contains(&h));
    }
    // The leverages add up to the 19 parameters.
    assert!((r.leverage.iter().sum::<f64>() - 19.0).abs() < 1e-8);

    // The guide's spread by development year 1 to 8 (np.nanstd, divisor n)
    // and mean by calendar year 1988 to 1997, to two decimals
    // (docs/guides/reserving-taylor-mcguire.md, section 5).
    let by_age = r.summary(ResidualGroup::Development);
    let spread = [1.34, 1.59, 0.88, 0.62, 0.83, 0.77, 0.65, 0.40];
    for (g, want) in by_age.iter().zip(spread) {
        let n = g.n as f64;
        let population = g.sd * ((n - 1.0) / n).sqrt();
        assert!(
            (population - want).abs() <= 0.005,
            "{}: {population} vs {want}",
            g.label
        );
    }
    assert_eq!(by_age.len(), 10);
    assert_eq!(by_age[0].label, "12");
    assert_eq!((by_age[8].n, by_age[9].n), (2, 0));
    let by_calendar = r.summary(ResidualGroup::Calendar);
    let means = [
        -0.38, -0.13, -0.40, 0.56, 0.42, -0.43, 0.23, -0.27, -0.26, 0.32,
    ];
    assert_eq!(by_calendar.len(), 10);
    for (g, want) in by_calendar.iter().zip(means) {
        assert!(
            (g.mean - want).abs() <= 0.005,
            "{}: {} vs {want}",
            g.label,
            g.mean
        );
    }
    assert_eq!(by_calendar[0].label, "1988");
    let by_origin = r.summary(ResidualGroup::Origin);
    assert_eq!(by_origin[0].label, "1988");
    assert_eq!(by_origin.iter().map(|g| g.n).sum::<usize>(), 53);

    // The actual-to-fitted heat map, Figure 6-22, in percent.
    let heat: [&[i64]; 10] = [
        &[98, 100, 100, 104, 113, 87, 96, 92, 100, 100],
        &[99, 99, 106, 103, 95, 95, 99, 102, 100],
        &[96, 108, 107, 91, 90, 102, 92, 104],
        &[97, 103, 96, 97, 103, 111, 111],
        &[95, 107, 100, 100, 97, 100],
        &[98, 105, 93, 101, 104],
        &[109, 91, 95, 104],
        &[106, 90, 105],
        &[103, 97],
        &[100],
    ];
    let printed: Vec<i64> = heat.iter().flat_map(|row| row.iter().copied()).collect();
    let ours: Vec<i64> = r
        .actual_to_fitted
        .iter()
        .map(|x| (100.0 * x).round() as i64)
        .collect();
    assert_eq!(ours, printed);
}

/// A column of a custom design: a name and its value at `(k, j)`.
type DesignColumn = (&'static str, Box<dyn Fn(f64, f64) -> f64>);

/// Taylor and McGuire's Chapter 7 models as custom designs on the NJ
/// triangle, `k` and `j` the 1-based accident and development years.
fn taylor_mcguire_design(columns: &[DesignColumn]) -> Design {
    let cells = TriangleFrame::new(&taylor_mcguire(), "paid", None).unwrap();
    let index = |name: &str| match cells.frame().column(name) {
        Some(Column::Numeric(v)) => v.clone(),
        _ => unreachable!("a triangle frame has numeric indices"),
    };
    let (k, j) = (index("origin_index"), index("development_index"));
    Design::new(
        columns
            .iter()
            .map(|(name, _)| (*name).to_string())
            .collect(),
        columns
            .iter()
            .map(|(_, f)| k.iter().zip(&j).map(|(k, j)| f(k + 1.0, j + 1.0)).collect())
            .collect(),
    )
    .unwrap()
}

fn column(name: &'static str, f: impl Fn(f64, f64) -> f64 + 'static) -> DesignColumn {
    (name, Box::new(f))
}

/// `1[j = at]`, development year `at`'s own effect.
fn year(name: &'static str, at: f64) -> DesignColumn {
    column(name, move |_, j| f64::from(u8::from(j == at)))
}

fn close(ours: &[f64], printed: &[f64], tolerance: f64) {
    assert_eq!(ours.len(), printed.len());
    for (a, b) in ours.iter().zip(printed) {
        assert!((a - b).abs() <= tolerance, "{ours:?} vs {printed:?}");
    }
}

#[test]
fn odp_glm_designs_match_taylor_mcguire_chapter_7() {
    let tri = taylor_mcguire();
    let odp = OdpGlm::default();

    // Model 7-2: the accident-year effect a quadratic in k, all ten
    // development effects free (Table 7-1).
    let names = [
        "ln b1", "ln b2", "ln b3", "ln b4", "ln b5", "ln b6", "ln b7", "ln b8", "ln b9", "ln b10",
    ];
    let mut quad = vec![column("a1", |k, _| k), column("a2", |k, _| k * k)];
    for (b, name) in names.into_iter().enumerate() {
        quad.push(year(name, b as f64 + 1.0));
    }
    let fit = odp
        .fit_design(&tri, "paid", &taylor_mcguire_design(&quad))
        .unwrap();
    let c = fit.glm.coefficients();
    close(&c[..2], &[0.2001, -0.0179], 5e-5);
    close(&c[2..3], &[10.471], 5e-4);
    let contrasts: Vec<f64> = c[3..].iter().map(|x| x - c[2]).collect();
    let table_7_1 = [
        -0.206, -0.750, -1.015, -1.452, -1.830, -2.142, -2.353, -2.514, -2.661,
    ];
    close(&contrasts, &table_7_1, 5e-4);
    assert!((fit.glm.dispersion() - 102.6).abs() < 0.05);
    let error = fit.prediction_error().unwrap().total;
    assert!((error.total - 372_532.0).abs() <= 0.5, "{}", error.total);
    assert!((error.rmsep() - 13_030.0).abs() <= 0.5, "{}", error.rmsep());

    // Model 7-10: development effects a line in j with a kink at 7.5 and a
    // term for development year 2 (Table 7-3; the guide's four decimals,
    // since c is 0.1544995 where the table rounds to 0.155).
    let smooth = || {
        vec![
            column("a0", |_, _| 1.0),
            column("a1", |k, _| k),
            column("a2", |k, _| k * k),
            column("b1", |_, j| j - 1.0),
            column("b2", |_, j| (j - 7.5).max(0.0)),
        ]
    };
    let mut smooth_7_10 = smooth();
    smooth_7_10.push(year("c", 2.0));
    let fit = odp
        .fit_design(&tri, "paid", &taylor_mcguire_design(&smooth_7_10))
        .unwrap();
    close(
        fit.glm.coefficients(),
        &[10.4687, 0.2001, -0.0179, -0.3577, 0.2356, 0.1545],
        5e-5,
    );
    close(
        fit.glm.coefficients(),
        &[10.469, 0.200, -0.018, -0.358, 0.236, 0.155],
        1e-3,
    );
    assert!((fit.glm.dispersion() - 107.3).abs() < 0.05);
    let error = fit.prediction_error().unwrap().total;
    assert!((error.total - 373_006.0).abs() <= 0.5, "{}", error.total);
    assert!((error.rmsep() - 13_215.0).abs() <= 0.5, "{}", error.rmsep());
    // The actual-to-fitted heat map of model 7-10, Figure 7-5, in percent.
    let heat: [&[i64]; 10] = [
        &[99, 101, 98, 111, 112, 84, 97, 96, 100, 97],
        &[99, 99, 102, 109, 93, 90, 99, 106, 99],
        &[95, 107, 102, 96, 88, 97, 92, 107],
        &[97, 103, 94, 104, 102, 107, 113],
        &[97, 108, 99, 108, 97, 98],
        &[97, 104, 89, 106, 101],
        &[110, 92, 93, 112],
        &[102, 87, 99],
        &[105, 98],
        &[101],
    ];
    let printed: Vec<i64> = heat.iter().flat_map(|row| row.iter().copied()).collect();
    let r = fit.residuals().unwrap();
    let ours: Vec<i64> = r
        .actual_to_fitted
        .iter()
        .map(|x| (100.0 * x).round() as i64)
        .collect();
    assert_eq!(ours, printed);

    // Model 7-14: 7-10 with a term for development year 4 and the
    // interactions the heat map points to (Table 7-5, the monograph's
    // order).
    let mut interact = smooth();
    interact.push(year("c1", 2.0));
    interact.push(year("c2", 4.0));
    interact.push(column("d1", |k, j| {
        f64::from(u8::from(j == 1.0 && k <= 6.0))
    }));
    interact.push(column("d2", |k, j| {
        f64::from(u8::from(j == 2.0 && k <= 6.0))
    }));
    interact.push(column("d3", |k, j| f64::from(u8::from(j == 3.0)) * k));
    let fit = odp
        .fit_design(&tri, "paid", &taylor_mcguire_design(&interact))
        .unwrap();
    // Table 7-5 prints a0 to three decimals with a trailing zero.
    let c = fit.glm.coefficients();
    close(&c[..1], &[10.490], 5e-4);
    let table_7_5 = [
        0.2066, -0.0183, -0.3685, 0.2720, 0.0375, 0.0528, -0.0671, 0.1273, -0.0113,
    ];
    close(&c[1..], &table_7_5, 5e-5);
    assert!((fit.glm.dispersion() - 53.9).abs() < 0.05);
    let error = fit.prediction_error().unwrap().total;
    assert!((error.total - 370_493.0).abs() <= 0.5, "{}", error.total);
    assert!((error.rmsep() - 11_021.0).abs() <= 0.5, "{}", error.rmsep());
    // Every parameter is estimated, so the leverages add up to ten.
    let r = fit.residuals().unwrap();
    assert!((r.leverage.iter().sum::<f64>() - 10.0).abs() < 1e-8);
    assert!(fit.coding.is_none());
}

#[test]
fn odp_glm_design_is_checked() {
    let tri = taylor_mcguire();
    let odp = OdpGlm::default();
    // A design with a row too few.
    let short = Design::new(vec!["a0".into()], vec![vec![1.0; 99]]).unwrap();
    let e = odp
        .fit_design(&tri, "paid", &short)
        .unwrap_err()
        .to_string();
    assert!(e.contains("99 rows") && e.contains("100"), "{e}");
    // A column only the future cells take: a calendar year after 1997.
    let future_only = [
        column("a0", |_, _| 1.0),
        column("next", |k, j| f64::from(u8::from(k + j > 11.0))),
    ];
    let e = odp
        .fit_design(&tri, "paid", &taylor_mcguire_design(&future_only))
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("column next is zero on every observed cell"),
        "{e}"
    );
    // The cross-classified design built by hand reproduces fit.
    let cells = TriangleFrame::new(&tri, "paid", None).unwrap();
    let design = Terms::new()
        .intercept()
        .factor_with_reference("origin", "1988")
        .factor_with_reference("development", "12")
        .fit(cells.frame())
        .unwrap()
        .design(cells.frame())
        .unwrap();
    let by_hand = odp.fit_design(&tri, "paid", &design).unwrap();
    let fit = odp.fit(&tri, "paid").unwrap();
    assert_eq!(by_hand.glm.coefficients(), fit.glm.coefficients());
    assert_eq!(by_hand.reserves, fit.reserves);
}

/// Rows of a two-key (lob × coverage) long table with paid and incurred:
/// Auto from RAA and Home from GenIns (scaled), with origins re-based to
/// 2011, each coverage a different share of the line. Returns the key
/// columns, origins, ages, paid and incurred.
type KeyedRows = (
    Vec<&'static str>,
    Vec<&'static str>,
    Vec<Month>,
    Vec<u32>,
    Vec<f64>,
    Vec<f64>,
);

fn lob_coverage_rows() -> KeyedRows {
    let mut rows: KeyedRows = Default::default();
    for (lob, dataset, scale) in [("Auto", "raa", 1.0), ("Home", "genins", 1e-3)] {
        let long = triangle(dataset).to_long();
        let first = long.origin[0].year();
        for (coverage, share) in [("BI", 0.7), ("PD", 0.3)] {
            for (k, ((origin, &age), &paid)) in long
                .origin
                .iter()
                .zip(&long.development)
                .zip(&long.values[0].1)
                .enumerate()
            {
                // A coverage-specific tilt by origin so the segments develop
                // differently.
                let tilt = 1.0 + share * (origin.year() - first) as f64 / 20.0;
                let paid = paid * scale * share * tilt;
                rows.0.push(lob);
                rows.1.push(coverage);
                rows.2.push(Month::january(2011 + origin.year() - first));
                rows.3.push(age);
                rows.4.push(paid);
                rows.5.push(paid * 1.2 + 10.0 * (k % 3) as f64);
            }
        }
    }
    rows
}

fn keyed_triangle(rows: &KeyedRows, keys: &[(&str, &[&str])]) -> Triangle {
    Triangle::from_long(&Long {
        keys,
        origin: &rows.2,
        development: DevelopmentColumn::Age(&rows.3),
        values: &[("paid", &rows.4), ("incurred", &rows.5)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

#[test]
fn select_and_group_by_on_lob_and_coverage() {
    let rows = lob_coverage_rows();
    let tri = keyed_triangle(&rows, &[("lob", &rows.0), ("coverage", &rows.1)]);
    assert_eq!(tri.shape(), [4, 2, 10, 10]);

    // Selecting one segment equals building it alone.
    let auto_bi = tri
        .select(&[("lob", &["Auto"]), ("coverage", &["BI"])])
        .unwrap();
    let keep: Vec<usize> = (0..rows.0.len())
        .filter(|&r| rows.0[r] == "Auto" && rows.1[r] == "BI")
        .collect();
    let pick = |c: &[f64]| keep.iter().map(|&r| c[r]).collect::<Vec<_>>();
    let alone = Triangle::from_long(&Long {
        keys: &[
            ("lob", &vec!["Auto"; keep.len()]),
            ("coverage", &vec!["BI"; keep.len()]),
        ],
        origin: &keep.iter().map(|&r| rows.2[r]).collect::<Vec<_>>(),
        development: DevelopmentColumn::Age(&keep.iter().map(|&r| rows.3[r]).collect::<Vec<_>>()),
        values: &[("paid", &pick(&rows.4)), ("incurred", &pick(&rows.5))],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap();
    assert_eq!(auto_bi, alone);
    let bi = tri.select(&[("coverage", &["BI"])]).unwrap();
    assert_eq!(bi.shape()[0], 2);

    // Grouping equals building with fewer keys (from_long sums the rows),
    // and its totals are the sums of the segments'.
    let by_lob = tri.group_by(&["lob"]).unwrap();
    assert_eq!(by_lob, keyed_triangle(&rows, &[("lob", &rows.0)]));
    let total = tri.group_by(&[]).unwrap();
    assert_eq!(total, keyed_triangle(&rows, &[]));
    let diagonal = |t: &Triangle, c: usize| -> f64 {
        let d = t.latest_diagonal();
        (0..t.shape()[0])
            .map(|i| d.values(i, c).iter().sum::<f64>())
            .sum()
    };
    for c in 0..2 {
        let segments = diagonal(&tri, c);
        assert!((diagonal(&by_lob, c) - segments).abs() < 1e-6 * segments);
        assert!((diagonal(&total, c) - segments).abs() < 1e-6 * segments);
    }

    // Chain ladder on a group equals chain ladder on the summed triangle.
    let auto = by_lob.select(&[("lob", &["Auto"])]).unwrap();
    let auto_rows: Vec<usize> = (0..rows.0.len()).filter(|&r| rows.0[r] == "Auto").collect();
    let summed = Triangle::from_long(&Long {
        keys: &[],
        origin: &auto_rows.iter().map(|&r| rows.2[r]).collect::<Vec<_>>(),
        development: DevelopmentColumn::Age(
            &auto_rows.iter().map(|&r| rows.3[r]).collect::<Vec<_>>(),
        ),
        values: &[
            (
                "paid",
                &auto_rows.iter().map(|&r| rows.4[r]).collect::<Vec<_>>(),
            ),
            (
                "incurred",
                &auto_rows.iter().map(|&r| rows.5[r]).collect::<Vec<_>>(),
            ),
        ],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap();
    for column in ["paid", "incurred"] {
        let grouped = ChainLadder::default().fit(&auto, column).unwrap();
        let direct = ChainLadder::default().fit(&summed, column).unwrap();
        for (a, b) in grouped.ultimate.iter().zip(&direct.ultimate) {
            assert!((a - b).abs() <= 1e-9 * b.abs(), "{column}: {a} vs {b}");
        }
        assert_eq!(grouped.development.ldf, direct.development.ldf);
    }
    let all = ChainLadder::default().fit(&total, "paid").unwrap();
    let direct = ChainLadder::default()
        .fit(&keyed_triangle(&rows, &[]), "paid")
        .unwrap();
    assert_eq!(all.ultimate, direct.ultimate);

    // Fitting needs one segment.
    assert!(ChainLadder::default().fit(&tri, "paid").is_err());
    // Errors by name.
    assert!(tri.select(&[("line", &["Auto"])]).is_err());
    assert!(tri.select(&[("coverage", &["GL"])]).is_err());
    assert!(tri.select_columns(&["reported"]).is_err());
    assert!(tri.group_by(&["coverage", "coverage"]).is_err());
}

#[test]
fn every_segment_at_once_on_lob_and_coverage() {
    let rows = lob_coverage_rows();
    let tri = keyed_triangle(&rows, &[("lob", &rows.0), ("coverage", &rows.1)]);
    let alone = |label: &Label| {
        let parts = label.parts();
        tri.select(&[
            ("lob", &[parts[0].as_str()]),
            ("coverage", &[parts[1].as_str()]),
        ])
        .unwrap()
    };

    // Each segment's fit equals fitting that segment alone, exactly.
    let cl = ChainLadder::default().fit_segments(&tri, "paid").unwrap();
    let mack = Mack::default().fit_segments(&tri, "paid").unwrap();
    assert_eq!(cl.key_names, ["lob", "coverage"]);
    assert_eq!(cl.labels, tri.index());
    for (label, fit) in cl.iter() {
        assert_eq!(
            *fit,
            ChainLadder::default().fit(&alone(label), "paid").unwrap()
        );
        let m = mack.get(label).unwrap();
        assert_eq!(*m, Mack::default().fit(&alone(label), "paid").unwrap());
    }

    // Long results: one row per segment × origin, segment, or segment × age.
    let long = mack.to_long();
    assert_eq!(long.n_rows(), 40);
    assert_eq!(long.keys[0].0, "lob");
    assert_eq!(long.keys[1].1[10], "PD");
    assert_eq!(long.origin.as_ref().unwrap()[10], Period::year(2011));
    let reserve: Vec<f64> = mack
        .fits
        .iter()
        .flat_map(|f| f.chain_ladder.reserves())
        .collect();
    assert_eq!(long.column("reserve").unwrap(), reserve);
    let se: Vec<f64> = mack
        .fits
        .iter()
        .flat_map(|f| f.standard_error.clone())
        .collect();
    assert_eq!(long.column("standard_error").unwrap(), se);
    let totals = mack.totals();
    assert_eq!(totals.n_rows(), 4);
    for (s, fit) in mack.fits.iter().enumerate() {
        assert_eq!(
            totals.column("reserve").unwrap()[s],
            fit.chain_ladder.total_reserve()
        );
        assert_eq!(
            totals.column("standard_error").unwrap()[s],
            fit.total_standard_error
        );
    }
    let sum: f64 = totals.column("reserve").unwrap().iter().sum();
    assert!((mack.total_reserve() - sum).abs() < 1e-9 * sum);
    assert!((cl.total_reserve() - sum).abs() < 1e-9 * sum);
    let dev = cl.development_table();
    assert_eq!(dev.n_rows(), 40);
    assert_eq!(dev.age.as_ref().unwrap()[..2], [12, 24]);
    assert_eq!(
        dev.column("ldf").unwrap()[10..19],
        cl.fits[1].development.ldf[..]
    );
    assert!(dev.column("ldf").unwrap()[19].is_nan());
    assert_eq!(dev.column("cdf").unwrap()[19], 1.0);

    // Picking one segment by name.
    let home_pd = cl.segment(&[("lob", "Home"), ("coverage", "PD")]).unwrap();
    assert_eq!(home_pd.labels, [Label::new(["Home", "PD"])]);
    assert_eq!(home_pd.fits[0], cl.fits[3]);
    assert_eq!(
        cl.position(&[("lob", "Auto")]),
        Err(prospicio_reserving::Error::AmbiguousSegment(2))
    );
    assert!(cl.position(&[("line", "Auto")]).is_err());
    assert!(cl.position(&[("lob", "Farm")]).is_err());

    // The bootstrap of every segment: one joint distribution over
    // lob × coverage × origin, each segment's scale its own.
    let boot = OdpBootstrap {
        n_sims: 2_000,
        seed: 11,
        process: ProcessDistribution::Gamma,
        ..Default::default()
    };
    let joint = boot.fit_segments(&tri, "paid").unwrap();
    assert_eq!(joint.reserves.dims(), ["lob", "coverage", "origin"]);
    assert_eq!(joint.reserves.n_components(), 40);
    for (label, fit) in joint.segments.iter() {
        let single = boot.fit(&alone(label), "paid").unwrap();
        assert_eq!(fit.scale, single.scale);
        assert_eq!(fit.residuals.len(), single.residuals.len());
        assert_eq!(fit.chain_ladder, single.chain_ladder);
    }
    let again = boot.fit_segments(&tri, "paid").unwrap();
    assert_eq!(joint.reserves.draw_matrix(), again.reserves.draw_matrix());
    let marginal = |fits: &OdpBootstrapFits| {
        fits.segment(&[("lob", "Home"), ("coverage", "BI")])
            .unwrap()
            .reserves
    };
    let home_bi = marginal(&joint);
    assert_eq!(home_bi.n_components(), 10);
    assert_eq!(home_bi.draw_matrix(), marginal(&again).draw_matrix());
    let key = vec!["Home".into(), "BI".into(), Period::year(2015).into()];
    assert_eq!(
        joint.reserves.marginal(&key).unwrap().mean(),
        home_bi.marginal(&key).unwrap().mean()
    );

    // Aggregating over origin keeps the dependence between segments: the
    // total is the same however it is grouped.
    let by_lob = joint.reserves.aggregate(&["lob"]).unwrap();
    assert_eq!(by_lob.n_components(), 2);
    let by_segment = joint.reserves.aggregate(&["lob", "coverage"]).unwrap();
    assert_eq!(by_segment.n_components(), 4);
    let total = joint.reserves.total().mean();
    assert!((by_lob.total().mean() - total).abs() < 1e-9 * total);
    let totals = joint.totals();
    assert_eq!(
        totals.column("scale").unwrap()[2],
        joint.segments.fits[2].scale
    );
    for s in 0..4 {
        let mean = totals.column("mean").unwrap()[s];
        let cl_reserve = totals.column("reserve").unwrap()[s];
        assert!(
            (mean / cl_reserve - 1.0).abs() < 0.1,
            "{mean} vs {cl_reserve}"
        );
    }

    // Each row's mean and std_dev are those of its own reserve component,
    // and each segment total's those of its part of the joint draws.
    let long = joint.to_long();
    assert_eq!(long.n_rows(), 40);
    for (j, key) in joint.reserves.components().iter().enumerate() {
        let component = joint.reserves.marginal(key).unwrap();
        assert_eq!(long.column("mean").unwrap()[j], component.mean());
        assert_eq!(long.column("std_dev").unwrap()[j], component.std_dev());
    }
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs();
    for (s, label) in joint.segments.labels.iter().enumerate() {
        let parts = label.parts();
        let one = joint
            .segment(&[("lob", &parts[0]), ("coverage", &parts[1])])
            .unwrap();
        let total = one.reserves.total();
        assert!(close(totals.column("mean").unwrap()[s], total.mean()));
        assert!(close(totals.column("std_dev").unwrap()[s], total.std_dev()));
        let by_key = by_segment
            .marginal(&vec![parts[0].as_str().into(), parts[1].as_str().into()])
            .unwrap();
        assert!(close(by_key.mean(), total.mean()));
    }
}
