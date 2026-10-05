//! Reserving parity: Chain Ladder and Mack against R ChainLadder and
//! chainladder-python on RAA, GenIns and ABC.
//!
//! Every row of both reference files is evaluated; see
//! `scripts/reserving_r.R` and `scripts/reserving_chainladder_python.py`.
//! The ODP bootstrap is checked against R's `BootChainLadder`: exactly for
//! the scale and residuals, within Monte Carlo tolerances for the reserve
//! distribution (`scripts/reserving_bootstrap_r.R`). The calendar-diagonal
//! backtest is checked against Chain Ladder on the truncated triangle.

use std::collections::HashMap;

use act_glm::Glm;
use act_models::Terms;
use act_models::stack::stacking_weights;
use act_prob::Distribution;
use act_reserving::{
    Average, ChainLadder, ChainLadderFit, Development, DevelopmentColumn, GlmCandidate, Grain,
    Long, Mack, MackFit, Month, OdpBootstrap, OdpBootstrapFit, Period, ProcessDistribution,
    SigmaInterpolation, Triangle, TriangleFrame, TriangleModel, diagonal_backtest,
};
use act_validation::{Case, check, reference, triangle};

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
        ["crps", "coverage", "actual_vs_expected", "total_crps"]
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
    // The ODP model over-predicts the low 1990 diagonal: A/E 1.12, 0.66.
    let ae = bt.split_scores(0, 2);
    assert!(
        (ae[0] - 1.115).abs() < 1e-3 && (ae[1] - 0.658).abs() < 1e-3,
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
            index: None,
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
