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

use act_glm::Glm;
use act_models::Terms;
use act_models::stack::stacking_weights;
use act_prob::Distribution;
use act_reserving::{
    Average, ChainLadder, ChainLadderFit, Development, DevelopmentColumn, GlmCandidate, Grain,
    Label, Long, Mack, MackFit, Month, OdpBootstrap, OdpBootstrapFit, OdpBootstrapFits, OdpGlm,
    OdpGlmFit, Period, ProcessDistribution, SigmaInterpolation, Triangle, TriangleFrame,
    TriangleModel, diagonal_backtest,
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
        Err(act_reserving::Error::AmbiguousSegment(2))
    );
    assert!(cl.position(&[("line", "Auto")]).is_err());
    assert!(cl.position(&[("lob", "Farm")]).is_err());

    // The bootstrap of every segment: one joint distribution over
    // lob × coverage × origin, each segment's scale its own.
    let boot = OdpBootstrap {
        n_sims: 2_000,
        seed: 11,
        process: ProcessDistribution::Gamma,
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
        let want = by_segment.row(0).unwrap()[s];
        assert!(mean.is_finite() && want.is_finite());
        let cl_reserve = totals.column("reserve").unwrap()[s];
        assert!(
            (mean / cl_reserve - 1.0).abs() < 0.1,
            "{mean} vs {cl_reserve}"
        );
    }
    assert_eq!(joint.to_long().n_rows(), 40);
}
