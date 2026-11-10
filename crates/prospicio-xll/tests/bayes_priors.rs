//! A new underwriting year from `PRO.RES.BAYES.BF`, its pattern prior, and
//! `PRO.RES.INDUSTRY.PRIOR` (`docs/design/xll.md`, decision 54), each the
//! library's result exactly: the fits are reproducible by seed, and the
//! industry prior is a closed form of the triangles. The library's own
//! tests check these against exact distributions and simulated panels
//! (`validation/tests/reserving_bayes_new_year.rs`,
//! `reserving_industry_prior.rs`).

use crate::common;

use common::{genins, n, num, t};
use prospicio_bayes::nuts::Sampler;
use prospicio_prob::Distribution;
use prospicio_reserving::bayes::{BayesBfNormalised, BayesBfNormalisedFit};
use prospicio_reserving::{
    DevelopmentColumn, Grain, IndustryPrior, IndustryPriorSettings, Long, Month, PatternPrior,
    Triangle,
};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

const GENINS_PREMIUM_NEW: f64 = 11_000_000.0;

/// `PRO.RES.BAYES.BF` on GenIns with a common multiplier: 300 draws, seed
/// 5, 2000 simulations, and the trailing arguments given.
fn bf(output: &str, new_premium: Grid, pattern_mean: Grid, pattern_sd: Grid) -> Grid {
    let (tri, premium) = genins();
    call(
        "pro_res_bayes_bf",
        vec![
            tri,
            premium,
            n(0.65),
            n(0.2),
            t(output),
            n(300.0),
            n(4.0),
            n(5.0),
            n(2000.0),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            n(0.3),
            new_premium,
            pattern_mean,
            pattern_sd,
        ],
    )
}

fn model() -> BayesBfNormalised {
    BayesBfNormalised {
        apriori: 0.65,
        cv: 0.2,
        common_cv: Some(0.3),
        sampler: Sampler {
            chains: 4,
            tune: 300,
            draws: 300,
            ..BayesBfNormalised::default().sampler
        },
        n_sims: 2000,
        seed: 5,
        ..BayesBfNormalised::default()
    }
}

fn genins_triangle() -> Triangle {
    let (tri, premium) = genins();
    let premium: Vec<f64> = premium.cells.iter().map(num).collect();
    prospicio_xll::reserving::triangle_of(&tri, Some(&premium), true).unwrap()
}

fn library_fit(prior: Option<&PatternPrior>) -> BayesBfNormalisedFit {
    let tri = genins_triangle();
    match prior {
        Some(p) => model().fit_with_pattern_prior(&tri, "values", "exposure", p),
        None => model().fit(&tri, "values", "exposure"),
    }
    .unwrap()
}

#[test]
fn the_new_year_is_the_librarys() {
    let fit = library_fit(None);
    let year = fit.new_year(GENINS_PREMIUM_NEW, 2000, 5).unwrap();
    let g = bf(
        "new_year",
        n(GENINS_PREMIUM_NEW),
        Grid::missing(),
        Grid::missing(),
    );
    assert_eq!(g.rows, 3, "{:?}", g.as_scalar());
    assert!(g.columns >= 4);
    assert_eq!(g.get(0, 0), &Value::Text(year.origin.to_string()));
    assert_eq!(g.get(1, 0), &Value::from("Loss ratio"));
    assert_eq!(num(g.get(1, 1)), year.loss_ratio.mean());
    assert_eq!(num(g.get(1, 2)), year.loss_ratio.std_dev());
    assert_eq!(num(g.get(2, 1)), year.ultimate.mean());
    // The default percentiles start at the 75th.
    assert_eq!(g.get(0, 3), &Value::from("P75"));
    assert_eq!(num(g.get(2, 3)), year.ultimate.quantile(0.75).unwrap());
    // The view needs the premium.
    let missing = bf(
        "new_year",
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
    );
    match missing.as_scalar() {
        Some(Value::Text(s)) => assert!(s.contains("new_premium"), "{s}"),
        other => panic!("expected an error, got {other:?}"),
    }
}

#[test]
fn a_pattern_prior_is_the_librarys() {
    // A prior near GenIns' own pattern: log share ratios to the first age.
    let mean = [0.9, 0.7, 0.5, 0.2, -0.1, -0.4, -0.8, -1.2, -1.6];
    let sd = [0.2; 9];
    let ages: Vec<u32> = (1..=10).map(|k| 12 * k).collect();
    let prior = PatternPrior::new(&ages, &mean, &sd).unwrap();
    let fit = library_fit(Some(&prior));
    // Under the triangle's columns, the first blank.
    let row = |v: &[f64]| {
        let mut cells = vec![Value::Empty];
        cells.extend(v.iter().map(|&x| Value::Number(x)));
        Grid::from_rows(vec![cells])
    };
    let g = bf("reserves", Grid::missing(), row(&mean), row(&sd));
    assert_eq!(
        num(g.get(11, 2)),
        fit.reserves.mean(),
        "{:?}",
        g.as_scalar()
    );
    // One per age after the first works too.
    let g = bf(
        "reserves",
        Grid::missing(),
        Grid::row(&mean),
        Grid::row(&sd),
    );
    assert_eq!(num(g.get(11, 2)), fit.reserves.mean());
    // Both or neither.
    let g = bf(
        "reserves",
        Grid::missing(),
        Grid::row(&mean),
        Grid::missing(),
    );
    match g.as_scalar() {
        Some(Value::Text(s)) => assert!(s.contains("together"), "{s}"),
        other => panic!("expected an error, got {other:?}"),
    }
}

/// Twelve companies, 6 x 6 triangles with patterns and loss ratios that
/// differ by company and origin; the cells as a stacked sheet range.
fn panel() -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let base = [0.4, 0.25, 0.15, 0.1, 0.06, 0.04];
    let (mut companies, mut premiums) = (Vec::new(), Vec::new());
    for c in 0..12 {
        let shift = 0.15 * ((c as f64) * 1.7).sin();
        let shares: Vec<f64> = base
            .iter()
            .enumerate()
            .map(|(d, s)| s * (shift * d as f64 / 5.0).exp())
            .collect();
        let total: f64 = shares.iter().sum();
        let lr = 0.7 * (1.0 + 0.12 * ((c as f64) * 2.3).cos());
        let premium = 1000.0 + 100.0 * c as f64;
        let mut rows = Vec::new();
        for o in 0..6 {
            let ult = premium * lr * (1.0 + 0.08 * ((o + 3 * c) as f64).sin());
            let mut cum = 0.0;
            rows.push(
                (0..6 - o)
                    .map(|d| {
                        cum += ult * shares[d] / total;
                        cum
                    })
                    .collect(),
            );
        }
        companies.push(rows);
        premiums.push(vec![premium; 6]);
    }
    (companies, premiums)
}

fn sheet(panel: &[Vec<Vec<f64>>], premiums: &[Vec<f64>]) -> (Grid, Grid, Grid) {
    let mut rows = Vec::new();
    let (mut labels, mut money) = (Vec::new(), Vec::new());
    for (c, company) in panel.iter().enumerate() {
        for (o, row) in company.iter().enumerate() {
            rows.push(
                (0..6)
                    .map(|d| row.get(d).map_or(Value::Empty, |&x| Value::Number(x)))
                    .collect(),
            );
            labels.push(vec![Value::Text(format!("Co {c}"))]);
            money.push(vec![Value::Number(premiums[c][o])]);
        }
    }
    (
        Grid::from_rows(rows),
        Grid::from_rows(labels),
        Grid::from_rows(money),
    )
}

fn library(panel: &[Vec<Vec<f64>>], premiums: &[Vec<f64>]) -> Vec<Triangle> {
    let _ = premiums;
    panel
        .iter()
        .map(|company| {
            let (mut origin, mut age, mut values) = (Vec::new(), Vec::new(), Vec::new());
            for (o, row) in company.iter().enumerate() {
                for (d, &x) in row.iter().enumerate() {
                    origin.push(Month::january(2001 + o as i32));
                    age.push(12 * (d as u32 + 1));
                    values.push(x);
                }
            }
            Triangle::from_long(&Long {
                keys: &[],
                origin: &origin,
                development: DevelopmentColumn::Age(&age),
                values: &[("values", &values)],
                origin_grain: Grain::Year,
                development_grain: Grain::Year,
                cumulative: true,
            })
            .unwrap()
        })
        .collect()
}

fn prior_call(args: Vec<Grid>) -> Grid {
    let g = call("pro_res_industry_prior", args);
    if let Some(Value::Text(s)) = g.as_scalar() {
        assert!(!s.starts_with("#PROSPICIO!"), "{s}");
    }
    g
}

#[test]
fn the_industry_prior_is_the_librarys() {
    let (panel, premiums) = panel();
    let (tri, companies, money) = sheet(&panel, &premiums);
    let want =
        IndustryPrior::from_triangles(&library(&panel, &premiums), "values", &premiums).unwrap();
    let summary = prior_call(vec![tri.clone(), companies.clone(), money.clone()]);
    let value = |label: &str| {
        let r = (0..summary.rows)
            .find(|&r| *summary.get(r, 0) == Value::from(label))
            .unwrap_or_else(|| panic!("no row {label}"));
        summary.get(r, 1).clone()
    };
    assert_eq!(value("Apriori"), Value::Number(want.apriori));
    assert_eq!(value("CV"), Value::Number(want.cv));
    assert_eq!(
        value("Common CV"),
        want.common_cv().map_or(Value::Empty, Value::Number)
    );
    assert_eq!(value("Between-company CV"), Value::Number(want.between_cv));
    assert_eq!(value("Within-company CV"), Value::Number(want.within_cv));
    assert_eq!(value("Tail factor"), Value::Number(want.tail_factor));
    assert_eq!(value("Companies used"), Value::Number(12.0));
    assert!(want.between_cv > 0.0 && want.cv > 0.0);

    let pattern = prior_call(vec![
        tri.clone(),
        companies.clone(),
        money.clone(),
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        t("pattern"),
    ]);
    assert_eq!((pattern.rows, pattern.columns), (7, 5));
    assert_eq!(pattern.get(1, 1), &Value::Empty);
    for k in 0..5 {
        assert_eq!(num(pattern.get(k + 2, 1)), want.pattern.mean[k]);
        assert_eq!(num(pattern.get(k + 2, 2)), want.pattern.sd[k]);
    }
    assert_eq!(num(pattern.get(6, 3)), want.percent_developed[5]);

    // A younger triangle's prior, and the settings.
    let settings = IndustryPriorSettings {
        min_developed: 0.6,
        subtract_process: true,
        ..IndustryPriorSettings::default()
    };
    let young = settings
        .from_triangles(&library(&panel, &premiums), "values", &premiums)
        .unwrap()
        .at_age(48)
        .unwrap();
    let got = prior_call(vec![
        tri.clone(),
        companies.clone(),
        money.clone(),
        n(0.6),
        Grid::scalar(Value::Bool(true)),
        n(48.0),
    ]);
    assert_eq!(*got.get(0, 1), Value::Number(young.apriori));
    assert_eq!(*got.get(2, 1), Value::Number(young.cv));
    let report = prior_call(vec![
        tri.clone(),
        companies.clone(),
        money.clone(),
        n(0.6),
        Grid::scalar(Value::Bool(true)),
        n(48.0),
        t("report"),
    ]);
    assert!(
        (0..report.rows)
            .any(|r| matches!(report.get(r, 1), Value::Text(s) if s.contains("48 months"))),
        "{report:?}"
    );

    // A company's rows must be together.
    let mut labels = companies.clone();
    labels.cells.swap(0, 6);
    match call("pro_res_industry_prior", vec![tri, labels, money]).as_scalar() {
        Some(Value::Text(s)) => assert!(s.contains("together"), "{s}"),
        other => panic!("expected an error, got {other:?}"),
    }
}
