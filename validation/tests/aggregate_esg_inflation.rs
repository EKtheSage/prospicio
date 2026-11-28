//! The ESG's inflation (`docs/design/esg-dfa.md`, validation 4): the
//! Ornstein-Uhlenbeck rate's moments and its price index's mean in closed
//! form, the correlation of the shocks, the superimposed AR(1)'s moments
//! and its shock's correlation with the year's innovations, and an
//! `IndexPaths` from the scenarios through `reinflate_draws` against
//! re-inflation by hand.

use prospicio_esg::{Cir, Esg, Index, Inflation, Scenarios, Step, Superimposed, Vasicek};
use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};
use prospicio_reserving::{CalendarIndex, IndexPaths, Indexation, Month, Period};

const SIMS: usize = 20_000;

/// The mean of a column over paths and its standard error.
fn column(f: impl Fn(usize) -> f64) -> (f64, f64) {
    let draws: Vec<f64> = (0..SIMS).map(f).collect();
    let n = SIMS as f64;
    let mean = draws.iter().sum::<f64>() / n;
    let var = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

/// The sample correlation of two columns.
fn corr(f: impl Fn(usize) -> f64, g: impl Fn(usize) -> f64) -> f64 {
    let (x, y): (Vec<f64>, Vec<f64>) = (0..SIMS).map(|s| (f(s), g(s))).unzip();
    let n = SIMS as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
    sxy / (sxx * syy).sqrt()
}

/// A correlation within four of its standard errors, `(1 - ρ²) / √n`.
fn near_corr(sample: f64, rho: f64, what: &str) {
    let se = (1.0 - rho * rho) / (SIMS as f64).sqrt();
    assert!(
        (sample - rho).abs() < 4.0 * se,
        "{what}: {sample} against {rho}"
    );
}

/// `(1 - e^(-c)) / c`, a kernel integral over one year.
fn e1(c: f64) -> f64 {
    -(-c).exp_m1() / c
}

fn vasicek() -> Vasicek {
    Vasicek::new(0.3, 0.04, 0.01, 0.02).unwrap()
}

fn cpi() -> Inflation {
    Inflation::new(0.5, 0.025, 0.01, 0.04).unwrap()
}

#[test]
fn the_inflation_rate_and_its_index_have_their_closed_form_moments() {
    let paths = Esg::new(vasicek())
        .with_inflation(cpi())
        .simulate(10, SIMS, 2026)
        .unwrap();
    let model = cpi();
    for years in [1usize, 10] {
        let (k, t) = (12 * years, years as f64);
        let (mean, se) = column(|s| paths.inflation_rate(s, k).unwrap());
        assert!((mean - model.mean(t)).abs() < 4.0 * se, "{mean} vs {}", model.mean(t));
        let var = column(|s| (paths.inflation_rate(s, k).unwrap() - model.mean(t)).powi(2)).0;
        assert!((var / model.variance(t) - 1.0).abs() < 0.05, "{var}");
        let (level, se) = column(|s| paths.index_level(Index::Price, s, k).unwrap());
        assert!(
            (level - model.index_mean(t)).abs() < 4.0 * se,
            "{level} vs {}",
            model.index_mean(t)
        );
    }
    // After ten years the rate is near its stationary law.
    assert!((model.variance(10.0) / model.stationary_variance() - 1.0).abs() < 1e-4);
}

#[test]
fn the_shocks_of_the_rate_and_inflation_are_correlated() {
    let rho = 0.6;
    let paths = Esg::new(vasicek())
        .with_inflation(cpi())
        .with_correlation(vec![vec![1.0, rho], vec![rho, 1.0]])
        .simulate(5, SIMS, 9)
        .unwrap();
    // Cov(r(t), i(t)) = ρ σ_r σ_i (1 - e^(-(a + κ) t)) / (a + κ).
    let (v, i, t) = (vasicek(), cpi(), 5.0);
    let cov = rho * v.sigma * i.sigma * e1((v.a + i.kappa) * t) * t;
    let theory = cov / (v.variance(t) * i.variance(t)).sqrt();
    let sample = corr(
        |s| paths.short_rate(s, 60).unwrap(),
        |s| paths.inflation_rate(s, 60).unwrap(),
    );
    near_corr(sample, theory, "rate and inflation at 5 years");
}

#[test]
fn the_superimposed_excess_is_an_ar1_correlated_with_the_year() {
    let sup = Superimposed::new(0.01, 0.6, 0.008).unwrap();
    let (rho_ri, rho_rs, rho_is) = (0.5, 0.2, 0.4);
    let paths = Esg::new(vasicek())
        .with_inflation(cpi())
        .with_superimposed(sup)
        .with_correlation(vec![
            vec![1.0, rho_ri, rho_rs],
            vec![rho_ri, 1.0, rho_is],
            vec![rho_rs, rho_is, 1.0],
        ])
        .simulate(4, SIMS, 31)
        .unwrap();
    for year in [1usize, 4] {
        let (mean, se) = column(|s| paths.superimposed(s, year).unwrap());
        assert!((mean - sup.mean).abs() < 4.0 * se, "{mean}");
        let var = column(|s| (paths.superimposed(s, year).unwrap() - sup.mean).powi(2)).0;
        assert!((var / sup.variance(year) - 1.0).abs() < 0.05, "{var}");
    }
    // Year 3's shock against the year's innovations of the rate and of
    // inflation: Corr = ρ E(a) / √E(2a), the kernels over one year.
    let shock = |s: usize| {
        let (now, before) = (
            paths.superimposed(s, 3).unwrap(),
            paths.superimposed(s, 2).unwrap(),
        );
        (now - sup.mean - sup.persistence * (before - sup.mean)) / sup.sigma
    };
    let (v, i) = (vasicek(), cpi());
    let innovation = |p: &Scenarios, s: usize, a: f64, rate: bool| {
        let at = |k| {
            if rate {
                p.short_rate(s, k).unwrap() - v.b
            } else {
                p.inflation_rate(s, k).unwrap() - i.mean
            }
        };
        at(36) - (-a).exp() * at(24)
    };
    near_corr(
        corr(shock, |s| innovation(&paths, s, i.kappa, false)),
        rho_is * e1(i.kappa) / e1(2.0 * i.kappa).sqrt(),
        "superimposed and inflation",
    );
    near_corr(
        corr(shock, |s| innovation(&paths, s, v.a, true)),
        rho_rs * e1(v.a) / e1(2.0 * v.a).sqrt(),
        "superimposed and the rate",
    );
    // The claims index is the price index times the excess compounded.
    for s in [0, 7] {
        let ratio = paths.index_level(Index::Claims, s, 24).unwrap()
            / paths.index_level(Index::Price, s, 24).unwrap();
        let excess = paths.superimposed(s, 1).unwrap() + paths.superimposed(s, 2).unwrap();
        assert!((ratio.ln() - excess).abs() < 1e-12);
    }
}

#[test]
fn index_paths_from_the_scenarios_reinflate_reserve_cells() {
    let esg = Esg::new(vasicek())
        .with_inflation(cpi())
        .with_superimposed(Superimposed::new(0.01, 0.6, 0.008).unwrap())
        .with_step(Step::Quarterly);
    let n = 50;
    let paths = esg.simulate(3, n, 5).unwrap();
    // An index given to 2022; the scenarios carry it on from 2023.
    let index = CalendarIndex::new(Period::year(2021), vec![1.0, 1.1]).unwrap();
    let levels: Vec<f64> = paths
        .index_levels(Index::Claims, Step::Annual)
        .unwrap()
        .iter()
        .map(|l| 1.1 * l)
        .collect();
    assert_eq!(levels.len(), 3 * n);
    let index_paths = IndexPaths::new(Period::year(2023), n, levels.clone()).unwrap();
    // Origin 2022's cells at ages 24 and 36 (paid in 2023 and 2024) and
    // origin 2021's at 48 (paid in 2024), at the 2022 level.
    let cell = |origin, age| vec![KeyValue::Period(Period::year(origin)), KeyValue::Int(age)];
    let components = vec![cell(2022, 24), cell(2022, 36), cell(2021, 48)];
    let base: Vec<f64> = (0..n * 3).map(|j| 100.0 + j as f64).collect();
    let draws = PredictiveDistribution::from_draws(
        vec!["origin".into(), "development".into()],
        components,
        base.clone(),
        Provenance::new("cells"),
    )
    .unwrap();
    let at_2022 = Indexation {
        index,
        base: Month::january(2022),
    };
    let out = at_2022.reinflate_draws(&draws, Some(&index_paths)).unwrap();
    for sim in [0, 17, n - 1] {
        let row = out.row(sim).unwrap();
        // By hand: the year's claims index level over 2022's 1.1.
        let level = |year: usize| {
            1.1 * paths
                .index_level(Index::Claims, sim, 4 * (year - 2022))
                .unwrap()
        };
        let expect = [
            base[sim * 3] * level(2023) / 1.1,
            base[sim * 3 + 1] * level(2024) / 1.1,
            base[sim * 3 + 2] * level(2024) / 1.1,
        ];
        for (got, want) in row.iter().zip(expect) {
            assert!((got - want).abs() < 1e-9 * want, "{got} vs {want}");
        }
    }
}

#[test]
fn the_correlation_is_checked() {
    let esg = Esg::new(vasicek()).with_inflation(cpi());
    let bad = |m: Vec<Vec<f64>>| esg.clone().with_correlation(m).simulate(1, 2, 1).is_err();
    assert!(bad(vec![vec![1.0]]));
    assert!(bad(vec![vec![1.0, 0.5], vec![0.4, 1.0]]));
    assert!(bad(vec![vec![1.0, 1.5], vec![1.5, 1.0]]));
    assert!(bad(vec![vec![0.9, 0.0], vec![0.0, 1.0]]));
    let cir = Esg::new(Cir::new(0.3, 0.04, 0.06, 0.03).unwrap()).with_inflation(cpi());
    assert!(cir.clone().with_correlation(vec![vec![1.0, 0.3], vec![0.3, 1.0]]).simulate(1, 2, 1).is_err());
    // Uncorrelated, CIR runs beside inflation.
    let paths = cir.simulate(2, 3, 1).unwrap();
    assert!(paths.index_level(Index::Price, 2, 24).unwrap() > 0.0);
    // Without inflation there is no price index, and a grain finer than
    // the step is refused.
    let rate_only = Esg::new(vasicek()).with_step(Step::Annual).simulate(1, 2, 1).unwrap();
    assert!(rate_only.index_level(Index::Price, 0, 1).is_err());
    assert!(rate_only.inflation_rate(0, 1).is_err());
    assert!(rate_only.superimposed(0, 1).is_err());
    let annual = esg.with_step(Step::Annual).simulate(2, 2, 1).unwrap();
    assert!(annual.index_levels(Index::Price, Step::Monthly).is_err());
}
