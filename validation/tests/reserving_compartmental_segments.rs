//! The compartmental model over segments and in the diagonal backtest
//! (`Compartmental::fit_segments` and `CompartmentalCandidate`,
//! `docs/design/reserving-compartmental.md`, decision 8) on the papers'
//! company 337 workers' compensation (`data/clrd_wkcomp_337.csv`, the cells
//! at calendar years up to 1996 and 1997 with its premium only, outstanding
//! = incurred - paid, direct premium), at the model's defaults (model 1:
//! 4 chains of 1,000 warm-up and 2,000 kept draws, target acceptance 0.99,
//! 10,000 simulations; the backtest's 4,000).
//!
//! * `Independent`, twin segments (company 337 under two names): the rank
//!   correlation of their reserves is within `4.5/sqrt(n_sims)` of 0 (the
//!   sd of the rank correlation of independent draws is about
//!   `1/sqrt(n_sims)`), and their mean reserves agree.
//! * `RankCorrelation` on three copies of company 337, target Spearman 0.6,
//!   0.3 and 0.4: each pair within 4 seed-to-seed sds of its target.
//! * The diagonal backtest, latest three diagonals, the compartmental
//!   model 1 next to the ODP GLM and `BayesOdp`: each split's A/E, the mean
//!   cell CRPS and total CRPS as ratios to the GLM's, and the mean log
//!   density less the GLM's, for the paid increments; and the held-out
//!   outstanding diagonals as a second table, scored with the same
//!   metrics (the model's own mean cell CRPS, total CRPS and mean log
//!   density, since no other model gives an outstanding). The 90% coverage
//!   and worst cell log density are reported, not checked.
//! * Model 2 (log-normal, loss ratios, incremental paid, effects by cell;
//!   the heavy sampler of decision 5) runs only under `full_bayes()`
//!   (`RISK_RS_FULL_BAYES`, read at run time): twin segments and the same
//!   backtest on the full square's latest diagonal (the 1997 diagonal held
//!   out leaves the papers' 90 cells for 87 parameters; two held out leave
//!   fewer cells than parameters), checked for sanity (finite scores, the tables agreeing on
//!   their rows) and reported.
//!
//! Every checked row's target is its mean over seeds 0 to `n - 1` of the
//! sweep and its tolerance at least 6 times its seed-to-seed sd there (and
//! 1.25 times its largest distance from the mean) ([`TOLERANCES`]); the
//! twins' Spearman row takes `4.5/sqrt(n_sims)`. The sweep is the ignored
//! `seed_sweep` test in a release build: `SWEEP_SEEDS=20 cargo test
//! --release -p prospicio-validation --features slow-bayes --test
//! reserving_compartmental_segments -- --ignored seed_sweep --nocapture`
//! (`SWEEP_START` moves the first seed, `SWEEP_ONLY` keeps the named
//! groups of `twins`, `ranked` and `backtest`, `SWEEP_CSV` writes every row
//! at every seed).

use std::sync::OnceLock;
use std::time::Instant;

use prospicio_glm::Glm;
use prospicio_models::Terms;
use prospicio_prob::PredictiveDistribution;
use prospicio_reserving::bayes::{
    BayesOdp, Compartmental, CompartmentalCandidate, CompartmentalPriors, Outstanding,
};
use prospicio_reserving::{
    Backtest, DevelopmentColumn, GlmCandidate, Grain, Lag, Long, Month, SegmentDependence,
    Triangle, TriangleFrame, TriangleModel, diagonal_backtest,
};
use prospicio_validation::full_bayes;

/// The rows of `data/clrd_wkcomp_337.csv` at calendar years up to 1996 and
/// 1997 with its premium only: origin, age, paid, incurred, premium.
fn training_rows() -> Vec<(i32, Lag, f64, f64, f64)> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/clrd_wkcomp_337.csv");
    let text = std::fs::read_to_string(path).expect("data/clrd_wkcomp_337.csv");
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    assert_eq!(
        lines.next(),
        Some("origin,development,paid,incurred,case_incurred,premium,premium_net")
    );
    let all: Vec<(i32, Lag, f64, f64, f64)> = lines
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (
                f[0].parse().unwrap(),
                f[1].parse().unwrap(),
                f[2].parse().unwrap(),
                f[3].parse().unwrap(),
                f[5].parse().unwrap(),
            )
        })
        .collect();
    let mut rows: Vec<_> = all
        .iter()
        .copied()
        .filter(|r| r.0 + (r.1 / 12) as i32 - 1 <= 1996)
        .collect();
    assert_eq!(rows.len(), 45);
    let premium_1997 = all.iter().find(|r| r.0 == 1997).unwrap().4;
    rows.push((1997, 12, f64::NAN, f64::NAN, premium_1997));
    rows
}

/// Company 337's training triangle under each of `names` as segments of a
/// `company` key (a single unkeyed segment when `names` is empty).
fn training(names: &[&str]) -> Triangle {
    let rows = training_rows();
    let copies: Vec<Option<&str>> = if names.is_empty() {
        vec![None]
    } else {
        names.iter().map(|n| Some(*n)).collect()
    };
    let (mut keys, mut origin, mut ages) = (vec![], vec![], vec![]);
    let (mut paid, mut incurred, mut premium) = (vec![], vec![], vec![]);
    for name in copies {
        for r in &rows {
            keys.push(name.unwrap_or(""));
            origin.push(Month::january(r.0));
            ages.push(r.1);
            paid.push(r.2);
            incurred.push(r.3);
            premium.push(r.4);
        }
    }
    let key_columns: Vec<(&str, &[&str])> = if names.is_empty() {
        vec![]
    } else {
        vec![("company", &keys)]
    };
    Triangle::from_long(&Long {
        keys: &key_columns,
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[
            ("paid", &paid),
            ("incurred", &incurred),
            ("premium", &premium),
        ],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

/// The cells at calendar years up to 1997 (the 55 cells of the upper
/// triangle; the file holds the whole square), 1997 diagonal included: holding that one out
/// leaves the papers' training triangle (90 cells), the fewest model 2
/// (87 parameters) can be fitted to.
fn full_square() -> Triangle {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/clrd_wkcomp_337.csv");
    let text = std::fs::read_to_string(path).expect("data/clrd_wkcomp_337.csv");
    let (mut origin, mut ages) = (vec![], vec![]);
    let (mut paid, mut incurred, mut premium) = (vec![], vec![], vec![]);
    for l in text.lines().filter(|l| !l.starts_with('#')).skip(1) {
        let f: Vec<&str> = l.split(',').collect();
        let (year, age): (i32, Lag) = (f[0].parse().unwrap(), f[1].parse().unwrap());
        if year + (age / 12) as i32 - 1 > 1997 {
            continue;
        }
        origin.push(Month::january(year));
        ages.push(age);
        paid.push(f[2].parse::<f64>().unwrap());
        incurred.push(f[3].parse::<f64>().unwrap());
        premium.push(f[5].parse::<f64>().unwrap());
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[
            ("paid", &paid),
            ("incurred", &incurred),
            ("premium", &premium),
        ],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

fn single() -> &'static Triangle {
    static TRIANGLE: OnceLock<Triangle> = OnceLock::new();
    TRIANGLE.get_or_init(|| training(&[]))
}

fn incurred() -> Outstanding<'static> {
    Outstanding::FromIncurred("incurred")
}

/// Each segment's reserve total, simulation by simulation.
fn by_company(pd: &PredictiveDistribution) -> Vec<Vec<f64>> {
    let by = pd.aggregate(&["company"]).unwrap();
    let n = by.n_components();
    (0..n)
        .map(|j| by.draw_matrix().chunks(n).map(|r| r[j]).collect())
        .collect()
}

fn ranks(x: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut r = vec![0.0; x.len()];
    for (k, &i) in idx.iter().enumerate() {
        r[i] = k as f64;
    }
    r
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let (rx, ry) = (ranks(x), ranks(y));
    let m = (x.len() as f64 - 1.0) / 2.0;
    let sxy: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - m) * (b - m)).sum();
    let sxx: f64 = rx.iter().map(|a| (a - m).powi(2)).sum();
    sxy / sxx
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn odp_glm() -> GlmCandidate {
    GlmCandidate {
        name: "odp".into(),
        terms: Terms::new()
            .intercept()
            .factor("origin")
            .factor("development"),
        glm: Glm::over_dispersed_poisson(),
    }
}

/// A measured quantity: its name, its value, its target and the tolerance
/// on `|value - target|`.
struct Measured {
    name: String,
    value: f64,
    target: f64,
    tolerance: f64,
}

impl Measured {
    fn used(&self) -> f64 {
        (self.value - self.target).abs() / self.tolerance
    }
}

/// A measured row, checked against its target and tolerance in
/// [`TOLERANCES`], or only reported when it has none there.
fn row(name: impl Into<String>, value: f64) -> Measured {
    let name = name.into();
    let (target, tolerance) = TOLERANCES
        .iter()
        .find(|t| t.0 == name)
        .map_or((0.0, f64::INFINITY), |t| (t.1, t.2));
    Measured {
        name,
        value,
        target,
        tolerance,
    }
}

/// `4.5 / sqrt(n_sims)` at the default 10,000 simulations.
const TWINS_SPEARMAN: f64 = 0.045;

/// `(row, target, tolerance)`: each target is the row's mean over the
/// sweep's seeds 0 to 29 and each tolerance at least 6 times its sd there
/// (and 1.25 times its largest distance from the mean), from the
/// release-build sweep (the ODP GLM's rows are deterministic: 0.001); the
/// coverage and worst log density rows are reported, not checked.
const TOLERANCES: &[(&str, f64, f64)] = &[
    ("twins_spearman", 0.0, TWINS_SPEARMAN),
    ("twins_mean_ratio", 0.0, 0.00611),
    ("spearman_01", 0.0, 0.01),
    ("spearman_02", 0.0, 0.0153),
    ("spearman_12", 0.0, 0.0179),
    ("paid_odp_ae0", 0.89353, 0.001),
    ("paid_odp_ae1", 0.9217, 0.001),
    ("paid_odp_ae2", 1.0152, 0.001),
    ("paid_odp_crps", 1.0, 0.001),
    ("paid_odp_tcrps", 1.0, 0.001),
    ("paid_odp_lpd", 0.0, 0.001),
    ("paid_bayes_odp_ae0", 0.89303, 0.00235),
    ("paid_bayes_odp_ae1", 0.92121, 0.00229),
    ("paid_bayes_odp_ae2", 1.0146, 0.00235),
    ("paid_bayes_odp_crps", 1.0037, 0.0507),
    ("paid_bayes_odp_tcrps", 1.0084, 0.075),
    ("paid_bayes_odp_lpd", 0.0023783, 0.0264),
    ("paid_compartmental_gm1_ae0", 1.3404, 0.0383),
    ("paid_compartmental_gm1_ae1", 1.0679, 0.0174),
    ("paid_compartmental_gm1_ae2", 1.1531, 0.0126),
    ("paid_compartmental_gm1_crps", 3.4863, 0.174),
    ("paid_compartmental_gm1_tcrps", 1.771, 0.167),
    ("paid_compartmental_gm1_lpd", -1.1504, 0.0367),
    ("os_compartmental_gm1_ae0", 1.0819, 0.0136),
    ("os_compartmental_gm1_ae1", 1.0252, 0.00975),
    ("os_compartmental_gm1_ae2", 0.97636, 0.00659),
    ("os_compartmental_gm1_crps", 1881.9, 78.5),
    ("os_compartmental_gm1_tcrps", 3617.3, 248.0),
    ("os_compartmental_gm1_lpd", -9.5594, 0.0105),
];

fn assert_all(measured: &[Measured]) {
    for m in measured {
        eprintln!(
            "{:40} {:>12.5} target {:>9.4} tolerance {:.4} (used {:.2})",
            m.name,
            m.value,
            m.target,
            m.tolerance,
            m.used()
        );
    }
    for m in measured {
        assert!(
            m.used() <= 1.0,
            "{}: {} is not within {} of {}",
            m.name,
            m.value,
            m.tolerance,
            m.target
        );
    }
}

fn model_1(seed: u64) -> Compartmental {
    Compartmental {
        seed,
        ..Compartmental::new(CompartmentalPriors::gesmann_morris())
    }
}

fn model_2(seed: u64) -> Compartmental {
    Compartmental {
        seed,
        ..Compartmental::model_2(CompartmentalPriors::gesmann_morris_2())
    }
}

/// Twin segments (company 337 under two names), `Independent`.
fn twins(prefix: &str, model: &Compartmental) -> Vec<Measured> {
    let t = Instant::now();
    let tri = training(&["a", "b"]);
    let fits = model
        .fit_segments(&tri, "paid", incurred(), "premium")
        .unwrap();
    let lines = by_company(&fits.reserves);
    eprintln!("{prefix} twins in {:.2?}", t.elapsed());
    vec![
        row(
            format!("{prefix}twins_spearman"),
            spearman(&lines[0], &lines[1]),
        ),
        row(
            format!("{prefix}twins_mean_ratio"),
            mean(&lines[0]) / mean(&lines[1]) - 1.0,
        ),
    ]
}

/// The target Spearman matrix of `ranked`.
const TARGET: [f64; 9] = [1.0, 0.6, 0.3, 0.6, 1.0, 0.4, 0.3, 0.4, 1.0];

/// Three copies of company 337, rank correlated.
fn ranked(seed: u64) -> Vec<Measured> {
    let t = Instant::now();
    let tri = training(&["a", "b", "c"]);
    let ranked = Compartmental {
        dependence: SegmentDependence::RankCorrelation {
            spearman: TARGET.to_vec(),
        },
        ..model_1(seed)
    }
    .fit_segments(&tri, "paid", incurred(), "premium")
    .unwrap();
    let lines = by_company(&ranked.reserves);
    eprintln!("rank correlation in {:.2?}", t.elapsed());
    [(0, 1), (0, 2), (1, 2)]
        .iter()
        .map(|&(i, j)| {
            row(
                format!("spearman_{i}{j}"),
                spearman(&lines[i], &lines[j]) - TARGET[i * 3 + j],
            )
        })
        .collect()
}

/// The scores of one backtest table's model `m`, as rows named `{name}_*`:
/// each split's A/E, the mean cell CRPS and total CRPS divided by `crps0`
/// and `tcrps0`, the coverage, and the mean log density less `lpd0`.
fn table_rows(
    name: &str,
    bt: &Backtest,
    m: usize,
    (crps0, tcrps0, lpd0): (f64, f64, f64),
) -> Vec<Measured> {
    let mut out = Vec::new();
    for (s, v) in bt.split_scores(m, 2).iter().enumerate() {
        out.push(row(format!("{name}_ae{s}"), *v));
    }
    out.push(row(format!("{name}_crps"), bt.mean(m, 0) / crps0));
    out.push(row(format!("{name}_tcrps"), bt.mean(m, 3) / tcrps0));
    out.push(row(format!("{name}_cover"), bt.mean(m, 1)));
    let lpd = bt.log_densities()[m]
        .as_ref()
        .unwrap_or_else(|| panic!("{name} gives log densities"));
    out.push(row(format!("{name}_lpd"), mean(lpd) - lpd0));
    let worst = lpd.iter().copied().fold(f64::INFINITY, f64::min);
    out.push(row(format!("{name}_lpd_min"), worst));
    out
}

/// The diagonal backtest, latest `n_diagonals` diagonals: the ODP GLM,
/// `BayesOdp` and the compartmental `model` for the paid increments, then
/// the model's held-out outstanding.
fn backtest(
    prefix: &str,
    tri: &Triangle,
    model: Compartmental,
    n_diagonals: usize,
    seed: u64,
) -> Vec<Measured> {
    let t = Instant::now();
    let cells = TriangleFrame::new(tri, "paid", None).unwrap();
    let glm = odp_glm();
    let bayes = BayesOdp::default();
    let candidate = CompartmentalCandidate::new(model, tri, "paid", incurred(), "premium");
    let models: Vec<&dyn TriangleModel> = vec![&glm, &bayes, &candidate];
    let paid = diagonal_backtest(&cells, &models, n_diagonals, 4000, seed, 0.9).unwrap();
    let t_paid = t.elapsed();
    let outstanding = candidate
        .outstanding_backtest(&cells, n_diagonals, 4000, seed, 0.9)
        .unwrap();
    eprintln!(
        "{prefix} backtest in {t_paid:.2?} (paid), {:.2?} (with the outstanding, on the shared fits)",
        t.elapsed()
    );
    eprintln!(
        "{prefix} scored rows {:?}, excluded {:?}",
        paid.scored_rows().iter().map(Vec::len).collect::<Vec<_>>(),
        paid.excluded()
    );
    // The paid table and the outstanding table score the same cells.
    assert_eq!(paid.scored_rows(), outstanding.scored_rows());
    assert_eq!(paid.excluded(), outstanding.excluded());
    let lpd: Vec<&Vec<f64>> = paid.log_densities().iter().flatten().collect();
    assert_eq!(lpd.len(), models.len(), "every model gives log densities");
    let relative = (paid.mean(0, 0), paid.mean(0, 3), mean(lpd[0]));
    let mut out = Vec::new();
    for (m, model) in paid.models().iter().enumerate() {
        out.extend(table_rows(
            &format!("{prefix}paid_{model}"),
            &paid,
            m,
            relative,
        ));
    }
    // No other model gives an outstanding: this table's own scores.
    out.extend(table_rows(
        &format!("{prefix}os_{}", candidate.name()),
        &outstanding,
        0,
        (1.0, 1.0, 0.0),
    ));
    out
}

#[test]
fn independent_segments_are_independent() {
    assert_all(&twins("", &model_1(0)));
}

#[test]
fn rank_correlation_reaches_the_target() {
    assert_all(&ranked(0));
}

#[test]
fn backtest_on_company_337() {
    assert_all(&backtest("", single(), model_1(0), 3, 0));
}

/// Model 2 (the heavy sampler of decision 5) runs only in the weekly and
/// manual runs (`RISK_RS_FULL_BAYES`).
#[test]
fn model_2_twins_and_backtest() {
    if !full_bayes() {
        eprintln!("model 2 runs only under RISK_RS_FULL_BAYES");
        return;
    }
    let measured = twins("gm2_", &model_2(0));
    assert_all(&measured);
    let rows = backtest("gm2_", &full_square(), model_2(0), 1, 0);
    for m in &rows {
        eprintln!("{:40} {:>12.5}", m.name, m.value);
        assert!(m.value.is_finite(), "{} is not finite", m.name);
    }
}

#[test]
#[ignore = "a release-build seed sweep"]
fn seed_sweep() {
    let env = |name: &str, default: u64| {
        std::env::var(name)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    };
    let (start, n) = (env("SWEEP_START", 0), env("SWEEP_SEEDS", 20));
    let only = std::env::var("SWEEP_ONLY").ok();
    let wanted = |g: &str| only.as_ref().is_none_or(|o| o.split(',').any(|b| b == g));
    let mut csv = std::env::var("SWEEP_CSV")
        .ok()
        .map(|p| std::fs::File::create(p).expect("SWEEP_CSV"));
    let runs: Vec<Vec<Measured>> = (start..start + n)
        .map(|seed| {
            let mut all = Vec::new();
            if wanted("twins") {
                all.extend(twins("", &model_1(seed)));
            }
            if wanted("ranked") {
                all.extend(ranked(seed));
            }
            if wanted("backtest") {
                all.extend(backtest("", single(), model_1(seed), 3, seed));
            }
            if let Some(f) = csv.as_mut() {
                use std::io::Write;
                for m in &all {
                    writeln!(f, "{seed},{},{}", m.name, m.value).unwrap();
                }
            }
            all
        })
        .collect();
    let mut failures = 0;
    for (q, m0) in runs[0].iter().enumerate() {
        let v: Vec<f64> = runs.iter().map(|r| r[q].value).collect();
        let m = mean(&v);
        let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt();
        let used = runs.iter().map(|r| r[q].used()).fold(0.0, f64::max);
        failures += runs.iter().filter(|r| r[q].used() > 1.0).count();
        println!(
            "{:40} mean {m:>11.5} sd {sd:>9.5} min {:>11.5} max {:>11.5} target {:>8.4} tol {:.4} \
             tol/sd {:>5.1} max used {used:.2}",
            m0.name,
            v.iter().copied().fold(f64::INFINITY, f64::min),
            v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            m0.target,
            m0.tolerance,
            m0.tolerance / sd
        );
    }
    assert_eq!(
        failures,
        0,
        "rows out of tolerance over seeds {start} to {}",
        start + n - 1
    );
}
