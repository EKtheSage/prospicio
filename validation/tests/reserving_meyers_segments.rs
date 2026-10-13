//! Meyers' models over segments and in the diagonal backtest
//! (`Meyers::fit_segments` and `TriangleModel for Meyers`,
//! `docs/design/reserving-bayes.md`, decisions 8 and 11), on the lines of
//! the CAS loss reserve database (`data/clrd_lines.csv`, paid losses and net
//! earned premium summed over companies) and on Meyers' group 353
//! (`data/clrd_comauto_353.csv`, the cells at calendar years up to 1997).
//!
//! Every checked row's target is its mean over seeds 0 to 59 and its
//! tolerance at least 4 times its seed-to-seed sd there (and 1.25 times its
//! largest distance from the mean), so every row passes at each of those
//! seeds ([`TOLERANCES`]). No checked row is heavy-tailed: none of them
//! strays more than 4.1 sds from its mean. The sweep is the ignored
//! `seed_sweep` test in a release build: `SWEEP_SEEDS=60 cargo test
//! --release -p prospicio-validation --test reserving_meyers_segments --
//! --ignored seed_sweep --nocapture` (about 23 s a seed; `SWEEP_START`
//! moves the first seed). All at `Meyers`' defaults (4 chains of 1,000
//! warm-up and 2,000 kept draws, 10,000 simulations; the backtest's 4,000
//! draws).
//!
//! * `Independent`, twin lines (CLRD wkcomp under two names, CRC): the rank
//!   correlation of their totals is within `4.5/sqrt(n_sims)` of 0 (sd
//!   0.0083 over seeds, as for independent draws) and their mean totals
//!   agree within 0.62% (sd 0.15%).
//! * `RankCorrelation` on comauto, ppauto and wkcomp (CSR), target Spearman
//!   0.6, 0.3 and 0.4: each pair within 0.013 (sd at most 0.0030).
//! * The diagonal backtest, latest three diagonals, Meyers' CRC, CSR and
//!   CAY next to the ODP GLM and `BayesOdp` (paid), or next to each other
//!   (group 353 incurred, where the GLM's IRLS does not converge): each
//!   split's A/E, the mean cell CRPS and total CRPS as ratios to the first
//!   model's, and the mean log density less the first model's. The ODP
//!   rows and each model's 90% coverage and worst cell log density are
//!   reported, not checked. Meyers' models forecast well above the
//!   outcomes on group 353 (A/E 0.30 to 0.90; CSR the closest on the
//!   latest split, as Meyers finds CSR corrects CRC on paid data): the
//!   models' late `sigma_d` and young origins' level rest on few cells
//!   when diagonals are held out, and the mean of a log-normal grows with
//!   both. Their intervals cover every held-out cell of group 353 and
//!   comauto, and 0.92 to 1 of wkcomp's, where the ODP models cover 0.41
//!   to 0.46, so on wkcomp Meyers' log density beats the GLM's by 0.54 to
//!   0.60 a cell, while on comauto it is lower by 0.36 to 0.46 and on
//!   group 353 paid by 1.28 to 1.39.
//!
//! CI runs the twins, the rank correlation and two of the four backtests
//! (group 353 paid and wkcomp): about 81 s in a debug build.

use std::time::Instant;

use prospicio_glm::Glm;
use prospicio_models::Terms;
use prospicio_prob::PredictiveDistribution;
use prospicio_reserving::bayes::{BayesOdp, Meyers, MeyersModel};
use prospicio_reserving::{
    DevelopmentColumn, GlmCandidate, Grain, Lag, Long, Month, SegmentDependence, Triangle,
    TriangleFrame, TriangleModel, diagonal_backtest,
};

/// The lines of `data/clrd_lines.csv` named in `lines`, as segments of a
/// `lob` key, renamed (to repeat a line under two names), with columns
/// `paid` and `premium`.
fn clrd(lines: &[(&str, &str)]) -> Triangle {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/clrd_lines.csv");
    let text = std::fs::read_to_string(path).expect("data/clrd_lines.csv");
    let mut rows = text.lines().filter(|l| !l.starts_with('#'));
    assert_eq!(rows.next(), Some("lob,origin,development,paid,premium"));
    let rows: Vec<Vec<&str>> = rows.map(|l| l.split(',').collect()).collect();
    let (mut keys, mut origin, mut ages, mut paid, mut premium) =
        (vec![], vec![], vec![], vec![], vec![]);
    for &(line, name) in lines {
        for f in rows.iter().filter(|f| f[0] == line) {
            keys.push(name);
            origin.push(Month::january(f[1].parse().unwrap()));
            ages.push(f[2].parse().unwrap());
            paid.push(f[3].parse::<f64>().unwrap());
            premium.push(f[4].parse::<f64>().unwrap());
        }
    }
    Triangle::from_long(&Long {
        keys: &[("lob", &keys)],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("paid", &paid), ("premium", &premium)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

/// Group 353's training triangle (the cells at calendar years up to 1997)
/// with columns `paid`, `incurred` and `premium`.
fn group_353() -> Triangle {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/clrd_comauto_353.csv");
    let text = std::fs::read_to_string(path).expect("data/clrd_comauto_353.csv");
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    assert_eq!(
        lines.next(),
        Some("origin,development,paid,incurred,premium")
    );
    let (mut origin, mut ages, mut paid, mut incurred, mut premium) =
        (vec![], vec![], vec![], vec![], vec![]);
    for line in lines {
        let f: Vec<&str> = line.split(',').collect();
        let (year, age): (i32, Lag) = (f[0].parse().unwrap(), f[1].parse().unwrap());
        if year + (age / 12) as i32 - 1 > 1997 {
            continue;
        }
        origin.push(Month::january(year));
        ages.push(age);
        paid.push(f[2].parse::<f64>().unwrap());
        incurred.push(f[3].parse::<f64>().unwrap());
        premium.push(f[4].parse::<f64>().unwrap());
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

/// Each origin's premium in a single-segment triangle: the first value of
/// the `premium` column, constant across ages in these data.
fn premium_of(tri: &Triangle) -> Vec<f64> {
    let column = tri.column_position("premium").unwrap();
    (0..tri.origins().len())
        .map(|o| tri.get(0, column, o, 0).unwrap())
        .collect()
}

/// Each line's total, simulation by simulation.
fn by_line(pd: &PredictiveDistribution) -> Vec<Vec<f64>> {
    let by = pd.aggregate(&["lob"]).unwrap();
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
    fn new(name: impl Into<String>, value: f64, target: f64, tolerance: f64) -> Self {
        Self {
            name: name.into(),
            value,
            target,
            tolerance,
        }
    }

    /// A quantity reported by the sweep but not checked.
    fn info(name: impl Into<String>, value: f64) -> Self {
        Self::new(name, value, 0.0, f64::INFINITY)
    }

    fn used(&self) -> f64 {
        (self.value - self.target).abs() / self.tolerance
    }
}

/// A measured row, checked against its target and tolerance in
/// [`TOLERANCES`], or only reported when it has none there.
fn row(name: impl Into<String>, value: f64) -> Measured {
    let name = name.into();
    match TOLERANCES.iter().find(|t| t.0 == name) {
        Some(&(_, target, tolerance)) => Measured::new(name, value, target, tolerance),
        None => Measured::info(name, value),
    }
}

/// `4.5 / sqrt(n_sims)` at the default 10,000 simulations: the sd of the
/// rank correlation of independent draws is about `1 / sqrt(n_sims)`.
const TWINS_SPEARMAN: f64 = 0.045;

/// `(row, target, tolerance)`: each target is the row's mean over seeds 0
/// to 59 and each tolerance at least 4 times its sd there (and 1.25 times
/// its largest distance from the mean), from the release-build sweep.
const TOLERANCES: &[(&str, f64, f64)] = &[
    ("twins_spearman", 0.0, TWINS_SPEARMAN),
    ("twins_mean_ratio", 0.0, 0.0062),
    ("spearman_01", 0.0, 0.0076),
    ("spearman_02", 0.0, 0.011),
    ("spearman_12", 0.0, 0.013),
    ("g353_paid_meyers_crc_ae0", 0.3376, 0.053),
    ("g353_paid_meyers_crc_ae1", 0.5506, 0.033),
    ("g353_paid_meyers_crc_ae2", 0.6592, 0.018),
    ("g353_paid_meyers_crc_crps", 2.4395, 0.16),
    ("g353_paid_meyers_crc_tcrps", 4.8103, 0.59),
    ("g353_paid_meyers_crc_lpd", -1.3115, 0.036),
    ("g353_paid_meyers_csr_ae0", 0.3618, 0.058),
    ("g353_paid_meyers_csr_ae1", 0.6608, 0.038),
    ("g353_paid_meyers_csr_ae2", 0.8991, 0.028),
    ("g353_paid_meyers_csr_crps", 2.2833, 0.14),
    ("g353_paid_meyers_csr_tcrps", 3.7322, 0.5),
    ("g353_paid_meyers_csr_lpd", -1.2782, 0.032),
    ("g353_paid_meyers_cay_ae0", 0.2983, 0.047),
    ("g353_paid_meyers_cay_ae1", 0.4980, 0.031),
    ("g353_paid_meyers_cay_ae2", 0.6495, 0.017),
    ("g353_paid_meyers_cay_crps", 2.7703, 0.15),
    ("g353_paid_meyers_cay_tcrps", 5.5450, 0.61),
    ("g353_paid_meyers_cay_lpd", -1.3924, 0.05),
    ("g353_incurred_meyers_crc_ae0", 0.3200, 0.098),
    ("g353_incurred_meyers_crc_ae1", 0.4334, 0.027),
    ("g353_incurred_meyers_crc_ae2", 0.6271, 0.02),
    ("g353_incurred_meyers_csr_ae0", 0.3395, 0.072),
    ("g353_incurred_meyers_csr_ae1", 0.5018, 0.032),
    ("g353_incurred_meyers_csr_ae2", 0.7884, 0.038),
    ("g353_incurred_meyers_csr_crps", 0.9512, 0.079),
    ("g353_incurred_meyers_csr_tcrps", 0.8541, 0.13),
    ("g353_incurred_meyers_csr_lpd", 0.0120, 0.05),
    ("g353_incurred_meyers_cay_ae0", 0.2915, 0.057),
    ("g353_incurred_meyers_cay_ae1", 0.3916, 0.033),
    ("g353_incurred_meyers_cay_ae2", 0.6574, 0.031),
    ("g353_incurred_meyers_cay_crps", 1.1016, 0.082),
    ("g353_incurred_meyers_cay_tcrps", 1.0921, 0.16),
    ("g353_incurred_meyers_cay_lpd", -0.0633, 0.047),
    ("comauto_meyers_crc_ae0", 0.8044, 0.11),
    ("comauto_meyers_crc_ae1", 0.9242, 0.0057),
    ("comauto_meyers_crc_ae2", 0.8347, 0.0034),
    ("comauto_meyers_crc_crps", 1.4600, 0.14),
    ("comauto_meyers_crc_tcrps", 1.2699, 0.11),
    ("comauto_meyers_crc_lpd", -0.3831, 0.07),
    ("comauto_meyers_csr_ae0", 0.7606, 0.12),
    ("comauto_meyers_csr_ae1", 0.9541, 0.0076),
    ("comauto_meyers_csr_ae2", 0.8732, 0.004),
    ("comauto_meyers_csr_crps", 1.5947, 0.17),
    ("comauto_meyers_csr_tcrps", 1.1342, 0.12),
    ("comauto_meyers_csr_lpd", -0.4558, 0.071),
    ("comauto_meyers_cay_ae0", 0.8089, 0.087),
    ("comauto_meyers_cay_ae1", 0.9263, 0.006),
    ("comauto_meyers_cay_ae2", 0.8494, 0.003),
    ("comauto_meyers_cay_crps", 1.4035, 0.13),
    ("comauto_meyers_cay_tcrps", 1.1745, 0.11),
    ("comauto_meyers_cay_lpd", -0.3597, 0.081),
    ("wkcomp_meyers_crc_ae0", 0.7555, 0.099),
    ("wkcomp_meyers_crc_ae1", 0.9848, 0.0054),
    ("wkcomp_meyers_crc_ae2", 0.7468, 0.0025),
    ("wkcomp_meyers_crc_crps", 1.3133, 0.098),
    ("wkcomp_meyers_crc_tcrps", 1.0296, 0.054),
    ("wkcomp_meyers_crc_lpd", 0.5390, 0.14),
    ("wkcomp_meyers_csr_ae0", 0.7311, 0.13),
    ("wkcomp_meyers_csr_ae1", 1.0884, 0.0075),
    ("wkcomp_meyers_csr_ae2", 0.8109, 0.0036),
    ("wkcomp_meyers_csr_crps", 1.3088, 0.16),
    ("wkcomp_meyers_csr_tcrps", 0.8758, 0.079),
    ("wkcomp_meyers_csr_lpd", 0.5990, 0.13),
    ("wkcomp_meyers_cay_ae0", 0.7690, 0.11),
    ("wkcomp_meyers_cay_ae1", 1.0114, 0.0068),
    ("wkcomp_meyers_cay_ae2", 0.7362, 0.002),
    ("wkcomp_meyers_cay_crps", 1.3156, 0.11),
    ("wkcomp_meyers_cay_tcrps", 1.0453, 0.063),
    ("wkcomp_meyers_cay_lpd", 0.5325, 0.13),
];

fn assert_all(measured: &[Measured]) {
    for m in measured {
        eprintln!(
            "{:34} {:>12.5} target {:>9.4} tolerance {:.4} (used {:.2})",
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

fn meyers(model: MeyersModel, seed: u64) -> Meyers {
    Meyers {
        model,
        seed,
        ..Meyers::default()
    }
}

/// Twin lines (workers' compensation under two names), `Independent`.
fn twins(seed: u64) -> Vec<Measured> {
    let t = Instant::now();
    let tri = clrd(&[("wkcomp", "a"), ("wkcomp", "b")]);
    let fits = meyers(MeyersModel::Crc, seed)
        .fit_segments(&tri, "paid", "premium")
        .unwrap();
    let lines = by_line(&fits.reserves);
    eprintln!("twins in {:.2?}", t.elapsed());
    vec![
        row("twins_spearman", spearman(&lines[0], &lines[1])),
        row("twins_mean_ratio", mean(&lines[0]) / mean(&lines[1]) - 1.0),
    ]
}

/// Commercial auto, private passenger auto and workers' compensation,
/// rank correlated.
fn ranked(seed: u64) -> Vec<Measured> {
    let t = Instant::now();
    let three = clrd(&[
        ("comauto", "comauto"),
        ("ppauto", "ppauto"),
        ("wkcomp", "wkcomp"),
    ]);
    let ranked = Meyers {
        dependence: SegmentDependence::RankCorrelation {
            spearman: TARGET.to_vec(),
        },
        ..meyers(MeyersModel::Csr, seed)
    }
    .fit_segments(&three, "paid", "premium")
    .unwrap();
    let lines = by_line(&ranked.reserves);
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

/// The diagonal backtest, latest three diagonals: the ODP GLM, `BayesOdp`
/// and Meyers' models.
fn backtest(name: &str, seed: u64) -> Vec<Measured> {
    let t = Instant::now();
    let (tri, column) = match name {
        "g353_paid" => (group_353(), "paid"),
        "g353_incurred" => (group_353(), "incurred"),
        line => (clrd(&[(line, line)]), "paid"),
    };
    let premium = premium_of(&tri);
    let cells = TriangleFrame::new(&tri, column, Some(&premium)).unwrap();
    let glm = odp_glm();
    let bayes = BayesOdp::default();
    let (crc, csr, cay) = (
        meyers(MeyersModel::Crc, 0),
        meyers(MeyersModel::Csr, 0),
        meyers(MeyersModel::Cay, 0),
    );
    // The ODP models do not fit incurred increments (the GLM's IRLS does
    // not converge on group 353's), so there Meyers' CRC is the baseline.
    let models: Vec<&dyn TriangleModel> = if column == "incurred" {
        vec![&crc, &csr, &cay]
    } else {
        vec![&glm, &bayes, &crc, &csr, &cay]
    };
    let bt = diagonal_backtest(&cells, &models, 3, 4000, seed, 0.9).unwrap();
    eprintln!("{name} backtest in {:.2?}", t.elapsed());
    let lpd: Vec<Vec<f64>> = bt.log_densities().iter().flatten().cloned().collect();
    assert_eq!(lpd.len(), models.len(), "every model gives log densities");
    let mut out = Vec::new();
    for (m, model) in bt.models().iter().enumerate() {
        let ae = bt.split_scores(m, 2);
        for (s, v) in ae.iter().enumerate() {
            out.push(row(format!("{name}_{model}_ae{s}"), *v));
        }
        out.push(row(
            format!("{name}_{model}_crps"),
            bt.mean(m, 0) / bt.mean(0, 0),
        ));
        out.push(row(
            format!("{name}_{model}_tcrps"),
            bt.mean(m, 3) / bt.mean(0, 3),
        ));
        out.push(row(format!("{name}_{model}_cover"), bt.mean(m, 1)));
        out.push(row(
            format!("{name}_{model}_lpd"),
            mean(&lpd[m]) - mean(&lpd[0]),
        ));
        let worst = lpd[m].iter().copied().fold(f64::INFINITY, f64::min);
        out.push(row(format!("{name}_{model}_lpd_min"), worst));
    }
    out
}

/// The target Spearman matrix of `ranked`.
const TARGET: [f64; 9] = [1.0, 0.6, 0.3, 0.6, 1.0, 0.4, 0.3, 0.4, 1.0];

const BACKTESTS: [&str; 4] = ["g353_paid", "g353_incurred", "comauto", "wkcomp"];

#[test]
fn independent_segments_are_independent() {
    assert_all(&twins(0));
}

#[test]
fn rank_correlation_reaches_the_target() {
    assert_all(&ranked(0));
}

// CI runs two of the four backtests (group 353 paid and wkcomp), each a
// test of its own so they run side by side; the sweep checks all four.
#[test]
fn backtest_on_group_353_paid() {
    assert_all(&backtest("g353_paid", 0));
}

#[test]
fn backtest_on_wkcomp() {
    assert_all(&backtest("wkcomp", 0));
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
    let (start, n) = (env("SWEEP_START", 0), env("SWEEP_SEEDS", 40));
    let runs: Vec<Vec<Measured>> = (start..start + n)
        .map(|seed| {
            let mut all = twins(seed);
            all.extend(ranked(seed));
            for name in BACKTESTS {
                all.extend(backtest(name, seed));
            }
            all
        })
        .collect();
    let mut failures = 0;
    for (q, m0) in runs[0].iter().enumerate() {
        let v: Vec<f64> = runs.iter().map(|r| r[q].value).collect();
        let m = mean(&v);
        let sd = if v.len() > 1 {
            (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
        } else {
            f64::NAN
        };
        let used = runs.iter().map(|r| r[q].used()).fold(0.0, f64::max);
        failures += runs.iter().filter(|r| r[q].used() > 1.0).count();
        println!(
            "{:34} mean {m:>11.5} sd {sd:>9.5} min {:>11.5} max {:>11.5} target {:>8.4} tol {:.4} \
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
