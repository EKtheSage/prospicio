//! Meyers' models over segments and in the diagonal backtest
//! (`Meyers::fit_segments` and `TriangleModel for Meyers`,
//! `docs/design/reserving-bayes.md`, decisions 8 and 11), on the lines of
//! the CAS loss reserve database (`data/clrd_lines.csv`, paid losses and net
//! earned premium summed over companies) and on Meyers' group 353
//! (`data/clrd_comauto_353.csv`, the cells at calendar years up to 1997).
//!
//! Every checked row's target is its mean over seeds 0 to 119 and its
//! tolerance at least 6 times its seed-to-seed sd there (and 1.25 times
//! its largest distance from the mean) ([`TOLERANCES`]); the twins'
//! Spearman row takes `4.5/sqrt(n_sims)`, 4.9 sds. The rows' tails are a
//! little heavier than normal (a mean of log-normals rides on its largest
//! draws): over seeds 0 to 119 and the fresh seeds 200 to 379 the largest
//! distance is 5.3 sds (group 353 paid, CSR's A/E on the earliest split,
//! seed 23), then 5.2 (comauto, CAY's A/E on the middle split, seed 372,
//! every fit converged), where 300 normal seeds of 77 rows would reach
//! about 4.1. Every row passes at each fresh seed. The sweep is the
//! ignored `seed_sweep` test in a release build: `SWEEP_SEEDS=120 cargo
//! test --release -p prospicio-validation --test reserving_meyers_segments
//! -- --ignored seed_sweep --nocapture` (about 40 s a seed; `SWEEP_START`
//! moves the first seed, `SWEEP_BACKTESTS` keeps only the named
//! backtests, `SWEEP_CSV` writes every row at every seed). All at
//! `Meyers`' defaults (4 chains of 1,000 warm-up and 2,000 kept draws,
//! 10,000 simulations; the backtest's 4,000 draws) but one: on group 353
//! the backtest's fits take target acceptance 0.99 ([`backtest_meyers`]).
//!
//! The backtest reports each split's diagnostics (`Backtest::diagnostics`,
//! decision 21 of `reserving-bayes.md`), so the sweep reads the worst R-hat
//! of Meyers' fits from them ([`worst_rhat`]) and counts, without
//! checking, a backtest's rows at a seed where a fit has an R-hat above
//! 1.05. At the default 0.95 that happened at 20 of
//! seeds 0 to 119 and 200 to 299 (R-hat up to 1.18, up to 978
//! divergences), and those fits made the rows' largest outliers (up to 9
//! sds; at seed 204 CRC on incurred, which moved every incurred row while
//! they were ratios to CRC's). At 0.99 it happens at 4 of the 300 seeds
//! (216, 243, 291 and 317; R-hat up to 1.36, with 854 divergences, CSR's
//! total CRPS ratio 5.4 sds out at 317). CLRD's summed lines keep 0.95:
//! their fits reach R-hat 1.052 at most, with no outlying row.
//!
//! * `Independent`, twin lines (CLRD wkcomp under two names, CRC): the rank
//!   correlation of their totals is within `4.5/sqrt(n_sims)` of 0 (sd
//!   0.0092 over seeds, as for independent draws) and their mean totals
//!   agree within 0.47% (sd 0.14%).
//! * `RankCorrelation` on comauto, ppauto and wkcomp (CSR), target Spearman
//!   0.6, 0.3 and 0.4: each pair within 0.0082 (sd at most 0.0029).
//! * The diagonal backtest, latest three diagonals, Meyers' CRC, CSR and
//!   CAY next to the ODP GLM and `BayesOdp` (paid), or next to each other
//!   (group 353 incurred, where the GLM's IRLS does not converge): each
//!   split's A/E, then on paid the mean cell CRPS and total CRPS as ratios
//!   to the GLM's and the mean log density less the GLM's, and on incurred
//!   each model's own mean cell CRPS, total CRPS and mean log density (a
//!   ratio to a sampled model would move with that model's chains). The
//!   ODP rows and each model's 90% coverage and worst cell log density are
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
//! (group 353 paid and wkcomp): about 140 s in a debug build, run side by
//! side (group 353 paid alone about 105 s at target acceptance 0.99).

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
    group_353_until(1997)
}

/// Group 353's cells at calendar years up to `last`, with columns `paid`,
/// `incurred` and `premium`.
fn group_353_until(last: i32) -> Triangle {
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
        if year + (age / 12) as i32 - 1 > last {
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
    /// Whether one of the fits behind the row failed (the sweep sets it).
    failed: bool,
}

impl Measured {
    fn new(name: impl Into<String>, value: f64, target: f64, tolerance: f64) -> Self {
        Self {
            name: name.into(),
            value,
            target,
            tolerance,
            failed: false,
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
/// to 119 and each tolerance at least 6 times its sd there (and 1.25
/// times its largest distance from the mean), from the release-build
/// sweep; checked at seeds 200 to 379.
const TOLERANCES: &[(&str, f64, f64)] = &[
    ("twins_spearman", 0.0, TWINS_SPEARMAN),
    ("twins_mean_ratio", 0.0, 0.0085),
    ("spearman_01", 0.0, 0.014),
    ("spearman_02", 0.0, 0.018),
    ("spearman_12", 0.0, 0.017),
    ("g353_paid_meyers_crc_ae0", 0.3355, 0.086),
    ("g353_paid_meyers_crc_ae1", 0.5508, 0.041),
    ("g353_paid_meyers_crc_ae2", 0.6600, 0.023),
    ("g353_paid_meyers_crc_crps", 2.4386, 0.22),
    ("g353_paid_meyers_crc_tcrps", 4.7946, 0.96),
    ("g353_paid_meyers_crc_lpd", -1.3109, 0.048),
    ("g353_paid_meyers_csr_ae0", 0.3644, 0.12),
    ("g353_paid_meyers_csr_ae1", 0.6599, 0.059),
    ("g353_paid_meyers_csr_ae2", 0.8974, 0.038),
    ("g353_paid_meyers_csr_crps", 2.2795, 0.21),
    ("g353_paid_meyers_csr_tcrps", 3.7218, 0.74),
    ("g353_paid_meyers_csr_lpd", -1.2772, 0.047),
    ("g353_paid_meyers_cay_ae0", 0.3013, 0.068),
    ("g353_paid_meyers_cay_ae1", 0.4980, 0.046),
    ("g353_paid_meyers_cay_ae2", 0.6494, 0.029),
    ("g353_paid_meyers_cay_crps", 2.7692, 0.23),
    ("g353_paid_meyers_cay_tcrps", 5.5180, 1.1),
    ("g353_paid_meyers_cay_lpd", -1.3930, 0.047),
    ("g353_incurred_meyers_crc_ae0", 0.3208, 0.11),
    ("g353_incurred_meyers_crc_ae1", 0.4336, 0.042),
    ("g353_incurred_meyers_crc_ae2", 0.6264, 0.032),
    ("g353_incurred_meyers_crc_crps", 227.9, 21.0),
    ("g353_incurred_meyers_crc_tcrps", 855.1, 150.0),
    ("g353_incurred_meyers_crc_lpd", -7.1920, 0.056),
    ("g353_incurred_meyers_csr_ae0", 0.3377, 0.12),
    ("g353_incurred_meyers_csr_ae1", 0.5020, 0.051),
    ("g353_incurred_meyers_csr_ae2", 0.7880, 0.055),
    ("g353_incurred_meyers_csr_crps", 217.1, 19.0),
    ("g353_incurred_meyers_csr_tcrps", 732.3, 120.0),
    ("g353_incurred_meyers_csr_lpd", -7.1810, 0.049),
    ("g353_incurred_meyers_cay_ae0", 0.2917, 0.085),
    ("g353_incurred_meyers_cay_ae1", 0.3924, 0.037),
    ("g353_incurred_meyers_cay_ae2", 0.6588, 0.038),
    ("g353_incurred_meyers_cay_crps", 251.3, 22.0),
    ("g353_incurred_meyers_cay_tcrps", 935.4, 170.0),
    ("g353_incurred_meyers_cay_lpd", -7.2568, 0.055),
    ("comauto_meyers_crc_ae0", 0.8006, 0.17),
    ("comauto_meyers_crc_ae1", 0.9241, 0.0074),
    ("comauto_meyers_crc_ae2", 0.8347, 0.0053),
    ("comauto_meyers_crc_crps", 1.4608, 0.2),
    ("comauto_meyers_crc_tcrps", 1.2701, 0.16),
    ("comauto_meyers_crc_lpd", -0.3820, 0.11),
    ("comauto_meyers_csr_ae0", 0.7570, 0.18),
    ("comauto_meyers_csr_ae1", 0.9540, 0.011),
    ("comauto_meyers_csr_ae2", 0.8730, 0.0056),
    ("comauto_meyers_csr_crps", 1.5985, 0.28),
    ("comauto_meyers_csr_tcrps", 1.1355, 0.2),
    ("comauto_meyers_csr_lpd", -0.4558, 0.12),
    ("comauto_meyers_cay_ae0", 0.8066, 0.15),
    ("comauto_meyers_cay_ae1", 0.9262, 0.0073),
    ("comauto_meyers_cay_ae2", 0.8494, 0.0049),
    ("comauto_meyers_cay_crps", 1.4018, 0.21),
    ("comauto_meyers_cay_tcrps", 1.1740, 0.16),
    ("comauto_meyers_cay_lpd", -0.3576, 0.12),
    ("wkcomp_meyers_crc_ae0", 0.7563, 0.15),
    ("wkcomp_meyers_crc_ae1", 0.9848, 0.0078),
    ("wkcomp_meyers_crc_ae2", 0.7468, 0.0035),
    ("wkcomp_meyers_crc_crps", 1.3131, 0.16),
    ("wkcomp_meyers_crc_tcrps", 1.0297, 0.093),
    ("wkcomp_meyers_crc_lpd", 0.5421, 0.19),
    ("wkcomp_meyers_csr_ae0", 0.7323, 0.17),
    ("wkcomp_meyers_csr_ae1", 1.0881, 0.012),
    ("wkcomp_meyers_csr_ae2", 0.8110, 0.0051),
    ("wkcomp_meyers_csr_crps", 1.3032, 0.22),
    ("wkcomp_meyers_csr_tcrps", 0.8722, 0.12),
    ("wkcomp_meyers_csr_lpd", 0.6025, 0.19),
    ("wkcomp_meyers_cay_ae0", 0.7696, 0.16),
    ("wkcomp_meyers_cay_ae1", 1.0116, 0.0094),
    ("wkcomp_meyers_cay_ae2", 0.7362, 0.0033),
    ("wkcomp_meyers_cay_crps", 1.3155, 0.18),
    ("wkcomp_meyers_cay_tcrps", 1.0449, 0.1),
    ("wkcomp_meyers_cay_lpd", 0.5350, 0.19),
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

/// Meyers' `model` as backtest `name` takes it, with `seed` (which the
/// backtest replaces by its own).
///
/// On group 353's training triangles of 28 and 36 cells the default
/// target acceptance of 0.95 is not enough: over seeds 0 to 119 and 200 to
/// 299 a fit had an R-hat above 1.05 at 20 seeds (up to 1.18, with 978
/// divergences), and those fits gave the rows' largest outliers (up to 9
/// sds). At 0.99 such a fit is rarer (4 seeds in 300, see
/// [`worst_rhat`]). CLRD's fits keep the default (R-hat at most
/// 1.052, no outlying row).
fn backtest_meyers(name: &str, model: MeyersModel, seed: u64) -> Meyers {
    let mut m = meyers(model, seed);
    if name.starts_with("g353") {
        m.sampler.target_accept = 0.99;
    }
    m
}

/// R-hat above which the sweep treats a fit as failed.
const FAILED_RHAT: f64 = 1.05;

/// The largest R-hat of Meyers' fits (the last three models) in a
/// backtest, from the diagnostics it reports for each split.
fn worst_rhat(bt: &prospicio_reserving::Backtest) -> f64 {
    let first = bt.models().len() - 3;
    (first..bt.models().len())
        .flat_map(|m| bt.diagnostics(m).iter().flatten())
        .fold(0.0, |worst, d| f64::max(worst, d.max_rhat))
}

/// The diagonal backtest, latest three diagonals: the ODP GLM, `BayesOdp`
/// and Meyers' models.
fn backtest(name: &str, seed: u64) -> (Vec<Measured>, f64) {
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
        backtest_meyers(name, MeyersModel::Crc, 0),
        backtest_meyers(name, MeyersModel::Csr, 0),
        backtest_meyers(name, MeyersModel::Cay, 0),
    );
    // The ODP models do not fit incurred increments (the GLM's IRLS does
    // not converge on group 353's).
    let models: Vec<&dyn TriangleModel> = if column == "incurred" {
        vec![&crc, &csr, &cay]
    } else {
        vec![&glm, &bayes, &crc, &csr, &cay]
    };
    let bt = diagonal_backtest(&cells, &models, 3, 4000, seed, 0.9).unwrap();
    eprintln!("{name} backtest in {:.2?}", t.elapsed());
    let lpd: Vec<Vec<f64>> = bt.log_densities().iter().flatten().cloned().collect();
    assert_eq!(lpd.len(), models.len(), "every model gives log densities");
    // Scores are relative to the GLM's on paid. On incurred there is no
    // GLM, and a ratio to another sampled model would move every row with
    // that model's chains, so there each model's own scores are reported.
    let (crps0, tcrps0, lpd0) = if column == "incurred" {
        (1.0, 1.0, 0.0)
    } else {
        (bt.mean(0, 0), bt.mean(0, 3), mean(&lpd[0]))
    };
    let mut out = Vec::new();
    for (m, model) in bt.models().iter().enumerate() {
        let ae = bt.split_scores(m, 2);
        for (s, v) in ae.iter().enumerate() {
            out.push(row(format!("{name}_{model}_ae{s}"), *v));
        }
        out.push(row(format!("{name}_{model}_crps"), bt.mean(m, 0) / crps0));
        out.push(row(format!("{name}_{model}_tcrps"), bt.mean(m, 3) / tcrps0));
        out.push(row(format!("{name}_{model}_cover"), bt.mean(m, 1)));
        out.push(row(format!("{name}_{model}_lpd"), mean(&lpd[m]) - lpd0));
        let worst = lpd[m].iter().copied().fold(f64::INFINITY, f64::min);
        out.push(row(format!("{name}_{model}_lpd_min"), worst));
    }
    (out, worst_rhat(&bt))
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
    assert_all(&backtest("g353_paid", 0).0);
}

#[test]
fn backtest_on_wkcomp() {
    assert_all(&backtest("wkcomp", 0).0);
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
    // `SWEEP_BACKTESTS` (comma separated) limits the sweep to those backtests.
    let only = std::env::var("SWEEP_BACKTESTS").ok();
    // `SWEEP_CSV` names a file for every row at every seed
    // (`seed,row,value,failed`).
    let mut csv = std::env::var("SWEEP_CSV")
        .ok()
        .map(|p| std::fs::File::create(p).expect("SWEEP_CSV"));
    let runs: Vec<Vec<Measured>> = (start..start + n)
        .map(|seed| {
            let mut all = Vec::new();
            if only.is_none() {
                all.extend(twins(seed));
                all.extend(ranked(seed));
            }
            for name in BACKTESTS {
                if only
                    .as_ref()
                    .is_none_or(|o| o.split(',').any(|b| b == name))
                {
                    let (mut rows, rhat) = backtest(name, seed);
                    if name.starts_with("g353") && rhat > FAILED_RHAT {
                        println!("seed {seed}: a fit of {name} failed (R-hat {rhat:.4})");
                        rows.iter_mut().for_each(|m| m.failed = true);
                    }
                    all.extend(rows);
                }
            }
            if let Some(f) = csv.as_mut() {
                use std::io::Write;
                for m in &all {
                    writeln!(f, "{seed},{},{},{}", m.name, m.value, m.failed).unwrap();
                }
            }
            all
        })
        .collect();
    let mut failures = 0;
    // Rows behind a failed fit are counted, not checked.
    for (q, m0) in runs[0].iter().enumerate() {
        let ok: Vec<&Measured> = runs.iter().map(|r| &r[q]).filter(|m| !m.failed).collect();
        let n_failed = runs.len() - ok.len();
        let v: Vec<f64> = ok.iter().map(|r| r.value).collect();
        let m = mean(&v);
        let sd = if v.len() > 1 {
            (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
        } else {
            f64::NAN
        };
        let used = ok.iter().map(|r| r.used()).fold(0.0, f64::max);
        failures += ok.iter().filter(|r| r.used() > 1.0).count();
        println!(
            "{:34} mean {m:>11.5} sd {sd:>9.5} min {:>11.5} max {:>11.5} target {:>8.4} tol {:.4} \
             tol/sd {:>5.1} max used {used:.2} seeds failed {n_failed}",
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
