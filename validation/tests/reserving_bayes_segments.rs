//! The Bayesian ODP over segments and in the diagonal backtest
//! (`BayesOdp::fit_segments` and `TriangleModel for BayesOdp`,
//! `docs/design/reserving-bayes.md`, decisions 8 and 11), on the lines of
//! the CAS loss reserve database (`data/clrd_lines.csv`, paid losses summed
//! over companies) and GenIns.
//!
//! Every tolerance is at least 4 times the quantity's seed-to-seed sd over
//! seeds 0 to 69, and every row passes at each of those seeds, using at
//! most 0.65 of its tolerance. The sweep is the ignored `seed_sweep` test
//! in a release build: `SWEEP_SEEDS=70 cargo test --release -p
//! prospicio-validation --test reserving_bayes_segments -- --ignored
//! seed_sweep --nocapture` (it defaults to 40 seeds from 0;
//! `SWEEP_START` moves the first seed, for a check on fresh seeds). All at
//! the defaults (4 chains of 1,000 draws, 10,000 simulations; the
//! backtest's 4,000 draws).
//!
//! * `Independent`, twin lines (workers' compensation under two names):
//!   the rank correlation of their totals is within `4.5/sqrt(n_sims)`
//!   of 0 (sd 0.010, about `1/sqrt(n_sims)`, as for independent draws;
//!   shared streams would give nearly 1), and their mean totals agree
//!   within 0.5% (sd 0.11%).
//! * `RankCorrelation` on comauto, ppauto and wkcomp, target Spearman
//!   0.6, 0.3 and 0.4: each pair within 0.015 (sd at most 0.0029).
//! * The diagonal backtest, latest three diagonals, `BayesOdp` next to
//!   the ODP GLM. On the CLRD lines the two agree: actual over expected
//!   within 0.002 per split (sd at most 0.00034), mean cell CRPS within
//!   3% and total CRPS within 5%, coverage within 0.15 (one or two cells),
//!   mean log density within 0.2 (sd 0.043 on wkcomp). On GenIns the
//!   Bayesian means are about 1% higher (A/E lower by 0.007, 0.013 and
//!   0.010 per split on average, and by 0.004 to 0.019 across seeds and
//!   splits: the posterior mean of `exp(eta)` over the young origins' wide
//!   posteriors, where the GLM gives the MLE), and its mean log density
//!   lower by 0.013 on average (sd 0.0044 over seeds 0 to 69, 0.0051 over
//!   40 to 69 alone; the tolerance is 0.025, 5 such sds); the CRPS agree
//!   within 3% and 10%. Per-cell log densities differ by up to 2 on single
//!   cells at some seeds (heavy-tailed), so only their mean is checked.
//!   Each model's coverage is reported, not checked: the 90% intervals
//!   hold 0.69 to 0.73 of comauto's held-out cells and 0.41 to 0.46 of
//!   wkcomp's across seeds.

use std::time::Instant;

use prospicio_glm::Glm;
use prospicio_models::Terms;
use prospicio_prob::PredictiveDistribution;
use prospicio_reserving::bayes::BayesOdp;
use prospicio_reserving::{
    DevelopmentColumn, GlmCandidate, Grain, Long, Month, SegmentDependence, Triangle,
    TriangleFrame, diagonal_backtest,
};
use prospicio_validation::triangle;

/// The lines of `data/clrd_lines.csv` named in `lines`, as segments of a
/// `lob` key, renamed by `rename` (to repeat a line under two names).
fn clrd(lines: &[(&str, &str)]) -> Triangle {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/clrd_lines.csv");
    let text = std::fs::read_to_string(path).expect("data/clrd_lines.csv");
    let rows: Vec<Vec<&str>> = text
        .lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
        .map(|l| l.split(',').collect())
        .collect();
    let (mut keys, mut origin, mut ages, mut paid) = (vec![], vec![], vec![], vec![]);
    for &(line, name) in lines {
        for f in rows.iter().filter(|f| f[0] == line) {
            keys.push(name);
            origin.push(Month::january(f[1].parse().unwrap()));
            ages.push(f[2].parse().unwrap());
            paid.push(f[3].parse::<f64>().unwrap());
        }
    }
    Triangle::from_long(&Long {
        keys: &[("lob", &keys)],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("paid", &paid)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
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

    /// A quantity reported by the sweep but not checked (an infinite
    /// tolerance).
    fn info(name: impl Into<String>, value: f64) -> Self {
        Self::new(name, value, 0.0, f64::INFINITY)
    }

    /// The share of the tolerance used.
    fn used(&self) -> f64 {
        (self.value - self.target).abs() / self.tolerance
    }
}

fn assert_all(measured: &[Measured]) {
    for m in measured {
        eprintln!(
            "{:28} {:>12.5} target {:>9.4} tolerance {:.4} (used {:.2})",
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

fn model(seed: u64) -> BayesOdp {
    BayesOdp {
        seed,
        ..BayesOdp::default()
    }
}

/// Twin lines (workers' compensation under two names), `Independent`: the
/// rank correlation of their totals and the ratio of their means.
fn twins(seed: u64) -> Vec<Measured> {
    let t = Instant::now();
    let tri = clrd(&[("wkcomp", "a"), ("wkcomp", "b")]);
    let fits = model(seed).fit_segments(&tri, "paid").unwrap();
    let lines = by_line(&fits.reserves);
    let n = lines[0].len() as f64;
    eprintln!("twins in {:.2?}", t.elapsed());
    vec![
        Measured::new(
            "twins_spearman",
            spearman(&lines[0], &lines[1]),
            0.0,
            TWINS_SPEARMAN_SDS / n.sqrt(),
        ),
        Measured::new(
            "twins_mean_ratio",
            mean(&lines[0]) / mean(&lines[1]) - 1.0,
            0.0,
            TWINS_MEAN_RATIO,
        ),
    ]
}

/// Commercial auto, private passenger auto and workers' compensation,
/// rank correlated: each pair's Spearman against the target.
fn ranked(seed: u64) -> Vec<Measured> {
    let t = Instant::now();
    let three = clrd(&[
        ("comauto", "comauto"),
        ("ppauto", "ppauto"),
        ("wkcomp", "wkcomp"),
    ]);
    let ranked = BayesOdp {
        dependence: SegmentDependence::RankCorrelation {
            spearman: TARGET.to_vec(),
        },
        ..model(seed)
    }
    .fit_segments(&three, "paid")
    .unwrap();
    let lines = by_line(&ranked.reserves);
    eprintln!("rank correlation in {:.2?}", t.elapsed());
    [(0, 1), (0, 2), (1, 2)]
        .iter()
        .map(|&(i, j)| {
            Measured::new(
                format!("spearman_{i}{j}"),
                spearman(&lines[i], &lines[j]),
                TARGET[i * 3 + j],
                RANK,
            )
        })
        .collect()
}

/// The diagonal backtest of `BayesOdp` next to the ODP GLM on the latest
/// three diagonals of `name`: each split's actual over expected, and the
/// means over the splits of the cell CRPS, the total CRPS, the coverage of
/// the central 90% and the held-out log density, Bayes against the GLM.
fn backtest(name: &str, seed: u64) -> Vec<Measured> {
    let t = Instant::now();
    let (tri, column, tol) = match name {
        "genins" => (triangle("genins"), "values", &GENINS),
        line => (clrd(&[(line, line)]), "paid", &CLRD),
    };
    let cells = TriangleFrame::new(&tri, column, None).unwrap();
    let (glm, bayes) = (odp_glm(), BayesOdp::default());
    let bt = diagonal_backtest(&cells, &[&glm, &bayes], 3, 4000, seed, 0.9).unwrap();
    assert_eq!(bt.models(), ["odp", "bayes_odp"]);
    eprintln!("{name} backtest in {:.2?}", t.elapsed());
    let mut out = Vec::new();
    let (g, b) = (bt.split_scores(0, 2), bt.split_scores(1, 2));
    for s in 0..bt.n_splits() {
        out.push(Measured::new(
            format!("{name}_ae_diff_split{s}"),
            b[s] - g[s],
            tol.ae_diff.0,
            tol.ae_diff.1,
        ));
    }
    out.push(Measured::new(
        format!("{name}_crps_ratio"),
        bt.mean(1, 0) / bt.mean(0, 0),
        1.0,
        tol.crps_ratio,
    ));
    out.push(Measured::new(
        format!("{name}_total_crps_ratio"),
        bt.mean(1, 3) / bt.mean(0, 3),
        1.0,
        tol.total_crps_ratio,
    ));
    out.push(Measured::new(
        format!("{name}_coverage_diff"),
        bt.mean(1, 1) - bt.mean(0, 1),
        0.0,
        tol.coverage_diff,
    ));
    // Each model's own coverage, for the record (not checked).
    out.push(Measured::info(
        format!("{name}_coverage_odp"),
        bt.mean(0, 1),
    ));
    out.push(Measured::info(
        format!("{name}_coverage_bayes"),
        bt.mean(1, 1),
    ));
    let lpd: Vec<Vec<f64>> = bt.log_densities().iter().flatten().cloned().collect();
    assert_eq!(lpd.len(), 2, "both models give log densities");
    out.push(Measured::new(
        format!("{name}_lpd_diff"),
        mean(&lpd[1]) - mean(&lpd[0]),
        tol.lpd_diff.0,
        tol.lpd_diff.1,
    ));
    out
}

/// The target Spearman matrix of `ranked`.
const TARGET: [f64; 9] = [1.0, 0.6, 0.3, 0.6, 1.0, 0.4, 0.3, 0.4, 1.0];

/// Tolerances of one triangle's backtest: (target, tolerance) where the
/// Bayesian model differs from the GLM systematically.
struct BacktestTolerance {
    ae_diff: (f64, f64),
    crps_ratio: f64,
    total_crps_ratio: f64,
    coverage_diff: f64,
    lpd_diff: (f64, f64),
}

const TWINS_SPEARMAN_SDS: f64 = 4.5;
const TWINS_MEAN_RATIO: f64 = 0.005;
const RANK: f64 = 0.015;
const GENINS: BacktestTolerance = BacktestTolerance {
    ae_diff: (-0.01, 0.015),
    crps_ratio: 0.03,
    total_crps_ratio: 0.10,
    coverage_diff: 0.1,
    lpd_diff: (-0.013, 0.025),
};
const CLRD: BacktestTolerance = BacktestTolerance {
    ae_diff: (0.0, 0.002),
    crps_ratio: 0.03,
    total_crps_ratio: 0.05,
    coverage_diff: 0.15,
    lpd_diff: (0.0, 0.2),
};

#[test]
fn independent_segments_are_independent() {
    assert_all(&twins(0));
}

#[test]
fn rank_correlation_reaches_the_target() {
    assert_all(&ranked(0));
}

#[test]
fn backtest_next_to_the_odp_glm() {
    for name in ["genins", "comauto", "wkcomp"] {
        assert_all(&backtest(name, 0));
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
    let (start, n) = (env("SWEEP_START", 0), env("SWEEP_SEEDS", 40));
    let runs: Vec<Vec<Measured>> = (start..start + n)
        .map(|seed| {
            let mut all = twins(seed);
            all.extend(ranked(seed));
            for name in ["genins", "comauto", "wkcomp"] {
                all.extend(backtest(name, seed));
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
            "{:28} mean {m:>11.5} sd {sd:>9.5} min {:>11.5} max {:>11.5} target {:>8.4} tol {:.4} \
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
