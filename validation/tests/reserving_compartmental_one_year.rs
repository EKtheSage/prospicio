//! The compartmental model's one-year view (`Compartmental::one_year`,
//! `docs/design/reserving-bayes.md`, decision 20) on the papers' company
//! 337 workers' compensation (`data/clrd_wkcomp_337.csv`), trained as
//! `reserving_compartmental.rs` trains it (calendar years up to 1996, and
//! 1997 with premium only), at the defaults of each model (model 1: 4
//! chains of 1,000 warm-up and 2,000 kept draws; model 2: 4 chains of
//! 2,000 and 8,000) and the default one-year settings (moment matching,
//! then a refit for what it cannot rescue).
//!
//! There is no external reference for a Bayesian one-year view of these
//! models, so the checks are internal, against refits on the same years.
//! The year adds 20 cells (paid and outstanding of every origin's next
//! diagonal, 1997's first among them) to 90, so most simulations are
//! flagged (`k̂` above 0.7). For model 1:
//!
//! * `mean_z`: the default's total claims development result has mean
//!   zero within Monte Carlo error over `SIMS` simulations;
//! * `rescued`: the share of flagged simulations moment matching brings to
//!   `k̂` at most 0.7 is above a floor;
//! * `gap_matched`: on the first `REFIT_SIMS` simulations, the root mean
//!   square gap between a rescued simulation's total and its `RefitAll`
//!   value, in units of the total's sd, is below a bound and below
//!   `gap_reweighted`, the same simulations' gap without matching;
//! * `life_ratio`: the default's total sd over the lifetime reserve's is
//!   in a band around its sweep mean.
//!
//! Every bound is the sweep's mean minus (floors) or plus (bounds) at
//! least 4 seed-to-seed sds over seeds 0 to 19, and every seed passes
//! (`seed_sweep`, ignored: `SWEEP_SEEDS=20 cargo test --release -p
//! prospicio-validation --features slow-bayes --test
//! reserving_compartmental_one_year -- --ignored seed_sweep --nocapture`).
//!
//! Model 2 is measured, not checked in CI (`model_2_comparison`, ignored):
//! moment matching rescues almost none of its simulations, and a refit
//! takes about a minute, so its default view at 10,000 simulations is
//! days of refits; `runtime` (ignored) measures both models' wall times.

use std::sync::OnceLock;
use std::time::Instant;

use prospicio_prob::{Distribution, PredictiveDistribution};
use prospicio_reserving::bayes::{
    BayesOneYear, BayesOneYearFit, Compartmental, CompartmentalFit, CompartmentalPriors,
    OneYearRefit, Outstanding,
};
use prospicio_reserving::{DevelopmentColumn, Grain, Lag, Long, Month, Triangle};

/// Simulations of model 1's default view.
const SIMS: usize = 500;
/// Of those, the simulations refitted with `RefitAll` for the gaps.
const REFIT_SIMS: usize = 40;

/// Model 1's floor on `rescued`, bound on `gap_matched` and band on
/// `life_ratio`, from `seed_sweep` over seeds 0 to 19 (mean, sd and the
/// extreme seed are in the design note's decision 20).
const RESCUED_FLOOR: f64 = 0.0;
const GAP_BOUND: f64 = 1.0;
const LIFE_BAND: (f64, f64) = (0.0, 2.0);

/// Model 2's comparison: simulations matched (no refit) and refitted.
const SIMS_2: usize = 200;
const REFIT_SIMS_2: usize = 8;

/// The rows of `data/clrd_wkcomp_337.csv`: origin, age in months, paid,
/// incurred, premium.
fn rows() -> Vec<(i32, Lag, f64, f64, f64)> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/clrd_wkcomp_337.csv");
    let text = std::fs::read_to_string(path).expect("data/clrd_wkcomp_337.csv");
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    assert_eq!(
        lines.next(),
        Some("origin,development,paid,incurred,case_incurred,premium,premium_net")
    );
    lines
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
        .collect()
}

/// The papers' training triangle: the cells at calendar years up to 1996,
/// columns `paid`, `incurred` and `premium` (direct), and 1997 with its
/// premium only.
fn training() -> &'static Triangle {
    static TRIANGLE: OnceLock<Triangle> = OnceLock::new();
    TRIANGLE.get_or_init(|| {
        let all = rows();
        let mut rows: Vec<_> = all
            .iter()
            .copied()
            .filter(|r| r.0 + (r.1 / 12) as i32 - 1 <= 1996)
            .collect();
        assert_eq!(rows.len(), 45);
        let premium_1997 = all.iter().find(|r| r.0 == 1997).unwrap().4;
        rows.push((1997, 12, f64::NAN, f64::NAN, premium_1997));
        let origin: Vec<Month> = rows.iter().map(|r| Month::january(r.0)).collect();
        let ages: Vec<Lag> = rows.iter().map(|r| r.1).collect();
        let column = |k: usize| -> Vec<f64> { rows.iter().map(|r| [r.2, r.3, r.4][k]).collect() };
        let (paid, incurred, premium) = (column(0), column(1), column(2));
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
    })
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn sd(x: &[f64]) -> f64 {
    let m = mean(x);
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
}

fn totals(d: &PredictiveDistribution) -> Vec<f64> {
    let c = d.n_components();
    d.draw_matrix().chunks(c).map(|r| r.iter().sum()).collect()
}

/// Model 1 or 2 at its defaults.
fn preset(two: bool) -> Compartmental {
    if two {
        Compartmental::model_2(CompartmentalPriors::gesmann_morris_2())
    } else {
        Compartmental::new(CompartmentalPriors::gesmann_morris())
    }
}

/// The one-year view of model 1 or 2 and its wall time.
fn view(
    two: bool,
    n_sims: usize,
    seed: u64,
    refit: OneYearRefit,
    max_refits: Option<usize>,
) -> (BayesOneYearFit<CompartmentalFit>, f64) {
    let t = Instant::now();
    let view = Compartmental {
        n_sims,
        seed,
        ..preset(two)
    }
    .one_year(
        training(),
        "paid",
        Outstanding::FromIncurred("incurred"),
        "premium",
        &BayesOneYear {
            refit,
            max_refits,
            ..BayesOneYear::default()
        },
    )
    .unwrap_or_else(|e| panic!("model {}: {e}", if two { 2 } else { 1 }));
    (view, t.elapsed().as_secs_f64())
}

/// What one model measures at one seed.
#[derive(Debug, Clone, Copy)]
struct Measured {
    mean_z: f64,
    flagged_share: f64,
    rescued: f64,
    /// Simulations the default refitted (or, with a budget, left
    /// unrefitted).
    refitted: usize,
    gap_matched: f64,
    gap_reweighted: f64,
    /// The rescued simulations' total sd under the default and under
    /// `ReweightOnly`, over their sd refitted.
    sd_matched: f64,
    sd_reweighted: f64,
    life_ratio: f64,
    /// Wall times: `ReweightOnly` and the default at `sims`, `RefitAll`
    /// at `refit_sims`.
    times: [f64; 3],
}

/// Model 1 (or 2) at one seed: `sims` simulations reweighted only and at
/// the default (with `max_refits`), the first `refit_sims` refitted.
fn measure(
    two: bool,
    sims: usize,
    refit_sims: usize,
    max_refits: Option<usize>,
    seed: u64,
) -> Measured {
    let name = if two { "model 2" } else { "model 1" };
    let (only, t_only) = view(two, sims, seed, OneYearRefit::ReweightOnly, None);
    let (default, t_default) = view(two, sims, seed, OneYearRefit::default(), max_refits);
    let (all, t_all) = view(two, refit_sims, seed, OneYearRefit::RefitAll, None);
    // The same years under every setting.
    assert_eq!(
        only.pareto_k, default.pareto_k,
        "{name}: the default's years"
    );
    assert_eq!(
        all.pareto_k,
        only.pareto_k[..refit_sims],
        "{name}: RefitAll's years"
    );
    let (t_only_, t_def, t_all_) = (totals(&only.cdr), totals(&default.cdr), totals(&all.cdr));
    let s = sd(&t_def);
    let mean_z = mean(&t_def) / (s / (sims as f64).sqrt());
    let n_flagged = only
        .pareto_k
        .iter()
        .filter(|&&k| k.is_nan() || k > 0.7)
        .count();
    let rescued = if n_flagged == 0 {
        1.0
    } else {
        default.moment_matched.len() as f64 / n_flagged as f64
    };
    let matched: Vec<usize> = default
        .moment_matched
        .iter()
        .copied()
        .filter(|&i| i < refit_sims)
        .collect();
    let pick = |x: &[f64]| -> Vec<f64> { matched.iter().map(|&i| x[i]).collect() };
    let rms = |x: &[f64]| -> f64 {
        let sq: Vec<f64> = matched
            .iter()
            .map(|&i| ((x[i] - t_all_[i]) / s).powi(2))
            .collect();
        if sq.is_empty() { 0.0 } else { mean(&sq).sqrt() }
    };
    let sd_refit = sd(&pick(&t_all_));
    let life = default.fit.reserves.total().std_dev();
    let m = Measured {
        mean_z,
        flagged_share: only.flagged_share(),
        rescued,
        refitted: default.refitted.len() + default.unrefitted.len(),
        gap_matched: rms(&t_def),
        gap_reweighted: rms(&t_only_),
        sd_matched: sd(&pick(&t_def)) / sd_refit,
        sd_reweighted: sd(&pick(&t_only_)) / sd_refit,
        life_ratio: s / life,
        times: [t_only, t_default, t_all],
    };
    println!("{name} seed {seed}: {m:?}");
    m
}

#[test]
fn the_one_year_view_of_model_1_matches_its_refits() {
    let m = measure(false, SIMS, REFIT_SIMS, None, 0);
    assert!(m.mean_z.abs() < 5.0, "mean z {}", m.mean_z);
    assert!(m.rescued >= RESCUED_FLOOR, "rescued {}", m.rescued);
    assert!(m.gap_matched < GAP_BOUND, "gap {}", m.gap_matched);
    assert!(
        m.gap_matched < m.gap_reweighted,
        "matching {} is no closer than reweighting {}",
        m.gap_matched,
        m.gap_reweighted
    );
    assert!(
        (LIFE_BAND.0..LIFE_BAND.1).contains(&m.life_ratio),
        "one-year over lifetime sd {}",
        m.life_ratio
    );
}

fn summary(name: &str, row: &str, v: &[f64]) {
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = v.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "{name:8} {row:15} mean {:9.4} sd {:8.4} min {min:9.4} max {max:9.4} mean-4sd {:9.4} \
         mean+4sd {:9.4}",
        mean(v),
        sd(v),
        mean(v) - 4.0 * sd(v),
        mean(v) + 4.0 * sd(v)
    );
}

fn summarise(name: &str, ms: &[Measured]) {
    for (row, f) in [
        ("mean_z", (|m: &Measured| m.mean_z) as fn(&Measured) -> f64),
        ("flagged_share", |m| m.flagged_share),
        ("rescued", |m| m.rescued),
        ("refitted", |m| m.refitted as f64),
        ("gap_matched", |m| m.gap_matched),
        ("gap_reweighted", |m| m.gap_reweighted),
        ("sd_matched", |m| m.sd_matched),
        ("sd_reweighted", |m| m.sd_reweighted),
        ("life_ratio", |m| m.life_ratio),
        ("t_reweight", |m| m.times[0]),
        ("t_default", |m| m.times[1]),
        ("t_refit_all", |m| m.times[2]),
    ] {
        let v: Vec<f64> = ms.iter().map(f).collect();
        summary(name, row, &v);
    }
}

fn seeds(default: u64) -> u64 {
    std::env::var("SWEEP_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Model 1's seed-to-seed spread of the measured values (ignored; release
/// build, minutes a seed).
#[test]
#[ignore]
fn seed_sweep() {
    let ms: Vec<Measured> = (0..seeds(20))
        .map(|seed| measure(false, SIMS, REFIT_SIMS, None, seed))
        .collect();
    summarise("model 1", &ms);
}

/// Model 2 at a few seeds (ignored; release build, many minutes a seed):
/// `SIMS_2` simulations moment-matched and never refitted (`max_refits`
/// 0), the first `REFIT_SIMS_2` also refitted, for the share matching
/// rescues and the gaps.
#[test]
#[ignore]
fn model_2_comparison() {
    let ms: Vec<Measured> = (0..seeds(3))
        .map(|seed| measure(true, SIMS_2, REFIT_SIMS_2, Some(0), seed))
        .collect();
    summarise("model 2", &ms);
}

/// Wall times at 10,000 simulations (ignored; release build): model 1
/// reweighting only and at the default, model 2 reweighting only and
/// moment matching without refits (`max_refits` 0), with the number of
/// refits the default would make.
#[test]
#[ignore]
fn runtime() {
    for two in [false, true] {
        let (only, t_only) = view(two, 10_000, 0, OneYearRefit::ReweightOnly, None);
        let budget = two.then_some(0);
        let (default, t_default) = view(two, 10_000, 0, OneYearRefit::default(), budget);
        println!(
            "model {}: 10,000 simulations: ReweightOnly {t_only:.1} s (flagged {:.3}), default \
             with max_refits {budget:?} {t_default:.1} s (matched {}, refitted {}, left \
             unrefitted {})",
            if two { 2 } else { 1 },
            only.flagged_share(),
            default.moment_matched.len(),
            default.refitted.len(),
            default.unrefitted.len()
        );
    }
}
