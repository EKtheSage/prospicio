//! Meyers' one-year view (`Meyers::one_year`, `docs/design/reserving-bayes.md`,
//! decision 19) on his illustrative insurer, group 353 Commercial Auto
//! (`data/clrd_comauto_353.csv`, the cells at calendar years up to 1997):
//! CRC and CSR on paid, CAY on incurred, as Meyers fits them, at the
//! defaults (4 chains of 1,000 warm-up and 2,000 kept draws, target
//! acceptance 0.95) and the default one-year settings (moment matching,
//! then a refit for what it cannot rescue).
//!
//! There is no external reference for a Bayesian one-year view of these
//! models, so the checks are internal, against refits on the same years.
//! Nearly every simulation is flagged (`k̂` above 0.7): the year's cells
//! have the smallest `sigma_d` of their origins, so each moves its
//! origin's posterior far. For each case:
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
//!   in a band around its sweep mean (0.77 to 0.82: about four fifths of
//!   the lifetime uncertainty emerges in the first year);
//! * against the ODP bootstrap's one-year view of the same paid triangle
//!   (`OdpBootstrap::one_year`, chain ladder), CRC's `life_ratio` over the
//!   bootstrap's own is in a wide band. The models differ (log-normal
//!   cumulative values with a premium-based level against over-dispersed
//!   Poisson increments with free origin levels), so their sds differ;
//!   only the share of the lifetime uncertainty that emerges in the first
//!   year is expected to be of the same order.
//!
//! A pull request runs CRC alone (the smoke run, with the same sizes and
//! bounds); every case runs when `RISK_RS_FULL_BAYES` is set, as in CI's
//! weekly and manual runs (`prospicio_validation::full_bayes`, the user's
//! decision of 2026-10-16). The view's `n_sims` is set explicitly here: by
//! default `Meyers::one_year` runs at most 2,000 simulations.
//!
//! Every bound is the sweep's mean minus (floors) or plus (bounds) at
//! least 4 seed-to-seed sds over seeds 0 to 19, and every seed passes
//! (`seed_sweep`, ignored: `SWEEP_SEEDS=20 cargo test --release -p
//! prospicio-validation --features slow-bayes --test
//! reserving_meyers_one_year -- --ignored seed_sweep --nocapture`); the
//! sweep also reports the wall times.

use std::sync::OnceLock;
use std::time::Instant;

use prospicio_prob::{Distribution, PredictiveDistribution};
use prospicio_reserving::bayes::{
    BayesOneYear, BayesOneYearFit, Meyers, MeyersFit, MeyersModel, OneYearRefit,
};
use prospicio_reserving::{
    ChainLadder, DevelopmentColumn, Grain, Lag, Long, Month, OdpBootstrap, OneYearMethod, Triangle,
};
use prospicio_validation::full_bayes;

/// Simulations of the default view.
const SIMS: usize = 500;
/// Of those, the simulations refitted with `RefitAll` for the gaps.
const REFIT_SIMS: usize = 60;

/// The cases: name, model and column. The first, CRC, is the smoke run.
const CASES: &[(&str, MeyersModel, &str)] = &[
    ("CRC paid", MeyersModel::Crc, "paid"),
    ("CSR paid", MeyersModel::Csr, "paid"),
    ("CAY incurred", MeyersModel::Cay, "incurred"),
];

/// The cases the strict test runs: every case in the full run, CRC alone
/// otherwise.
fn strict_cases() -> &'static [(&'static str, MeyersModel, &'static str)] {
    if full_bayes() { CASES } else { &CASES[..1] }
}

/// Per case: the floor on `rescued`, the bound on `gap_matched` and the
/// band on `life_ratio`. Over seeds 0 to 19 (mean, sd, and the extreme
/// seed): `rescued` 0.904 (0.034, min 0.795), 0.900 (0.039, min 0.803)
/// and 0.833 (0.052, min 0.734); `gap_matched` 0.080 (0.029, max 0.158),
/// 0.075 (0.019, max 0.137) and 0.084 (0.027, max 0.160), against
/// `gap_reweighted` 0.21, 0.19 and 0.21 (at every seed above
/// `gap_matched`, the largest ratio 0.88); `life_ratio` 0.769 (0.081),
/// 0.798 (0.071) and 0.825 (0.083).
const BOUNDS: &[(&str, f64, f64, (f64, f64))] = &[
    ("CRC paid", 0.75, 0.20, (0.44, 1.10)),
    ("CSR paid", 0.74, 0.16, (0.51, 1.09)),
    ("CAY incurred", 0.62, 0.20, (0.49, 1.16)),
];

/// The band on CRC's `life_ratio` over the ODP bootstrap's: 0.957 (sd
/// 0.097, from 0.635 to 1.141) over seeds 0 to 19; the bootstrap's own is
/// 0.803 (sd 0.008).
const ODP_BAND: (f64, f64) = (0.56, 1.35);

/// The rows of `data/clrd_comauto_353.csv`: origin, age in months, paid,
/// incurred, premium.
fn rows() -> Vec<(i32, Lag, f64, f64, f64)> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/clrd_comauto_353.csv");
    let text = std::fs::read_to_string(path).expect("data/clrd_comauto_353.csv");
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    assert_eq!(
        lines.next(),
        Some("origin,development,paid,incurred,premium")
    );
    lines
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (
                f[0].parse().unwrap(),
                f[1].parse().unwrap(),
                f[2].parse().unwrap(),
                f[3].parse().unwrap(),
                f[4].parse().unwrap(),
            )
        })
        .collect()
}

/// The training triangle: the cells at calendar years up to 1997, columns
/// `paid`, `incurred` and `premium`.
fn training() -> &'static Triangle {
    static TRIANGLE: OnceLock<Triangle> = OnceLock::new();
    TRIANGLE.get_or_init(|| {
        let rows: Vec<_> = rows()
            .into_iter()
            .filter(|r| r.0 + (r.1 / 12) as i32 - 1 <= 1997)
            .collect();
        assert_eq!(rows.len(), 55);
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

/// The one-year view of `model` on `column` and its wall time.
fn view(
    model: MeyersModel,
    column: &str,
    n_sims: usize,
    seed: u64,
    refit: OneYearRefit,
) -> (BayesOneYearFit<MeyersFit>, f64) {
    let t = Instant::now();
    let view = Meyers {
        model,
        n_sims,
        seed,
        ..Meyers::default()
    }
    .one_year(
        training(),
        column,
        "premium",
        &BayesOneYear {
            refit: Some(refit),
            n_sims: Some(n_sims),
            ..BayesOneYear::default()
        },
    )
    .unwrap_or_else(|e| panic!("{model:?} {column}: {e}"));
    (view, t.elapsed().as_secs_f64())
}

/// What one case measures at one seed.
#[derive(Debug, Clone, Copy)]
struct Measured {
    mean_z: f64,
    flagged_share: f64,
    rescued: f64,
    /// Simulations the default refitted, of `SIMS`.
    refitted: usize,
    gap_matched: f64,
    gap_reweighted: f64,
    /// The rescued simulations' total sd under the default and under
    /// `ReweightOnly`, over their sd refitted.
    sd_matched: f64,
    sd_reweighted: f64,
    life_ratio: f64,
    /// Wall times: `ReweightOnly` and the default at `SIMS`, `RefitAll` at
    /// `REFIT_SIMS`.
    times: [f64; 3],
}

fn measure(name: &str, model: MeyersModel, column: &str, seed: u64) -> Measured {
    let (only, t_only) = view(model, column, SIMS, seed, OneYearRefit::ReweightOnly);
    let (default, t_default) = view(model, column, SIMS, seed, OneYearRefit::default());
    let (all, t_all) = view(model, column, REFIT_SIMS, seed, OneYearRefit::RefitAll);
    // The same years under every setting.
    assert_eq!(
        only.pareto_k, default.pareto_k,
        "{name}: the default's years"
    );
    assert_eq!(
        all.pareto_k,
        only.pareto_k[..REFIT_SIMS],
        "{name}: RefitAll's years"
    );
    let (t_only_, t_def, t_all_) = (totals(&only.cdr), totals(&default.cdr), totals(&all.cdr));
    let s = sd(&t_def);
    let mean_z = mean(&t_def) / (s / (SIMS as f64).sqrt());
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
        .filter(|&i| i < REFIT_SIMS)
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
        refitted: default.refitted.len(),
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

/// The ODP bootstrap's ratio of one-year to lifetime sd on group 353 paid,
/// at 10,000 simulations.
fn odp_life_ratio(seed: u64) -> f64 {
    let boot = OdpBootstrap {
        n_sims: 10_000,
        seed,
        ..Default::default()
    };
    let one = boot
        .one_year(
            training(),
            "paid",
            &OneYearMethod::ChainLadder(ChainLadder::default()),
        )
        .unwrap();
    let life = boot.fit(training(), "paid").unwrap();
    sd(&totals(&one.cdr)) / life.reserves.total().std_dev()
}

fn bounds(name: &str) -> (f64, f64, (f64, f64)) {
    let &(_, floor, gap, band) = BOUNDS.iter().find(|b| b.0 == name).unwrap();
    (floor, gap, band)
}

#[test]
fn the_one_year_view_matches_its_refits() {
    let mut crc_ratio = None;
    for &(name, model, column) in strict_cases() {
        let m = measure(name, model, column, 0);
        let (floor, gap, (lo, hi)) = bounds(name);
        assert!(m.mean_z.abs() < 5.0, "{name}: mean z {}", m.mean_z);
        assert!(m.rescued >= floor, "{name}: rescued {}", m.rescued);
        assert!(m.gap_matched < gap, "{name}: gap {}", m.gap_matched);
        assert!(
            m.gap_matched < m.gap_reweighted,
            "{name}: matching {} is no closer than reweighting {}",
            m.gap_matched,
            m.gap_reweighted
        );
        assert!(
            (lo..hi).contains(&m.life_ratio),
            "{name}: one-year over lifetime sd {}",
            m.life_ratio
        );
        if model == MeyersModel::Crc {
            crc_ratio = Some(m.life_ratio);
        }
    }
    let odp = odp_life_ratio(0);
    let r = crc_ratio.unwrap() / odp;
    println!("ODP bootstrap one-year over lifetime sd {odp:.4}; CRC over it {r:.4}");
    assert!(
        (ODP_BAND.0..ODP_BAND.1).contains(&r),
        "CRC's one-year share over the ODP bootstrap's: {r}"
    );
}

/// Seed-to-seed spread of the measured values (ignored; release build).
#[test]
#[ignore]
fn seed_sweep() {
    let seeds: u64 = std::env::var("SWEEP_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let mut by_case: Vec<Vec<Measured>> = vec![Vec::new(); CASES.len()];
    let mut odp = Vec::new();
    for seed in 0..seeds {
        for (k, &(name, model, column)) in CASES.iter().enumerate() {
            by_case[k].push(measure(name, model, column, seed));
        }
        odp.push(odp_life_ratio(seed));
        println!("ODP seed {seed}: life ratio {:.4}", odp.last().unwrap());
    }
    let summary = |name: &str, row: &str, v: &[f64]| {
        let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = v.iter().copied().fold(f64::INFINITY, f64::min);
        println!(
            "{name:13} {row:15} mean {:9.4} sd {:8.4} min {min:9.4} max {max:9.4} mean-4sd {:9.4} \
             mean+4sd {:9.4}",
            mean(v),
            sd(v),
            mean(v) - 4.0 * sd(v),
            mean(v) + 4.0 * sd(v)
        );
    };
    for (k, &(name, _, _)) in CASES.iter().enumerate() {
        let ms = &by_case[k];
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
    summary("ODP", "life_ratio", &odp);
    let crc: Vec<f64> = by_case[0]
        .iter()
        .zip(&odp)
        .map(|(m, o)| m.life_ratio / o)
        .collect();
    summary("CRC over ODP", "life_ratio", &crc);
}

/// Wall times at 10,000 simulations: `ReweightOnly` and the default, and
/// `RefitFlagged` at 1,000 simulations (it refits nearly every one, so its
/// time grows in proportion; ignored, release build, many minutes).
#[test]
#[ignore]
fn runtime() {
    for &(name, model, column) in CASES {
        let (only, t_only) = view(model, column, 10_000, 0, OneYearRefit::ReweightOnly);
        let (default, t_default) = view(model, column, 10_000, 0, OneYearRefit::default());
        let (flagged, t_flagged) = view(model, column, 1_000, 0, OneYearRefit::RefitFlagged);
        println!(
            "{name}: 10,000 simulations: ReweightOnly {t_only:.1} s, default {t_default:.1} s \
             (flagged {:.3}, matched {}, refitted {}); 1,000 simulations: RefitFlagged \
             {t_flagged:.1} s ({} refits)",
            only.flagged_share(),
            default.moment_matched.len(),
            default.refitted.len(),
            flagged.refitted.len()
        );
    }
}
