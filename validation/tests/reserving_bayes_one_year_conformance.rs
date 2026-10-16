//! The conformance suite of the Bayesian one-year view
//! (`docs/design/reserving-bayes.md`, decision 17), in a release build:
//! the checks of the generic engine that need no external reference and
//! too many NUTS refits for a unit test. [`conform`] runs on every model
//! with a one-year view (`BayesOdp` on RAA and GenIns, `BayesBf` on GenIns
//! with premium, Meyers' CRC and CSR on group 353's paid and CAY on
//! its incurred, decision 19, and the compartmental models 1 and 2 on
//! company 337, decision 20); a new Bayesian model's one-year view is added to
//! [`every_model_conforms`] before it ships (AGENTS.md, "Checks"). The
//! unit-test half of the suite, run on each model's `YearModel`, is
//! `prospicio_reserving::bayes::one_year::conformance` (crate-private).
//!
//! For each model, at the defaults otherwise:
//!
//! 1. the total claims development result has mean zero: within 5 Monte
//!    Carlo standard errors over 10,000 reweighted simulations;
//! 2. the opening reserve is the posterior mean of the lifetime reserve:
//!    their totals within 5 standard errors of the lifetime total;
//! 3. a seed replays the view bit for bit, its first 500 simulations
//!    as part of 10,000;
//! 4. on 300 simulations (60 for Meyers' models, 30 for compartmental
//!    model 1 and 6 for model 2, whose refits cost seconds; they also
//!    sample 4 chains of 300 warm-up and 300 kept draws, about a sixth of
//!    their default's cost or less), `RefitAll` sees
//!    `ReweightOnly`'s years (the
//!    same `k̂`), and `RefitFlagged` gives `RefitAll`'s rows for the
//!    flagged simulations and `ReweightOnly`'s for the others;
//!    `MomentMatchThenRefit` sees the same years, moment-matches or
//!    refits each flagged simulation (never both), and gives `RefitAll`'s
//!    rows for those it refits and `ReweightOnly`'s for the unflagged
//!    (decision 18); it matches at least one (not required of
//!    compartmental model 2, whose simulations matching almost never
//!    rescues), and replays bit for bit
//!    (rows, `k̂`, the matched and the refitted) on one thread against
//!    the default pool, matching's own parallel steps nested in the
//!    simulations' included;
//! 5. on those simulations, the root mean square gap between a reweighted and a
//!    refitted simulation's total, in units of the total's sd, is below
//!    `REFITS.gaps.0` where `k̂` is at most 0.7 (over at least
//!    `MIN_UNFLAGGED` such simulations) and below `REFITS.gaps.1` above it
//!    (`MEYERS_REFITS` for Meyers' models, nearly all of whose simulations
//!    are flagged);
//! 6. a triangle whose every origin is at the last age gives a zero claims
//!    development result, `k̂` 0 and no refit under every setting (not for
//!    the compartmental models, whose fit needs a future cell: their unit
//!    half samples such a square directly).
//!
//! Every bound is at least the largest model's mean (or mean offset) plus
//! 4 seed-to-seed sds over seeds 0 to 19 (largest: mean z 4.40, opening z
//! 4.75, both GenIns under `BayesOdp`; gaps 0.071, `BayesBf`, and 0.455,
//! RAA; Meyers' flagged gap 0.606, CSR), and every seed passes
//! (`seed_sweep`, ignored: `SWEEP_SEEDS=20 cargo test --release -p
//! prospicio-validation --features slow-bayes --test
//! reserving_bayes_one_year_conformance -- --ignored seed_sweep
//! --nocapture`; `SWEEP_MEYERS_ONLY=1` runs Meyers' models alone).

use std::sync::OnceLock;
use std::time::Instant;

use prospicio_bayes::nuts::Sampler;
use prospicio_prob::{Distribution, PredictiveDistribution};
use prospicio_reserving::bayes::{
    BayesBf, BayesBfFit, BayesOdp, BayesOdpFit, BayesOneYear, BayesOneYearFit, Compartmental,
    CompartmentalFit, CompartmentalPriors, Meyers, MeyersFit, MeyersModel, OneYearRefit,
    Outstanding,
};
use prospicio_reserving::{DevelopmentColumn, Grain, Lag, Long, Month, Triangle};
use prospicio_validation::{triangle, triangle_columns};

/// The simulations of the mean check.
const SIMS: usize = 10_000;
/// A model's refit checks: the simulations refitted for the gap and the
/// refit settings, the bounds on the root mean square gap, in units of
/// the total's sd, where `k̂` is at most 0.7 and above it, and whether
/// moment matching must rescue at least one simulation.
#[derive(Clone, Copy)]
struct Refits {
    sims: usize,
    gaps: (f64, f64),
    matches: bool,
}

const REFITS: Refits = Refits {
    sims: 300,
    gaps: (0.08, 0.5),
    matches: true,
};
/// Meyers' models, whose refits cost seconds and nearly all of whose
/// simulations are flagged (a year cell has its origin's smallest
/// `sigma_d`): the unflagged gap is checked only over at least
/// `MIN_UNFLAGGED` simulations.
const MEYERS_REFITS: Refits = Refits {
    sims: 60,
    gaps: (0.08, 0.65),
    matches: true,
};
/// The compartmental models on company 337, most of whose simulations are
/// flagged; model 2's refits cost seconds each even with the short
/// sampler, and moment matching rescues almost none of them.
const COMPARTMENTAL_REFITS: Refits = Refits {
    sims: 30,
    gaps: (1.0, 1.0),
    matches: true,
};
const COMPARTMENTAL_2_REFITS: Refits = Refits {
    sims: 6,
    gaps: (1.0, 2.0),
    matches: false,
};
/// Fewest unflagged simulations whose gap is checked.
const MIN_UNFLAGGED: usize = 10;

/// The bits of each value, so that a NaN Pareto `k` compares equal to itself.
fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn sd(x: &[f64]) -> f64 {
    let m = mean(x);
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
}

fn totals(d: &PredictiveDistribution) -> Vec<f64> {
    let n = d.n_components();
    d.draw_matrix().chunks(n).map(|r| r.iter().sum()).collect()
}

/// A model's one-year view, at `n_sims`, `seed` and `refit`.
type Run<'a, F> = dyn Fn(usize, u64, OneYearRefit) -> BayesOneYearFit<F> + Sync + 'a;

/// What `conform` measures, for the sweep.
struct Measured {
    mean_z: f64,
    opening_z: f64,
    gap_unflagged: f64,
    gap_flagged: f64,
    flagged: usize,
}

/// The suite on one model: `run` gives its view of the open triangle,
/// `closed` of the triangle without a year, and `reserves` the lifetime
/// reserves of its fit. Panics naming the model and the property; without
/// `strict` (the sweep), only on the exact properties.
fn conform<F: Sync + Send>(
    name: &str,
    run: &Run<'_, F>,
    closed: Option<&Run<'_, F>>,
    reserves: fn(&F) -> &PredictiveDistribution,
    refits: Refits,
    seed: u64,
    strict: bool,
) -> Measured {
    let t = Instant::now();
    let Refits {
        sims: refit_sims,
        gaps,
        matches,
    } = refits;
    let only = OneYearRefit::ReweightOnly;
    // 1. The mean is zero.
    let full = run(SIMS, seed, only);
    let total = totals(&full.cdr);
    let mean_z = mean(&total) / (sd(&total) / (SIMS as f64).sqrt());
    assert!(!strict || mean_z.abs() < 5.0, "{name}: mean z {mean_z}");
    // 2. The opening reserve is the lifetime reserve's posterior mean.
    let lifetime = reserves(&full.fit).total();
    let opening: f64 = full.opening_reserve.iter().sum();
    let n_life = reserves(&full.fit).n_sims() as f64;
    let opening_z = (opening - lifetime.mean()) / (lifetime.std_dev() / n_life.sqrt());
    assert!(
        !strict || opening_z.abs() < 5.0,
        "{name}: opening z {opening_z}"
    );
    // 3. A seed replays the view: simulation i depends only on the seed
    // and i, not on the number of simulations or how they are split
    // over threads (the unit half checks one thread against many).
    let one = run(500, seed, only);
    let n = full.cdr.n_components();
    assert_eq!(
        one.cdr.draw_matrix(),
        &full.cdr.draw_matrix()[..500 * n],
        "{name}: replay"
    );
    // 4. The refit settings see the same years.
    let a = run(refit_sims, seed, only);
    let b = run(refit_sims, seed, OneYearRefit::RefitAll);
    let c = run(refit_sims, seed, OneYearRefit::RefitFlagged);
    assert_eq!(
        bits(&a.pareto_k),
        bits(&b.pareto_k),
        "{name}: RefitAll's years"
    );
    assert_eq!(b.refitted, (0..refit_sims).collect::<Vec<_>>());
    let is_flagged: Vec<bool> = a.pareto_k.iter().map(|&k| k.is_nan() || k > 0.7).collect();
    assert_eq!(
        c.refitted,
        (0..refit_sims)
            .filter(|&i| is_flagged[i])
            .collect::<Vec<_>>(),
        "{name}: RefitFlagged refits the flagged"
    );
    let row = |v: &BayesOneYearFit<F>, i: usize| v.cdr.draw_matrix()[i * n..(i + 1) * n].to_vec();
    for (i, &f) in is_flagged.iter().enumerate() {
        let expected = if f { row(&b, i) } else { row(&a, i) };
        assert_eq!(
            row(&c, i),
            expected,
            "{name}: RefitFlagged's simulation {i}"
        );
    }
    let m = run(refit_sims, seed, OneYearRefit::MomentMatchThenRefit);
    assert_eq!(
        bits(&m.pareto_k),
        bits(&a.pareto_k),
        "{name}: MomentMatchThenRefit's years"
    );
    assert!(
        !matches || !m.moment_matched.is_empty(),
        "{name}: MomentMatchThenRefit matched nothing"
    );
    let m1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| run(refit_sims, seed, OneYearRefit::MomentMatchThenRefit));
    assert_eq!(
        m1.cdr.draw_matrix(),
        m.cdr.draw_matrix(),
        "{name}: MomentMatchThenRefit's replay on one thread"
    );
    assert_eq!(
        bits(&m1.pareto_k),
        bits(&m.pareto_k),
        "{name}: matched replay's k"
    );
    assert_eq!(
        m1.moment_matched, m.moment_matched,
        "{name}: matched replay's matched"
    );
    assert_eq!(m1.refitted, m.refitted, "{name}: matched replay's refitted");
    let mut either = m.moment_matched.clone();
    either.extend(&m.refitted);
    either.sort_unstable();
    assert_eq!(
        either, c.refitted,
        "{name}: matched or refitted, never both"
    );
    for (i, &f) in is_flagged.iter().enumerate() {
        if m.refitted.contains(&i) {
            assert_eq!(row(&m, i), row(&b, i), "{name}: refit {i} after matching");
        } else if !f {
            assert_eq!(row(&m, i), row(&a, i), "{name}: unflagged {i}");
        }
    }
    // 5. The reweighting is close to the refits.
    let (ta, tb) = (totals(&a.cdr), totals(&b.cdr));
    let s = sd(&total);
    let gap = |flagged: bool| {
        let squares: Vec<f64> = (0..refit_sims)
            .filter(|&i| is_flagged[i] == flagged)
            .map(|i| ((ta[i] - tb[i]) / s).powi(2))
            .collect();
        if squares.is_empty() {
            0.0
        } else {
            mean(&squares).sqrt()
        }
    };
    let (gap_unflagged, gap_flagged) = (gap(false), gap(true));
    let flagged = is_flagged.iter().filter(|&&f| f).count();
    assert!(
        !strict || refit_sims - flagged < MIN_UNFLAGGED || gap_unflagged < gaps.0,
        "{name}: rms gap {gap_unflagged} sd where k <= 0.7"
    );
    assert!(
        !strict || gap_flagged < gaps.1,
        "{name}: rms gap {gap_flagged} sd where k > 0.7 ({flagged} simulations)"
    );
    // 6. A year without cells changes and refits nothing.
    let every = [
        only,
        OneYearRefit::RefitFlagged,
        OneYearRefit::RefitAll,
        OneYearRefit::MomentMatchThenRefit,
    ];
    for (closed, refit) in closed.iter().flat_map(|c| every.map(|r| (c, r))) {
        let v = closed(200, seed, refit);
        assert!(v.refitted.is_empty(), "{name}: {refit:?} refitted");
        assert!(v.moment_matched.is_empty(), "{name}: {refit:?} matched");
        assert!(v.pareto_k.iter().all(|&k| k == 0.0), "{name}: {refit:?}");
        assert!(
            v.cdr.draw_matrix().iter().all(|x| x.abs() < 1e-6),
            "{name}: {refit:?} gives a CDR without a year"
        );
    }
    println!(
        "{name} seed {seed}: mean z {mean_z:.3}, opening z {opening_z:.3}, gap {gap_unflagged:.4} \
         ({} unflagged) and {gap_flagged:.4} ({flagged} flagged), {:.1?}",
        refit_sims - flagged,
        t.elapsed()
    );
    Measured {
        mean_z,
        opening_z,
        gap_unflagged,
        gap_flagged,
        flagged,
    }
}

/// A fully developed 4 x 4 square with `paid` and `premium`.
fn square() -> Triangle {
    let rows: [[f64; 4]; 4] = [
        [357.0, 1124.0, 1735.0, 2218.0],
        [352.0, 1236.0, 2170.0, 3353.0],
        [290.0, 1292.0, 2219.0, 3235.0],
        [310.0, 1418.0, 2195.0, 3757.0],
    ];
    let mut origin = Vec::new();
    let (mut ages, mut paid, mut premium) = (Vec::new(), Vec::new(), Vec::new());
    for (k, row) in rows.iter().enumerate() {
        for (d, &v) in row.iter().enumerate() {
            origin.push(Month::january(2001 + k as i32));
            ages.push(12 * (d as u32 + 1));
            paid.push(v);
            premium.push(5000.0 + 500.0 * k as f64);
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("paid", &paid), ("premium", &premium)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

fn settings(refit: OneYearRefit) -> BayesOneYear {
    BayesOneYear {
        refit,
        ..BayesOneYear::default()
    }
}

/// `BayesOdp` on `column` of `tri`.
fn odp<'a>(tri: &'a Triangle, column: &'a str) -> Box<Run<'a, BayesOdpFit>> {
    Box::new(move |n_sims, seed, refit| {
        BayesOdp {
            n_sims,
            seed,
            ..BayesOdp::default()
        }
        .one_year(tri, column, &settings(refit))
        .unwrap()
    })
}

/// `BayesBf` with GenIns' premium, apriori 0.5 and cv 0.2, as the Stan
/// parity case of `reserving_bayes_bf.rs`.
fn bf(tri: &Triangle) -> Box<Run<'_, BayesBfFit>> {
    Box::new(move |n_sims, seed, refit| {
        BayesBf {
            apriori: 0.5,
            cv: 0.2,
            n_sims,
            seed,
            ..BayesBf::default()
        }
        .one_year(tri, "paid", "premium", &settings(refit))
        .unwrap()
    })
}

/// Group 353 Commercial Auto (`data/clrd_comauto_353.csv`), the cells at
/// calendar years up to 1997, columns `paid`, `incurred` and `premium`, as
/// in `reserving_meyers.rs`.
fn comauto_353() -> &'static Triangle {
    static TRIANGLE: OnceLock<Triangle> = OnceLock::new();
    TRIANGLE.get_or_init(|| {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/clrd_comauto_353.csv");
        let text = std::fs::read_to_string(path).expect("data/clrd_comauto_353.csv");
        let mut lines = text.lines().filter(|l| !l.starts_with('#'));
        assert_eq!(
            lines.next(),
            Some("origin,development,paid,incurred,premium")
        );
        let rows: Vec<(i32, Lag, [f64; 3])> = lines
            .map(|l| {
                let f: Vec<&str> = l.split(',').collect();
                let v = |k: usize| -> f64 { f[k].parse().unwrap() };
                (
                    f[0].parse().unwrap(),
                    f[1].parse().unwrap(),
                    [v(2), v(3), v(4)],
                )
            })
            .filter(|r| r.0 + (r.1 / 12) as i32 - 1 <= 1997)
            .collect();
        let origin: Vec<Month> = rows.iter().map(|r| Month::january(r.0)).collect();
        let ages: Vec<Lag> = rows.iter().map(|r| r.1).collect();
        let column = |k: usize| -> Vec<f64> { rows.iter().map(|r| r.2[k]).collect() };
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

/// Meyers' `model` on `column` of `tri` with premium, 4 chains of 300
/// warm-up and 300 kept draws.
fn meyers<'a>(tri: &'a Triangle, model: MeyersModel, column: &'a str) -> Box<Run<'a, MeyersFit>> {
    Box::new(move |n_sims, seed, refit| {
        Meyers {
            model,
            sampler: Sampler {
                tune: 300,
                draws: 300,
                ..Meyers::default().sampler
            },
            n_sims,
            seed,
            ..Meyers::default()
        }
        .one_year(tri, column, "premium", &settings(refit))
        .unwrap()
    })
}

/// Company 337 workers' compensation (`data/clrd_wkcomp_337.csv`), columns
/// `paid`, `incurred` and `premium`: the papers' training triangle (the
/// cells at calendar years up to 1996, and 1997 with its premium only).
fn wkcomp_337() -> Triangle {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/clrd_wkcomp_337.csv");
    let text = std::fs::read_to_string(path).expect("data/clrd_wkcomp_337.csv");
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    assert_eq!(
        lines.next(),
        Some("origin,development,paid,incurred,case_incurred,premium,premium_net")
    );
    let all: Vec<(i32, Lag, [f64; 3])> = lines
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let v = |k: usize| -> f64 { f[k].parse().unwrap() };
            (
                f[0].parse().unwrap(),
                f[1].parse().unwrap(),
                [v(2), v(3), v(5)],
            )
        })
        .collect();
    let mut rows: Vec<_> = all
        .iter()
        .copied()
        .filter(|r| r.0 + (r.1 / 12) as i32 - 1 <= 1996)
        .collect();
    let premium = all.iter().find(|r| r.0 == 1997).unwrap().2[2];
    rows.push((1997, 12, [f64::NAN, f64::NAN, premium]));
    let origin: Vec<Month> = rows.iter().map(|r| Month::january(r.0)).collect();
    let ages: Vec<Lag> = rows.iter().map(|r| r.1).collect();
    let column = |k: usize| -> Vec<f64> { rows.iter().map(|r| r.2[k]).collect() };
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
}

/// Compartmental model 1 (or, with `two`, model 2) on `tri` with
/// outstanding from incurred, 4 chains of 300 warm-up and 300 kept draws.
fn compartmental(tri: &Triangle, two: bool) -> Box<Run<'_, CompartmentalFit>> {
    Box::new(move |n_sims, seed, refit| {
        let preset = if two {
            Compartmental::model_2(CompartmentalPriors::gesmann_morris_2())
        } else {
            Compartmental::new(CompartmentalPriors::gesmann_morris())
        };
        Compartmental {
            sampler: Sampler {
                tune: 300,
                draws: 300,
                ..preset.sampler
            },
            n_sims,
            seed,
            ..preset
        }
        .one_year(
            tri,
            "paid",
            Outstanding::FromIncurred("incurred"),
            "premium",
            &settings(refit),
        )
        .unwrap()
    })
}

/// Runs the suite on every model at `seed`.
fn every_model(seed: u64, strict: bool) -> Vec<(&'static str, Measured)> {
    let (raa, genins, premium, closed) = (
        triangle("raa"),
        triangle("genins"),
        triangle_columns("genins_premium", &["paid", "premium"]),
        square(),
    );
    let odp_reserves: fn(&BayesOdpFit) -> &PredictiveDistribution = |f| &f.reserves;
    let bf_reserves: fn(&BayesBfFit) -> &PredictiveDistribution = |f| &f.reserves;
    let meyers_reserves: fn(&MeyersFit) -> &PredictiveDistribution = |f| &f.reserves;
    let group = comauto_353();
    // SWEEP_MEYERS_ONLY and SWEEP_COMPARTMENTAL_ONLY (the sweep): those
    // models alone.
    let meyers_only = !strict && std::env::var("SWEEP_MEYERS_ONLY").is_ok();
    let compartmental_only = !strict && std::env::var("SWEEP_COMPARTMENTAL_ONLY").is_ok();
    let mut out = if meyers_only || compartmental_only {
        Vec::new()
    } else {
        vec![
            (
                "BayesOdp RAA",
                conform(
                    "BayesOdp RAA",
                    &*odp(&raa, "values"),
                    Some(&*odp(&closed, "paid")),
                    odp_reserves,
                    REFITS,
                    seed,
                    strict,
                ),
            ),
            (
                "BayesOdp GenIns",
                conform(
                    "BayesOdp GenIns",
                    &*odp(&genins, "values"),
                    Some(&*odp(&closed, "paid")),
                    odp_reserves,
                    REFITS,
                    seed,
                    strict,
                ),
            ),
            (
                "BayesBf GenIns",
                conform(
                    "BayesBf GenIns",
                    &*bf(&premium),
                    Some(&*bf(&closed)),
                    bf_reserves,
                    REFITS,
                    seed,
                    strict,
                ),
            ),
        ]
    };
    let meyers_models: &[(&str, MeyersModel, &str)] = if compartmental_only {
        &[]
    } else {
        &[
            ("Meyers CRC paid", MeyersModel::Crc, "paid"),
            ("Meyers CSR paid", MeyersModel::Csr, "paid"),
            ("Meyers CAY incurred", MeyersModel::Cay, "incurred"),
        ]
    };
    for &(name, model, column) in meyers_models {
        let measured = conform(
            name,
            &*meyers(group, model, column),
            Some(&*meyers(&closed, model, "paid")),
            meyers_reserves,
            MEYERS_REFITS,
            seed,
            strict,
        );
        out.push((name, measured));
    }
    if meyers_only {
        return out;
    }
    let compartmental_reserves: fn(&CompartmentalFit) -> &PredictiveDistribution = |f| &f.reserves;
    // A compartmental fit needs a future cell, so check 6 runs in the
    // unit half only (`compartmental::tests`, sampling the closed square
    // directly).
    let training = wkcomp_337();
    for (name, two, refits) in [
        ("Compartmental 1", false, COMPARTMENTAL_REFITS),
        ("Compartmental 2", true, COMPARTMENTAL_2_REFITS),
    ] {
        let measured = conform(
            name,
            &*compartmental(&training, two),
            None,
            compartmental_reserves,
            refits,
            seed,
            strict,
        );
        out.push((name, measured));
    }
    out
}

#[test]
fn every_model_conforms() {
    every_model(0, true);
}

/// Seed-to-seed spread of the measured values (ignored; release build).
#[test]
#[ignore]
fn seed_sweep() {
    let seeds: u64 = std::env::var("SWEEP_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let mut by_model: Vec<(&str, Vec<Measured>)> = Vec::new();
    for seed in 0..seeds {
        for (name, m) in every_model(seed, false) {
            match by_model.iter_mut().find(|(n, _)| *n == name) {
                Some((_, v)) => v.push(m),
                None => by_model.push((name, vec![m])),
            }
        }
    }
    for (name, ms) in &by_model {
        for (row, f) in [
            ("mean_z", (|m: &Measured| m.mean_z) as fn(&Measured) -> f64),
            ("opening_z", |m| m.opening_z),
            ("gap_unflagged", |m| m.gap_unflagged),
            ("gap_flagged", |m| m.gap_flagged),
            ("flagged", |m| m.flagged as f64),
        ] {
            let v: Vec<f64> = ms.iter().map(f).collect();
            let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            println!(
                "{name:16} {row:14} mean {:8.4} sd {:8.4} max {max:8.4} mean + 4 sd {:8.4}",
                mean(&v),
                sd(&v),
                mean(&v) + 4.0 * sd(&v)
            );
        }
    }
}
