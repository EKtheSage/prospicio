//! The conformance suite of the Bayesian one-year view
//! (`docs/design/reserving-bayes.md`, decision 17), in a release build:
//! the checks of the generic engine that need no external reference and
//! too many NUTS refits for a unit test. [`conform`] runs on every model
//! with a one-year view (`BayesOdp` on RAA and GenIns, `BayesBf` on GenIns
//! with premium); a new Bayesian model's one-year view is added to
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
//! 4. on 300 simulations, `RefitAll` sees `ReweightOnly`'s years (the
//!    same `k̂`), and `RefitFlagged` gives `RefitAll`'s rows for the
//!    flagged simulations and `ReweightOnly`'s for the others;
//! 5. on those 300, the root mean square gap between a reweighted and a
//!    refitted simulation's total, in units of the total's sd, is below
//!    `UNFLAGGED_GAP` where `k̂` is at most 0.7 and below `FLAGGED_GAP`
//!    above it;
//! 6. a triangle whose every origin is at the last age gives a zero claims
//!    development result, `k̂` 0 and no refit under every setting.
//!
//! Every bound is at least the largest model's mean (or mean offset) plus
//! 4 seed-to-seed sds over seeds 0 to 19 (largest: mean z 4.40, opening z
//! 4.75, both GenIns under `BayesOdp`; gaps 0.071, `BayesBf`, and 0.455,
//! RAA), and every seed passes (`seed_sweep`, ignored:
//! `SWEEP_SEEDS=20 cargo test --release -p prospicio-validation --features
//! slow-bayes --test reserving_bayes_one_year_conformance -- --ignored
//! seed_sweep --nocapture`).

use std::time::Instant;

use prospicio_prob::{Distribution, PredictiveDistribution};
use prospicio_reserving::bayes::{
    BayesBf, BayesBfFit, BayesOdp, BayesOdpFit, BayesOneYear, BayesOneYearFit, OneYearRefit,
};
use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
use prospicio_validation::{triangle, triangle_columns};

/// The simulations of the mean check.
const SIMS: usize = 10_000;
/// The simulations refitted for the gap and the refit settings.
const REFIT_SIMS: usize = 300;
/// Bounds on the root mean square gap, in units of the total's sd.
const UNFLAGGED_GAP: f64 = 0.08;
const FLAGGED_GAP: f64 = 0.5;

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
fn conform<F: Sync>(
    name: &str,
    run: &Run<'_, F>,
    closed: &Run<'_, F>,
    reserves: fn(&F) -> &PredictiveDistribution,
    seed: u64,
    strict: bool,
) -> Measured {
    let t = Instant::now();
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
    let a = run(REFIT_SIMS, seed, only);
    let b = run(REFIT_SIMS, seed, OneYearRefit::RefitAll);
    let c = run(REFIT_SIMS, seed, OneYearRefit::RefitFlagged);
    assert_eq!(a.pareto_k, b.pareto_k, "{name}: RefitAll's years");
    assert_eq!(b.refitted, (0..REFIT_SIMS).collect::<Vec<_>>());
    let is_flagged: Vec<bool> = a.pareto_k.iter().map(|&k| k.is_nan() || k > 0.7).collect();
    assert_eq!(
        c.refitted,
        (0..REFIT_SIMS)
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
    // 5. The reweighting is close to the refits.
    let (ta, tb) = (totals(&a.cdr), totals(&b.cdr));
    let s = sd(&total);
    let gap = |flagged: bool| {
        let squares: Vec<f64> = (0..REFIT_SIMS)
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
        !strict || gap_unflagged < UNFLAGGED_GAP,
        "{name}: rms gap {gap_unflagged} sd where k <= 0.7"
    );
    assert!(
        !strict || gap_flagged < FLAGGED_GAP,
        "{name}: rms gap {gap_flagged} sd where k > 0.7 ({flagged} simulations)"
    );
    // 6. A year without cells changes and refits nothing.
    for refit in [only, OneYearRefit::RefitFlagged, OneYearRefit::RefitAll] {
        let v = closed(200, seed, refit);
        assert!(v.refitted.is_empty(), "{name}: {refit:?} refitted");
        assert!(v.pareto_k.iter().all(|&k| k == 0.0), "{name}: {refit:?}");
        assert!(
            v.cdr.draw_matrix().iter().all(|x| x.abs() < 1e-6),
            "{name}: {refit:?} gives a CDR without a year"
        );
    }
    println!(
        "{name} seed {seed}: mean z {mean_z:.3}, opening z {opening_z:.3}, gap {gap_unflagged:.4} \
         ({} unflagged) and {gap_flagged:.4} ({flagged} flagged), {:.1?}",
        REFIT_SIMS - flagged,
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
    vec![
        (
            "BayesOdp RAA",
            conform(
                "BayesOdp RAA",
                &*odp(&raa, "values"),
                &*odp(&closed, "paid"),
                odp_reserves,
                seed,
                strict,
            ),
        ),
        (
            "BayesOdp GenIns",
            conform(
                "BayesOdp GenIns",
                &*odp(&genins, "values"),
                &*odp(&closed, "paid"),
                odp_reserves,
                seed,
                strict,
            ),
        ),
        (
            "BayesBf GenIns",
            conform(
                "BayesBf GenIns",
                &*bf(&premium),
                &*bf(&closed),
                bf_reserves,
                seed,
                strict,
            ),
        ),
    ]
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
