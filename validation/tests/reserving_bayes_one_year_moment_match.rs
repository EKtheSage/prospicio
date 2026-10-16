//! Moment matching in the Bayesian one-year view
//! (`OneYearRefit::MomentMatchThenRefit`, `docs/design/reserving-bayes.md`,
//! decision 18), against refitting on the same years.
//!
//! For `BayesOdp` on RAA and GenIns and `BayesBf` on GenIns with premium
//! (apriori 0.5, cv 0.2), at the defaults otherwise, the view is run under
//! `ReweightOnly`, `RefitFlagged` and `MomentMatchThenRefit` on the same
//! 1,000 years, and:
//!
//! * exactly: moment matching sees `ReweightOnly`'s `k̂`; every flagged
//!   simulation is either moment-matched or refitted, never both; a
//!   refitted one's row is `RefitFlagged`'s (which is `RefitAll`'s, as the
//!   conformance suite checks) and an unflagged one's `ReweightOnly`'s,
//!   bit for bit;
//! * `rescued`, the share of flagged simulations that moment matching
//!   brings to `k̂` at most 0.7, is above a floor;
//! * `gap_matched`, the root mean square gap between a rescued
//!   simulation's total claims development result and its refitted value,
//!   in units of the total's sd, is below a bound, and below
//!   `gap_reweighted`, the same simulations' gap under `ReweightOnly`.
//!
//! The ignored `seed_sweep` runs 2,000 simulations with `RefitAll` too and
//! also reports `sd_ratio`, the total's sd under each setting over
//! `RefitAll`'s, and the wall times; its first 1,000 simulations are the
//! test's (a simulation depends only on the seed and its index). Every
//! bound is the sweep's mean minus (floor) or plus (bounds) at least 4
//! seed-to-seed sds over seeds 0 to 19 (`SWEEP_SEEDS=20 cargo test
//! --release -p prospicio-validation --features slow-bayes --test
//! reserving_bayes_one_year_moment_match -- --ignored seed_sweep
//! --nocapture`).

use std::time::{Duration, Instant};

use prospicio_prob::PredictiveDistribution;
use prospicio_reserving::Triangle;
use prospicio_reserving::bayes::{
    BayesBf, BayesBfFit, BayesOdp, BayesOdpFit, BayesOneYear, BayesOneYearFit, OneYearRefit,
};
use prospicio_validation::{triangle, triangle_columns};

/// Simulations in the test.
const SIMS: usize = 1_000;
/// Simulations in the sweep.
const SWEEP_SIMS: usize = 2_000;

/// Per model: the floor on `rescued` and the bound on `gap_matched`, at
/// `SIMS`.
/// Over seeds 0 to 19: `rescued` at least 0.9964 (GenIns; 1 elsewhere),
/// `gap_matched` mean + 4 sd 0.101 (RAA), 0.062 (GenIns) and 0.064
/// (`BayesBf`), seed 0 at 0.049, 0.038 and 0.041.
const BOUNDS: &[(&str, f64, f64)] = &[
    ("BayesOdp RAA", 0.99, 0.11),
    ("BayesOdp GenIns", 0.99, 0.07),
    ("BayesBf GenIns", 0.99, 0.07),
];

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn sd(x: &[f64]) -> f64 {
    let m = mean(x);
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
}

fn totals(d: &PredictiveDistribution, n: usize) -> Vec<f64> {
    let c = d.n_components();
    d.draw_matrix()
        .chunks(c)
        .take(n)
        .map(|r| r.iter().sum())
        .collect()
}

/// A model's one-year view at `n_sims`, `seed` and `refit`.
type Run<'a, F> = dyn Fn(usize, u64, OneYearRefit) -> BayesOneYearFit<F> + Sync + 'a;

/// One model's views on the same years.
struct Views<F> {
    only: BayesOneYearFit<F>,
    flagged: BayesOneYearFit<F>,
    matched: BayesOneYearFit<F>,
    all: Option<BayesOneYearFit<F>>,
    times: [f64; 3],
}

fn timed<F>(
    run: &Run<'_, F>,
    n: usize,
    seed: u64,
    refit: OneYearRefit,
) -> (BayesOneYearFit<F>, Duration) {
    let t = Instant::now();
    let v = run(n, seed, refit);
    (v, t.elapsed())
}

fn views<F>(run: &Run<'_, F>, n: usize, seed: u64, with_all: bool) -> Views<F> {
    let (only, a) = timed(run, n, seed, OneYearRefit::ReweightOnly);
    let (flagged, b) = timed(run, n, seed, OneYearRefit::RefitFlagged);
    let (matched, c) = timed(run, n, seed, OneYearRefit::MomentMatchThenRefit);
    let all = with_all.then(|| run(n, seed, OneYearRefit::RefitAll));
    Views {
        only,
        flagged,
        matched,
        all,
        times: [a.as_secs_f64(), b.as_secs_f64(), c.as_secs_f64()],
    }
}

/// What the views measure over their first `n` simulations.
#[derive(Debug, Clone, Copy)]
struct Measured {
    flagged: usize,
    rescued: f64,
    gap_matched: f64,
    gap_reweighted: f64,
    /// The rescued simulations' totals: their sd under each setting over
    /// their sd refitted, and the mean gap to the refits (in units of the
    /// total's sd).
    spread_matched: f64,
    spread_reweighted: f64,
    bias_matched: f64,
    bias_reweighted: f64,
    /// The total's sd over `RefitAll`'s (NaN without it).
    sd_ratio: f64,
    sd_ratio_flagged: f64,
    sd_ratio_reweight: f64,
}

/// Checks the exact properties on the first `n` simulations of `v` and
/// measures them.
fn measure<F>(name: &str, v: &Views<F>, n: usize) -> Measured {
    let c = v.only.cdr.n_components();
    let row = |f: &BayesOneYearFit<F>, i: usize| f.cdr.draw_matrix()[i * c..(i + 1) * c].to_vec();
    let below = |list: &[usize]| list.iter().copied().filter(|&i| i < n).collect::<Vec<_>>();
    let (refitted, matched) = (below(&v.matched.refitted), below(&v.matched.moment_matched));

    assert_eq!(
        v.matched.pareto_k[..n],
        v.only.pareto_k[..n],
        "{name}: the same years"
    );
    let flagged: Vec<usize> = (0..n)
        .filter(|&i| {
            let k = v.only.pareto_k[i];
            k.is_nan() || k > 0.7
        })
        .collect();
    assert_eq!(below(&v.flagged.refitted), flagged, "{name}: RefitFlagged");
    let mut both = matched.clone();
    both.extend(&refitted);
    both.sort_unstable();
    assert_eq!(both, flagged, "{name}: matched or refitted, never both");
    for i in 0..n {
        if refitted.contains(&i) {
            assert_eq!(row(&v.matched, i), row(&v.flagged, i), "{name}: refit {i}");
        } else if !matched.contains(&i) {
            assert_eq!(
                row(&v.matched, i),
                row(&v.only, i),
                "{name}: simulation {i}"
            );
        }
    }
    if let Some(all) = &v.all {
        for &i in &flagged {
            assert_eq!(row(&v.flagged, i), row(all, i), "{name}: RefitAll {i}");
        }
    }

    let (t_o, t_f, t_m) = (
        totals(&v.only.cdr, n),
        totals(&v.flagged.cdr, n),
        totals(&v.matched.cdr, n),
    );
    let s = sd(&t_f);
    // A rescued simulation's refitted value is RefitFlagged's.
    let gap = |t: &[f64]| {
        if matched.is_empty() {
            return 0.0;
        }
        let sq: Vec<f64> = matched
            .iter()
            .map(|&i| ((t[i] - t_f[i]) / s).powi(2))
            .collect();
        mean(&sq).sqrt()
    };
    let pick = |t: &[f64]| matched.iter().map(|&i| t[i]).collect::<Vec<_>>();
    let spread = |t: &[f64]| sd(&pick(t)) / sd(&pick(&t_f));
    let bias = |t: &[f64]| {
        let (a, b) = (pick(t), pick(&t_f));
        mean(
            &a.iter()
                .zip(&b)
                .map(|(x, y)| (x - y) / s)
                .collect::<Vec<_>>(),
        )
    };
    let ratio = |t: &[f64]| {
        v.all
            .as_ref()
            .map_or(f64::NAN, |all| sd(t) / sd(&totals(&all.cdr, n)))
    };
    Measured {
        flagged: flagged.len(),
        rescued: matched.len() as f64 / flagged.len().max(1) as f64,
        gap_matched: gap(&t_m),
        gap_reweighted: gap(&t_o),
        spread_matched: spread(&t_m),
        spread_reweighted: spread(&t_o),
        bias_matched: bias(&t_m),
        bias_reweighted: bias(&t_o),
        sd_ratio: ratio(&t_m),
        sd_ratio_flagged: ratio(&t_f),
        sd_ratio_reweight: ratio(&t_o),
    }
}

fn settings(refit: OneYearRefit) -> BayesOneYear {
    BayesOneYear {
        refit,
        ..BayesOneYear::default()
    }
}

fn odp(tri: &Triangle) -> Box<Run<'_, BayesOdpFit>> {
    Box::new(move |n_sims, seed, refit| {
        BayesOdp {
            n_sims,
            seed,
            ..BayesOdp::default()
        }
        .one_year(tri, "values", &settings(refit))
        .unwrap()
    })
}

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

/// One model's measures and wall times.
type Outcome = (&'static str, Measured, Option<Measured>, [f64; 3]);

/// One model's measures at `seed`: at `SIMS` only, or (the sweep) at
/// `SWEEP_SIMS` with `RefitAll` and also on its first `SIMS`; and the wall
/// times of `ReweightOnly`, `RefitFlagged` and `MomentMatchThenRefit`.
fn outcome<F>(name: &'static str, run: &Run<'_, F>, seed: u64, sweep: bool) -> Outcome {
    let n = if sweep { SWEEP_SIMS } else { SIMS };
    let v = views(run, n, seed, sweep);
    let small = measure(name, &v, SIMS);
    let large = sweep.then(|| measure(name, &v, SWEEP_SIMS));
    println!(
        "{name} seed {seed}: {small:?}; {large:?}; times {:?}",
        v.times
    );
    (name, small, large, v.times)
}

fn every_model(seed: u64, sweep: bool) -> Vec<Outcome> {
    let (raa, genins, premium) = (
        triangle("raa"),
        triangle("genins"),
        triangle_columns("genins_premium", &["paid", "premium"]),
    );
    vec![
        outcome("BayesOdp RAA", &*odp(&raa), seed, sweep),
        outcome("BayesOdp GenIns", &*odp(&genins), seed, sweep),
        outcome("BayesBf GenIns", &*bf(&premium), seed, sweep),
    ]
}

#[test]
fn moment_matching_agrees_with_the_refits() {
    for (name, m, _, _) in every_model(0, false) {
        let &(_, floor, bound) = BOUNDS.iter().find(|b| b.0 == name).unwrap();
        assert!(m.rescued >= floor, "{name}: rescued {}", m.rescued);
        assert!(m.gap_matched < bound, "{name}: gap {}", m.gap_matched);
        assert!(
            m.gap_matched < m.gap_reweighted,
            "{name}: matching ({}) no closer to the refits than reweighting ({})",
            m.gap_matched,
            m.gap_reweighted
        );
    }
}

/// Seed-to-seed spread of the measures (ignored; release build).
#[test]
#[ignore]
fn seed_sweep() {
    let seeds: u64 = std::env::var("SWEEP_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    type Row = (Vec<Measured>, Vec<Measured>, Vec<[f64; 3]>);
    let mut by_model: Vec<(&str, Row)> = Vec::new();
    for seed in 0..seeds {
        for (name, small, large, times) in every_model(seed, true) {
            let large = large.expect("the sweep runs RefitAll");
            match by_model.iter_mut().find(|(n, _)| *n == name) {
                Some((_, (a, b, c))) => {
                    a.push(small);
                    b.push(large);
                    c.push(times);
                }
                None => by_model.push((name, (vec![small], vec![large], vec![times]))),
            }
        }
    }
    let print = |name: &str, row: &str, v: &[f64]| {
        let (lo, hi) = v
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
                (a.min(x), b.max(x))
            });
        println!(
            "{name:16} {row:24} mean {:9.4} sd {:8.4} min {lo:9.4} max {hi:9.4} \
             mean - 4 sd {:9.4} mean + 4 sd {:9.4}",
            mean(v),
            sd(v),
            mean(v) - 4.0 * sd(v),
            mean(v) + 4.0 * sd(v)
        );
    };
    for (name, (small, large, times)) in &by_model {
        for (label, ms) in [("1000", small), ("2000", large)] {
            for (row, f) in [
                (
                    "flagged",
                    (|m: &Measured| m.flagged as f64) as fn(&Measured) -> f64,
                ),
                ("rescued", |m| m.rescued),
                ("gap_matched", |m| m.gap_matched),
                ("gap_reweighted", |m| m.gap_reweighted),
                ("spread_matched", |m| m.spread_matched),
                ("spread_reweighted", |m| m.spread_reweighted),
                ("bias_matched", |m| m.bias_matched),
                ("bias_reweighted", |m| m.bias_reweighted),
                ("sd_ratio", |m| m.sd_ratio),
                ("sd_ratio_flagged", |m| m.sd_ratio_flagged),
                ("sd_ratio_reweight", |m| m.sd_ratio_reweight),
            ] {
                let v: Vec<f64> = ms.iter().map(f).collect();
                print(name, &format!("{row}@{label}"), &v);
            }
        }
        for (j, row) in ["time_reweight", "time_refit_flagged", "time_matching"]
            .iter()
            .enumerate()
        {
            let v: Vec<f64> = times.iter().map(|t| t[j]).collect();
            print(name, row, &v);
        }
    }
}
