//! The one-year view of the Bayesian Bornhuetter–Ferguson
//! (`BayesBf::one_year`, `docs/design/reserving-bayes.md`, decision 17)
//! on GenIns with R ChainLadder's `ClarkCapeCod` premium, apriori 0.5,
//! reweighting every simulation (`ReweightOnly`, 10,000 simulations).
//!
//! * A flat prior (`cv = f64::INFINITY`) is the Bayesian chain ladder:
//!   each origin's and the total's sd of the claims development result is
//!   `BayesOdp::one_year`'s on the same triangle, as a ratio with target 1.
//! * A tight prior (`cv = 0.001`, `development_sd` 100, no process error)
//!   is England, Verrall and Wüthrich's strong-prior limit, where each
//!   `mu_w` is its prior `m_w` and each `gamma_d` is independently
//!   `Gamma(S_d / phi, M_d / phi)` (`S_d` the observed increments of
//!   column `d`, `M_d` the prior ultimates observed there). The year adds
//!   `A_d gamma_d` to column `d` (`A_d` the prior ultimates of the origins
//!   whose year cell is at `d`), which moves the posterior mean of
//!   `gamma_d` to `(S_d + A_d gamma_d) / (M_d + A_d)`, so origin `w`'s
//!   claims development result is linear in the `gamma_d` with
//!   coefficient `-m_w` at its year's column and `-m_w A_d / (M_d + A_d)`
//!   after it, and its variance is the sum of the squared coefficients
//!   times `phi S_d / M_d^2`. Each origin's and the total's sd is checked
//!   against that closed form, as a ratio with target 1.
//! * Between them, the total's sd falls as the prior tightens: the sd at
//!   `cv` 0.2 and 0.05 (with process error) over the flat prior's.
//! * The mean total claims development result is zero within Monte Carlo
//!   error under each prior.
//!
//! Every tolerance is at least the row's mean offset from its target plus
//! 4 times its seed-to-seed sd over seeds 0 to 39, from the ignored
//! `seed_sweep` (`SWEEP_SEEDS=40 cargo test --release -p
//! prospicio-validation --features slow-bayes --test
//! reserving_bayes_bf_one_year -- --ignored seed_sweep --nocapture`).

use std::time::Instant;

use prospicio_reserving::bayes::{
    BayesBf, BayesBfFit, BayesOdp, BayesOneYear, BayesOneYearFit, OneYearRefit,
};
use prospicio_reserving::{ProcessDistribution, Triangle};
use prospicio_validation::triangle_columns;

const SIMS: usize = 10_000;

/// `(row, target, tolerance)`; origins numbered from the oldest (0, with
/// no cell in the year, is not checked). The ratio rows' target is 1 (the
/// other model, the closed form), the shrink rows' the sweep's mean (any
/// value below 1 shows the narrowing; the target pins its size).
const ROWS: &[(&str, f64, f64)] = &[
    ("flat_sd_ratio[1]", 1.0, 0.13),
    ("flat_sd_ratio[2]", 1.0, 0.09),
    ("flat_sd_ratio[3]", 1.0, 0.10),
    ("flat_sd_ratio[4]", 1.0, 0.09),
    ("flat_sd_ratio[5]", 1.0, 0.10),
    ("flat_sd_ratio[6]", 1.0, 0.11),
    ("flat_sd_ratio[7]", 1.0, 0.10),
    ("flat_sd_ratio[8]", 1.0, 0.12),
    ("flat_sd_ratio[9]", 1.0, 0.15),
    ("flat_sd_ratio[total]", 1.0, 0.12),
    ("flat_mean_z[total]", 0.0, 5.0),
    ("tight_sd_ratio[1]", 1.0, 0.10),
    ("tight_sd_ratio[2]", 1.0, 0.07),
    ("tight_sd_ratio[3]", 1.0, 0.10),
    ("tight_sd_ratio[4]", 1.0, 0.08),
    ("tight_sd_ratio[5]", 1.0, 0.09),
    ("tight_sd_ratio[6]", 1.0, 0.11),
    ("tight_sd_ratio[7]", 1.0, 0.09),
    ("tight_sd_ratio[8]", 1.0, 0.10),
    ("tight_sd_ratio[9]", 1.0, 0.10),
    ("tight_sd_ratio[total]", 1.0, 0.11),
    ("tight_mean_z[total]", 0.0, 5.0),
    ("shrink[0.2]", 0.697, 0.08),
    ("shrink[0.05]", 0.502, 0.06),
];

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn sd(x: &[f64]) -> f64 {
    let m = mean(x);
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
}

/// Each origin's draws, then the total's.
fn columns(draws: &[f64], n: usize) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = (0..n)
        .map(|j| draws.chunks(n).map(|r| r[j]).collect())
        .collect();
    out.push(draws.chunks(n).map(|r| r.iter().sum()).collect());
    out
}

fn reweight() -> BayesOneYear {
    BayesOneYear {
        refit: OneYearRefit::ReweightOnly,
        ..BayesOneYear::default()
    }
}

/// The one-year view of `BayesBf` at `cv`; `tight` adds the strong-prior
/// limit's `development_sd` 100 and no process error.
fn bf(tri: &Triangle, cv: f64, tight: bool, seed: u64) -> BayesOneYearFit<BayesBfFit> {
    let model = BayesBf {
        apriori: 0.5,
        cv,
        n_sims: SIMS,
        seed,
        ..BayesBf::default()
    };
    let model = if tight {
        BayesBf {
            development_sd: 100.0,
            process: ProcessDistribution::None,
            ..model
        }
    } else {
        model
    };
    model
        .one_year(tri, "paid", "premium", &reweight())
        .expect("one-year view")
}

/// The strong-prior limit's sd of each origin's and the total's claims
/// development result (module docs).
fn closed_form(tri: &Triangle, fit: &BayesBfFit) -> Vec<f64> {
    let (latest, m, phi) = (
        &fit.chain_ladder.latest_position,
        &fit.prior_ultimate,
        fit.scale,
    );
    let (no, nd) = (latest.len(), latest[0] + 1);
    let (mut s, mut big_m) = (vec![0.0; nd], vec![0.0; nd]);
    for w in 0..no {
        let mut previous = 0.0;
        for d in 0..=latest[w] {
            let cum = tri.get(0, 0, w, d).unwrap();
            s[d] += cum - previous;
            big_m[d] += m[w];
            previous = cum;
        }
    }
    // A_d: the prior ultimates of the origins whose year cell is at d.
    let mut a = vec![0.0; nd];
    for w in 0..no {
        if latest[w] + 1 < nd {
            a[latest[w] + 1] += m[w];
        }
    }
    let var_gamma: Vec<f64> = (0..nd)
        .map(|d| phi * s[d] / (big_m[d] * big_m[d]))
        .collect();
    let coefficient = |w: usize, d: usize| {
        if d == latest[w] + 1 {
            m[w]
        } else {
            m[w] * a[d] / (big_m[d] + a[d])
        }
    };
    let mut out: Vec<f64> = (0..no)
        .map(|w| {
            (latest[w] + 1..nd)
                .map(|d| coefficient(w, d).powi(2) * var_gamma[d])
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let total: f64 = (0..nd)
        .map(|d| {
            let c: f64 = (0..no)
                .filter(|&w| d > latest[w])
                .map(|w| coefficient(w, d))
                .sum();
            c * c * var_gamma[d]
        })
        .sum();
    out.push(total.sqrt());
    out
}

fn label(j: usize, n: usize) -> String {
    if j == n {
        "total".into()
    } else {
        j.to_string()
    }
}

/// Every row at `seed`.
fn rows(seed: u64) -> Vec<(String, f64)> {
    let tri = triangle_columns("genins_premium", &["paid", "premium"]);
    let mut out = Vec::new();
    let flat = bf(&tri, f64::INFINITY, false, seed);
    let n = flat.cdr.n_components();
    let flat_cols = columns(flat.cdr.draw_matrix(), n);
    let odp = BayesOdp {
        n_sims: SIMS,
        seed: seed + 10_000,
        ..BayesOdp::default()
    }
    .one_year(&tri, "paid", &reweight())
    .expect("one-year view");
    let odp_cols = columns(odp.cdr.draw_matrix(), n);
    for j in 1..=n {
        out.push((
            format!("flat_sd_ratio[{}]", label(j, n)),
            sd(&flat_cols[j]) / sd(&odp_cols[j]),
        ));
    }
    let z = |c: &[f64]| mean(c) / (sd(c) / (SIMS as f64).sqrt());
    out.push(("flat_mean_z[total]".into(), z(&flat_cols[n])));

    let tight = bf(&tri, 1e-3, true, seed);
    let tight_cols = columns(tight.cdr.draw_matrix(), n);
    let limit = closed_form(&tri, &tight.fit);
    for j in 1..=n {
        out.push((
            format!("tight_sd_ratio[{}]", label(j, n)),
            sd(&tight_cols[j]) / limit[j],
        ));
    }
    out.push(("tight_mean_z[total]".into(), z(&tight_cols[n])));

    let flat_sd = sd(&flat_cols[n]);
    for (cv, name) in [(0.2, "shrink[0.2]"), (0.05, "shrink[0.05]")] {
        let v = bf(&tri, cv, false, seed);
        out.push((
            name.into(),
            sd(&columns(v.cdr.draw_matrix(), n)[n]) / flat_sd,
        ));
    }
    out
}

fn expected(row: &str) -> (f64, f64) {
    let found: Vec<_> = ROWS.iter().filter(|r| r.0 == row).collect();
    assert_eq!(found.len(), 1, "{row}");
    (found[0].1, found[0].2)
}

fn used(row: &str, value: f64) -> f64 {
    let (target, tol) = expected(row);
    (value - target).abs() / tol
}

#[test]
fn the_one_year_view_spans_the_chain_ladder_and_the_strong_prior_limit() {
    let t = Instant::now();
    let rows = rows(0);
    assert_eq!(rows.len(), ROWS.len());
    let mut failures = Vec::new();
    for (row, value) in &rows {
        let u = used(row, *value);
        println!("{row}: {value:.4} uses {u:.2} of its tolerance");
        if u > 1.0 {
            failures.push(format!("{row}: {value} uses {u:.2} of its tolerance"));
        }
    }
    println!("{:?}", t.elapsed());
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Seed-to-seed spread of every row (ignored; release build).
#[test]
#[ignore]
fn seed_sweep() {
    let seeds: u64 = std::env::var("SWEEP_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let mut by_row: Vec<(String, Vec<f64>)> = Vec::new();
    for seed in 0..seeds {
        for (row, value) in rows(seed) {
            match by_row.iter_mut().find(|(r, _)| *r == row) {
                Some((_, v)) => v.push(value),
                None => by_row.push((row, vec![value])),
            }
        }
    }
    for (row, values) in &by_row {
        let (target, tol) = expected(row);
        let worst = values.iter().map(|v| used(row, *v)).fold(0.0, f64::max);
        println!(
            "{row:24} n {:3} mean {:9.4} sd {:8.4} | target {target} tol {tol} (|offset| + 4 sd = {:.4}) worst {worst:.2}",
            values.len(),
            mean(values),
            sd(values),
            (mean(values) - target).abs() + 4.0 * sd(values)
        );
    }
}
