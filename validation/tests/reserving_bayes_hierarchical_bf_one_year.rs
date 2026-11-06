//! Measurements of `HierarchicalBf::one_year` on the three-program Stan
//! reference panel (`reserving-bayes.md`, decision 30, "As built (PR 17)"):
//! the mean of the claims development result under refits only, the Pareto k
//! as a function of the panel and of the draw count, and the reweighting
//! against the refits where k is low. All tests are ignored (they refit);
//! run them in release:
//!
//! `cargo test --release -p prospicio-validation --features slow-bayes --test
//! reserving_bayes_hierarchical_bf_one_year -- --ignored --nocapture <name>`
//!
//! `HB_SIMS`, `HB_PANEL`, `HB_SEEDS` (a comma-separated list) and `HB_DRAWS` (draws and
//! warm-up per chain, default 200) set the size.

use prospicio_bayes::nuts::Sampler;
use prospicio_prob::PredictiveDistribution;
use prospicio_reserving::Triangle;
use prospicio_reserving::bayes::{
    BayesBfNormalised, BayesOneYear, BayesOneYearFit, HierarchicalBf, HierarchicalBfFit,
    OneYearRefit,
};
use prospicio_validation::{
    stan_triangle, stan_triangle_ages, stan_triangle_cut, stan_triangle_of,
};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn seeds() -> Vec<u64> {
    std::env::var("HB_SEEDS")
        .unwrap_or_else(|_| "0,1".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect()
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

/// `(mean, standard error, z)` of the draws.
fn z(x: &[f64]) -> (f64, f64, f64) {
    let se = sd(x) / (x.len() as f64).sqrt();
    (mean(x), se, mean(x) / se)
}

/// The conformance suite's `HierarchicalBf` with `draws` kept draws and as
/// many warm-up draws on each of 2 chains.
fn run(
    tri: &Triangle,
    draws: usize,
    sims: usize,
    seed: u64,
    refit: OneYearRefit,
) -> BayesOneYearFit<HierarchicalBfFit> {
    HierarchicalBf {
        apriori: 0.7,
        level_cv: Some(0.5),
        tau_scale: 0.5,
        sampler: Sampler {
            chains: 2,
            tune: draws,
            draws,
            ..HierarchicalBf::default().sampler
        },
        n_sims: sims,
        seed,
        ..HierarchicalBf::default()
    }
    .one_year(
        tri,
        "paid",
        "premium",
        &BayesOneYear {
            n_sims: Some(sims),
            refit: Some(refit),
            ..BayesOneYear::default()
        },
    )
    .unwrap()
}

fn quantiles(k: &[f64]) -> String {
    let mut v: Vec<f64> = k.iter().copied().filter(|x| !x.is_nan()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    format!(
        "k min {:.2} median {:.2} max {:.2}; share above 0.7: {:.2} ({} NaN)",
        q(0.0),
        q(0.5),
        q(1.0),
        k.iter().filter(|&&x| x.is_nan() || x > 0.7).count() as f64 / k.len() as f64,
        k.len() - v.len()
    )
}

/// Item 1: the mean of the claims development result under refits only is
/// zero by the tower property, up to Monte Carlo error.
#[test]
#[ignore]
fn refit_only_mean_of_the_cdr() {
    let tri = stan_triangle();
    let sims = env_usize("HB_SIMS", 200);
    let draws = env_usize("HB_DRAWS", 200);
    for seed in seeds() {
        let fit = run(&tri, draws, sims, seed, OneYearRefit::RefitAll);
        let by_program = fit.cdr.aggregate(&["program"]).unwrap();
        let n = by_program.n_components();
        let total = totals(&fit.cdr);
        let (m, se, zz) = z(&total);
        println!(
            "seed {seed}, {sims} sims, refit only: total mean {m:.1} se {se:.1} z {zz:.2} \
             (sd of the total {:.1}); {} refits with warnings",
            sd(&total),
            fit.refit_warnings
        );
        for c in 0..n {
            let col: Vec<f64> = by_program.draw_matrix().chunks(n).map(|r| r[c]).collect();
            let (m, se, zz) = z(&col);
            println!("  program {c}: mean {m:.1} se {se:.1} z {zz:.2}");
        }
        let only = run(&tri, draws, sims, seed, OneYearRefit::ReweightOnly);
        let (m, se, zz) = z(&totals(&only.cdr));
        println!("  same years reweighted only: mean {m:.1} se {se:.1} z {zz:.2}");
    }
}

/// Item 4: the Pareto k of the reweighting as a function of the panel and of
/// the draw count, and the same for `BayesBfNormalised` on each triangle.
#[test]
#[ignore]
fn pareto_k_by_panel_and_draws() {
    let sims = env_usize("HB_SIMS", 60);
    let seed = 0;
    let panels: [(&str, Vec<&str>); 9] = [
        ("all three", vec!["evw", "genins", "raa"]),
        ("evw alone", vec!["evw"]),
        ("genins alone", vec!["genins"]),
        ("raa alone", vec!["raa"]),
        ("evw and raa", vec!["evw", "raa"]),
        ("genins and raa", vec!["genins", "raa"]),
        ("evw and genins", vec!["evw", "genins"]),
        ("genins twice", vec!["genins", "genins2"]),
        ("all and a twin", vec!["evw", "genins", "raa", "genins2"]),
    ];
    for draws in [200, 2000] {
        for (name, keep) in &panels {
            let tri = stan_triangle_of(keep);
            let fit = run(&tri, draws, sims, seed, OneYearRefit::ReweightOnly);
            println!(
                "HierarchicalBf {name:14} {} draws per chain: {}",
                draws,
                quantiles(&fit.pareto_k)
            );
        }
        for keep in ["evw", "genins", "raa"] {
            let tri = stan_triangle_of(&[keep]);
            let fit = BayesBfNormalised {
                apriori: 0.7,
                cv: 0.5,
                sampler: Sampler {
                    chains: 2,
                    tune: draws,
                    draws,
                    ..BayesBfNormalised::default().sampler
                },
                n_sims: sims,
                seed,
                ..BayesBfNormalised::default()
            }
            .one_year(
                &tri,
                "paid",
                "premium",
                &BayesOneYear {
                    n_sims: Some(sims),
                    refit: Some(OneYearRefit::ReweightOnly),
                    ..BayesOneYear::default()
                },
            )
            .unwrap();
            println!(
                "BayesBfNormalised {keep:11} {} draws per chain: {}",
                draws,
                quantiles(&fit.pareto_k)
            );
        }
    }
}

/// Items 3 and 5: the reweighted totals against the refits, split by k, on
/// the full panel and each program alone, at the draw count of `HB_DRAWS`.
#[test]
#[ignore]
fn reweighting_against_refits() {
    let sims = env_usize("HB_SIMS", 20);
    let draws = env_usize("HB_DRAWS", 200);
    let panels: [(&str, Vec<&str>, usize); 6] = [
        ("all three", vec!["evw", "genins", "raa"], 10),
        ("evw alone", vec!["evw"], 10),
        ("genins alone", vec!["genins"], 10),
        ("raa alone", vec!["raa"], 10),
        ("all, 6 ages", vec!["evw", "genins", "raa"], 6),
        ("evw+genins 6", vec!["evw", "genins"], 6),
    ];
    let only = std::env::var("HB_PANEL").ok();
    for (name, keep, cut) in &panels {
        if only.as_deref().is_some_and(|o| o != *name) {
            continue;
        }
        let tri = stan_triangle_cut(keep, *cut);
        for seed in seeds() {
            let a = run(&tri, draws, sims, seed, OneYearRefit::ReweightOnly);
            let b = run(&tri, draws, sims, seed, OneYearRefit::RefitAll);
            // The scale: the sd of the total over many reweighted years.
            let wide = run(&tri, draws, 4000, seed, OneYearRefit::ReweightOnly);
            let s = sd(&totals(&wide.cdr));
            let (ta, tb) = (totals(&a.cdr), totals(&b.cdr));
            let rms = |flagged: bool| {
                let v: Vec<f64> = (0..sims)
                    .filter(|&i| (a.pareto_k[i].is_nan() || a.pareto_k[i] > 0.7) == flagged)
                    .map(|i| ((ta[i] - tb[i]) / s).powi(2))
                    .collect();
                (v.len(), if v.is_empty() { 0.0 } else { mean(&v).sqrt() })
            };
            let bias: Vec<f64> = (0..sims).map(|i| (ta[i] - tb[i]) / s).collect();
            let (nu, gu) = rms(false);
            let (nf, gf) = rms(true);
            println!(
                "{name:13} seed {seed} {draws} draws: {}; rms gap {gu:.3} sd ({nu} sims k<=0.7), \
                 {gf:.3} sd ({nf} sims k>0.7); mean gap {:.3} sd, refit mean z {:.2}",
                quantiles(&a.pareto_k),
                mean(&bias),
                z(&tb).2
            );
        }
    }
}

/// The scale `phi_c` of each program in each panel (the pooled scale per unit
/// of premium makes a program's scale depend on the others) and the fit's
/// diagnostics at 200 and 2,000 draws per chain.
#[test]
#[ignore]
fn scales_and_diagnostics_by_panel() {
    for draws in [200, 2000] {
        for keep in [
            vec!["evw"],
            vec!["genins"],
            vec!["raa"],
            vec!["evw", "raa"],
            vec!["evw", "genins"],
            vec!["genins", "raa"],
            vec!["evw", "genins", "raa"],
        ] {
            let tri = stan_triangle_of(&keep);
            let fit = HierarchicalBf {
                apriori: 0.7,
                level_cv: Some(0.5),
                tau_scale: 0.5,
                sampler: Sampler {
                    chains: 2,
                    tune: draws,
                    draws,
                    ..HierarchicalBf::default().sampler
                },
                n_sims: 1000,
                ..HierarchicalBf::default()
            }
            .fit(&tri, "paid", "premium")
            .unwrap();
            let scales: Vec<String> = fit
                .programs
                .iter()
                .map(|p| format!("{:.1}", p.scale))
                .collect();
            let d = &fit.diagnostics;
            println!(
                "{draws} draws {keep:?}: phi_c {scales:?}; max rhat {:.3}, min ess {:.0},                  divergences {}, warnings {}",
                d.max_rhat,
                d.min_ess_bulk,
                d.divergences,
                d.warnings.len()
            );
        }
    }
}

/// Item 4, the mechanism: a program that stops at fewer ages than the grid
/// (EVW at 6 of 10) has its next diagonal at ages it has never observed. Cut
/// every program to 6 ages and the grid has no such age.
#[test]
#[ignore]
fn pareto_k_with_every_program_cut_to_six_ages() {
    let sims = env_usize("HB_SIMS", 60);
    for draws in [200, 2000] {
        for keep in [
            vec!["evw", "raa"],
            vec!["evw", "genins"],
            vec!["evw", "genins", "raa"],
        ] {
            for cut in [6, 10] {
                let tri = stan_triangle_cut(&keep, cut);
                let fit = run(&tri, draws, sims, 0, OneYearRefit::ReweightOnly);
                println!(
                    "{keep:?} cut to {cut} ages, {draws} draws per chain: {}",
                    quantiles(&fit.pareto_k)
                );
            }
        }
    }
}

/// Item 4, the control for the mechanism: the same programs, changing only
/// whether one of them stops short of the grid. Two copies of a program at
/// full length (`genins` and `genins2`), one cut to 6 ages beside the other
/// at full length, and both cut to 6 ages. If the cause is a program's next
/// diagonal at ages it has never observed, k is high for the unequal panels
/// and at the level of the equal ones otherwise. The sampler and seed are
/// those of `pareto_k_by_panel_and_draws`.
#[test]
#[ignore]
fn pareto_k_when_one_program_stops_short() {
    let sims = env_usize("HB_SIMS", 60);
    let panels: [(&str, Vec<(&str, usize)>); 14] = [
        (
            "genins 10 + genins 10",
            vec![("genins", 10), ("genins2", 10)],
        ),
        ("genins 10 + genins 6", vec![("genins", 10), ("genins2", 6)]),
        ("genins 6 + genins 10", vec![("genins", 6), ("genins2", 10)]),
        ("genins 6 + genins 6", vec![("genins", 6), ("genins2", 6)]),
        ("raa 10 + raa 10", vec![("raa", 10), ("raa2", 10)]),
        ("raa 10 + raa 6", vec![("raa", 10), ("raa2", 6)]),
        ("raa 6 + raa 6", vec![("raa", 6), ("raa2", 6)]),
        ("genins 10 + raa 10", vec![("genins", 10), ("raa", 10)]),
        ("genins 10 + raa 6", vec![("genins", 10), ("raa", 6)]),
        ("genins 6 + raa 10", vec![("genins", 6), ("raa", 10)]),
        ("genins 6 + raa 6", vec![("genins", 6), ("raa", 6)]),
        ("evw + genins 10", vec![("evw", 6), ("genins", 10)]),
        ("evw + genins 6", vec![("evw", 6), ("genins", 6)]),
        ("evw + genins 7", vec![("evw", 6), ("genins", 7)]),
    ];
    for draws in [200, 2000] {
        for (name, spec) in &panels {
            let tri = stan_triangle_ages(spec);
            let fit = run(&tri, draws, sims, 0, OneYearRefit::ReweightOnly);
            println!(
                "{name:22} {draws} draws per chain: {}",
                quantiles(&fit.pareto_k)
            );
        }
    }
}
