//! The benchmark of #972: the reserving hot paths it names, timed in a
//! release build, each with a hash of its seeded output so that a change
//! meant to keep the numbers can be shown to keep them bit for bit.
//!
//! * The ODP bootstrap (lifetime and one-year views) and Mack's bootstrap
//!   with a Gamma process draw one Gamma per future cell per simulation
//!   (`process_draw`, `MackProcess::draw`, the one-year bootstrap's next
//!   cells).
//! * The Bayesian ODP draws its reserves through the same `process_draw`.
//! * `HierarchicalBf` differentiates its whole log density with the chunked
//!   dual (`dual::gradient`) at every leapfrog step, and its one-year view
//!   refits inside a parallel loop over simulations.
//!
//! Every test is ignored; run them in release, alone, with
//!
//! `cargo test --release -p prospicio-validation --test validation
//! reserving_perf_benchmark:: -- --ignored --nocapture --test-threads 1`
//!
//! Each prints `BENCH <name> median <s> (min <s>, runs <n>) hash 0x<16 hex>`:
//! the median and fastest wall clock of `PERF_RUNS` runs (default 5) and an
//! FNV-1a hash of the output's bits, the same in every run.

use prospicio_prob::PredictiveDistribution;
use prospicio_reserving::bayes::{BayesOdp, BayesOneYear, HierarchicalBf, OneYearRefit};
use prospicio_reserving::{
    ChainLadder, MackBootstrap, MackProcess, OdpBootstrap, OneYearMethod, ProcessDistribution,
};
use prospicio_validation::{stan_triangle, triangle};
use std::time::Instant;

fn runs() -> usize {
    std::env::var("PERF_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5)
}

/// FNV-1a over the bits of `values`.
fn fnv(values: impl IntoIterator<Item = f64>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in values {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

fn draws(d: &PredictiveDistribution) -> Vec<f64> {
    d.draw_matrix().to_vec()
}

/// Times `f` `runs()` times and prints its line; every run must give the
/// same hash.
fn bench(name: &str, f: impl Fn() -> u64) {
    if cfg!(debug_assertions) {
        println!("warning: a debug build; the times mean nothing");
    }
    let mut times = Vec::new();
    let mut hash = None;
    for _ in 0..runs() {
        let start = Instant::now();
        let h = f();
        times.push(start.elapsed().as_secs_f64());
        assert!(hash.is_none_or(|x| x == h), "{name}: the hash moved between runs");
        hash = Some(h);
    }
    times.sort_by(f64::total_cmp);
    println!(
        "BENCH {name} median {:.3} (min {:.3}, runs {}) hash {:#018x}",
        times[times.len() / 2],
        times[0],
        times.len(),
        hash.expect("at least one run"),
    );
}

#[test]
#[ignore = "a release benchmark; see the module doc"]
fn odp_bootstrap_lifetime() {
    let tri = triangle("abc");
    bench("odp-bootstrap-lifetime abc 100k gamma", || {
        let fit = OdpBootstrap {
            n_sims: 100_000,
            seed: 972,
            process: ProcessDistribution::Gamma,
            ..Default::default()
        }
        .fit(&tri, "values")
        .unwrap();
        fnv(draws(&fit.reserves))
    });
}

#[test]
#[ignore = "a release benchmark; see the module doc"]
fn odp_bootstrap_one_year() {
    let tri = triangle("abc");
    bench("odp-bootstrap-one-year abc 20k gamma", || {
        let fit = OdpBootstrap {
            n_sims: 20_000,
            seed: 972,
            process: ProcessDistribution::Gamma,
            ..Default::default()
        }
        .one_year(
            &tri,
            "values",
            &OneYearMethod::ChainLadder(ChainLadder::default()),
        )
        .unwrap();
        fnv(draws(&fit.cdr))
    });
}

#[test]
#[ignore = "a release benchmark; see the module doc"]
fn mack_bootstrap_lifetime() {
    let tri = triangle("abc");
    bench("mack-bootstrap-lifetime abc 100k gamma", || {
        let fit = MackBootstrap {
            n_sims: 100_000,
            seed: 972,
            process: MackProcess::Gamma,
            ..Default::default()
        }
        .fit(&tri, "values")
        .unwrap();
        fnv(draws(&fit.reserves))
    });
}

#[test]
#[ignore = "a release benchmark; see the module doc"]
fn bayes_odp_fit() {
    let tri = triangle("raa");
    bench("bayes-odp-fit raa default", || {
        let fit = BayesOdp {
            seed: 972,
            ..BayesOdp::default()
        }
        .fit(&tri, "values")
        .unwrap();
        fnv(draws(&fit.reserves))
    });
}

#[test]
#[ignore = "a release benchmark; see the module doc"]
fn hierarchical_bf_fit() {
    let tri = stan_triangle();
    bench("hierarchical-bf-fit stan panel default", || {
        let fit = HierarchicalBf {
            apriori: 0.7,
            level_cv: Some(0.5),
            tau_scale: 0.5,
            seed: 972,
            ..HierarchicalBf::default()
        }
        .fit(&tri, "paid", "premium")
        .unwrap();
        fnv(fit
            .programs
            .iter()
            .flat_map(|p| p.expected_ultimate.iter().copied()))
    });
}

#[test]
#[ignore = "a release benchmark; see the module doc"]
fn hierarchical_bf_one_year_refits() {
    let tri = stan_triangle();
    bench("hierarchical-bf-one-year stan panel 64 sims refit-all", || {
        let fit = HierarchicalBf {
            apriori: 0.7,
            level_cv: Some(0.5),
            tau_scale: 0.5,
            n_sims: 64,
            seed: 972,
            ..HierarchicalBf::default()
        }
        .one_year(
            &tri,
            "paid",
            "premium",
            &BayesOneYear {
                n_sims: Some(64),
                refit: Some(OneYearRefit::RefitAll),
                ..BayesOneYear::default()
            },
        )
        .unwrap();
        fnv(draws(&fit.cdr))
    });
}
