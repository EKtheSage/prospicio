//! Timings of the reinsurance tower on simulated years (#970).
//!
//! `cargo run --release -p prospicio-aggregate --example tower_bench` prints, for each
//! scenario, the median of several runs and a checksum of the result's bits.
//! The checksum shows whether a change moved any number: equal checksums
//! before and after a change mean the results are bit for bit the same.
//! Set `RAYON_NUM_THREADS=1` to time a single thread. Arguments name the
//! scenarios to run (all by default); `--sims N` scales the year count.
//!
//! No benchmark crate: plain `std::time::Instant`, so the bench adds no
//! dependency. It is an example rather than a `[[bench]]` target: the R
//! package's `bootstrap.R` vendors the crates without `benches/`, and a
//! `[[bench]]` table naming a missing file stops Cargo reading the
//! manifest, while an example needs no manifest entry.

use std::hint::black_box;
use std::time::{Duration, Instant};

use prospicio_aggregate::{EventSet, Layer, Tower, simulate_events};
use prospicio_prob::{Lognormal, Poisson, PredictiveDistribution};

const RUNS: usize = 5;

/// FNV-1a over the bits of each value, in order.
fn checksum(values: impl IntoIterator<Item = f64>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in values {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

fn pd_checksum(pd: &PredictiveDistribution) -> u64 {
    checksum(pd.draw_matrix().iter().copied())
}

fn events_checksum(e: &EventSet) -> u64 {
    checksum((0..e.n_sims()).flat_map(|i| e.events(i).iter().copied()))
}

fn time<T>(mut f: impl FnMut() -> T, sum: impl Fn(&T) -> u64) -> (Duration, u64) {
    let mut times = Vec::with_capacity(RUNS);
    let mut check = 0;
    for _ in 0..RUNS {
        let start = Instant::now();
        let out = black_box(f());
        times.push(start.elapsed());
        check = sum(&out);
    }
    times.sort();
    (times[RUNS / 2], check)
}

fn events(n_sims: usize, seed: u64) -> EventSet {
    simulate_events(
        &Poisson::new(5.0).unwrap(),
        &Lognormal::new(12.0, 1.5).unwrap(),
        n_sims,
        seed,
    )
    .unwrap()
}

fn labelled(events: EventSet) -> EventSet {
    let n: usize = (0..events.n_sims()).map(|i| events.events(i).len()).sum();
    let perils = ["wind", "quake", "flood"];
    let labels = (0..n).map(|k| perils[k % 3].to_string()).collect();
    events.with_labels(vec![("peril".into(), labels)]).unwrap()
}

fn single() -> Tower {
    Tower::new(vec![
        Layer::xol("1m xs 1m", 1e6, 1e6)
            .unwrap()
            .reinstatements(2)
            .unwrap()
            .premium_rate(0.05, 1e8)
            .unwrap(),
        Layer::xol("3m xs 2m", 3e6, 2e6)
            .unwrap()
            .aggregate_limit(6e6)
            .unwrap(),
    ])
    .unwrap()
}

fn inuring() -> Tower {
    Tower::inuring(vec![
        vec![Layer::quota_share("QS", 0.3).unwrap()],
        vec![
            Layer::xol("1m xs 1m", 1e6, 1e6)
                .unwrap()
                .reinstatements(1)
                .unwrap()
                .premium_rate(0.05, 1e8)
                .unwrap(),
            Layer::xol("3m xs 2m", 3e6, 2e6).unwrap(),
        ],
        vec![Layer::stop_loss("SL", 5e6, 4e6).unwrap()],
    ])
    .unwrap()
}

fn term_tower() -> Tower {
    Tower::new(vec![
        Layer::quota_share("QS", 0.5)
            .unwrap()
            .premium_rate(1.0, 2e6)
            .unwrap()
            .profit_commission(0.25, 0.1)
            .unwrap()
            .profit_commission_carry_forward(None)
            .unwrap(),
        Layer::xol("1m xs 1m", 1e6, 1e6).unwrap(),
    ])
    .unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut n_sims = 200_000usize;
    let mut chosen = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--sims" => n_sims = it.next().expect("--sims N").parse().expect("a count"),
            "--bench" => {}
            s if s.starts_with("--") => {}
            s => chosen.push(s.to_string()),
        }
    }
    let run = |name: &str| chosen.is_empty() || chosen.iter().any(|c| name.contains(c.as_str()));

    println!(
        "tower bench: {n_sims} years, {} threads, median of {RUNS} runs",
        rayon::current_num_threads()
    );
    let plain = events(n_sims, 7);
    let tagged = labelled(plain.clone());
    let totals: Vec<f64> = (0..n_sims * 5)
        .map(|i| 1e6 * (1.0 + (i % 97) as f64 / 10.0))
        .collect();
    let term_years: Vec<EventSet> = (0..5).map(|t| events(n_sims / 5, 100 + t)).collect();

    let report = |name: &str, (d, c): (Duration, u64)| {
        println!(
            "{name:<22} {:>10.1} ms  checksum {c:016x}",
            d.as_secs_f64() * 1e3
        );
    };
    if run("apply_single") {
        let t = single();
        report(
            "apply_single",
            time(|| t.apply(&plain).unwrap(), pd_checksum),
        );
    }
    if run("apply_inuring") {
        let t = inuring();
        report(
            "apply_inuring",
            time(|| t.apply(&plain).unwrap(), pd_checksum),
        );
    }
    if run("apply_labelled") {
        let t = inuring();
        report(
            "apply_labelled",
            time(|| t.apply(&tagged).unwrap(), pd_checksum),
        );
    }
    if run("ceded_events") {
        let t = inuring();
        report(
            "ceded_events",
            time(
                || t.ceded_events(&tagged, &["1m xs 1m", "SL"]).unwrap(),
                events_checksum,
            ),
        );
    }
    if run("apply_annual") {
        let t = single();
        report(
            "apply_annual",
            time(|| t.apply_annual(&totals).unwrap(), pd_checksum),
        );
    }
    if run("apply_term") {
        let t = term_tower();
        report(
            "apply_term",
            time(|| t.apply_term(&term_years).unwrap(), pd_checksum),
        );
    }
}
