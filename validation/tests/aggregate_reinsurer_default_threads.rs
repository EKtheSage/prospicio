//! A panel's default draws give the same bits whatever the number of
//! threads (#980): the years' ranks are computed once, and each year draws
//! from its own stream, in parallel.

use prospicio_aggregate::{Layer, Panel, Reinsurer, Tower, simulate_events};
use prospicio_prob::{Lognormal, Poisson, PredictiveDistribution};

/// Enough years that every thread count below splits them into many tasks.
const N_SIMS: usize = 6_000;

fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(f)
}

fn bits(pd: &PredictiveDistribution) -> Vec<u64> {
    pd.draw_matrix().iter().map(|x| x.to_bits()).collect()
}

fn panel() -> Panel {
    Panel::new(vec![
        Reinsurer::new("Re A", 0.05, 0.4)
            .unwrap()
            .collateral(5e5)
            .unwrap()
            .correlation(0.4)
            .unwrap(),
        Reinsurer::new("Re B", 0.2, 0.3).unwrap(),
        Reinsurer::new("Re C", 0.1, 0.5)
            .unwrap()
            .correlation(0.7)
            .unwrap(),
    ])
    .unwrap()
    .line("Re A", "1m xs 1m", 0.5)
    .unwrap()
    .line("Re B", "1m xs 1m", 0.5)
    .unwrap()
    .line("Re A", "3m xs 2m", 0.3)
    .unwrap()
    .line("Re C", "3m xs 2m", 0.7)
    .unwrap()
}

#[test]
fn the_default_draws_give_the_same_bits_on_any_number_of_threads() {
    let events = simulate_events(
        &Poisson::new(4.0).unwrap(),
        &Lognormal::new(12.0, 1.5).unwrap(),
        N_SIMS,
        11,
    )
    .unwrap();
    let n: usize = (0..N_SIMS).map(|i| events.events(i).len()).sum();
    let perils = ["flood", "quake", "wind"];
    let labels = (0..n).map(|k| perils[k % 3].to_string()).collect();
    let labelled = events
        .clone()
        .with_labels(vec![("peril".into(), labels)])
        .unwrap();
    let tower = Tower::new(vec![
        Layer::xol("1m xs 1m", 1e6, 1e6).unwrap(),
        Layer::xol("3m xs 2m", 3e6, 2e6).unwrap(),
    ])
    .unwrap();
    let flat = tower.apply(&events).unwrap();
    let by_label = tower.apply(&labelled).unwrap();
    let panel = panel();
    let run = || {
        (
            bits(&panel.apply(&flat, 5).unwrap()),
            bits(&panel.apply(&by_label, 5).unwrap()),
            bits(&panel.by_reinsurer(&flat, 5).unwrap()),
        )
    };
    let one = in_pool(1, run);
    for threads in [2, 3, 8] {
        assert!(in_pool(threads, run) == one, "{threads} threads differ from one");
    }
}
