//! The tower's simulated paths give the same bits whatever the number of
//! threads (#970): each year, or each path of a term, is a row of its own,
//! filled in parallel from its own inputs.

use prospicio_aggregate::{EventSet, Layer, Tower, simulate_events};
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

fn event_bits(events: &EventSet) -> Vec<Vec<u64>> {
    (0..events.n_sims())
        .map(|i| events.events(i).iter().map(|x| x.to_bits()).collect())
        .collect()
}

fn events(seed: u64) -> EventSet {
    let events = simulate_events(
        &Poisson::new(4.0).unwrap(),
        &Lognormal::new(12.0, 1.5).unwrap(),
        N_SIMS,
        seed,
    )
    .unwrap();
    let n: usize = (0..N_SIMS).map(|i| events.events(i).len()).sum();
    let perils = ["flood", "quake", "wind"];
    let labels = (0..n).map(|k| perils[k % 3].to_string()).collect();
    // Times rising through each year.
    let times = (0..N_SIMS)
        .flat_map(|i| {
            let m = events.events(i).len();
            (0..m).map(move |j| (j as f64 + 0.5) / m as f64)
        })
        .collect();
    events
        .with_labels(vec![("peril".into(), labels)])
        .unwrap()
        .with_times(times)
        .unwrap()
}

/// A quota share inuring to two excess layers, one with reinstatements pro
/// rata as to time and one scoped to wind, then a stop loss.
fn tower() -> Tower {
    Tower::inuring(vec![
        vec![
            Layer::quota_share("QS", 0.3)
                .unwrap()
                .premium_rate(1.0, 5e6)
                .unwrap()
                .profit_commission(0.2, 0.1)
                .unwrap(),
        ],
        vec![
            Layer::xol("1m xs 1m", 1e6, 1e6)
                .unwrap()
                .paid_reinstatements(5e4, vec![1.0, 0.5])
                .unwrap()
                .pro_rata_as_to_time()
                .unwrap(),
            Layer::xol("wind 3m xs 2m", 3e6, 2e6)
                .unwrap()
                .aggregate_limit(6e6)
                .unwrap()
                .scope("peril", &["wind"])
                .unwrap(),
        ],
        vec![Layer::stop_loss("SL", 5e6, 4e6).unwrap()],
    ])
    .unwrap()
}

#[test]
fn the_tower_gives_the_same_bits_on_any_number_of_threads() {
    let tower = tower();
    let labelled = events(11);
    let plain = EventSet::from_years(
        (0..N_SIMS).map(|i| labelled.events(i).to_vec()).collect(),
        11,
    )
    .unwrap();
    let years: Vec<EventSet> = (0..3).map(|t| events(20 + t)).collect();
    let annual: Vec<f64> = (0..N_SIMS).map(|i| (1 + i % 50) as f64 * 2e5).collect();
    let flat = Tower::new(vec![
        Layer::stop_loss("SL", 5e6, 4e6).unwrap(),
        Layer::xol("1m xs 1m", 1e6, 1e6).unwrap(),
    ])
    .unwrap();
    let run = || {
        (
            bits(&tower.apply(&labelled).unwrap()),
            bits(&flat.apply(&plain).unwrap()),
            event_bits(&tower.ceded_events(&labelled, &["1m xs 1m", "SL"]).unwrap()),
            bits(&tower.apply_term(&years).unwrap()),
            bits(&flat.apply_annual(&annual).unwrap()),
        )
    };
    let one = in_pool(1, run);
    for threads in [2, 3, 8] {
        assert!(in_pool(threads, run) == one, "{threads} threads differ from one");
    }
}
