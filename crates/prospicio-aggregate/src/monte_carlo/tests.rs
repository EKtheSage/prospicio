fn part(dims: Vec<String>, keys: Vec<Vec<&str>>, draws: Vec<f64>) -> PredictiveDistribution {
    let keys = keys
        .into_iter()
        .map(|k| k.into_iter().map(prospicio_prob::KeyValue::from).collect())
        .collect();
    PredictiveDistribution::from_draws(dims, keys, draws, Provenance::new("part")).unwrap()
}

#[test]
fn contract_years_from_the_simulated_start_are_the_set() {
    let e = EventSet::from_years(vec![vec![1.0, 2.0], vec![3.0]], 4)
        .unwrap()
        .with_times(vec![0.1, 0.9, 0.5])
        .unwrap();
    assert_eq!(e.contract_years(0.0, PartialYears::Drop).unwrap(), e);
    assert!(e.contract_years(1.0, PartialYears::Drop).is_err());
    assert!(e.contract_years(-0.1, PartialYears::Drop).is_err());
    let untimed = EventSet::from_years(vec![vec![1.0]], 0).unwrap();
    assert!(untimed.contract_years(0.5, PartialYears::Keep).is_err());
}

#[test]
fn a_set_shifted_by_the_start_regroups_to_itself() {
    // Each loss of year i at time t moved to i + t + start on a
    // calendar timeline, one year longer, then cut at start.
    let mut rng = prospicio_core::StreamRng::new(11, 0);
    let n = 30;
    let mut years = Vec::new();
    let mut times = Vec::new();
    for _ in 0..n {
        let k = (rng.next_open01() * 4.0) as usize;
        let mut t: Vec<f64> = (0..k).map(|_| rng.next_open01()).collect();
        t.sort_by(f64::total_cmp);
        years.push(
            (0..k)
                .map(|_| 1.0 + rng.next_open01() * 9.0)
                .collect::<Vec<_>>(),
        );
        times.extend(t);
    }
    let labels: Vec<String> = (0..times.len()).map(|k| format!("c{}", k % 3)).collect();
    let original = EventSet::from_years(years.clone(), 2)
        .unwrap()
        .with_times(times.clone())
        .unwrap()
        .with_labels(vec![("class".into(), labels.clone())])
        .unwrap();
    for &start in &[0.25, 0.5, 0.875] {
        let mut cal: Vec<Vec<(f64, f64, String)>> = vec![Vec::new(); n + 1];
        let mut k = 0;
        for (i, y) in years.iter().enumerate() {
            for &x in y {
                let at = i as f64 + times[k] + start;
                let year = at.floor() as usize;
                cal[year].push((at - year as f64, x, labels[k].clone()));
                k += 1;
            }
        }
        for y in &mut cal {
            y.sort_by(|a, b| a.0.total_cmp(&b.0));
        }
        let calendar = EventSet::from_years(
            cal.iter()
                .map(|y| y.iter().map(|e| e.1).collect())
                .collect(),
            2,
        )
        .unwrap()
        .with_times(cal.iter().flatten().map(|e| e.0).collect())
        .unwrap()
        .with_labels(vec![(
            "class".into(),
            cal.iter().flatten().map(|e| e.2.clone()).collect(),
        )])
        .unwrap();
        let back = calendar.contract_years(start, PartialYears::Drop).unwrap();
        assert_eq!(back.n_sims(), n);
        for i in 0..n {
            assert_eq!(
                back.events(i),
                original.events(i),
                "start {start}, year {i}"
            );
            let (a, b) = (back.times(i).unwrap(), original.times(i).unwrap());
            assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12));
            assert_eq!(back.labels(i), original.labels(i));
        }
        // Keeping the partial years: the first is empty here, the last
        // holds what ran past the original's end.
        let kept = calendar.contract_years(start, PartialYears::Keep).unwrap();
        assert_eq!(kept.n_sims(), n + 2);
        assert!(kept.events(0).is_empty());
        let total = |e: &EventSet| {
            (0..e.n_sims())
                .map(|i| e.events(i).iter().sum::<f64>())
                .sum::<f64>()
        };
        assert!((total(&kept) - total(&calendar)).abs() < 1e-9);
    }
}

#[test]
fn the_aggregate_part_follows_its_years() {
    let p = part(
        vec!["peril".into()],
        vec![vec!["fire"], vec!["wind"]],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    let events = EventSet::from_years(vec![vec![7.0], vec![], vec![8.0, 9.0]], 0)
        .unwrap()
        .with_times(vec![0.5, 0.1, 0.2])
        .unwrap()
        .with_aggregate_part(&p)
        .unwrap();
    // The events get "" in the part's dimension.
    assert_eq!(events.label_combinations(), [[""], ["fire"], ["wind"]]);
    assert_eq!(events.aggregate_part_indices(), Some(&[1, 2][..]));
    let moved = events.permute(&[2, 0, 0]).unwrap();
    assert_eq!(moved.aggregate_part(0), Some(&[5.0, 6.0][..]));
    assert_eq!(moved.aggregate_part(2), Some(&[1.0, 2.0][..]));
    assert_eq!(moved.events(0), &[8.0, 9.0]);
    let years = events.term_years(3).unwrap();
    assert_eq!(years[1].aggregate_part(0), Some(&[3.0, 4.0][..]));
    let totals: Vec<f64> =
        prospicio_prob::Empirical::draws(events.totals().unwrap().total()).to_vec();
    assert_eq!(totals, [10.0, 7.0, 28.0]);
    let maxima: Vec<f64> =
        prospicio_prob::Empirical::draws(events.maxima().unwrap().total()).to_vec();
    assert_eq!(maxima, [7.0, 0.0, 9.0]);
    assert_eq!(events.counts(), [1, 0, 2]);
    // An hours clause keeps the part, summed over its labels.
    let grouped = events.grouped(0.0).unwrap();
    assert!(!grouped.has_labels());
    assert_eq!(grouped.aggregate_part(2), Some(&[11.0][..]));
    // Kept by label, the part keeps its columns.
    let kept = events.grouped_with(0.0, GroupLabels::Largest).unwrap();
    assert_eq!(kept.aggregate_part(2), Some(&[5.0, 6.0][..]));
    assert_eq!(kept.aggregate_part_indices(), Some(&[1, 2][..]));
    assert_eq!(kept.events(2), &[8.0, 9.0]);
    // Labels go before the part.
    assert!(
        events
            .clone()
            .with_labels(vec![("class".into(), vec!["a".into(); 3])])
            .is_err()
    );
    let short = part(vec![], vec![vec![]], vec![1.0, 2.0]);
    assert!(events.clone().with_aggregate_part(&short).is_err());
    let negative = part(vec![], vec![vec![]], vec![1.0, -2.0, 3.0]);
    assert!(events.clone().with_aggregate_part(&negative).is_err());
    let kind = part(vec!["kind".into()], vec![vec!["a"]], vec![1.0, 2.0, 3.0]);
    assert!(events.clone().with_aggregate_part(&kind).is_err());
}

#[test]
fn merging_adds_aggregate_parts_with_the_same_labels() {
    let a = EventSet::from_years(vec![vec![7.0], vec![]], 0)
        .unwrap()
        .with_aggregate_part(&part(
            vec!["peril".into()],
            vec![vec!["fire"]],
            vec![1.0, 2.0],
        ))
        .unwrap();
    let b = EventSet::from_years(vec![vec![], vec![3.0]], 0)
        .unwrap()
        .with_aggregate_part(&part(
            vec!["peril".into()],
            vec![vec!["fire"], vec!["wind"]],
            vec![10.0, 20.0, 30.0, 40.0],
        ))
        .unwrap();
    let all = a.merge(&b).unwrap();
    let labels: Vec<Vec<String>> = all
        .aggregate_part_labels()
        .unwrap()
        .into_iter()
        .map(<[String]>::to_vec)
        .collect();
    assert_eq!(labels, [["fire"], ["wind"]]);
    assert_eq!(all.aggregate_part(0), Some(&[11.0, 20.0][..]));
    assert_eq!(all.aggregate_part(1), Some(&[32.0, 40.0][..]));
    // Unlabelled parts add into one amount a year; one set's alone is kept.
    let c = EventSet::from_years(vec![vec![1.0], vec![2.0]], 0).unwrap();
    let d = c
        .clone()
        .with_aggregate_part(&part(vec![], vec![vec![]], vec![5.0, 6.0]))
        .unwrap();
    let both = d.merge(&d).unwrap();
    assert_eq!(both.aggregate_part(1), Some(&[12.0][..]));
    assert!(!both.has_labels());
    assert_eq!(c.merge(&d).unwrap().aggregate_part(0), Some(&[5.0][..]));
    assert!(!c.merge(&c).unwrap().has_aggregate_part());
}

#[test]
fn merge_keeps_the_labels_of_either_set() {
    let cat = EventSet::from_years(vec![vec![100.0], vec![]], 1)
        .unwrap()
        .with_labels(vec![("peril".into(), vec!["wind".into()])])
        .unwrap();
    let other = EventSet::from_years(vec![vec![3.0], vec![5.0]], 2)
        .unwrap()
        .with_labels(vec![("class".into(), vec!["motor".into(), "home".into()])])
        .unwrap();
    let all = cat.merge(&other).unwrap();
    assert_eq!(all.label_dims(), ["peril", "class"]);
    assert_eq!(all.events(0), &[100.0, 3.0]);
    let year: Vec<Vec<String>> = all.labels(0).unwrap().iter().map(|v| v.to_vec()).collect();
    assert_eq!(
        year,
        [
            vec!["wind".to_string(), String::new()],
            vec![String::new(), "motor".into()]
        ]
    );
    // An unlabelled set's losses get "" in every dimension.
    let plain = EventSet::from_years(vec![vec![1.0], vec![2.0]], 0).unwrap();
    let mixed = plain.merge(&cat).unwrap();
    assert_eq!(
        mixed.label_combinations(),
        [vec![String::new()], vec!["wind".to_string()]]
    );
    assert!(!plain.merge(&plain).unwrap().has_labels());
    assert_eq!(
        all.slice(1..2).label_combinations(),
        all.label_combinations()
    );
    assert!(
        all.with_uniform_times()
            .grouped(72.0)
            .map(|g| !g.has_labels())
            .unwrap()
    );
}

#[test]
fn labels_are_checked() {
    let events = EventSet::from_years(vec![vec![1.0, 2.0]], 0).unwrap();
    let two = || vec!["a".to_string(), "b".into()];
    assert!(events.clone().with_labels(vec![]).is_err());
    assert!(
        events
            .clone()
            .with_labels(vec![("kind".into(), two())])
            .is_err()
    );
    assert!(
        events
            .clone()
            .with_labels(vec![("".into(), two())])
            .is_err()
    );
    assert!(
        events
            .clone()
            .with_labels(vec![("p".into(), two()), ("p".into(), two())])
            .is_err()
    );
    assert!(
        events
            .clone()
            .with_labels(vec![("p".into(), vec!["a".into()])])
            .is_err()
    );
    assert!(EventSet::from_long_labeled(&[0], &[1.0], vec![("p".into(), two())], 1, 0).is_err());
    assert!(
        EventSet::from_long_labeled(&[1], &[1.0], vec![("p".into(), vec!["a".into()])], 1, 0)
            .is_err()
    );
}

use super::*;
use crate::fft;
use prospicio_prob::{Grid, NegativeBinomial, Poisson};

fn severity() -> Grid {
    Grid::new(1.0, vec![0.1, 0.3, 0.25, 0.2, 0.1, 0.05]).unwrap()
}

#[test]
fn a_serial_severity_gives_the_same_draws() {
    use prospicio_prob::Custom;
    use std::sync::Arc;
    let grid = severity();
    let g = grid.clone();
    let serial = Custom::new(
        "grid",
        Arc::new(move |x| Ok(g.cdf(x))),
        Some(Arc::new(move |p| {
            grid.quantile(p).map_err(|e| e.to_string())
        })),
        false,
    )
    .unwrap();
    let freq = Poisson::new(4.0).unwrap();
    let a = simulate_events(&freq, &serial, 2_000, 3).unwrap();
    let b = simulate_events(&freq, &severity(), 2_000, 3).unwrap();
    assert_eq!(a, b);
}

#[test]
fn identical_across_thread_counts_and_replayable() {
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| simulate_events(&Poisson::new(4.0).unwrap(), &severity(), 5_000, 9))
            .unwrap()
    };
    let one = run(1);
    assert_eq!(one, run(4));
    assert_eq!(one, run(16));

    // Replay year 1234 by hand from its own stream.
    let mut rng = StreamRng::new(9, 1234);
    let n = Poisson::new(4.0)
        .unwrap()
        .quantile(rng.next_open01())
        .unwrap();
    let replay: Vec<f64> = (0..n)
        .map(|_| severity().quantile(rng.next_open01()).unwrap())
        .collect();
    assert_eq!(one.events(1234), replay);
}

#[test]
fn totals_match_fft_by_kolmogorov_smirnov() {
    // A discrete severity makes the FFT result exact, so the simulated
    // totals must be a sample from it.
    let n_sims = 200_000;
    for freq in [
        &Poisson::new(3.0).unwrap() as &(dyn Counting + Sync),
        &NegativeBinomial::new(2.5, 1.5).unwrap(),
    ] {
        let (exact, report) = fft(freq, &severity(), 400).unwrap();
        assert!(report.tail_mass < 1e-12);
        let totals = simulate_events(freq, &severity(), n_sims, 1)
            .unwrap()
            .totals()
            .unwrap();
        let ks = (0..400)
            .map(|k| (totals.cdf(k as f64) - exact.cdf(k as f64)).abs())
            .fold(0.0, f64::max);
        // Critical value at the 0.1% level: 1.95 / sqrt(n).
        assert!(ks < 1.95 / (n_sims as f64).sqrt(), "KS {ks}");
    }
}

#[test]
fn totals_carry_provenance() {
    let events = simulate_events(&Poisson::new(1.0).unwrap(), &severity(), 10, 77).unwrap();
    let totals = events.totals().unwrap();
    assert_eq!(totals.n_sims(), 10);
    assert_eq!(totals.provenance().seed, Some(77));
    assert_eq!(totals.provenance().model, "frequency_severity_monte_carlo");
    assert_eq!(totals.provenance().samplers, Some(current_samplers()));
    assert_eq!(
        events.counts().iter().sum::<usize>(),
        (0..10).map(|i| events.events(i).len()).sum::<usize>()
    );
}

#[test]
fn years_from_elsewhere_record_no_samplers() {
    // Two event loss tables with the default seed: same streams, but
    // nothing says they are the same draws.
    let a = EventSet::from_years(vec![vec![1.0, 2.0], vec![3.0]], 0).unwrap();
    let b = EventSet::from_years(vec![vec![4.0], vec![]], 0).unwrap();
    assert_eq!(a.samplers(), None);
    let (pa, pb) = (a.totals().unwrap(), b.totals().unwrap());
    assert_eq!(pa.provenance().samplers, None);
    assert_eq!(pa.provenance().seed, Some(0));
    assert!(pa.provenance().shares_streams(pb.provenance()));
    assert!(!pa.provenance().replays_same_draws(pb.provenance()));
}

#[test]
fn times_are_checked_and_replay() {
    let events = EventSet::from_years(vec![vec![1.0, 2.0], vec![3.0]], 0).unwrap();
    assert!(events.clone().with_times(vec![0.5, 0.4, 0.1]).is_err());
    assert!(events.clone().with_times(vec![0.1, 1.5, 0.1]).is_err());
    assert!(events.clone().with_times(vec![0.1, 0.2]).is_err());
    // A later year may start before an earlier one ended.
    assert!(events.clone().with_times(vec![0.4, 0.9, 0.1]).is_ok());
    assert!(!events.has_times() && events.times(0).is_none());

    let freq = Poisson::new(4.0).unwrap();
    let few = simulate_events(&freq, &severity(), 50, 8)
        .unwrap()
        .with_uniform_times();
    let many = simulate_events(&freq, &severity(), 500, 8)
        .unwrap()
        .with_uniform_times();
    let mut all = Vec::new();
    for i in 0..50 {
        let t = few.times(i).unwrap();
        assert_eq!(t, many.times(i).unwrap());
        assert_eq!(t.len(), few.events(i).len());
        assert!(t.windows(2).all(|w| w[0] <= w[1]));
        assert!(t.iter().all(|&x| x > 0.0 && x < 1.0));
        all.extend_from_slice(t);
    }
    // Spread over the year.
    let mean = all.iter().sum::<f64>() / all.len() as f64;
    assert!((mean - 0.5).abs() < 0.06, "{mean}");
}

#[test]
fn seasonal_times_follow_the_weights() {
    let events = EventSet::from_years(vec![vec![1.0, 2.0], vec![3.0]], 0).unwrap();
    assert!(events.clone().with_seasonal_times(&[]).is_err());
    assert!(events.clone().with_seasonal_times(&[0.0, 0.0]).is_err());
    assert!(events.clone().with_seasonal_times(&[1.0, -1.0]).is_err());
    assert!(
        events
            .clone()
            .with_seasonal_times(&[1.0, f64::NAN])
            .is_err()
    );

    let freq = Poisson::new(4.0).unwrap();
    let sim = simulate_events(&freq, &severity(), 50_000, 8).unwrap();

    // Equal weights are the uniform times, to rounding.
    let uniform = sim.clone().with_uniform_times();
    let flat = sim.clone().with_seasonal_times(&[3.0; 12]).unwrap();
    for i in 0..1_000 {
        for (a, b) in uniform.times(i).unwrap().iter().zip(flat.times(i).unwrap()) {
            assert!((a - b).abs() < 1e-15, "{a} {b}");
        }
    }

    // A season: nothing in the first quarter, most in the third.
    let weights = [0.0, 1.0, 6.0, 1.0];
    let seasonal = sim.with_seasonal_times(&weights).unwrap();
    let mut all = Vec::new();
    for i in 0..seasonal.n_sims() {
        let t = seasonal.times(i).unwrap();
        assert_eq!(t.len(), seasonal.events(i).len());
        assert!(t.windows(2).all(|w| w[0] <= w[1]));
        assert!(t.iter().all(|&x| (0.25..=1.0).contains(&x)));
        all.extend_from_slice(t);
    }
    // The piecewise-linear cdf of the season.
    let cdf = |x: f64| {
        let mut c = 0.0;
        for (k, w) in weights.iter().enumerate() {
            let (lo, hi) = (k as f64 / 4.0, (k + 1) as f64 / 4.0);
            c += w / 8.0 * ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
        }
        c
    };
    all.sort_by(f64::total_cmp);
    let n = all.len() as f64;
    let ks = all
        .iter()
        .enumerate()
        .map(|(j, &x)| {
            let f = cdf(x);
            (f - j as f64 / n).abs().max(((j + 1) as f64 / n - f).abs())
        })
        .fold(0.0, f64::max);
    // Times within a year are not independent of its count, but pooled
    // over i.i.d. losses they are a sample of the season: the 0.1%
    // critical value 1.95 / sqrt(n).
    assert!(ks < 1.95 / n.sqrt(), "KS {ks}");
}

#[test]
fn rejects_zero_sims() {
    assert!(simulate_events(&Poisson::new(1.0).unwrap(), &severity(), 0, 1).is_err());
}

#[test]
fn merging_keeps_every_loss_and_orders_by_time() {
    let a = EventSet::from_years(vec![vec![1.0, 2.0], vec![3.0]], 5).unwrap();
    let b = EventSet::from_years(vec![vec![10.0], vec![]], 6).unwrap();
    let m = a.merge(&b).unwrap();
    assert_eq!(m.events(0), &[1.0, 2.0, 10.0]);
    assert_eq!(m.events(1), &[3.0]);
    assert_eq!(m.seed(), 5);
    assert!(!m.has_times());
    let si_a = a.clone().with_sums_insured(vec![5.0, 5.0, 5.0]).unwrap();
    let si_b = b.clone().with_sums_insured(vec![20.0]).unwrap();
    assert_eq!(
        si_a.merge(&si_b).unwrap().sums_insured(0),
        Some(&[5.0, 5.0, 20.0][..])
    );
    assert!(si_a.merge(&b).is_err());
    let timed = a.clone().with_times(vec![0.1, 0.2, 0.3]).unwrap();
    assert!(timed.merge(&b).is_err());
    let short = EventSet::from_years(vec![vec![1.0]], 0).unwrap();
    assert!(a.merge(&short).is_err());
    // A tie in time keeps this set's loss first.
    let t_a = EventSet::from_years(vec![vec![1.0]], 0)
        .unwrap()
        .with_times(vec![0.5])
        .unwrap();
    let t_b = EventSet::from_years(vec![vec![2.0]], 0)
        .unwrap()
        .with_times(vec![0.5])
        .unwrap();
    assert_eq!(t_a.merge(&t_b).unwrap().events(0), &[1.0, 2.0]);
    assert_eq!(t_b.merge(&t_a).unwrap().events(0), &[2.0, 1.0]);
}

#[test]
fn an_hours_clause_groups_within_the_window_and_keeps_totals() {
    let h = 1.0 / 8760.0;
    let e = EventSet::from_years(vec![vec![1.0, 2.0, 4.0], vec![], vec![8.0, 16.0]], 0)
        .unwrap()
        .with_times(vec![0.0, 72.0 * h, 72.5 * h, 0.3, 0.3])
        .unwrap();
    let g = e.grouped(72.0).unwrap();
    // 72 hours after the first is still in its window; 72.5 is not.
    assert_eq!(g.events(0), &[3.0, 4.0]);
    assert_eq!(g.times(0), Some(&[0.0, 72.5 * h][..]));
    assert!(g.events(1).is_empty());
    assert_eq!(g.events(2), &[24.0]);
    let total = |s: &EventSet| -> f64 { (0..s.n_sims()).flat_map(|y| s.events(y)).sum() };
    assert_eq!(total(&g), total(&e));
    // A zero window still groups losses at the same time.
    assert_eq!(e.grouped(0.0).unwrap().events(2), &[24.0]);
    assert!(
        EventSet::from_years(vec![vec![1.0]], 0)
            .unwrap()
            .grouped(72.0)
            .is_err()
    );
    assert!(e.grouped(-1.0).is_err());
}

fn peril(values: &[&str]) -> Vec<(String, Vec<String>)> {
    vec![(
        "peril".into(),
        values.iter().map(|v| v.to_string()).collect(),
    )]
}

#[test]
fn an_hours_clause_keeps_labels_that_agree() {
    let h = 1.0 / 8760.0;
    let e = EventSet::from_years(vec![vec![1.0, 2.0, 4.0], vec![8.0]], 0)
        .unwrap()
        .with_times(vec![0.0, 10.0 * h, 0.5, 0.2])
        .unwrap()
        .with_labels(peril(&["wind"; 4]))
        .unwrap();
    // One label value is kept by either rule; the default drops it.
    for rule in [GroupLabels::Agree, GroupLabels::Largest] {
        let g = e.grouped_with(72.0, rule).unwrap();
        assert_eq!(g.events(0), &[3.0, 4.0]);
        assert_eq!(g.label_combinations(), [["wind"]]);
        assert_eq!(g.labels(1).unwrap(), [["wind".to_string()]]);
    }
    let dropped = e.grouped_with(72.0, GroupLabels::Drop).unwrap();
    assert_eq!(dropped, e.grouped(72.0).unwrap());
    assert!(!dropped.has_labels());
    // Without labels, every rule groups as the plain clause does.
    let plain = EventSet::from_years(vec![vec![1.0, 2.0]], 0)
        .unwrap()
        .with_times(vec![0.0, 0.0])
        .unwrap();
    assert_eq!(
        plain.grouped_with(0.0, GroupLabels::Agree).unwrap(),
        plain.grouped(0.0).unwrap()
    );
    assert!(plain.grouped_by("peril", 0.0, GroupLabels::Agree).is_err());
}

#[test]
fn an_occurrence_of_mixed_labels_takes_its_largest_loss_or_errs() {
    let h = 1.0 / 8760.0;
    let e = EventSet::from_years(vec![vec![5.0, 9.0, 9.0, 2.0]], 0)
        .unwrap()
        .with_times(vec![0.1, 0.1 + h, 0.1 + 2.0 * h, 0.6])
        .unwrap()
        .with_labels(peril(&["wind", "flood", "quake", "quake"]))
        .unwrap();
    let err = e
        .grouped_with(72.0, GroupLabels::Agree)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("year 1") && err.contains("wind") && err.contains("flood"),
        "{err}"
    );
    let g = e.grouped_with(72.0, GroupLabels::Largest).unwrap();
    assert_eq!(g.events(0), &[23.0, 2.0]);
    // The first of the two largest (flood's 9 before quake's 9).
    assert_eq!(
        g.labels(0).unwrap(),
        [["flood".to_string()], ["quake".to_string()]]
    );
}

#[test]
fn an_hours_clause_by_peril_keeps_perils_apart() {
    let h = 1.0 / 8760.0;
    let class = |v: &[&str]| {
        (
            "class".to_string(),
            v.iter().map(|s| s.to_string()).collect(),
        )
    };
    let mut labels = peril(&["wind", "flood", "wind", "wind", "flood"]);
    labels.push(class(&["home", "home", "auto", "home", "home"]));
    let e = EventSet::from_years(vec![vec![1.0, 2.0, 4.0, 16.0, 8.0]], 0)
        .unwrap()
        .with_times(vec![
            0.1,
            0.1 + 10.0 * h,
            0.1 + 20.0 * h,
            0.1 + 80.0 * h,
            0.1 + 100.0 * h,
        ])
        .unwrap()
        .with_labels(labels)
        .unwrap();
    // Wind: 1 and 4 at 0 and 20 hours, 16 at 80 opens a new window;
    // flood: 2 at 10 hours, 8 at 100 (90 hours on) opens a new one.
    // Listed by the time each window opens.
    let g = e.grouped_by("peril", 72.0, GroupLabels::Largest).unwrap();
    assert_eq!(g.events(0), &[5.0, 2.0, 16.0, 8.0]);
    let t = g.times(0).unwrap();
    assert_eq!(t, &[0.1, 0.1 + 10.0 * h, 0.1 + 80.0 * h, 0.1 + 100.0 * h]);
    // Wind's first occurrence mixes classes and takes auto's (the 4).
    assert_eq!(
        g.labels(0).unwrap()[0],
        ["wind".to_string(), "auto".to_string()]
    );
    // The classes disagree within wind's first window.
    assert!(e.grouped_by("peril", 72.0, GroupLabels::Agree).is_err());
    assert!(e.grouped_by("region", 72.0, GroupLabels::Agree).is_err());
    // Grouping by peril on one peril is the plain clause.
    let wind = EventSet::from_years(vec![vec![1.0, 4.0, 16.0]], 0)
        .unwrap()
        .with_times(vec![0.1, 0.1 + 20.0 * h, 0.1 + 80.0 * h])
        .unwrap()
        .with_labels(peril(&["wind"; 3]))
        .unwrap();
    assert_eq!(
        wind.grouped_by("peril", 72.0, GroupLabels::Agree).unwrap(),
        wind.grouped_with(72.0, GroupLabels::Agree).unwrap()
    );
}

#[test]
fn permute_carries_every_field_of_a_year() {
    let e = EventSet::from_years(vec![vec![1.0, 2.0], vec![], vec![3.0]], 9)
        .unwrap()
        .with_times(vec![0.1, 0.4, 0.8])
        .unwrap()
        .with_sums_insured(vec![10.0, 20.0, 30.0])
        .unwrap()
        .with_labels(vec![(
            "peril".into(),
            vec!["wind".into(), "quake".into(), "wind".into()],
        )])
        .unwrap()
        .with_input_hash(Some("h".into()));
    let p = e.permute(&[2, 0, 0, 1]).unwrap();
    assert_eq!(p.n_sims(), 4);
    assert_eq!(p.events(0), &[3.0]);
    assert_eq!(p.events(1), &[1.0, 2.0]);
    assert_eq!(p.events(2), &[1.0, 2.0]);
    assert!(p.events(3).is_empty());
    assert_eq!(p.times(0), Some(&[0.8][..]));
    assert_eq!(p.sums_insured(2), Some(&[10.0, 20.0][..]));
    let wind = ["wind".to_string()];
    let quake = ["quake".to_string()];
    assert_eq!(p.labels(1).unwrap(), vec![&wind[..], &quake[..]]);
    assert_eq!(p.label_combinations(), e.label_combinations());
    assert_eq!((p.seed(), p.input_hash()), (9, Some("h")));
    assert!(e.permute(&[3]).is_err());
    assert!(e.permute(&[]).is_err());
}
