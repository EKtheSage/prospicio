//! The bootstraps' future cells (`fit_with_cells` and
//! `fit_segments_with_cells` of `OdpBootstrap` and `MackBootstrap`;
//! `reserving-inflation.md`, decisions A to I, issue #979).
//!
//! * Keeping the cells draws no other random number: the reserves and the
//!   patterns are bit for bit those of `fit` and `fit_segments`, with and
//!   without a tail, and with segments under each `SegmentDependence`; the
//!   reserves' provenance is the same, the cells' adds `cells =
//!   incremental`.
//! * Each origin's cells sum to its reserve in every simulation: the ODP
//!   bootstrap's exactly when summed in age order (its reserve is that
//!   sum), Mack's to round-off (its cells are differences of cumulative
//!   values); also through `aggregate`, and row by row under a rank
//!   correlation, which moves each segment's cells with its reserves.
//! * The mean of each cell is the chain ladder's future increment
//!   (`FutureCells::from_chain_ladder`) on GenIns at 10,000 simulations:
//!   Mack's within four standard errors of the mean (the worst 2.6); the
//!   ODP bootstrap's within four plus 3% of the increment, since its
//!   simulated reserves themselves sit above the chain ladder's (by 1% to
//!   2.3% an origin on GenIns, 2010 the most; the mean of a product of
//!   pseudo factors exceeds the product of their means), and the cells sum
//!   to them: the worst cell is 2010 at 84 months, 4.9 standard errors
//!   above, using 56% of its allowance. The ODP bootstrap's reserves are
//!   checked against R's `BootChainLadder`, not the chain ladder
//!   (`reserving.rs`). So that offset hides no misplaced cell, each cell's
//!   share of its origin's mean reserve is also checked against the chain
//!   ladder's share within four standard errors and no other allowance
//!   (the worst 2.9 for each bootstrap).
//! * The segments' cells stay in their own columns when the segments
//!   differ in shape (a fully developed square, which has no cell without
//!   a tail, beside RAA), and go through `reinflate_draws` by their
//!   dimension names; the cells are the same bits on one thread and four.
//! * `Indexation::reinflate_draws` takes the cells: paths equal to the
//!   index's own future rate give the deterministic re-inflation; a unit
//!   index gives back the reserves by origin; each simulation's re-inflated
//!   cells are `Indexation::reinflate` of that simulation's cells.
//! * A fully developed triangle with no tail has no future cell, and
//!   `fit_with_cells` fails while `fit` works; with a tail, every origin
//!   has one more cell, one development period past the oldest age.

use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};
use prospicio_reserving::{
    CalendarIndex, ChainLadderFit, DevelopmentColumn, DevelopmentPatterns, Error, FutureCell,
    FutureCells, Grain, IndexPaths, Indexation, Long, MackBootstrap, Month, OdpBootstrap, Period,
    SegmentDependence, Tail, TailConstant, Triangle,
};
use prospicio_validation::triangle;

/// `tri`'s values twice, under the `lob` values `A` and `B`, with every
/// value of `B` doubled.
fn twins(tri: &Triangle) -> Triangle {
    let long = tri.to_long();
    let values = &long.values.iter().find(|(c, _)| c == "values").unwrap().1;
    let n = long.origin.len();
    let keys: Vec<&str> = (0..2 * n).map(|i| if i < n { "A" } else { "B" }).collect();
    let doubled: Vec<f64> = values
        .iter()
        .copied()
        .chain(values.iter().map(|v| 2.0 * v))
        .collect();
    Triangle::from_long(&Long {
        keys: &[("lob", &keys)],
        origin: &[long.origin.clone(), long.origin.clone()].concat(),
        development: DevelopmentColumn::Age(&[long.development.clone(), long.development].concat()),
        values: &[("values", &doubled)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

fn constant_tail(factor: f64) -> Tail {
    Tail::Constant(TailConstant {
        factor,
        ..Default::default()
    })
}

/// What a fit gives, whichever bootstrap: reserves, patterns, cells.
struct Run {
    reserves: PredictiveDistribution,
    patterns: DevelopmentPatterns,
    cells: Option<PredictiveDistribution>,
}

fn odp(tail: Tail, dependence: SegmentDependence, n_sims: usize) -> OdpBootstrap {
    OdpBootstrap {
        n_sims,
        seed: 979,
        tail,
        dependence,
        ..Default::default()
    }
}

fn mack(tail: Tail, dependence: SegmentDependence, n_sims: usize) -> MackBootstrap {
    MackBootstrap {
        n_sims,
        seed: 980,
        tail,
        dependence,
        ..Default::default()
    }
}

/// The ODP bootstrap's single fit, with or without cells.
fn odp_one(boot: &OdpBootstrap, tri: &Triangle, cells: bool) -> (Run, ChainLadderFit) {
    let fit = if cells {
        boot.fit_with_cells(tri, "values")
    } else {
        boot.fit(tri, "values")
    }
    .unwrap();
    let run = Run {
        reserves: fit.reserves,
        patterns: fit.patterns,
        cells: fit.cells,
    };
    (run, fit.chain_ladder)
}

/// Mack's bootstrap's single fit, with or without cells.
fn mack_one(boot: &MackBootstrap, tri: &Triangle, cells: bool) -> (Run, ChainLadderFit) {
    let fit = if cells {
        boot.fit_with_cells(tri, "values")
    } else {
        boot.fit(tri, "values")
    }
    .unwrap();
    let run = Run {
        reserves: fit.reserves,
        patterns: fit.patterns,
        cells: fit.cells,
    };
    (run, fit.mack.chain_ladder)
}

fn odp_segments(boot: &OdpBootstrap, tri: &Triangle, cells: bool) -> Run {
    let fit = if cells {
        boot.fit_segments_with_cells(tri, "values")
    } else {
        boot.fit_segments(tri, "values")
    }
    .unwrap();
    Run {
        reserves: fit.reserves,
        patterns: fit.patterns,
        cells: fit.cells,
    }
}

fn mack_segments(boot: &MackBootstrap, tri: &Triangle, cells: bool) -> Run {
    let fit = if cells {
        boot.fit_segments_with_cells(tri, "values")
    } else {
        boot.fit_segments(tri, "values")
    }
    .unwrap();
    Run {
        reserves: fit.reserves,
        patterns: fit.patterns,
        cells: fit.cells,
    }
}

fn bits(pd: &PredictiveDistribution) -> Vec<u64> {
    pd.draw_matrix().iter().map(|x| x.to_bits()).collect()
}

/// `with` (cells kept) has `without`'s reserves and patterns bit for bit
/// and its reserves' provenance, and cells whose provenance is the
/// reserves' with `cells = incremental`; `without` has no cells.
fn check_identical(what: &str, without: &Run, with: &Run) {
    assert!(without.cells.is_none(), "{what}");
    assert_eq!(bits(&with.reserves), bits(&without.reserves), "{what}");
    assert_eq!(with.reserves.dims(), without.reserves.dims(), "{what}");
    assert_eq!(
        with.reserves.components(),
        without.reserves.components(),
        "{what}"
    );
    assert_eq!(with.patterns, without.patterns, "{what}");
    assert_eq!(
        with.reserves.provenance(),
        without.reserves.provenance(),
        "{what}"
    );
    let cells = with.cells.as_ref().expect("cells");
    let want: Provenance = with
        .reserves
        .provenance()
        .clone()
        .param("cells", "incremental");
    assert_eq!(cells.provenance(), &want, "{what}");
    assert_eq!(cells.n_sims(), with.reserves.n_sims(), "{what}");
}

/// Each origin's cells sum to its reserve in every simulation: exactly
/// when `exact` (summed in component order, which is age order), else to
/// `rel` relative; an origin with no cell has a reserve of 0. Then the same
/// through `aggregate` over the reserves' dimensions, to 1e-9 relative.
fn check_sums(what: &str, reserves: &PredictiveDistribution, cells: &PredictiveDistribution, exact: bool) {
    let rel = 1e-9;
    let dims = cells.dims();
    assert_eq!(dims[..dims.len() - 1], *reserves.dims(), "{what}");
    assert_eq!(dims.last().map(String::as_str), Some("development"));
    // The reserve each cell belongs to.
    let owner: Vec<usize> = cells
        .components()
        .iter()
        .map(|key| {
            reserves
                .component_index(&key[..key.len() - 1].to_vec())
                .unwrap_or_else(|| panic!("{what}: no reserve for {key:?}"))
        })
        .collect();
    let n_reserves = reserves.n_components();
    for sim in 0..reserves.n_sims() {
        let row = cells.row(sim).unwrap();
        let mut sums = vec![0.0; n_reserves];
        let mut has = vec![false; n_reserves];
        for (&o, &x) in owner.iter().zip(row) {
            sums[o] += x;
            has[o] = true;
        }
        for (o, &reserve) in reserves.row(sim).unwrap().iter().enumerate() {
            if !has[o] {
                assert_eq!(reserve, 0.0, "{what}: sim {sim}, origin {o}");
            } else if exact {
                assert_eq!(sums[o].to_bits(), reserve.to_bits(), "{what}: sim {sim}, origin {o}");
            } else {
                let scale = reserve.abs().max(1.0);
                assert!(
                    (sums[o] - reserve).abs() <= rel * scale,
                    "{what}: sim {sim}, origin {o}: {} {reserve}",
                    sums[o]
                );
            }
        }
    }
    let keep: Vec<&str> = reserves.dims().iter().map(String::as_str).collect();
    let by_origin = cells.aggregate(&keep).unwrap();
    for (j, key) in by_origin.components().iter().enumerate() {
        let o = reserves.component_index(key).expect("a reserve");
        for sim in 0..reserves.n_sims() {
            let (got, want) = (by_origin.row(sim).unwrap()[j], reserves.row(sim).unwrap()[o]);
            assert!(
                (got - want).abs() <= rel * want.abs().max(1.0),
                "{what}: aggregate, sim {sim}, {key:?}: {got} {want}"
            );
        }
    }
}

#[test]
fn cells_leave_a_single_fit_unchanged_and_sum_to_its_reserves() {
    let raa = triangle("raa");
    let independent = SegmentDependence::Independent;
    for (name, tail) in [("no tail", Tail::default()), ("tail", constant_tail(1.05))] {
        let boot = odp(tail, independent.clone(), 500);
        let (without, _) = odp_one(&boot, &raa, false);
        let (with, _) = odp_one(&boot, &raa, true);
        let what = format!("odp, {name}");
        check_identical(&what, &without, &with);
        let cells = with.cells.as_ref().unwrap();
        assert_eq!(cells.dims(), ["origin", "development"]);
        check_sums(&what, &with.reserves, cells, true);

        let boot = mack(tail, independent.clone(), 500);
        let (without, _) = mack_one(&boot, &raa, false);
        let (with, _) = mack_one(&boot, &raa, true);
        let what = format!("mack, {name}");
        check_identical(&what, &without, &with);
        let cells = with.cells.as_ref().unwrap();
        assert_eq!(cells.dims(), ["origin", "development"]);
        check_sums(&what, &with.reserves, cells, false);
    }
}

#[test]
fn cells_leave_segments_unchanged_under_every_dependence() {
    let tri = twins(&triangle("raa"));
    let dependences = [
        SegmentDependence::Independent,
        SegmentDependence::Synchronized,
        SegmentDependence::RankCorrelation {
            spearman: vec![1.0, 0.7, 0.7, 1.0],
        },
    ];
    for dependence in dependences {
        for (name, tail) in [("no tail", Tail::default()), ("tail", constant_tail(1.05))] {
            let boot = odp(tail, dependence.clone(), 400);
            let without = odp_segments(&boot, &tri, false);
            let with = odp_segments(&boot, &tri, true);
            let what = format!("odp, {dependence:?}, {name}");
            check_identical(&what, &without, &with);
            let cells = with.cells.as_ref().unwrap();
            assert_eq!(cells.dims(), ["lob", "origin", "development"]);
            check_sums(&what, &with.reserves, cells, true);

            let boot = mack(tail, dependence.clone(), 400);
            let without = mack_segments(&boot, &tri, false);
            let with = mack_segments(&boot, &tri, true);
            let what = format!("mack, {dependence:?}, {name}");
            check_identical(&what, &without, &with);
            check_sums(&what, &with.reserves, with.cells.as_ref().unwrap(), false);
        }
    }
}

#[test]
fn a_segment_picked_alone_keeps_its_cells() {
    let tri = twins(&triangle("raa"));
    let rank = SegmentDependence::RankCorrelation {
        spearman: vec![1.0, 0.5, 0.5, 1.0],
    };
    let fit = odp(constant_tail(1.05), rank.clone(), 300)
        .fit_segments_with_cells(&tri, "values")
        .unwrap();
    let alone = fit.segment(&[("lob", "B")]).unwrap();
    let cells = alone.cells.as_ref().unwrap();
    assert_eq!(cells.dims(), ["lob", "origin", "development"]);
    assert!(
        cells
            .components()
            .iter()
            .all(|k| k[0] == KeyValue::from("B"))
    );
    check_sums("odp, B alone", &alone.reserves, cells, true);
    assert!(
        odp(constant_tail(1.05), rank.clone(), 300)
            .fit_segments(&tri, "values")
            .unwrap()
            .segment(&[("lob", "B")])
            .unwrap()
            .cells
            .is_none()
    );
    let fit = mack(Tail::default(), rank, 300)
        .fit_segments_with_cells(&tri, "values")
        .unwrap();
    let alone = fit.segment(&[("lob", "A")]).unwrap();
    check_sums("mack, A alone", &alone.reserves, alone.cells.as_ref().unwrap(), false);
}

/// Each cell's mean against the chain ladder's future increment, within
/// four standard errors of the mean plus `bias` of the increment (and 1e-9
/// of it, for cells with no spread).
fn check_means(what: &str, cells: &PredictiveDistribution, cl: &ChainLadderFit, bias: f64) {
    let want = FutureCells::from_chain_ladder(cl);
    let n = cells.n_sims() as f64;
    let mut checked = 0;
    let (mut worst_z, mut worst_rel) = (0.0f64, 0.0f64);
    let mut worst_use = (0.0f64, String::new());
    for cell in &want.cells {
        let key = vec![
            KeyValue::Period(want.origins[cell.origin]),
            KeyValue::Int(cell.age.into()),
        ];
        let j = cells
            .component_index(&key)
            .unwrap_or_else(|| panic!("{what}: no cell {key:?}"));
        let x: Vec<f64> = (0..cells.n_sims())
            .map(|s| cells.row(s).unwrap()[j])
            .collect();
        let mean = x.iter().sum::<f64>() / n;
        let sd = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let se = sd / n.sqrt();
        let gap = (mean - cell.value).abs();
        worst_z = worst_z.max(gap / se);
        worst_rel = worst_rel.max(gap / cell.value.abs());
        let allowance = 4.0 * se + (bias + 1e-9) * cell.value.abs();
        if gap / allowance > worst_use.0 {
            worst_use = (gap / allowance, format!("{key:?} {:.2} se", gap / se));
        }
        assert!(
            gap <= allowance,
            "{what}: {key:?}: mean {mean}, chain ladder {}, se {se}",
            cell.value
        );
        checked += 1;
    }
    println!(
        "{what}: worst {worst_z:.2} standard errors, worst relative gap {worst_rel:.4}, \
         most of its allowance used {:.3} at {}",
        worst_use.0, worst_use.1
    );
    assert_eq!(checked, want.cells.len());
    assert_eq!(cells.n_components(), want.cells.len(), "{what}");
}

/// The worst gaps at 10,000 simulations (Windows): ODP 4.91 standard
/// errors and 3.1% of an increment, at most 56% of a cell's allowance;
/// Mack 2.63 standard errors, at most 66%. The shares: ODP 2.85 standard
/// errors, Mack 2.79.
#[test]
fn the_mean_cells_are_the_chain_ladders_on_genins() {
    let genins = triangle("genins");
    let independent = SegmentDependence::Independent;
    let (run, cl) = odp_one(&odp(Tail::default(), independent.clone(), 10_000), &genins, true);
    check_means("odp", run.cells.as_ref().unwrap(), &cl, 0.03);
    check_shares("odp", run.cells.as_ref().unwrap(), &run.reserves, &cl);
    let (run, cl) = mack_one(&mack(Tail::default(), independent, 10_000), &genins, true);
    check_means("mack", run.cells.as_ref().unwrap(), &cl, 0.0);
    check_shares("mack", run.cells.as_ref().unwrap(), &run.reserves, &cl);
}

/// GenIns' index, 2001 to 2010 at 4% a year, base March 2006, and the
/// future rate `rate`.
fn genins_index(rate: f64) -> (Indexation, Vec<f64>) {
    let given: Vec<f64> = (0..10).map(|k| 100.0 * 1.04_f64.powi(k)).collect();
    let index = Indexation {
        index: CalendarIndex::new(Period::year(2001), given.clone())
            .unwrap()
            .with_future_rate(rate)
            .unwrap(),
        base: Month::new(2006, 3).unwrap(),
    };
    (index, given)
}

/// Simulation `sim` of `cells` as the future cells of a fit with `origins`.
fn cells_of(cells: &PredictiveDistribution, origins: &[Period], sim: usize) -> FutureCells {
    let row = cells.row(sim).unwrap();
    let cells = cells
        .components()
        .iter()
        .zip(row)
        .map(|(key, &value)| {
            let (KeyValue::Period(period), KeyValue::Int(age)) = (&key[0], &key[1]) else {
                panic!("an origin and an age: {key:?}");
            };
            FutureCell {
                origin: origins.iter().position(|p| p == period).unwrap(),
                age: *age as u32,
                valuation: period.start().add_months(age - 1),
                value,
            }
        })
        .collect();
    FutureCells {
        origins: origins.to_vec(),
        cells,
    }
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1.0)
}

/// The three checks of `reinflate_draws` on one bootstrap's cells.
fn check_reinflation(what: &str, run: &Run, cl: &ChainLadderFit) {
    let cells = run.cells.as_ref().unwrap();
    let n_sims = cells.n_sims();
    let rate = 0.05;
    let (index, given) = genins_index(rate);
    // (a) Paths at the index's own future rate (eleven years: 2011 to
    // 2020, the tail paid one period past the oldest age).
    let path: Vec<f64> = (1..=11).map(|k| given[9] * (1.0 + rate).powi(k)).collect();
    let paths = IndexPaths::new(Period::year(2011), n_sims, path.repeat(n_sims)).unwrap();
    let with = index.reinflate_draws(cells, Some(&paths)).unwrap();
    let without = index.reinflate_draws(cells, None).unwrap();
    assert_eq!(with.components(), cells.components());
    for (a, b) in with.draw_matrix().iter().zip(without.draw_matrix()) {
        assert!(close(*a, *b, 1e-12), "{what}: {a} {b}");
    }
    // (b) A unit index gives back the reserves by origin.
    let unit = Indexation {
        index: CalendarIndex::new(Period::year(2001), vec![1.0; 10])
            .unwrap()
            .with_future_rate(0.0)
            .unwrap(),
        base: Month::new(2006, 3).unwrap(),
    };
    let by_origin = unit
        .reinflate_draws(cells, None)
        .unwrap()
        .aggregate(&["origin"])
        .unwrap();
    for (j, key) in by_origin.components().iter().enumerate() {
        let o = run.reserves.component_index(key).unwrap();
        for sim in 0..n_sims {
            let (got, want) = (by_origin.row(sim).unwrap()[j], run.reserves.row(sim).unwrap()[o]);
            assert!(close(got, want, 1e-9), "{what}: {key:?} {got} {want}");
        }
    }
    // (c) Each simulation is `reinflate` of its own cells.
    for sim in [0, 1, 17, n_sims / 2, n_sims - 1] {
        let want = index.reinflate(&cells_of(cells, &cl.origins, sim)).unwrap();
        assert_eq!(want.cells.cells.len(), cells.n_components());
        for (got, cell) in without.row(sim).unwrap().iter().zip(&want.cells.cells) {
            assert!(close(*got, cell.value, 1e-12), "{what}: sim {sim}");
        }
    }
}

#[test]
fn reinflate_draws_takes_the_bootstraps_cells() {
    let genins = triangle("genins");
    let independent = SegmentDependence::Independent;
    for (name, tail) in [("no tail", Tail::default()), ("tail", constant_tail(1.03))] {
        let (run, cl) = odp_one(&odp(tail, independent.clone(), 300), &genins, true);
        check_reinflation(&format!("odp, {name}"), &run, &cl);
        let (run, cl) = mack_one(&mack(tail, independent.clone(), 300), &genins, true);
        check_reinflation(&format!("mack, {name}"), &run, &cl);
    }
}

/// A 3 by 3 annual triangle with every cell observed.
fn square() -> Triangle {
    let origin = [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2022].map(Month::january);
    let ages = [12, 24, 36, 12, 24, 36, 12, 24, 36];
    let values = [
        100.0, 150.0, 165.0, 110.0, 170.0, 182.0, 120.0, 175.0, 196.0,
    ];
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("values", &values)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

#[test]
fn a_fully_developed_triangle_has_no_cells_without_a_tail() {
    let tri = square();
    let independent = SegmentDependence::Independent;
    let boot = odp(Tail::default(), independent.clone(), 50);
    assert!(boot.fit(&tri, "values").is_ok());
    assert!(matches!(
        boot.fit_with_cells(&tri, "values"),
        Err(Error::Bootstrap(_))
    ));
    assert!(matches!(
        boot.fit_segments_with_cells(&tri, "values"),
        Err(Error::Bootstrap(_))
    ));
    let boot = mack(Tail::default(), independent.clone(), 50);
    assert!(boot.fit(&tri, "values").is_ok());
    assert!(matches!(
        boot.fit_with_cells(&tri, "values"),
        Err(Error::Bootstrap(_))
    ));
    assert!(matches!(
        boot.fit_segments_with_cells(&tri, "values"),
        Err(Error::Bootstrap(_))
    ));
    // With a tail each origin has its tail cell, at 48 months.
    let (run, _) = odp_one(&odp(constant_tail(1.05), independent.clone(), 50), &tri, true);
    let cells = run.cells.unwrap();
    assert_eq!(cells.n_components(), 3);
    assert!(cells.components().iter().all(|k| k[1] == KeyValue::Int(48)));
    let (run, _) = mack_one(&mack(constant_tail(1.05), independent, 50), &tri, true);
    assert_eq!(run.cells.unwrap().n_components(), 3);
}

/// With a tail, every origin's last cell is at the oldest age plus one
/// development period, and the cells are the chain ladder's grid after
/// each origin's latest age, then that one.
fn check_tail_cells(what: &str, cells: &PredictiveDistribution, cl: &ChainLadderFit) {
    let ages = &cl.development.development;
    let tail_age = ages[ages.len() - 1] + (ages[ages.len() - 1] - ages[ages.len() - 2]);
    let mut want = Vec::new();
    for (o, &origin) in cl.origins.iter().enumerate() {
        for &age in ages[cl.latest_position[o] + 1..].iter().chain([&tail_age]) {
            want.push(vec![KeyValue::Period(origin), KeyValue::Int(age.into())]);
        }
    }
    assert_eq!(cells.components(), want, "{what}");
    assert_eq!(tail_age, 132, "{what}");
}

#[test]
fn a_tail_adds_one_cell_per_origin_past_the_oldest_age() {
    let raa = triangle("raa");
    let independent = SegmentDependence::Independent;
    let (run, cl) = odp_one(&odp(constant_tail(1.05), independent.clone(), 200), &raa, true);
    check_tail_cells("odp", run.cells.as_ref().unwrap(), &cl);
    let (run, cl) = mack_one(&mack(constant_tail(1.05), independent, 200), &raa, true);
    check_tail_cells("mack", run.cells.as_ref().unwrap(), &cl);
    // The oldest origin's only cell is its tail step, its whole reserve.
    let cells = run.cells.as_ref().unwrap();
    for sim in 0..cells.n_sims() {
        let (cell, reserve) = (cells.row(sim).unwrap()[0], run.reserves.row(sim).unwrap()[0]);
        assert!(close(cell, reserve, 1e-12));
    }
}

/// Segments of their own shapes, under the `lob` values in `parts`: each
/// triangle's observed values as they are.
fn stack(parts: &[(&str, &Triangle)]) -> Triangle {
    let (mut keys, mut origin, mut ages, mut values) = (vec![], vec![], vec![], vec![]);
    for &(name, tri) in parts {
        let long = tri.to_long();
        let v = &long.values.iter().find(|(c, _)| c == "values").unwrap().1;
        keys.extend(std::iter::repeat_n(name, v.len()));
        origin.extend(long.origin);
        ages.extend(long.development);
        values.extend(v.iter().copied());
    }
    Triangle::from_long(&Long {
        keys: &[("lob", &keys)],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("values", &values)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

/// A 4 by 4 annual triangle with every cell observed.
fn square4() -> Triangle {
    let origin = [2016, 2017, 2018, 2019].map(Month::january).repeat(4);
    let ages: Vec<u32> = [12, 24, 36, 48].iter().flat_map(|&a| [a; 4]).collect();
    let values = [
        100.0, 110.0, 120.0, 105.0, // 12 months
        150.0, 170.0, 175.0, 160.0, // 24
        165.0, 182.0, 196.0, 176.0, // 36
        170.0, 188.0, 199.0, 181.0, // 48
    ];
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("values", &values)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

/// A fully developed square beside RAA, in both orders: each segment's
/// cells sit in their own columns (a square with no tail has none, a
/// segment of no cells beside one of 45), move with their own reserves
/// under a rank correlation, and `segment` gives the square no cells.
#[test]
fn segments_of_different_shapes_keep_their_own_cells() {
    let (raa, sq) = (triangle("raa"), square4());
    let rank = SegmentDependence::RankCorrelation {
        spearman: vec![1.0, 0.6, 0.6, 1.0],
    };
    for (order, tri) in [
        ("square first", stack(&[("A", &sq), ("B", &raa)])),
        ("raa first", stack(&[("B", &raa), ("A", &sq)])),
    ] {
        for dependence in [SegmentDependence::Independent, rank.clone()] {
            for (name, tail) in [("no tail", Tail::default()), ("tail", constant_tail(1.05))] {
                let what = format!("{order}, {dependence:?}, {name}");
                let n_square = if name == "tail" { 4 } else { 0 };
                let boot = odp(tail, dependence.clone(), 300);
                let without = odp_segments(&boot, &tri, false);
                let with = odp_segments(&boot, &tri, true);
                check_identical(&format!("odp, {what}"), &without, &with);
                let cells = with.cells.as_ref().unwrap();
                check_sums(&format!("odp, {what}"), &with.reserves, cells, true);
                let in_a = cells
                    .components()
                    .iter()
                    .filter(|k| k[0] == KeyValue::from("A"))
                    .count();
                assert_eq!(in_a, n_square, "odp, {what}");
                let fit = boot.fit_segments_with_cells(&tri, "values").unwrap();
                let a = fit.segment(&[("lob", "A")]).unwrap();
                assert_eq!(a.cells.is_some(), n_square > 0, "odp, {what}");
                if let Some(cells) = &a.cells {
                    check_sums(&format!("odp A, {what}"), &a.reserves, cells, true);
                }
                let b = fit.segment(&[("lob", "B")]).unwrap();
                check_sums(&format!("odp B, {what}"), &b.reserves, b.cells.as_ref().unwrap(), true);

                let boot = mack(tail, dependence.clone(), 300);
                let without = mack_segments(&boot, &tri, false);
                let with = mack_segments(&boot, &tri, true);
                check_identical(&format!("mack, {what}"), &without, &with);
                check_sums(&format!("mack, {what}"), &with.reserves, with.cells.as_ref().unwrap(), false);
                let fit = boot.fit_segments_with_cells(&tri, "values").unwrap();
                let a = fit.segment(&[("lob", "A")]).unwrap();
                assert_eq!(a.cells.is_some(), n_square > 0, "mack, {what}");
                let b = fit.segment(&[("lob", "B")]).unwrap();
                check_sums(&format!("mack B, {what}"), &b.reserves, b.cells.as_ref().unwrap(), false);
            }
        }
    }
}

/// The segments' cells go through `reinflate_draws` by their dimension
/// names: paths at the index's own rate give the deterministic
/// re-inflation, and a unit index gives back each segment's reserves by
/// origin.
#[test]
fn reinflate_draws_takes_the_segments_cells() {
    let tri = twins(&triangle("genins"));
    let rank = SegmentDependence::RankCorrelation {
        spearman: vec![1.0, 0.5, 0.5, 1.0],
    };
    let odp_run = odp_segments(&odp(constant_tail(1.03), rank.clone(), 200), &tri, true);
    let mack_run = mack_segments(&mack(Tail::default(), rank, 200), &tri, true);
    for (what, run) in [("odp", odp_run), ("mack", mack_run)] {
        let cells = run.cells.as_ref().unwrap();
        assert_eq!(cells.dims(), ["lob", "origin", "development"]);
        let n_sims = cells.n_sims();
        let rate = 0.05;
        let (index, given) = genins_index(rate);
        let path: Vec<f64> = (1..=11).map(|k| given[9] * (1.0 + rate).powi(k)).collect();
        let paths = IndexPaths::new(Period::year(2011), n_sims, path.repeat(n_sims)).unwrap();
        let with = index.reinflate_draws(cells, Some(&paths)).unwrap();
        let without = index.reinflate_draws(cells, None).unwrap();
        assert_eq!(with.components(), cells.components(), "{what}");
        for (a, b) in with.draw_matrix().iter().zip(without.draw_matrix()) {
            assert!(close(*a, *b, 1e-12), "{what}: {a} {b}");
        }
        let unit = Indexation {
            index: CalendarIndex::new(Period::year(2001), vec![1.0; 10])
                .unwrap()
                .with_future_rate(0.0)
                .unwrap(),
            base: Month::new(2006, 3).unwrap(),
        };
        let by_origin = unit
            .reinflate_draws(cells, None)
            .unwrap()
            .aggregate(&["lob", "origin"])
            .unwrap();
        // Without a tail each segment's oldest origin has no cell and a
        // reserve of 0; with one, every origin has cells.
        for (o, key) in run.reserves.components().iter().enumerate() {
            if by_origin.component_index(key).is_none() {
                assert!(what == "mack", "{what}: no cell for {key:?}");
                assert!((0..n_sims).all(|s| run.reserves.row(s).unwrap()[o] == 0.0));
            }
        }
        for (j, key) in by_origin.components().iter().enumerate() {
            let o = run.reserves.component_index(key).unwrap();
            for sim in 0..n_sims {
                let (got, want) = (by_origin.row(sim).unwrap()[j], run.reserves.row(sim).unwrap()[o]);
                assert!(close(got, want, 1e-9), "{what}: {key:?} {got} {want}");
            }
        }
    }
}

fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(f)
}

/// The cells, like the reserves, are the same bits on one thread and on
/// four, under a rank correlation with a tail.
#[test]
fn the_cells_do_not_depend_on_the_number_of_threads() {
    let tri = twins(&triangle("raa"));
    let rank = SegmentDependence::RankCorrelation {
        spearman: vec![1.0, 0.7, 0.7, 1.0],
    };
    let odp_boot = odp(constant_tail(1.05), rank.clone(), 600);
    let mack_boot = mack(constant_tail(1.05), rank, 600);
    let runs = |threads| {
        in_pool(threads, || {
            (
                odp_segments(&odp_boot, &tri, true),
                mack_segments(&mack_boot, &tri, true),
            )
        })
    };
    let (one, four) = (runs(1), runs(4));
    for (what, a, b) in [("odp", &one.0, &four.0), ("mack", &one.1, &four.1)] {
        assert_eq!(bits(&a.reserves), bits(&b.reserves), "{what}");
        let (ca, cb) = (a.cells.as_ref().unwrap(), b.cells.as_ref().unwrap());
        assert_eq!(ca.components(), cb.components(), "{what}");
        assert_eq!(bits(ca), bits(cb), "{what}");
    }
}

/// Each cell's share of its origin's mean reserve, `mean(cell) /
/// mean(reserve)`, against the chain ladder's share of the origin's
/// reserve, within four standard errors of the ratio (delta method, and
/// 1e-9 for cells with no spread). An origin's overall offset from the
/// chain ladder cancels, so this needs no allowance for it: it catches a
/// cell moved between ages of one origin, which the sums cannot.
fn check_shares(what: &str, cells: &PredictiveDistribution, reserves: &PredictiveDistribution, cl: &ChainLadderFit) {
    let want = FutureCells::from_chain_ladder(cl);
    let n_sims = cells.n_sims();
    let n = n_sims as f64;
    let column = |pd: &PredictiveDistribution, j: usize| -> Vec<f64> {
        (0..n_sims).map(|s| pd.row(s).unwrap()[j]).collect()
    };
    let mut worst = 0.0f64;
    for (o, &origin) in want.origins.iter().enumerate() {
        let of_origin: Vec<&FutureCell> = want.cells.iter().filter(|c| c.origin == o).collect();
        let total: f64 = of_origin.iter().map(|c| c.value).sum();
        if of_origin.is_empty() || total == 0.0 {
            continue;
        }
        let r = column(reserves, reserves.component_index(&vec![KeyValue::Period(origin)]).unwrap());
        let mean_r = r.iter().sum::<f64>() / n;
        for cell in of_origin {
            let key = vec![KeyValue::Period(origin), KeyValue::Int(cell.age.into())];
            let x = column(cells, cells.component_index(&key).unwrap());
            let share = x.iter().sum::<f64>() / n / mean_r;
            let z: Vec<f64> = x.iter().zip(&r).map(|(x, r)| x - share * r).collect();
            let mean_z = z.iter().sum::<f64>() / n;
            let sd_z = (z.iter().map(|v| (v - mean_z).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
            let se = sd_z / n.sqrt() / mean_r.abs();
            let gap = (share - cell.value / total).abs();
            worst = worst.max(gap / se);
            assert!(
                gap <= 4.0 * se + 1e-9,
                "{what}: {key:?}: share {share}, chain ladder {}, se {se}",
                cell.value / total
            );
        }
    }
    println!("{what}: shares, worst {worst:.2} standard errors");
}
