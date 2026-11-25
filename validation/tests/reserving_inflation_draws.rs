//! Stochastic re-inflation of simulated future cells
//! (`Indexation::reinflate_draws`, `reserving-inflation.md` PR 8, decision
//! 11, #979).
//!
//! No package re-inflates simulated cells by index paths, so the checks
//! are exact identities on GenIns: with no paths each simulation equals
//! `reinflate` of its own cells, and with one path per simulation each
//! simulation equals `reinflate` by that simulation's index.

use prospicio_prob::{KeyValue, PredictiveDistribution};
use prospicio_reserving::{
    CalendarIndex, FutureCell, FutureCells, Indexation, Month, OdpGlm, Period,
};
use prospicio_validation::triangle;

/// GenIns' ODP GLM draws by cell, and the index of the deflated history:
/// 1.0 in 2001 rising 4% a year to the valuation in 2010.
fn setup() -> (PredictiveDistribution, Indexation) {
    let fit = OdpGlm::default().fit(&triangle("genins"), "values").unwrap();
    let draws = fit.predict_distribution(500, 11).unwrap();
    let history: Vec<f64> = (0..10).map(|k| 1.04_f64.powi(k)).collect();
    let index = CalendarIndex::new(Period::year(2001), history)
        .unwrap()
        .with_future_rate(0.04)
        .unwrap();
    (draws, Indexation { index, base: Month::january(2010) })
}

/// Simulation `i`'s cells of `draws` as `FutureCells`, origin by origin.
fn cells_of(draws: &PredictiveDistribution, i: usize) -> FutureCells {
    let mut origins: Vec<Period> = Vec::new();
    let mut cells = Vec::new();
    for (key, &value) in draws.components().iter().zip(draws.row(i).unwrap()) {
        let (KeyValue::Period(origin), KeyValue::Int(age)) = (&key[0], &key[1]) else {
            panic!("a cell key {key:?}");
        };
        let o = match origins.iter().position(|p| p == origin) {
            Some(o) => o,
            None => {
                origins.push(*origin);
                origins.len() - 1
            }
        };
        cells.push(FutureCell {
            origin: o,
            age: *age as u32,
            valuation: origin.start().add_months(age - 1),
            value,
        });
    }
    FutureCells { origins, cells }
}

#[test]
fn without_paths_each_simulation_is_the_deterministic_reinflation() {
    let (draws, at_2010) = setup();
    let out = at_2010.reinflate_draws(&draws, &[]).unwrap();
    assert_eq!(out.components(), draws.components());
    assert_eq!(out.n_sims(), draws.n_sims());
    for i in [0, 1, 250, 499] {
        let want = at_2010.reinflate(&cells_of(&draws, i)).unwrap();
        for (got, cell) in out.row(i).unwrap().iter().zip(&want.cells.cells) {
            assert!((got - cell.value).abs() <= 1e-12 * cell.value.abs().max(1.0));
        }
    }
    // Inflation past 2010 only raises the reserve.
    assert!(out.aggregate(&[]).unwrap().row(0).unwrap()[0] > draws.row(0).unwrap().iter().sum());
}

#[test]
fn each_simulation_takes_its_own_path() {
    let (draws, at_2010) = setup();
    let paths: Vec<CalendarIndex> = (0..draws.n_sims())
        .map(|i| {
            let rate = 0.10 * i as f64 / draws.n_sims() as f64;
            at_2010.index.clone().with_future_rate(rate).unwrap()
        })
        .collect();
    let out = at_2010.reinflate_draws(&draws, &paths).unwrap();
    for i in [0, 7, 499] {
        let own = Indexation { index: paths[i].clone(), base: at_2010.base };
        let want = own.reinflate(&cells_of(&draws, i)).unwrap();
        for (got, cell) in out.row(i).unwrap().iter().zip(&want.cells.cells) {
            assert!((got - cell.value).abs() <= 1e-12 * cell.value.abs().max(1.0));
        }
    }
    // A path at no future inflation leaves the 2010 money as it is.
    assert_eq!(out.row(0).unwrap(), draws.row(0).unwrap());
    // The provenance keeps the draws' seed.
    assert_eq!(out.provenance().seed, draws.provenance().seed);
}

#[test]
fn the_paths_are_one_per_simulation() {
    let (draws, at_2010) = setup();
    let two = vec![at_2010.index.clone(), at_2010.index.clone()];
    let err = at_2010.reinflate_draws(&draws, &two).unwrap_err();
    assert!(err.to_string().contains("one per simulation"), "{err}");
    let by_origin = draws.aggregate(&["origin"]).unwrap();
    let err = at_2010.reinflate_draws(&by_origin, &[]).unwrap_err();
    assert!(err.to_string().contains("\"development\" or \"age\""), "{err}");
}
