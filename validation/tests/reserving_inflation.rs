//! Indexation of a triangle by a calendar index and re-inflation of its
//! projection (`reserving-inflation.md`, PR 1, #979).
//!
//! No package restates a triangle by a calendar index in the same terms,
//! so the checks are Taylor's (1977) printed re-inflation of Example 1
//! and exact identities on GenIns (validation target 1).

use prospicio_reserving::{
    CalendarIndex, ChainLadder, FutureCell, FutureCells, Grain, Indexation, Month, Period,
};
use prospicio_validation::triangle;

/// Taylor (1977), Example 1 (a motor account), section 8: the separation
/// fit's `r_j` and its last index `lambda_3 = 113.9`, projected at 10% a
/// year. The future cell of origin `i` at development year `j` is `r_j
/// lambda_{i+j}`; at the base level (`lambda_3`) it is `r_j 113.9`, and
/// re-inflating it at 10% a year from the last period must give the cells
/// Taylor prints (p. 229): 5.3; 10.8, 5.8; 36.1, 11.9, 6.4, to his
/// rounding (10.8 is 10.85 here, so 0.06).
#[test]
fn reinflation_reproduces_taylor_1977_example_1() {
    let r = [0.5835, 0.2878, 0.0866, 0.0421];
    let lambda = [86.4, 98.9, 102.0, 113.9];
    let origins: Vec<Period> = (2001..=2004).map(Period::year).collect();
    let mut cells = Vec::new();
    for (i, origin) in origins.iter().enumerate() {
        for (j, rj) in r.iter().enumerate().skip(4 - i) {
            let age = 12 * (j as u32 + 1);
            cells.push(FutureCell {
                origin: i,
                age,
                valuation: origin.start().add_months(age as i64 - 1),
                value: rj * lambda[3],
            });
        }
    }
    let at_last = Indexation {
        index: CalendarIndex::new(Period::year(2001), lambda.to_vec())
            .unwrap()
            .with_future_rate(0.10)
            .unwrap(),
        base: Month::january(2004),
    };
    let out = at_last
        .reinflate(&FutureCells { origins, cells })
        .unwrap();
    let printed = [(1, 3, 5.3), (2, 2, 10.8), (2, 3, 5.8), (3, 1, 36.1), (3, 2, 11.9), (3, 3, 6.4)];
    assert_eq!(out.cells.cells.len(), printed.len());
    for (cell, (i, j, want)) in out.cells.cells.iter().zip(printed) {
        assert_eq!((cell.origin, cell.age), (i, 12 * (j as u32 + 1)));
        assert!((cell.value - want).abs() <= 0.06, "{i} {j}: {} vs {want}", cell.value);
    }
}

/// GenIns (Taylor-Ashe): a unit index leaves the chain ladder's reserves
/// as they are, restating then inflating is the identity, and the
/// re-inflated reserve of a deflated triangle at a constant rate is its
/// base reserve with each future cell grown by its years past the base.
#[test]
fn genins_identities() {
    let tri = triangle("genins");
    let fit = ChainLadder::default().fit(&tri, "values").unwrap();
    let cells = FutureCells::from_chain_ladder(&fit);
    let unit = Indexation {
        index: CalendarIndex::new(Period::year(2001), vec![1.0])
            .unwrap()
            .with_future_rate(0.0)
            .unwrap(),
        base: Month::january(2010),
    };
    let same = unit.reinflate(&cells).unwrap();
    for (got, want) in same.reserves.iter().zip(fit.reserves()) {
        assert!((got - want).abs() <= 1e-9 * want.abs());
    }

    let cpi = Indexation {
        index: CalendarIndex::new(
            Period::containing(Month::new(2001, 1).unwrap(), Grain::Quarter),
            (0..40).map(|q| 100.0 * 1.01_f64.powi(q)).collect(),
        )
        .unwrap()
        .with_future_rate(0.04)
        .unwrap(),
        base: Month::new(2010, 12).unwrap(),
    };
    let restated = cpi.restate(&tri, "values").unwrap();
    let back = cpi.inflate(&restated, "values").unwrap();
    for o in 0..10 {
        for d in 0..10 - o {
            let (a, b) = (tri.get(0, 0, o, d).unwrap(), back.get(0, 0, o, d).unwrap());
            assert!((a - b).abs() <= 1e-12 * a.abs(), "{o} {d}");
        }
    }
    // An annual cell is paid in the quarter of its valuation month, the
    // fourth: the first cell, valued 2001-12, by 1.01^3 against 1.01^39.
    let want = tri.get(0, 0, 0, 0).unwrap() * 1.01_f64.powi(36);
    assert!((restated.get(0, 0, 0, 0).unwrap() - want).abs() <= 1e-9 * want);

    let fit = ChainLadder::default().fit(&restated, "values").unwrap();
    let out = cpi.reinflate(&FutureCells::from_chain_ladder(&fit)).unwrap();
    for (base, cell) in FutureCells::from_chain_ladder(&fit)
        .cells
        .iter()
        .zip(&out.cells.cells)
    {
        let years = cell.valuation.months_since(Month::new(2010, 12).unwrap()) as f64 / 12.0;
        let want = base.value * 1.04_f64.powf(years);
        assert!((cell.value - want).abs() <= 1e-9 * want.abs());
    }
    assert!(out.total_reserve() > out.total_base_reserve());
}
