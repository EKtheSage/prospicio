//! An event loss table as a compound Poisson severity on a grid against
//! `aggregate` 2.1.0's `agg NAME elt [...]` (#781,
//! `validation/scripts/aggregate_elt_grid.py`): a table of means placed by
//! the linear scatter, with and without an occurrence layer, and a table
//! with secondary uncertainty bucketed by the nearest node, against
//! `aggregate` (through eltq) and exactly against SciPy's beta.

use prospicio_aggregate::cat::{EltBucketing, EventLossTable};
use prospicio_aggregate::{Layer, Tower};
use prospicio_prob::Grid;
use prospicio_validation::{check, reference};

const STEP: f64 = 10.0;
const POINTS: usize = 1024;
/// rate, mean (an ELT3: no secondary uncertainty, exposure unused).
const MEANS: &[[f64; 2]] = &[
    [0.5, 130.0],
    [0.25, 270.5],
    [0.05, 1234.5],
    [0.01, 2999.9],
    [0.002, 7777.7],
];
/// rate, mean, max, sdi, sdc.
const SECONDARY: &[[f64; 5]] = &[
    [0.01, 800.0, 2000.0, 200.0, 100.0],
    [0.05, 250.0, 500.0, 0.0, 0.0],
    [0.002, 1500.0, 5000.0, 300.0, 300.0],
    [0.2, 40.0, 300.0, 30.0, 10.0],
    [0.0005, 6000.0, 20000.0, 3000.0, 1000.0],
];

fn table(rows: &[[f64; 5]]) -> EventLossTable {
    let col = |j: usize| rows.iter().map(|r| r[j]).collect::<Vec<_>>();
    EventLossTable::new(&col(0), &col(1), &col(3), &col(4), &col(2)).unwrap()
}

fn means() -> EventLossTable {
    let rows: Vec<[f64; 5]> = MEANS
        .iter()
        .map(|r| [r[0], r[1], 2.0 * r[1], 0.0, 0.0])
        .collect();
    table(&rows)
}

#[test]
fn elt_grid_matches_aggregate() {
    let means = means();
    let secondary = table(SECONDARY);
    // `aggregate` leaves the annual mass above the grid out; `fft` lumps
    // it onto the last node, so it is taken off there.
    let unlumped = |grid: Grid, tail: f64| -> Vec<f64> {
        let mut p = grid.probs().to_vec();
        *p.last_mut().unwrap() -= tail;
        p
    };
    let grids = |elt: &EventLossTable, rule| -> (Vec<f64>, Vec<f64>) {
        let (sev, _) = elt.severity_grid(STEP, POINTS, rule).unwrap();
        let (agg, report, _) = elt.aggregate_grid(STEP, POINTS, rule).unwrap();
        assert!(report.aliasing_error < 1e-6);
        (sev.probs().to_vec(), unlumped(agg, report.tail_mass))
    };
    let (m_sev, m_agg) = grids(&means, EltBucketing::MeansLinear);
    let (s_sev, s_agg) = grids(&secondary, EltBucketing::Nearest);
    let (sev, _) = means
        .severity_grid(STEP, POINTS, EltBucketing::MeansLinear)
        .unwrap();
    let tower = Tower::new(vec![Layer::xol("1000 xs 500", 1000.0, 500.0).unwrap()]).unwrap();
    let r = tower.on_grid(&means.count(), &sev, POINTS).unwrap();
    let ceded = unlumped(r.ceded[0].clone(), r.ceded_reports[0].tail_mass);
    check(&reference("elt_grid_aggregate.csv"), |c| {
        let k = c.number("arg")? as usize;
        let grid = match (c.get("params").strip_prefix("table=")?, c.get("quantity")) {
            ("means", "sev") => &m_sev,
            ("means", "agg") => &m_agg,
            ("means_ceded", "agg") => &ceded,
            ("secondary" | "secondary_exact", "sev") => &s_sev,
            ("secondary", "agg") => &s_agg,
            _ => return None,
        };
        grid.get(k).copied()
    });
}
