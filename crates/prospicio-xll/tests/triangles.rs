//! The triangle helpers on RAA, reached through the catalogue as Excel
//! reaches them: incremental and back, link ratios and the latest
//! diagonal, against the values worked out from RAA itself.

mod common;

use common::{RAA, num, raa, text};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

#[test]
fn incremental_and_back() {
    let inc = call("pro_tri_incremental", vec![raa()]);
    assert_eq!((inc.rows, inc.columns), (10, 10));
    for (r, row) in RAA.iter().enumerate() {
        for c in 0..10 {
            let want = row.get(c).map(|&x| if c == 0 { x } else { x - row[c - 1] });
            match want {
                Some(w) => assert_eq!(num(inc.get(r, c)), w, "{r} {c}"),
                None => assert_eq!(inc.get(r, c), &Value::Empty),
            }
        }
    }
    // RAA has a negative increment (1982, 72 to 84 months).
    assert_eq!(num(inc.get(1, 6)), 15_496.0 - 15_599.0);
    assert_eq!(call("pro_tri_cumulative", vec![inc]), raa());
}

#[test]
fn link_ratios_and_latest() {
    let lr = call("pro_tri_linkratios", vec![raa()]);
    assert_eq!(
        (lr.rows, lr.columns),
        (10, 9),
        "one column fewer than the ages"
    );
    assert_eq!(num(lr.get(0, 0)), 8269.0 / 5012.0);
    assert_eq!(num(lr.get(8, 0)), 5395.0 / 3133.0);
    assert_eq!(
        lr.get(9, 0),
        &Value::Empty,
        "the youngest origin has no link"
    );
    // The volume-weighted average of a column is the chain ladder's factor.
    let (num_sum, den_sum) = RAA
        .iter()
        .filter(|r| r.len() > 1)
        .fold((0.0, 0.0), |(a, b), r| (a + r[1], b + r[0]));
    let ldf = call("pro_cl_ldf", vec![raa()]);
    assert!((num(&ldf.cells[0]) - num_sum / den_sum).abs() < 1e-12);

    let latest = call("pro_tri_latest", vec![raa()]);
    assert_eq!((latest.rows, latest.columns), (10, 1));
    let want: Vec<f64> = RAA.iter().map(|r| *r.last().unwrap()).collect();
    assert_eq!(latest.cells.iter().map(num).collect::<Vec<_>>(), want);
    // The total of the latest diagonal is PRO.CL's total latest.
    let cl = call("pro_cl", vec![raa()]);
    assert_eq!(num(cl.get(11, 1)), want.iter().sum::<f64>());
}

#[test]
fn a_bad_triangle_is_text() {
    let one = Grid::from_rows(vec![vec![Value::Number(1.0)]]);
    assert!(text(&call("pro_tri_latest", vec![one])).contains("two development columns"));
}
