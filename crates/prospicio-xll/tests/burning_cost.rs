//! `PRO.BURNINGCOST`, reached through the catalogue as Excel reaches it,
//! against Clark, *Basics of Reinsurance Pricing* (CAS study note, 2014),
//! the per-risk example of section A.4.a: fourteen losses, 400,000 xs
//! 100,000, trended at 4% to 7/1/97, eight accident years with their
//! on-level premiums and excess development factors. The note's figures:
//! each year's layer loss to the dollar, its developed loss and loss cost,
//! and the 12.4% total.

mod common;

use common::{n, num, t, text};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn column(v: &[f64]) -> Grid {
    Grid::from_rows(v.iter().map(|&x| vec![Value::Number(x)]).collect())
}

/// Excel serials of the accident dates.
fn serial(y: i32, m: u32, d: u32) -> f64 {
    let (y, m) = if m <= 2 {
        (i64::from(y) - 1, m + 12)
    } else {
        (i64::from(y), m)
    };
    let days =
        |y: i64, m: i64, d: i64| 365 * y + y / 4 - y / 100 + y / 400 + (153 * (m - 3) + 2) / 5 + d;
    // 1 January 1900 is serial 1, and Excel counts a 29 February 1900.
    (days(y, i64::from(m), i64::from(d)) - days(1899, 13, 1) + 2) as f64
}

/// (accident date, amount, the note's trend factor).
const LOSSES: [((i32, u32, u32), f64, f64); 14] = [
    ((1988, 9, 20), 240_946.0, 1.411),
    ((1988, 10, 11), 821_499.0, 1.408),
    ((1989, 3, 15), 158_129.0, 1.385),
    ((1990, 6, 21), 114_051.0, 1.317),
    ((1990, 10, 24), 78_043.0, 1.300),
    ((1991, 1, 10), 162_533.0, 1.289),
    ((1992, 2, 23), 324_298.0, 1.234),
    ((1992, 4, 30), 100_549.0, 1.225),
    ((1992, 9, 22), 75_476.0, 1.206),
    ((1993, 1, 1), 171_885.0, 1.193),
    ((1993, 5, 18), 94_218.0, 1.175),
    ((1993, 8, 1), 170_297.0, 1.166),
    ((1994, 8, 15), 87_133.0, 1.119),
    ((1995, 7, 12), 771_249.0, 1.080),
];

/// (year, premium, layer loss, LDF, developed, loss cost), as printed.
const YEARS: [(f64, f64, f64, f64, f64, f64); 8] = [
    (1988.0, 1_422_554.0, 639_975.0, 1.000, 639_975.0, 0.450),
    (1989.0, 1_823_103.0, 119_009.0, 1.000, 119_009.0, 0.065),
    (1990.0, 2_054_034.0, 51_661.0, 1.000, 51_661.0, 0.025),
    (1991.0, 2_147_147.0, 109_505.0, 1.000, 109_505.0, 0.051),
    (1992.0, 2_151_541.0, 323_357.0, 1.010, 326_591.0, 0.152),
    (1993.0, 2_159_198.0, 214_331.0, 1.050, 225_048.0, 0.104),
    (1994.0, 2_167_158.0, 0.0, 1.150, 0.0, 0.000),
    (1995.0, 2_187_654.0, 400_000.0, 1.300, 520_000.0, 0.238),
];

fn args(trend: Grid, trend_to: Grid) -> Vec<Grid> {
    let dates: Vec<f64> = LOSSES
        .iter()
        .map(|&((y, m, d), ..)| serial(y, m, d))
        .collect();
    vec![
        column(&dates),
        column(&LOSSES.map(|l| l.1)),
        n(400_000.0),
        n(100_000.0),
        column(&YEARS.map(|y| y.0)),
        column(&YEARS.map(|y| y.1)),
        column(&YEARS.map(|y| y.3)),
        trend,
        trend_to,
    ]
}

#[test]
fn matches_clark() {
    assert_eq!(serial(1997, 7, 1), 35_612.0);
    // The note's printed trend factors, one per loss.
    let b = call(
        "pro_burningcost",
        args(column(&LOSSES.map(|l| l.2)), Grid::missing()),
    );
    assert_eq!((b.rows, b.columns), (11, 7), "{:?}", b.as_scalar());
    assert_eq!(b.get(0, 4), &Value::from("400,000 xs 100,000 layer loss"));
    for (k, &(year, _, layer, _, developed, cost)) in YEARS.iter().enumerate() {
        let r = k + 1;
        assert_eq!(num(b.get(r, 0)), year);
        assert!((num(b.get(r, 4)) - layer).abs() <= 1.5, "{year}");
        assert!((num(b.get(r, 5)) - developed).abs() <= 2.0, "{year}");
        assert!((num(b.get(r, 6)) - cost).abs() <= 0.0005 + 1e-12, "{year}");
    }
    assert_eq!(b.get(9, 0), &Value::from("Total"));
    assert!((num(b.get(9, 1)) - 16_112_389.0).abs() < 0.5);
    assert_eq!(num(b.get(9, 2)), 14.0); // every loss goes through, two cede nothing
    assert!((num(b.get(9, 5)) - 1_991_789.0).abs() <= 6.0);
    assert!((num(b.get(9, 6)) - 0.124).abs() <= 0.0005);
    let mean: f64 = (1..=8).map(|r| num(b.get(r, 6))).sum::<f64>() / 8.0;
    assert!((num(b.get(10, 6)) - mean).abs() < 1e-15);

    // At 4% to 7/1/97 (an Excel date) the trend factors agree with the
    // note's to its three decimals, so the total loss cost does too.
    let r = call("pro_burningcost", args(n(0.04), n(35_612.0)));
    assert!(
        (num(r.get(9, 6)) - 0.124).abs() <= 0.0005,
        "{:?}",
        r.get(9, 6)
    );
}

#[test]
fn layers_terms_and_errors() {
    // Two layers in a row, the second unlimited, and an aggregate limit.
    let mut a = args(column(&LOSSES.map(|l| l.2)), Grid::missing());
    a[2] = Grid::from_rows(vec![vec![
        Value::Number(400_000.0),
        Value::from("unlimited"),
    ]]);
    a[3] = Grid::from_rows(vec![vec![
        Value::Number(100_000.0),
        Value::Number(500_000.0),
    ]]);
    let b = call("pro_burningcost", a.clone());
    assert_eq!(b.columns, 10, "{:?}", b.as_scalar());
    assert_eq!(b.get(0, 7), &Value::from("unlimited xs 500,000 layer loss"));
    // 1988: 1,156,671 trended, 656,671 above 500,000.
    assert!((num(b.get(1, 7)) - (821_499.0 * 1.408 - 500_000.0)).abs() < 1e-6);
    a.extend([Grid::missing(), Grid::missing(), n(500_000.0)]);
    let capped = call("pro_burningcost", a);
    assert!((num(capped.get(1, 4)) - 500_000.0).abs() < 1e-9); // 639,975 capped
    assert!((num(capped.get(1, 7)) - 500_000.0).abs() < 1e-9);

    let e = |a: Vec<Grid>| text(&call("pro_burningcost", a)).to_owned();
    let mut bad = args(n(0.04), Grid::missing());
    assert!(e(bad.clone()).contains("trend_to"));
    bad[1] = column(&[1.0, 2.0]);
    assert!(e(bad).contains("one value per accident date"));
    let mut bad = args(Grid::missing(), Grid::missing());
    bad[4] = column(&[1988.5]);
    bad[5] = n(1.0);
    bad[6] = Grid::missing();
    assert!(e(bad).contains("whole accident years"));
    let mut bad = args(t("x"), Grid::missing());
    bad[7] = column(&[1.0, 1.0]);
    assert!(e(bad).contains("one per loss"));
}
