//! Friedland's exhibits and the weighted selection of ultimates, reached
//! through the catalogue as Excel reaches them: Friedland (2010) chapter
//! 7 Exhibit I, sheets 1 and 2 (U.S. Industry Auto) to the printed
//! digits, chapter 15 Exhibit I, sheet 2 (XYZ Insurer) by origin and by
//! age, and chainladder-python's `VotingChainladder` on GenIns
//! (`validation/reference/reserving_exhibits_python.csv`), its methods'
//! ultimates from `PRO.RES.CL.ULTIMATE`, `PRO.RES.BF` and
//! `PRO.RES.CAPECOD` as a sheet chains them. The same cases as
//! `validation/tests/reserving_exhibits.rs`.

mod common;

use common::{column_of, data_triangle, fails, genins, n, num, ok, reference, row, t};
use prospicio_xll::sheet::{Grid, Value};

fn blank() -> Grid {
    Grid::scalar(Value::Empty)
}

/// Each value equals the printed one at `digits` decimals.
fn printed(what: &str, got: &[f64], book: &[f64], digits: i32) {
    assert_eq!(got.len(), book.len(), "{what}");
    let scale = 10f64.powi(digits);
    for (k, (g, b)) in got.iter().zip(book).enumerate() {
        assert!(
            ((g * scale).round() - b * scale).abs() < 1e-6,
            "{what}[{k}]: {g} against {b}"
        );
    }
}

fn numbers(table: &Grid, r: usize, from: usize, to: usize) -> Vec<f64> {
    (from..to).map(|c| num(table.get(r, c))).collect()
}

/// A column's printed exhibit: its averages, selected factors, tail,
/// factors to ultimate and percents developed.
type Printed = (
    &'static str,
    [[f64; 9]; 6],
    [f64; 9],
    f64,
    [f64; 10],
    [f64; 10],
);

#[test]
fn friedland_development_exhibits() {
    let book: [Printed; 2] = [
        (
            "reported",
            [
                [
                    1.168, 1.058, 1.027, 1.011, 1.004, 1.003, 1.002, 1.001, 1.000,
                ],
                [
                    1.164, 1.056, 1.027, 1.012, 1.005, 1.003, 1.002, 1.001, 1.000,
                ],
                [
                    1.165, 1.057, 1.027, 1.010, 1.004, 1.003, 1.002, 1.001, 1.000,
                ],
                [
                    1.168, 1.058, 1.027, 1.011, 1.004, 1.003, 1.002, 1.001, 1.000,
                ],
                [
                    1.164, 1.056, 1.027, 1.012, 1.005, 1.003, 1.002, 1.001, 1.000,
                ],
                [
                    1.164, 1.057, 1.027, 1.011, 1.004, 1.003, 1.002, 1.001, 1.000,
                ],
            ],
            [
                1.164, 1.056, 1.027, 1.012, 1.005, 1.003, 1.002, 1.001, 1.000,
            ],
            1.000,
            [
                1.292, 1.110, 1.051, 1.023, 1.011, 1.006, 1.003, 1.001, 1.000, 1.000,
            ],
            [77.4, 90.1, 95.1, 97.8, 98.9, 99.4, 99.7, 99.9, 100.0, 100.0],
        ),
        (
            "paid",
            [
                [
                    1.712, 1.189, 1.091, 1.044, 1.019, 1.010, 1.005, 1.002, 1.002,
                ],
                [
                    1.702, 1.186, 1.091, 1.044, 1.019, 1.009, 1.005, 1.002, 1.002,
                ],
                [
                    1.708, 1.188, 1.091, 1.044, 1.019, 1.010, 1.005, 1.002, 1.002,
                ],
                [
                    1.712, 1.189, 1.091, 1.044, 1.019, 1.010, 1.005, 1.002, 1.002,
                ],
                [
                    1.702, 1.186, 1.091, 1.044, 1.019, 1.009, 1.005, 1.002, 1.002,
                ],
                [
                    1.706, 1.188, 1.091, 1.044, 1.019, 1.010, 1.005, 1.002, 1.002,
                ],
            ],
            [
                1.702, 1.186, 1.091, 1.044, 1.019, 1.009, 1.005, 1.002, 1.002,
            ],
            1.002,
            [
                2.390, 1.404, 1.184, 1.085, 1.040, 1.020, 1.011, 1.006, 1.004, 1.002,
            ],
            [41.8, 71.2, 84.5, 92.2, 96.2, 98.0, 98.9, 99.4, 99.6, 99.8],
        ),
    ];
    for (column, averages, selected, tail, cdf, percent) in book {
        let (tri, _) = data_triangle("friedland_us_auto", column);
        let ex = ok(
            "pro_res_dev_exhibit",
            vec![tri.clone(), blank(), n(tail), row(&selected)],
        );
        assert_eq!(ex.rows, 10);
        assert_eq!(*ex.get(0, 1), Value::from("12-24"));
        assert_eq!(*ex.get(0, 10), Value::from("120-Ult"));
        assert_eq!(*ex.get(3, 0), Value::from("Medial latest 5x1"));
        for (k, book) in averages.iter().enumerate() {
            let mut got = numbers(&ex, k + 1, 1, 10);
            if column == "paid" && k == 2 {
                // The book prints 1.010 where the printed claims give
                // 1.0094965, under the rounding edge (see the crate test).
                assert!((got[5] - 1.0095).abs() < 1e-5);
                got[5] = 1.010;
            }
            printed(&format!("{column} average {k}"), &got, book, 3);
            assert_eq!(*ex.get(k + 1, 10), Value::Empty);
        }
        assert_eq!(numbers(&ex, 7, 1, 10), selected);
        assert_eq!(num(ex.get(7, 10)), tail);
        printed(&format!("{column} CDF"), &numbers(&ex, 8, 1, 11), &cdf, 3);
        // The percents divide by the rounded CDFs, given as a pattern of
        // factors to ultimate.
        let rounded = ok(
            "pro_res_dev_exhibit",
            vec![tri, blank(), blank(), row(&cdf), t("cdf")],
        );
        let p: Vec<f64> = numbers(&rounded, 9, 1, 11)
            .iter()
            .map(|x| 100.0 * x)
            .collect();
        printed(&format!("{column} percent"), &p, &percent, 1);
    }
}

/// Chapter 15, Exhibit I, sheet 2 (XYZ Insurer): the methods' printed
/// ultimates and the book's selection, column (11).
fn xyz() -> (Grid, Grid, Grid, Grid) {
    let e = f64::NAN;
    let methods: [[f64; 11]; 7] = [
        [
            15822.0, 25107.0, 37246.0, 38798.0, 48169.0, 44373.0, 70780.0, 71362.0, 53392.0,
            42680.0, 41531.0,
        ],
        [
            15980.0, 25140.0, 37334.0, 39290.0, 47326.0, 44078.0, 71401.0, 68685.0, 51776.0,
            38549.0, 45987.0,
        ],
        [
            15822.0, 25107.0, 37246.0, 38798.0, 48169.0, 44373.0, 70792.0, 71554.0, 53906.0,
            40300.0, 36842.0,
        ],
        [
            15975.0, 25128.0, 37294.0, 39274.0, 47313.0, 45070.0, 71688.0, 77898.0, 56031.0,
            34988.0, 33988.0,
        ],
        [
            15822.0, 25082.0, 37083.0, 38778.0, 48655.0, 46107.0, 76620.0, 80745.0, 64505.0,
            58516.0, 59242.0,
        ],
        [e, e, e, e, e, e, e, e, e, 30512.0, 30140.0],
        [
            e, e, e, 39192.0, 46869.0, 44479.0, 71906.0, 71684.0, 49913.0, 31805.0, 29828.0,
        ],
    ];
    let ultimates = Grid::from_rows(
        (0..11)
            .map(|o| {
                methods
                    .iter()
                    .map(|m| {
                        if m[o].is_nan() {
                            Value::Empty
                        } else {
                            Value::Number(m[o])
                        }
                    })
                    .collect()
            })
            .collect(),
    );
    let reported = [
        15822.0, 25107.0, 37246.0, 38798.0, 48169.0, 44373.0, 70288.0, 70655.0, 48804.0, 31732.0,
        18632.0,
    ];
    let paid = [
        15822.0, 24817.0, 36782.0, 38519.0, 44437.0, 39320.0, 52811.0, 40026.0, 22819.0, 11865.0,
        3409.0,
    ];
    let origins: Vec<f64> = (1998..=2008).map(f64::from).collect();
    (
        ultimates,
        Grid::column(&reported),
        Grid::column(&paid),
        Grid::column(&origins),
    )
}

#[test]
fn friedland_chapter_15_selection_by_origin_and_by_age() {
    let (ultimates, reported, paid, origins) = xyz();
    let names = Grid::from_rows(vec![
        [
            "development both",
            "development paid",
            "bf reported",
            "bf paid",
            "frequency-severity 1",
            "frequency-severity 2",
            "frequency-severity 3",
        ]
        .map(Value::from)
        .to_vec(),
    ]);
    // Column (4) to 2004, (6) and (7) halved in 2005 and 2006, (6) in
    // 2007, (6) and (9) halved in 2008.
    let mut by_origin = vec![vec![0.0; 7]; 11];
    for (o, w) in by_origin.iter_mut().enumerate() {
        match o {
            0..=6 => w[0] = 1.0,
            7 | 8 => (w[2], w[3]) = (0.5, 0.5),
            9 => w[2] = 1.0,
            _ => (w[2], w[5]) = (0.5, 0.5),
        }
    }
    let by_origin = Grid::from_rows(
        by_origin
            .iter()
            .map(|r| r.iter().map(|&x| Value::Number(x)).collect())
            .collect(),
    );
    // The same rule by age at 12/31/08: 12, 24, 36 and 48, 60 and older.
    let by_age = Grid::from_rows(
        [
            [12.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0],
            [24.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [36.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
            [60.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        .iter()
        .map(|r| r.iter().map(|&x| Value::Number(x)).collect())
        .collect(),
    );
    let ages: Vec<f64> = (0..11).map(|k| f64::from(132 - 12 * k)).collect();
    let book = [
        15822.0, 25107.0, 37246.0, 38798.0, 48169.0, 44373.0, 70780.0, 74726.0, 54968.0, 40300.0,
        33491.0,
    ];
    let book_ibnr = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 492.0, 4071.0, 6164.0, 8568.0, 14859.0,
    ];
    for (weights, ages) in [(by_origin, blank()), (by_age, Grid::column(&ages))] {
        let s = ok(
            "pro_res_select",
            vec![
                ultimates.clone(),
                weights,
                ages,
                reported.clone(),
                paid.clone(),
                origins.clone(),
                names.clone(),
            ],
        );
        let (u, i, up) = (
            column_of(&s, "Ultimate"),
            column_of(&s, "IBNR"),
            column_of(&s, "Unpaid"),
        );
        for o in 0..11 {
            assert!((num(s.get(o + 1, u)) - book[o]).abs() <= 0.5, "{o}");
            assert!((num(s.get(o + 1, i)) - book_ibnr[o]).abs() <= 0.5, "{o}");
        }
        assert_eq!(*s.get(12, 0), Value::from("Total"));
        assert!((num(s.get(12, u)) - 483781.0).abs() <= 1.0);
        assert!((num(s.get(12, i)) - 34155.0).abs() <= 1.0);
        assert!((num(s.get(12, up)) - (483781.0 - 330629.0)).abs() <= 3.0);
        assert_eq!(
            num(s.get(11, column_of(&s, "Weight frequency-severity 2"))),
            0.5
        );
    }
}

#[test]
fn selection_errors_name_the_origin() {
    let (ultimates, reported, _, origins) = xyz();
    // Weights that sum to 0.75 in 2008 are refused, not rescaled.
    let mut w = vec![vec![Value::Number(0.0); 7]; 11];
    for r in w.iter_mut() {
        r[0] = Value::Number(1.0);
    }
    w[10][0] = Value::Number(0.5);
    w[10][2] = Value::Number(0.25);
    let err = fails(
        "pro_res_select",
        vec![
            ultimates.clone(),
            Grid::from_rows(w),
            blank(),
            reported,
            blank(),
            origins.clone(),
        ],
    );
    assert!(err.contains("2008 sum to 0.75"), "{err}");
    // A weight on a method that does not project there.
    let mut one = vec![0.0; 7];
    one[5] = 1.0;
    let err = fails(
        "pro_res_select",
        vec![
            ultimates.clone(),
            row(&one),
            blank(),
            blank(),
            blank(),
            origins,
        ],
    );
    assert!(err.contains("no ultimate there"), "{err}");
    let err = fails("pro_res_select", vec![ultimates, row(&[1.0, 0.0])]);
    assert!(err.contains("weights must be"), "{err}");
}

#[test]
fn voting_chainladder_parity() {
    let (tri, premium) = genins();
    let ult = |table: &Grid| -> Vec<f64> {
        let c = column_of(table, "Ultimate");
        (1..11).map(|r| num(table.get(r, c))).collect()
    };
    let cl = ok("pro_res_cl_ultimate", vec![tri.clone()]);
    let cl: Vec<f64> = (0..10).map(|r| num(cl.get(r, 0))).collect();
    let bf = ult(&ok(
        "pro_res_bf",
        vec![tri.clone(), premium.clone(), n(0.75)],
    ));
    let cc = ult(&ok("pro_res_capecod", vec![tri.clone(), premium]));
    let ultimates = Grid::from_rows(
        (0..10)
            .map(|o| {
                vec![
                    Value::Number(cl[o]),
                    Value::Number(bf[o]),
                    Value::Number(cc[o]),
                ]
            })
            .collect(),
    );
    let latest: Vec<f64> = (0..10)
        .map(|r| {
            (0..10)
                .rev()
                .find_map(|c| match tri.get(r, c) {
                    Value::Number(x) => Some(*x),
                    _ => None,
                })
                .unwrap()
        })
        .collect();
    let phased = Grid::from_rows(
        (0..10)
            .map(|i| {
                match i {
                    0..=3 => [1.0, 0.0, 0.0],
                    4..=6 => [0.5, 0.5, 0.0],
                    7..=8 => [0.0, 0.5, 0.5],
                    _ => [0.0, 0.0, 1.0],
                }
                .map(Value::Number)
                .to_vec()
            })
            .collect(),
    );
    let names = Grid::from_rows(vec![["cl", "bf", "cc"].map(Value::from).to_vec()]);
    let labels: Vec<f64> = (2001..=2010).map(f64::from).collect();
    let cases = [
        ("phased", phased),
        ("blend", row(&[0.5, 0.25, 0.25])),
        ("single_bf", row(&[0.0, 1.0, 0.0])),
    ];
    let selected: Vec<(&str, Grid)> = cases
        .into_iter()
        .map(|(name, w)| {
            (
                name,
                ok(
                    "pro_res_select",
                    vec![
                        ultimates.clone(),
                        w,
                        blank(),
                        blank(),
                        Grid::column(&latest),
                        Grid::column(&labels),
                        names.clone(),
                    ],
                ),
            )
        })
        .collect();
    let mut checked = 0;
    for r in reference("reserving_exhibits_python.csv") {
        let (_, s) = selected.iter().find(|(n, _)| *n == r[1]).unwrap();
        let origin = || r[3].parse::<usize>().unwrap() - 2000;
        let got = match r[2].as_str() {
            "ultimate:cl" => cl[origin() - 1],
            "ultimate:bf" => bf[origin() - 1],
            "ultimate:cc" => cc[origin() - 1],
            "ultimate" => num(s.get(origin(), column_of(s, "Ultimate"))),
            "reserve" => num(s.get(origin(), column_of(s, "Unpaid"))),
            "total_ultimate" => num(s.get(11, column_of(s, "Ultimate"))),
            "total_reserve" => num(s.get(11, column_of(s, "Unpaid"))),
            other => panic!("unknown quantity {other}"),
        };
        let (want, abs, rel): (f64, f64, f64) = (
            r[4].parse().unwrap(),
            r[5].parse().unwrap(),
            r[6].parse().unwrap(),
        );
        assert!(
            (got - want).abs() <= abs.max(rel * want.abs()),
            "{} {} {}: {got} vs {want}",
            r[1],
            r[2],
            r[3]
        );
        checked += 1;
    }
    assert!(checked >= 120, "{checked}");
}
