//! `PRO.PRC.TREND.FACTORS`, `PRO.PRC.LR.CUMULATIVE` and `PRO.REVERSE`
//! (#803, #806), reached through the catalogue as Excel reaches it: the
//! AP pricing demo's test vectors of #801 and #806.

use crate::common;

use common::{fails, n, num, ok, t, text};
use prospicio_xll::sheet::{Grid, Value};

fn column(v: &[f64]) -> Grid {
    Grid::from_rows(v.iter().map(|&x| vec![Value::Number(x)]).collect())
}

fn values(g: &Grid) -> Vec<f64> {
    g.cells.iter().map(num).collect()
}

fn b(x: bool) -> Grid {
    Grid::scalar(Value::Bool(x))
}

fn blank() -> Grid {
    Grid::missing()
}

fn schedule() -> (Grid, Grid) {
    let rates = [
        4.5, 4.0, 3.5, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 6.5, 5.5, 5.0, 4.5, 4.0, 4.0, 4.0,
    ];
    let years: Vec<f64> = (2012..2028).map(f64::from).collect();
    let rates: Vec<f64> = rates.iter().map(|r| r / 100.0).collect();
    (column(&years), column(&rates))
}

fn accident_years() -> Grid {
    column(&(2012..2026).map(f64::from).collect::<Vec<_>>())
}

fn close(got: f64, want: f64) {
    assert!(((got - want) / want).abs() < 1e-12, "{got} against {want}");
}

const INCEPTION: f64 = 2026.0027397260274;

#[test]
fn trend_factors_reproduce_the_issue_vectors() {
    let (ty, tr) = schedule();
    // (basis, kind, incomplete, cutoff, years 2012, 2018, 2025)
    let cases: [(&str, &str, bool, f64, [f64; 3]); 8] = [
        (
            "RAD",
            "loss",
            false,
            12.0,
            [1.965720460376, 1.549905768665, 1.060710030249],
        ),
        (
            "RAD",
            "premium",
            false,
            12.0,
            [1.970440084121, 1.557339412121, 1.060710030249],
        ),
        (
            "LOD",
            "loss",
            false,
            12.0,
            [1.927547497255, 1.519807645895, 1.040111758173],
        ),
        (
            "LOD",
            "premium",
            false,
            12.0,
            [1.932175469096, 1.527096933018, 1.040111758173],
        ),
        (
            "RAD",
            "loss",
            true,
            6.0,
            [1.965720460376, 1.549905768665, 1.071161637407],
        ),
        (
            "RAD",
            "premium",
            true,
            6.0,
            [1.970440084121, 1.557339412121, 1.072446776656],
        ),
        (
            "lod",
            "loss",
            true,
            9.0,
            [1.927547497255, 1.519807645895, 1.045223518981],
        ),
        (
            "LOD",
            "Premium",
            true,
            9.0,
            [1.932175469096, 1.527096933018, 1.045850340865],
        ),
    ];
    for (basis, kind, incomplete, cutoff, want) in cases {
        let g = ok(
            "pro_prc_trend_factors",
            vec![
                accident_years(),
                ty.clone(),
                tr.clone(),
                n(INCEPTION),
                t(basis),
                b(incomplete),
                n(cutoff),
                t(kind),
                blank(),
            ],
        );
        assert_eq!((g.rows, g.columns), (14, 1));
        let f = values(&g);
        for (k, w) in [0, 6, 13].into_iter().zip(want) {
            close(f[k], w);
        }
    }
    // The defaults: RAD, loss, complete years, factors.
    let g = ok(
        "pro_prc_trend_factors",
        vec![accident_years(), ty.clone(), tr.clone(), n(INCEPTION)],
    );
    let full = [
        1.965720460376,
        1.885588597245,
        1.817440068887,
        1.755980742886,
        1.692516473093,
        1.623521668846,
        1.549905768665,
        1.469121493285,
        1.379472004094,
        1.292248980824,
        1.219116374789,
        1.158308611831,
        1.105787026943,
        1.060710030249,
    ];
    for (f, w) in values(&g).into_iter().zip(full) {
        close(f, w);
    }
}

#[test]
fn trend_factors_detail_dates_and_blank_rows() {
    let (ty, tr) = schedule();
    // 1 January 2026 as an Excel date is 2026.0 on the half-month grid.
    let g = ok(
        "pro_prc_trend_factors",
        vec![
            column(&[2012.0, 2025.0]),
            ty.clone(),
            tr.clone(),
            n(46_023.0),
            blank(),
            blank(),
            blank(),
            blank(),
            t("detail"),
        ],
    );
    assert_eq!((g.rows, g.columns), (3, 4));
    assert_eq!(text(&Grid::scalar(g.cells[3].clone())), "Factor");
    assert_eq!(num(&g.cells[4]), 2012.0);
    assert_eq!(num(&g.cells[5]), 2012.5);
    assert_eq!(num(&g.cells[6]), 2027.0);
    close(num(&g.cells[7]), 1.965509247180);
    close(num(&g.cells[11]), 1.060596058827);

    // The extension case, with a blank row inside the schedule.
    let ty = Grid::from_rows(vec![
        vec![Value::Number(2012.0)],
        vec![Value::Empty],
        vec![Value::Number(2013.0)],
        vec![Value::Number(2014.0)],
    ]);
    let tr = Grid::from_rows(vec![
        vec![Value::Number(0.05)],
        vec![Value::Empty],
        vec![Value::Number(0.06)],
        vec![Value::Number(0.07)],
    ]);
    for kind in ["loss", "premium"] {
        let g = ok(
            "pro_prc_trend_factors",
            vec![
                column(&[2008.0, 2010.0, 2025.0]),
                ty.clone(),
                tr.clone(),
                n(2016.0027397260274),
                blank(),
                blank(),
                blank(),
                t(kind),
            ],
        );
        let f = values(&g);
        close(f[0], 1.7281053820990948);
        close(f[1], 1.5093941672627256);
        assert_eq!(f[2], 1.0);
    }

    // A half-blank row, a bad cutoff, a bad basis.
    let half = Grid::from_rows(vec![vec![Value::Number(0.05)], vec![Value::Empty]]);
    let two = column(&[2012.0, 2013.0]);
    assert!(
        fails(
            "pro_prc_trend_factors",
            vec![column(&[2012.0]), two.clone(), half, n(2016.0)]
        )
        .contains("not both")
    );
    let rates = column(&[0.05, 0.06]);
    let base = |cutoff: Grid, basis: Grid| {
        vec![
            column(&[2012.0]),
            two.clone(),
            rates.clone(),
            n(2016.0),
            basis,
            b(true),
            cutoff,
        ]
    };
    assert!(fails("pro_prc_trend_factors", base(n(13.0), blank())).contains("cutoff"));
    assert!(fails("pro_prc_trend_factors", base(n(6.0), t("XYZ"))).contains("basis"));
}

#[test]
fn cumulative_loss_ratio_runs_down_the_rows() {
    let lr = Grid::row(&[0.0, 0.6, 0.9, 0.3]);
    let ep = Grid::row(&[0.0, 100.0, 300.0, 600.0]);
    let g = ok("pro_prc_lr_cumulative", vec![lr, ep]);
    assert_eq!((g.rows, g.columns), (4, 1));
    let want = [0.0, 0.6, 330.0 / 400.0, 510.0 / 1000.0];
    for (f, w) in values(&g).into_iter().zip(want) {
        assert!((f - w).abs() < 1e-15);
    }
    // A blank row adds nothing; a half-blank row and a mismatch are errors.
    let lr = Grid::from_rows(vec![
        vec![Value::Number(0.5)],
        vec![Value::Empty],
        vec![Value::Number(0.7)],
    ]);
    let ep = Grid::from_rows(vec![
        vec![Value::Number(100.0)],
        vec![Value::Empty],
        vec![Value::Number(100.0)],
    ]);
    assert_eq!(
        values(&ok("pro_prc_lr_cumulative", vec![lr, ep])),
        vec![0.5, 0.5, 0.6]
    );
    let lr = column(&[0.5, 0.6]);
    let ep = Grid::from_rows(vec![vec![Value::Number(100.0)], vec![Value::Empty]]);
    assert!(fails("pro_prc_lr_cumulative", vec![lr, ep]).contains("not both"));
    assert!(
        fails(
            "pro_prc_lr_cumulative",
            vec![column(&[0.5]), column(&[1.0, 2.0])]
        )
        .contains("same number")
    );
}

#[test]
fn reverse_pads_and_cuts() {
    let row = Grid::row(&[0.35, 0.58, 0.74, 0.85, 0.92, 0.96, 0.98, 1.0]);
    let g = ok("pro_reverse", vec![row.clone(), n(10.0), blank()]);
    assert_eq!((g.rows, g.columns), (10, 1));
    assert_eq!(
        values(&g),
        vec![1.0, 1.0, 1.0, 0.98, 0.96, 0.92, 0.85, 0.74, 0.58, 0.35]
    );
    assert_eq!(
        values(&ok("pro_reverse", vec![row.clone(), n(3.0)])),
        vec![1.0, 0.98, 0.96]
    );
    assert_eq!(
        values(&ok("pro_reverse", vec![row.clone()])),
        vec![1.0, 0.98, 0.96, 0.92, 0.85, 0.74, 0.58, 0.35]
    );
    assert_eq!(
        values(&ok(
            "pro_reverse",
            vec![Grid::row(&[1.0, 2.0]), n(3.0), n(0.0)]
        )),
        vec![0.0, 2.0, 1.0]
    );
    let gap = Grid::from_rows(vec![vec![
        Value::Number(1.0),
        Value::Empty,
        Value::Number(2.0),
    ]]);
    assert!(fails("pro_reverse", vec![gap]).contains("blank"));
    assert!(fails("pro_reverse", vec![row.clone(), n(0.0)]).contains("length"));
    let square = Grid::from_rows(vec![vec![Value::Number(1.0); 2]; 2]);
    assert!(fails("pro_reverse", vec![square]).contains("single row or column"));
}
