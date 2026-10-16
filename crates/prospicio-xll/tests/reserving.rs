//! The worksheet functions on RAA (Mack 1993): the same numbers as
//! `prospicio_reserving`, which matches R ChainLadder, and the published
//! totals, reached through the catalogue as Excel reaches them.

use prospicio_reserving::{ChainLadder, Mack};
use prospicio_xll::reserving::triangle;
use prospicio_xll::sheet::{CellError, Grid, Value};
use prospicio_xll::{FUNCTIONS, call};

const RAA: [&[f64]; 10] = [
    &[
        5012.0, 8269.0, 10907.0, 11805.0, 13539.0, 16181.0, 18009.0, 18608.0, 18662.0, 18834.0,
    ],
    &[
        106.0, 4285.0, 5396.0, 10666.0, 13782.0, 15599.0, 15496.0, 16169.0, 16704.0,
    ],
    &[
        3410.0, 8992.0, 13873.0, 16141.0, 18735.0, 22214.0, 22863.0, 23466.0,
    ],
    &[5655.0, 11555.0, 15766.0, 21266.0, 23425.0, 26083.0, 27067.0],
    &[1092.0, 9565.0, 15836.0, 22169.0, 25955.0, 26180.0],
    &[1513.0, 6445.0, 11702.0, 12935.0, 15852.0],
    &[557.0, 4020.0, 10946.0, 12314.0],
    &[1351.0, 6947.0, 13112.0],
    &[3133.0, 5395.0],
    &[2063.0],
];

/// RAA as Excel passes a range: blanks below the diagonal.
fn raa() -> Grid {
    Grid::from_rows(
        RAA.iter()
            .map(|row| {
                (0..10)
                    .map(|c| row.get(c).map_or(Value::Empty, |&x| Value::Number(x)))
                    .collect()
            })
            .collect(),
    )
}

fn args(triangle: Grid, rest: usize) -> Vec<Grid> {
    std::iter::once(triangle)
        .chain(std::iter::repeat_with(Grid::missing).take(rest))
        .collect()
}

fn num(v: &Value) -> f64 {
    match v {
        Value::Number(x) => *x,
        other => panic!("expected a number, got {other:?}"),
    }
}

fn text(g: &Grid) -> &str {
    match g.as_scalar() {
        Some(Value::Text(s)) => s,
        other => panic!("expected text, got {other:?}"),
    }
}

#[test]
fn every_function_is_reachable_by_its_symbol() {
    for f in FUNCTIONS {
        let out = call(f.symbol, args(raa(), f.args.len().saturating_sub(1)));
        if let Some(Value::Text(s)) = out.as_scalar() {
            assert!(!s.starts_with("#PROSPICIO!"), "{}: {s}", f.name);
        }
        assert!(f.name.starts_with("PRO."));
        assert!(f.help.len() < 256 && f.args.iter().all(|a| a.help.len() < 254));
    }
}

#[test]
fn chain_ladder_matches_the_crate_and_r() {
    let fit = ChainLadder::default()
        .fit(&triangle(&raa()).unwrap(), "values")
        .unwrap();

    let ult = call("pro_cl_ultimate", args(raa(), 2));
    assert_eq!((ult.rows, ult.columns), (10, 1));
    let ult: Vec<f64> = ult.cells.iter().map(num).collect();
    assert_eq!(ult, fit.ultimate);

    let ldf = call("pro_cl_ldf", args(raa(), 2));
    assert_eq!((ldf.rows, ldf.columns), (1, 10));
    assert!((num(&ldf.cells[0]) - 2.999359).abs() < 1e-6);
    assert_eq!(num(&ldf.cells[9]), 1.0, "no tail");

    let cdf = call("pro_cl_cdf", args(raa(), 2));
    assert_eq!(cdf.columns, 10);
    assert!((num(&cdf.cells[0]) - 8.920234).abs() < 1e-6);

    // R ChainLadder: sum(MackChainLadder(RAA)$FullTriangle[, 10] - latest).
    let table = call("pro_cl", args(raa(), 3));
    assert_eq!((table.rows, table.columns), (12, 5));
    assert_eq!(table.get(0, 4), &Value::from("Reserve"));
    assert_eq!(table.get(11, 0), &Value::from("Total"));
    assert!((num(table.get(11, 4)) - 52_135.228).abs() < 1e-2);
    assert_eq!(num(table.get(10, 0)), 10.0, "origins default to 1..n");
}

#[test]
fn mack_matches_the_crate_and_r() {
    let fit = Mack::default()
        .fit(&triangle(&raa()).unwrap(), "values")
        .unwrap();
    let table = call("pro_mack", args(raa(), 3));
    assert_eq!((table.rows, table.columns), (12, 8));
    for i in 0..10 {
        assert_eq!(num(table.get(i + 1, 6)), fit.standard_error[i]);
    }
    assert_eq!(table.get(1, 7), &Value::Empty, "no reserve, no CV");
    // R ChainLadder: MackChainLadder(RAA, est.sigma = "log-linear")$Total.Mack.S.E
    // (validation/reference/reserving_chainladder_r.csv).
    assert!((num(table.get(11, 6)) - 26_880.740_329_891_4).abs() < 1e-6);
    assert!((num(table.get(11, 7)) - 26_880.740_329_891_4 / 52_135.228).abs() < 1e-6);
}

#[test]
fn options_and_labels() {
    let labels = Grid::from_rows(
        (1981..=1990)
            .map(|y| vec![Value::Number(y as f64)])
            .collect(),
    );
    let table = call(
        "pro_mack",
        vec![
            raa(),
            Value::from("simple").into(),
            Grid::scalar(Value::Number(1.05)),
            labels,
        ],
    );
    assert_eq!(num(table.get(1, 0)), 1981.0);
    let simple_tail = call(
        "pro_cl_ldf",
        vec![
            raa(),
            Value::from("Simple").into(),
            Grid::scalar(Value::Number(1.05)),
        ],
    );
    assert_eq!(num(&simple_tail.cells[9]), 1.05);

    let curve = call(
        "pro_cl_factors",
        vec![raa(), Grid::missing(), Value::from("exponential").into()],
    );
    assert_eq!(curve.get(10, 0), &Value::from("120-Ult"));
    assert!(num(curve.get(10, 1)) > 1.0);

    // #N/A marks an unobserved cell, as blanks do.
    let mut na = raa();
    na.cells[99] = Value::Error(CellError::NA);
    assert_eq!(call("pro_cl", args(na, 3)), call("pro_cl", args(raa(), 3)));
}

#[test]
fn errors_come_back_as_text() {
    let bad = call(
        "pro_cl",
        vec![
            raa(),
            Value::from("median").into(),
            Grid::missing(),
            Grid::missing(),
        ],
    );
    assert!(text(&bad).starts_with("#PROSPICIO! average must be"));

    let mut hole = raa();
    hole.cells[1] = Value::Empty;
    assert!(
        text(&call("pro_mack", args(hole, 3)))
            .contains("row 1, column 3 has a value after a blank")
    );

    let mut label = raa();
    label.cells[0] = Value::from("AY");
    assert!(text(&call("pro_cl_ldf", args(label, 2))).contains("is text"));

    let short = Grid::from_rows(vec![vec![Value::Number(1.0)]; 3]);
    assert!(text(&call("pro_cl_ldf", args(short, 2))).contains("two development columns"));

    let wrong = Grid::from_rows(vec![vec![Value::Number(1.0)]; 3]);
    assert!(
        text(&call(
            "pro_cl",
            vec![raa(), Grid::missing(), Grid::missing(), wrong]
        ))
        .contains("10 labels")
    );
}

#[test]
fn version_names_the_workspace_version() {
    assert_eq!(
        text(&call("pro_version", vec![])),
        concat!("prospicio ", env!("CARGO_PKG_VERSION"))
    );
}
