//! The expected-loss methods (`PRO.EL`, `PRO.BF`, `PRO.BENKTANDER`,
//! `PRO.CAPECOD`) on GenIns with R ChainLadder's ClarkCapeCod premium,
//! reached through the catalogue as Excel reaches them, against every
//! total chainladder-python 0.10.1 gives in
//! `validation/reference/reserving_expected_loss_python.csv`.

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use common::{n, num, t, text};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn validation(path: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../validation")
        .join(path);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// GenIns paid as a sheet range, and its premium as a column.
fn genins() -> (Grid, Grid) {
    let mut paid: BTreeMap<i32, BTreeMap<u32, f64>> = BTreeMap::new();
    let mut premium: BTreeMap<i32, f64> = BTreeMap::new();
    for line in validation("data/genins_premium.csv")
        .lines()
        .skip_while(|l| l.starts_with('#'))
        .skip(1)
    {
        let f: Vec<&str> = line.split(',').collect();
        let origin: i32 = f[0].parse().unwrap();
        paid.entry(origin)
            .or_default()
            .insert(f[1].parse().unwrap(), f[2].parse().unwrap());
        premium.insert(origin, f[3].parse().unwrap());
    }
    let ages = paid.values().map(BTreeMap::len).max().unwrap();
    let rows = paid
        .values()
        .map(|row| {
            let cells: Vec<f64> = row.values().copied().collect();
            (0..ages)
                .map(|c| cells.get(c).map_or(Value::Empty, |&x| Value::Number(x)))
                .collect()
        })
        .collect();
    let premium = Grid::from_rows(premium.values().map(|&p| vec![Value::Number(p)]).collect());
    (Grid::from_rows(rows), premium)
}

/// The method spec's settings: `bornhuetter_ferguson;apriori=0.75;average=volume`.
fn settings(spec: &str) -> (String, BTreeMap<String, String>) {
    let mut parts = spec.split(';');
    let method = parts.next().unwrap().to_owned();
    let kv = parts
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap();
            (k.to_owned(), v.to_owned())
        })
        .collect();
    (method, kv)
}

fn run(method: &str, s: &BTreeMap<String, String>) -> Grid {
    let (tri, premium) = genins();
    let f = |k: &str| n(s[k].parse().unwrap());
    let average = t(&s["average"]);
    let none = Grid::missing;
    match method {
        "expected_loss" => call(
            "pro_el",
            vec![tri, premium, f("apriori"), average, none(), none()],
        ),
        "bornhuetter_ferguson" => call(
            "pro_bf",
            vec![tri, premium, f("apriori"), average, none(), none()],
        ),
        "benktander" => call(
            "pro_benktander",
            vec![
                tri,
                premium,
                f("apriori"),
                f("n_iters"),
                average,
                none(),
                none(),
            ],
        ),
        "cape_cod" => call(
            "pro_capecod",
            vec![
                tri,
                premium,
                f("trend"),
                f("decay"),
                average,
                none(),
                none(),
            ],
        ),
        other => panic!("unknown method {other}"),
    }
}

#[test]
fn totals_match_chainladder_python_on_genins() {
    let mut checked = 0;
    for line in validation("reference/reserving_expected_loss_python.csv").lines() {
        let f: Vec<&str> = line.split(',').collect();
        if !line.starts_with("genins_premium,") || !f[2].starts_with("total_") {
            continue;
        }
        let (method, s) = settings(f[1]);
        let out = run(&method, &s);
        assert_eq!((out.rows, out.columns), (12, 7), "{}", f[1]);
        let col = if f[2] == "total_ultimate" { 5 } else { 6 };
        let got = num(out.get(11, col));
        let expected: f64 = f[4].parse().unwrap();
        let rel = f[6].parse::<f64>().unwrap_or(1e-9).max(1e-9);
        assert!(
            (got - expected).abs() <= rel * expected.abs(),
            "{} {}: {got} vs {expected}",
            f[1],
            f[2]
        );
        checked += 1;
    }
    assert_eq!(checked, 28, "every GenIns total in the reference");
}

#[test]
fn exhibit_columns_and_errors() {
    let (tri, premium) = genins();
    let out = call(
        "pro_bf",
        vec![
            tri.clone(),
            premium.clone(),
            n(0.75),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
        ],
    );
    assert_eq!(out.get(0, 4), &Value::from("A priori"));
    assert_eq!(num(out.get(1, 1)), 10_000_000.0);
    assert_eq!(num(out.get(1, 4)), 0.75);
    // The oldest origin is fully developed: its BF ultimate is its latest.
    assert!((num(out.get(1, 5)) - num(out.get(1, 2))).abs() < 1e-6);

    let short = Grid::from_rows(vec![vec![Value::Number(1.0)]; 3]);
    let none = Grid::missing;
    assert!(
        text(&call(
            "pro_bf",
            vec![tri.clone(), short, n(0.75), none(), none(), none()]
        ))
        .contains("10 values")
    );
    assert!(
        text(&call(
            "pro_bf",
            vec![tri.clone(), premium.clone(), none(), none(), none(), none()]
        ))
        .contains("apriori is required")
    );
    let mut zero = premium.clone();
    zero.cells[3] = Value::Number(0.0);
    assert!(
        text(&call(
            "pro_el",
            vec![tri, zero, n(0.5), none(), none(), none()]
        ))
        .contains("positive")
    );
}
