//! Shared test data: RAA (Mack 1993) as Excel passes a range, and small
//! helpers on worksheet values.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use prospicio_xll::sheet::{Grid, Value};

pub const RAA: [&[f64]; 10] = [
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

/// RAA with blanks below the diagonal.
pub fn raa() -> Grid {
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

pub fn n(x: f64) -> Grid {
    Grid::scalar(Value::Number(x))
}

pub fn t(s: &str) -> Grid {
    Grid::scalar(Value::from(s))
}

pub fn num(v: &Value) -> f64 {
    match v {
        Value::Number(x) => *x,
        other => panic!("expected a number, got {other:?}"),
    }
}

pub fn text(g: &Grid) -> &str {
    match g.as_scalar() {
        Some(Value::Text(s)) => s,
        other => panic!("expected text, got {other:?}"),
    }
}

/// A file under the repository's `validation/` directory.
pub fn validation(path: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../validation")
        .join(path);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// GenIns paid as a sheet range and its premium (R's `?ClarkCapeCod`) as a
/// column, from `validation/data/genins_premium.csv`.
pub fn genins() -> (Grid, Grid) {
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

/// The CAS workers' compensation triangle summed over companies
/// (`validation/data/clrd_wkcomp.csv`, origins 1988 to 1997): each
/// origin's latest paid value and its premium there, oldest first.
pub fn clrd_wkcomp() -> (Vec<f64>, Vec<f64>) {
    let mut latest: BTreeMap<i32, (u32, f64, f64)> = BTreeMap::new();
    for line in validation("data/clrd_wkcomp.csv")
        .lines()
        .skip_while(|l| l.starts_with('#'))
        .skip(1)
    {
        let f: Vec<&str> = line.split(',').collect();
        let origin: i32 = f[0].parse().unwrap();
        let age: u32 = f[1].parse().unwrap();
        let cell = (age, f[2].parse().unwrap(), f[3].parse().unwrap());
        let e = latest.entry(origin).or_insert(cell);
        if age >= e.0 {
            *e = cell;
        }
    }
    latest
        .values()
        .map(|&(_, paid, premium)| (paid, premium))
        .unzip()
}

/// The CAS workers' compensation triangle (`validation/data/clrd_wkcomp.csv`)
/// as a sheet range of cumulative paid, and each origin's latest premium
/// as a column, as chainladder-python's `sample_weight=premium.latest_diagonal`.
pub fn clrd_wkcomp_triangle() -> (Grid, Grid) {
    let mut paid: BTreeMap<i32, BTreeMap<u32, f64>> = BTreeMap::new();
    for line in validation("data/clrd_wkcomp.csv")
        .lines()
        .skip_while(|l| l.starts_with('#'))
        .skip(1)
    {
        let f: Vec<&str> = line.split(',').collect();
        paid.entry(f[0].parse().unwrap())
            .or_default()
            .insert(f[1].parse().unwrap(), f[2].parse().unwrap());
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
    let (_, premium) = clrd_wkcomp();
    let premium = Grid::from_rows(premium.iter().map(|&p| vec![Value::Number(p)]).collect());
    (Grid::from_rows(rows), premium)
}

/// The rows of a reference CSV under `validation/reference/`, comments
/// and the header dropped, split on commas.
pub fn reference(name: &str) -> Vec<Vec<String>> {
    validation(&format!("reference/{name}"))
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("dataset,"))
        .map(|l| l.split(',').map(str::to_owned).collect())
        .collect()
}
