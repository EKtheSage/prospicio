//! Shared test data: RAA (Mack 1993) as Excel passes a range, and small
//! helpers on worksheet values.

#![allow(dead_code)]

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
