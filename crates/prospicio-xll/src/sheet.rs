//! Worksheet values: what Excel passes in and what a function hands back,
//! with no Excel types, so the functions can be tested on any host.

use std::fmt;

/// Excel's error values, as `#N/A` and the others.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellError {
    Null,
    Div0,
    Value,
    Ref,
    Name,
    Num,
    NA,
}

impl fmt::Display for CellError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Null => "#NULL!",
            Self::Div0 => "#DIV/0!",
            Self::Value => "#VALUE!",
            Self::Ref => "#REF!",
            Self::Name => "#NAME?",
            Self::Num => "#NUM!",
            Self::NA => "#N/A",
        })
    }
}

/// One cell's value. `Empty` is a blank cell or an omitted argument.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Empty,
    Number(f64),
    Text(String),
    Bool(bool),
    Error(CellError),
}

impl Value {
    /// A number, or `Empty` for a non-finite one, which Excel cannot hold.
    pub fn number(x: f64) -> Self {
        if x.is_finite() {
            Self::Number(x)
        } else {
            Self::Empty
        }
    }

    /// Blank, or text of only spaces.
    pub fn is_blank(&self) -> bool {
        match self {
            Self::Empty => true,
            Self::Text(s) => s.trim().is_empty(),
            _ => false,
        }
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Self::Text(s.to_owned())
    }
}

/// A rectangle of values, row-major: a range, an array or a single cell
/// (1 x 1).
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    pub rows: usize,
    pub columns: usize,
    pub cells: Vec<Value>,
}

impl Grid {
    /// A 1 x 1 grid.
    pub fn scalar(value: Value) -> Self {
        Self {
            rows: 1,
            columns: 1,
            cells: vec![value],
        }
    }

    /// An omitted argument.
    pub fn missing() -> Self {
        Self::scalar(Value::Empty)
    }

    /// A grid from rows of equal length.
    pub fn from_rows(rows: Vec<Vec<Value>>) -> Self {
        let columns = rows.first().map_or(0, Vec::len);
        debug_assert!(rows.iter().all(|r| r.len() == columns));
        Self {
            rows: rows.len(),
            columns,
            cells: rows.into_iter().flatten().collect(),
        }
    }

    /// A single row of numbers.
    pub fn row(values: &[f64]) -> Self {
        Self::from_rows(vec![values.iter().map(|&x| Value::number(x)).collect()])
    }

    /// A single column of numbers.
    pub fn column(values: &[f64]) -> Self {
        Self::from_rows(values.iter().map(|&x| vec![Value::number(x)]).collect())
    }

    pub fn get(&self, row: usize, column: usize) -> &Value {
        &self.cells[row * self.columns + column]
    }

    /// The single value of a 1 x 1 grid; `None` for a larger one.
    pub fn as_scalar(&self) -> Option<&Value> {
        (self.cells.len() == 1).then(|| &self.cells[0])
    }

    /// Every cell blank, as an omitted argument is.
    pub fn is_blank(&self) -> bool {
        self.cells.iter().all(Value::is_blank)
    }
}

/// Where a cell sits inside an argument, for error messages: row and
/// column counted from 1 within the range.
pub fn position(row: usize, column: usize) -> String {
    format!("row {}, column {}", row + 1, column + 1)
}

impl From<Value> for Grid {
    fn from(value: Value) -> Self {
        Self::scalar(value)
    }
}
