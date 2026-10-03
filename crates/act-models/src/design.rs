//! Design matrices: turning named columns into the numeric matrix a model
//! fits, the same way on training data and on new data.
//!
//! [`Terms`] says what goes in (an intercept, numeric columns, factors);
//! [`Terms::fit`] learns the factor levels from training data and returns
//! a [`Coding`]; [`Coding::design`] builds a [`Design`] from any frame with
//! the same columns. A fitted model keeps its `Coding`, so prediction on new
//! data reproduces the training columns exactly.

use std::collections::BTreeSet;

use act_core::{Error, Result};

/// A named column of data.
#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    Numeric(Vec<f64>),
    Categorical(Vec<String>),
}

impl Column {
    fn len(&self) -> usize {
        match self {
            Self::Numeric(v) => v.len(),
            Self::Categorical(v) => v.len(),
        }
    }
}

/// Named columns of equal length: the input to [`Coding::design`].
///
/// ```
/// use act_models::{Column, Frame};
///
/// let frame = Frame::new(vec![
///     ("age".into(), Column::Numeric(vec![30.0, 45.0])),
///     ("region".into(), Column::Categorical(vec!["N".into(), "S".into()])),
/// ])
/// .unwrap();
/// assert_eq!(frame.n_rows(), 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    columns: Vec<(String, Column)>,
    n_rows: usize,
}

impl Frame {
    /// Fails if the columns differ in length or two share a name.
    pub fn new(columns: Vec<(String, Column)>) -> Result<Self> {
        let n_rows = columns.first().map_or(0, |(_, c)| c.len());
        for (i, (name, c)) in columns.iter().enumerate() {
            if c.len() != n_rows {
                return Err(invalid(
                    "columns",
                    c.len() as f64,
                    "must all have the same length",
                ));
            }
            if columns[..i].iter().any(|(other, _)| other == name) {
                return Err(invalid("columns", i as f64, "repeat a column name"));
            }
        }
        Ok(Self { columns, n_rows })
    }

    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// The column called `name`.
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|(n, _)| n == name).map(|(_, c)| c)
    }

    fn require(&self, name: &str) -> Result<&Column> {
        self.column(name)
            .ok_or_else(|| Error::Data(format!("no column named {name:?}")))
    }
}

/// One term of a model formula.
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// A column of ones.
    Intercept,
    /// A numeric column, as is.
    Numeric(String),
    /// A categorical column in treatment coding: one indicator column per
    /// level except the reference. The reference is the given level, or
    /// the first in sorted order (R's and statsmodels' default).
    Factor {
        name: String,
        reference: Option<String>,
    },
}

/// The terms of a model, before any data is seen.
///
/// ```
/// use act_models::{Column, Frame, Terms};
///
/// let train = Frame::new(vec![
///     ("age".into(), Column::Numeric(vec![30.0, 45.0, 60.0])),
///     ("region".into(), Column::Categorical(vec!["N".into(), "S".into(), "W".into()])),
/// ])
/// .unwrap();
/// let coding = Terms::new().intercept().numeric("age").factor("region").fit(&train).unwrap();
/// assert_eq!(coding.names(), ["(Intercept)", "age", "region[S]", "region[W]"]);
/// let d = coding.design(&train).unwrap();
/// assert_eq!(d.column(3), [0.0, 0.0, 1.0]);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Terms {
    terms: Vec<Term>,
}

impl Terms {
    /// No terms.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an intercept.
    pub fn intercept(mut self) -> Self {
        self.terms.push(Term::Intercept);
        self
    }

    /// Adds a numeric column.
    pub fn numeric(mut self, name: &str) -> Self {
        self.terms.push(Term::Numeric(name.into()));
        self
    }

    /// Adds a factor with the default reference level.
    pub fn factor(mut self, name: &str) -> Self {
        self.terms.push(Term::Factor {
            name: name.into(),
            reference: None,
        });
        self
    }

    /// Adds a factor with the given reference level.
    pub fn factor_with_reference(mut self, name: &str, reference: &str) -> Self {
        self.terms.push(Term::Factor {
            name: name.into(),
            reference: Some(reference.into()),
        });
        self
    }

    /// Adds any term.
    pub fn term(mut self, term: Term) -> Self {
        self.terms.push(term);
        self
    }

    /// Learns factor levels from `frame`. Fails if a column is missing or
    /// has the wrong kind, or a reference level does not occur.
    pub fn fit(&self, frame: &Frame) -> Result<Coding> {
        let mut coded = Vec::with_capacity(self.terms.len());
        for term in &self.terms {
            coded.push(match term {
                Term::Intercept => Coded::Intercept,
                Term::Numeric(name) => match frame.require(name)? {
                    Column::Numeric(_) => Coded::Numeric(name.clone()),
                    Column::Categorical(_) => {
                        return Err(Error::Data(format!("{name:?} is categorical, not numeric")));
                    }
                },
                Term::Factor { name, reference } => {
                    let Column::Categorical(values) = frame.require(name)? else {
                        return Err(Error::Data(format!("{name:?} is numeric, not categorical")));
                    };
                    let levels: BTreeSet<&String> = values.iter().collect();
                    let reference = match reference {
                        Some(r) if levels.contains(r) => r.clone(),
                        Some(r) => {
                            return Err(Error::Data(format!(
                                "reference level {r:?} does not occur in {name:?}"
                            )));
                        }
                        None => levels
                            .first()
                            .map(|s| (*s).clone())
                            .ok_or_else(|| Error::Data(format!("{name:?} is empty")))?,
                    };
                    let others = levels
                        .into_iter()
                        .filter(|l| **l != reference)
                        .cloned()
                        .collect();
                    Coded::Factor {
                        name: name.clone(),
                        reference,
                        levels: others,
                    }
                }
            });
        }
        Ok(Coding { coded })
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Coded {
    Intercept,
    Numeric(String),
    Factor {
        name: String,
        reference: String,
        levels: Vec<String>,
    },
}

/// Terms with their factor levels learned: builds the same design matrix
/// on training data and on new data.
#[derive(Debug, Clone, PartialEq)]
pub struct Coding {
    coded: Vec<Coded>,
}

impl Coding {
    /// Column names of the design matrix: `(Intercept)`, numeric names,
    /// and `name[level]` for each non-reference factor level.
    pub fn names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for c in &self.coded {
            match c {
                Coded::Intercept => names.push("(Intercept)".into()),
                Coded::Numeric(n) => names.push(n.clone()),
                Coded::Factor { name, levels, .. } => {
                    names.extend(levels.iter().map(|l| format!("{name}[{l}]")));
                }
            }
        }
        names
    }

    /// Reference level of each factor, in term order.
    pub fn references(&self) -> Vec<(String, String)> {
        self.coded
            .iter()
            .filter_map(|c| match c {
                Coded::Factor {
                    name, reference, ..
                } => Some((name.clone(), reference.clone())),
                _ => None,
            })
            .collect()
    }

    /// The design matrix for `frame`. Fails if a column is missing, has
    /// the wrong kind, or holds a factor level not seen in training.
    pub fn design(&self, frame: &Frame) -> Result<Design> {
        let n = frame.n_rows();
        let mut columns: Vec<Vec<f64>> = Vec::new();
        for c in &self.coded {
            match c {
                Coded::Intercept => columns.push(vec![1.0; n]),
                Coded::Numeric(name) => match frame.require(name)? {
                    Column::Numeric(v) => columns.push(v.clone()),
                    Column::Categorical(_) => {
                        return Err(Error::Data(format!("{name:?} is categorical, not numeric")));
                    }
                },
                Coded::Factor {
                    name,
                    reference,
                    levels,
                } => {
                    let Column::Categorical(values) = frame.require(name)? else {
                        return Err(Error::Data(format!("{name:?} is numeric, not categorical")));
                    };
                    let start = columns.len();
                    columns.extend((0..levels.len()).map(|_| vec![0.0; n]));
                    for (i, v) in values.iter().enumerate() {
                        if v == reference {
                            continue;
                        }
                        let k = levels.binary_search(v).map_err(|_| {
                            Error::Data(format!("level {v:?} of {name:?} was not seen in training"))
                        })?;
                        columns[start + k][i] = 1.0;
                    }
                }
            }
        }
        Design::new(self.names(), columns)
    }
}

/// A design matrix (rows are observations, columns are coefficients), with
/// an offset and prior weights.
///
/// Stored column by column. The offset (default 0) enters the linear
/// predictor with a fixed coefficient of 1, as `ln(exposure)` does in a
/// log-link frequency model; the weights (default 1) divide the variance.
#[derive(Debug, Clone, PartialEq)]
pub struct Design {
    names: Vec<String>,
    columns: Vec<Vec<f64>>,
    n_rows: usize,
    offset: Vec<f64>,
    weights: Vec<f64>,
}

impl Design {
    /// A design from named columns. Fails if they differ in length, are
    /// empty, or hold a non-finite value.
    pub fn new(names: Vec<String>, columns: Vec<Vec<f64>>) -> Result<Self> {
        if names.len() != columns.len() {
            return Err(invalid(
                "names",
                names.len() as f64,
                "must name every column",
            ));
        }
        let n_rows = columns.first().map_or(0, Vec::len);
        if columns.iter().any(|c| c.len() != n_rows) {
            return Err(invalid(
                "columns",
                n_rows as f64,
                "must all have the same length",
            ));
        }
        if let Some(bad) = columns.iter().flatten().find(|x| !x.is_finite()) {
            return Err(invalid("columns", *bad, "must be finite"));
        }
        Ok(Self {
            names,
            columns,
            n_rows,
            offset: vec![0.0; n_rows],
            weights: vec![1.0; n_rows],
        })
    }

    /// Sets the offset, one value per row.
    pub fn with_offset(mut self, offset: Vec<f64>) -> Result<Self> {
        if offset.len() != self.n_rows {
            return Err(invalid(
                "offset",
                offset.len() as f64,
                "needs one value per row",
            ));
        }
        if let Some(bad) = offset.iter().find(|x| !x.is_finite()) {
            return Err(invalid("offset", *bad, "must be finite"));
        }
        self.offset = offset;
        Ok(self)
    }

    /// Sets the prior weights, one positive value per row.
    pub fn with_weights(mut self, weights: Vec<f64>) -> Result<Self> {
        if weights.len() != self.n_rows {
            return Err(invalid(
                "weights",
                weights.len() as f64,
                "needs one value per row",
            ));
        }
        if let Some(bad) = weights.iter().find(|x| !(x.is_finite() && **x > 0.0)) {
            return Err(invalid("weights", *bad, "must be finite and positive"));
        }
        self.weights = weights;
        Ok(self)
    }

    /// Column names.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns.
    pub fn n_cols(&self) -> usize {
        self.columns.len()
    }

    /// Column `j`.
    pub fn column(&self, j: usize) -> &[f64] {
        &self.columns[j]
    }

    /// Offset per row.
    pub fn offset(&self) -> &[f64] {
        &self.offset
    }

    /// Prior weight per row.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// `X β + offset`.
    pub fn linear_predictor(&self, beta: &[f64]) -> Vec<f64> {
        let mut eta = self.offset.clone();
        for (col, b) in self.columns.iter().zip(beta) {
            for (e, x) in eta.iter_mut().zip(col) {
                *e += b * x;
            }
        }
        eta
    }

    /// The rows in `rows`, in that order, with their offsets and weights.
    pub fn select(&self, rows: &[usize]) -> Self {
        let pick = |v: &[f64]| rows.iter().map(|&i| v[i]).collect::<Vec<_>>();
        Self {
            names: self.names.clone(),
            columns: self.columns.iter().map(|c| pick(c)).collect(),
            n_rows: rows.len(),
            offset: pick(&self.offset),
            weights: pick(&self.weights),
        }
    }
}

fn invalid(name: &'static str, value: f64, reason: &'static str) -> Error {
    Error::InvalidParameter {
        name,
        value,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(regions: &[&str]) -> Frame {
        Frame::new(vec![
            (
                "x".into(),
                Column::Numeric((0..regions.len()).map(|i| i as f64).collect()),
            ),
            (
                "region".into(),
                Column::Categorical(regions.iter().map(|s| s.to_string()).collect()),
            ),
        ])
        .unwrap()
    }

    #[test]
    fn coding_replays_on_new_data() {
        let train = frame(&["S", "N", "W", "N"]);
        let coding = Terms::new()
            .intercept()
            .factor_with_reference("region", "S")
            .numeric("x")
            .fit(&train)
            .unwrap();
        assert_eq!(
            coding.names(),
            ["(Intercept)", "region[N]", "region[W]", "x"]
        );
        // New data with a subset of levels, in another order.
        let d = coding.design(&frame(&["W", "S"])).unwrap();
        assert_eq!(d.column(1), [0.0, 0.0]);
        assert_eq!(d.column(2), [1.0, 0.0]);
        assert!(coding.design(&frame(&["E"])).is_err());
    }

    #[test]
    fn rejects_bad_frames_and_designs() {
        assert!(
            Frame::new(vec![
                ("a".into(), Column::Numeric(vec![1.0])),
                ("a".into(), Column::Numeric(vec![2.0])),
            ])
            .is_err()
        );
        assert!(Terms::new().numeric("region").fit(&frame(&["N"])).is_err());
        let d = Design::new(vec!["a".into()], vec![vec![1.0, 2.0]]).unwrap();
        assert!(d.clone().with_weights(vec![1.0, 0.0]).is_err());
        assert!(d.with_offset(vec![1.0]).is_err());
    }

    #[test]
    fn linear_predictor_and_select() {
        let d = Design::new(
            vec!["a".into(), "b".into()],
            vec![vec![1.0, 1.0, 1.0], vec![0.0, 1.0, 2.0]],
        )
        .unwrap()
        .with_offset(vec![0.5, 0.0, 0.0])
        .unwrap();
        assert_eq!(d.linear_predictor(&[1.0, 2.0]), [1.5, 3.0, 5.0]);
        let s = d.select(&[2, 0]);
        assert_eq!(s.column(1), [2.0, 0.0]);
        assert_eq!(s.offset(), [0.0, 0.5]);
    }
}
