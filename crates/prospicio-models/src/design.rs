//! Design matrices: turning named columns into the numeric matrix a model
//! fits, the same way on training data and on new data.
//!
//! [`Terms`] says what goes in (an intercept, numeric columns, factors);
//! [`Terms::fit`] learns the factor levels from training data and returns
//! a [`Coding`]; [`Coding::design`] builds a [`Design`] from any frame with
//! the same columns. A fitted model keeps its `Coding`, so prediction on new
//! data reproduces the training columns exactly.

use std::collections::BTreeSet;

use prospicio_core::{Error, Result};

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
/// use prospicio_models::{Column, Frame};
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
    /// The product of two columns, each numeric as is or a factor in
    /// treatment coding with its first level as reference: `a:b`, `a[x]:b`
    /// or `a[x]:b[y]`, R's `a:b` beside main effects.
    Interaction { a: String, b: String },
    /// Raw powers `x, x², …, x^degree` of a numeric column, named `x`,
    /// `x^2`, ….
    Poly { name: String, degree: usize },
    /// Hinge functions `max(0, x − k)` at each knot, named `hinge(x,k)`.
    Hinges { name: String, knots: Vec<f64> },
    /// Steps `1[x ≥ k]` at each knot, named `step(x,k)`.
    Steps { name: String, knots: Vec<f64> },
    /// A numeric column cut into bins `(−∞, c₁], (c₁, c₂], …, (c_k, ∞)` and
    /// coded as a factor (named as R's `cut`, `x[(30,45]]`), the bin with
    /// index `reference` (default 0, the lowest) as reference.
    Bins {
        name: String,
        cuts: Vec<f64>,
        reference: Option<usize>,
    },
    /// `ln(x / base)`, named `log(x/base)`: a log transform centred so a
    /// coefficient's base value is `base` (CAS Monograph 5, section 5.6.2).
    Log { name: String, base: f64 },
    /// An ordered factor in cumulative coding: for each level after the
    /// first in `levels`' order, `1[level ≥ it]`, named `x[>=level]`. Each
    /// coefficient is the step from the level below.
    Ordinal { name: String, levels: Vec<String> },
    /// Orthogonal polynomials of degree 1 to `degree` on the training
    /// values, as R's `poly(x, degree)`, named `poly(x)1`, `poly(x)2`, ….
    /// The recurrence learned in training rebuilds them on new data.
    OrthogonalPoly { name: String, degree: usize },
    /// A natural cubic spline with interior `knots`, linear beyond the
    /// boundary knots (the training range unless given): `knots.len() + 1`
    /// columns named `spline(x)1`, …. The basis is the restricted cubic
    /// (truncated power) one, which spans the same functions as R's
    /// `splines::ns(x, knots)`: fitted values agree, coefficients differ.
    NaturalSpline {
        name: String,
        knots: Vec<f64>,
        boundary: Option<(f64, f64)>,
    },
}

/// The terms of a model, before any data is seen.
///
/// ```
/// use prospicio_models::{Column, Frame, Terms};
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

    /// Adds the interaction of two columns; see [`Term::Interaction`].
    pub fn interaction(self, a: &str, b: &str) -> Self {
        self.term(Term::Interaction {
            a: a.into(),
            b: b.into(),
        })
    }

    /// Adds raw powers of a numeric column up to `degree`.
    pub fn poly(self, name: &str, degree: usize) -> Self {
        self.term(Term::Poly {
            name: name.into(),
            degree,
        })
    }

    /// Adds hinge functions `max(0, x − k)` at `knots`.
    pub fn hinges(self, name: &str, knots: &[f64]) -> Self {
        self.term(Term::Hinges {
            name: name.into(),
            knots: knots.to_vec(),
        })
    }

    /// Adds steps `1[x ≥ k]` at `knots`.
    pub fn steps(self, name: &str, knots: &[f64]) -> Self {
        self.term(Term::Steps {
            name: name.into(),
            knots: knots.to_vec(),
        })
    }

    /// Adds a numeric column cut into bins at `cuts`, the lowest bin as
    /// reference.
    pub fn bins(self, name: &str, cuts: &[f64]) -> Self {
        self.term(Term::Bins {
            name: name.into(),
            cuts: cuts.to_vec(),
            reference: None,
        })
    }

    /// Adds `ln(x / base)`.
    pub fn log(self, name: &str, base: f64) -> Self {
        self.term(Term::Log {
            name: name.into(),
            base,
        })
    }

    /// Adds an ordered factor in cumulative coding, levels in `levels`'
    /// order.
    pub fn ordinal(self, name: &str, levels: &[&str]) -> Self {
        self.term(Term::Ordinal {
            name: name.into(),
            levels: levels.iter().map(|l| (*l).to_string()).collect(),
        })
    }

    /// Adds orthogonal polynomials of a numeric column up to `degree`.
    pub fn orthogonal_poly(self, name: &str, degree: usize) -> Self {
        self.term(Term::OrthogonalPoly {
            name: name.into(),
            degree,
        })
    }

    /// Adds a natural cubic spline with interior `knots`, its boundary
    /// knots the training range.
    pub fn natural_spline(self, name: &str, knots: &[f64]) -> Self {
        self.term(Term::NaturalSpline {
            name: name.into(),
            knots: knots.to_vec(),
            boundary: None,
        })
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
                Term::Numeric(name) => {
                    numeric(frame, name)?;
                    Coded::Numeric(name.clone())
                }
                Term::Factor { name, reference } => {
                    let (reference, levels) = factor_levels(frame, name, reference.as_deref())?;
                    Coded::Factor {
                        name: name.clone(),
                        reference,
                        levels,
                    }
                }
                Term::Interaction { a, b } => Coded::Interaction {
                    a: side(frame, a)?,
                    b: side(frame, b)?,
                },
                Term::Poly { name, degree } => {
                    numeric(frame, name)?;
                    if *degree == 0 {
                        return Err(invalid("degree", 0.0, "must be at least 1"));
                    }
                    Coded::Poly {
                        name: name.clone(),
                        degree: *degree,
                    }
                }
                Term::Hinges { name, knots } | Term::Steps { name, knots } => {
                    numeric(frame, name)?;
                    increasing(knots, "knots")?;
                    if matches!(term, Term::Hinges { .. }) {
                        Coded::Hinges {
                            name: name.clone(),
                            knots: knots.clone(),
                        }
                    } else {
                        Coded::Steps {
                            name: name.clone(),
                            knots: knots.clone(),
                        }
                    }
                }
                Term::Bins {
                    name,
                    cuts,
                    reference,
                } => {
                    numeric(frame, name)?;
                    increasing(cuts, "cuts")?;
                    let reference = reference.unwrap_or(0);
                    if reference > cuts.len() {
                        return Err(invalid(
                            "reference",
                            reference as f64,
                            "must index one of the bins",
                        ));
                    }
                    Coded::Bins {
                        name: name.clone(),
                        cuts: cuts.clone(),
                        reference,
                    }
                }
                Term::Log { name, base } => {
                    numeric(frame, name)?;
                    if !(base.is_finite() && *base > 0.0) {
                        return Err(invalid("base", *base, "must be positive"));
                    }
                    Coded::Log {
                        name: name.clone(),
                        base: *base,
                    }
                }
                Term::OrthogonalPoly { name, degree } => {
                    let x = numeric(frame, name)?;
                    orthogonal_poly(name, x, *degree)?
                }
                Term::NaturalSpline {
                    name,
                    knots,
                    boundary,
                } => {
                    let x = numeric(frame, name)?;
                    let (lo, hi) = match boundary {
                        Some(b) => *b,
                        None => x
                            .iter()
                            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                                (lo.min(v), hi.max(v))
                            }),
                    };
                    let mut all = vec![lo];
                    all.extend(knots);
                    all.push(hi);
                    increasing(&all, "knots")?;
                    if knots.is_empty() {
                        return Err(invalid(
                            "knots",
                            0.0,
                            "a natural spline needs an interior knot",
                        ));
                    }
                    Coded::NaturalSpline {
                        name: name.clone(),
                        knots: all,
                    }
                }
                Term::Ordinal { name, levels } => {
                    categorical(frame, name)?;
                    if levels.len() < 2 {
                        return Err(invalid(
                            "levels",
                            levels.len() as f64,
                            "an ordinal needs at least two levels",
                        ));
                    }
                    if levels
                        .iter()
                        .enumerate()
                        .any(|(i, l)| levels[..i].contains(l))
                    {
                        return Err(Error::Data(format!("{name:?} repeats a level")));
                    }
                    Coded::Ordinal {
                        name: name.clone(),
                        levels: levels.clone(),
                    }
                }
            });
        }
        Ok(Coding { coded })
    }
}

fn numeric<'a>(frame: &'a Frame, name: &str) -> Result<&'a [f64]> {
    match frame.require(name)? {
        Column::Numeric(v) => Ok(v),
        Column::Categorical(_) => Err(Error::Data(format!("{name:?} is categorical, not numeric"))),
    }
}

fn categorical<'a>(frame: &'a Frame, name: &str) -> Result<&'a [String]> {
    match frame.require(name)? {
        Column::Categorical(v) => Ok(v),
        Column::Numeric(_) => Err(Error::Data(format!("{name:?} is numeric, not categorical"))),
    }
}

/// The reference and the other levels, sorted, of a factor.
fn factor_levels(
    frame: &Frame,
    name: &str,
    reference: Option<&str>,
) -> Result<(String, Vec<String>)> {
    let values = categorical(frame, name)?;
    let levels: BTreeSet<&String> = values.iter().collect();
    let reference = match reference {
        Some(r) if levels.iter().any(|l| l.as_str() == r) => r.to_string(),
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
    Ok((reference, others))
}

fn side(frame: &Frame, name: &str) -> Result<Side> {
    Ok(match frame.require(name)? {
        Column::Numeric(_) => Side::Numeric(name.to_string()),
        Column::Categorical(_) => {
            let (reference, levels) = factor_levels(frame, name, None)?;
            Side::Factor {
                name: name.to_string(),
                reference,
                levels,
            }
        }
    })
}

fn increasing(v: &[f64], name: &'static str) -> Result<()> {
    if v.is_empty() {
        return Err(invalid(name, 0.0, "must not be empty"));
    }
    if let Some(&bad) = v.iter().find(|x| !x.is_finite()) {
        return Err(invalid(name, bad, "must be finite"));
    }
    if v.windows(2).any(|w| w[0] >= w[1]) {
        return Err(invalid(name, v.len() as f64, "must be strictly increasing"));
    }
    Ok(())
}

/// One side of an interaction.
#[derive(Debug, Clone, PartialEq)]
enum Side {
    Numeric(String),
    Factor {
        name: String,
        reference: String,
        levels: Vec<String>,
    },
}

impl Side {
    fn names(&self) -> Vec<String> {
        match self {
            Self::Numeric(n) => vec![n.clone()],
            Self::Factor { name, levels, .. } => {
                levels.iter().map(|l| format!("{name}[{l}]")).collect()
            }
        }
    }

    fn columns(&self, frame: &Frame) -> Result<Vec<Vec<f64>>> {
        match self {
            Self::Numeric(n) => Ok(vec![numeric(frame, n)?.to_vec()]),
            Self::Factor {
                name,
                reference,
                levels,
            } => indicators(categorical(frame, name)?, name, reference, levels),
        }
    }
}

/// Treatment-coded indicator columns of a factor's values.
fn indicators(
    values: &[String],
    name: &str,
    reference: &str,
    levels: &[String],
) -> Result<Vec<Vec<f64>>> {
    let mut columns = vec![vec![0.0; values.len()]; levels.len()];
    for (i, v) in values.iter().enumerate() {
        if v == reference {
            continue;
        }
        let k = levels.binary_search(v).map_err(|_| {
            Error::Data(format!("level {v:?} of {name:?} was not seen in training"))
        })?;
        columns[k][i] = 1.0;
    }
    Ok(columns)
}

/// The recurrence of R's `poly(x, degree)` on the training values `x`.
fn orthogonal_poly(name: &str, x: &[f64], degree: usize) -> Result<Coded> {
    let distinct: BTreeSet<u64> = x.iter().map(|v| v.to_bits()).collect();
    if degree == 0 || degree >= distinct.len() {
        return Err(invalid(
            "degree",
            degree as f64,
            "must be at least 1 and less than the number of distinct values",
        ));
    }
    let n = x.len();
    let (mut alpha, mut norm2) = (Vec::with_capacity(degree), vec![1.0, n as f64]);
    let (mut prev, mut cur) = (vec![0.0; n], vec![1.0; n]);
    for k in 0..degree {
        let a = x.iter().zip(&cur).map(|(x, p)| x * p * p).sum::<f64>() / norm2[k + 1];
        let ratio = norm2[k + 1] / norm2[k];
        let next: Vec<f64> = (0..n)
            .map(|i| (x[i] - a) * cur[i] - if k == 0 { 0.0 } else { ratio * prev[i] })
            .collect();
        alpha.push(a);
        norm2.push(next.iter().map(|p| p * p).sum());
        prev = std::mem::replace(&mut cur, next);
    }
    Ok(Coded::OrthogonalPoly {
        name: name.to_string(),
        alpha,
        norm2,
    })
}

fn poly_columns(x: &[f64], alpha: &[f64], norm2: &[f64]) -> Vec<Vec<f64>> {
    let n = x.len();
    let (mut prev, mut cur) = (vec![0.0; n], vec![1.0; n]);
    let mut out = Vec::with_capacity(alpha.len());
    for (k, a) in alpha.iter().enumerate() {
        let ratio = norm2[k + 1] / norm2[k];
        let next: Vec<f64> = (0..n)
            .map(|i| (x[i] - a) * cur[i] - if k == 0 { 0.0 } else { ratio * prev[i] })
            .collect();
        let scale = norm2[k + 2].sqrt();
        out.push(next.iter().map(|p| p / scale).collect());
        prev = std::mem::replace(&mut cur, next);
    }
    out
}

/// The restricted cubic spline basis on knots `t₁ < … < t_K`: `x`, then
/// `(d_j − d_{K−1}) / (t_K − t₁)²` for `j < K − 1`, with
/// `d_j(x) = ((x − t_j)³₊ − (x − t_K)³₊) / (t_K − t_j)`; linear beyond the
/// boundary knots.
fn spline_columns(x: &[f64], knots: &[f64]) -> Vec<Vec<f64>> {
    let k = knots.len();
    let last = knots[k - 1];
    let scale = (last - knots[0]).powi(2);
    let cube = |v: f64| v.max(0.0).powi(3);
    let d = |j: usize, v: f64| (cube(v - knots[j]) - cube(v - last)) / (last - knots[j]);
    let mut out = vec![x.to_vec()];
    out.extend((0..k - 2).map(|j| x.iter().map(|&v| (d(j, v) - d(k - 2, v)) / scale).collect()));
    out
}

/// A bin's label, as R's `cut`: `(a,b]`, open at both infinite ends.
fn bin_label(cuts: &[f64], k: usize) -> String {
    let lo = if k == 0 {
        "-Inf".to_string()
    } else {
        cuts[k - 1].to_string()
    };
    if k == cuts.len() {
        format!("({lo},Inf)")
    } else {
        format!("({lo},{}]", cuts[k])
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
    Interaction {
        a: Side,
        b: Side,
    },
    Poly {
        name: String,
        degree: usize,
    },
    Hinges {
        name: String,
        knots: Vec<f64>,
    },
    Steps {
        name: String,
        knots: Vec<f64>,
    },
    Bins {
        name: String,
        cuts: Vec<f64>,
        reference: usize,
    },
    Log {
        name: String,
        base: f64,
    },
    Ordinal {
        name: String,
        levels: Vec<String>,
    },
    /// R's `poly` recurrence: `p₀ = 1`, `p₁ = x − α₁`,
    /// `p_{k+1} = (x − α_{k+1}) p_k − (n_{k+1} / n_k) p_{k−1}`, each column
    /// `p_k / √n_{k+1}`; `norm2` is `[1, n₁, …, n_{degree+1}]`.
    OrthogonalPoly {
        name: String,
        alpha: Vec<f64>,
        norm2: Vec<f64>,
    },
    /// Every knot, boundaries included.
    NaturalSpline {
        name: String,
        knots: Vec<f64>,
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
                Coded::Interaction { a, b } => {
                    let (na, nb) = (a.names(), b.names());
                    for y in &nb {
                        names.extend(na.iter().map(|x| format!("{x}:{y}")));
                    }
                }
                Coded::Poly { name, degree } => {
                    names.push(name.clone());
                    names.extend((2..=*degree).map(|k| format!("{name}^{k}")));
                }
                Coded::Hinges { name, knots } => {
                    names.extend(knots.iter().map(|k| format!("hinge({name},{k})")));
                }
                Coded::Steps { name, knots } => {
                    names.extend(knots.iter().map(|k| format!("step({name},{k})")));
                }
                Coded::Bins {
                    name,
                    cuts,
                    reference,
                } => names.extend(
                    (0..=cuts.len())
                        .filter(|k| k != reference)
                        .map(|k| format!("{name}[{}]", bin_label(cuts, k))),
                ),
                Coded::Log { name, base } => names.push(format!("log({name}/{base})")),
                Coded::Ordinal { name, levels } => {
                    names.extend(levels[1..].iter().map(|l| format!("{name}[>={l}]")));
                }
                Coded::OrthogonalPoly { name, alpha, .. } => {
                    names.extend((1..=alpha.len()).map(|k| format!("poly({name}){k}")));
                }
                Coded::NaturalSpline { name, knots } => {
                    names.extend((1..knots.len()).map(|k| format!("spline({name}){k}")));
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
                Coded::Numeric(name) => columns.push(numeric(frame, name)?.to_vec()),
                Coded::Factor {
                    name,
                    reference,
                    levels,
                } => columns.extend(indicators(
                    categorical(frame, name)?,
                    name,
                    reference,
                    levels,
                )?),
                Coded::Interaction { a, b } => {
                    let (ca, cb) = (a.columns(frame)?, b.columns(frame)?);
                    for y in &cb {
                        for x in &ca {
                            columns.push(x.iter().zip(y).map(|(x, y)| x * y).collect());
                        }
                    }
                }
                Coded::Poly { name, degree } => {
                    let x = numeric(frame, name)?;
                    columns.extend(
                        (1..=*degree).map(|k| x.iter().map(|v| v.powi(k as i32)).collect()),
                    );
                }
                Coded::Hinges { name, knots } => {
                    let x = numeric(frame, name)?;
                    columns.extend(
                        knots
                            .iter()
                            .map(|k| x.iter().map(|v| (v - k).max(0.0)).collect()),
                    );
                }
                Coded::Steps { name, knots } => {
                    let x = numeric(frame, name)?;
                    columns.extend(
                        knots
                            .iter()
                            .map(|k| x.iter().map(|v| if v >= k { 1.0 } else { 0.0 }).collect()),
                    );
                }
                Coded::Bins {
                    name,
                    cuts,
                    reference,
                } => {
                    let x = numeric(frame, name)?;
                    let start = columns.len();
                    columns.extend((0..cuts.len()).map(|_| vec![0.0; n]));
                    for (i, v) in x.iter().enumerate() {
                        if v.is_nan() {
                            return Err(Error::Data(format!("{name:?} holds NaN")));
                        }
                        // The bin (c_{k-1}, c_k]: the number of cuts below v.
                        let k = cuts.partition_point(|c| c < v);
                        if k != *reference {
                            let col = if k < *reference { k } else { k - 1 };
                            columns[start + col][i] = 1.0;
                        }
                    }
                }
                Coded::Log { name, base } => {
                    let x = numeric(frame, name)?;
                    if let Some(&bad) = x.iter().find(|v| v.is_nan() || **v <= 0.0) {
                        return Err(Error::Data(format!(
                            "log({name}/{base}) needs positive values, got {bad}"
                        )));
                    }
                    columns.push(x.iter().map(|v| (v / base).ln()).collect());
                }
                Coded::OrthogonalPoly { alpha, norm2, name } => {
                    columns.extend(poly_columns(numeric(frame, name)?, alpha, norm2));
                }
                Coded::NaturalSpline { name, knots } => {
                    columns.extend(spline_columns(numeric(frame, name)?, knots));
                }
                Coded::Ordinal { name, levels } => {
                    let values = categorical(frame, name)?;
                    let start = columns.len();
                    columns.extend((1..levels.len()).map(|_| vec![0.0; n]));
                    for (i, v) in values.iter().enumerate() {
                        let rank = levels.iter().position(|l| l == v).ok_or_else(|| {
                            Error::Data(format!("level {v:?} of {name:?} is not one of its levels"))
                        })?;
                        for col in columns[start..start + rank].iter_mut() {
                            col[i] = 1.0;
                        }
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

    fn rating_frame() -> Frame {
        Frame::new(vec![
            ("age".into(), Column::Numeric(vec![20.0, 30.0, 45.0, 60.0])),
            (
                "region".into(),
                Column::Categorical(["N", "S", "W", "S"].iter().map(|s| s.to_string()).collect()),
            ),
            (
                "size".into(),
                Column::Categorical(
                    ["small", "large", "medium", "small"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                ),
            ),
        ])
        .unwrap()
    }

    #[test]
    fn interactions_multiply_the_coded_columns() {
        let f = rating_frame();
        let c = Terms::new()
            .interaction("region", "age")
            .interaction("region", "size")
            .interaction("age", "age")
            .fit(&f)
            .unwrap();
        assert_eq!(
            c.names(),
            [
                "region[S]:age",
                "region[W]:age",
                // The first factor varies fastest, as R's a:b.
                "region[S]:size[medium]",
                "region[W]:size[medium]",
                "region[S]:size[small]",
                "region[W]:size[small]",
                "age:age",
            ]
        );
        let d = c.design(&f).unwrap();
        assert_eq!(d.column(0), [0.0, 30.0, 0.0, 60.0]);
        assert_eq!(d.column(3), [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(d.column(4), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(d.column(6), [400.0, 900.0, 2025.0, 3600.0]);
    }

    #[test]
    fn polynomials_hinges_steps_and_logs() {
        let f = rating_frame();
        let c = Terms::new()
            .poly("age", 3)
            .hinges("age", &[25.0, 50.0])
            .steps("age", &[30.0])
            .log("age", 40.0)
            .fit(&f)
            .unwrap();
        assert_eq!(
            c.names(),
            [
                "age",
                "age^2",
                "age^3",
                "hinge(age,25)",
                "hinge(age,50)",
                "step(age,30)",
                "log(age/40)"
            ]
        );
        let d = c.design(&f).unwrap();
        assert_eq!(d.column(2), [8000.0, 27000.0, 91125.0, 216000.0]);
        assert_eq!(d.column(3), [0.0, 5.0, 20.0, 35.0]);
        assert_eq!(d.column(4), [0.0, 0.0, 0.0, 10.0]);
        // A step includes its knot.
        assert_eq!(d.column(5), [0.0, 1.0, 1.0, 1.0]);
        assert!((d.column(6)[2] - (45.0f64 / 40.0).ln()).abs() < 1e-15);
        assert!(Terms::new().hinges("age", &[50.0, 25.0]).fit(&f).is_err());
        assert!(Terms::new().poly("age", 0).fit(&f).is_err());
        assert!(Terms::new().log("age", 0.0).fit(&f).is_err());
    }

    #[test]
    fn bins_code_a_numeric_column_as_a_factor() {
        let f = rating_frame();
        let c = Terms::new().bins("age", &[30.0, 50.0]).fit(&f).unwrap();
        // Right-closed, as R's cut: 30 falls in (-Inf,30].
        assert_eq!(c.names(), ["age[(30,50]]", "age[(50,Inf)]"]);
        let d = c.design(&f).unwrap();
        assert_eq!(d.column(0), [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(d.column(1), [0.0, 0.0, 0.0, 1.0]);
        let middle = Terms::new()
            .term(Term::Bins {
                name: "age".into(),
                cuts: vec![30.0, 50.0],
                reference: Some(1),
            })
            .fit(&f)
            .unwrap();
        assert_eq!(middle.names(), ["age[(-Inf,30]]", "age[(50,Inf)]"]);
        assert_eq!(middle.design(&f).unwrap().column(0), [1.0, 1.0, 0.0, 0.0]);
        // Values outside the training range still fall in a bin.
        let new = Frame::new(vec![("age".into(), Column::Numeric(vec![-5.0, 99.0]))]).unwrap();
        assert_eq!(c.design(&new).unwrap().column(1), [0.0, 1.0]);
    }

    #[test]
    fn ordinals_are_cumulative_steps() {
        let f = rating_frame();
        let c = Terms::new()
            .ordinal("size", &["small", "medium", "large"])
            .fit(&f)
            .unwrap();
        assert_eq!(c.names(), ["size[>=medium]", "size[>=large]"]);
        let d = c.design(&f).unwrap();
        assert_eq!(d.column(0), [0.0, 1.0, 1.0, 0.0]);
        assert_eq!(d.column(1), [0.0, 1.0, 0.0, 0.0]);
        let unknown = Frame::new(vec![(
            "size".into(),
            Column::Categorical(vec!["huge".into()]),
        )])
        .unwrap();
        assert!(c.design(&unknown).is_err());
        assert!(
            Terms::new()
                .ordinal("size", &["small", "small"])
                .fit(&f)
                .is_err()
        );
    }
}
