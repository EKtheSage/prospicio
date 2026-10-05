//! Errors raised by triangle construction and reserving methods.

use std::fmt;

/// Result alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Why a triangle could not be built or a method could not be fitted.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The long table has no rows or no value columns.
    Empty,
    /// A long-table column has a different number of rows than `origin`.
    LengthMismatch {
        column: String,
        expected: usize,
        found: usize,
    },
    /// Two value columns share a name.
    DuplicateColumn(String),
    /// An index label is named twice.
    DuplicateLabel(String),
    /// A value is infinite (NaN marks a missing value).
    NonFinite { column: String, row: usize },
    /// A development age is zero, or a valuation is before its origin starts.
    NonPositiveAge { row: usize },
    /// A development age is not on the triangle's development grid (a whole
    /// number of development periods from the youngest age).
    OffGrid { row: usize, age: u32 },
    /// No column or index position has this name.
    UnknownLabel(String),
    /// A method needs a triangle with a single index position; slice first.
    MultipleSegments(usize),
    /// A grain change that is not a coarsening of the current grain.
    InvalidGrain(&'static str),
    /// An origin has no observed value, so it has no latest diagonal.
    EmptyOrigin(String),
    /// A development factor could not be estimated.
    Factor { age: usize, reason: &'static str },
    /// The tail factor is not finite and positive.
    InvalidTail(f64),
    /// The method needs more development ages than the triangle has.
    TooFewAges { needed: usize, found: usize },
    /// The bootstrap cannot run on this input or with these settings.
    Bootstrap(&'static str),
    /// The ODP GLM cannot be fitted to this triangle.
    OdpGlm(String),
    /// An error from a shared crate (simulation, distributions).
    Core(act_core::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "no rows or no value columns supplied"),
            Self::LengthMismatch {
                column,
                expected,
                found,
            } => write!(f, "column {column} has {found} rows, expected {expected}"),
            Self::DuplicateColumn(c) => write!(f, "column {c} is supplied twice"),
            Self::DuplicateLabel(l) => write!(f, "index label {l} is supplied twice"),
            Self::NonFinite { column, row } => {
                write!(f, "column {column}, row {row} is infinite")
            }
            Self::NonPositiveAge { row } => {
                write!(f, "row {row} is not after the start of its origin period")
            }
            Self::OffGrid { row, age } => write!(
                f,
                "row {row}: age {age} months is not on the development grid"
            ),
            Self::UnknownLabel(l) => write!(f, "no column or index named {l}"),
            Self::MultipleSegments(n) => write!(
                f,
                "triangle has {n} index positions; slice to one before fitting"
            ),
            Self::InvalidGrain(why) => write!(f, "invalid grain change: {why}"),
            Self::EmptyOrigin(o) => write!(f, "origin {o} has no observed values"),
            Self::Factor { age, reason } => {
                write!(f, "factor from development index {age}: {reason}")
            }
            Self::InvalidTail(t) => write!(f, "tail factor {t} is not finite and positive"),
            Self::TooFewAges { needed, found } => write!(
                f,
                "method needs at least {needed} development ages, triangle has {found}"
            ),
            Self::Bootstrap(why) => write!(f, "bootstrap: {why}"),
            Self::OdpGlm(why) => write!(f, "ODP GLM: {why}"),
            Self::Core(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for Error {}

impl From<act_core::Error> for Error {
    fn from(e: act_core::Error) -> Self {
        Self::Core(e)
    }
}
