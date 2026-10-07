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
    /// Two key columns share a name.
    DuplicateKey(String),
    /// A key column and a value column share a name.
    KeyClash(String),
    /// A key value is given twice in a selection.
    DuplicateKeyValue { key: String, value: String },
    /// A value is infinite (NaN marks a missing value).
    NonFinite { column: String, row: usize },
    /// A development age is zero, or a valuation is before its origin starts.
    NonPositiveAge { row: usize },
    /// A development age is not on the triangle's development grid (a whole
    /// number of development periods from the youngest age).
    OffGrid { row: usize, age: u32 },
    /// No measure column has this name.
    UnknownColumn(String),
    /// No key column has this name.
    UnknownKey(String),
    /// No segment has this value of the key.
    UnknownKeyValue { key: String, value: String },
    /// A selection matches no segment.
    NoSegments,
    /// A selection lists no values for a key, or no columns.
    EmptySelection,
    /// A method needs a triangle with a single segment; select or group
    /// first.
    MultipleSegments(usize),
    /// A segment choice matches several segments of a fit or triangle.
    AmbiguousSegment(usize),
    /// A view needs one measure column and the triangle has several.
    MultipleColumns(usize),
    /// Fitting one segment of a triangle failed; `label` names it.
    InSegment { label: String, source: Box<Error> },
    /// A grain change that is not a coarsening of the current grain.
    InvalidGrain(&'static str),
    /// An origin has no observed value, so it has no latest diagonal.
    EmptyOrigin(String),
    /// A development factor could not be estimated.
    Factor { age: usize, reason: &'static str },
    /// The tail factor is not finite and positive.
    InvalidTail(f64),
    /// A tail cannot be fitted with these settings, or its variance cannot
    /// be extrapolated.
    Tail(&'static str),
    /// The method needs more development ages than the triangle has.
    TooFewAges { needed: usize, found: usize },
    /// An origin has no observed, finite, positive value in the exposure
    /// column.
    InvalidExposure { column: String, origin: String },
    /// A method setting is out of its range; `expected` describes the range.
    InvalidSetting {
        name: &'static str,
        value: f64,
        expected: &'static str,
    },
    /// The bootstrap cannot run on this input or with these settings.
    Bootstrap(&'static str),
    /// The ODP GLM cannot be fitted to this triangle.
    OdpGlm(String),
    /// Clark's growth-curve model cannot be fitted to this triangle.
    Clark(String),
    /// The claims development result (Merz–Wüthrich) does not apply to
    /// this fit.
    ClaimsDevelopment(&'static str),
    /// Re-reserving failed in `failed` of `n_sims` simulations of the
    /// one-year bootstrap; `source` is one of the failures (the first in
    /// the order of their messages, so the same whatever the threads).
    OneYear {
        failed: usize,
        n_sims: usize,
        source: Box<Error>,
    },
    /// An error from a shared crate (simulation, distributions).
    Core(prospicio_core::Error),
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
            Self::DuplicateKey(k) => write!(f, "key {k} is supplied twice"),
            Self::KeyClash(k) => write!(f, "{k} is both a key and a value column"),
            Self::DuplicateKeyValue { key, value } => {
                write!(f, "value {value:?} of key {key} is supplied twice")
            }
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
            Self::UnknownColumn(c) => write!(f, "no column named {c}"),
            Self::UnknownKey(k) => write!(f, "no key named {k}"),
            Self::UnknownKeyValue { key, value } => {
                write!(f, "no segment has {key} = {value:?}")
            }
            Self::NoSegments => write!(f, "no segment matches the selection"),
            Self::EmptySelection => write!(f, "a selection must list at least one value"),
            Self::MultipleSegments(n) => write!(
                f,
                "triangle has {n} segments; select one or group_by before fitting"
            ),
            Self::AmbiguousSegment(n) => {
                write!(f, "{n} segments match; name more keys to pick one")
            }
            Self::MultipleColumns(n) => {
                write!(f, "triangle has {n} columns; name the one to view")
            }
            Self::InSegment { label, source } => write!(f, "segment {label}: {source}"),
            Self::InvalidGrain(why) => write!(f, "invalid grain change: {why}"),
            Self::EmptyOrigin(o) => write!(f, "origin {o} has no observed values"),
            Self::Factor { age, reason } => {
                write!(f, "factor from development index {age}: {reason}")
            }
            Self::InvalidTail(t) => write!(f, "tail factor {t} is not finite and positive"),
            Self::Tail(why) => write!(f, "tail: {why}"),
            Self::TooFewAges { needed, found } => write!(
                f,
                "method needs at least {needed} development ages, triangle has {found}"
            ),
            Self::InvalidExposure { column, origin } => write!(
                f,
                "origin {origin} has no observed, finite, positive exposure in column {column}"
            ),
            Self::InvalidSetting {
                name,
                value,
                expected,
            } => write!(f, "{name} = {value} is invalid: expected {expected}"),
            Self::Bootstrap(why) => write!(f, "bootstrap: {why}"),
            Self::OdpGlm(why) => write!(f, "ODP GLM: {why}"),
            Self::Clark(why) => write!(f, "Clark: {why}"),
            Self::ClaimsDevelopment(why) => write!(f, "claims development result: {why}"),
            Self::OneYear {
                failed,
                n_sims,
                source,
            } => write!(
                f,
                "one-year bootstrap: re-reserving failed in {failed} of {n_sims} simulations, for example: {source}"
            ),
            Self::Core(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for Error {}

impl From<prospicio_core::Error> for Error {
    fn from(e: prospicio_core::Error) -> Self {
        Self::Core(e)
    }
}
