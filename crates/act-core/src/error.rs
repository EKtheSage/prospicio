//! The error type shared across the workspace.

use std::fmt;

/// Result alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Why an operation could not be carried out.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A model or distribution parameter is outside its valid domain.
    InvalidParameter {
        name: &'static str,
        value: f64,
        reason: &'static str,
    },
    /// A probability argument is NaN or outside `[0, 1]`.
    InvalidProbability(f64),
    /// A month number is outside 1 to 12.
    InvalidMonth(u8),
    /// Input data cannot be used as given: a missing or mistyped column, a
    /// factor level not seen in training, mismatched lengths. The message
    /// names what is wrong.
    Data(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter {
                name,
                value,
                reason,
            } => write!(f, "invalid parameter {name} = {value}: {reason}"),
            Self::InvalidProbability(p) => write!(f, "probability {p} is not in [0, 1]"),
            Self::InvalidMonth(m) => write!(f, "month {m} is not in 1 to 12"),
            Self::Data(message) => write!(f, "invalid data: {message}"),
        }
    }
}

impl std::error::Error for Error {}
