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
    /// An age before the triangle's first age, so that nothing is kept.
    AgeBeforeFirst { age: u32, first: u32 },
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
    /// A selected development pattern is invalid, or does not fit the
    /// triangle or the tail; `age` (months) is the age it concerns, when
    /// one does.
    Pattern {
        age: Option<prospicio_core::Lag>,
        reason: &'static str,
    },
    /// A tail cannot be fitted with these settings, or its variance cannot
    /// be extrapolated.
    Tail(&'static str),
    /// The method needs more development ages than the triangle has.
    TooFewAges { needed: usize, found: usize },
    /// An origin has no observed, finite, positive value in the exposure
    /// column.
    InvalidExposure { column: String, origin: String },
    /// An a-priori loss ratio by origin does not line up with the origins
    /// fitted: an origin without one, one for an origin the triangle or
    /// diagonal does not have, an origin given twice or the wrong number of
    /// values in origin order (`reserving-v02.md` decision 22).
    Apriori(String),
    /// A method setting is out of its range; `expected` describes the range.
    InvalidSetting {
        name: &'static str,
        value: f64,
        expected: &'static str,
    },
    /// A setting, or a combination of settings, that the method it was
    /// given to does not support: `setting` names it, `method` the method
    /// and `reason` why, usually with what to use instead. This is the one
    /// variant for that situation in every method (`reserving-v02.md`
    /// decision 29): the variants named after a method ([`Bootstrap`],
    /// [`Bayes`], [`ClaimsDevelopment`]) carry only failures on the data.
    /// The Python bindings raise it as `prospicio.NotSupportedError` and the
    /// R package signals it with the condition class
    /// `prospicio_not_supported`.
    ///
    /// [`Bootstrap`]: Self::Bootstrap
    /// [`Bayes`]: Self::Bayes
    /// [`ClaimsDevelopment`]: Self::ClaimsDevelopment
    NotSupported {
        setting: &'static str,
        method: &'static str,
        reason: &'static str,
    },
    /// A setting that is planned for the method but not built in this
    /// release (`reserving-v02.md` decision 29). No setting raises it
    /// today; the bindings treat it as [`NotSupported`](Self::NotSupported).
    NotYetBuilt {
        setting: &'static str,
        method: &'static str,
    },
    /// The method needs link ratios and the triangle has none: every origin
    /// is observed at one age only, a latest diagonal. `method` names it.
    LatestDiagonalOnly { method: &'static str },
    /// A latest diagonal cannot be built or fitted as given.
    Diagonal(String),
    /// The bootstrap cannot run on this input: the triangle's shape or
    /// values, or a setting's value out of its range. A setting the
    /// bootstrap does not support is [`NotSupported`](Self::NotSupported).
    Bootstrap(&'static str),
    /// A bootstrap's own link ratio exclusions (`n_periods`, `drop_high`
    /// and `drop_low`) differ from the re-reserving method's in its one-year
    /// view; each is written out.
    ExclusionsMismatch { bootstrap: String, method: String },
    /// The ODP GLM cannot be fitted to this triangle.
    OdpGlm(String),
    /// A Bayesian reserving model cannot be fitted to this triangle, for
    /// example an origin whose observed increments sum to zero or less,
    /// which leaves the log-linear mean undefined, or its sampler or mode
    /// search failed. A setting the model does not support is
    /// [`NotSupported`](Self::NotSupported), and one out of its range
    /// [`InvalidSetting`](Self::InvalidSetting).
    Bayes(String),
    /// An excess-layer pattern cannot be derived from this input
    /// (`reserving-excess.md`).
    Excess(String),
    /// Development patterns per simulation cannot be read as asked
    /// (`reserving-v02.md` decision 28): an unknown segment, a falling
    /// pattern or a horizon too short.
    PatternDraws(String),
    /// An exhibit or a selection of ultimates cannot be built as given
    /// (`reserving-v02.md` decision 34): lengths that differ, an unknown or
    /// repeated method, or weights that do not sum to 1.
    Exhibit(String),
    /// A triangle cannot be read from a file (the `io` feature): the file
    /// cannot be opened, a column is missing or a cell cannot be read. The
    /// message names the file line or cell and the column.
    File(String),
    /// The long table's periods span far more cells than it has rows, as one
    /// mistyped origin or valuation does (`9020-01` among monthly origins of
    /// the 2020s): the dense grid it asks for is refused before it is
    /// allocated (#832). The message names the earliest and latest period.
    TooLarge(String),
    /// Clark's growth-curve model cannot be fitted to this triangle.
    Clark(String),
    /// A method on claim counts (frequency-severity, disposal rates, case
    /// outstanding development or a Berquist-Sherman adjustment) cannot be
    /// applied to this triangle or with these settings.
    Counts(String),
    /// The Munich chain ladder cannot be fitted: the paid and incurred
    /// columns are not observed at the same cells, or a cell is not
    /// positive ([`MunichChainLadder`](crate::MunichChainLadder)).
    Munich(String),
    /// A growth curve cannot be fitted to this benchmark pattern
    /// ([`BenchmarkCurve`](crate::BenchmarkCurve)).
    BenchmarkCurve(String),
    /// The paid-incurred chain ([`PaidIncurredChain`](crate::PaidIncurredChain))
    /// cannot be fitted: a triangle of the wrong shape, a value that is not
    /// positive, or a setting of the wrong length or out of range.
    PaidIncurred(String),
    /// A development curve cannot be fitted to these factors, or its
    /// settings do not fit the triangle
    /// ([`DevelopmentCurve`](crate::DevelopmentCurve)).
    DevelopmentCurve(String),
    /// An industry prior cannot be built from these triangles
    /// ([`IndustryPrior`](crate::IndustryPrior)): too few companies, ages
    /// that differ, or a pattern or spread that cannot be estimated.
    IndustryPrior(String),
    /// The claims development result (Merz–Wüthrich) does not apply to
    /// this triangle: its latest values are not on one calendar diagonal.
    /// A setting of the fit that the closed form does not support (a tail,
    /// `alpha` other than 1, link ratio exclusions) is
    /// [`NotSupported`](Self::NotSupported).
    ClaimsDevelopment(&'static str),
    /// Re-reserving failed in `failed` of `n_sims` simulations of the
    /// one-year bootstrap; `source` is one of the failures (the first in
    /// the order of their messages, so the same whatever the threads).
    OneYear {
        failed: usize,
        n_sims: usize,
        source: Box<Error>,
    },
    /// The tail could not be refitted on the pseudo factors of `failed` of
    /// `n_sims` simulations of a bootstrap's lifetime view; `source` is one
    /// of the failures (the first in the order of their messages).
    TailRefit {
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
            Self::AgeBeforeFirst { age, first } => write!(
                f,
                "age {age} months is before the triangle's first age, {first} months, so no age is kept"
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
            Self::Pattern {
                age: Some(age),
                reason,
            } => write!(f, "selected pattern, age {age} months: {reason}"),
            Self::Pattern { age: None, reason } => write!(f, "selected pattern: {reason}"),
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
            Self::NotSupported {
                setting,
                method,
                reason,
            } => write!(f, "{setting} is not supported by {method}: {reason}"),
            Self::NotYetBuilt { setting, method } => write!(
                f,
                "{setting} is not yet built for {method}: it is planned for a later release"
            ),
            Self::LatestDiagonalOnly { method } => write!(
                f,
                "{method} needs link ratios, and the triangle has none: every origin is observed \
                 at one age only (a latest diagonal); the chain ladder and the expected-loss \
                 methods project it with a fully selected pattern"
            ),
            Self::Diagonal(why) => write!(f, "latest diagonal: {why}"),
            Self::Bootstrap(why) => write!(f, "bootstrap: {why}"),
            Self::ExclusionsMismatch { bootstrap, method } => write!(
                f,
                "bootstrap: the bootstrap's exclusions ({bootstrap}) must be the re-reserving method's ({method}), or none"
            ),
            Self::OdpGlm(why) => write!(f, "ODP GLM: {why}"),
            Self::Bayes(why) => write!(f, "Bayesian reserving: {why}"),
            Self::Apriori(why) => write!(f, "apriori: {why}"),
            Self::Excess(why) => write!(f, "excess layer: {why}"),
            Self::PatternDraws(why) => write!(f, "pattern draws: {why}"),
            Self::Exhibit(why) => write!(f, "exhibit: {why}"),
            Self::File(why) => write!(f, "reading a triangle: {why}"),
            Self::TooLarge(why) => write!(f, "triangle too large: {why}"),
            Self::Clark(why) => write!(f, "Clark: {why}"),
            Self::Counts(why) => write!(f, "claim counts: {why}"),
            Self::Munich(why) => write!(f, "Munich chain ladder: {why}"),
            Self::BenchmarkCurve(why) => write!(f, "benchmark curve: {why}"),
            Self::PaidIncurred(why) => write!(f, "paid-incurred chain: {why}"),
            Self::DevelopmentCurve(why) => write!(f, "development curve: {why}"),
            Self::IndustryPrior(why) => write!(f, "industry prior: {why}"),
            Self::ClaimsDevelopment(why) => write!(f, "claims development result: {why}"),
            Self::OneYear {
                failed,
                n_sims,
                source,
            } => write!(
                f,
                "one-year bootstrap: re-reserving failed in {failed} of {n_sims} simulations, for example: {source}"
            ),
            Self::TailRefit {
                failed,
                n_sims,
                source,
            } => write!(
                f,
                "bootstrap: the tail could not be refitted in {failed} of {n_sims} simulations, for example: {source}"
            ),
            Self::Core(e) => e.fmt(f),
        }
    }
}

impl Error {
    /// Whether the error is a setting the method does not support
    /// ([`NotSupported`](Self::NotSupported), [`NotYetBuilt`](Self::NotYetBuilt)
    /// or [`ExclusionsMismatch`](Self::ExclusionsMismatch), a bootstrap's
    /// exclusions that differ from its re-reserving method's), looking
    /// through [`InSegment`](Self::InSegment). The Python and R bindings use
    /// it to raise `prospicio.NotSupportedError` and the condition class
    /// `prospicio_not_supported`.
    pub fn is_not_supported(&self) -> bool {
        match self {
            Self::NotSupported { .. }
            | Self::NotYetBuilt { .. }
            | Self::ExclusionsMismatch { .. } => true,
            Self::InSegment { source, .. } => source.is_not_supported(),
            _ => false,
        }
    }
}

impl std::error::Error for Error {}

impl From<prospicio_core::Error> for Error {
    fn from(e: prospicio_core::Error) -> Self {
        Self::Core(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_supported_is_found_through_a_segment() {
        let unsupported = Error::NotSupported {
            setting: "s",
            method: "m",
            reason: "r",
        };
        assert!(unsupported.is_not_supported());
        assert!(
            Error::NotYetBuilt {
                setting: "s",
                method: "m"
            }
            .is_not_supported()
        );
        let wrapped = Error::InSegment {
            label: "a".into(),
            source: Box::new(unsupported),
        };
        assert!(wrapped.is_not_supported());
        assert!(!Error::Bootstrap("no residuals to resample").is_not_supported());
        assert!(!Error::Bayes("x".into()).is_not_supported());
        assert_eq!(
            Error::NotYetBuilt {
                setting: "s",
                method: "m"
            }
            .to_string(),
            "s is not yet built for m: it is planned for a later release"
        );
    }
}
