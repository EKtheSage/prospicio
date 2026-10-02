//! R bindings, built by `R CMD INSTALL R/actuarialrs`.
//!
//! Wrappers convert arguments and return [`act_core::Error`]s as R condition
//! objects (extendr `result_condition`), which the R layer raises. They
//! hold no numerical code; the idiomatic R API (functions and S3 methods)
//! lives in `R/actuarialrs/R`.
//!
//! Each lane keeps its wrappers in its own module, registered below.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};

mod aggregate;
mod distributions;

pub(crate) fn to_r(e: act_core::Error) -> Error {
    Error::Other(e.to_string())
}

/// Converts an R double holding a non-negative whole number below 2^53.
pub(crate) fn whole(x: f64, name: &str) -> Result<u64> {
    if x.is_finite() && x >= 0.0 && x.fract() == 0.0 && x < 9_007_199_254_740_992.0 {
        Ok(x as u64)
    } else {
        Err(Error::Other(format!(
            "{name} must be a non-negative whole number below 2^53, got {x}"
        )))
    }
}

extendr_module! {
    mod actuarialrs;
    use aggregate;
    use distributions;
}
