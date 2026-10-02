//! Rating limits and layers from a severity.
//!
//! Both functions are ratios of limited expected values, so they apply to
//! any [`Severity`] and serve primary pricing (increased limit factors,
//! deductible credits) and reinsurance pricing alike.

use act_core::{Error, Result};
use act_prob::Severity;

/// Increased limit factor: the expected loss capped at `limit` relative
/// to the expected loss capped at `basic_limit`,
/// `ILF(limit) = LEV(limit) / LEV(basic_limit)`.
///
/// A limit of `+inf` gives the factor for unlimited cover (the mean over
/// `LEV(basic_limit)`).
///
/// ```
/// use act_prob::Lognormal;
/// use act_pricing::layer::ilf;
///
/// let sev = Lognormal::from_mean_cv(50_000.0, 3.0).unwrap();
/// let f = ilf(&sev, 1e6, 1e5).unwrap();
/// assert!(f > 1.0);
/// assert_eq!(ilf(&sev, 1e5, 1e5).unwrap(), 1.0);
/// ```
pub fn ilf<S: Severity + ?Sized>(severity: &S, limit: f64, basic_limit: f64) -> Result<f64> {
    positive("limit", limit)?;
    positive("basic_limit", basic_limit)?;
    if !basic_limit.is_finite() {
        return Err(Error::InvalidParameter {
            name: "basic_limit",
            value: basic_limit,
            reason: "must be finite",
        });
    }
    Ok(severity.lev(limit) / severity.lev(basic_limit))
}

/// Loss elimination ratio of a deductible: the share of expected ground-up
/// loss below `deductible`, `LEV(deductible) / E[X]`. This is the credit
/// for a straight deductible.
///
/// ```
/// use act_prob::Lognormal;
/// use act_pricing::layer::loss_elimination_ratio;
///
/// let sev = Lognormal::from_mean_cv(10_000.0, 1.0).unwrap();
/// let ler = loss_elimination_ratio(&sev, 1_000.0).unwrap();
/// assert!(ler > 0.0 && ler < 0.1);
/// ```
pub fn loss_elimination_ratio<S: Severity + ?Sized>(severity: &S, deductible: f64) -> Result<f64> {
    if deductible.is_nan() || deductible < 0.0 {
        return Err(Error::InvalidParameter {
            name: "deductible",
            value: deductible,
            reason: "must be non-negative",
        });
    }
    Ok(severity.lev(deductible) / severity.mean())
}

fn positive(name: &'static str, value: f64) -> Result<()> {
    if value.is_nan() || value <= 0.0 {
        return Err(Error::InvalidParameter {
            name,
            value,
            reason: "must be positive",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use act_prob::{Distribution, Lognormal};

    #[test]
    fn ilf_is_a_ratio_of_limited_means() {
        let sev = Lognormal::new(9.0, 1.5).unwrap();
        let f = ilf(&sev, 250_000.0, 100_000.0).unwrap();
        assert!((f - sev.lev(250_000.0) / sev.lev(100_000.0)).abs() < 1e-15);
        // Increasing in the limit, unlimited cover the largest.
        let g = ilf(&sev, 1e6, 100_000.0).unwrap();
        let unlimited = ilf(&sev, f64::INFINITY, 100_000.0).unwrap();
        assert!(1.0 < f && f < g && g < unlimited);
        assert!((unlimited - sev.mean() / sev.lev(100_000.0)).abs() < 1e-12);
        assert!(ilf(&sev, 0.0, 1.0).is_err());
        assert!(ilf(&sev, 1.0, f64::INFINITY).is_err());
    }

    #[test]
    fn loss_elimination_ratio_bounds() {
        let sev = Lognormal::new(9.0, 1.5).unwrap();
        assert_eq!(loss_elimination_ratio(&sev, 0.0).unwrap(), 0.0);
        let full = loss_elimination_ratio(&sev, f64::INFINITY).unwrap();
        assert!((full - 1.0).abs() < 1e-15);
        let ler = loss_elimination_ratio(&sev, 5_000.0).unwrap();
        assert!((ler - sev.lev(5_000.0) / sev.mean()).abs() < 1e-15);
        assert!(loss_elimination_ratio(&sev, -1.0).is_err());
    }
}
