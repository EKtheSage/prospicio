//! Wald inference on a coefficient from its estimate and standard error,
//! on the normal: the two-sided p-value of `β = 0` and the confidence
//! interval `β ± z se` (#976). [`GlmFit::p_values`](crate::GlmFit::p_values)
//! uses Student's t when the dispersion is estimated; these are the normal
//! forms that the two-part models and the Excel add-in report.

use prospicio_core::{Error, Result};
use prospicio_math::special::{norm_cdf, norm_quantile};

/// The two-sided normal p-value of a Wald statistic `z = β / se`,
/// `2 Φ(-|z|)`.
///
/// ```
/// use prospicio_glm::wald::p_value;
///
/// assert!((p_value(1.959963984540054) - 0.05).abs() < 1e-12);
/// assert_eq!(p_value(0.0), 1.0);
/// ```
pub fn p_value(z: f64) -> f64 {
    2.0 * norm_cdf(-z.abs())
}

/// The normal quantile `z = Φ⁻¹((1 + level) / 2)` of a two-sided interval
/// at confidence `level`, in `(0, 1)`.
///
/// ```
/// use prospicio_glm::wald::critical_value;
///
/// assert!((critical_value(0.95).unwrap() - 1.959963984540054).abs() < 1e-12);
/// assert!(critical_value(1.0).is_err());
/// ```
pub fn critical_value(level: f64) -> Result<f64> {
    if !(level > 0.0 && level < 1.0) {
        return Err(Error::InvalidParameter {
            name: "level",
            value: level,
            reason: "must be between 0 and 1",
        });
    }
    Ok(norm_quantile(0.5 + level / 2.0))
}

/// The Wald interval `estimate ± z se` at confidence `level`, as
/// `(lower, upper)`.
///
/// ```
/// use prospicio_glm::wald::interval;
///
/// let (lo, hi) = interval(1.0, 0.5, 0.95).unwrap();
/// assert!((lo - 0.020018007729973).abs() < 1e-12 && (hi - 1.979981992270027).abs() < 1e-12);
/// ```
pub fn interval(estimate: f64, se: f64, level: f64) -> Result<(f64, f64)> {
    let z = critical_value(level)?;
    Ok((estimate - z * se, estimate + z * se))
}
