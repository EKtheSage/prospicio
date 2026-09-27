//! Loss development factors.

use std::fmt;

/// Why a development factor could not be computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorError {
    /// The two columns have different lengths, so origins cannot be paired.
    LengthMismatch { from: usize, to: usize },
    /// No origins were supplied.
    Empty,
    /// The `from` column sums to zero, so the ratio is undefined.
    ZeroDenominator,
    /// A value is NaN or infinite.
    NonFinite,
}

impl fmt::Display for FactorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { from, to } => {
                write!(f, "column lengths differ: from has {from}, to has {to}")
            }
            Self::Empty => write!(f, "no origins supplied"),
            Self::ZeroDenominator => write!(f, "from column sums to zero"),
            Self::NonFinite => write!(f, "input contains NaN or infinite values"),
        }
    }
}

impl std::error::Error for FactorError {}

/// Volume-weighted age-to-age factor between two adjacent development ages.
///
/// `from[i]` and `to[i]` are the cumulative losses of origin `i` at the
/// earlier and later age. Only origins observed at both ages should be passed.
///
/// ```text
/// f = sum(to) / sum(from)
/// ```
///
/// This is the chain-ladder link ratio: equivalent to averaging the
/// individual ratios `to[i] / from[i]` weighted by `from[i]`.
///
/// # Example
///
/// ```
/// use risk_rs::development::volume_weighted_factor;
///
/// let f = volume_weighted_factor(&[100.0, 200.0], &[150.0, 250.0]).unwrap();
/// assert!((f - 400.0 / 300.0).abs() < 1e-12);
/// ```
pub fn volume_weighted_factor(from: &[f64], to: &[f64]) -> Result<f64, FactorError> {
    if from.len() != to.len() {
        return Err(FactorError::LengthMismatch {
            from: from.len(),
            to: to.len(),
        });
    }
    if from.is_empty() {
        return Err(FactorError::Empty);
    }
    if from.iter().chain(to).any(|x| !x.is_finite()) {
        return Err(FactorError::NonFinite);
    }

    let denominator: f64 = from.iter().sum();
    if denominator == 0.0 {
        return Err(FactorError::ZeroDenominator);
    }
    let numerator: f64 = to.iter().sum();

    Ok(numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RAA triangle (Mack 1993), cumulative losses at 12 and 24 months for
    /// the nine origins observed at both ages.
    const RAA_12: [f64; 9] = [
        5012.0, 106.0, 3410.0, 5655.0, 1092.0, 1513.0, 557.0, 1351.0, 3133.0,
    ];
    const RAA_24: [f64; 9] = [
        8269.0, 4285.0, 8992.0, 11555.0, 9565.0, 6445.0, 4020.0, 6947.0, 5395.0,
    ];

    #[test]
    fn matches_raa_12_24_factor() {
        // chainladder-python and R ChainLadder both give 2.999359 for RAA 12-24.
        let f = volume_weighted_factor(&RAA_12, &RAA_24).unwrap();
        assert!((f - 2.999359).abs() < 1e-6, "got {f}");
    }

    #[test]
    fn single_origin_is_its_own_ratio() {
        assert_eq!(volume_weighted_factor(&[50.0], &[75.0]), Ok(1.5));
    }

    #[test]
    fn rejects_mismatched_lengths() {
        assert_eq!(
            volume_weighted_factor(&[1.0, 2.0], &[1.0]),
            Err(FactorError::LengthMismatch { from: 2, to: 1 })
        );
    }

    #[test]
    fn rejects_empty() {
        assert_eq!(volume_weighted_factor(&[], &[]), Err(FactorError::Empty));
    }

    #[test]
    fn rejects_zero_denominator() {
        assert_eq!(
            volume_weighted_factor(&[0.0, 0.0], &[1.0, 2.0]),
            Err(FactorError::ZeroDenominator)
        );
    }

    #[test]
    fn rejects_nan() {
        assert_eq!(
            volume_weighted_factor(&[1.0, f64::NAN], &[1.0, 2.0]),
            Err(FactorError::NonFinite)
        );
    }
}
