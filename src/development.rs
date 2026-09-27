//! Loss development factors.

use std::fmt;

use crate::triangle::Triangle;

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

/// Simple-average age-to-age factor between two adjacent development ages.
///
/// ```text
/// f = mean(to[i] / from[i])
/// ```
///
/// Every origin carries equal weight regardless of size. Fails with
/// [`FactorError::ZeroDenominator`] if any `from[i]` is zero.
pub fn simple_average_factor(from: &[f64], to: &[f64]) -> Result<f64, FactorError> {
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
    if from.contains(&0.0) {
        return Err(FactorError::ZeroDenominator);
    }

    let total: f64 = from.iter().zip(to).map(|(a, b)| b / a).sum();
    Ok(total / from.len() as f64)
}

/// How individual link ratios are averaged into one factor per age.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Averaging {
    /// `sum(to) / sum(from)`, see [`volume_weighted_factor`].
    #[default]
    Volume,
    /// `mean(to / from)`, see [`simple_average_factor`].
    Simple,
}

/// A factor that could not be computed at a given development age.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgeFactorError {
    /// The earlier of the two ages the factor links.
    pub age: usize,
    pub error: FactorError,
}

impl fmt::Display for AgeFactorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "factor from age {}: {}", self.age, self.error)
    }
}

impl std::error::Error for AgeFactorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Age-to-age factors for every pair of adjacent ages in `triangle`.
///
/// Element `k` develops losses from age `k` to age `k + 1`, so the result has
/// `n_ages - 1` elements.
///
/// # Example
///
/// ```
/// use risk_rs::development::{age_to_age_factors, Averaging};
/// use risk_rs::triangle::Triangle;
///
/// let t = Triangle::from_cumulative(vec![
///     vec![100.0, 150.0, 165.0],
///     vec![200.0, 250.0],
///     vec![300.0],
/// ])
/// .unwrap();
/// let f = age_to_age_factors(&t, Averaging::Volume).unwrap();
/// assert_eq!(f, vec![400.0 / 300.0, 1.1]);
/// ```
pub fn age_to_age_factors(
    triangle: &Triangle,
    averaging: Averaging,
) -> Result<Vec<f64>, AgeFactorError> {
    (0..triangle.n_ages().saturating_sub(1))
        .map(|age| {
            let (from, to) = triangle.link_columns(age);
            match averaging {
                Averaging::Volume => volume_weighted_factor(&from, &to),
                Averaging::Simple => simple_average_factor(&from, &to),
            }
            .map_err(|error| AgeFactorError { age, error })
        })
        .collect()
}

/// Age-to-ultimate (cumulative) development factors.
///
/// Element `k` develops losses from age `k` to ultimate: the product of
/// `factors[k..]` times `tail`. The result has `factors.len() + 1` elements;
/// the last one is `tail` itself, applying to the final age.
///
/// ```
/// use risk_rs::development::cumulative_factors;
///
/// let cdf = cumulative_factors(&[2.0, 1.5], 1.1);
/// for (got, want) in cdf.iter().zip([3.3, 1.65, 1.1]) {
///     assert!((got - want).abs() < 1e-12);
/// }
/// ```
pub fn cumulative_factors(factors: &[f64], tail: f64) -> Vec<f64> {
    let mut out = vec![tail; factors.len() + 1];
    for k in (0..factors.len()).rev() {
        out[k] = out[k + 1] * factors[k];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangle::tests::raa;

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
    fn raa_volume_factors() {
        // R ChainLadder: ata(RAA) volume-weighted row.
        let expected = [
            2.999359, 1.623523, 1.270888, 1.171675, 1.113385, 1.041935, 1.033264, 1.016936,
            1.009217,
        ];
        let f = age_to_age_factors(&raa(), Averaging::Volume).unwrap();
        assert_eq!(f.len(), expected.len());
        for (got, want) in f.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn raa_simple_factors() {
        // Mean of the individual RAA link ratios, computed independently.
        let expected = [
            8.206099, 1.695894, 1.314510, 1.182926, 1.126962, 1.043328, 1.034355, 1.017995,
            1.009217,
        ];
        let f = age_to_age_factors(&raa(), Averaging::Simple).unwrap();
        for (got, want) in f.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn age_factor_error_names_the_age() {
        let t = Triangle::from_cumulative(vec![vec![1.0, 0.0, 3.0], vec![1.0, 0.0]]).unwrap();
        assert_eq!(
            age_to_age_factors(&t, Averaging::Volume),
            Err(AgeFactorError {
                age: 1,
                error: FactorError::ZeroDenominator
            })
        );
    }

    #[test]
    fn simple_average_rejects_any_zero() {
        assert_eq!(
            simple_average_factor(&[0.0, 1.0], &[1.0, 2.0]),
            Err(FactorError::ZeroDenominator)
        );
        assert_eq!(simple_average_factor(&[1.0, 2.0], &[2.0, 3.0]), Ok(1.75));
    }

    #[test]
    fn cumulative_factors_without_factors_is_tail() {
        assert_eq!(cumulative_factors(&[], 1.05), vec![1.05]);
    }

    #[test]
    fn rejects_nan() {
        assert_eq!(
            volume_weighted_factor(&[1.0, f64::NAN], &[1.0, 2.0]),
            Err(FactorError::NonFinite)
        );
    }
}
