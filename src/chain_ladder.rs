//! Chain-ladder reserving and Mack's standard errors.
//!
//! **Provisional:** this module will be replaced by `act-reserving` as
//! described in `docs/architecture.md`. Do not build on this API.

use std::fmt;

use crate::development::{AgeFactorError, Averaging, age_to_age_factors, cumulative_factors};
use crate::triangle::Triangle;

/// Why a chain-ladder projection could not be made.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChainLadderError {
    /// An age-to-age factor could not be computed.
    Factor(AgeFactorError),
    /// The tail factor is not finite and positive.
    InvalidTail(f64),
    /// Mack's method needs at least two development factors to estimate the
    /// variance parameters.
    TooFewAges,
}

impl fmt::Display for ChainLadderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Factor(e) => e.fmt(f),
            Self::InvalidTail(t) => write!(f, "tail factor {t} is not finite and positive"),
            Self::TooFewAges => write!(f, "Mack's method needs at least three development ages"),
        }
    }
}

impl std::error::Error for ChainLadderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Factor(e) => Some(e),
            _ => None,
        }
    }
}

impl From<AgeFactorError> for ChainLadderError {
    fn from(e: AgeFactorError) -> Self {
        Self::Factor(e)
    }
}

/// A deterministic chain-ladder projection.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainLadder {
    /// Age-to-age factors; element `k` develops age `k` to age `k + 1`.
    pub factors: Vec<f64>,
    /// Tail factor from the last age to ultimate.
    pub tail: f64,
    /// Age-to-ultimate factors; element `k` develops age `k` to ultimate.
    pub cdf: Vec<f64>,
    /// Latest observed cumulative value per origin.
    pub latest: Vec<f64>,
    /// Projected ultimate per origin.
    pub ultimate: Vec<f64>,
}

impl ChainLadder {
    /// Projects `triangle` to ultimate with volume-weighted factors and no tail.
    ///
    /// # Example
    ///
    /// ```
    /// use risk_rs::chain_ladder::ChainLadder;
    /// use risk_rs::triangle::Triangle;
    ///
    /// let t = Triangle::from_cumulative(vec![
    ///     vec![100.0, 150.0],
    ///     vec![200.0],
    /// ])
    /// .unwrap();
    /// let cl = ChainLadder::fit(&t).unwrap();
    /// assert_eq!(cl.ultimate, vec![150.0, 300.0]);
    /// assert_eq!(cl.total_reserve(), 100.0);
    /// ```
    pub fn fit(triangle: &Triangle) -> Result<Self, ChainLadderError> {
        Self::fit_with(triangle, Averaging::Volume, 1.0)
    }

    /// Projects `triangle` to ultimate with the given averaging and tail.
    pub fn fit_with(
        triangle: &Triangle,
        averaging: Averaging,
        tail: f64,
    ) -> Result<Self, ChainLadderError> {
        if !tail.is_finite() || tail <= 0.0 {
            return Err(ChainLadderError::InvalidTail(tail));
        }
        let factors = age_to_age_factors(triangle, averaging)?;
        let cdf = cumulative_factors(&factors, tail);
        let latest = triangle.latest();
        let ultimate = latest
            .iter()
            .zip(triangle.latest_ages())
            .map(|(value, age)| value * cdf[age])
            .collect();
        Ok(Self {
            factors,
            tail,
            cdf,
            latest,
            ultimate,
        })
    }

    /// Reserve (ultimate minus latest) per origin, often labelled IBNR.
    pub fn reserves(&self) -> Vec<f64> {
        self.ultimate
            .iter()
            .zip(&self.latest)
            .map(|(u, l)| u - l)
            .collect()
    }

    /// Total projected ultimate across origins.
    pub fn total_ultimate(&self) -> f64 {
        self.ultimate.iter().sum()
    }

    /// Total reserve across origins.
    pub fn total_reserve(&self) -> f64 {
        self.total_ultimate() - self.latest.iter().sum::<f64>()
    }
}

/// Mack's (1993) distribution-free chain ladder: the volume-weighted
/// projection plus the mean squared error of each reserve.
#[derive(Debug, Clone, PartialEq)]
pub struct Mack {
    /// The underlying chain-ladder projection.
    pub chain_ladder: ChainLadder,
    /// Variance parameters `sigma_k^2`, one per age-to-age factor. The last
    /// one is extrapolated when only one origin informs it.
    pub sigma2: Vec<f64>,
    /// Standard error of each origin's reserve.
    pub standard_errors: Vec<f64>,
    /// Standard error of the total reserve, including the correlation
    /// between origins that share estimated factors.
    pub total_standard_error: f64,
}

impl Mack {
    /// Fits Mack's model to `triangle` with no tail.
    ///
    /// Needs at least three development ages so the last variance parameter
    /// can be extrapolated.
    pub fn fit(triangle: &Triangle) -> Result<Self, ChainLadderError> {
        let chain_ladder = ChainLadder::fit(triangle)?;
        let factors = &chain_ladder.factors;
        let n = factors.len();
        if n < 2 {
            return Err(ChainLadderError::TooFewAges);
        }

        // Column sums of `from` values each factor was estimated on.
        let mut volume = Vec::with_capacity(n);
        let mut sigma2: Vec<Option<f64>> = Vec::with_capacity(n);
        for (age, &f) in factors.iter().enumerate() {
            let (from, to) = triangle.link_columns(age);
            volume.push(from.iter().sum::<f64>());
            sigma2.push((from.len() > 1).then(|| {
                let ss: f64 = from
                    .iter()
                    .zip(&to)
                    .map(|(c, d)| c * (d / c - f).powi(2))
                    .sum();
                ss / (from.len() - 1) as f64
            }));
        }
        let sigma2 = fill_sigma2(sigma2)?;

        // Projected cumulative values: observed where known, else chained.
        let latest_ages = triangle.latest_ages();
        let projected: Vec<Vec<f64>> = triangle
            .rows()
            .iter()
            .map(|row| {
                let mut values = row.clone();
                for k in row.len() - 1..n {
                    values.push(values[k] * factors[k]);
                }
                values
            })
            .collect();

        // Per-factor weight in the parameter-error term.
        let weight: Vec<f64> = (0..n).map(|k| sigma2[k] / factors[k].powi(2)).collect();

        let mse: Vec<f64> = projected
            .iter()
            .zip(&latest_ages)
            .map(|(values, &a)| {
                let ultimate = values[n];
                let sum: f64 = (a..n)
                    .map(|k| weight[k] * (1.0 / values[k] + 1.0 / volume[k]))
                    .sum();
                ultimate.powi(2) * sum
            })
            .collect();

        // Covariance between origins through the shared factor estimates.
        let mut total_mse: f64 = mse.iter().sum();
        for i in 0..projected.len() {
            for j in i + 1..projected.len() {
                let start = latest_ages[i].max(latest_ages[j]);
                let shared: f64 = (start..n).map(|k| 2.0 * weight[k] / volume[k]).sum();
                total_mse += projected[i][n] * projected[j][n] * shared;
            }
        }

        Ok(Self {
            chain_ladder,
            sigma2,
            standard_errors: mse.iter().map(|m| m.sqrt()).collect(),
            total_standard_error: total_mse.sqrt(),
        })
    }

    /// Coefficient of variation of the total reserve.
    pub fn total_cv(&self) -> f64 {
        self.total_standard_error / self.chain_ladder.total_reserve()
    }
}

/// Replaces variance parameters estimated from a single origin using Mack's
/// log-linear rule `sigma_k^2 = min(s_{k-1}^4 / s_{k-2}^2, s_{k-2}^2, s_{k-1}^2)`.
fn fill_sigma2(sigma2: Vec<Option<f64>>) -> Result<Vec<f64>, ChainLadderError> {
    let mut out = Vec::with_capacity(sigma2.len());
    for value in sigma2 {
        let filled = match value {
            Some(v) => v,
            None => {
                let [.., a, b] = out[..] else {
                    return Err(ChainLadderError::TooFewAges);
                };
                let floor = f64::min(a, b);
                if a == 0.0 {
                    floor
                } else {
                    f64::min(b * b / a, floor)
                }
            }
        };
        out.push(filled);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangle::tests::raa;

    fn close(got: f64, want: f64, tol: f64) {
        assert!((got - want).abs() < tol, "got {got}, want {want}");
    }

    #[test]
    fn raa_chain_ladder_totals() {
        // R ChainLadder: summary(MackChainLadder(RAA))$Totals.
        let cl = ChainLadder::fit(&raa()).unwrap();
        close(cl.total_ultimate(), 213_122.23, 0.01);
        close(cl.total_reserve(), 52_135.23, 0.01);
        assert_eq!(cl.reserves()[0], 0.0);
    }

    #[test]
    fn raa_mack_standard_error() {
        // R ChainLadder: MackChainLadder(RAA) total Mack S.E. 26,909.01.
        let mack = Mack::fit(&raa()).unwrap();
        close(mack.total_standard_error, 26_909.01, 0.01);
        close(mack.total_cv(), 0.5161, 1e-4);
        assert_eq!(mack.standard_errors[0], 0.0);
    }

    #[test]
    fn tail_scales_every_ultimate() {
        let t = raa();
        let base = ChainLadder::fit(&t).unwrap();
        let tailed = ChainLadder::fit_with(&t, Averaging::Volume, 1.05).unwrap();
        for (a, b) in base.ultimate.iter().zip(&tailed.ultimate) {
            close(*b, a * 1.05, 1e-6);
        }
    }

    #[test]
    fn rejects_bad_tail() {
        assert_eq!(
            ChainLadder::fit_with(&raa(), Averaging::Volume, 0.0),
            Err(ChainLadderError::InvalidTail(0.0))
        );
    }

    #[test]
    fn mack_needs_three_ages() {
        let t = Triangle::from_cumulative(vec![vec![1.0, 2.0], vec![1.0, 3.0], vec![1.0]]).unwrap();
        assert_eq!(Mack::fit(&t), Err(ChainLadderError::TooFewAges));
    }

    #[test]
    fn sigma2_extrapolation() {
        assert_eq!(
            fill_sigma2(vec![Some(4.0), Some(2.0), None]),
            Ok(vec![4.0, 2.0, 1.0])
        );
        assert_eq!(
            fill_sigma2(vec![Some(1.0), Some(2.0), None]),
            Ok(vec![1.0, 2.0, 1.0])
        );
    }
}
