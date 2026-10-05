//! Deterministic chain ladder.

use crate::development::{Development, DevelopmentFit, cumulative_factors};
use crate::error::{Error, Result};
use crate::triangle::{Segment, Triangle};
use act_core::Period;

/// Chain-ladder method: project each origin's latest value to ultimate with
/// the estimated age-to-age factors and a tail factor.
///
/// ```
/// use act_reserving::{ChainLadder, DevelopmentColumn, Grain, Long, Month, Triangle};
///
/// let origin = [2020, 2020, 2021].map(Month::january);
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 12]),
///     values: &[("paid", &[100.0, 150.0, 200.0])],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })
/// .unwrap();
/// let cl = ChainLadder::default().fit(&tri, "paid").unwrap();
/// assert_eq!(cl.ultimate, vec![150.0, 300.0]);
/// assert_eq!(cl.total_reserve(), 100.0);
/// assert_eq!(cl.origins[1].to_string(), "2021");
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChainLadder {
    /// How the age-to-age factors are estimated.
    pub development: Development,
    /// Factor from the oldest age to ultimate; 1 means no tail.
    pub tail: f64,
}

impl Default for ChainLadder {
    fn default() -> Self {
        Self {
            development: Development::default(),
            tail: 1.0,
        }
    }
}

/// A fitted chain-ladder projection, per origin.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainLadderFit {
    /// Origin periods, oldest first; every per-origin vector follows them.
    pub origins: Vec<Period>,
    /// The estimated development pattern.
    pub development: DevelopmentFit,
    /// Tail factor from the oldest age to ultimate.
    pub tail: f64,
    /// Age-to-ultimate factors; element `k` develops age `k` to ultimate.
    pub cdf: Vec<f64>,
    /// Development position of each origin's latest observation.
    pub latest_position: Vec<usize>,
    /// Latest observed cumulative value per origin.
    pub latest: Vec<f64>,
    /// Projected ultimate per origin.
    pub ultimate: Vec<f64>,
}

impl ChainLadder {
    /// Fits `column` of a single-segment triangle.
    pub fn fit(&self, triangle: &Triangle, column: &str) -> Result<ChainLadderFit> {
        let segment = triangle.segment(column)?;
        let mut fit = self.fit_segment(&segment)?;
        fit.development.development = triangle.development().to_vec();
        Ok(fit)
    }

    pub(crate) fn fit_segment(&self, segment: &Segment) -> Result<ChainLadderFit> {
        if !self.tail.is_finite() || self.tail <= 0.0 {
            return Err(Error::InvalidTail(self.tail));
        }
        let development = self.development.fit_segment(segment)?;
        let cdf = cumulative_factors(&development.ldf, self.tail);
        let (latest_position, latest): (Vec<usize>, Vec<f64>) = (0..segment.n_origins)
            .map(|o| segment.latest(o))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .unzip();
        let ultimate = latest_position
            .iter()
            .zip(&latest)
            .map(|(&d, &v)| v * cdf[d])
            .collect();
        Ok(ChainLadderFit {
            origins: segment.origins.clone(),
            development,
            tail: self.tail,
            cdf,
            latest_position,
            latest,
            ultimate,
        })
    }
}

impl ChainLadderFit {
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

    /// Projected cumulative value of `origin` at every development position
    /// from its latest observation on, before the tail: element `k` is
    /// position `latest_position + k`.
    pub(crate) fn projection(&self, origin: usize) -> Vec<f64> {
        let ldf = &self.development.ldf;
        let mut values = vec![self.latest[origin]];
        for f in &ldf[self.latest_position[origin]..] {
            values.push(values[values.len() - 1] * f);
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::development::Average;
    use crate::triangle::tests::{annual, raa};

    fn close(got: f64, want: f64, tol: f64) {
        assert!((got - want).abs() < tol, "got {got}, want {want}");
    }

    #[test]
    fn raa_totals() {
        // R ChainLadder: summary(MackChainLadder(RAA))$Totals.
        let cl = ChainLadder::default().fit(&raa(), "values").unwrap();
        close(cl.total_ultimate(), 213_122.23, 0.01);
        close(cl.total_reserve(), 52_135.23, 0.01);
        assert_eq!(cl.reserves()[0], 0.0);
        assert_eq!(cl.cdf.len(), 10);
        assert_eq!(cl.origins.len(), 10);
    }

    #[test]
    fn tail_scales_every_ultimate() {
        let t = raa();
        let base = ChainLadder::default().fit(&t, "values").unwrap();
        let tailed = ChainLadder {
            tail: 1.05,
            ..Default::default()
        }
        .fit(&t, "values")
        .unwrap();
        for (a, b) in base.ultimate.iter().zip(&tailed.ultimate) {
            close(*b, a * 1.05, 1e-6);
        }
    }

    #[test]
    fn rejects_bad_tail() {
        let cl = ChainLadder {
            tail: 0.0,
            ..Default::default()
        };
        assert_eq!(cl.fit(&raa(), "values"), Err(Error::InvalidTail(0.0)));
    }

    #[test]
    fn simple_average_projection() {
        let t = annual(2020, &[&[100.0, 200.0], &[100.0, 100.0], &[50.0]]);
        let cl = ChainLadder {
            development: Development {
                average: Average::Simple,
                ..Default::default()
            },
            ..Default::default()
        }
        .fit(&t, "values")
        .unwrap();
        assert_eq!(cl.ultimate, vec![200.0, 100.0, 75.0]);
        assert_eq!(cl.projection(2), vec![50.0, 75.0]);
    }

    #[test]
    fn hole_in_a_row_uses_latest_observation() {
        use crate::triangle::{DevelopmentColumn, Long};
        use crate::{Grain, Triangle};
        use act_core::Month;

        // 2020 is unobserved at 24 months, so it does not inform either
        // factor, but its value at 36 months is its latest.
        let origin = [2019, 2019, 2019, 2020, 2020, 2021, 2021, 2022].map(Month::january);
        let t = Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 24, 36, 12, 36, 12, 24, 12]),
            values: &[("paid", &[1.0, 2.0, 4.0, 1.0, 3.0, 1.0, 2.0, 1.0])],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let cl = ChainLadder::default().fit(&t, "paid").unwrap();
        assert_eq!(cl.development.ldf, vec![2.0, 2.0]);
        assert_eq!(cl.latest_position, vec![2, 2, 1, 0]);
        assert_eq!(cl.ultimate, vec![4.0, 3.0, 4.0, 4.0]);
    }
}
