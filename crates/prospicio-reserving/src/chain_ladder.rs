//! Deterministic chain ladder.

use crate::development::{Development, DevelopmentFit, cumulative_factors};
use crate::error::Result;
use crate::segments::{SegmentFits, fit_each};
use crate::tail::{Tail, TailFit};
use crate::triangle::{Segment, Triangle};
use prospicio_core::{Lag, Period};

/// Chain-ladder method: project each origin's latest value to ultimate with
/// the estimated age-to-age factors and a [`Tail`].
///
/// ```
/// use prospicio_reserving::{ChainLadder, DevelopmentColumn, Grain, Long, Month, Triangle};
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
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ChainLadder {
    /// How the age-to-age factors are estimated.
    pub development: Development,
    /// Development past the oldest age. The default, a constant 1, is no
    /// tail; a number converts to a constant tail.
    pub tail: Tail,
}

/// A fitted chain-ladder projection, per origin.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainLadderFit {
    /// Origin periods, oldest first; every per-origin vector follows them.
    pub origins: Vec<Period>,
    /// The estimated development pattern.
    pub development: DevelopmentFit,
    /// The fitted tail: the selected factors, which the projection uses,
    /// and the factor from the oldest age to ultimate.
    pub tail: TailFit,
    /// Age-to-ultimate factors from the selected factors and the tail;
    /// element `k` develops age `k` to ultimate.
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
        self.fit_segment(&segment, &segment.ages)
    }

    /// Fits `column` in every segment of `triangle`, each on its own; see
    /// [`SegmentFits`] for the long tables. A failure names its segment.
    pub fn fit_segments(
        &self,
        triangle: &Triangle,
        column: &str,
    ) -> Result<SegmentFits<ChainLadderFit>> {
        fit_each(triangle, column, |s| self.fit_segment(s, &s.ages))
    }

    pub(crate) fn fit_segment(&self, segment: &Segment, ages: &[Lag]) -> Result<ChainLadderFit> {
        let mut development = self.development.fit_segment(segment)?;
        development.development = ages.to_vec();
        let tail = self.tail.fit(&development)?;
        let cdf = cumulative_factors(&tail.ldf[..development.ldf.len()], tail.factor);
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
            tail,
            cdf,
            latest_position,
            latest,
            ultimate,
        })
    }
}

impl ChainLadderFit {
    /// Selected age-to-age factors within the triangle, which the projection
    /// uses: the estimated ones, replaced by the tail's from its attachment.
    /// Element `k` links age `k` to `k + 1`.
    pub fn ldf(&self) -> &[f64] {
        &self.tail.ldf[..self.development.ldf.len()]
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

    /// Projected cumulative value of `origin` at every development position
    /// from its latest observation on, with the selected factors and before
    /// the tail: element `k` is position `latest_position + k`.
    pub(crate) fn projection(&self, origin: usize) -> Vec<f64> {
        let ldf = self.ldf();
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
    use crate::error::Error;
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
            tail: 1.05.into(),
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
            tail: 0.0.into(),
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
        use prospicio_core::Month;

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
