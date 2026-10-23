//! Deterministic chain ladder.

use crate::development::{Development, DevelopmentFit, cumulative_factors};
use crate::error::{Error, Result};
use crate::pattern::{PatternKind, PatternWeight, SelectedPattern};
use crate::segments::{SegmentFits, fit_each};
use crate::tail::{Tail, TailFit};
use crate::triangle::{Segment, Triangle};
use prospicio_core::{Lag, Period};

/// Chain-ladder method: project each origin's latest value to ultimate with
/// the estimated age-to-age factors and a [`Tail`].
///
/// With a selected pattern ([`Development::pattern`]) the projection uses
/// the selected factors where it gives them, blended with the estimate by
/// their weight; the estimate is still reported
/// ([`ChainLadderFit::development`]).
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
#[derive(Debug, Clone, PartialEq, Default)]
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
    /// The estimated development pattern, with the selected factors and
    /// their weights when there is a pattern.
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
    /// For a cumulative selected pattern (factors to ultimate or percents
    /// developed), its given factor to ultimate at its last age over the
    /// factor to ultimate in use there (the factors from that age on and
    /// the tail). It reconciles the pattern's level at its last age only:
    /// the given percent developed there times it is the projection's
    /// (`1 / cdf` there), and, ending at the oldest age with a tail, the
    /// rescaled percent developed there times the tail factor is 100%. At
    /// an earlier age, the given percent developed times it is the
    /// projection's only when every link from that age to the last is
    /// fully selected (weight 1); a blended link uses `w * selected + (1 -
    /// w) * estimated`, which the given values do not show. 1 when the
    /// pattern ends at the oldest age with no tail, since its level is then
    /// the tail ([`pattern_tail`](Self::pattern_tail)). `None` for an
    /// age-to-age pattern, which stops at its last factor and lets the tail
    /// multiply on top, or without a pattern.
    pub pattern_rescale: Option<f64>,
    /// The tail a cumulative selected pattern implies: with no [`Tail`]
    /// (a constant 1 at the oldest age), a pattern of factors to ultimate
    /// or percents developed that reaches the oldest age keeps its level
    /// there as the tail, as chainladder-python's `DevelopmentConstant`
    /// does. The tail factor is the pattern's factor to ultimate at the
    /// oldest age (1 over its percent developed there), a constant
    /// ([`TailConstant`](crate::TailConstant) at the oldest age) with no
    /// parameter error: [`tail`](Self::tail)'s `std_err` is 0, its `sigma`
    /// extrapolated as a constant tail's. `None` with a tail (the level is
    /// rescaled to it), for a pattern that stops before the oldest age (its
    /// level is rescaled to the estimated factors after it), for an
    /// age-to-age pattern, or without a pattern.
    pub pattern_tail: Option<f64>,
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

    /// The tail in use on a segment with development `ages`: the given
    /// one, or with none the constant one a cumulative pattern to the
    /// oldest age implies ([`ChainLadderFit::pattern_tail`]).
    pub(crate) fn tail_in_use(&self, ages: &[Lag]) -> Result<Tail> {
        if !self.tail.is_none() {
            return Ok(self.tail);
        }
        Ok(match &self.development.pattern {
            Some(p) => p
                .resolve(ages)?
                .implied_tail(ages)
                .map_or(self.tail, Tail::from),
            None => self.tail,
        })
    }

    pub(crate) fn fit_segment(&self, segment: &Segment, ages: &[Lag]) -> Result<ChainLadderFit> {
        let mut development = self.development.fit_segment(segment)?;
        development.development = ages.to_vec();
        let resolved = match &self.development.pattern {
            Some(p) => Some(p.resolve(ages)?),
            None => None,
        };
        // With no tail, a cumulative pattern to the oldest age keeps its
        // level there as the tail: a constant factor, which carries no
        // parameter error (its sigma is extrapolated as a constant tail's).
        let pattern_tail = resolved
            .as_ref()
            .filter(|_| self.tail.is_none())
            .and_then(|r| r.implied_tail(ages));
        let mut tail = pattern_tail
            .map_or(self.tail, Tail::from)
            .fit(&development)?;
        if pattern_tail.is_some() {
            tail.std_err = 0.0;
        }
        // A factor is the pattern's or the tail's, never both.
        if let Some(k) = (tail.attachment..development.selected.len())
            .find(|&k| development.selected[k].is_some())
        {
            return Err(Error::Pattern {
                age: Some(ages[k]),
                reason: "the factor is selected and replaced by the tail: \
                         select factors before the tail's attachment age only",
            });
        }
        let cdf = cumulative_factors(&tail.ldf[..development.ldf.len()], tail.factor);
        // The given level of a cumulative pattern at its last age, against
        // the factor to ultimate in use there.
        let pattern_rescale = resolved
            .and_then(|r| r.level)
            .map(|(last, given)| given / cdf[last]);
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
            pattern_rescale,
            pattern_tail,
        })
    }
}

impl ChainLadderFit {
    /// The tail this fit used, given it was fitted with `tail`: `tail`, or
    /// the constant one its pattern implied ([`pattern_tail`](Self::pattern_tail)).
    pub(crate) fn tail_in_use(&self, tail: Tail) -> Tail {
        self.pattern_tail.map_or(tail, Tail::from)
    }

    /// Age-to-age factors within the triangle, which the projection uses:
    /// the estimated ones (blended with a selected pattern where it gives
    /// them), replaced by the tail's from its attachment. Element `k` links
    /// age `k` to `k + 1`.
    pub fn ldf(&self) -> &[f64] {
        &self.tail.ldf[..self.development.ldf.len()]
    }

    /// The development this fit projects with, frozen as a chain ladder to
    /// apply to another book or to a latest diagonal (`reserving-v02.md`,
    /// decision 17): chainladder-python's
    /// `Chainladder().fit(source).predict(target)`.
    ///
    /// It holds a fully selected pattern of age-to-age factors, the factors
    /// in use ([`ldf`](Self::ldf)) at the fit's ages: estimated or blended
    /// with a selection under the fit's average and exclusions, and an
    /// estimated tail's own factors from its attachment age on. Its tail is
    /// the fit's tail factor as a constant at the oldest age
    /// (`Tail::from(self.tail.factor)`). Its other settings are the
    /// defaults, with no `n_periods` and no drops, as
    /// [`ChainLadder::fit_diagonal`] requires.
    ///
    /// Apply it to a latest diagonal with [`ChainLadder::fit_diagonal`], and
    /// to another triangle through its latest diagonal,
    /// [`LatestDiagonal::from_triangle`](crate::LatestDiagonal::from_triangle).
    /// For the expected-loss methods, put it in their `chain_ladder` field:
    /// Bornhuetter-Ferguson and Benktander apply their own apriori to the
    /// target's exposure, and Cape Cod estimates its apriori on the target.
    /// The tail is never refitted on the target.
    ///
    /// On the fit's own ages its link factors are the fit's to the bit; the
    /// tail is spread again by the constant tail's decay, so the factors to
    /// ultimate and the ultimates agree to about 1e-15 relative, and to the
    /// bit when the tail factor is 1. It does not carry sigma, the standard
    /// errors or an estimated tail's form, and a fit applied with it has no
    /// standard error. A tail factor below exp(-0.75), about 0.4724, cannot
    /// be spread at the constant tail's decay of 0.5, and a fit with one age
    /// has no factors, so fitting the result is then an error.
    ///
    /// Apply it to any book other than the fit's own through that book's
    /// latest diagonal, which reads ages between the fit's by its
    /// log-linear rule and refuses ages outside them. The plain triangle path
    /// (`fit`) is right only on a target whose development ages are the
    /// fit's, and has two hazards that give no error: it reads each factor
    /// from its age to the target's next age, so a target with a finer
    /// development grain is misread, and on a target with ages past the
    /// fit's oldest it estimates the target's own factors there and attaches
    /// the fit's tail at the target's oldest age, which mixes the target's
    /// development into the fit's. A target without one of the fit's ages
    /// is an error ([`Error::Pattern`]).
    ///
    /// ```
    /// use prospicio_reserving::{
    ///     ChainLadder, DevelopmentColumn, Grain, LatestDiagonal, Long, Month, PatternKind, Tail,
    ///     Triangle,
    /// };
    ///
    /// // A source book with a factor of 1.5 from 12 to 24 months.
    /// let origin = [2020, 2020, 2021].map(Month::january);
    /// let source = Triangle::from_long(&Long {
    ///     keys: &[],
    ///     origin: &origin,
    ///     development: DevelopmentColumn::Age(&[12, 24, 12]),
    ///     values: &[("paid", &[100.0, 150.0, 200.0])],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })
    /// .unwrap();
    /// let fit = ChainLadder { tail: 1.1.into(), ..Default::default() }
    ///     .fit(&source, "paid")
    ///     .unwrap();
    /// let frozen = fit.to_chain_ladder();
    /// let pattern = frozen.development.pattern.as_ref().unwrap();
    /// assert_eq!(pattern.kind, PatternKind::AgeToAge);
    /// assert_eq!(pattern.values, [(12, 1.5)]);
    /// assert_eq!(frozen.tail, Tail::from(1.1));
    ///
    /// // Another book's diagonal at 31 December 2024.
    /// let diagonal = LatestDiagonal::new(
    ///     Grain::Year,
    ///     &[2023, 2024].map(Month::january),
    ///     &[24, 12],
    ///     &[400.0, 300.0],
    /// )
    /// .unwrap();
    /// let applied = frozen.fit_diagonal(&diagonal).unwrap();
    /// assert_eq!(applied.ldf(), [1.5]);
    /// // 400 × 1.1 and 300 × 1.5 × 1.1, each spread over the tail's periods.
    /// assert!((applied.ultimate[0] - 440.0).abs() < 1e-9);
    /// assert!((applied.ultimate[1] - 495.0).abs() < 1e-9);
    /// ```
    pub fn to_chain_ladder(&self) -> ChainLadder {
        let ages = &self.development.development;
        ChainLadder {
            development: Development {
                pattern: Some(SelectedPattern {
                    kind: PatternKind::AgeToAge,
                    values: ages
                        .iter()
                        .copied()
                        .zip(self.ldf().iter().copied())
                        .collect(),
                    weight: PatternWeight::All(1.0),
                }),
                ..Default::default()
            },
            tail: Tail::from(self.tail.factor),
        }
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
