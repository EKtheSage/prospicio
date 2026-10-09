//! Dependence between the segments of a multi-segment bootstrap: joint
//! reserves across lines of business, for capital
//! (`docs/design/reserving-v02.md`, decision 9).
//!
//! [`OdpBootstrap::fit_segments`](crate::OdpBootstrap::fit_segments),
//! [`MackBootstrap::fit_segments`](crate::MackBootstrap::fit_segments) and
//! their `one_year_segments` simulate every segment of a triangle into one
//! joint distribution. [`SegmentDependence`] chooses how the segments
//! depend on each other in it:
//!
//! * [`Independent`](SegmentDependence::Independent), the default: each
//!   segment resamples its own residuals on its own; the draws are those of
//!   the bootstrap before the choice existed.
//! * [`Synchronized`](SegmentDependence::Synchronized): every segment
//!   resamples the residuals of the same positions, the synchronous
//!   bootstrap of Taylor and McGuire (2007) and the correlated bootstrap of
//!   Kirschner, Kerley and Isaacs (2008). The dependence comes from the
//!   data: the residuals of one cell (or link) in every line are drawn
//!   together, so the lines' parameter error has the correlation their
//!   residuals show. Process error stays independent.
//! * [`RankCorrelation`](SegmentDependence::RankCorrelation): each segment
//!   is bootstrapped on its own, and then the segments' simulations are
//!   paired by Iman and Conover (1982) on the segment totals, as
//!   Kirschner, Kerley and Isaacs's first approach does, moving each
//!   segment's simulations as whole rows
//!   ([`PredictiveDistribution::reorder_groups`]) so every segment keeps
//!   its own distribution and its joint structure across origins. The
//!   dependence is the user's, as a Spearman matrix.
//!
//! The joint distribution then goes to
//! [`PredictiveDistribution::capital`] for the portfolio's risk measure and
//! its allocation to the lines.

use std::ops::Range;

use prospicio_core::StreamRng;
use prospicio_prob::{KeyValue, PredictiveDistribution};

use crate::bootstrap::components_of;
use crate::error::{Error, Result};
use crate::segments::{ReserveFit, SegmentFits};
use crate::triangle::Segment;

/// How the segments of a multi-segment bootstrap depend on each other;
/// see the [module documentation](crate::dependence).
///
/// It applies to `fit_segments` and `one_year_segments` of
/// [`OdpBootstrap`](crate::OdpBootstrap) and
/// [`MackBootstrap`](crate::MackBootstrap); the single-segment `fit` and
/// `one_year` have no other segment to depend on and ignore it.
///
/// ```
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, OdpBootstrap, ProcessDistribution, SegmentDependence, Triangle};
///
/// // Two lines with the same origins and ages: Home is Auto doubled.
/// let origin = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2023].map(Month::january);
/// let ages = [12, 24, 36, 48, 12, 24, 36, 12, 24, 12];
/// let paid = [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0];
/// let tri = Triangle::from_long(&Long {
///     keys: &[("lob", &[["Auto"; 10], ["Home"; 10]].concat())],
///     origin: &[origin, origin].concat(),
///     development: DevelopmentColumn::Age(&[ages, ages].concat()),
///     values: &[("paid", &[paid, paid.map(|v| v * 2.0)].concat())],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let boot = OdpBootstrap {
///     n_sims: 500,
///     process: ProcessDistribution::None,
///     dependence: SegmentDependence::Synchronized,
///     ..Default::default()
/// };
/// let by_lob = boot.fit_segments(&tri, "paid")?.reserves.aggregate(&["lob"])?;
/// // The same residual positions in both lines: Home's reserve is
/// // Auto's doubled in every simulation.
/// assert!(by_lob.draw_matrix().chunks(2).all(|r| (r[1] - 2.0 * r[0]).abs() < 1e-9 * r[1].abs().max(1.0)));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub enum SegmentDependence {
    /// Each segment resamples its own residuals independently of the
    /// others (with stream `i` for every segment of simulation `i`, in
    /// index order).
    #[default]
    Independent,
    /// Every segment resamples the residuals of the same positions:
    /// whenever a simulation draws a residual for a cell (ODP) or a link
    /// ratio (Mack), it draws one position, uniformly from the positions
    /// where every segment has a residual, and each segment takes its own
    /// residual there. The segments must have the same origins, ages and
    /// observed cells (an error otherwise), so a position means the same
    /// origin and age in each. Where every segment has a residual wherever
    /// any has, each segment's own distribution is unchanged; a position
    /// that only some segments have a residual at is left out of all of
    /// them, and Mack's centring is over the positions kept. Process error
    /// is drawn independently in each segment.
    Synchronized,
    /// Each segment is bootstrapped independently, then the segments'
    /// simulations are reordered as whole rows (Iman–Conover on the segment
    /// totals, [`PredictiveDistribution::reorder_groups`]) so that the
    /// totals have about this Spearman rank correlation. `spearman` is the
    /// `S × S` matrix over the segments in index order, row-major,
    /// symmetric with a unit diagonal. Iman–Conover sets the correlation of
    /// normal scores, so the matrix is converted to
    /// `r = 2 sin(pi rho / 6)`, which must be positive definite. The
    /// reordering's seed is the first number of stream `n_sims` of the
    /// bootstrap's seed, a stream no simulation draws from.
    RankCorrelation {
        /// Spearman's rho between the segments' totals, `S × S`
        /// row-major.
        spearman: Vec<f64>,
    },
}

impl SegmentDependence {
    /// `draws`, simulated for the segments of `fits` (components running
    /// over each segment's origins in turn), reordered to the rank
    /// correlation if that is the choice, and as they are otherwise; `seed`
    /// is the bootstrap's.
    pub(crate) fn reorder<T: ReserveFit>(
        &self,
        draws: PredictiveDistribution,
        fits: &SegmentFits<T>,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        match self {
            Self::RankCorrelation { spearman } => {
                let ranges: Vec<_> = (0..fits.len()).map(|s| components_of(fits, s)).collect();
                rank_correlate(draws, &ranges, spearman, seed)
            }
            Self::Independent | Self::Synchronized => Ok(draws),
        }
    }
}

/// The residuals a bootstrap model of one segment resamples, laid out to
/// be synchronized with other segments'.
pub(crate) trait Resample {
    /// The residuals, row-major over origin × development; NaN where there
    /// is none.
    fn residuals(&self) -> &[f64];

    /// The position (row-major origin × development) of every residual a
    /// simulation draws, in the order it draws them: for the ODP every
    /// observed cell, for Mack every observed link ratio, factor by factor.
    /// Its pool lists the residuals in the same order.
    fn draw_positions(&self) -> Vec<usize>;
}

/// The residuals a simulation resamples, replaceable by a synchronized
/// pool.
pub(crate) trait SharedPool {
    /// Replaces the pool by `residuals` at `positions`, in that order
    /// (centred again if the model centres its pool).
    fn share(&mut self, residuals: &[f64], positions: &[usize]);
}

/// What a synchronized bootstrap draws once per simulation for every
/// segment: one position index for each of `n_draws` residuals, uniform on
/// `0..n_pool`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Shared {
    n_draws: usize,
    n_pool: usize,
}

impl Shared {
    /// The positions of one simulation, drawn from `rng` as an independent
    /// segment draws its residuals.
    pub(crate) fn picks(&self, rng: &mut StreamRng) -> Vec<usize> {
        (0..self.n_draws)
            .map(|_| ((rng.next_open01() * self.n_pool as f64) as usize).min(self.n_pool - 1))
            .collect()
    }
}

/// Checks that `segments` (each with its model) can be synchronized, and
/// returns what each simulation draws and the positions of the shared pool:
/// those of the first segment's draw order where every segment has a
/// residual.
pub(crate) fn synchronize<'a, B: Resample + 'a>(
    segments: impl IntoIterator<Item = (&'a Segment, &'a B)>,
) -> Result<(Shared, Vec<usize>)> {
    let segments: Vec<(&Segment, &B)> = segments.into_iter().collect();
    let (first, model) = segments[0];
    let order = model.draw_positions();
    for &(s, m) in &segments[1..] {
        if s.origins != first.origins || s.ages != first.ages || m.draw_positions() != order {
            return Err(Error::Bootstrap(
                "synchronized segments must have the same origins, ages and observed cells",
            ));
        }
    }
    let positions: Vec<usize> = order
        .iter()
        .copied()
        .filter(|&p| segments.iter().all(|(_, m)| !m.residuals()[p].is_nan()))
        .collect();
    let needs_draws = segments
        .iter()
        .any(|(_, m)| m.residuals().iter().any(|r| !r.is_nan()));
    if positions.is_empty() && needs_draws {
        return Err(Error::Bootstrap(
            "synchronized segments have no position where every segment has a residual",
        ));
    }
    Ok((
        Shared {
            n_draws: order.len(),
            n_pool: positions.len().max(1),
        },
        positions,
    ))
}

/// `draws` with its segments' simulations reordered so that the segment
/// totals have about the Spearman correlation `spearman`; `ranges[s]` are
/// the components of segment `s`, and `seed` the bootstrap's.
fn rank_correlate(
    draws: PredictiveDistribution,
    ranges: &[Range<usize>],
    spearman: &[f64],
    seed: u64,
) -> Result<PredictiveDistribution> {
    let s = ranges.len();
    if spearman.len() != s * s {
        return Err(Error::InvalidSetting {
            name: "spearman",
            value: spearman.len() as f64,
            expected: "a segments × segments matrix",
        });
    }
    let mut normal = vec![1.0; s * s];
    for i in 0..s {
        if spearman[i * s + i] != 1.0 {
            return Err(Error::InvalidSetting {
                name: "spearman",
                value: spearman[i * s + i],
                expected: "a unit diagonal",
            });
        }
        for j in 0..i {
            let (a, b) = (spearman[i * s + j], spearman[j * s + i]);
            if a != b || !(-1.0..=1.0).contains(&a) {
                return Err(Error::InvalidSetting {
                    name: "spearman",
                    value: a,
                    expected: "a symmetric matrix with entries in [-1, 1]",
                });
            }
            let r = 2.0 * (std::f64::consts::PI * a / 6.0).sin();
            normal[i * s + j] = r;
            normal[j * s + i] = r;
        }
    }
    // One group per segment, whatever the triangle's keys.
    let keys = ranges
        .iter()
        .enumerate()
        .flat_map(|(g, r)| {
            r.clone()
                .map(move |j| vec![KeyValue::Int(g as i64), KeyValue::Int(j as i64)])
        })
        .collect();
    let grouped = PredictiveDistribution::from_draws(
        vec!["segment".into(), "component".into()],
        keys,
        draws.draw_matrix().to_vec(),
        draws.provenance().clone(),
    )?;
    let reorder_seed = StreamRng::new(seed, draws.n_sims() as u64).next_u64();
    let reordered = grouped.reorder_groups("segment", &normal, reorder_seed)?;
    let provenance = reordered
        .provenance()
        .clone()
        .param("rank_correlation", format!("{spearman:?}"));
    Ok(PredictiveDistribution::from_draws(
        draws.dims().to_vec(),
        draws.components().to_vec(),
        reordered.draw_matrix().to_vec(),
        provenance,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangle::tests::{GENINS, RAA, raa};
    use crate::{
        ChainLadder, DevelopmentColumn, Grain, Long, MackBootstrap, MackProcess, Month,
        OdpBootstrap, OneYearMethod, ProcessDistribution, Triangle,
    };

    /// A cumulative annual triangle with a `lob` key: each segment's name,
    /// first origin year and rows.
    fn keyed(segments: &[(&str, i32, &[&[f64]])]) -> Triangle {
        let (mut keys, mut origin, mut ages, mut values) = (vec![], vec![], vec![], vec![]);
        for &(name, first, rows) in segments {
            for (k, row) in rows.iter().enumerate() {
                for (d, &v) in row.iter().enumerate() {
                    keys.push(name);
                    origin.push(Month::january(first + k as i32));
                    ages.push(12 * (d as u32 + 1));
                    values.push(v);
                }
            }
        }
        Triangle::from_long(&Long {
            keys: &[("lob", &keys)],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &values)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    fn odp(dependence: SegmentDependence) -> OdpBootstrap {
        OdpBootstrap {
            n_sims: 300,
            seed: 11,
            dependence,
            ..Default::default()
        }
    }

    fn mack(dependence: SegmentDependence) -> MackBootstrap {
        MackBootstrap {
            n_sims: 300,
            seed: 11,
            dependence,
            ..Default::default()
        }
    }

    fn draws(pd: &PredictiveDistribution) -> Vec<f64> {
        pd.draw_matrix().to_vec()
    }

    #[test]
    fn one_synchronized_segment_is_its_independent_bootstrap() {
        // The shared positions are drawn as an independent segment draws its
        // residuals, one uniform per cell (ODP) or link (Mack, where RAA has
        // no zero value or zero sigma) in the same order, into the same
        // pool: with one segment the draws are identical.
        let tri = raa();
        let cl = OneYearMethod::ChainLadder(ChainLadder::default());
        let ind = SegmentDependence::Independent;
        let sync = SegmentDependence::Synchronized;
        assert_eq!(
            draws(
                &odp(ind.clone())
                    .fit_segments(&tri, "values")
                    .unwrap()
                    .reserves
            ),
            draws(
                &odp(sync.clone())
                    .fit_segments(&tri, "values")
                    .unwrap()
                    .reserves
            )
        );
        assert_eq!(
            draws(
                &odp(ind.clone())
                    .one_year_segments(&tri, "values", &cl)
                    .unwrap()
                    .cdr
            ),
            draws(
                &odp(sync.clone())
                    .one_year_segments(&tri, "values", &cl)
                    .unwrap()
                    .cdr
            )
        );
        for process in [MackProcess::Gamma, MackProcess::Residuals] {
            let m = |dependence| MackBootstrap {
                process,
                ..mack(dependence)
            };
            assert_eq!(
                draws(
                    &m(ind.clone())
                        .fit_segments(&tri, "values")
                        .unwrap()
                        .reserves
                ),
                draws(
                    &m(sync.clone())
                        .fit_segments(&tri, "values")
                        .unwrap()
                        .reserves
                )
            );
            assert_eq!(
                draws(
                    &m(ind.clone())
                        .one_year_segments(&tri, "values", &cl)
                        .unwrap()
                        .cdr
                ),
                draws(
                    &m(sync.clone())
                        .one_year_segments(&tri, "values", &cl)
                        .unwrap()
                        .cdr
                )
            );
        }
    }

    #[test]
    fn synchronized_segments_share_their_positions() {
        // GenIns, RAA and GenIns doubled on the same origins: without
        // process error each simulation's pseudo triangles use the same
        // positions, so the doubled line's reserve is GenIns' doubled, draw
        // by draw.
        let doubled: Vec<Vec<f64>> = GENINS
            .iter()
            .map(|r| r.iter().map(|v| 2.0 * v).collect())
            .collect();
        let doubled: Vec<&[f64]> = doubled.iter().map(Vec::as_slice).collect();
        let tri = keyed(&[
            ("a", 2001, &GENINS),
            ("b", 2001, &RAA),
            ("c", 2001, &doubled),
        ]);
        let boot = OdpBootstrap {
            process: ProcessDistribution::None,
            ..odp(SegmentDependence::Synchronized)
        };
        let fit = boot.fit_segments(&tri, "paid").unwrap();
        let by = fit.reserves.aggregate(&["lob"]).unwrap();
        for row in by.draw_matrix().chunks(3) {
            assert!((row[2] / row[0] - 2.0).abs() < 1e-9, "{row:?}");
        }
        assert!(
            fit.reserves
                .provenance()
                .parameters
                .contains(&("dependence".into(), "Synchronized".into()))
        );
        assert!(
            mack(SegmentDependence::Synchronized)
                .fit_segments(&tri, "paid")
                .is_ok()
        );
    }

    #[test]
    fn synchronized_needs_the_same_shape() {
        let shape = Err(Error::Bootstrap(
            "synchronized segments must have the same origins, ages and observed cells",
        ));
        let cl = OneYearMethod::ChainLadder(ChainLadder::default());
        let sync = SegmentDependence::Synchronized;
        // Other origins.
        let tri = keyed(&[("a", 2001, &GENINS), ("b", 1981, &RAA)]);
        assert_eq!(
            odp(sync.clone()).fit_segments(&tri, "paid").map(|_| ()),
            shape
        );
        assert_eq!(
            mack(sync.clone()).fit_segments(&tri, "paid").map(|_| ()),
            shape
        );
        assert_eq!(
            odp(sync.clone())
                .one_year_segments(&tri, "paid", &cl)
                .map(|_| ()),
            shape
        );
        assert_eq!(
            mack(sync.clone())
                .one_year_segments(&tri, "paid", &cl)
                .map(|_| ()),
            shape
        );
        // The same origins and ages, but every origin of the second
        // segment but the oldest stops a diagonal short.
        let short: Vec<&[f64]> = RAA
            .iter()
            .enumerate()
            .map(|(k, r)| if k == 0 { *r } else { &r[..r.len() - 1] })
            .take(9)
            .collect();
        let genins: Vec<&[f64]> = GENINS.iter().take(9).copied().collect();
        let tri = keyed(&[("a", 2001, &genins), ("b", 2001, &short)]);
        assert_eq!(
            odp(sync.clone()).fit_segments(&tri, "paid").map(|_| ()),
            shape
        );
        // Independent and rank-correlated segments may differ in shape.
        assert!(
            odp(SegmentDependence::Independent)
                .fit_segments(&tri, "paid")
                .is_ok()
        );
        let rank = SegmentDependence::RankCorrelation {
            spearman: vec![1.0, 0.5, 0.5, 1.0],
        };
        assert!(odp(rank).fit_segments(&tri, "paid").is_ok());
    }

    #[test]
    fn rank_correlation_checks_its_matrix() {
        let tri = keyed(&[("a", 2001, &GENINS), ("b", 2001, &RAA)]);
        let fit = |spearman: Vec<f64>| {
            odp(SegmentDependence::RankCorrelation { spearman })
                .fit_segments(&tri, "paid")
                .map(|_| ())
        };
        for bad in [
            vec![1.0, 0.5, 0.5],
            vec![0.9, 0.5, 0.5, 1.0],
            vec![1.0, 0.5, 0.4, 1.0],
            vec![1.0, 1.5, 1.5, 1.0],
        ] {
            assert!(
                matches!(
                    fit(bad.clone()),
                    Err(Error::InvalidSetting {
                        name: "spearman",
                        ..
                    })
                ),
                "{bad:?}"
            );
        }
        // A Spearman rho of 1 converts to just below 1, still positive
        // definite; a matrix that is not, after conversion, is refused.
        assert!(fit(vec![1.0, 1.0, 1.0, 1.0]).is_ok());
        let three = keyed(&[("a", 2001, &GENINS), ("b", 2001, &RAA), ("c", 2001, &RAA)]);
        let bad = SegmentDependence::RankCorrelation {
            spearman: vec![1.0, 0.9, -0.9, 0.9, 1.0, 0.9, -0.9, 0.9, 1.0],
        };
        assert!(matches!(
            odp(bad).fit_segments(&three, "paid"),
            Err(Error::Core(_))
        ));
    }

    #[test]
    fn rank_correlation_moves_whole_simulations() {
        let tri = keyed(&[("a", 2001, &GENINS), ("b", 2001, &RAA)]);
        let rank = || {
            odp(SegmentDependence::RankCorrelation {
                spearman: vec![1.0, 0.8, 0.8, 1.0],
            })
            .fit_segments(&tri, "paid")
            .unwrap()
        };
        let ind = odp(SegmentDependence::Independent)
            .fit_segments(&tri, "paid")
            .unwrap();
        let ranked = rank();
        assert_eq!(ranked.reserves.dims(), ind.reserves.dims());
        assert_eq!(ranked.reserves.components(), ind.reserves.components());
        // Each segment's rows (its ten origins) are the independent
        // bootstrap's rows, each used once.
        let n = ind.reserves.n_components();
        for range in [0..10, 10..20] {
            let rows = |pd: &PredictiveDistribution| {
                let mut rows: Vec<Vec<u64>> = pd
                    .draw_matrix()
                    .chunks(n)
                    .map(|r| r[range.clone()].iter().map(|v| v.to_bits()).collect())
                    .collect();
                rows.sort();
                rows
            };
            assert_eq!(rows(&ind.reserves), rows(&ranked.reserves));
        }
        // Reproducible: the reordering's seed comes from the bootstrap's.
        assert_eq!(draws(&rank().reserves), draws(&ranked.reserves));
    }
}
