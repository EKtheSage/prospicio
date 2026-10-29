//! Dependence between the segments of a multi-segment bootstrap: joint
//! reserves across lines of business, for capital
//! (`docs/design/reserving-v02.md`, decision 10).
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
use prospicio_math::linalg::cholesky;
use prospicio_prob::portfolio::CorrelationKind;
use prospicio_prob::{KeyValue, PredictiveDistribution};

use crate::bootstrap::components_of;
use crate::error::{Error, Result};
use crate::pattern_draws::DevelopmentPatterns;
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
    /// `r = 2 sin(pi rho / 6)`, which must be positive definite. The paired
    /// rows are then put in a random order, since Iman–Conover leaves them
    /// sorted by the first segment's total: row `i` is no longer simulation
    /// `i`, but the rows are exchangeable, so any subset of them is a fair
    /// sample. Both use stream `n_sims` of the bootstrap's seed, a stream no
    /// simulation draws from: its first number seeds the reordering, the
    /// rest shuffle the rows.
    RankCorrelation {
        /// Spearman's rho between the segments' totals, `S × S`
        /// row-major.
        spearman: Vec<f64>,
    },
}

impl SegmentDependence {
    /// Checks the choice against `n_segments` segments before any
    /// simulation: a [`RankCorrelation`](Self::RankCorrelation) matrix must
    /// be `n_segments × n_segments`, symmetric with a unit diagonal, and
    /// positive definite after the map to normal scores.
    #[cfg_attr(not(feature = "bayes"), allow(dead_code))] // only the Bayesian fits call it
    pub(crate) fn validate(&self, n_segments: usize) -> Result<()> {
        match self {
            Self::RankCorrelation { spearman } => normal_scores(spearman, n_segments).map(|_| ()),
            Self::Independent | Self::Synchronized => Ok(()),
        }
    }

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

    /// As [`reorder`](Self::reorder), with each segment's development
    /// patterns moved with its reserves (`reserving-v02.md` decision 28):
    /// the reserves are `reorder`'s, and row `i` of a segment's patterns is
    /// the simulation its reserves in row `i` came from.
    pub(crate) fn reorder_with<T: ReserveFit>(
        &self,
        draws: PredictiveDistribution,
        fits: &SegmentFits<T>,
        seed: u64,
        patterns: &mut DevelopmentPatterns,
    ) -> Result<PredictiveDistribution> {
        match self {
            Self::RankCorrelation { spearman } => {
                let ranges: Vec<_> = (0..fits.len()).map(|s| components_of(fits, s)).collect();
                let (draws, source) = rank_correlate_indexed(draws, &ranges, spearman, seed)?;
                for (s, source) in source.iter().enumerate() {
                    patterns.permute(s, source);
                }
                Ok(draws)
            }
            Self::Independent | Self::Synchronized => Ok(draws),
        }
    }

    /// As [`reorder`](Self::reorder), with the components of segment `s`
    /// given as `ranges[s]` instead of read from the fits, so that a
    /// segment can move other components with its reserves (Meyers'
    /// ultimates); its total is the sum over its range.
    #[cfg(feature = "bayes")]
    pub(crate) fn reorder_ranges(
        &self,
        draws: PredictiveDistribution,
        ranges: &[Range<usize>],
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        match self {
            Self::RankCorrelation { spearman } => rank_correlate(draws, ranges, spearman, seed),
            Self::Independent | Self::Synchronized => Ok(draws),
        }
    }

    /// As [`reorder_ranges`](Self::reorder_ranges), with each segment's
    /// development patterns moved with its rows (`reserving-v02.md`
    /// decision 28).
    #[cfg(feature = "bayes")]
    pub(crate) fn reorder_ranges_with(
        &self,
        draws: PredictiveDistribution,
        ranges: &[Range<usize>],
        seed: u64,
        patterns: &mut DevelopmentPatterns,
    ) -> Result<PredictiveDistribution> {
        match self {
            Self::RankCorrelation { spearman } => {
                let (draws, source) = rank_correlate_indexed(draws, ranges, spearman, seed)?;
                for (s, source) in source.iter().enumerate() {
                    patterns.permute(s, source);
                }
                Ok(draws)
            }
            Self::Independent | Self::Synchronized => Ok(draws),
        }
    }

    /// The simulations of several segments, `rows` (one row of `width`
    /// values per simulation, segment `s` holding the columns `ranges[s]`),
    /// reordered to the rank correlation if that is the choice, and as they
    /// are otherwise. Iman–Conover ranks segment `s` by the sum of its
    /// columns `ranked[s]` alone (its total reserve), and all of the
    /// segment's columns in `ranges[s]` then move with it, so a segment's
    /// values stay consistent row by row. `seed` is the fit's.
    #[cfg(feature = "bayes")]
    pub(crate) fn reorder_rows(
        &self,
        rows: Vec<f64>,
        width: usize,
        ranges: &[Range<usize>],
        ranked: &[Range<usize>],
        seed: u64,
    ) -> Result<Vec<f64>> {
        match self {
            Self::RankCorrelation { spearman } => {
                rank_correlate_rows(&rows, width, ranges, ranked, spearman, seed)
            }
            Self::Independent | Self::Synchronized => Ok(rows),
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

/// The normal-score correlation `r = 2 sin(pi rho / 6)` of the Spearman
/// matrix `spearman` over `s` segments, after checking that it is `s × s`,
/// symmetric with a unit diagonal and entries in `[-1, 1]`, and that `r` is
/// positive definite.
fn normal_scores(spearman: &[f64], s: usize) -> Result<Vec<f64>> {
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
    if cholesky(&normal, s).is_none() {
        return Err(Error::InvalidSetting {
            name: "spearman",
            value: f64::NAN,
            expected: "a matrix whose 2 sin(pi rho / 6) is positive definite",
        });
    }
    Ok(normal)
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
    Ok(rank_correlate_indexed(draws, ranges, spearman, seed)?.0)
}

/// [`rank_correlate`], and where each row came from: `source[s][i]` is the
/// simulation of the input whose segment `s` row `i` of the result holds.
fn rank_correlate_indexed(
    draws: PredictiveDistribution,
    ranges: &[Range<usize>],
    spearman: &[f64],
    seed: u64,
) -> Result<(PredictiveDistribution, Vec<Vec<usize>>)> {
    let normal = normal_scores(spearman, ranges.len())?;
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
    // Stream `n_sims`, which no simulation draws from: its first number
    // seeds Iman–Conover, the rest shuffle the rows.
    let n_sims = draws.n_sims();
    let mut rng = StreamRng::new(seed, n_sims as u64);
    // `reorder_groups` is this with its index dropped.
    let (reordered, index) = grouped.reorder_groups_indexed(
        "segment",
        &normal,
        CorrelationKind::Pearson,
        rng.next_u64(),
    )?;
    let order = shuffled_order(n_sims, &mut rng);
    let n = draws.n_components();
    let paired = reordered.draw_matrix();
    let rows = order
        .iter()
        .flat_map(|&r| paired[r * n..(r + 1) * n].iter().copied())
        .collect();
    let source = index
        .iter()
        .map(|index| order.iter().map(|&r| index[r]).collect())
        .collect();
    let provenance = reordered
        .provenance()
        .clone()
        .param("rank_correlation", format!("{spearman:?}"));
    let draws = PredictiveDistribution::from_draws(
        draws.dims().to_vec(),
        draws.components().to_vec(),
        rows,
        provenance,
    )?;
    Ok((draws, source))
}

/// Iman–Conover leaves the first segment's rows sorted by its total, so row
/// `i` would rank the simulations rather than stand for one: the paired rows
/// are put in a random order (Fisher–Yates, from `rng`), which keeps the
/// joint law.
fn shuffled_order(n_sims: usize, rng: &mut StreamRng) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n_sims).collect();
    for i in (1..n_sims).rev() {
        let j = ((rng.next_open01() * (i + 1) as f64) as usize).min(i);
        order.swap(i, j);
    }
    order
}

/// `rows` (`width` values per simulation, segment `s` in the columns
/// `ranges[s]`) with the segments' simulations reordered as whole rows of
/// their own columns, so that the sums of `ranked[s]` (the segment totals)
/// have about the Spearman correlation `spearman`, then the rows shuffled
/// as in [`rank_correlate`] (stream `n_sims` of `seed`).
#[cfg(feature = "bayes")]
fn rank_correlate_rows(
    rows: &[f64],
    width: usize,
    ranges: &[Range<usize>],
    ranked: &[Range<usize>],
    spearman: &[f64],
    seed: u64,
) -> Result<Vec<f64>> {
    let s = ranges.len();
    let normal = normal_scores(spearman, s)?;
    let n_sims = rows.len() / width;
    let totals: Vec<f64> = rows
        .chunks_exact(width)
        .flat_map(|row| {
            ranked
                .iter()
                .map(move |r| row[r.clone()].iter().sum::<f64>())
        })
        .collect();
    let grouped = PredictiveDistribution::from_draws(
        vec!["segment".into(), "component".into()],
        (0..s)
            .map(|g| vec![KeyValue::Int(g as i64), KeyValue::Int(0)])
            .collect(),
        totals.clone(),
        prospicio_prob::Provenance::new("rank_correlation_rows"),
    )?;
    let mut rng = StreamRng::new(seed, n_sims as u64);
    let reordered = grouped.reorder_groups("segment", &normal, rng.next_u64())?;
    let moved = reordered.draw_matrix();
    // Iman–Conover returns each segment's totals in a new order. The k-th
    // smallest total of the output is the k-th smallest of the input, so
    // matching them by rank (ties by position) finds the input row each
    // output row came from.
    let by_total = |values: &dyn Fn(usize) -> f64| {
        let mut order: Vec<usize> = (0..n_sims).collect();
        order.sort_by(|&a, &b| values(a).total_cmp(&values(b)).then(a.cmp(&b)));
        order
    };
    let source: Vec<Vec<usize>> = (0..s)
        .map(|g| {
            let from = by_total(&|i| totals[i * s + g]);
            let to = by_total(&|i| moved[i * s + g]);
            let mut source = vec![0; n_sims];
            for (&out, &input) in to.iter().zip(&from) {
                source[out] = input;
            }
            source
        })
        .collect();
    let order = shuffled_order(n_sims, &mut rng);
    let mut out = vec![0.0; rows.len()];
    for (i, &r) in order.iter().enumerate() {
        for (g, range) in ranges.iter().enumerate() {
            let from = source[g][r] * width;
            out[i * width + range.start..i * width + range.end]
                .copy_from_slice(&rows[from + range.start..from + range.end]);
        }
    }
    Ok(out)
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
            bad.validate(3),
            Err(Error::InvalidSetting { name: "spearman", expected, .. })
                if expected.contains("positive definite")
        ));
        assert!(matches!(
            odp(bad).fit_segments(&three, "paid"),
            Err(Error::InvalidSetting {
                name: "spearman",
                ..
            })
        ));
        // `validate` checks the size against the segments, before any
        // simulation (`BayesOdp::fit_segments` calls it first).
        let two = SegmentDependence::RankCorrelation {
            spearman: vec![1.0, 0.5, 0.5, 1.0],
        };
        assert!(two.validate(2).is_ok());
        assert!(two.validate(3).is_err());
        assert!(SegmentDependence::Independent.validate(3).is_ok());
    }

    #[test]
    fn rank_correlation_moves_whole_simulations() {
        // For both bootstraps, lifetime and one-year views: each segment's
        // rows (its ten origins) are the independent bootstrap's rows, each
        // used once, in another order.
        let tri = keyed(&[("a", 2001, &GENINS), ("b", 2001, &RAA)]);
        let cl = OneYearMethod::ChainLadder(ChainLadder::default());
        let rank = SegmentDependence::RankCorrelation {
            spearman: vec![1.0, 0.8, 0.8, 1.0],
        };
        type Run<'a> = Box<dyn Fn(SegmentDependence) -> PredictiveDistribution + 'a>;
        let runs: [Run; 4] = [
            Box::new(|d| odp(d).fit_segments(&tri, "paid").unwrap().reserves),
            Box::new(|d| mack(d).fit_segments(&tri, "paid").unwrap().reserves),
            Box::new(|d| odp(d).one_year_segments(&tri, "paid", &cl).unwrap().cdr),
            Box::new(|d| mack(d).one_year_segments(&tri, "paid", &cl).unwrap().cdr),
        ];
        for run in &runs {
            let ind = run(SegmentDependence::Independent);
            let ranked = run(rank.clone());
            assert_eq!(ranked.dims(), ind.dims());
            assert_eq!(ranked.components(), ind.components());
            assert_ne!(draws(&ranked), draws(&ind));
            let n = ind.n_components();
            assert_eq!(n, 20);
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
                assert_eq!(rows(&ind), rows(&ranked));
            }
            // Reproducible: the reordering's seed comes from the bootstrap's.
            assert_eq!(draws(&run(rank.clone())), draws(&ranked));
        }
    }

    /// Rows of two segments with three columns each, the first the one
    /// ranked and the others functions of it: reordering moves the whole
    /// row, so the functions still hold, and the rows are a permutation.
    #[cfg(feature = "bayes")]
    #[test]
    fn whole_rows_move_with_the_ranked_columns() {
        let n_sims = 400;
        let mut rng = StreamRng::new(9, 0);
        let mut rows = Vec::new();
        for _ in 0..n_sims {
            for scale in [1.0, 10.0] {
                let x = scale * rng.next_open01();
                rows.extend([x, 2.0 * x + 1.0, -x]);
            }
        }
        let ranges = [0..3, 3..6];
        let ranked = [0..1, 3..4];
        let rank = SegmentDependence::RankCorrelation {
            spearman: vec![1.0, 0.7, 0.7, 1.0],
        };
        let out = rank
            .reorder_rows(rows.clone(), 6, &ranges, &ranked, 5)
            .unwrap();
        assert_eq!(out.len(), rows.len());
        for row in out.chunks_exact(6) {
            for s in [0, 3] {
                assert_eq!(row[s + 1], 2.0 * row[s] + 1.0);
                assert_eq!(row[s + 2], -row[s]);
            }
        }
        for s in [0, 3] {
            let column = |v: &[f64]| {
                let mut c: Vec<f64> = v.chunks_exact(6).map(|r| r[s]).collect();
                c.sort_by(f64::total_cmp);
                c
            };
            assert_eq!(column(&rows), column(&out));
        }
        let x: Vec<f64> = out.chunks_exact(6).map(|r| r[0]).collect();
        let y: Vec<f64> = out.chunks_exact(6).map(|r| r[3]).collect();
        let rho = {
            let ranks = |v: &[f64]| {
                let mut idx: Vec<usize> = (0..v.len()).collect();
                idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
                let mut r = vec![0.0; v.len()];
                for (k, &i) in idx.iter().enumerate() {
                    r[i] = k as f64;
                }
                r
            };
            let (rx, ry) = (ranks(&x), ranks(&y));
            let m = (n_sims as f64 - 1.0) / 2.0;
            let sxy: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - m) * (b - m)).sum();
            let sxx: f64 = rx.iter().map(|a| (a - m).powi(2)).sum();
            sxy / sxx
        };
        assert!((rho - 0.7).abs() < 0.06, "{rho}");
        // Independent leaves the rows alone.
        let same = SegmentDependence::Independent
            .reorder_rows(rows.clone(), 6, &ranges, &ranked, 5)
            .unwrap();
        assert_eq!(same, rows);
    }

    #[test]
    fn rank_correlated_rows_are_in_a_random_order() {
        // Iman–Conover alone leaves the first segment's rows sorted by its
        // total (its column of target ranks is the identity), so row `i`
        // would rank the simulations: a prefix of them would be biased, and
        // two seeds would pair their rows in common order. After the shuffle
        // the row index is unrelated to every segment's total: its Spearman
        // rho with each is within 4 standard errors (1 / sqrt(n - 1)) of 0.
        let tri = keyed(&[("a", 2001, &GENINS), ("b", 2001, &RAA)]);
        let ranked = odp(SegmentDependence::RankCorrelation {
            spearman: vec![1.0, 0.8, 0.8, 1.0],
        })
        .fit_segments(&tri, "paid")
        .unwrap()
        .reserves;
        let (n, k) = (ranked.n_sims(), ranked.n_components());
        for range in [0..10, 10..20] {
            let totals: Vec<f64> = ranked
                .draw_matrix()
                .chunks(k)
                .map(|r| r[range.clone()].iter().sum())
                .collect();
            assert!(totals.windows(2).any(|w| w[0] > w[1]));
            let mut by_total: Vec<usize> = (0..n).collect();
            by_total.sort_by(|&a, &b| totals[a].total_cmp(&totals[b]));
            let mut rank = vec![0.0; n];
            for (r, &i) in by_total.iter().enumerate() {
                rank[i] = r as f64;
            }
            let mean = (n - 1) as f64 / 2.0;
            let num: f64 = (0..n).map(|i| (i as f64 - mean) * (rank[i] - mean)).sum();
            let den: f64 = (0..n).map(|i| (i as f64 - mean).powi(2)).sum();
            let rho = num / den;
            assert!(rho.abs() < 4.0 / ((n - 1) as f64).sqrt(), "{rho}");
        }
    }
}
