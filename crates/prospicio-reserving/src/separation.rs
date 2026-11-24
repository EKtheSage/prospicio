//! Taylor's separation method (`reserving-inflation.md`, PR 3, issue #979).
//!
//! Increments divided by a normaliser `n_i` per origin (claim counts,
//! exposure; 1 by default) are `s_ij` with
//!
//! ```text
//! E[s_ij] = r_j lambda_{i+j},     sum_j r_j = 1
//! ```
//!
//! development proportions `r_j` and an index `lambda_h` per calendar
//! period (Taylor 1977). The arithmetic method solves Taylor's equations
//! (5) and (6) backward from the last period `k`, with `v_j` the column
//! sums and `d_h` the diagonal sums of `s`:
//!
//! ```text
//! lambda_h = d_h / (1 - sum_{j>h} r_j),    r_h = v_h / sum_{m>=h} lambda_m
//! ```
//!
//! which are the marginal equations of a Poisson likelihood with
//! development and calendar factors on `s_ij` (Verbeek 1972; decision 8).
//! Future indices are not fitted: they come from a rate, an index or an
//! exponential trend of the last fitted indices ([`FutureInflation`],
//! decision 14), and the future increment of origin `i` at age `j` is
//! `n_i r_j lambda_{i+j}`. On a triangle already restated by an economic
//! index ([`Indexation::restate`](crate::Indexation::restate)) the indices
//! measure superimposed inflation.
//!
//! The method needs a full triangle: origin grain equal to development
//! grain, as many ages as origins, every origin observed from the first age
//! to the latest diagonal with no hole (decision 7).

use crate::error::{Error, Result};
use crate::inflation::{CalendarIndex, FutureCell, FutureCells};
use crate::segments::{SegmentFits, fit_each};
use crate::triangle::{Segment, Triangle};
use prospicio_core::{Lag, Period};

fn sep_err<T>(why: impl Into<String>) -> Result<T> {
    Err(Error::Inflation(format!("separation: {}", why.into())))
}

/// Which separation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SeparationKind {
    /// Taylor's arithmetic method: the backward recursion on column and
    /// diagonal sums.
    #[default]
    Arithmetic,
}

/// The indices of the calendar periods after the triangle's last.
#[derive(Debug, Clone, PartialEq)]
pub enum FutureInflation {
    /// A constant annual rate from the last fitted index, compounded by
    /// the months of each period: `lambda_{k+t} = lambda_k (1 + rate)^(t g /
    /// 12)`, `g` the period's months.
    Rate(f64),
    /// The relative changes of a [`CalendarIndex`] from the triangle's last
    /// calendar period: `lambda_h = lambda_k I(p_h) / I(p_k)`, `p_h` the
    /// index period that holds the end of calendar period `h`. An index
    /// finer than the triangle is read at each period's last sub-period (a
    /// monthly index on annual periods at December), not averaged; give
    /// the index at the triangle's grain for an average.
    Index(CalendarIndex),
    /// The exponential trend of the last `periods` fitted indices (a
    /// least-squares line through their logarithms), continued from the
    /// last fitted index: `lambda_{k+t} = lambda_k exp(b t)`, `b` the
    /// line's slope. `periods` is at least 2 and at most the number of
    /// calendar periods; there is no default (decision 14). Every
    /// continuation starts from `lambda_k`, which rests on one cell (the
    /// last origin's first) and is the noisiest index: check it before
    /// projecting with any of the three.
    Trend {
        /// How many of the last fitted indices the trend is fitted to.
        periods: usize,
    },
}

/// Taylor's (1977) separation method.
///
/// ```
/// use prospicio_reserving::{
///     DevelopmentColumn, FutureInflation, Grain, Long, Month, Separation, SeparationKind,
///     Triangle,
/// };
///
/// // Taylor (1977), Example 1: claims per claim incurred, incremental.
/// let s = [50.4, 28.2, 9.0, 4.8, 58.0, 29.2, 9.7, 59.5, 33.2, 66.2];
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &[2001, 2001, 2001, 2001, 2002, 2002, 2002, 2003, 2003, 2004].map(Month::january),
///     development: DevelopmentColumn::Age(&[12, 24, 36, 48, 12, 24, 36, 12, 24, 12]),
///     values: &[("s", &s)],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: false,
/// })?;
/// let fit = Separation {
///     kind: SeparationKind::Arithmetic,
///     normaliser: None,
///     future: FutureInflation::Rate(0.10),
/// }
/// .fit(&tri, "s")?;
/// assert!((fit.proportions[0] - 0.5835).abs() < 5e-5);
/// assert!((fit.indices[0] - 86.4).abs() < 0.05);
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Separation {
    /// The method.
    pub kind: SeparationKind,
    /// `n_i`, one per origin of the segment, finite and positive; `None`
    /// is 1 for every origin. Taylor takes the claims incurred in the
    /// origin period as estimated at the end of its first development
    /// period.
    pub normaliser: Option<Vec<f64>>,
    /// The indices after the last calendar period.
    pub future: FutureInflation,
}

/// A separation fit of one triangle segment.
#[derive(Debug, Clone, PartialEq)]
pub struct SeparationFit {
    /// The origins; every per-origin vector follows them.
    pub origins: Vec<Period>,
    /// The development ages in months.
    pub ages: Vec<Lag>,
    /// `n_i`, one per origin (1 when none was given).
    pub normaliser: Vec<f64>,
    /// `r_j`, one per age, summing to 1.
    pub proportions: Vec<f64>,
    /// The triangle's calendar periods, the first origin's from its first
    /// age on (at the origins' grain).
    pub calendar: Vec<Period>,
    /// `lambda_h`, one per calendar period.
    pub indices: Vec<f64>,
    /// The calendar periods after the triangle's last up to the last
    /// future cell's, with their indices.
    pub future_indices: Vec<(Period, f64)>,
    /// `n_i r_j lambda_{i+j}` on the observed cells, `None` on the future
    /// ones; one row per origin, one value per age.
    pub fitted: Vec<Vec<Option<f64>>>,
    /// `n_i r_j lambda_{i+j}` on the future cells.
    pub future: FutureCells,
    /// The sum of each origin's future cells. The first origin's is 0: the
    /// method has no tail past the oldest age.
    pub reserves: Vec<f64>,
}

impl SeparationFit {
    /// The total reserve.
    pub fn total_reserve(&self) -> f64 {
        self.reserves.iter().sum()
    }
}

impl Separation {
    /// Fits `column` of a single-segment triangle.
    ///
    /// Fails if the triangle has several segments, is not a full triangle
    /// (decision 7), the normaliser's length is not the number of origins
    /// or a value is not finite and positive, a sum the recursion divides
    /// by is not positive, or the future indices cannot be formed (a rate
    /// at or below -1, an index that misses a period, a trend over fewer
    /// than 2 or more than all periods or through a non-positive index).
    pub fn fit(&self, triangle: &Triangle, column: &str) -> Result<SeparationFit> {
        self.fit_segment(&triangle.segment(column)?)
    }

    /// Fits `column` in every segment of `triangle`. A normaliser applies
    /// to every segment, so each must have as many origins as it has
    /// values.
    ///
    /// Fails as [`fit`](Self::fit) in any segment.
    pub fn fit_segments(
        &self,
        triangle: &Triangle,
        column: &str,
    ) -> Result<SegmentFits<SeparationFit>> {
        fit_each(triangle, column, |s| self.fit_segment(s))
    }

    fn fit_segment(&self, seg: &Segment) -> Result<SeparationFit> {
        let n = seg.n_origins;
        let step = seg.origins[0].grain().months();
        if seg.n_dev != n {
            return sep_err(format!(
                "a full triangle has as many ages as origins; this has {} ages and {n} origins with a value (an origin with no value yet is left out)",
                seg.n_dev
            ));
        }
        if n < 2 {
            return Err(Error::TooFewAges {
                needed: 2,
                found: n,
            });
        }
        if seg.ages[0] != step {
            return sep_err(format!(
                "the first age must be one development period ({step} months), not {}",
                seg.ages[0]
            ));
        }
        if seg.ages[1] - seg.ages[0] != step {
            return sep_err(format!(
                "the development grain ({} months) must be the origin grain ({step} months)",
                seg.ages[1] - seg.ages[0]
            ));
        }
        let normaliser = match &self.normaliser {
            None => vec![1.0; n],
            Some(v) if v.len() != n => {
                return Err(Error::LengthMismatch {
                    column: "normaliser".into(),
                    expected: n,
                    found: v.len(),
                });
            }
            Some(v) => {
                if let Some(&bad) = v.iter().find(|x| !x.is_finite() || **x <= 0.0) {
                    return Err(Error::InvalidSetting {
                        name: "normaliser",
                        value: bad,
                        expected: "finite and positive values",
                    });
                }
                v.clone()
            }
        };

        // s_ij on the observed cells: increments over the normaliser.
        let mut s = vec![vec![0.0; n]; n];
        for (o, row) in s.iter_mut().enumerate() {
            let mut before = 0.0;
            for (d, cell) in row.iter_mut().enumerate() {
                match (seg.get(o, d), d + o < n) {
                    (Some(v), true) => {
                        *cell = (v - before) / normaliser[o];
                        before = v;
                    }
                    (None, false) => {}
                    (Some(_), false) => {
                        return sep_err(format!(
                            "origin {} has a cell past the latest diagonal",
                            seg.origins[o]
                        ));
                    }
                    (None, true) => {
                        return sep_err(format!(
                            "origin {} has no value at age {}: the method needs every cell up to the latest diagonal",
                            seg.origins[o], seg.ages[d]
                        ));
                    }
                }
            }
        }

        let (proportions, indices) = match self.kind {
            SeparationKind::Arithmetic => arithmetic(&s)?,
        };

        let calendar: Vec<Period> = (0..n)
            .map(|h| Period::containing(seg.valuation_of(0, h), seg.origins[0].grain()))
            .collect();
        let future_lambda = self.future_indices(seg, &calendar, &indices)?;
        let lambda = |h: usize| {
            if h < n {
                indices[h]
            } else {
                future_lambda[h - n].1
            }
        };

        let mut fitted = vec![vec![None; n]; n];
        let mut cells = Vec::new();
        let mut reserves = vec![0.0; n];
        for o in 0..n {
            for d in 0..n {
                let value = normaliser[o] * proportions[d] * lambda(o + d);
                if o + d < n {
                    fitted[o][d] = Some(value);
                } else {
                    reserves[o] += value;
                    cells.push(FutureCell {
                        origin: o,
                        age: seg.ages[d],
                        valuation: seg.valuation_of(o, d),
                        value,
                    });
                }
            }
        }
        Ok(SeparationFit {
            origins: seg.origins.clone(),
            ages: seg.ages.clone(),
            normaliser,
            proportions,
            calendar,
            indices,
            future_indices: future_lambda,
            fitted,
            future: FutureCells {
                origins: seg.origins.clone(),
                cells,
            },
            reserves,
        })
    }

    /// The indices of calendar periods `n..2n - 1`.
    fn future_indices(
        &self,
        seg: &Segment,
        calendar: &[Period],
        indices: &[f64],
    ) -> Result<Vec<(Period, f64)>> {
        let n = indices.len();
        let grain = seg.origins[0].grain();
        let last = indices[n - 1];
        let periods: Vec<Period> = (n..2 * n - 1)
            .map(|h| Period::containing(seg.valuation_of(n - 1, h + 1 - n), grain))
            .collect();
        let values: Vec<f64> = match &self.future {
            FutureInflation::Rate(rate) => {
                if !rate.is_finite() || *rate <= -1.0 {
                    return Err(Error::InvalidSetting {
                        name: "future_rate",
                        value: *rate,
                        expected: "finite and above -1",
                    });
                }
                let per_period = (1.0 + rate).powf(grain.months() as f64 / 12.0);
                (1..n).map(|t| last * per_period.powi(t as i32)).collect()
            }
            FutureInflation::Index(index) => {
                let at_last = index.value(calendar[n - 1].end())?;
                periods
                    .iter()
                    .map(|p| Ok(last * index.value(p.end())? / at_last))
                    .collect::<Result<_>>()?
            }
            FutureInflation::Trend { periods: m } => {
                let m = *m;
                if m < 2 || m > n {
                    return Err(Error::InvalidSetting {
                        name: "trend_periods",
                        value: m as f64,
                        expected: "at least 2 and at most the number of calendar periods",
                    });
                }
                let tail = &indices[n - m..];
                if tail.iter().any(|&l| l <= 0.0) {
                    return sep_err("a trend needs positive indices over its periods");
                }
                let slope = log_slope(tail);
                (1..n).map(|t| last * (slope * t as f64).exp()).collect()
            }
        };
        Ok(periods.into_iter().zip(values).collect())
    }
}

/// The least-squares slope of `ln y` on 0, 1, 2, ...
fn log_slope(y: &[f64]) -> f64 {
    let m = y.len() as f64;
    let x_bar = (m - 1.0) / 2.0;
    let y_bar = y.iter().map(|v| v.ln()).sum::<f64>() / m;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (x, v) in y.iter().enumerate() {
        let dx = x as f64 - x_bar;
        sxy += dx * (v.ln() - y_bar);
        sxx += dx * dx;
    }
    sxy / sxx
}

/// Taylor's recursion on the `n` by `n` upper-left triangle of `s`
/// (`s[i][j]` observed where `i + j < n`): the proportions and indices.
fn arithmetic(s: &[Vec<f64>]) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = s.len();
    let column = |j: usize| (0..n - j).map(|i| s[i][j]).sum::<f64>();
    let diagonal = |h: usize| (0..=h).map(|i| s[i][h - i]).sum::<f64>();
    let mut r = vec![0.0; n];
    let mut lambda = vec![0.0; n];
    let (mut r_after, mut lambda_from) = (0.0_f64, 0.0_f64);
    for h in (0..n).rev() {
        let left = 1.0 - r_after;
        if left <= 0.0 || !left.is_finite() {
            return sep_err(format!(
                "the proportions after period {h} sum to {r_after}, at least 1"
            ));
        }
        lambda[h] = diagonal(h) / left;
        if lambda[h] <= 0.0 || !lambda[h].is_finite() {
            return sep_err(format!(
                "calendar period {h}'s increments sum to {}: every diagonal needs a positive sum",
                diagonal(h)
            ));
        }
        lambda_from += lambda[h];
        if lambda_from <= 0.0 || !lambda_from.is_finite() {
            return sep_err(format!(
                "the indices from period {h} on sum to {lambda_from}, not positive"
            ));
        }
        r[h] = column(h) / lambda_from;
        r_after += r[h];
    }
    Ok((r, lambda))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain_ladder::ChainLadder;
    use crate::triangle::tests::{annual, genins};
    use prospicio_core::{Grain, Month};

    /// Taylor (1977), Example 1, cumulated.
    fn taylor() -> Triangle {
        annual(
            2001,
            &[
                &[50.4, 78.6, 87.6, 92.4],
                &[58.0, 87.2, 96.9],
                &[59.5, 92.7],
                &[66.2],
            ],
        )
    }

    fn arithmetic_at(rate: f64) -> Separation {
        Separation {
            kind: SeparationKind::Arithmetic,
            normaliser: None,
            future: FutureInflation::Rate(rate),
        }
    }

    #[test]
    fn taylors_example_1_to_his_digits() {
        let fit = arithmetic_at(0.10).fit(&taylor(), "values").unwrap();
        let r = [0.583476, 0.287767, 0.086615, 0.042142];
        let lambda = [86.3789, 98.9391, 101.9984, 113.9];
        for (got, want) in fit.proportions.iter().zip(r) {
            assert!((got - want).abs() < 5e-7, "{got} {want}");
        }
        for (got, want) in fit.indices.iter().zip(lambda) {
            assert!((got - want).abs() < 5e-5, "{got} {want}");
        }
        assert!((fit.proportions.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        // The recursion's marginals: column and diagonal sums are kept.
        let total: f64 = fit.fitted.iter().flatten().flatten().sum();
        assert!((total - 348.2).abs() < 1e-10);
        let future: Vec<f64> = fit.future.cells.iter().map(|c| c.value).collect();
        let want = [5.3, 10.8, 5.8, 36.1, 11.9, 6.4];
        for (got, want) in future.iter().zip(want) {
            assert!((got - want).abs() <= 0.06, "{got} {want}");
        }
        assert_eq!(
            fit.calendar
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            ["2001", "2002", "2003", "2004"]
        );
        assert_eq!(fit.future_indices.len(), 3);
        assert_eq!(fit.future_indices[2].0, Period::year(2007));
        assert!((fit.total_reserve() - future.iter().sum::<f64>()).abs() < 1e-12);
    }

    #[test]
    fn the_normaliser_leaves_the_proportions_and_scales_the_reserves() {
        let n = vec![1.0, 2.0, 4.0, 8.0];
        let plain = arithmetic_at(0.05).fit(&taylor(), "values").unwrap();
        // Increments multiplied by n_i give back plain's s after division.
        let scaled = annual(
            2001,
            &[
                &[50.4, 78.6, 87.6, 92.4],
                &[116.0, 174.4, 193.8],
                &[238.0, 370.8],
                &[529.6],
            ],
        );
        let fit = Separation {
            normaliser: Some(n),
            ..arithmetic_at(0.05)
        }
        .fit(&scaled, "values")
        .unwrap();
        for (a, b) in fit.proportions.iter().zip(&plain.proportions) {
            assert!((a - b).abs() < 1e-14);
        }
        for (o, (a, b)) in fit.reserves.iter().zip(&plain.reserves).enumerate() {
            if o > 0 {
                assert!((a / b - [1.0, 2.0, 4.0, 8.0][o]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn the_future_by_rate_index_and_trend() {
        let tri = genins();
        let by_rate = arithmetic_at(0.0).fit(&tri, "values").unwrap();
        let last = by_rate.indices[9];
        assert!(by_rate.future_indices.iter().all(|&(_, l)| l == last));

        let flat = CalendarIndex::new(Period::year(2001), vec![1.0; 10])
            .unwrap()
            .with_future_rate(0.0)
            .unwrap();
        let by_index = Separation {
            future: FutureInflation::Index(flat),
            ..arithmetic_at(0.0)
        }
        .fit(&tri, "values")
        .unwrap();
        assert_eq!(by_index.reserves, by_rate.reserves);

        let doubling = CalendarIndex::new(Period::year(2001), vec![1.0; 10])
            .unwrap()
            .with_future_rate(1.0)
            .unwrap();
        let fit = Separation {
            future: FutureInflation::Index(doubling),
            ..arithmetic_at(0.0)
        }
        .fit(&tri, "values")
        .unwrap();
        assert!((fit.future_indices[2].1 - 8.0 * last).abs() < 1e-9 * last);
        assert_eq!(fit.future_indices[2].0, Period::year(2013));

        let trend = Separation {
            future: FutureInflation::Trend { periods: 10 },
            ..arithmetic_at(0.0)
        }
        .fit(&tri, "values")
        .unwrap();
        let b = log_slope(&trend.indices);
        assert!((trend.future_indices[0].1 - last * b.exp()).abs() < 1e-9 * last);
        for m in [0, 1, 11] {
            let bad = Separation {
                future: FutureInflation::Trend { periods: m },
                ..arithmetic_at(0.0)
            };
            assert!(bad.fit(&tri, "values").is_err());
        }
    }

    #[test]
    fn an_exactly_separable_triangle_is_recovered() {
        // Increments r_j lambda_{i+j} with lambda growing 7% a year.
        let r = [0.5, 0.3, 0.15, 0.05];
        let rows: Vec<Vec<f64>> = (0..4)
            .map(|i| {
                let mut c = 0.0;
                (0..4 - i)
                    .map(|j| {
                        c += r[j] * 100.0 * 1.07_f64.powi((i + j) as i32);
                        c
                    })
                    .collect()
            })
            .collect();
        let rows: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        let fit = arithmetic_at(0.07)
            .fit(&annual(2001, &rows), "values")
            .unwrap();
        for (j, want) in r.iter().enumerate() {
            assert!((fit.proportions[j] - want).abs() < 1e-12);
        }
        for (h, l) in fit.indices.iter().enumerate() {
            assert!((l - 100.0 * 1.07_f64.powi(h as i32)).abs() < 1e-9);
        }
        for cell in &fit.future.cells {
            let (i, j) = (cell.origin, (cell.age / 12 - 1) as usize);
            let want = r[j] * 100.0 * 1.07_f64.powi((i + j) as i32);
            assert!((cell.value - want).abs() < 1e-9 * want);
        }
    }

    #[test]
    fn a_diagonal_must_sum_to_more_than_zero() {
        // The middle diagonal (2002) sums to 0, then to -1, with a positive
        // last diagonal; then the last diagonal sums to 0.
        let cases: [[&[f64]; 3]; 3] = [
            [&[10.0, 10.0, 15.0], &[0.0, 5.0], &[4.0]],
            [&[10.0, 9.5, 15.0], &[-0.5, 5.0], &[4.0]],
            [&[10.0, 15.0, 15.0], &[11.0, 11.0], &[0.0]],
        ];
        for rows in cases {
            let tri = annual(2001, &rows);
            let err = arithmetic_at(0.0).fit(&tri, "values").unwrap_err();
            assert!(matches!(err, Error::Inflation(_)), "{err}");
        }
    }

    #[test]
    fn one_segment_per_fit_and_every_segment_with_fit_segments() {
        let stacked = Triangle::stack(
            &[("A", &taylor()), ("B", &taylor())],
            "program",
            &["values"],
        )
        .unwrap();
        assert!(matches!(
            arithmetic_at(0.1).fit(&stacked, "values"),
            Err(Error::MultipleSegments(2))
        ));
        let fits = arithmetic_at(0.1).fit_segments(&stacked, "values").unwrap();
        assert_eq!(fits.len(), 2);
        let alone = arithmetic_at(0.1).fit(&taylor(), "values").unwrap();
        for (_, fit) in fits.iter() {
            assert_eq!(fit, &alone);
        }
    }

    #[test]
    fn it_needs_a_full_triangle() {
        let quarterly = Triangle::from_long(&crate::triangle::Long {
            keys: &[],
            origin: &[
                Month::january(2001),
                Month::january(2001),
                Month::january(2002),
            ],
            development: crate::triangle::DevelopmentColumn::Age(&[3, 6, 3]),
            values: &[("values", &[1.0, 2.0, 1.0])],
            origin_grain: Grain::Year,
            development_grain: Grain::Quarter,
            cumulative: true,
        })
        .unwrap();
        assert!(arithmetic_at(0.0).fit(&quarterly, "values").is_err());
        // Ages from 24 on: the first increment would hold two years.
        let late = Triangle::from_long(&crate::triangle::Long {
            keys: &[],
            origin: &[2001, 2001, 2002].map(Month::january),
            development: crate::triangle::DevelopmentColumn::Age(&[24, 36, 24]),
            values: &[("values", &[1.0, 2.0, 1.0])],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        assert!(arithmetic_at(0.0).fit(&late, "values").is_err());
        let short = Separation {
            normaliser: Some(vec![1.0; 3]),
            ..arithmetic_at(0.0)
        };
        assert!(matches!(
            short.fit(&taylor(), "values"),
            Err(Error::LengthMismatch { .. })
        ));
        let zero = Separation {
            normaliser: Some(vec![1.0, 0.0, 1.0, 1.0]),
            ..arithmetic_at(0.0)
        };
        assert!(zero.fit(&taylor(), "values").is_err());
        assert!(arithmetic_at(-1.0).fit(&taylor(), "values").is_err());
        // A chain ladder needs no full triangle; separation does.
        let ragged = annual(2001, &[&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &[1.0]]);
        assert!(ChainLadder::default().fit(&ragged, "values").is_ok());
        assert!(arithmetic_at(0.0).fit(&ragged, "values").is_err());
    }
}
