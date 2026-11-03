//! The Berquist-Sherman adjustments (Berquist and Sherman, 1977; Friedland,
//! *Estimating Unpaid Claims Using Basic Techniques*, chapter 13;
//! `reserving-v02.md` decision 34): a triangle restated before it is
//! projected, so that the chain ladder, the expected-loss methods or Mack
//! run on it unchanged.
//!
//! - [`CaseAdequacy`] restates incurred claims at the case reserve adequacy
//!   of the latest diagonal: the average case outstanding of each age on
//!   the latest diagonal, detrended to the earlier origins, times the open
//!   counts, plus paid.
//! - [`SettlementRate`] restates paid claims and closed counts at the
//!   settlement rate of the latest diagonal: closed counts are the latest
//!   diagonal's disposal rates times each origin's ultimate counts, and
//!   paid claims are read off a curve through each origin's observed
//!   (closed count, paid) pairs at those counts.
//! - [`BerquistSherman`] does both, so that incurred is the adjusted
//!   average case times the adjusted open counts plus the adjusted paid.
//!
//! Each returns a cumulative copy of the triangle with the adjusted
//! columns replaced; the latest diagonal is unchanged.

use crate::error::{Error, Result};
use crate::frequency_severity::{
    Rows, Selection, UltimateCounts, last_observed, origin_years, pooled_trend, ratio_column,
    valuation_positions,
};
use crate::triangle::Triangle;
use prospicio_core::Lag;

fn bs_err<T>(why: impl Into<String>) -> Result<T> {
    Err(Error::Counts(why.into()))
}

/// Where the open claim counts come from.
#[derive(Debug, Clone, PartialEq)]
pub enum OpenCounts {
    /// A column of open counts.
    Column(String),
    /// Reported less closed counts (chainladder-python's convention).
    ReportedLessClosed { reported: String, closed: String },
}

impl OpenCounts {
    fn rows(&self, tri: &Triangle) -> Result<Rows> {
        match self {
            Self::Column(c) => tri.cells(c),
            Self::ReportedLessClosed { reported, closed } => {
                let r = tri.cells(reported)?;
                let c = tri.cells(closed)?;
                Ok(r.iter()
                    .zip(&c)
                    .map(|(r, c)| {
                        r.iter()
                            .zip(c)
                            .map(|(r, c)| Some(r.as_ref()? - c.as_ref()?))
                            .collect()
                    })
                    .collect())
            }
        }
    }
}

/// The annual severity trend that detrends the average case outstanding.
#[derive(Debug, Clone, PartialEq)]
pub enum CaseTrend {
    /// A selected trend, as Berquist and Sherman (15%) and Friedland (5%)
    /// select it by judgment.
    Given(f64),
    /// The exponential trend of `amount` over `counts` fitted over every
    /// age at once ([`pooled_trend`]): paid claims over closed counts, the
    /// severity Berquist and Sherman compare with the average case. The
    /// average case outstanding itself carries the change in adequacy being
    /// removed, so its own trend is not a sound choice; pass it explicitly
    /// to use it.
    Fitted { amount: String, counts: String },
}

impl CaseTrend {
    fn rate(&self, tri: &Triangle) -> Result<f64> {
        let rate = match self {
            Self::Given(r) => *r,
            Self::Fitted { amount, counts } => {
                let t = ratio_column(tri, amount, counts, "__severity")?;
                match pooled_trend(&t, "__severity")? {
                    Some(fit) => fit.rate,
                    None => {
                        return bs_err(
                            "no severity trend can be fitted: fewer than two origins at every age",
                        );
                    }
                }
            }
        };
        if !rate.is_finite() || rate <= -1.0 {
            return Err(Error::InvalidSetting {
                name: "trend",
                value: rate,
                expected: "finite and above -1",
            });
        }
        Ok(rate)
    }
}

/// The Berquist-Sherman case reserve adequacy adjustment.
///
/// The average case outstanding of a cell is `(incurred - paid) / open`.
/// The adjusted average at origin `o` and age `k` is the latest diagonal's
/// average at age `k` (origin `o*`) divided by `(1 + trend)` for each year
/// from `o` to `o*`; adjusted incurred is that times the open counts plus
/// paid. Ages after `max_age` keep their own average, so their incurred is
/// unchanged (Friedland leaves XYZ Insurer's ages past 84 months as they
/// are). A cell needs incurred, paid and open counts; where the average
/// cannot be formed (no open claim on the latest diagonal) the case
/// outstanding is taken as zero.
#[derive(Debug, Clone, PartialEq)]
pub struct CaseAdequacy {
    pub incurred: String,
    pub paid: String,
    pub open_counts: OpenCounts,
    pub trend: CaseTrend,
    /// The oldest age adjusted; `None` adjusts every age.
    pub max_age: Option<Lag>,
}

/// The Berquist-Sherman settlement rate adjustment.
///
/// Disposal rates are closed counts over ultimate counts. The adjusted
/// closed count of every cell up to the valuation is the selected disposal
/// rate of its age (by default the latest diagonal's) times the origin's
/// ultimate count. The adjusted paid of a cell is read off the curve through
/// two successive observed (closed, paid) points of its origin whose closed
/// counts span the adjusted count, or, when none does, the nearest pair;
/// Berquist and Sherman fit the exponential `paid = a exp(b closed)` through
/// them, and a straight line is the alternative. Each origin's latest cell
/// keeps its observed paid.
#[derive(Debug, Clone, PartialEq)]
pub struct SettlementRate {
    pub paid: String,
    pub closed_counts: String,
    pub ultimate_counts: UltimateCounts,
    /// The disposal rate of each age; Berquist and Sherman take the latest
    /// diagonal's ([`Selection::LatestDiagonal`]), which leaves the latest
    /// diagonal of the closed counts and paid claims unchanged.
    pub disposal_rates: Selection,
    pub interpolation: Interpolation,
}

/// The curve through two successive (closed count, paid) points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// `paid = a exp(b closed)`, Berquist and Sherman's choice.
    #[default]
    Exponential,
    /// A straight line.
    Linear,
}

/// Both Berquist-Sherman adjustments (Friedland, chapter 13, XYZ Insurer's
/// third projection): paid claims and closed counts as [`SettlementRate`]
/// adjusts them, and incurred claims as the adjusted average case
/// outstanding of [`CaseAdequacy`] times the adjusted open counts (reported
/// less adjusted closed) plus the adjusted paid. The case adjustment's open
/// counts must be [`OpenCounts::ReportedLessClosed`], and its closed column
/// the settlement adjustment's.
#[derive(Debug, Clone, PartialEq)]
pub struct BerquistSherman {
    pub case_adequacy: CaseAdequacy,
    pub settlement_rate: SettlementRate,
}

/// Cell-by-cell adjusted average case outstanding of a [`CaseAdequacy`]:
/// rows per origin, `None` where it cannot be formed. Also returns the open
/// counts and the paid claims it was formed from.
struct Averages {
    adjusted: Rows,
    open: Rows,
    paid: Rows,
    incurred: Rows,
}

impl CaseAdequacy {
    fn averages(&self, tri: &Triangle) -> Result<Averages> {
        let trend = self.trend.rate(tri)?;
        let incurred = tri.cells(&self.incurred)?;
        let paid = tri.cells(&self.paid)?;
        let open = self.open_counts.rows(tri)?;
        let origins = tri.origins();
        let ages = tri.development();
        let years: Vec<f64> = origins.iter().map(|&o| origin_years(o)).collect();
        let average: Rows = (0..origins.len())
            .map(|o| {
                (0..ages.len())
                    .map(|d| {
                        let (i, p, n) = (incurred[o][d]?, paid[o][d]?, open[o][d]?);
                        (n != 0.0).then(|| (i - p) / n)
                    })
                    .collect()
            })
            .collect();
        let val = valuation_positions(tri);
        // The latest diagonal's origin at each age.
        let diag: Vec<Option<usize>> = (0..ages.len())
            .map(|d| val.iter().position(|&v| v == Some(d)))
            .collect();
        let adjusted = (0..origins.len())
            .map(|o| {
                (0..ages.len())
                    .map(|d| {
                        if self.max_age.is_some_and(|m| ages[d] > m) {
                            return average[o][d];
                        }
                        let star = diag[d]?;
                        let a = average[star][d]?;
                        Some(a / (1.0 + trend).powf(years[star] - years[o]))
                    })
                    .collect()
            })
            .collect();
        Ok(Averages {
            adjusted,
            open,
            paid,
            incurred,
        })
    }

    /// The average case outstanding, as observed (column `average_case`)
    /// and adjusted (`adjusted_average_case`), added to a cumulative copy of
    /// `tri`: Friedland's "adjusted average case outstanding" triangle.
    pub fn average_case(&self, tri: &Triangle) -> Result<Triangle> {
        let a = self.averages(tri)?;
        let observed: Rows = a
            .incurred
            .iter()
            .zip(&a.paid)
            .zip(&a.open)
            .map(|((i, p), n)| {
                i.iter()
                    .zip(p)
                    .zip(n)
                    .map(|((i, p), n)| {
                        let n = (*n)?;
                        let (i, p) = ((*i)?, (*p)?);
                        (n != 0.0).then(|| (i - p) / n)
                    })
                    .collect()
            })
            .collect();
        tri.with_cells("average_case", &observed)?
            .with_cells("adjusted_average_case", &a.adjusted)
    }

    /// `tri` with the incurred column restated at the latest diagonal's case
    /// reserve adequacy, cumulative. Cells without paid claims or open
    /// counts become unobserved.
    pub fn adjust(&self, tri: &Triangle) -> Result<Triangle> {
        let a = self.averages(tri)?;
        let rows = combine(&a.adjusted, &a.open, &a.paid);
        tri.with_cells(&self.incurred, &rows)
    }
}

/// Adjusted average times open plus paid, cell by cell; zero case where the
/// average is missing.
fn combine(average: &Rows, open: &Rows, paid: &Rows) -> Rows {
    average
        .iter()
        .zip(open)
        .zip(paid)
        .map(|((a, n), p)| {
            a.iter()
                .zip(n)
                .zip(p)
                .map(|((a, n), p)| {
                    let (n, p) = ((*n)?, (*p)?);
                    Some(p + if n == 0.0 { 0.0 } else { a.unwrap_or(0.0) * n })
                })
                .collect()
        })
        .collect()
}

/// The result of a settlement rate adjustment, before it is written back.
struct Settled {
    closed: Rows,
    paid: Rows,
}

impl SettlementRate {
    fn settle(&self, tri: &Triangle) -> Result<Settled> {
        let closed = tri.cells(&self.closed_counts)?;
        let paid = tri.cells(&self.paid)?;
        let ultimate = self.ultimate_counts.resolve(tri)?;
        let ages = tri.development();
        let (no, nd) = (closed.len(), ages.len());
        let rates: Rows = closed
            .iter()
            .zip(&ultimate)
            .map(|(r, &u)| {
                r.iter()
                    .map(|c| c.filter(|_| u > 0.0).map(|c| c / u))
                    .collect()
            })
            .collect();
        let val = valuation_positions(tri);
        let latest: Vec<Option<usize>> = closed.iter().map(|r| last_observed(r)).collect();
        let selected = self
            .disposal_rates
            .select(&rates, ages, &latest, "disposal rates")?;
        let mut adj_closed = vec![vec![None; nd]; no];
        let mut adj_paid = vec![vec![None; nd]; no];
        for o in 0..no {
            let Some(last) = val[o] else {
                continue;
            };
            // The observed (closed, paid) pairs of successive ages.
            let pairs: Vec<((f64, f64), (f64, f64))> = (0..nd.saturating_sub(1))
                .filter_map(|d| {
                    let p0 = (closed[o][d]?, paid[o][d]?);
                    let p1 = (closed[o][d + 1]?, paid[o][d + 1]?);
                    usable(p0, p1, self.interpolation).then_some((p0, p1))
                })
                .collect();
            for d in 0..=last {
                if latest[o] == Some(d) {
                    adj_closed[o][d] = closed[o][d];
                    adj_paid[o][d] = paid[o][d];
                    continue;
                }
                let Some(rate) = selected[d] else {
                    continue;
                };
                let x = rate * ultimate[o];
                adj_closed[o][d] = Some(x);
                adj_paid[o][d] = interpolate(&pairs, x, self.interpolation);
            }
        }
        Ok(Settled {
            closed: adj_closed,
            paid: adj_paid,
        })
    }

    /// `tri` with the paid and closed count columns restated at the latest
    /// diagonal's settlement rate, cumulative. Every cell up to the
    /// valuation is filled where its age has a disposal rate and its origin
    /// a curve, including ages before the origin's first observed count.
    pub fn adjust(&self, tri: &Triangle) -> Result<Triangle> {
        let s = self.settle(tri)?;
        tri.with_cells(&self.closed_counts, &s.closed)?
            .with_cells(&self.paid, &s.paid)
    }
}

/// Whether the curve through two points is defined.
fn usable(p0: (f64, f64), p1: (f64, f64), how: Interpolation) -> bool {
    p0.0 != p1.0
        && match how {
            Interpolation::Exponential => p0.1 > 0.0 && p1.1 > 0.0,
            Interpolation::Linear => true,
        }
}

/// Two successive (closed count, paid) points.
type Segment = ((f64, f64), (f64, f64));

/// The paid claims at `x` closed counts on the pair whose closed counts span
/// `x` (the first one), else on the nearest pair.
fn interpolate(pairs: &[Segment], x: f64, how: Interpolation) -> Option<f64> {
    let distance = |((x0, _), (x1, _)): &Segment| {
        let (lo, hi) = (x0.min(*x1), x0.max(*x1));
        if x < lo {
            lo - x
        } else if x > hi {
            x - hi
        } else {
            0.0
        }
    };
    let pair = pairs
        .iter()
        .min_by(|a, b| distance(a).total_cmp(&distance(b)))?;
    let ((x0, y0), (x1, y1)) = *pair;
    Some(match how {
        Interpolation::Exponential => {
            let b = (y1.ln() - y0.ln()) / (x1 - x0);
            y0 * (b * (x - x0)).exp()
        }
        Interpolation::Linear => y0 + (y1 - y0) * (x - x0) / (x1 - x0),
    })
}

impl BerquistSherman {
    /// `tri` with incurred, paid and closed counts all restated, cumulative.
    pub fn adjust(&self, tri: &Triangle) -> Result<Triangle> {
        let OpenCounts::ReportedLessClosed { reported, closed } = &self.case_adequacy.open_counts
        else {
            return bs_err(
                "both Berquist-Sherman adjustments need open counts as reported less closed counts",
            );
        };
        if *closed != self.settlement_rate.closed_counts {
            return bs_err(format!(
                "the case adjustment's closed counts ({closed}) are not the settlement adjustment's ({})",
                self.settlement_rate.closed_counts
            ));
        }
        if self.case_adequacy.paid != self.settlement_rate.paid {
            return bs_err(format!(
                "the case adjustment's paid claims ({}) are not the settlement adjustment's ({})",
                self.case_adequacy.paid, self.settlement_rate.paid
            ));
        }
        let a = self.case_adequacy.averages(tri)?;
        let s = self.settlement_rate.settle(tri)?;
        let reported_rows = tri.cells(reported)?;
        let open: Rows = reported_rows
            .iter()
            .zip(&s.closed)
            .map(|(r, c)| {
                r.iter()
                    .zip(c)
                    .map(|(r, c)| Some(r.as_ref()? - c.as_ref()?))
                    .collect()
            })
            .collect();
        let incurred = combine(&a.adjusted, &open, &s.paid);
        tri.with_cells(closed, &s.closed)?
            .with_cells(&self.settlement_rate.paid, &s.paid)?
            .with_cells(&self.case_adequacy.incurred, &incurred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_runs_through_both_points_and_extrapolates_on_the_nearest() {
        let pairs = [
            ((10.0, 100.0), (20.0, 400.0)),
            ((20.0, 400.0), (25.0, 500.0)),
        ];
        for how in [Interpolation::Exponential, Interpolation::Linear] {
            assert!((interpolate(&pairs, 10.0, how).unwrap() - 100.0).abs() < 1e-9);
            assert!((interpolate(&pairs, 25.0, how).unwrap() - 500.0).abs() < 1e-9);
        }
        // Exponential through (10, 100) and (20, 400): 100 * 2^((x - 10) / 5).
        let y = interpolate(&pairs, 5.0, Interpolation::Exponential).unwrap();
        assert!((y - 50.0).abs() < 1e-9);
        let y = interpolate(&pairs, 22.0, Interpolation::Linear).unwrap();
        assert!((y - 440.0).abs() < 1e-9);
        assert!(interpolate(&[], 1.0, Interpolation::Linear).is_none());
    }
}
