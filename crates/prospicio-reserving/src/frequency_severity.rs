//! Methods on claim counts (Friedland, *Estimating Unpaid Claims Using
//! Basic Techniques*, chapters 11 and 12; `reserving-v02.md` decision 35):
//! frequency-severity ([`FrequencySeverity`]), the disposal rate method
//! ([`Disposal`]) and case outstanding development
//! ([`CaseOutstanding`], [`case_development_factor`]), with the pieces they
//! share with the Berquist-Sherman adjustments
//! ([`crate::berquist_sherman`]): ultimate claim counts
//! ([`UltimateCounts`]), a selection of one value per age from a triangle of
//! ratios ([`Selection`]), and exponential trends ([`TrendFit`],
//! [`trend_by_age`], [`pooled_trend`]).
//!
//! Each method reads several columns of one single-segment [`Triangle`]:
//! counts and amounts sit side by side as its measure columns.

use crate::chain_ladder::{ChainLadder, ChainLadderFit};
use crate::error::{Error, Result};
use crate::triangle::Triangle;
use prospicio_core::{Lag, Period};

/// Rows of a triangle column: one row per origin, one value per age.
pub(crate) type Rows = Vec<Vec<Option<f64>>>;

fn counts_err<T>(why: impl Into<String>) -> Result<T> {
    Err(Error::Counts(why.into()))
}

/// The origin's start in years, as a fraction (a half-year origin starting
/// in July is half a year later than one starting in January).
pub(crate) fn origin_years(origin: Period) -> f64 {
    let m = origin.start();
    f64::from(m.year()) + (f64::from(m.month()) - 1.0) / 12.0
}

/// The development position of each origin's latest cell on the triangle's
/// grid at its valuation: the oldest age whose valuation is not after the
/// triangle's (`None` for an origin that starts after it).
pub(crate) fn valuation_positions(tri: &Triangle) -> Vec<Option<usize>> {
    let nd = tri.development().len();
    let val = tri.valuation();
    (0..tri.origins().len())
        .map(|o| (0..nd).rev().find(|&d| tri.valuation_of(o, d) <= val))
        .collect()
}

/// The position of the last observed cell in `row`.
pub(crate) fn last_observed(row: &[Option<f64>]) -> Option<usize> {
    row.iter().rposition(Option::is_some)
}

// ---------------------------------------------------------------------------
// Selections
// ---------------------------------------------------------------------------

/// How one value per development age is chosen from a triangle of ratios:
/// disposal rates, incremental severities brought to a common cost level,
/// or the ratios of case outstanding development.
#[derive(Debug, Clone, PartialEq)]
pub enum Selection {
    /// The simple average of the values at each age over the latest
    /// `n_periods` origins that have one (`None` uses every origin), without
    /// the `drop_high` highest and the `drop_low` lowest of them, as
    /// Friedland's "simple average, latest n" and "medial average, latest
    /// 5x1" (`n_periods = 5`, one dropped each side). Drops that would leave
    /// nothing are not made.
    Average {
        n_periods: Option<usize>,
        drop_high: usize,
        drop_low: usize,
    },
    /// The value on the latest diagonal at each age, as Berquist and
    /// Sherman select their disposal rates.
    LatestDiagonal,
    /// The values given by age in months, `(age, value)`; every age the
    /// method needs must be given.
    Given(Vec<(Lag, f64)>),
}

impl Default for Selection {
    /// The simple average of every origin.
    fn default() -> Self {
        Self::Average {
            n_periods: None,
            drop_high: 0,
            drop_low: 0,
        }
    }
}

impl Selection {
    /// The simple average of the latest `n` origins at each age.
    pub fn latest(n: usize) -> Self {
        Self::Average {
            n_periods: Some(n),
            drop_high: 0,
            drop_low: 0,
        }
    }

    /// One value per age of `ages` from `rows` (a triangle of ratios over
    /// the same ages); `None` at an age with nothing to select from.
    /// `latest` gives the position of each origin's latest-diagonal cell.
    pub(crate) fn select(
        &self,
        rows: &[Vec<Option<f64>>],
        ages: &[Lag],
        latest: &[Option<usize>],
        what: &str,
    ) -> Result<Vec<Option<f64>>> {
        match self {
            Self::Average {
                n_periods,
                drop_high,
                drop_low,
            } => {
                if *n_periods == Some(0) {
                    return Err(Error::InvalidSetting {
                        name: "n_periods",
                        value: 0.0,
                        expected: "at least 1",
                    });
                }
                Ok((0..ages.len())
                    .map(|d| {
                        let mut column: Vec<f64> = rows
                            .iter()
                            .filter_map(|r| r[d])
                            .filter(|v| v.is_finite())
                            .collect();
                        if let Some(n) = n_periods {
                            let skip = column.len().saturating_sub(*n);
                            column.drain(..skip);
                        }
                        if column.len() > drop_high + drop_low {
                            column.sort_by(f64::total_cmp);
                            column.drain(column.len() - drop_high..);
                            column.drain(..*drop_low);
                        }
                        (!column.is_empty())
                            .then(|| column.iter().sum::<f64>() / column.len() as f64)
                    })
                    .collect())
            }
            // The most recent origin at each age: older origins also stop at
            // the oldest age when the triangle's ages are cut off.
            Self::LatestDiagonal => Ok((0..ages.len())
                .map(|d| {
                    rows.iter()
                        .zip(latest)
                        .rev()
                        .find_map(|(r, &l)| (l == Some(d)).then_some(r[d]).flatten())
                })
                .collect()),
            Self::Given(values) => {
                for (age, v) in values {
                    if !ages.contains(age) {
                        return counts_err(format!(
                            "{what}: the age {age} of a given value is not one of the triangle's ages"
                        ));
                    }
                    if !v.is_finite() {
                        return counts_err(format!("{what}: the value at age {age} is not finite"));
                    }
                }
                Ok(ages
                    .iter()
                    .map(|a| values.iter().find(|(age, _)| age == a).map(|&(_, v)| v))
                    .collect())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Trends
// ---------------------------------------------------------------------------

/// An exponential trend fitted by least squares on the logarithms of the
/// values, as Friedland's "exponential regression analysis" (Excel's
/// `LOGEST`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrendFit {
    /// The annual rate of change, `exp(slope) - 1`.
    pub rate: f64,
    /// The coefficient of determination of the regression on the
    /// logarithms; 1 with two points.
    pub r_squared: f64,
    /// The number of points fitted.
    pub n: usize,
}

/// The exponential trend of `(year, value)` points: `ln(value)` regressed on
/// the year. Values that are not finite and positive are left out; `None`
/// when fewer than two points or a single year remain.
pub fn exponential_trend(points: &[(f64, f64)]) -> Option<TrendFit> {
    let pts: Vec<(f64, f64)> = points
        .iter()
        .filter(|(x, y)| x.is_finite() && y.is_finite() && *y > 0.0)
        .map(|&(x, y)| (x, y.ln()))
        .collect();
    let n = pts.len();
    if n < 2 {
        return None;
    }
    let nf = n as f64;
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / nf;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / nf;
    let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
    if sxx <= 0.0 {
        return None;
    }
    let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let syy: f64 = pts.iter().map(|p| (p.1 - my).powi(2)).sum();
    let b = sxy / sxx;
    let r_squared = if syy > 0.0 {
        sxy * sxy / (sxx * syy)
    } else {
        1.0
    };
    Some(TrendFit {
        rate: b.exp() - 1.0,
        r_squared,
        n,
    })
}

/// The exponential trend of `column` at each development age, the values
/// regressed on the origin year (Friedland's "annual change based on
/// exponential regression analysis of severities and accident year"):
/// `(age, fit)` per age, `None` where fewer than two origins have a
/// positive value. Pass an average (a severity, an average case
/// outstanding) built by [`ratio_column`].
pub fn trend_by_age(tri: &Triangle, column: &str) -> Result<Vec<(Lag, Option<TrendFit>)>> {
    let rows = tri.cells(column)?;
    let years: Vec<f64> = tri.origins().into_iter().map(origin_years).collect();
    Ok(tri
        .development()
        .iter()
        .enumerate()
        .map(|(d, &age)| {
            let pts: Vec<(f64, f64)> = rows
                .iter()
                .zip(&years)
                .filter_map(|(r, &x)| Some((x, r[d]?)))
                .collect();
            (age, exponential_trend(&pts))
        })
        .collect())
}

/// One exponential trend for `column` over every age at once: `ln(value)`
/// regressed on the origin year with an intercept per age, so each age
/// keeps its own level and all share the rate. Ages with fewer than two
/// positive values add nothing. The R-squared is of the year once the
/// ages' levels are removed. `None` with nothing to fit.
pub fn pooled_trend(tri: &Triangle, column: &str) -> Result<Option<TrendFit>> {
    let rows = tri.cells(column)?;
    let years: Vec<f64> = tri.origins().into_iter().map(origin_years).collect();
    let (mut sxy, mut sxx, mut syy, mut n) = (0.0, 0.0, 0.0, 0usize);
    for d in 0..tri.development().len() {
        let pts: Vec<(f64, f64)> = rows
            .iter()
            .zip(&years)
            .filter_map(|(r, &x)| {
                let y = r[d]?;
                (y.is_finite() && y > 0.0).then(|| (x, y.ln()))
            })
            .collect();
        if pts.len() < 2 {
            continue;
        }
        let k = pts.len() as f64;
        let mx = pts.iter().map(|p| p.0).sum::<f64>() / k;
        let my = pts.iter().map(|p| p.1).sum::<f64>() / k;
        for (x, y) in &pts {
            sxx += (x - mx).powi(2);
            sxy += (x - mx) * (y - my);
            syy += (y - my).powi(2);
        }
        n += pts.len();
    }
    if sxx <= 0.0 {
        return Ok(None);
    }
    let b = sxy / sxx;
    Ok(Some(TrendFit {
        rate: b.exp() - 1.0,
        r_squared: if syy > 0.0 {
            sxy * sxy / (sxx * syy)
        } else {
            1.0
        },
        n,
    }))
}

/// A copy of `tri` with the column `name` equal to `numerator` over
/// `denominator`, cell by cell where both are observed and the denominator
/// is not zero: a severity (claims over claim counts), a disposal rate or an
/// average case outstanding. The copy is cumulative.
pub fn ratio_column(
    tri: &Triangle,
    numerator: &str,
    denominator: &str,
    name: &str,
) -> Result<Triangle> {
    let num = tri.cells(numerator)?;
    let den = tri.cells(denominator)?;
    let rows: Rows = num
        .iter()
        .zip(&den)
        .map(|(n, d)| {
            n.iter()
                .zip(d)
                .map(|(n, d)| match (n, d) {
                    (Some(n), Some(d)) if *d != 0.0 => Some(n / d),
                    _ => None,
                })
                .collect()
        })
        .collect();
    tri.with_cells(name, &rows)
}

// ---------------------------------------------------------------------------
// Ultimate claim counts
// ---------------------------------------------------------------------------

/// The ultimate number of claims of each origin.
#[derive(Debug, Clone, PartialEq)]
pub enum UltimateCounts {
    /// Given, one per origin of the triangle, oldest first.
    Given(Vec<f64>),
    /// Projected by the chain ladder on the count column `column` (reported
    /// or closed counts), with all of [`ChainLadder`]'s settings.
    Development { column: String, method: ChainLadder },
    /// The mean of several projections, origin by origin, as Friedland
    /// selects the average of the closed and reported count projections.
    Mean(Vec<UltimateCounts>),
}

impl UltimateCounts {
    /// The chain ladder on `column` with its default settings.
    pub fn development(column: impl Into<String>) -> Self {
        Self::Development {
            column: column.into(),
            method: ChainLadder::default(),
        }
    }

    /// One ultimate per origin of `tri`.
    pub fn resolve(&self, tri: &Triangle) -> Result<Vec<f64>> {
        let origins = tri.origins();
        match self {
            Self::Given(values) => {
                if values.len() != origins.len() {
                    return Err(Error::LengthMismatch {
                        column: "ultimate counts".into(),
                        expected: origins.len(),
                        found: values.len(),
                    });
                }
                if let Some(v) = values.iter().find(|v| !v.is_finite() || **v < 0.0) {
                    return counts_err(format!(
                        "an ultimate claim count of {v} is not finite and non-negative"
                    ));
                }
                Ok(values.clone())
            }
            Self::Development { column, method } => {
                let fit = method.fit(tri, column)?;
                origins
                    .iter()
                    .map(|o| {
                        fit.origins
                            .iter()
                            .position(|p| p == o)
                            .map(|k| fit.ultimate[k])
                            .ok_or_else(|| {
                                Error::Counts(format!(
                                    "the origin {o} has no count in the column {column}"
                                ))
                            })
                    })
                    .collect()
            }
            Self::Mean(parts) => {
                if parts.is_empty() {
                    return counts_err("a mean of ultimate counts needs at least one projection");
                }
                let all = parts
                    .iter()
                    .map(|p| p.resolve(tri))
                    .collect::<Result<Vec<_>>>()?;
                Ok((0..origins.len())
                    .map(|o| all.iter().map(|v| v[o]).sum::<f64>() / all.len() as f64)
                    .collect())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Trended selections
// ---------------------------------------------------------------------------

/// A value selected at the level of a base origin and carried to the other
/// origins by an annual trend: at origin `o`,
/// `selected * (1 + trend)^(years(o) - years(base)) / factors[o]`, where
/// `factors[o]` brings origin `o`'s value to the base level for anything
/// besides the trend (Friedland's tort reform factors; 1 at the base and
/// when `factors` is empty). Friedland's approach #2 selects a frequency and
/// a severity this way from the older, mature years.
#[derive(Debug, Clone, PartialEq)]
pub struct Trended {
    /// The value at the base origin's level.
    pub selected: f64,
    /// The origin whose level `selected` is at.
    pub base: Period,
    /// The annual trend.
    pub trend: f64,
    /// One factor per origin of the triangle, oldest first, or empty.
    pub factors: Vec<f64>,
}

impl Trended {
    /// `selected` at `base` with an annual `trend` and no other factor.
    pub fn new(selected: f64, base: Period, trend: f64) -> Self {
        Self {
            selected,
            base,
            trend,
            factors: Vec::new(),
        }
    }

    /// The value at each of `origins`.
    pub fn values(&self, origins: &[Period]) -> Result<Vec<f64>> {
        check_trend(self.trend)?;
        if !self.factors.is_empty() && self.factors.len() != origins.len() {
            return Err(Error::LengthMismatch {
                column: "factors".into(),
                expected: origins.len(),
                found: self.factors.len(),
            });
        }
        if let Some(f) = self.factors.iter().find(|f| !f.is_finite() || **f <= 0.0) {
            return counts_err(format!("a trend factor of {f} is not finite and positive"));
        }
        let base = origin_years(self.base);
        Ok(origins
            .iter()
            .enumerate()
            .map(|(o, &p)| {
                let f = self.factors.get(o).copied().unwrap_or(1.0);
                self.selected * (1.0 + self.trend).powf(origin_years(p) - base) / f
            })
            .collect())
    }
}

fn check_trend(trend: f64) -> Result<()> {
    if !trend.is_finite() || trend <= -1.0 {
        return Err(Error::InvalidSetting {
            name: "trend",
            value: trend,
            expected: "finite and above -1",
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Frequency-severity (chapter 11, approaches 1 and 2)
// ---------------------------------------------------------------------------

/// How the ultimate claim counts of [`FrequencySeverity`] are projected.
#[derive(Debug, Clone, PartialEq)]
pub enum CountProjection {
    /// From the count triangles (approach #1).
    Counts(UltimateCounts),
    /// Exposure times a trended frequency (approach #2): at origin `o` the
    /// count is `exposure[o] * exposure_factors[o] * frequency[o]`, the
    /// exposure the latest value of the column `exposure`, brought to the
    /// base level by `exposure_factors` (on-level factors for premium, wage
    /// inflation for payroll; empty for none), and the frequency per unit of
    /// that exposure from [`Trended`] (the claim count trend in its
    /// `trend`).
    Frequency {
        exposure: String,
        exposure_factors: Vec<f64>,
        frequency: Trended,
    },
}

/// How the ultimate severity of [`FrequencySeverity`] is projected.
#[derive(Debug, Clone, PartialEq)]
pub enum SeverityProjection {
    /// The chain ladder on the triangle of severities (the amounts over the
    /// counts, cell by cell), with all of [`ChainLadder`]'s settings
    /// (approach #1).
    Development(ChainLadder),
    /// A severity selected at a base origin and trended (approach #2).
    Trended(Trended),
}

/// Frequency-severity: ultimate claims as ultimate claim counts times an
/// ultimate severity, each projected on its own (Friedland, chapter 11,
/// approaches #1 and #2).
///
/// ```
/// use prospicio_reserving::{
///     ChainLadder, CountProjection, DevelopmentColumn, FrequencySeverity, Grain, Long, Month,
///     SeverityProjection, Triangle, UltimateCounts,
/// };
///
/// let origin = [2020, 2020, 2021].map(Month::january);
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 12]),
///     values: &[("reported", &[100.0, 150.0, 120.0]), ("counts", &[10.0, 12.0, 11.0])],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })
/// .unwrap();
/// let fs = FrequencySeverity {
///     amounts: "reported".into(),
///     counts: "counts".into(),
///     ultimate_counts: CountProjection::Counts(UltimateCounts::development("counts")),
///     severity: SeverityProjection::Development(ChainLadder::default()),
/// };
/// let fit = fs.fit(&tri).unwrap();
/// // 11 counts at 1.2 to ultimate, a severity of 120 / 11 at 1.25.
/// assert!((fit.ultimate[1] - 13.2 * 120.0 / 11.0 * 1.25).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FrequencySeverity {
    /// The claim amounts (reported or paid) whose ultimate is projected.
    pub amounts: String,
    /// The claim counts the severity triangle divides the amounts by
    /// (reported counts for reported amounts, closed for paid).
    pub counts: String,
    pub ultimate_counts: CountProjection,
    pub severity: SeverityProjection,
}

/// A fitted [`FrequencySeverity`], per origin.
#[derive(Debug, Clone, PartialEq)]
pub struct FrequencySeverityFit {
    /// Origin periods, oldest first.
    pub origins: Vec<Period>,
    /// The latest amount of each origin.
    pub latest: Vec<f64>,
    /// The latest severity of each origin (NaN when it has none).
    pub latest_severity: Vec<f64>,
    pub ultimate_counts: Vec<f64>,
    pub ultimate_severity: Vec<f64>,
    /// Ultimate counts times ultimate severity.
    pub ultimate: Vec<f64>,
    /// The chain ladder on the severities, for a developed severity.
    pub severity_fit: Option<ChainLadderFit>,
}

impl FrequencySeverityFit {
    /// Ultimate less latest amount per origin: IBNR on reported amounts,
    /// the unpaid estimate on paid ones.
    pub fn reserves(&self) -> Vec<f64> {
        self.ultimate
            .iter()
            .zip(&self.latest)
            .map(|(u, l)| u - l)
            .collect()
    }

    pub fn total_ultimate(&self) -> f64 {
        self.ultimate.iter().sum()
    }

    pub fn total_reserve(&self) -> f64 {
        self.reserves().iter().sum()
    }
}

/// The latest observed value of each row; NaN for an empty row.
fn latest_values(rows: &[Vec<Option<f64>>]) -> Vec<f64> {
    rows.iter()
        .map(|r| last_observed(r).and_then(|d| r[d]).unwrap_or(f64::NAN))
        .collect()
}

impl FrequencySeverity {
    /// Fits a single-segment triangle.
    pub fn fit(&self, tri: &Triangle) -> Result<FrequencySeverityFit> {
        let origins = tri.origins();
        let latest = latest_values(&tri.cells(&self.amounts)?);
        let ultimate_counts = match &self.ultimate_counts {
            CountProjection::Counts(u) => u.resolve(tri)?,
            CountProjection::Frequency {
                exposure,
                exposure_factors,
                frequency,
            } => {
                let exposure_values = latest_values(&tri.cells(exposure)?);
                if !exposure_factors.is_empty() && exposure_factors.len() != origins.len() {
                    return Err(Error::LengthMismatch {
                        column: "exposure_factors".into(),
                        expected: origins.len(),
                        found: exposure_factors.len(),
                    });
                }
                let freq = frequency.values(&origins)?;
                exposure_values
                    .iter()
                    .zip(&freq)
                    .enumerate()
                    .map(|(o, (e, f))| {
                        if !e.is_finite() {
                            return Err(Error::InvalidExposure {
                                column: exposure.clone(),
                                origin: origins[o].to_string(),
                            });
                        }
                        Ok(e * exposure_factors.get(o).copied().unwrap_or(1.0) * f)
                    })
                    .collect::<Result<_>>()?
            }
        };
        let severity_tri = ratio_column(tri, &self.amounts, &self.counts, "__severity")?;
        let latest_severity = latest_values(&severity_tri.cells("__severity")?);
        let (ultimate_severity, severity_fit) = match &self.severity {
            SeverityProjection::Development(method) => {
                let fit = method.fit(&severity_tri, "__severity")?;
                let sev = origins
                    .iter()
                    .map(|o| {
                        fit.origins
                            .iter()
                            .position(|p| p == o)
                            .map_or(f64::NAN, |k| fit.ultimate[k])
                    })
                    .collect();
                (sev, Some(fit))
            }
            SeverityProjection::Trended(t) => (t.values(&origins)?, None),
        };
        let ultimate: Vec<f64> = ultimate_counts
            .iter()
            .zip(&ultimate_severity)
            .map(|(c, s)| c * s)
            .collect();
        if let Some(o) = (0..origins.len()).find(|&o| !(ultimate[o] + latest[o]).is_finite()) {
            return counts_err(format!(
                "the origin {} has no ultimate or no latest value in {}",
                origins[o], self.amounts
            ));
        }
        Ok(FrequencySeverityFit {
            origins,
            latest,
            latest_severity,
            ultimate_counts,
            ultimate_severity,
            ultimate,
            severity_fit,
        })
    }
}

// ---------------------------------------------------------------------------
// Disposal rates (chapter 11, approach #3)
// ---------------------------------------------------------------------------

/// The disposal rate method (Friedland, chapter 11, approach #3): the
/// closed claim counts still to come, by the selected disposal rates (closed
/// counts over ultimate counts), times the incremental paid severity of
/// each age, selected at a base origin's cost level and trended to each
/// origin's.
///
/// Future incremental closed counts at age `k` of an origin whose latest
/// closed count `C` is at age `d` are `(U - C) / (1 - r_d) * (r_k -
/// r_{k-1})`, with `U` its ultimate count and `r` the selected rates; a last
/// step "to ultimate" closes the rest, `1 - r` at the oldest age. Future
/// paid claims are those counts times the selected severities at the base
/// level carried to the origin as [`Trended`] carries them (with `trend`
/// and `factors`). Payments on claims still open (partial payments) are
/// assumed small, as the method requires.
#[derive(Debug, Clone, PartialEq)]
pub struct Disposal {
    pub closed_counts: String,
    pub paid: String,
    pub ultimate_counts: UltimateCounts,
    /// Selected from the triangle of disposal rates.
    pub disposal_rates: Selection,
    /// Selected from the incremental paid severities brought to the base
    /// origin's cost level.
    pub severities: Selection,
    /// From this age on (and in the step to ultimate), the severity is the
    /// pooled one of those ages: their trended incremental payments over
    /// their incremental closed counts (Friedland's "trended tail
    /// severity"). `None` uses `severities` at every age, and the oldest
    /// age's severity in the step to ultimate.
    pub tail_severity_from: Option<Lag>,
    /// The annual severity trend.
    pub trend: f64,
    /// The origin whose cost level the severities are selected at; `None`
    /// is the latest origin.
    pub base: Option<Period>,
    /// One factor per origin, oldest first, that brings an origin's
    /// severity to the base level besides the trend (tort reform); empty
    /// for none.
    pub factors: Vec<f64>,
}

/// A fitted [`Disposal`]. Every square has one row per origin and one
/// column per age plus a last one for the step to ultimate where noted;
/// cells before an origin's latest closed count are its data.
#[derive(Debug, Clone, PartialEq)]
pub struct DisposalFit {
    pub origins: Vec<Period>,
    /// Development ages; the squares that have a step to ultimate have one
    /// more column after the last.
    pub ages: Vec<Lag>,
    pub ultimate_counts: Vec<f64>,
    /// Cumulative closed counts over ultimate counts (no step to ultimate).
    pub disposal_rates: Rows,
    /// The selected disposal rate at each age.
    pub selected_rates: Vec<f64>,
    /// Incremental paid over incremental closed counts, at each origin's
    /// own cost level (no step to ultimate).
    pub severities: Rows,
    /// The same, brought to the base origin's cost level.
    pub trended_severities: Rows,
    /// The selected severity at each age and in the step to ultimate, at
    /// the base level.
    pub selected_severities: Vec<f64>,
    /// Incremental closed counts, observed then projected, with the step to
    /// ultimate.
    pub incremental_counts: Rows,
    /// Incremental paid severities, observed then projected (the selected
    /// ones at each origin's cost level), with the step to ultimate.
    pub projected_severities: Rows,
    /// Incremental paid claims, observed then projected, with the step to
    /// ultimate.
    pub incremental_paid: Rows,
    /// The latest paid claims of each origin.
    pub latest: Vec<f64>,
    /// Latest paid plus the projected payments.
    pub ultimate: Vec<f64>,
}

impl DisposalFit {
    /// Ultimate less paid per origin: the unpaid claim estimate.
    pub fn reserves(&self) -> Vec<f64> {
        self.ultimate
            .iter()
            .zip(&self.latest)
            .map(|(u, l)| u - l)
            .collect()
    }

    pub fn total_ultimate(&self) -> f64 {
        self.ultimate.iter().sum()
    }

    pub fn total_reserve(&self) -> f64 {
        self.reserves().iter().sum()
    }
}

/// Incremental values of cumulative `rows`: the first age's value as is,
/// and each later one less the one before, where both are observed.
pub(crate) fn incremental(rows: &[Vec<Option<f64>>]) -> Rows {
    rows.iter()
        .map(|r| {
            (0..r.len())
                .map(|d| match d {
                    0 => r[0],
                    _ => Some(r[d]? - r[d - 1]?),
                })
                .collect()
        })
        .collect()
}

impl Disposal {
    /// Fits a single-segment triangle.
    pub fn fit(&self, tri: &Triangle) -> Result<DisposalFit> {
        check_trend(self.trend)?;
        let origins = tri.origins();
        let ages = tri.development().to_vec();
        let (no, nd) = (origins.len(), ages.len());
        if !self.factors.is_empty() && self.factors.len() != no {
            return Err(Error::LengthMismatch {
                column: "factors".into(),
                expected: no,
                found: self.factors.len(),
            });
        }
        if let Some(f) = self.factors.iter().find(|f| !f.is_finite() || **f <= 0.0) {
            return counts_err(format!(
                "a severity factor of {f} is not finite and positive"
            ));
        }
        let closed = tri.cells(&self.closed_counts)?;
        let paid = tri.cells(&self.paid)?;
        let ultimate_counts = self.ultimate_counts.resolve(tri)?;
        let latest_pos: Vec<Option<usize>> = closed.iter().map(|r| last_observed(r)).collect();

        // Disposal rates and their selection.
        let disposal_rates: Rows = closed
            .iter()
            .zip(&ultimate_counts)
            .map(|(r, &u)| {
                r.iter()
                    .map(|c| c.filter(|_| u > 0.0).map(|c| c / u))
                    .collect()
            })
            .collect();
        let selected_rates = self
            .disposal_rates
            .select(&disposal_rates, &ages, &latest_pos, "disposal rates")?
            .into_iter()
            .enumerate()
            .map(|(d, r)| {
                r.ok_or_else(|| {
                    Error::Counts(format!("no disposal rate to select at age {}", ages[d]))
                })
            })
            .collect::<Result<Vec<f64>>>()?;

        // Incremental counts, payments and severities.
        let inc_closed = incremental(&closed);
        let inc_paid = incremental(&paid);
        let base = self.base.unwrap_or(origins[no - 1]);
        let base_years = origin_years(base);
        let factor = |o: usize| self.factors.get(o).copied().unwrap_or(1.0);
        // Multiplies an origin's value onto the base cost level.
        let to_base: Vec<f64> = (0..no)
            .map(|o| (1.0 + self.trend).powf(base_years - origin_years(origins[o])) * factor(o))
            .collect();
        let severities: Rows = inc_paid
            .iter()
            .zip(&inc_closed)
            .map(|(p, c)| {
                p.iter()
                    .zip(c)
                    .map(|(p, c)| match (p, c) {
                        (Some(p), Some(c)) if *c != 0.0 => Some(p / c),
                        _ => None,
                    })
                    .collect()
            })
            .collect();
        let trended_severities: Rows = severities
            .iter()
            .enumerate()
            .map(|(o, r)| r.iter().map(|s| s.map(|s| s * to_base[o])).collect())
            .collect();
        let selected =
            self.severities
                .select(&trended_severities, &ages, &latest_pos, "severities")?;
        let tail_from = match self.tail_severity_from {
            Some(age) => Some(ages.iter().position(|&a| a == age).ok_or_else(|| {
                Error::Counts(format!(
                    "the tail severity age {age} is not one of the triangle's ages"
                ))
            })?),
            None => None,
        };
        let pooled = match tail_from {
            Some(k) => {
                let (mut amount, mut count) = (0.0, 0.0);
                for o in 0..no {
                    for d in k..nd {
                        if let (Some(p), Some(c)) = (inc_paid[o][d], inc_closed[o][d]) {
                            amount += p * to_base[o];
                            count += c;
                        }
                    }
                }
                if count == 0.0 {
                    return counts_err(format!(
                        "no closed claim from age {} on to pool a tail severity",
                        ages[k]
                    ));
                }
                Some(amount / count)
            }
            None => None,
        };
        let mut selected_severities = Vec::with_capacity(nd + 1);
        for (d, &own) in selected.iter().enumerate() {
            selected_severities.push(match (tail_from, pooled) {
                (Some(k), Some(p)) if d >= k => Some(p),
                _ => own,
            });
        }
        selected_severities.push(pooled.or(selected_severities[nd - 1]));
        // Ages without a selection are needed only when an origin has a
        // future at them.
        let need = |d: usize| -> Result<f64> {
            selected_severities[d].ok_or_else(|| {
                Error::Counts(match ages.get(d) {
                    Some(a) => format!("no severity to select at age {a}"),
                    None => "no severity for the step to ultimate".into(),
                })
            })
        };

        // Complete the squares.
        let mut incremental_counts = vec![vec![None; nd + 1]; no];
        let mut projected_severities = vec![vec![None; nd + 1]; no];
        let mut incremental_paid = vec![vec![None; nd + 1]; no];
        let mut latest = vec![f64::NAN; no];
        let mut ultimate = vec![f64::NAN; no];
        for o in 0..no {
            let Some(d0) = latest_pos[o] else {
                return counts_err(format!(
                    "the origin {} has no closed count in {}",
                    origins[o], self.closed_counts
                ));
            };
            let Some(paid_latest) = paid[o][d0] else {
                return counts_err(format!(
                    "the origin {} has no paid claims at its latest closed count",
                    origins[o]
                ));
            };
            latest[o] = paid_latest;
            for d in 0..=d0 {
                incremental_counts[o][d] = inc_closed[o][d];
                projected_severities[o][d] = severities[o][d];
                incremental_paid[o][d] = inc_paid[o][d];
            }
            let open = ultimate_counts[o] - closed[o][d0].unwrap_or(0.0);
            let remaining = 1.0 - selected_rates[d0];
            if open > 0.0 && remaining <= 0.0 {
                return counts_err(format!(
                    "the origin {} has {open} claims to close but the disposal rate selected at \
                     age {} is {}, which leaves none",
                    origins[o], ages[d0], selected_rates[d0]
                ));
            }
            let mut total = paid_latest;
            for d in d0 + 1..=nd {
                let step = match d {
                    d if d == nd => 1.0 - selected_rates[nd - 1],
                    d => selected_rates[d] - selected_rates[d - 1],
                };
                let count = if remaining > 0.0 {
                    open / remaining * step
                } else {
                    0.0
                };
                let sev = if count == 0.0 {
                    selected_severities[d].unwrap_or(0.0)
                } else {
                    need(d)?
                } / to_base[o];
                incremental_counts[o][d] = Some(count);
                projected_severities[o][d] = Some(sev);
                incremental_paid[o][d] = Some(count * sev);
                total += count * sev;
            }
            ultimate[o] = total;
        }
        Ok(DisposalFit {
            origins,
            ages,
            ultimate_counts,
            disposal_rates,
            selected_rates,
            severities,
            trended_severities,
            selected_severities: selected_severities
                .into_iter()
                .map(|s| s.unwrap_or(f64::NAN))
                .collect(),
            incremental_counts,
            projected_severities,
            incremental_paid,
            latest,
            ultimate,
        })
    }
}

// ---------------------------------------------------------------------------
// Case outstanding development (chapter 12)
// ---------------------------------------------------------------------------

/// The case outstanding development technique (Friedland, chapter 12,
/// approach #1, after Wiser): the case outstanding at each age is the case
/// outstanding at the age before times a selected ratio, and the payments
/// in each period are the case outstanding at its start times another; a
/// last step to ultimate pays `paid_to_ultimate` times the case outstanding
/// left at the oldest age, and no case outstanding remains after it.
/// Ultimate claims are the paid claims to date plus the projected
/// payments. It needs no claim counts; case outstanding is incurred less
/// paid.
#[derive(Debug, Clone, PartialEq)]
pub struct CaseOutstanding {
    pub paid: String,
    pub incurred: String,
    /// Selected from the ratios of the incremental paid claims at each age
    /// to the case outstanding at the age before.
    pub paid_on_case: Selection,
    /// Selected from the ratios of the case outstanding at each age to the
    /// case outstanding at the age before.
    pub case_on_case: Selection,
    /// The payments after the oldest age as a ratio to the case outstanding
    /// at it (Friedland's 1.10).
    pub paid_to_ultimate: f64,
}

/// A fitted [`CaseOutstanding`]. The squares have one row per origin and
/// one column per age; the paid square has one more for the step to
/// ultimate. The ratio triangles are at the age of their numerator (empty
/// at the first age); the selected ratios too.
#[derive(Debug, Clone, PartialEq)]
pub struct CaseOutstandingFit {
    pub origins: Vec<Period>,
    pub ages: Vec<Lag>,
    pub paid_on_case: Rows,
    pub case_on_case: Rows,
    /// NaN at the first age and wherever nothing was selected.
    pub selected_paid_on_case: Vec<f64>,
    pub selected_case_on_case: Vec<f64>,
    /// Case outstanding, observed then projected.
    pub case_outstanding: Rows,
    /// Incremental paid claims, observed then projected, with the step to
    /// ultimate.
    pub incremental_paid: Rows,
    pub latest_paid: Vec<f64>,
    pub latest_incurred: Vec<f64>,
    pub ultimate: Vec<f64>,
}

impl CaseOutstandingFit {
    /// Ultimate less incurred per origin.
    pub fn ibnr(&self) -> Vec<f64> {
        self.ultimate
            .iter()
            .zip(&self.latest_incurred)
            .map(|(u, l)| u - l)
            .collect()
    }

    /// Ultimate less paid per origin: the unpaid claim estimate.
    pub fn reserves(&self) -> Vec<f64> {
        self.ultimate
            .iter()
            .zip(&self.latest_paid)
            .map(|(u, l)| u - l)
            .collect()
    }

    pub fn total_ultimate(&self) -> f64 {
        self.ultimate.iter().sum()
    }

    pub fn total_reserve(&self) -> f64 {
        self.reserves().iter().sum()
    }
}

impl CaseOutstanding {
    /// Fits a single-segment triangle.
    pub fn fit(&self, tri: &Triangle) -> Result<CaseOutstandingFit> {
        if !self.paid_to_ultimate.is_finite() || self.paid_to_ultimate < 0.0 {
            return Err(Error::InvalidSetting {
                name: "paid_to_ultimate",
                value: self.paid_to_ultimate,
                expected: "finite and non-negative",
            });
        }
        let origins = tri.origins();
        let ages = tri.development().to_vec();
        let (no, nd) = (origins.len(), ages.len());
        let paid = tri.cells(&self.paid)?;
        let incurred = tri.cells(&self.incurred)?;
        let case: Rows = incurred
            .iter()
            .zip(&paid)
            .map(|(i, p)| {
                i.iter()
                    .zip(p)
                    .map(|(i, p)| Some(i.as_ref()? - p.as_ref()?))
                    .collect()
            })
            .collect();
        let inc_paid = incremental(&paid);
        let ratio = |num: &Rows| -> Rows {
            (0..no)
                .map(|o| {
                    (0..nd)
                        .map(|d| match d {
                            0 => None,
                            _ => match (num[o][d], case[o][d - 1]) {
                                (Some(n), Some(c)) if c != 0.0 => Some(n / c),
                                _ => None,
                            },
                        })
                        .collect()
                })
                .collect()
        };
        let paid_on_case = ratio(&inc_paid);
        let case_on_case = ratio(&case);
        let latest_pos: Vec<Option<usize>> = case.iter().map(|r| last_observed(r)).collect();
        // Ratio rows sit at their numerator's age, so their latest diagonal
        // is the case's.
        let sel_paid =
            self.paid_on_case
                .select(&paid_on_case, &ages, &latest_pos, "paid on case ratios")?;
        let sel_case =
            self.case_on_case
                .select(&case_on_case, &ages, &latest_pos, "case on case ratios")?;
        let need = |sel: &[Option<f64>], d: usize, what: &str| -> Result<f64> {
            sel[d].ok_or_else(|| {
                Error::Counts(format!("no {what} ratio to select at age {}", ages[d]))
            })
        };
        let mut case_square = vec![vec![None; nd]; no];
        let mut paid_square = vec![vec![None; nd + 1]; no];
        let mut latest_paid = vec![f64::NAN; no];
        let mut latest_incurred = vec![f64::NAN; no];
        let mut ultimate = vec![f64::NAN; no];
        for o in 0..no {
            let (Some(p0), Some(c0)) =
                latest_pos[o].map_or((None, None), |d| (paid[o][d], case[o][d]))
            else {
                return counts_err(format!(
                    "the origin {} has no cell with both {} and {}",
                    origins[o], self.paid, self.incurred
                ));
            };
            let d0 = latest_pos[o].unwrap_or(0);
            latest_paid[o] = p0;
            latest_incurred[o] = p0 + c0;
            for d in 0..=d0 {
                case_square[o][d] = case[o][d];
                paid_square[o][d] = inc_paid[o][d];
            }
            let (mut c, mut total) = (c0, p0);
            for d in d0 + 1..nd {
                let p = c * need(&sel_paid, d, "paid on case")?;
                c *= need(&sel_case, d, "case on case")?;
                case_square[o][d] = Some(c);
                paid_square[o][d] = Some(p);
                total += p;
            }
            let p = c * self.paid_to_ultimate;
            paid_square[o][nd] = Some(p);
            ultimate[o] = total + p;
        }
        let nan = |s: Vec<Option<f64>>| s.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect();
        Ok(CaseOutstandingFit {
            origins,
            ages,
            paid_on_case,
            case_on_case,
            selected_paid_on_case: nan(sel_paid),
            selected_case_on_case: nan(sel_case),
            case_outstanding: case_square,
            incremental_paid: paid_square,
            latest_paid,
            latest_incurred,
            ultimate,
        })
    }
}

/// The factor that develops case outstanding to the unpaid claims (case
/// outstanding plus IBNR) from reported and paid factors to ultimate
/// (Friedland, chapter 12, approach #2): `1 + (R - 1) P / (P - R)`, with
/// `R` the reported and `P` the paid factor to ultimate. It needs `P > R`.
pub fn case_development_factor(reported_cdf: f64, paid_cdf: f64) -> Result<f64> {
    if !(reported_cdf.is_finite() && paid_cdf.is_finite()) || paid_cdf <= reported_cdf {
        return counts_err(format!(
            "a case development factor needs a paid factor to ultimate ({paid_cdf}) above the reported one ({reported_cdf})"
        ));
    }
    Ok(1.0 + (reported_cdf - 1.0) * paid_cdf / (paid_cdf - reported_cdf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangle::{DevelopmentColumn, Long};
    use prospicio_core::{Grain, Month};

    /// A triangle from rows of cumulative values of several columns; every
    /// row starts at 12 months.
    pub(crate) fn tri(first: i32, columns: &[(&str, &[&[f64]])]) -> Triangle {
        let mut origin = Vec::new();
        let mut age = Vec::new();
        let rows = columns[0].1;
        for (o, r) in rows.iter().enumerate() {
            for d in 0..r.len() {
                origin.push(Month::january(first + o as i32));
                age.push(12 * (d as u32 + 1));
            }
        }
        let values: Vec<(&str, Vec<f64>)> = columns
            .iter()
            .map(|(n, rows)| (*n, rows.iter().flat_map(|r| r.iter().copied()).collect()))
            .collect();
        let refs: Vec<(&str, &[f64])> = values.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&age),
            values: &refs,
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    #[test]
    fn selections_average_the_latest_values_without_the_extremes() {
        let rows = vec![
            vec![Some(1.0), Some(5.0)],
            vec![Some(2.0), Some(6.0)],
            vec![Some(3.0), None],
            vec![Some(10.0), None],
        ];
        let ages = [12, 24];
        let latest = [None, Some(1), None, Some(0)];
        let s = Selection::latest(2)
            .select(&rows, &ages, &latest, "x")
            .unwrap();
        assert_eq!(s, vec![Some(6.5), Some(5.5)]);
        let medial = Selection::Average {
            n_periods: Some(3),
            drop_high: 1,
            drop_low: 1,
        };
        assert_eq!(
            medial.select(&rows, &ages, &latest, "x").unwrap()[0],
            Some(3.0)
        );
        // Drops that would leave nothing are not made.
        assert_eq!(
            medial.select(&rows, &ages, &latest, "x").unwrap()[1],
            Some(5.5)
        );
        let diag = Selection::LatestDiagonal
            .select(&rows, &ages, &latest, "x")
            .unwrap();
        assert_eq!(diag, vec![Some(10.0), Some(6.0)]);
        let given = Selection::Given(vec![(24, 0.5)])
            .select(&rows, &ages, &latest, "x")
            .unwrap();
        assert_eq!(given, vec![None, Some(0.5)]);
        assert!(
            Selection::Given(vec![(36, 0.5)])
                .select(&rows, &ages, &latest, "x")
                .is_err()
        );
    }

    #[test]
    fn exponential_trend_is_exact_on_a_geometric_series() {
        let pts: Vec<(f64, f64)> = (0..5)
            .map(|k| (2000.0 + k as f64, 100.0 * 1.07f64.powi(k)))
            .collect();
        let fit = exponential_trend(&pts).unwrap();
        assert!((fit.rate - 0.07).abs() < 1e-12);
        assert!((fit.r_squared - 1.0).abs() < 1e-12);
        assert_eq!(fit.n, 5);
        assert!(exponential_trend(&pts[..1]).is_none());
    }

    #[test]
    fn frequency_severity_equals_the_chain_ladder_when_rows_are_multiples() {
        // Every row of counts and of amounts is a multiple of one base row,
        // so every link ratio is the same in every origin and the severity
        // chain ladder times the count chain ladder is the amount one.
        let base_c = [10.0, 14.0, 15.0, 15.5];
        let base_a = [100.0, 180.0, 210.0, 220.0];
        let scale = [1.0, 1.3, 0.8, 1.7];
        let rows = |base: &[f64; 4]| -> Vec<Vec<f64>> {
            scale
                .iter()
                .enumerate()
                .map(|(o, s)| {
                    base[..4 - o]
                        .iter()
                        .map(|v| v * s * (1.0 + o as f64 * 0.1))
                        .collect()
                })
                .collect()
        };
        let (c, a) = (rows(&base_c), rows(&base_a));
        let c: Vec<&[f64]> = c.iter().map(Vec::as_slice).collect();
        let a: Vec<&[f64]> = a.iter().map(Vec::as_slice).collect();
        let t = tri(2020, &[("counts", &c), ("amounts", &a)]);
        let fs = FrequencySeverity {
            amounts: "amounts".into(),
            counts: "counts".into(),
            ultimate_counts: CountProjection::Counts(UltimateCounts::development("counts")),
            severity: SeverityProjection::Development(ChainLadder::default()),
        };
        let fit = fs.fit(&t).unwrap();
        let cl = ChainLadder::default().fit(&t, "amounts").unwrap();
        for (u, v) in fit.ultimate.iter().zip(&cl.ultimate) {
            assert!((u - v).abs() < 1e-9 * v, "{u} {v}");
        }
    }

    #[test]
    fn case_development_factor_matches_its_formula() {
        let f = case_development_factor(1.5, 3.0).unwrap();
        assert!((f - (1.0 + 0.5 * 3.0 / 1.5)).abs() < 1e-15);
        assert!(case_development_factor(2.0, 2.0).is_err());
    }
}
