//! Ratemaking: premium at the current rate level, trend and the overall
//! rate indication (`docs/design/ratemaking.md`), after Werner and Modlin,
//! *Basic Ratemaking* (5th ed., CAS, 2016), chapters 5 to 8 and 12.
//!
//! Time is a year fraction (2011.5 is 1 July 2011), so nothing here needs a
//! calendar; [`year_fraction`] puts a date on the book's half-month grid.
//! Policies are written uniformly over time and earned uniformly over a
//! term `T` in years.
//!
//! - [`Period`]: an experience period (premium earned in a calendar
//!   window, or written in a policy window) and its average written and
//!   earned dates, which set trend lengths. A forecast period is the policy
//!   period of the policies written while the new rates are in effect.
//! - [`RateHistory`]: rate changes and the parallelogram method, the
//!   portion of a period's premium at each rate level and the factor to
//!   the current level.
//! - [`TrendFit`]: exponential and linear trends fitted by least squares,
//!   and [`trend_factor`].
//! - [`LossRatioYear`] and [`PurePremiumYear`]: an experience year
//!   projected to the forecast period, and [`LossRatioExperience`] and
//!   [`PurePremiumExperience`], the years combined as a ratio of totals.
//! - [`Provisions`]: the variable expense and profit provisions, the
//!   permissible loss ratio, and the loss ratio and pure premium methods;
//!   [`premium_based_expenses`] splits expense ratios into fixed and
//!   variable.
//! - [`credibility_weighted`] and [`trended_present_rates`]: the indication
//!   weighted against a complement.
//!
//! Nothing rounds: the book's exhibits round at each step, so a parity
//! test rounds between the steps it reproduces.

use prospicio_core::{Error, Result};

/// The year fraction of a date on the book's half-month grid: the first of
/// a month is its start, a day near the middle is half way, and the last
/// day is the next month's start (30 June ≡ 1 July, 31 December ≡ 1
/// January). Werner and Modlin round dates to the nearest half-month, so
/// 1/1/11 to 6/30/13 is 2.5 years and 11/15/11 to 6/30/13 is 1.625. Any
/// day is rounded the same way, so the 24th of a 31-day month is already
/// the next month's start.
///
/// It is a ratemaking date, not a calendar: do not assign accident years
/// with it, since a loss on 31 December 2016 falls at 2017.0, in 2017's
/// year. Take the accident year from the date itself.
///
/// ```
/// use prospicio_pricing::ratemaking::year_fraction;
///
/// assert_eq!(year_fraction(2013, 6, 30).unwrap(), 2013.5);
/// assert_eq!(year_fraction(2011, 11, 15).unwrap(), 2011.875);
/// assert_eq!(year_fraction(2015, 12, 31).unwrap(), 2016.0);
/// ```
pub fn year_fraction(year: i32, month: u8, day: u8) -> Result<f64> {
    if !(1..=12).contains(&month) {
        return Err(Error::InvalidMonth(month));
    }
    let days = days_in_month(year, month);
    if day == 0 || day > days {
        return Err(Error::InvalidParameter {
            name: "day",
            value: f64::from(day),
            reason: "must be a day of the month",
        });
    }
    let within = f64::from(day - 1) / f64::from(days - 1);
    let half_months = (2.0 * within).round() / 2.0;
    Ok(f64::from(year) + (f64::from(month - 1) + half_months) / 12.0)
}

fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        4 | 6 | 9 | 11 => 30,
        2 if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
        2 => 28,
        _ => 31,
    }
}

/// An experience or forecast period, in year fractions `[from, to)`.
///
/// ```
/// use prospicio_pricing::ratemaking::Period;
///
/// // Calendar year 2011 earned premium, annual policies: written on
/// // average on 1/1/11; rates effective 1/1/13 for a year are written on
/// // average on 7/1/13, so the one-step premium trend runs 2.5 years.
/// let cy = Period::calendar(2011.0, 2012.0).unwrap();
/// let future = Period::policy(2013.0, 2014.0).unwrap();
/// assert_eq!(future.average_written_date(1.0) - cy.average_written_date(1.0), 2.5);
/// // Accident year 2011 to the forecast's average accident date: 4.5 years
/// // for rates effective 1/1/15.
/// let forecast = Period::policy(2015.0, 2016.0).unwrap();
/// assert_eq!(forecast.average_earned_date(1.0) - cy.average_earned_date(1.0), 4.5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Period {
    /// Premium earned (or losses occurring) in `[from, to)`: a calendar or
    /// accident year, or any calendar window.
    Calendar { from: f64, to: f64 },
    /// Policies written in `[from, to)`, over their whole term: a policy
    /// year, or the policies written while new rates are in effect.
    Policy { from: f64, to: f64 },
}

impl Period {
    /// Premium earned (losses occurring) in `[from, to)`.
    pub fn calendar(from: f64, to: f64) -> Result<Self> {
        check_window(from, to)?;
        Ok(Self::Calendar { from, to })
    }

    /// Policies written in `[from, to)`.
    pub fn policy(from: f64, to: f64) -> Result<Self> {
        check_window(from, to)?;
        Ok(Self::Policy { from, to })
    }

    fn bounds(&self) -> (f64, f64) {
        match *self {
            Self::Calendar { from, to } | Self::Policy { from, to } => (from, to),
        }
    }

    /// The average date on which the period's premium was written, for
    /// policies of `term` years: the window's midpoint for a policy period,
    /// half a term earlier for a calendar period (1/1 of a calendar year for
    /// annual policies, 4/1 for six-month ones). Premium is trended from
    /// and to these dates.
    pub fn average_written_date(&self, term: f64) -> f64 {
        let (from, to) = self.bounds();
        let mid = 0.5 * (from + to);
        match self {
            Self::Calendar { .. } => mid - 0.5 * term,
            Self::Policy { .. } => mid,
        }
    }

    /// The average date on which the period's premium is earned, which is
    /// also the average accident date of its losses: the window's midpoint
    /// for a calendar period, half a term after it for a policy period
    /// (1/1 of the next year for an annual policy year). Losses are
    /// trended from and to these dates.
    pub fn average_earned_date(&self, term: f64) -> f64 {
        self.average_written_date(term) + 0.5 * term
    }
}

fn check_window(from: f64, to: f64) -> Result<()> {
    if !(from.is_finite() && to.is_finite() && to > from) {
        return Err(Error::InvalidParameter {
            name: "to",
            value: to,
            reason: "the period must be finite and end after it starts",
        });
    }
    Ok(())
}

/// How a rate change reaches policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Applies {
    /// To policies written on or after the effective date: a diagonal line
    /// in the parallelogram diagram. The usual case.
    #[default]
    Written,
    /// To all premium earned on or after the effective date, policies in
    /// force included (a law change): a vertical line.
    InForce,
}

/// One rate change: its effective date (a year fraction), the overall
/// average change (`0.05` for +5%) and how it applies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateChange {
    pub effective: f64,
    pub change: f64,
    pub applies: Applies,
}

impl RateChange {
    /// A change to policies written on or after `effective`.
    pub fn written(effective: f64, change: f64) -> Self {
        Self {
            effective,
            change,
            applies: Applies::Written,
        }
    }

    /// A change to all premium earned on or after `effective`.
    pub fn in_force(effective: f64, change: f64) -> Self {
        Self {
            effective,
            change,
            applies: Applies::InForce,
        }
    }
}

/// A history of rate changes, from an initial level of 1, and the
/// parallelogram method (Werner and Modlin, ch. 5): the portion of a
/// period's premium written or earned at each rate level, under uniform
/// writing and earning, and the factor that brings it to the current level.
///
/// Areas are exact: each cell of the plane of written date and earned date
/// where the level is constant is clipped to the period's region.
///
/// ```
/// use prospicio_pricing::ratemaking::{Period, RateChange, RateHistory};
///
/// // Werner and Modlin, Table 5.19: +5% on 7/1/10, +10% on 1/1/11, -1% on
/// // 4/1/12. Calendar year 2011, annual policies.
/// let history = RateHistory::new(&[
///     RateChange::written(2010.5, 0.05),
///     RateChange::written(2011.0, 0.10),
///     RateChange::written(2012.25, -0.01),
/// ]).unwrap();
/// let cy = Period::calendar(2011.0, 2012.0).unwrap();
/// let level = history.rate_level(&cy, 1.0).unwrap();
/// assert!((level.average_level - 1.09625).abs() < 1e-12);
/// assert!((level.on_level_factor() - 1.0431).abs() < 1e-4);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RateHistory {
    changes: Vec<RateChange>,
}

/// One cell of a period's premium at a single rate level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Portion {
    /// How many changes that apply to written policies were in effect when
    /// the premium was written (0 for the initial level).
    pub written_changes: usize,
    /// How many in-force changes were in effect when it was earned.
    pub in_force_changes: usize,
    /// The cumulative rate level of the cell.
    pub level: f64,
    /// Its share of the period's premium.
    pub share: f64,
}

/// A period's premium by rate level.
#[derive(Debug, Clone, PartialEq)]
pub struct RateLevel {
    /// The cells with a positive share, by written then in-force changes.
    pub portions: Vec<Portion>,
    /// `Σ share × level`.
    pub average_level: f64,
    /// The level after every change in the history.
    pub current_level: f64,
}

impl RateLevel {
    /// `current_level / average_level`, the on-level factor that brings
    /// the period's premium to the current rate level.
    pub fn on_level_factor(&self) -> f64 {
        self.current_level / self.average_level
    }
}

impl RateHistory {
    /// A history from its changes, in order of effective date; changes on
    /// the same date apply in the order given.
    pub fn new(changes: &[RateChange]) -> Result<Self> {
        for c in changes {
            if !c.effective.is_finite() {
                return Err(Error::InvalidParameter {
                    name: "effective",
                    value: c.effective,
                    reason: "must be finite",
                });
            }
            if !(c.change.is_finite() && c.change > -1.0) {
                return Err(Error::InvalidParameter {
                    name: "change",
                    value: c.change,
                    reason: "must be finite and greater than -1",
                });
            }
        }
        if changes.windows(2).any(|w| w[1].effective < w[0].effective) {
            return Err(Error::Data(
                "rate changes must be in order of effective date".into(),
            ));
        }
        Ok(Self {
            changes: changes.to_vec(),
        })
    }

    /// A history from the cumulative level after each change (the first
    /// level is relative to an initial level of 1), as an exhibit prints
    /// them. Rounded printed levels then carry into the factors.
    pub fn from_levels(effective: &[f64], levels: &[f64], applies: &[Applies]) -> Result<Self> {
        if effective.len() != levels.len() || applies.len() != levels.len() {
            return Err(Error::Data(
                "effective, levels and applies must have the same length".into(),
            ));
        }
        let mut previous = 1.0;
        let mut changes = Vec::with_capacity(levels.len());
        for ((&e, &l), &a) in effective.iter().zip(levels).zip(applies) {
            if !(l.is_finite() && l > 0.0) {
                return Err(Error::InvalidParameter {
                    name: "levels",
                    value: l,
                    reason: "must be finite and positive",
                });
            }
            changes.push(RateChange {
                effective: e,
                change: l / previous - 1.0,
                applies: a,
            });
            previous = l;
        }
        Self::new(&changes)
    }

    /// The changes, in order.
    pub fn changes(&self) -> &[RateChange] {
        &self.changes
    }

    /// The level after every change: `Π (1 + change)`.
    pub fn current_level(&self) -> f64 {
        self.changes.iter().map(|c| 1.0 + c.change).product()
    }

    /// The portion of `period`'s premium at each rate level, for policies
    /// of `term` years, and its average and current levels.
    pub fn rate_level(&self, period: &Period, term: f64) -> Result<RateLevel> {
        if !(term.is_finite() && term > 0.0) {
            return Err(Error::InvalidParameter {
                name: "term",
                value: term,
                reason: "must be finite and positive",
            });
        }
        // Dates are measured from the period's start, so that areas of a
        // few square years are not lost to cancellation near 2000.
        let (origin, end) = period.bounds();
        let (from, to) = (0.0, end - origin);
        // The region in the plane of written date w and earned date t:
        // w <= t <= w + term, and the period's window.
        let mut region = vec![(1.0, -1.0, 0.0), (-1.0, 1.0, term)];
        let (w_lo, w_hi, t_lo, t_hi) = match period {
            Period::Calendar { .. } => {
                region.extend([(0.0, -1.0, -from), (0.0, 1.0, to)]);
                (from - term, to, from, to)
            }
            Period::Policy { .. } => {
                region.extend([(-1.0, 0.0, -from), (1.0, 0.0, to)]);
                (from, to, from, to + term)
            }
        };
        let total = (to - from) * term;

        let cuts = |kind: Applies| -> Vec<(f64, f64)> {
            // (start of the interval, cumulative factor from that date on)
            let mut out = vec![(f64::NEG_INFINITY, 1.0)];
            for c in self.changes.iter().filter(|c| c.applies == kind) {
                let factor = out.last().unwrap().1 * (1.0 + c.change);
                out.push((c.effective - origin, factor));
            }
            out
        };
        let written = cuts(Applies::Written);
        let in_force = cuts(Applies::InForce);

        let mut portions = Vec::new();
        let mut average_level = 0.0;
        for (i, &(w0, fw)) in written.iter().enumerate() {
            let w1 = written.get(i + 1).map_or(f64::INFINITY, |x| x.0);
            let (a, b) = (w0.max(w_lo), w1.min(w_hi));
            if b <= a {
                continue;
            }
            for (j, &(t0, ft)) in in_force.iter().enumerate() {
                let t1 = in_force.get(j + 1).map_or(f64::INFINITY, |x| x.0);
                let (c, d) = (t0.max(t_lo), t1.min(t_hi));
                if d <= c {
                    continue;
                }
                let mut poly = vec![(a, c), (b, c), (b, d), (a, d)];
                for &(p, q, r) in &region {
                    poly = clip(&poly, p, q, r);
                }
                let share = area(&poly) / total;
                if share > 1e-15 {
                    let level = fw * ft;
                    average_level += share * level;
                    portions.push(Portion {
                        written_changes: i,
                        in_force_changes: j,
                        level,
                        share,
                    });
                }
            }
        }
        Ok(RateLevel {
            portions,
            average_level,
            current_level: self.current_level(),
        })
    }
}

/// Sutherland–Hodgman: the part of a convex polygon where `p w + q t <= r`.
fn clip(poly: &[(f64, f64)], p: f64, q: f64, r: f64) -> Vec<(f64, f64)> {
    let inside = |&(w, t): &(f64, f64)| p * w + q * t - r;
    let mut out = Vec::with_capacity(poly.len() + 1);
    for (k, cur) in poly.iter().enumerate() {
        let prev = &poly[(k + poly.len() - 1) % poly.len()];
        let (sc, sp) = (inside(cur), inside(prev));
        if sc <= 0.0 {
            if sp > 0.0 {
                out.push(cross(prev, cur, sp, sc));
            }
            out.push(*cur);
        } else if sp <= 0.0 {
            out.push(cross(prev, cur, sp, sc));
        }
    }
    out
}

fn cross(a: &(f64, f64), b: &(f64, f64), sa: f64, sb: f64) -> (f64, f64) {
    let u = sa / (sa - sb);
    (a.0 + u * (b.0 - a.0), a.1 + u * (b.1 - a.1))
}

/// Shoelace area of a polygon.
fn area(poly: &[(f64, f64)]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for (k, &(x0, y0)) in poly.iter().enumerate() {
        let (x1, y1) = poly[(k + 1) % poly.len()];
        s += x0 * y1 - x1 * y0;
    }
    0.5 * s.abs()
}

/// `(1 + rate)^years`, the factor of an exponential trend over `years`.
///
/// ```
/// use prospicio_pricing::ratemaking::trend_factor;
///
/// // Werner and Modlin, ch. 5: +2% a year over 2.5 years.
/// assert!((trend_factor(0.02, 2.5) - 1.0508).abs() < 1e-4);
/// ```
pub fn trend_factor(rate: f64, years: f64) -> f64 {
    (1.0 + rate).powf(years)
}

/// The shape of a fitted trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendKind {
    /// `ln y = intercept + slope × time`.
    Exponential,
    /// `y = intercept + slope × time`.
    Linear,
}

/// A trend fitted by least squares to values at times in years: the
/// exponential fit regresses `ln y` on time, the linear fit `y`. For
/// quarterly points pass times a quarter apart (or `0, 1, 2, …` divided by
/// 4); Werner and Modlin's annual trend `exp(4b) − 1` on an index `x` is
/// [`TrendFit::annual_change`].
///
/// ```
/// use prospicio_pricing::ratemaking::TrendFit;
///
/// let t = [0.0, 0.25, 0.5, 0.75];
/// let y: Vec<f64> = t.iter().map(|&t: &f64| 100.0 * 1.03f64.powf(t)).collect();
/// let fit = TrendFit::exponential(&t, &y).unwrap();
/// assert!((fit.annual_change() - 0.03).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrendFit {
    pub kind: TrendKind,
    pub intercept: f64,
    /// Per year, on the log scale for an exponential fit.
    pub slope: f64,
    /// Coefficient of determination, on the fitted scale.
    pub r_squared: f64,
    /// The number of points.
    pub n: usize,
}

impl TrendFit {
    /// Fit `ln y = a + b t`; every value must be positive.
    pub fn exponential(times: &[f64], values: &[f64]) -> Result<Self> {
        if let Some(&v) = values.iter().find(|v| !(v.is_finite() && **v > 0.0)) {
            return Err(Error::InvalidParameter {
                name: "values",
                value: v,
                reason: "must be finite and positive for an exponential trend",
            });
        }
        let logs: Vec<f64> = values.iter().map(|v| v.ln()).collect();
        Self::least_squares(TrendKind::Exponential, times, &logs)
    }

    /// Fit `y = a + b t`.
    pub fn linear(times: &[f64], values: &[f64]) -> Result<Self> {
        if let Some(&v) = values.iter().find(|v| !v.is_finite()) {
            return Err(Error::InvalidParameter {
                name: "values",
                value: v,
                reason: "must be finite",
            });
        }
        Self::least_squares(TrendKind::Linear, times, values)
    }

    fn least_squares(kind: TrendKind, x: &[f64], y: &[f64]) -> Result<Self> {
        if x.len() != y.len() {
            return Err(Error::Data(
                "times and values must have the same length".into(),
            ));
        }
        if x.len() < 2 {
            return Err(Error::Data("a trend needs at least two points".into()));
        }
        if let Some(&t) = x.iter().find(|t| !t.is_finite()) {
            return Err(Error::InvalidParameter {
                name: "times",
                value: t,
                reason: "must be finite",
            });
        }
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
        for (&xi, &yi) in x.iter().zip(y) {
            let (dx, dy) = (xi - mx, yi - my);
            sxx += dx * dx;
            sxy += dx * dy;
            syy += dy * dy;
        }
        if sxx <= 0.0 {
            return Err(Error::Data("the times must not all be equal".into()));
        }
        let slope = sxy / sxx;
        let r_squared = if syy > 0.0 {
            sxy * sxy / (sxx * syy)
        } else {
            1.0
        };
        Ok(Self {
            kind,
            intercept: my - slope * mx,
            slope,
            r_squared,
            n: x.len(),
        })
    }

    /// The fitted value at `time`.
    pub fn fitted(&self, time: f64) -> f64 {
        let v = self.intercept + self.slope * time;
        match self.kind {
            TrendKind::Exponential => v.exp(),
            TrendKind::Linear => v,
        }
    }

    /// The annual trend: `exp(slope) − 1` for an exponential fit, the
    /// slope (an amount a year) for a linear one.
    pub fn annual_change(&self) -> f64 {
        match self.kind {
            TrendKind::Exponential => self.slope.exp_m1(),
            TrendKind::Linear => self.slope,
        }
    }
}

/// One experience year for the loss ratio method: its earned premium
/// brought to the current rate level and trended, and its losses developed,
/// trended and loaded for loss adjustment expense (Werner and Modlin,
/// ch. 8). Factors that do not apply are 1 (a net trend sits in
/// `loss_trend` with `premium_trend` at 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossRatioYear {
    pub earned_premium: f64,
    /// To the current rate level.
    pub on_level_factor: f64,
    pub premium_trend: f64,
    /// Reported (or paid) losses, with whatever LAE they include.
    pub losses: f64,
    /// Loss development to ultimate.
    pub development: f64,
    pub loss_trend: f64,
    /// `1 +` the LAE ratio to losses not in `losses` (ULAE, say).
    pub lae_factor: f64,
}

impl LossRatioYear {
    /// Premium at the current rate level, trended to the forecast period.
    pub fn projected_premium(&self) -> f64 {
        self.earned_premium * self.on_level_factor * self.premium_trend
    }

    /// Ultimate losses and LAE, trended to the forecast period.
    pub fn projected_losses(&self) -> f64 {
        self.losses * self.development * self.loss_trend * self.lae_factor
    }

    /// `projected_losses / projected_premium`.
    pub fn loss_ratio(&self) -> f64 {
        self.projected_losses() / self.projected_premium()
    }
}

/// One experience year for the pure premium method: its losses developed,
/// trended and loaded for LAE, over its earned exposure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PurePremiumYear {
    pub exposure: f64,
    pub losses: f64,
    pub development: f64,
    pub loss_trend: f64,
    pub lae_factor: f64,
}

impl PurePremiumYear {
    /// Ultimate losses and LAE, trended to the forecast period.
    pub fn projected_losses(&self) -> f64 {
        self.losses * self.development * self.loss_trend * self.lae_factor
    }

    /// `projected_losses / exposure`.
    pub fn pure_premium(&self) -> f64 {
        self.projected_losses() / self.exposure
    }
}

fn check_positive(name: &'static str, value: f64) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name,
            value,
            reason: "must be finite and positive",
        })
    }
}

fn check_finite_non_negative(name: &'static str, value: f64) -> Result<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name,
            value,
            reason: "must be finite and not negative",
        })
    }
}

/// Experience years for the loss ratio method, combined as the ratio of
/// their totals (premium-weighted loss ratios, Werner and Modlin ch. 8).
///
/// ```
/// use prospicio_pricing::ratemaking::{LossRatioExperience, LossRatioYear};
///
/// let year = |premium, losses| LossRatioYear {
///     earned_premium: premium,
///     on_level_factor: 1.0,
///     premium_trend: 1.0,
///     losses,
///     development: 1.0,
///     loss_trend: 1.0,
///     lae_factor: 1.0,
/// };
/// let e = LossRatioExperience::new(&[year(100.0, 50.0), year(300.0, 250.0)]).unwrap();
/// assert_eq!(e.loss_ratio(), 0.75);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LossRatioExperience {
    years: Vec<LossRatioYear>,
}

impl LossRatioExperience {
    /// At least one year; premium and factors positive, losses not negative.
    pub fn new(years: &[LossRatioYear]) -> Result<Self> {
        if years.is_empty() {
            return Err(Error::Data("an experience period needs a year".into()));
        }
        for y in years {
            check_positive("earned_premium", y.earned_premium)?;
            check_positive("on_level_factor", y.on_level_factor)?;
            check_positive("premium_trend", y.premium_trend)?;
            check_finite_non_negative("losses", y.losses)?;
            check_positive("development", y.development)?;
            check_positive("loss_trend", y.loss_trend)?;
            check_positive("lae_factor", y.lae_factor)?;
        }
        Ok(Self {
            years: years.to_vec(),
        })
    }

    /// The years.
    pub fn years(&self) -> &[LossRatioYear] {
        &self.years
    }

    /// Total projected premium at the current rate level.
    pub fn projected_premium(&self) -> f64 {
        self.years.iter().map(|y| y.projected_premium()).sum()
    }

    /// Total projected ultimate losses and LAE.
    pub fn projected_losses(&self) -> f64 {
        self.years.iter().map(|y| y.projected_losses()).sum()
    }

    /// The projected loss and LAE ratio, the ratio of the totals.
    pub fn loss_ratio(&self) -> f64 {
        self.projected_losses() / self.projected_premium()
    }
}

/// Experience years for the pure premium method, combined as the ratio of
/// their totals (exposure-weighted pure premiums).
#[derive(Debug, Clone, PartialEq)]
pub struct PurePremiumExperience {
    years: Vec<PurePremiumYear>,
}

impl PurePremiumExperience {
    /// At least one year; exposure and factors positive, losses not
    /// negative.
    pub fn new(years: &[PurePremiumYear]) -> Result<Self> {
        if years.is_empty() {
            return Err(Error::Data("an experience period needs a year".into()));
        }
        for y in years {
            check_positive("exposure", y.exposure)?;
            check_finite_non_negative("losses", y.losses)?;
            check_positive("development", y.development)?;
            check_positive("loss_trend", y.loss_trend)?;
            check_positive("lae_factor", y.lae_factor)?;
        }
        Ok(Self {
            years: years.to_vec(),
        })
    }

    /// The years.
    pub fn years(&self) -> &[PurePremiumYear] {
        &self.years
    }

    /// Total earned exposure.
    pub fn exposure(&self) -> f64 {
        self.years.iter().map(|y| y.exposure).sum()
    }

    /// Total projected ultimate losses and LAE.
    pub fn projected_losses(&self) -> f64 {
        self.years.iter().map(|y| y.projected_losses()).sum()
    }

    /// The projected pure premium, the ratio of the totals.
    pub fn pure_premium(&self) -> f64 {
        self.projected_losses() / self.exposure()
    }
}

/// The provisions of the price that are proportional to it: variable
/// expenses `V` and the target underwriting profit `Q` (Werner and Modlin,
/// ch. 7 and 8). Fixed expenses enter each method separately, as a ratio to
/// premium or an amount per exposure.
///
/// ```
/// use prospicio_pricing::ratemaking::Provisions;
///
/// // Werner and Modlin, ch. 8: V = 25%, Q = 10%.
/// let p = Provisions::new(0.25, 0.10).unwrap();
/// // Pure premium method: ($300 + $25) / 0.65 = $500.
/// assert!((p.pure_premium_rate(300.0, 25.0) - 500.0).abs() < 1e-12);
/// // Loss ratio method: (65% + 6.5%) / 65% - 1 = +10%.
/// assert!((p.loss_ratio_change(0.65, 0.065) - 0.10).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Provisions {
    pub variable: f64,
    pub profit: f64,
}

impl Provisions {
    /// Variable expense and profit as fractions of premium; the profit may
    /// be negative (investment income), and `variable + profit < 1`.
    pub fn new(variable: f64, profit: f64) -> Result<Self> {
        if !(variable.is_finite() && (0.0..1.0).contains(&variable)) {
            return Err(Error::InvalidParameter {
                name: "variable",
                value: variable,
                reason: "must be in [0, 1)",
            });
        }
        if !(profit.is_finite() && variable + profit < 1.0) {
            return Err(Error::InvalidParameter {
                name: "profit",
                value: profit,
                reason: "must be finite with variable + profit < 1",
            });
        }
        Ok(Self { variable, profit })
    }

    /// `1 − V − Q`, the variable permissible loss ratio.
    pub fn variable_permissible_loss_ratio(&self) -> f64 {
        1.0 - self.variable - self.profit
    }

    /// `1 − F − V − Q`, the permissible loss ratio with a fixed expense
    /// ratio `F`.
    pub fn permissible_loss_ratio(&self, fixed_ratio: f64) -> f64 {
        self.variable_permissible_loss_ratio() - fixed_ratio
    }

    /// The loss ratio method's indicated change, `(LR + F) / (1 − V − Q) −
    /// 1`, from the projected loss and LAE ratio at current rates and the
    /// fixed expense ratio.
    pub fn loss_ratio_change(&self, loss_ratio: f64, fixed_ratio: f64) -> f64 {
        (loss_ratio + fixed_ratio) / self.variable_permissible_loss_ratio() - 1.0
    }

    /// The pure premium method's indicated average rate, `(PP + fixed) /
    /// (1 − V − Q)`, from the projected pure premium (with LAE) and the
    /// fixed expense per exposure. Other costs per exposure, such as net
    /// reinsurance, add to either.
    pub fn pure_premium_rate(&self, pure_premium: f64, fixed_per_exposure: f64) -> f64 {
        (pure_premium + fixed_per_exposure) / self.variable_permissible_loss_ratio()
    }
}

/// An expense ratio to premium and the share of it that is fixed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpenseCategory {
    pub ratio: f64,
    pub fixed_share: f64,
}

/// The fixed and variable expense ratios of the premium-based projection
/// method: each category's ratio split by its fixed share, summed over the
/// categories, as `(fixed, variable)`. With every share 0 it is the
/// all-variable method.
pub fn premium_based_expenses(categories: &[ExpenseCategory]) -> Result<(f64, f64)> {
    let (mut fixed, mut variable) = (0.0, 0.0);
    for c in categories {
        check_finite_non_negative("ratio", c.ratio)?;
        if !(0.0..=1.0).contains(&c.fixed_share) {
            return Err(Error::InvalidParameter {
                name: "fixed_share",
                value: c.fixed_share,
                reason: "must be in [0, 1]",
            });
        }
        fixed += c.ratio * c.fixed_share;
        variable += c.ratio * (1.0 - c.fixed_share);
    }
    Ok((fixed, variable))
}

/// `Z × estimate + (1 − Z) × complement`.
pub fn credibility_weighted(estimate: f64, credibility: f64, complement: f64) -> f64 {
    credibility * estimate + (1.0 - credibility) * complement
}

/// The trended present rates complement as an indicated change (Werner
/// and Modlin, ch. 12): the residual of the last review,
/// `(1 + prior indicated) / (1 + prior implemented)`, times `trend`, the
/// factor from the last review's target effective date to the new one (the
/// loss trend factor over the premium trend factor in the loss ratio
/// method), less 1. For a rate, multiply the present rate by one plus it.
///
/// ```
/// use prospicio_pricing::ratemaking::{trend_factor, trended_present_rates};
///
/// // $200 present rate, 5% trend for 2 years, +10% indicated, +6% taken.
/// let change = trended_present_rates(0.10, 0.06, trend_factor(0.05, 2.0));
/// assert!((200.0 * (1.0 + change) - 228.82).abs() < 0.01);
/// ```
pub fn trended_present_rates(prior_indicated: f64, prior_implemented: f64, trend: f64) -> f64 {
    (1.0 + prior_indicated) / (1.0 + prior_implemented) * trend - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cy(y: f64) -> Period {
        Period::calendar(y, y + 1.0).unwrap()
    }

    #[test]
    fn no_changes_is_level_one() {
        let h = RateHistory::new(&[]).unwrap();
        for term in [0.25, 0.5, 1.0, 2.0] {
            for p in [cy(2011.0), Period::policy(2011.0, 2012.0).unwrap()] {
                let l = h.rate_level(&p, term).unwrap();
                assert!((l.average_level - 1.0).abs() < 1e-14);
                assert!((l.portions.iter().map(|p| p.share).sum::<f64>() - 1.0).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn change_before_the_window_is_fully_in() {
        let h = RateHistory::new(&[RateChange::written(2000.0, 0.2)]).unwrap();
        let l = h.rate_level(&cy(2011.0), 1.0).unwrap();
        assert!((l.on_level_factor() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn in_force_change_at_window_start_applies_to_all() {
        let h = RateHistory::new(&[RateChange::in_force(2011.0, -0.05)]).unwrap();
        let l = h.rate_level(&cy(2011.0), 1.0).unwrap();
        assert!((l.average_level - 0.95).abs() < 1e-14);
        assert!((l.on_level_factor() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn law_change_table_5_23() {
        let h = RateHistory::new(&[
            RateChange::written(2010.5, 0.05),
            RateChange::written(2011.0, 0.10),
            RateChange::in_force(2011.5, -0.05),
            RateChange::written(2012.25, -0.01),
        ])
        .unwrap();
        let l = h.rate_level(&cy(2011.0), 1.0).unwrap();
        let shares: Vec<f64> = l.portions.iter().map(|p| p.share).collect();
        let want = [0.125, 0.25, 0.125, 0.125, 0.375];
        assert_eq!(shares.len(), want.len());
        for (s, w) in shares.iter().zip(want) {
            assert!((s - w).abs() < 1e-14, "{shares:?}");
        }
        assert!((l.current_level - 1.0862775).abs() < 1e-12);
        assert!((l.on_level_factor() - 1.017084).abs() < 1e-6);
    }

    #[test]
    fn from_levels_reproduces_printed_levels() {
        let h = RateHistory::from_levels(
            &[2011.25, 2012.5],
            &[0.95, 1.045],
            &[Applies::Written, Applies::Written],
        )
        .unwrap();
        assert!((h.current_level() - 1.045).abs() < 1e-15);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(RateHistory::new(&[RateChange::written(2011.0, -1.0)]).is_err());
        assert!(
            RateHistory::new(&[
                RateChange::written(2012.0, 0.1),
                RateChange::written(2011.0, 0.1)
            ])
            .is_err()
        );
        assert!(Period::calendar(2011.0, 2011.0).is_err());
        assert!(year_fraction(2011, 2, 29).is_err());
        assert!(year_fraction(2012, 2, 29).is_ok());
        assert!(TrendFit::exponential(&[0.0, 1.0], &[1.0, 0.0]).is_err());
        assert!(TrendFit::linear(&[1.0, 1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn half_month_grid() {
        assert_eq!(year_fraction(2011, 1, 1).unwrap(), 2011.0);
        assert_eq!(year_fraction(2011, 7, 1).unwrap(), 2011.5);
        assert_eq!(year_fraction(2011, 6, 30).unwrap(), 2011.5);
        assert_eq!(year_fraction(2017, 9, 30).unwrap(), 2017.75);
        assert_eq!(year_fraction(2014, 3, 1).unwrap(), 2014.0 + 2.0 / 12.0);
    }
}
