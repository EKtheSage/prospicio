//! Earned exposure and premium from policies: what each policy's term
//! puts in a calendar window, or in the window it was written in. See
//! `docs/design/primary-rating.md`.
//!
//! The parallelogram method of [`ratemaking`](crate::ratemaking) assumes
//! policies written evenly over time, because it has only totals. With the
//! policies themselves nothing is assumed: a policy with an amount (an
//! exposure count, or a premium) earns it evenly over its term, and the
//! part earned in a window is the share of the term's days that fall in
//! it.
//!
//! Following Werner and Modlin (*Basic Ratemaking*, chapter 4), a window
//! gets four figures:
//!
//! | Figure | A policy's contribution |
//! |---|---|
//! | written | its whole amount, in the window that holds its effective date |
//! | earned | its amount times the share of its term inside the window |
//! | unearned | at the window's end: the part of a written policy not yet earned |
//! | in force | at the window's end: its amount, if written and not yet expired |
//!
//! "At the window's end" is the close of its last day: a policy that
//! expires the next morning is still in force, and one effective the next
//! morning is not yet written. A valuation date is read the same way, as
//! the close of the day before it.
//!
//! A window is aggregated by calendar period
//! ([`PeriodKind::Calendar`]: what was written, and what was earned, in
//! the window) or by policy period ([`PeriodKind::Policy`]: the policies
//! written in the window, with everything they earn wherever it falls).
//!
//! Dates are days: a term runs from its effective date up to, and not
//! including, its expiration date (the 12:01 a.m. convention), so an
//! annual policy has 365 or 366 days.
//!
//! A policy can change mid-term ([`Policy::endorsed`]: a new full-term
//! amount from a date) and be cancelled ([`Policy::cancelled`]). Each is
//! written when it happens: a change as the difference in full-term
//! amounts for the term still to run, a cancellation as the return of
//! what was not yet earned, on the last day in force. So a calendar
//! period's written amount can be negative, while a policy period holds
//! everything its policies wrote, net.
//!
//! [`EarningMethod`] says how time is counted: by exact days, on the
//! half-month grid the ratemaking exhibits use, or by the "15th of the
//! month" rule (twenty-fourths) for data summarized by month.
//!
//! ```
//! use prospicio_pricing::earned::{Date, EarningMethod, Policy, Windows, earned_exposure};
//! use prospicio_pricing::ratemaking::PeriodKind;
//!
//! // One car-year written on 1 October 2010.
//! let policy = Policy::new(
//!     Date::new(2010, 10, 1).unwrap(),
//!     Date::new(2011, 10, 1).unwrap(),
//!     1.0,
//! )
//! .unwrap();
//! let table = earned_exposure(
//!     &[policy],
//!     &Windows::years(2010, 2011).unwrap(),
//!     PeriodKind::Calendar,
//!     EarningMethod::Daily,
//!     None,
//! )
//! .unwrap();
//! let rows = table.rows();
//! // 92 of its 365 days fall in 2010.
//! assert_eq!(rows[0].written, 1.0);
//! assert!((rows[0].earned - 92.0 / 365.0).abs() < 1e-15);
//! assert!((rows[0].unearned - 273.0 / 365.0).abs() < 1e-15);
//! assert_eq!((rows[0].in_force, rows[1].written, rows[1].in_force), (1.0, 0.0, 0.0));
//! ```

use std::collections::BTreeMap;
use std::fmt;

use prospicio_core::{Error, Result};

use crate::ratemaking::PeriodKind;

/// A calendar date in the proleptic Gregorian calendar, held as a day
/// number, so that dates subtract to days. No time of day and no time
/// zone.
///
/// ```
/// use prospicio_pricing::earned::Date;
///
/// let a = Date::new(2024, 2, 28).unwrap();
/// let b = Date::parse("2024-03-01").unwrap();
/// assert_eq!(b.days_since(a), 2); // 2024 is a leap year
/// assert_eq!(b.to_string(), "2024-03-01");
/// assert!(Date::new(2023, 2, 29).is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Date {
    /// Days since 1970-01-01.
    days: i32,
}

/// The days in `month` of `year`.
fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ if is_leap(year) => 29,
        _ => 28,
    }
}

fn is_leap(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

impl Date {
    /// The date `year`-`month`-`day`. Fails for a month outside 1 to 12, a
    /// day the month does not have, or a year outside 1 to 9999.
    pub fn new(year: i32, month: u8, day: u8) -> Result<Self> {
        if !(1..=9999).contains(&year) {
            return Err(Error::InvalidParameter {
                name: "year",
                value: f64::from(year),
                reason: "must be between 1 and 9999",
            });
        }
        if !(1..=12).contains(&month) {
            return Err(Error::InvalidMonth(month));
        }
        if day == 0 || day > days_in_month(year, month) {
            return Err(Error::InvalidParameter {
                name: "day",
                value: f64::from(day),
                reason: "the month has no such day",
            });
        }
        // Days from the civil date (Hinnant's algorithm), with the year
        // starting in March so that the leap day comes last.
        let y = i64::from(year) - i64::from(month <= 2);
        let era = y.div_euclid(400);
        let year_of_era = y.rem_euclid(400);
        let shifted_month = (i64::from(month) + 9) % 12;
        let day_of_year = (153 * shifted_month + 2) / 5 + i64::from(day) - 1;
        let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
        Ok(Self {
            days: (era * 146_097 + day_of_era - 719_468) as i32,
        })
    }

    /// The date written `YYYY-MM-DD`. Fails on any other form, or on a
    /// date that does not exist.
    pub fn parse(iso: &str) -> Result<Self> {
        let bad = || Error::Data(format!("{iso:?} is not a date of the form YYYY-MM-DD"));
        let bytes = iso.as_bytes();
        if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
            return Err(bad());
        }
        let number = |range: std::ops::Range<usize>| -> Result<u32> {
            let part = &iso[range];
            if part.bytes().all(|b| b.is_ascii_digit()) {
                part.parse().map_err(|_| bad())
            } else {
                Err(bad())
            }
        };
        let (year, month, day) = (number(0..4)?, number(5..7)?, number(8..10)?);
        Self::new(year as i32, month as u8, day as u8).map_err(|_| bad())
    }

    /// The year, month and day.
    pub fn ymd(self) -> (i32, u8, u8) {
        // Hinnant's civil-from-days.
        let z = i64::from(self.days) + 719_468;
        let era = z.div_euclid(146_097);
        let day_of_era = z.rem_euclid(146_097);
        let year_of_era =
            (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
        let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
        let shifted_month = (5 * day_of_year + 2) / 153;
        let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
        let month = if shifted_month < 10 {
            shifted_month + 3
        } else {
            shifted_month - 9
        };
        let year = year_of_era + era * 400 + i64::from(month <= 2);
        (year as i32, month as u8, day as u8)
    }

    /// The days from `earlier` to this date; negative if this date is the
    /// earlier one.
    pub fn days_since(self, earlier: Date) -> i64 {
        i64::from(self.days) - i64::from(earlier.days)
    }

    /// The date as an exact year fraction: 1 July 2011 is `2011 + 181 /
    /// 365`. [`year_fraction`](crate::ratemaking::year_fraction) rounds to
    /// the half month instead, as the ratemaking exhibits do.
    pub fn year_fraction(self) -> f64 {
        let (year, _, _) = self.ymd();
        let start = Date::new(year, 1, 1).expect("the year is in range");
        let length = if is_leap(year) { 366.0 } else { 365.0 };
        f64::from(year) + self.days_since(start) as f64 / length
    }

    /// The date `days` days later, or earlier for a negative number.
    pub fn plus_days(self, days: i64) -> Date {
        Date {
            days: (i64::from(self.days) + days) as i32,
        }
    }

    /// The date on the half-month grid of
    /// [`year_fraction`](crate::ratemaking::year_fraction): the first of a
    /// month is its start, the middle half way, the last day the next
    /// month's start.
    fn half_month(self) -> f64 {
        let (year, month, day) = self.ymd();
        crate::ratemaking::year_fraction(year, month, day).expect("a Date is a valid date")
    }

    /// The middle of the date's month, as a year fraction.
    fn mid_month(self) -> f64 {
        let (year, month, _) = self.ymd();
        f64::from(year) + (f64::from(month) - 0.5) / 12.0
    }
}

impl fmt::Display for Date {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (year, month, day) = self.ymd();
        write!(f, "{year:04}-{month:02}-{day:02}")
    }
}

/// A policy: a term and an amount earned evenly over it, with any
/// mid-term changes and a cancellation.
///
/// The amount is whatever is being earned, for the full term: an exposure
/// count (1 for one car-year on an annual policy) or the term's premium.
///
/// ```
/// use prospicio_pricing::earned::{Date, Policy};
///
/// let d = |y, m, day| Date::new(y, m, day).unwrap();
/// // Werner and Modlin's Policy D: 400 for the year from 1 July 2011, changed
/// // on 1 January 2012 to a full-term premium of 800.
/// let policy = Policy::new(d(2011, 7, 1), d(2012, 7, 1), 400.0)
///     .unwrap()
///     .endorsed(d(2012, 1, 1), 800.0)
///     .unwrap();
/// assert_eq!(policy.amount_at(d(2011, 12, 31)), 400.0);
/// assert_eq!(policy.amount_at(d(2012, 1, 1)), 800.0);
/// // The policyholder pays about half of each: 184 and 182 of 366 days.
/// let paid = policy.earned_between(d(2011, 7, 1), d(2012, 7, 1));
/// assert!((paid - (400.0 * 184.0 + 800.0 * 182.0) / 366.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Policy {
    effective: Date,
    expiration: Date,
    amount: f64,
    label: String,
    /// Mid-term changes: the date and the full-term amount from it on, in
    /// date order.
    changes: Vec<(Date, f64)>,
    /// The first day the policy is no longer in force, if cancelled.
    cancellation: Option<Date>,
}

fn check_amount(amount: f64) -> Result<f64> {
    if amount.is_finite() && amount >= 0.0 {
        Ok(amount)
    } else {
        Err(Error::InvalidParameter {
            name: "amount",
            value: amount,
            reason: "must be finite and non-negative",
        })
    }
}

impl Policy {
    /// A policy in force from `effective` up to, and not including,
    /// `expiration`, earning `amount` over that term.
    ///
    /// Fails unless the expiration is after the effective date and the
    /// amount is finite and non-negative.
    pub fn new(effective: Date, expiration: Date, amount: f64) -> Result<Self> {
        if expiration <= effective {
            return Err(Error::Data(format!(
                "a policy effective {effective} cannot expire {expiration}: \
                 the expiration must be later"
            )));
        }
        Ok(Self {
            effective,
            expiration,
            amount: check_amount(amount)?,
            label: String::new(),
            changes: Vec::new(),
            cancellation: None,
        })
    }

    /// The policy with a label (a state, a class, a program): the table
    /// has a set of rows for each label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// The policy with a mid-term change: from `date` on its full-term
    /// amount is `full_term_amount` (a changed exposure, or the full-term
    /// premium after an endorsement). The change is written on `date`, as
    /// the difference between the new and the old amount for the part of
    /// the term still to run.
    ///
    /// Fails unless `date` is inside the term, after any earlier change
    /// and before any cancellation, and the amount is finite and
    /// non-negative.
    pub fn endorsed(mut self, date: Date, full_term_amount: f64) -> Result<Self> {
        let after = self.changes.last().map_or(self.effective, |c| c.0);
        if date <= after || date >= self.end() {
            return Err(Error::Data(format!(
                "a change on {date} is not after {after} and before {}",
                self.end()
            )));
        }
        self.changes.push((date, check_amount(full_term_amount)?));
        Ok(self)
    }

    /// The policy cancelled: `date` is the first day it is no longer in
    /// force. What was not yet earned is written off on the last day in
    /// force, the day before.
    ///
    /// Fails unless `date` is after the effective date, after any change
    /// and before the expiration, or if the policy is already cancelled.
    pub fn cancelled(mut self, date: Date) -> Result<Self> {
        if self.cancellation.is_some() {
            return Err(Error::Data("the policy is already cancelled".into()));
        }
        let after = self.changes.last().map_or(self.effective, |c| c.0);
        if date <= after || date >= self.expiration {
            return Err(Error::Data(format!(
                "a cancellation on {date} is not after {after} and before the expiration {}",
                self.expiration
            )));
        }
        self.cancellation = Some(date);
        Ok(self)
    }

    /// The effective date.
    pub fn effective(&self) -> Date {
        self.effective
    }

    /// The expiration date of the term as written, the first day the
    /// policy would not be in force.
    pub fn expiration(&self) -> Date {
        self.expiration
    }

    /// The cancellation date, the first day not in force, if cancelled.
    pub fn cancellation(&self) -> Option<Date> {
        self.cancellation
    }

    /// The mid-term changes: each date with the full-term amount from it.
    pub fn changes(&self) -> &[(Date, f64)] {
        &self.changes
    }

    /// The full-term amount at inception.
    pub fn amount(&self) -> f64 {
        self.amount
    }

    /// The label; empty unless one was given.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The term as written, in days.
    pub fn term_days(&self) -> i64 {
        self.expiration.days_since(self.effective)
    }

    /// The first day the policy is not in force: its cancellation, or its
    /// expiration.
    fn end(&self) -> Date {
        self.cancellation.unwrap_or(self.expiration)
    }

    /// The full-term amount in effect on `date`: Werner and Modlin's
    /// in-force amount. Zero when the policy is not in force.
    pub fn amount_at(&self, date: Date) -> f64 {
        if !self.in_force_at(date) {
            return 0.0;
        }
        self.changes
            .iter()
            .rev()
            .find(|c| c.0 <= date)
            .map_or(self.amount, |c| c.1)
    }

    /// The amount earned in `[from, to)` by exact days: each day of the
    /// term in the window earns the full-term amount then in effect over
    /// the term's days. Zero when they do not overlap.
    pub fn earned_between(&self, from: Date, to: Date) -> f64 {
        let m = EarningMethod::Daily;
        self.earned_in(m, m.bound(from), m.bound(to))
    }

    /// The amount still to be earned from `date` on, by exact days: all
    /// that the policy earns before its effective date, nothing from its
    /// expiration or cancellation on.
    pub fn unearned_at(&self, date: Date) -> f64 {
        let m = EarningMethod::Daily;
        self.earned_in(m, m.bound(date), f64::INFINITY)
    }

    /// Whether the policy is in force on `date`: effective on or before
    /// it, and not yet expired or cancelled.
    pub fn in_force_at(&self, date: Date) -> bool {
        self.effective <= date && date < self.end()
    }

    /// The amount earned between the coordinates `a` and `b` of `method`.
    fn earned_in(&self, method: EarningMethod, a: f64, b: f64) -> f64 {
        let start = method.event(self.effective);
        let term = method.event(self.expiration) - start;
        let end = method.event(self.end());
        if term <= 0.0 {
            // A term shorter than the method's grid is earned at once.
            return if a <= start && start < b {
                self.amount
            } else {
                0.0
            };
        }
        let mut earned = 0.0;
        let mut from = start;
        let mut amount = self.amount;
        let steps = self.changes.iter().map(|c| (method.event(c.0), c.1));
        for (to, next) in steps.chain(std::iter::once((end, 0.0))) {
            let to = to.min(end);
            let overlap = to.min(b) - from.max(a);
            if overlap > 0.0 {
                earned += amount * overlap / term;
            }
            from = to;
            amount = next;
        }
        earned
    }

    /// The written transactions: the date each is booked on and its
    /// amount. The whole amount at inception; at a change, the difference
    /// in full-term amounts for the term still to run; at a cancellation,
    /// on the last day in force, the return of what was not yet earned.
    fn transactions(&self, method: EarningMethod) -> Vec<(Date, f64)> {
        let start = method.event(self.effective);
        let expiry = method.event(self.expiration);
        let term = expiry - start;
        let to_run = |date: Date| {
            if term > 0.0 {
                ((expiry - method.event(date)) / term).clamp(0.0, 1.0)
            } else {
                0.0
            }
        };
        let mut out = vec![(self.effective, self.amount)];
        let mut amount = self.amount;
        for &(date, next) in &self.changes {
            out.push((date, (next - amount) * to_run(date)));
            amount = next;
        }
        if let Some(date) = self.cancellation {
            out.push((date.plus_days(-1), -amount * to_run(date)));
        }
        out
    }

    /// The full-term amount in force at the close of the day before
    /// `end`, on `method`'s grid: how a window ending at `end` sees the
    /// policy. One that expires on `end` is still in force, one effective
    /// on `end` is not yet, and a change on `end` has not yet happened.
    fn in_force_at_close(&self, method: EarningMethod, end: Date) -> f64 {
        let at = method.bound(end);
        if !(method.event(self.effective) < at && at <= method.event(self.end())) {
            return 0.0;
        }
        self.changes
            .iter()
            .rev()
            .find(|c| method.event(c.0) < at)
            .map_or(self.amount, |c| c.1)
    }
}

/// How a policy's amount is earned over its term.
///
/// ```
/// use prospicio_pricing::earned::{Date, EarningMethod, Policy, Windows, earned_exposure};
/// use prospicio_pricing::ratemaking::PeriodKind;
///
/// // An annual policy from 1 October 2010: its share earned in 2010.
/// let d = |y, m, day| Date::new(y, m, day).unwrap();
/// let policy = [Policy::new(d(2010, 10, 1), d(2011, 10, 1), 1.0).unwrap()];
/// let earned_2010 = |method| {
///     let years = Windows::years(2010, 2011).unwrap();
///     earned_exposure(&policy, &years, PeriodKind::Calendar, method, None).unwrap().rows()[0].earned
/// };
/// assert!((earned_2010(EarningMethod::Daily) - 92.0 / 365.0).abs() < 1e-15);
/// assert!((earned_2010(EarningMethod::HalfMonth) - 0.25).abs() < 1e-12); // three months of twelve
/// assert!((earned_2010(EarningMethod::TwentyFourths) - 5.0 / 24.0).abs() < 1e-12); // from 15 October
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum EarningMethod {
    /// Evenly by day, on the exact days of the term.
    #[default]
    Daily,
    /// Evenly by half month, on the grid of
    /// [`year_fraction`](crate::ratemaking::year_fraction): every date is
    /// taken to the nearest start or middle of a month, so a month is a
    /// twelfth of a year whatever its length. Werner and Modlin's exhibits
    /// count this way.
    HalfMonth,
    /// The "15th of the month" rule, for data summarized by month: every
    /// policy date (effective, expiration, change, cancellation) is taken
    /// to the middle of its month, so an annual policy earns 1/24 in the
    /// month it is written, 2/24 in each of the next eleven and 1/24 in
    /// the thirteenth.
    TwentyFourths,
}

impl EarningMethod {
    /// A policy date (effective, expiration, change, cancellation) on the
    /// method's time axis.
    fn event(self, date: Date) -> f64 {
        match self {
            Self::Daily => f64::from(date.days),
            Self::HalfMonth => date.half_month(),
            Self::TwentyFourths => date.mid_month(),
        }
    }

    /// A window bound or a valuation date on the method's time axis.
    fn bound(self, date: Date) -> f64 {
        match self {
            Self::Daily => f64::from(date.days),
            Self::HalfMonth | Self::TwentyFourths => date.half_month(),
        }
    }
}

/// Consecutive windows `[b_0, b_1), [b_1, b_2), …` between dates.
///
/// ```
/// use prospicio_pricing::earned::{Date, Windows};
///
/// let w = Windows::quarters(2024, 2024).unwrap();
/// assert_eq!(w.len(), 4);
/// assert_eq!(w.bounds()[1], Date::new(2024, 4, 1).unwrap());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Windows {
    bounds: Vec<Date>,
}

impl Windows {
    /// The windows between consecutive `bounds`. Fails unless there are at
    /// least two, strictly increasing.
    pub fn new(bounds: Vec<Date>) -> Result<Self> {
        if bounds.len() < 2 {
            return Err(Error::Data(
                "a window needs a start and an end: give at least two dates".into(),
            ));
        }
        if let Some(w) = bounds.windows(2).find(|w| w[1] <= w[0]) {
            return Err(Error::Data(format!(
                "the window bounds must increase: {} is not after {}",
                w[1], w[0]
            )));
        }
        Ok(Self { bounds })
    }

    /// One window per `months` months, from 1 January of `first_year`
    /// through 31 December of `last_year`.
    fn every(first_year: i32, last_year: i32, months: u8) -> Result<Self> {
        if last_year < first_year {
            return Err(Error::Data(format!(
                "the last year {last_year} is before the first year {first_year}"
            )));
        }
        let mut bounds = Vec::new();
        for year in first_year..=last_year {
            for month in (1..=12).step_by(usize::from(months)) {
                bounds.push(Date::new(year, month, 1)?);
            }
        }
        bounds.push(Date::new(last_year + 1, 1, 1)?);
        Self::new(bounds)
    }

    /// Calendar years `first_year` to `last_year`, both included.
    pub fn years(first_year: i32, last_year: i32) -> Result<Self> {
        Self::every(first_year, last_year, 12)
    }

    /// Calendar quarters of the years `first_year` to `last_year`.
    pub fn quarters(first_year: i32, last_year: i32) -> Result<Self> {
        Self::every(first_year, last_year, 3)
    }

    /// Calendar months of the years `first_year` to `last_year`.
    pub fn months(first_year: i32, last_year: i32) -> Result<Self> {
        Self::every(first_year, last_year, 1)
    }

    /// The dates the windows lie between, one more than there are windows.
    pub fn bounds(&self) -> &[Date] {
        &self.bounds
    }

    /// The number of windows.
    pub fn len(&self) -> usize {
        self.bounds.len() - 1
    }

    /// Whether there is no window; never, since a `Windows` has at least
    /// one.
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// The amount in force on `date`: the sum of the full-term amounts, as
/// they stand on that date, of the policies with an effective date on or
/// before it and an expiration or cancellation after it
/// (Werner and Modlin's in-force exposure as of a date, a snapshot with no
/// regard to how long each policy runs).
///
/// A window's `in_force` in an [`EarnedTable`] is this on the window's
/// last day.
///
/// ```
/// use prospicio_pricing::earned::{Date, Policy, in_force};
///
/// let d = |y, m, day| Date::new(y, m, day).unwrap();
/// let policies = [
///     Policy::new(d(2010, 10, 1), d(2011, 10, 1), 1.0).unwrap(),
///     Policy::new(d(2011, 1, 1), d(2012, 1, 1), 1.0).unwrap(),
/// ];
/// assert_eq!(in_force(&policies, d(2011, 1, 1)), 2.0);
/// assert_eq!(in_force(&policies, d(2011, 10, 1)), 1.0); // the first has expired
/// ```
pub fn in_force(policies: &[Policy], date: Date) -> f64 {
    policies.iter().map(|p| p.amount_at(date)).sum()
}

/// One window's figures for one label: [`EarnedTable::rows`].
#[derive(Debug, Clone, PartialEq)]
pub struct EarnedRow {
    /// The policies' label; empty when none was given.
    pub label: String,
    /// The window's first day.
    pub from: Date,
    /// The day after the window's last.
    pub to: Date,
    /// The amount written: the policies effective in the window.
    pub written: f64,
    /// The amount earned: in the window for a calendar period, and by the
    /// window's policies, up to the valuation date, for a policy period.
    pub earned: f64,
    /// The amount written and not yet earned: at the window's end for a
    /// calendar period, at the valuation date for a policy period.
    pub unearned: f64,
    /// The amount in force: at the window's end for a calendar period, at
    /// the valuation date for a policy period.
    pub in_force: f64,
}

/// Written, earned, unearned and in-force amounts by window and label:
/// [`earned_exposure`].
#[derive(Debug, Clone, PartialEq)]
pub struct EarnedTable {
    kind: PeriodKind,
    rows: Vec<EarnedRow>,
}

impl EarnedTable {
    /// Whether the windows are calendar periods or policy periods.
    pub fn kind(&self) -> PeriodKind {
        self.kind
    }

    /// The rows: the labels in order, and for each the windows in order.
    pub fn rows(&self) -> &[EarnedRow] {
        &self.rows
    }

    /// The rows of one label.
    pub fn label(&self, label: &str) -> Vec<&EarnedRow> {
        self.rows.iter().filter(|r| r.label == label).collect()
    }
}

/// Written, earned, unearned and in-force amounts of `policies` in each
/// window, for each label.
///
/// With [`PeriodKind::Calendar`] a window holds what happened in it: the
/// amounts written by policies effective in it, the amounts earned on its
/// days by any policy, and the unearned and in-force amounts at its end.
///
/// With [`PeriodKind::Policy`] a window holds the policies effective in
/// it, with all they earn, wherever it falls: at ultimate the earned
/// amount is the written one. Give `valuation` to see the policy periods
/// as of a date: earned stops there, and the unearned and in-force
/// amounts are at that date.
///
/// `valuation` also cuts a calendar table off: nothing is written or
/// earned from that date on.
///
/// A policy effective before the first window or after the last is not
/// written in any window, though a calendar window still gets what it
/// earns there.
///
/// ```
/// use prospicio_pricing::earned::{Date, EarningMethod, Policy, Windows, earned_exposure};
/// use prospicio_pricing::ratemaking::PeriodKind;
///
/// let d = |y, m, day| Date::new(y, m, day).unwrap();
/// // Two annual policies of 1,200 premium, written 1 July 2023 and 1 April 2024.
/// let policies = [
///     Policy::new(d(2023, 7, 1), d(2024, 7, 1), 1200.0).unwrap(),
///     Policy::new(d(2024, 4, 1), d(2025, 4, 1), 1200.0).unwrap(),
/// ];
/// let years = Windows::years(2023, 2025).unwrap();
/// let by_policy_year =
///     earned_exposure(&policies, &years, PeriodKind::Policy, EarningMethod::Daily, None).unwrap();
/// // At ultimate a policy year has earned all it wrote.
/// assert_eq!(by_policy_year.rows()[1].written, 1200.0);
/// assert!((by_policy_year.rows()[1].earned - 1200.0).abs() < 1e-12);
/// ```
pub fn earned_exposure(
    policies: &[Policy],
    windows: &Windows,
    kind: PeriodKind,
    method: EarningMethod,
    valuation: Option<Date>,
) -> Result<EarnedTable> {
    let bounds = windows.bounds();
    let n = windows.len();
    if let Some(v) = valuation
        && v <= bounds[0]
    {
        return Err(Error::Data(format!(
            "the valuation date {v} is not after the first window's start {}",
            bounds[0]
        )));
    }
    // The window that holds a date.
    let window_of = |date: Date| match bounds.partition_point(|b| *b <= date) {
        0 => None,
        k if k > n => None,
        k => Some(k - 1),
    };
    // Per label: written, earned, unearned and in force, by window.
    let mut by_label: BTreeMap<&str, Vec<[f64; 4]>> = BTreeMap::new();
    for policy in policies {
        // Not yet written at the valuation date: not in the table.
        if valuation.is_some_and(|v| policy.effective >= v) {
            continue;
        }
        let cells = by_label
            .entry(policy.label.as_str())
            .or_insert_with(|| vec![[0.0; 4]; n]);
        // What has been booked by the valuation date.
        let transactions: Vec<(Date, f64)> = policy
            .transactions(method)
            .into_iter()
            .filter(|t| valuation.is_none_or(|v| t.0 < v))
            .collect();
        match kind {
            PeriodKind::Calendar => {
                for &(date, amount) in &transactions {
                    if let Some(k) = window_of(date) {
                        cells[k][0] += amount;
                    }
                }
                for (k, cell) in cells.iter_mut().enumerate() {
                    let (from, to) = (bounds[k], bounds[k + 1]);
                    // A window cut by the valuation date ends there.
                    let end = valuation.map_or(to, |v| v.min(to));
                    if end <= from {
                        continue;
                    }
                    let (a, b) = (method.bound(from), method.bound(end));
                    cell[1] += policy.earned_in(method, a, b);
                    // Written by the window's end, less earned by then.
                    let written: f64 = transactions.iter().filter(|t| t.0 < end).map(|t| t.1).sum();
                    cell[2] += written - policy.earned_in(method, f64::NEG_INFINITY, b);
                    cell[3] += policy.in_force_at_close(method, end);
                }
            }
            PeriodKind::Policy => {
                // Everything a policy writes is booked in its own period.
                let Some(k) = window_of(policy.effective) else {
                    continue;
                };
                let cell = &mut cells[k];
                let written: f64 = transactions.iter().map(|t| t.1).sum();
                cell[0] += written;
                match valuation {
                    None => cell[1] += written,
                    Some(v) => {
                        let earned = policy.earned_in(method, f64::NEG_INFINITY, method.bound(v));
                        cell[1] += earned;
                        cell[2] += written - earned;
                        cell[3] += policy.in_force_at_close(method, v);
                    }
                }
            }
        }
    }
    let rows = by_label
        .into_iter()
        .flat_map(|(label, cells)| {
            cells.into_iter().enumerate().map(move |(k, c)| EarnedRow {
                label: label.to_string(),
                from: bounds[k],
                to: bounds[k + 1],
                written: c[0],
                earned: c[1],
                unearned: c[2],
                in_force: c[3],
            })
        })
        .collect();
    Ok(EarnedTable { kind, rows })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(year: i32, month: u8, day: u8) -> Date {
        Date::new(year, month, day).unwrap()
    }

    #[test]
    fn dates_count_days_and_round_trip() {
        assert_eq!(d(1970, 1, 1).days_since(d(1969, 12, 31)), 1);
        assert_eq!(d(2000, 3, 1).days_since(d(2000, 2, 28)), 2); // 2000 is a leap year
        assert_eq!(d(1900, 3, 1).days_since(d(1900, 2, 28)), 1); // 1900 is not
        assert_eq!(d(2025, 1, 1).days_since(d(2024, 1, 1)), 366);
        assert_eq!(d(2024, 1, 1).days_since(d(2025, 1, 1)), -366);
        // Every day of four centuries comes back as itself, one day apart.
        let mut previous = d(1899, 12, 31);
        for year in [1900, 1999, 2000, 2023, 2024, 2100] {
            for month in 1..=12 {
                for day in 1..=days_in_month(year, month) {
                    let date = d(year, month, day);
                    assert_eq!(date.ymd(), (year, month, day));
                    assert_eq!(Date::parse(&date.to_string()).unwrap(), date);
                    if (year, month, day) != (1900, 1, 1) && month == 1 && day == 1 {
                        previous = d(year - 1, 12, 31);
                    }
                    assert_eq!(date.days_since(previous), 1, "{date}");
                    previous = date;
                }
            }
        }
        assert!(d(2011, 7, 1) > d(2011, 6, 30));
    }

    #[test]
    fn bad_dates_are_refused() {
        assert!(Date::new(2023, 2, 29).is_err() && Date::new(2024, 2, 29).is_ok());
        assert!(Date::new(2023, 13, 1).is_err() && Date::new(2023, 0, 1).is_err());
        assert!(Date::new(2023, 4, 31).is_err() && Date::new(2023, 4, 0).is_err());
        assert!(Date::new(0, 1, 1).is_err() && Date::new(10_000, 1, 1).is_err());
        for bad in [
            "2023-2-29",
            "2023/01/01",
            "20230101",
            "2023-02-30",
            "2023-1x-01",
            "",
            "+023-01-01",
        ] {
            assert!(Date::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn year_fractions_are_exact() {
        assert_eq!(d(2011, 1, 1).year_fraction(), 2011.0);
        assert!((d(2011, 7, 1).year_fraction() - (2011.0 + 181.0 / 365.0)).abs() < 1e-12);
        assert!((d(2012, 12, 31).year_fraction() - (2012.0 + 365.0 / 366.0)).abs() < 1e-12);
    }

    /// Six annual policies of one exposure each, written a quarter apart
    /// from 1 October 2010: Werner and Modlin's chapter 4 (Table 4.7).
    fn six_policies() -> Vec<Policy> {
        [
            (2010, 10),
            (2011, 1),
            (2011, 4),
            (2011, 7),
            (2011, 10),
            (2012, 1),
        ]
        .into_iter()
        .map(|(year, month)| Policy::new(d(year, month, 1), d(year + 1, month, 1), 1.0).unwrap())
        .collect()
    }

    #[test]
    fn calendar_years_of_six_quarterly_policies() {
        let table = earned_exposure(
            &six_policies(),
            &Windows::years(2010, 2013).unwrap(),
            PeriodKind::Calendar,
            EarningMethod::Daily,
            None,
        )
        .unwrap();
        assert_eq!(table.kind(), PeriodKind::Calendar);
        let rows = table.rows();
        assert_eq!(rows.len(), 4);
        let written: Vec<f64> = rows.iter().map(|r| r.written).collect();
        assert_eq!(written, [1.0, 4.0, 1.0, 0.0]);
        // Exact days: 2010 has the last 92 days of the first policy's 365.
        let exact = [
            92.0 / 365.0,
            273.0 / 365.0 + 1.0 + 275.0 / 366.0 + 184.0 / 366.0 + 92.0 / 366.0,
            91.0 / 366.0 + 182.0 / 366.0 + 274.0 / 366.0 + 1.0,
            0.0,
        ];
        // Counting months instead, a quarter of a policy a quarter: a day
        // count is within a day or two in the year of that.
        let by_month = [0.25, 3.25, 2.5, 0.0];
        for (k, row) in rows.iter().enumerate() {
            assert!(
                (row.earned - exact[k]).abs() < 1e-12,
                "year {k}: {}",
                row.earned
            );
            assert!(
                (row.earned - by_month[k]).abs() < 6e-3,
                "year {k}: {}",
                row.earned
            );
        }
        // In force at the close of each year: the first policy; the four
        // written in 2011 (the one of 1 January 2011 expires the next
        // morning); the one written on 1 January 2012; none.
        let in_force: Vec<f64> = rows.iter().map(|r| r.in_force).collect();
        assert_eq!(in_force, [1.0, 4.0, 1.0, 0.0]);
        // Unearned at the end of 2010: 273 of the first policy's 365 days.
        assert!((rows[0].unearned - 273.0 / 365.0).abs() < 1e-15);
        // Everything written is earned in the end.
        let earned: f64 = rows.iter().map(|r| r.earned).sum();
        assert!((earned - 6.0).abs() < 1e-12);
        // Unearned rolls forward: opening + written − earned = closing.
        let mut unearned = 0.0;
        for row in rows {
            unearned += row.written - row.earned;
            assert!((row.unearned - unearned).abs() < 1e-12);
        }
    }

    #[test]
    fn policy_years_at_ultimate_and_at_a_valuation_date() {
        let years = Windows::years(2010, 2012).unwrap();
        let ultimate = earned_exposure(
            &six_policies(),
            &years,
            PeriodKind::Policy,
            EarningMethod::Daily,
            None,
        )
        .unwrap();
        for (row, written) in ultimate.rows().iter().zip([1.0, 4.0, 1.0]) {
            assert_eq!((row.written, row.earned), (written, written));
            assert_eq!((row.unearned, row.in_force), (0.0, 0.0));
        }
        // As of 1 January 2012: policy year 2011's four policies have
        // earned 365, 275, 184 and 92 days.
        let at = earned_exposure(
            &six_policies(),
            &years,
            PeriodKind::Policy,
            EarningMethod::Daily,
            Some(d(2012, 1, 1)),
        )
        .unwrap();
        let rows = at.rows();
        assert_eq!(
            (rows[0].earned, rows[0].unearned, rows[0].in_force),
            (1.0, 0.0, 0.0)
        );
        let earned_2011 = 1.0 + 275.0 / 366.0 + 184.0 / 366.0 + 92.0 / 366.0;
        assert!((rows[1].earned - earned_2011).abs() < 1e-12);
        assert!((rows[1].earned + rows[1].unearned - 4.0).abs() < 1e-12);
        // The policy of 1 January 2011 expires on the valuation date and
        // is still in force at the close of the day before.
        assert_eq!(rows[1].in_force, 4.0);
        // The policy written on the valuation date is not yet in the table.
        assert_eq!((rows[2].written, rows[2].earned), (0.0, 0.0));
    }

    #[test]
    fn a_valuation_date_cuts_a_calendar_table_off() {
        let rows_at = |valuation| {
            earned_exposure(
                &six_policies(),
                &Windows::years(2010, 2012).unwrap(),
                PeriodKind::Calendar,
                EarningMethod::Daily,
                valuation,
            )
            .unwrap()
            .rows()
            .to_vec()
        };
        let (full, cut) = (rows_at(None), rows_at(Some(d(2011, 7, 1))));
        assert_eq!(cut[0], full[0]);
        // Half of 2011: two policies written, the one of 1 July itself not.
        assert_eq!(cut[1].written, 2.0);
        let half = 181.0 / 365.0 + 181.0 / 365.0 + 91.0 / 366.0;
        assert!((cut[1].earned - half).abs() < 1e-12);
        assert_eq!(cut[1].in_force, 3.0);
        assert_eq!(
            (cut[2].written, cut[2].earned, cut[2].in_force),
            (0.0, 0.0, 0.0)
        );
        assert!(
            earned_exposure(
                &six_policies(),
                &Windows::years(2010, 2012).unwrap(),
                PeriodKind::Calendar,
                EarningMethod::Daily,
                Some(d(2010, 1, 1)),
            )
            .is_err()
        );
    }

    #[test]
    fn labels_split_the_table_and_short_terms_earn_on_their_own_days() {
        // A six-month policy of 600 and a two-year policy of 2,400.
        let policies = [
            Policy::new(d(2024, 3, 1), d(2024, 9, 1), 600.0)
                .unwrap()
                .with_label("b"),
            Policy::new(d(2023, 1, 1), d(2025, 1, 1), 2_400.0)
                .unwrap()
                .with_label("a"),
        ];
        let table = earned_exposure(
            &policies,
            &Windows::quarters(2024, 2024).unwrap(),
            PeriodKind::Calendar,
            EarningMethod::Daily,
            None,
        )
        .unwrap();
        assert_eq!(table.rows().len(), 8);
        let (a, b) = (table.label("a"), table.label("b"));
        assert_eq!(
            (a.len(), b.len(), table.rows()[0].label.as_str()),
            (4, 4, "a")
        );
        // The two-year policy: 731 days, written before the first window.
        assert!(a.iter().all(|r| r.written == 0.0));
        assert!((a[0].earned - 2_400.0 * 91.0 / 731.0).abs() < 1e-9);
        // It expires on 1 January 2025, so it is in force through the year.
        assert_eq!((a[2].in_force, a[3].in_force), (2_400.0, 2_400.0));
        // The six-month policy: 184 days, 31 of them in the first quarter.
        assert_eq!(b[0].written, 600.0);
        assert!((b[0].earned - 600.0 * 31.0 / 184.0).abs() < 1e-9);
        assert!((b[1].earned - 600.0 * 91.0 / 184.0).abs() < 1e-9);
        assert!((b[2].earned - 600.0 * 62.0 / 184.0).abs() < 1e-9);
        assert_eq!(
            (b[3].earned, b[2].in_force, b[1].in_force),
            (0.0, 0.0, 600.0)
        );
        assert!(table.label("c").is_empty());
    }

    #[test]
    fn policies_outside_the_windows() {
        // Written before the windows: earned in them, written in none.
        // Written after them: nothing.
        let policies = [
            Policy::new(d(2019, 7, 1), d(2020, 7, 1), 1.0).unwrap(),
            Policy::new(d(2022, 1, 1), d(2023, 1, 1), 1.0).unwrap(),
        ];
        let years = Windows::years(2020, 2021).unwrap();
        let calendar = earned_exposure(
            &policies,
            &years,
            PeriodKind::Calendar,
            EarningMethod::Daily,
            None,
        )
        .unwrap();
        let rows = calendar.rows();
        assert_eq!((rows[0].written, rows[1].written), (0.0, 0.0));
        assert!((rows[0].earned - 182.0 / 366.0).abs() < 1e-15 && rows[1].earned == 0.0);
        let policy = earned_exposure(
            &policies,
            &years,
            PeriodKind::Policy,
            EarningMethod::Daily,
            None,
        )
        .unwrap();
        assert!(
            policy
                .rows()
                .iter()
                .all(|r| r.written == 0.0 && r.earned == 0.0)
        );
        // No policies: no rows.
        assert!(
            earned_exposure(&[], &years, PeriodKind::Policy, EarningMethod::Daily, None)
                .unwrap()
                .rows()
                .is_empty()
        );
    }

    #[test]
    fn bad_policies_and_windows_are_refused() {
        assert!(Policy::new(d(2024, 1, 1), d(2024, 1, 1), 1.0).is_err());
        assert!(Policy::new(d(2024, 1, 2), d(2024, 1, 1), 1.0).is_err());
        assert!(Policy::new(d(2024, 1, 1), d(2025, 1, 1), -1.0).is_err());
        assert!(Policy::new(d(2024, 1, 1), d(2025, 1, 1), f64::NAN).is_err());
        let p = Policy::new(d(2024, 1, 1), d(2024, 1, 2), 5.0).unwrap();
        assert_eq!((p.term_days(), p.amount(), p.label()), (1, 5.0, ""));
        assert_eq!(
            (p.effective(), p.expiration()),
            (d(2024, 1, 1), d(2024, 1, 2))
        );
        assert!(Windows::new(vec![d(2024, 1, 1)]).is_err());
        assert!(Windows::new(vec![d(2024, 1, 1), d(2024, 1, 1)]).is_err());
        assert!(Windows::years(2024, 2023).is_err());
        let months = Windows::months(2023, 2024).unwrap();
        assert_eq!((months.len(), months.is_empty()), (24, false));
        assert_eq!(months.bounds()[24], d(2025, 1, 1));
    }
}
