//! Calendar months, period grains and origin periods.
//!
//! Development ages are whole months measured from the start of the origin
//! period, so a triangle only ever needs month resolution: an age of 12
//! months on an origin starting January 2021 is valued at the end of
//! December 2021.

use std::fmt;

use crate::error::{Error, Result};

/// Length of an origin or development period.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Grain {
    /// One month (`M`).
    Month,
    /// Three months (`Q`).
    Quarter,
    /// Six months (`S`).
    Semester,
    /// Twelve months (`Y`).
    Year,
}

impl Grain {
    /// Length of the period in months.
    pub const fn months(self) -> u32 {
        match self {
            Self::Month => 1,
            Self::Quarter => 3,
            Self::Semester => 6,
            Self::Year => 12,
        }
    }

    /// Whether every period of `self` is a whole number of `finer` periods.
    pub const fn is_multiple_of(self, finer: Grain) -> bool {
        self.months() % finer.months() == 0
    }
}

impl fmt::Display for Grain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Month => "M",
            Self::Quarter => "Q",
            Self::Semester => "S",
            Self::Year => "Y",
        })
    }
}

/// A calendar month. As a valuation date it means the end of that month.
///
/// ```
/// use act_reserving::Month;
///
/// let m = Month::new(2021, 11).unwrap();
/// assert_eq!(m.add_months(3), Month::new(2022, 2).unwrap());
/// assert_eq!(m.to_string(), "2021-11");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Month {
    // Field order gives the chronological `Ord`.
    year: i32,
    month: u8,
}

impl Month {
    /// The month `month` (1 to 12) of `year`.
    pub fn new(year: i32, month: u8) -> Result<Self> {
        if !(1..=12).contains(&month) {
            return Err(Error::InvalidMonth(month));
        }
        Ok(Self { year, month })
    }

    /// January of `year`.
    pub const fn january(year: i32) -> Self {
        Self { year, month: 1 }
    }

    /// Calendar year.
    pub const fn year(self) -> i32 {
        self.year
    }

    /// Month of the year, 1 to 12.
    pub const fn month(self) -> u8 {
        self.month
    }

    /// Months since January of year 0, so month arithmetic is integer
    /// arithmetic.
    const fn ordinal(self) -> i64 {
        self.year as i64 * 12 + (self.month as i64 - 1)
    }

    fn from_ordinal(ordinal: i64) -> Self {
        Self {
            year: ordinal.div_euclid(12) as i32,
            month: (ordinal.rem_euclid(12) + 1) as u8,
        }
    }

    /// The month `n` months later (earlier if `n` is negative).
    pub fn add_months(self, n: i64) -> Self {
        Self::from_ordinal(self.ordinal() + n)
    }

    /// Whole months from `earlier` to `self` (negative if `self` is earlier).
    pub const fn months_since(self, earlier: Month) -> i64 {
        self.ordinal() - earlier.ordinal()
    }

    /// First month of the `grain` period containing `self`. Periods are
    /// aligned to the calendar year: quarters start in January, April, July
    /// and October.
    pub fn floor(self, grain: Grain) -> Self {
        let size = grain.months() as u8;
        Self {
            year: self.year,
            month: (self.month - 1) / size * size + 1,
        }
    }
}

impl fmt::Display for Month {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{:02}", self.year, self.month)
    }
}

/// An origin period: a start month and a grain.
///
/// Periods compare equal across triangles and reserve distributions, so a
/// reserve component keyed by origin joins back to the triangle directly.
///
/// ```
/// use act_reserving::{Grain, Month, Period};
///
/// let p = Period::containing(Month::new(2021, 8).unwrap(), Grain::Quarter);
/// assert_eq!(p.start(), Month::new(2021, 7).unwrap());
/// assert_eq!(p.end(), Month::new(2021, 9).unwrap());
/// assert_eq!(p.to_string(), "2021Q3");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Period {
    start: Month,
    grain: Grain,
}

impl Period {
    /// The `grain` period that contains `month`.
    pub fn containing(month: Month, grain: Grain) -> Self {
        Self {
            start: month.floor(grain),
            grain,
        }
    }

    /// The calendar year `year`.
    pub const fn year(year: i32) -> Self {
        Self {
            start: Month::january(year),
            grain: Grain::Year,
        }
    }

    /// First month of the period.
    pub const fn start(self) -> Month {
        self.start
    }

    /// Last month of the period.
    pub fn end(self) -> Month {
        self.start.add_months(self.grain.months() as i64 - 1)
    }

    /// Length of the period.
    pub const fn grain(self) -> Grain {
        self.grain
    }
}

impl fmt::Display for Period {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Month { year, month } = self.start;
        match self.grain {
            Grain::Year => write!(f, "{year}"),
            Grain::Semester => write!(f, "{year}H{}", (month - 1) / 6 + 1),
            Grain::Quarter => write!(f, "{year}Q{}", (month - 1) / 3 + 1),
            Grain::Month => write!(f, "{}", self.start),
        }
    }
}

/// Development age: whole months from the start of the origin period to the
/// end of the valuation month. Age 12 on an annual origin is its first
/// year-end.
pub type Lag = u32;

#[cfg(test)]
mod tests {
    use super::*;

    fn m(year: i32, month: u8) -> Month {
        Month::new(year, month).unwrap()
    }

    #[test]
    fn month_arithmetic_crosses_years() {
        assert_eq!(m(2020, 1).add_months(-1), m(2019, 12));
        assert_eq!(m(2020, 12).add_months(13), m(2022, 1));
        assert_eq!(m(2022, 1).months_since(m(2020, 12)), 13);
        assert_eq!(m(-1, 12).add_months(1), m(0, 1));
    }

    #[test]
    fn rejects_bad_month() {
        assert_eq!(Month::new(2020, 0), Err(Error::InvalidMonth(0)));
        assert_eq!(Month::new(2020, 13), Err(Error::InvalidMonth(13)));
    }

    #[test]
    fn floor_aligns_to_calendar() {
        assert_eq!(m(2021, 12).floor(Grain::Quarter), m(2021, 10));
        assert_eq!(m(2021, 6).floor(Grain::Semester), m(2021, 1));
        assert_eq!(m(2021, 7).floor(Grain::Semester), m(2021, 7));
        assert_eq!(m(2021, 7).floor(Grain::Year), m(2021, 1));
        assert_eq!(m(2021, 7).floor(Grain::Month), m(2021, 7));
    }

    #[test]
    fn period_labels() {
        assert_eq!(Period::year(1981).to_string(), "1981");
        assert_eq!(
            Period::containing(m(2021, 7), Grain::Semester).to_string(),
            "2021H2"
        );
        assert_eq!(
            Period::containing(m(2021, 3), Grain::Month).to_string(),
            "2021-03"
        );
        assert_eq!(Period::year(1981).end(), m(1981, 12));
    }

    #[test]
    fn grain_multiples() {
        assert!(Grain::Year.is_multiple_of(Grain::Quarter));
        assert!(Grain::Year.is_multiple_of(Grain::Semester));
        assert!(!Grain::Quarter.is_multiple_of(Grain::Semester));
    }
}
