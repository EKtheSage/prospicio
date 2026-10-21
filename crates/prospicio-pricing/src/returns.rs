//! Return metrics on simulated cash flows: the net present value at an
//! annual rate, the internal rate of return found on a grid with a flag
//! for whether a root was found, the multiple on invested capital, the
//! leverage of premium on capital and a fee tiered on the margin. See
//! `docs/design/time-value.md`, decisions 8 to 12.
//!
//! Every function takes one simulation's flows at times in years from the
//! start, discounted by `(1 + r)^-t` as in [`payout`](crate::payout);
//! [`TimeUnit`] turns month or day offsets into years. The `_rows`
//! functions take a row-major matrix, one simulation per row. The metrics
//! are plain arithmetic on the flows and do not depend on how the flows
//! were built.

use prospicio_core::{Error, Result};

/// The unit of the time offsets of a cash flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeUnit {
    /// Years from the start.
    #[default]
    Years,
    /// Months from the start, a month a twelfth of a year.
    Months,
    /// Days from the start, actual/365.
    Days,
}

impl TimeUnit {
    /// The offsets in years.
    ///
    /// ```
    /// use prospicio_pricing::returns::TimeUnit;
    ///
    /// assert_eq!(TimeUnit::Months.years(&[0.0, 6.0, 18.0]), [0.0, 0.5, 1.5]);
    /// assert_eq!(TimeUnit::Days.years(&[365.0]), [1.0]);
    /// ```
    pub fn years(self, offsets: &[f64]) -> Vec<f64> {
        let per_year = match self {
            TimeUnit::Years => 1.0,
            TimeUnit::Months => 12.0,
            TimeUnit::Days => 365.0,
        };
        offsets.iter().map(|t| t / per_year).collect()
    }
}

fn check_times(times: &[f64]) -> Result<()> {
    if times.is_empty() {
        return Err(Error::Data("a cash flow needs at least one time".into()));
    }
    if let Some(&t) = times.iter().find(|t| !t.is_finite()) {
        return Err(Error::InvalidParameter {
            name: "times",
            value: t,
            reason: "must be finite",
        });
    }
    Ok(())
}

fn check_flows(flows: &[f64], times: &[f64]) -> Result<()> {
    if flows.len() != times.len() {
        return Err(Error::Data(format!(
            "{} flows at {} times; one flow a time",
            flows.len(),
            times.len()
        )));
    }
    if let Some(&x) = flows.iter().find(|x| !x.is_finite()) {
        return Err(Error::InvalidParameter {
            name: "flows",
            value: x,
            reason: "must be finite",
        });
    }
    Ok(())
}

fn value(flows: &[f64], times: &[f64], rate: f64) -> f64 {
    let base = 1.0 + rate;
    flows
        .iter()
        .zip(times)
        .map(|(x, t)| x * base.powf(-t))
        .sum()
}

fn slope(flows: &[f64], times: &[f64], rate: f64) -> f64 {
    let base = 1.0 + rate;
    flows
        .iter()
        .zip(times)
        .map(|(x, t)| -t * x * base.powf(-t - 1.0))
        .sum()
}

/// The net present value `Σ x_k (1 + rate)^-t_k` of flows `x_k` at times
/// `t_k` in years.
///
/// ```
/// use prospicio_pricing::returns::npv;
///
/// let v = npv(&[-100.0, 121.0], &[0.0, 2.0], 0.1).unwrap();
/// assert!(v.abs() < 1e-12);
/// ```
pub fn npv(flows: &[f64], times: &[f64], rate: f64) -> Result<f64> {
    check_times(times)?;
    check_flows(flows, times)?;
    if !(rate.is_finite() && rate > -1.0) {
        return Err(Error::InvalidParameter {
            name: "rate",
            value: rate,
            reason: "must be finite and above -1",
        });
    }
    Ok(value(flows, times, rate))
}

/// [`npv`] of each row of a row-major matrix of flows, one simulation a
/// row and one column a time.
pub fn npv_rows(flows: &[f64], times: &[f64], rate: f64) -> Result<Vec<f64>> {
    check_times(times)?;
    if flows.len() % times.len() != 0 {
        return Err(Error::Data(format!(
            "{} flows do not fill rows of {} times",
            flows.len(),
            times.len()
        )));
    }
    flows
        .chunks(times.len())
        .map(|row| npv(row, times, rate))
        .collect()
}

/// How [`irr`] searches for a root: the value of the flows on a grid of
/// `points` annual rates from `lower` to `upper`, the sign change whose
/// bracket's midpoint is nearest 0, then `bisections` halvings of that
/// bracket and `newton_steps` Newton steps clipped to it. The defaults are
/// the reference simulation model's: 400 points from −0.95 to 1, 8
/// bisections and 3 Newton steps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IrrOptions {
    /// The lowest rate searched, above −1.
    pub lower: f64,
    /// The highest rate searched.
    pub upper: f64,
    /// The number of grid points, at least 2.
    pub points: usize,
    /// Bisections of the chosen bracket.
    pub bisections: usize,
    /// Newton steps after them, each clipped to the bracket.
    pub newton_steps: usize,
}

impl Default for IrrOptions {
    fn default() -> Self {
        IrrOptions {
            lower: -0.95,
            upper: 1.0,
            points: 400,
            bisections: 8,
            newton_steps: 3,
        }
    }
}

impl IrrOptions {
    fn check(&self) -> Result<()> {
        if !(self.lower.is_finite() && self.lower > -1.0) {
            return Err(Error::InvalidParameter {
                name: "lower",
                value: self.lower,
                reason: "must be finite and above -1",
            });
        }
        if !(self.upper.is_finite() && self.upper > self.lower) {
            return Err(Error::InvalidParameter {
                name: "upper",
                value: self.upper,
                reason: "must be finite and above lower",
            });
        }
        if self.points < 2 {
            return Err(Error::InvalidParameter {
                name: "points",
                value: self.points as f64,
                reason: "must be at least 2",
            });
        }
        Ok(())
    }

    fn grid(&self) -> Vec<f64> {
        let step = (self.upper - self.lower) / (self.points - 1) as f64;
        (0..self.points)
            .map(|k| {
                if k == self.points - 1 {
                    self.upper
                } else {
                    self.lower + step * k as f64
                }
            })
            .collect()
    }
}

/// An internal rate of return found by [`irr`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Irr {
    /// The annual rate, NaN when no root was found.
    pub rate: f64,
    /// Whether the grid had a sign change (or an exact zero).
    pub found: bool,
    /// The net present value at `rate`, NaN when no root was found.
    pub npv: f64,
    /// The roots seen on the grid: its sign changes and exact zeros. More
    /// than one means the flows have several rates of return, and `rate`
    /// is the one nearest 0.
    pub sign_changes: usize,
}

fn irr_checked(flows: &[f64], times: &[f64], options: &IrrOptions, grid: &[f64]) -> Irr {
    let values: Vec<f64> = grid.iter().map(|&g| value(flows, times, g)).collect();
    // Each root on the grid as a bracket: an exact zero is a bracket of
    // width zero, a sign change the interval between its two points.
    let mut brackets: Vec<(f64, f64)> = Vec::new();
    for k in 0..grid.len() {
        if values[k] == 0.0 {
            brackets.push((grid[k], grid[k]));
        } else if k + 1 < grid.len() && values[k] * values[k + 1] < 0.0 {
            brackets.push((grid[k], grid[k + 1]));
        }
    }
    let sign_changes = brackets.len();
    let Some(&(mut lo, mut hi)) = brackets.iter().min_by(|a, b| {
        (0.5 * (a.0 + a.1))
            .abs()
            .total_cmp(&(0.5 * (b.0 + b.1)).abs())
    }) else {
        return Irr {
            rate: f64::NAN,
            found: false,
            npv: f64::NAN,
            sign_changes: 0,
        };
    };
    if lo < hi {
        let mut v_lo = value(flows, times, lo);
        for _ in 0..options.bisections {
            let mid = 0.5 * (lo + hi);
            let v_mid = value(flows, times, mid);
            if v_lo * v_mid <= 0.0 {
                hi = mid;
            } else {
                lo = mid;
                v_lo = v_mid;
            }
        }
    }
    let mut rate = 0.5 * (lo + hi);
    for _ in 0..options.newton_steps {
        let d = slope(flows, times, rate);
        if d == 0.0 || !d.is_finite() {
            break;
        }
        rate = (rate - value(flows, times, rate) / d).clamp(lo, hi);
    }
    Irr {
        rate,
        found: true,
        npv: value(flows, times, rate),
        sign_changes,
    }
}

/// The internal rate of return of flows at times in years: the annual
/// rate at which their [`npv`] is 0, found as [`IrrOptions`] says. Flows
/// with no root in the searched range, such as flows of one sign, give
/// `found = false` and NaN; flows with several give the one nearest 0 and
/// say how many the grid saw.
///
/// ```
/// use prospicio_pricing::returns::{IrrOptions, irr};
///
/// // −100 now, +230 in a year and −132 in two: roots at 10% and 20%.
/// let r = irr(&[-100.0, 230.0, -132.0], &[0.0, 1.0, 2.0], &IrrOptions::default()).unwrap();
/// assert!(r.found && (r.rate - 0.1).abs() < 1e-12);
/// assert_eq!(r.sign_changes, 2);
/// ```
pub fn irr(flows: &[f64], times: &[f64], options: &IrrOptions) -> Result<Irr> {
    check_times(times)?;
    check_flows(flows, times)?;
    options.check()?;
    Ok(irr_checked(flows, times, options, &options.grid()))
}

/// [`irr`] of each row of a row-major matrix of flows, one simulation a
/// row and one column a time.
pub fn irr_rows(flows: &[f64], times: &[f64], options: &IrrOptions) -> Result<Vec<Irr>> {
    check_times(times)?;
    options.check()?;
    if flows.len() % times.len() != 0 {
        return Err(Error::Data(format!(
            "{} flows do not fill rows of {} times",
            flows.len(),
            times.len()
        )));
    }
    let grid = options.grid();
    flows
        .chunks(times.len())
        .map(|row| {
            check_flows(row, times)?;
            Ok(irr_checked(row, times, options, &grid))
        })
        .collect()
}

fn invested(funded: f64, opening: f64) -> Result<f64> {
    let c = funded + opening;
    if !(c.is_finite() && c > 0.0) {
        return Err(Error::InvalidParameter {
            name: "capital",
            value: c,
            reason: "capital funded plus opening capital must be finite and positive",
        });
    }
    Ok(c)
}

/// The multiple on invested capital, `1 + released / (funded + opening)`:
/// the equity released over the capital funded during the term and the
/// capital held at its start.
///
/// ```
/// use prospicio_pricing::returns::moic;
///
/// assert_eq!(moic(30.0, 40.0, 60.0).unwrap(), 1.3);
/// ```
pub fn moic(released: f64, funded: f64, opening: f64) -> Result<f64> {
    Ok(1.0 + released / invested(funded, opening)?)
}

/// The leverage of written premium on capital, `rate × premium / (funded +
/// opening)`. The rate is the caller's; there is no default.
///
/// ```
/// use prospicio_pricing::returns::leverage;
///
/// assert_eq!(leverage(0.5, 300.0, 50.0, 100.0).unwrap(), 1.0);
/// ```
pub fn leverage(rate: f64, premium: f64, funded: f64, opening: f64) -> Result<f64> {
    if !rate.is_finite() {
        return Err(Error::InvalidParameter {
            name: "rate",
            value: rate,
            reason: "must be finite",
        });
    }
    Ok(rate * premium / invested(funded, opening)?)
}

/// A fee tiered on the margin: slope `s_i` on the part of the margin
/// between breakpoints `b_i` and `b_{i+1}`, the last tier unbounded,
/// `fee(m) = Σ_i s_i (clip(m, b_i, b_{i+1}) − b_i)`. Nothing is due below
/// the first breakpoint, the floor. The tiers are the caller's; there are
/// no defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct TieredFee {
    breakpoints: Vec<f64>,
    slopes: Vec<f64>,
}

impl TieredFee {
    /// From strictly increasing finite breakpoints and one finite slope
    /// each, the slope of the tier that starts there.
    ///
    /// ```
    /// use prospicio_pricing::returns::TieredFee;
    ///
    /// // 10% of the margin from 5 to 15, 20% above 15.
    /// let fee = TieredFee::new(vec![5.0, 15.0], vec![0.1, 0.2]).unwrap();
    /// assert_eq!(fee.fee(3.0), 0.0);
    /// assert!((fee.fee(10.0) - 0.5).abs() < 1e-15);
    /// assert!((fee.fee(25.0) - 3.0).abs() < 1e-15);
    /// ```
    pub fn new(breakpoints: Vec<f64>, slopes: Vec<f64>) -> Result<Self> {
        if breakpoints.is_empty() || breakpoints.len() != slopes.len() {
            return Err(Error::Data(format!(
                "{} breakpoints and {} slopes; a fee needs at least one tier, one slope a breakpoint",
                breakpoints.len(),
                slopes.len()
            )));
        }
        if breakpoints.iter().chain(&slopes).any(|x| !x.is_finite())
            || breakpoints.windows(2).any(|w| w[1] <= w[0])
        {
            return Err(Error::Data(
                "breakpoints and slopes must be finite, and breakpoints strictly increasing".into(),
            ));
        }
        Ok(TieredFee {
            breakpoints,
            slopes,
        })
    }

    /// The breakpoints, the first the floor.
    pub fn breakpoints(&self) -> &[f64] {
        &self.breakpoints
    }

    /// Each tier's slope.
    pub fn slopes(&self) -> &[f64] {
        &self.slopes
    }

    /// The fee on margin `m`.
    pub fn fee(&self, m: f64) -> f64 {
        let n = self.breakpoints.len();
        (0..n)
            .map(|i| {
                let lo = self.breakpoints[i];
                let hi = if i + 1 < n {
                    self.breakpoints[i + 1]
                } else {
                    f64::INFINITY
                };
                self.slopes[i] * (m.clamp(lo, hi) - lo)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_root_in_closed_form() {
        let r = irr(&[-100.0, 121.0], &[0.0, 2.0], &IrrOptions::default()).unwrap();
        assert!(r.found);
        assert!((r.rate - 0.1).abs() < 1e-12, "{}", r.rate);
        assert!(r.npv.abs() < 1e-10);
        assert_eq!(r.sign_changes, 1);
    }

    #[test]
    fn of_two_roots_the_one_nearest_zero() {
        // −100 + 230 v − 132 v² = 0 at v = 1/1.1 and 1/1.2.
        let flows = [-100.0, 230.0, -132.0];
        let r = irr(&flows, &[0.0, 1.0, 2.0], &IrrOptions::default()).unwrap();
        assert!((r.rate - 0.1).abs() < 1e-12);
        assert_eq!(r.sign_changes, 2);
        // Searched from 15% only the other is seen.
        let above = IrrOptions {
            lower: 0.15,
            ..IrrOptions::default()
        };
        let r = irr(&flows, &[0.0, 1.0, 2.0], &above).unwrap();
        assert!((r.rate - 0.2).abs() < 1e-12);
        assert_eq!(r.sign_changes, 1);
    }

    #[test]
    fn no_root_is_flagged() {
        let r = irr(&[10.0, 5.0], &[0.0, 1.0], &IrrOptions::default()).unwrap();
        assert!(!r.found && r.rate.is_nan() && r.npv.is_nan());
        assert_eq!(r.sign_changes, 0);
        // −100 now and 400 in a year: a root at 300%, outside [−0.95, 1].
        let r = irr(&[-100.0, 400.0], &[0.0, 1.0], &IrrOptions::default()).unwrap();
        assert!(!r.found);
        let wide = IrrOptions {
            upper: 4.0,
            ..IrrOptions::default()
        };
        let r = irr(&[-100.0, 400.0], &[0.0, 1.0], &wide).unwrap();
        assert!((r.rate - 3.0).abs() < 1e-12);
    }

    #[test]
    fn an_exact_zero_on_the_grid_is_a_root() {
        // At 0 the flows sum to 0, and 0 is a grid point of −1/2 to 1/2.
        let options = IrrOptions {
            lower: -0.5,
            upper: 0.5,
            points: 11,
            ..IrrOptions::default()
        };
        let r = irr(&[-100.0, 100.0], &[0.0, 1.0], &options).unwrap();
        assert!(r.found);
        assert_eq!(r.rate, 0.0);
    }

    #[test]
    fn rows_match_one_at_a_time() {
        let times = [0.0, 0.5, 1.0];
        let flows = [-100.0, 20.0, 95.0, 5.0, 5.0, 5.0, -50.0, 0.0, 60.0];
        let options = IrrOptions::default();
        let rows = irr_rows(&flows, &times, &options).unwrap();
        for (k, row) in flows.chunks(3).enumerate() {
            let one = irr(row, &times, &options).unwrap();
            assert_eq!(rows[k].found, one.found);
            assert!(rows[k].rate == one.rate || (rows[k].rate.is_nan() && one.rate.is_nan()));
        }
        assert!(!rows[1].found);
        let v = npv_rows(&flows, &times, 0.05).unwrap();
        assert!((v[2] - (-50.0 + 60.0 / 1.05)).abs() < 1e-12);
        assert!(irr_rows(&flows[..8], &times, &options).is_err());
    }

    #[test]
    fn the_day_count_moves_the_rate() {
        let flows = [-100.0, 0.0, 0.0, 103.0];
        let months = TimeUnit::Months.years(&[0.0, 1.0, 2.0, 3.0]);
        let days = TimeUnit::Days.years(&[0.0, 31.0, 59.0, 90.0]);
        let options = IrrOptions::default();
        let by_month = irr(&flows, &months, &options).unwrap().rate;
        let by_day = irr(&flows, &days, &options).unwrap().rate;
        assert!((by_month - (1.03f64.powf(4.0) - 1.0)).abs() < 1e-12);
        assert!((by_day - (1.03f64.powf(365.0 / 90.0) - 1.0)).abs() < 1e-12);
        assert!((by_month - by_day).abs() > 1e-4);
    }

    #[test]
    fn inputs_are_checked() {
        let o = IrrOptions::default();
        assert!(irr(&[1.0], &[0.0, 1.0], &o).is_err());
        assert!(irr(&[f64::NAN], &[0.0], &o).is_err());
        assert!(npv(&[1.0], &[0.0], -1.0).is_err());
        let bad = IrrOptions {
            lower: -1.0,
            ..IrrOptions::default()
        };
        assert!(irr(&[1.0], &[0.0], &bad).is_err());
        assert!(moic(1.0, 0.0, 0.0).is_err());
        assert!(TieredFee::new(vec![1.0, 1.0], vec![0.1, 0.2]).is_err());
        assert!(TieredFee::new(vec![1.0], vec![]).is_err());
    }

    #[test]
    fn three_tiers_by_hand() {
        let fee = TieredFee::new(vec![0.0, 10.0, 30.0], vec![0.05, 0.1, 0.25]).unwrap();
        assert_eq!(fee.fee(-5.0), 0.0);
        assert!((fee.fee(10.0) - 0.5).abs() < 1e-15);
        assert!((fee.fee(20.0) - 1.5).abs() < 1e-15);
        assert!((fee.fee(50.0) - (0.5 + 2.0 + 5.0)).abs() < 1e-15);
    }
}
