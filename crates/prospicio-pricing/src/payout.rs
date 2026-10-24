//! The time value of money in a price: when a cover's losses are paid, the
//! expected loss discounted to inception, and capital held while the losses
//! are outstanding, so the cost of capital accrues over the payout rather
//! than for one year. See `docs/design/time-value.md`.
//!
//! A [`PaymentPattern`] gives the share of the ultimate loss paid at each
//! time, in years from the treaty's inception. With a pattern that does not
//! depend on the size of the loss, every simulated year's present value is
//! its loss times [`PaymentPattern::discount_factor`], so a loss
//! distribution is discounted by scaling it.
//!
//! [`price_over_payout`] restates a one-period [`Price`] under the pattern:
//! with a constant cost of capital `r`, capital `C = a − P` is held in year
//! `k` in proportion to the share outstanding at its start, and its cost
//! `r C outstanding(k)` is charged at the start of the year, discounted,
//! as the one-period price charges `r C` at inception. The premium then
//! solves `P = E[X] d + r (a − P) A`, with `d` the discount factor and `A`
//! the discounted capital years ([`PaymentPattern::capital_years`]). A
//! pattern that pays everything within the first year gives `A = 1`, and
//! at a zero rate `d = 1` too: the one-period price.

use prospicio_core::{Error, Result};

use crate::risk_load::{PremiumRule, Price};

/// How a [`PaymentPattern`]'s cumulative share paid runs between its
/// times ([`PaymentPattern::paid_by`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// Each share is paid at its time, as the pattern discounts.
    Step,
    /// Each share is paid evenly over the period that ends at its time,
    /// from the previous time (or 0): the usual assumption.
    #[default]
    Linear,
    /// The share unpaid falls geometrically over each period; linear in a
    /// period that ends with nothing unpaid.
    LogLinear,
}

/// The share of the ultimate loss paid at each time, in years from the
/// treaty's inception.
#[derive(Debug, Clone, PartialEq)]
pub struct PaymentPattern {
    times: Vec<f64>,
    shares: Vec<f64>,
}

impl PaymentPattern {
    /// From non-negative, strictly increasing times and non-negative shares
    /// that sum to 1 (to within 1e-9; they are rescaled to sum to 1).
    ///
    /// ```
    /// use prospicio_pricing::payout::PaymentPattern;
    ///
    /// let p = PaymentPattern::new(vec![1.0, 2.0], vec![0.5, 0.5]).unwrap();
    /// // Half paid after one year and half after two, at 10%.
    /// let d = 0.5 / 1.1 + 0.5 / 1.21;
    /// assert!((p.discount_factor(0.1) - d).abs() < 1e-15);
    /// assert_eq!(p.duration(), 1.5);
    /// ```
    pub fn new(times: Vec<f64>, shares: Vec<f64>) -> Result<Self> {
        if times.is_empty() || times.len() != shares.len() {
            return Err(Error::Data(format!(
                "{} times and {} shares; a pattern needs at least one of each, as many times as shares",
                times.len(),
                shares.len()
            )));
        }
        if times.iter().any(|t| !(t.is_finite() && *t >= 0.0))
            || times.windows(2).any(|w| w[1] <= w[0])
        {
            return Err(Error::Data(
                "times must be finite, non-negative and strictly increasing".into(),
            ));
        }
        if let Some(&bad) = shares.iter().find(|s| !(s.is_finite() && **s >= 0.0)) {
            return Err(Error::InvalidParameter {
                name: "shares",
                value: bad,
                reason: "must be finite and non-negative",
            });
        }
        let total: f64 = shares.iter().sum();
        if (total - 1.0).abs() > 1e-9 {
            return Err(Error::InvalidParameter {
                name: "shares",
                value: total,
                reason: "must sum to 1",
            });
        }
        let shares = shares.iter().map(|s| s / total).collect();
        Ok(Self { times, shares })
    }

    /// From development factors to ultimate by age, such as a paid chain
    /// ladder's `cdf`: the share paid by age `ages[k]` is
    /// `1 / factors[k]`, and the shares paid are its increments, the first
    /// at the first age. The last factor must be 1 (include any tail as a
    /// last age); ages are read as times from inception, so shift the
    /// pattern ([`shifted`](Self::shifted)) when the losses do not start
    /// developing at inception.
    ///
    /// ```
    /// use prospicio_pricing::payout::PaymentPattern;
    ///
    /// let p = PaymentPattern::from_development(&[1.0, 2.0, 3.0], &[4.0, 1.25, 1.0]).unwrap();
    /// for (got, want) in p.shares().iter().zip([0.25, 0.55, 0.2]) {
    ///     assert!((got - want).abs() < 1e-15);
    /// }
    /// ```
    pub fn from_development(ages: &[f64], factors: &[f64]) -> Result<Self> {
        if ages.len() != factors.len() || ages.is_empty() {
            return Err(Error::Data(format!(
                "{} ages and {} factors",
                ages.len(),
                factors.len()
            )));
        }
        if let Some(&bad) = factors.iter().find(|f| !(f.is_finite() && **f >= 1.0)) {
            return Err(Error::InvalidParameter {
                name: "factors",
                value: bad,
                reason: "factors to ultimate must be finite and at least 1",
            });
        }
        let last = factors[factors.len() - 1];
        if (last - 1.0).abs() > 1e-12 {
            return Err(Error::InvalidParameter {
                name: "factors",
                value: last,
                reason: "the last factor must be 1; add the tail as a last age",
            });
        }
        if factors.windows(2).any(|w| w[1] > w[0]) {
            return Err(Error::Data(
                "factors to ultimate must not increase with age".into(),
            ));
        }
        let mut paid = 0.0;
        let shares = factors
            .iter()
            .map(|f| {
                let cumulative = 1.0 / f;
                let share = cumulative - paid;
                paid = cumulative;
                share
            })
            .collect();
        Self::new(ages.to_vec(), shares)
    }

    /// The pattern `by` years later: for losses that occur, on average,
    /// `by` years after inception (½ for losses occurring evenly over a
    /// year when the ages run from the accident date).
    pub fn shifted(&self, by: f64) -> Result<Self> {
        Self::new(
            self.times.iter().map(|t| t + by).collect(),
            self.shares.clone(),
        )
    }

    /// The pattern of what is unpaid at `age`, with times measured from
    /// `age`: the payments after it, each over the share outstanding. For
    /// a reserve valued at `age`, such as one ceded under a loss portfolio
    /// transfer; payments at `age` itself count as paid.
    ///
    /// ```
    /// use prospicio_pricing::payout::PaymentPattern;
    ///
    /// let p = PaymentPattern::new(vec![1.0, 2.0, 3.0, 4.0], vec![0.25; 4]).unwrap();
    /// let unpaid = p.unpaid_from(2.0).unwrap();
    /// assert_eq!(unpaid.times(), [1.0, 2.0]);
    /// assert_eq!(unpaid.shares(), [0.5, 0.5]);
    /// ```
    pub fn unpaid_from(&self, age: f64) -> Result<Self> {
        if !(age.is_finite() && age >= 0.0) {
            return Err(Error::InvalidParameter {
                name: "age",
                value: age,
                reason: "must be finite and non-negative",
            });
        }
        let outstanding = self.outstanding(age);
        let (times, shares): (Vec<f64>, Vec<f64>) = self
            .times
            .iter()
            .zip(&self.shares)
            .filter(|(t, s)| **t > age && **s > 0.0)
            .map(|(t, s)| (t - age, s / outstanding))
            .unzip();
        if times.is_empty() {
            return Err(Error::Data(format!("nothing is unpaid at age {age}")));
        }
        Self::new(times, shares)
    }

    /// The times.
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// The share paid at each time.
    pub fn shares(&self) -> &[f64] {
        &self.shares
    }

    /// The mean time to payment, undiscounted.
    pub fn duration(&self) -> f64 {
        self.times
            .iter()
            .zip(&self.shares)
            .map(|(t, s)| t * s)
            .sum()
    }

    /// The share not yet paid at time `t`: payments at `t` itself count as
    /// paid.
    pub fn outstanding(&self, t: f64) -> f64 {
        let paid: f64 = self
            .times
            .iter()
            .zip(&self.shares)
            .filter(|(time, _)| **time <= t)
            .map(|(_, s)| s)
            .sum();
        (1.0 - paid).max(0.0)
    }

    /// The share paid by time `t` in years, read between the pattern's
    /// times as `how` says: 0 at time 0 (unless a share is paid at 0) and
    /// 1 from the last time on, since the shares, any tail included, sum
    /// to 1.
    ///
    /// ```
    /// use prospicio_pricing::payout::{Interpolation, PaymentPattern};
    ///
    /// let p = PaymentPattern::new(vec![1.0, 2.0], vec![0.6, 0.4]).unwrap();
    /// assert_eq!(p.paid_by(0.5, Interpolation::Step), 0.0);
    /// assert!((p.paid_by(0.5, Interpolation::Linear) - 0.3).abs() < 1e-15);
    /// assert!((p.paid_by(1.5, Interpolation::Linear) - 0.8).abs() < 1e-15);
    /// // Unpaid 1 to 0.4 over the first year falls by sqrt(0.4) in half of it.
    /// let half = 1.0 - 0.4_f64.sqrt();
    /// assert!((p.paid_by(0.5, Interpolation::LogLinear) - half).abs() < 1e-15);
    /// assert_eq!(p.paid_by(2.0, Interpolation::Step), 1.0);
    /// ```
    pub fn paid_by(&self, t: f64, how: Interpolation) -> f64 {
        let mut a = 0.0;
        let mut fa = 0.0;
        for (&b, &s) in self.times.iter().zip(&self.shares) {
            let fb = (fa + s).min(1.0);
            if t < b {
                let w = (t - a) / (b - a);
                return match how {
                    Interpolation::Step => fa,
                    Interpolation::Linear => fa + (fb - fa) * w,
                    Interpolation::LogLinear if fb < 1.0 => {
                        1.0 - (1.0 - fa) * ((1.0 - fb) / (1.0 - fa)).powf(w)
                    }
                    Interpolation::LogLinear => fa + (fb - fa) * w,
                };
            }
            a = b;
            fa = fb;
        }
        1.0
    }

    /// The cumulative share paid at the end of each of `months` months from
    /// time 0, `paid_by((m + 1) / 12)` for `m = 0, 1, ...`: a paid pattern
    /// for the cash-flow engine ([`crate::cashflow::MonthlyPatterns`]), or
    /// a pattern measured from the accident date for
    /// [`MonthlyPatterns::from_annual`](crate::cashflow::MonthlyPatterns::from_annual).
    /// At month `12k − 1` it is the pattern's cumulative share at `k` years.
    ///
    /// ```
    /// use prospicio_pricing::payout::{Interpolation, PaymentPattern};
    ///
    /// let p = PaymentPattern::from_development(&[1.0, 2.0, 3.0], &[4.0, 1.25, 1.0]).unwrap();
    /// let m = p.to_monthly(36, Interpolation::Linear).unwrap();
    /// assert!((m[11] - 0.25).abs() < 1e-15 && (m[23] - 0.8).abs() < 1e-15);
    /// assert!((m[0] - 0.25 / 12.0).abs() < 1e-15);
    /// assert_eq!(m[35], 1.0);
    /// ```
    pub fn to_monthly(&self, months: usize, how: Interpolation) -> Result<Vec<f64>> {
        if months == 0 {
            return Err(Error::InvalidParameter {
                name: "months",
                value: 0.0,
                reason: "must be at least 1",
            });
        }
        Ok((0..months)
            .map(|m| self.paid_by((m + 1) as f64 / 12.0, how))
            .collect())
    }

    /// The present value at inception of a unit of ultimate loss, at an
    /// annual effective rate: `Σ share (1 + rate)^-time`.
    pub fn discount_factor(&self, rate: f64) -> f64 {
        self.times
            .iter()
            .zip(&self.shares)
            .map(|(t, s)| s * (1.0 + rate).powf(-t))
            .sum()
    }

    /// The discounted capital years per unit of initial capital: capital is
    /// held in year `k` in proportion to the share outstanding at its
    /// start, and its cost is charged at the start of the year,
    /// `Σ_k outstanding(k) (1 + rate)^-k`, over the years until the last
    /// payment. Year 0 counts in full, so a pattern that pays everything
    /// within the first year gives 1.
    pub fn capital_years(&self, rate: f64) -> f64 {
        let last = self.times[self.times.len() - 1];
        let years = last.ceil().max(1.0) as usize;
        (0..years)
            .map(|k| {
                let held = if k == 0 {
                    1.0
                } else {
                    self.outstanding(k as f64)
                };
                held * (1.0 + rate).powi(-(k as i32))
            })
            .sum()
    }
}

/// A [`Price`] restated over a payout, from [`price_over_payout`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PayoutPrice {
    /// Expected loss, undiscounted.
    pub expected_loss: f64,
    /// Expected loss discounted to inception.
    pub discounted_loss: f64,
    /// Premium at inception.
    pub premium: f64,
    /// Assets backing the loss, the one-year risk measure.
    pub assets: f64,
    /// The losses' discount factor.
    pub discount_factor: f64,
    /// The discounted capital years per unit of initial capital (1 for a
    /// distortion price, which carries no capital over time).
    pub capital_years: f64,
}

impl PayoutPrice {
    /// Initial capital, assets less premium.
    pub fn capital(&self) -> f64 {
        self.assets - self.premium
    }

    /// Margin over the discounted loss, `P − E[X] d`.
    pub fn margin(&self) -> f64 {
        self.premium - self.discounted_loss
    }
}

/// A one-period price restated over a payout, at an annual effective
/// discount `rate`.
///
/// With [`PremiumRule::CostOfCapital`] `r`, the premium solves
/// `P = E[X] d + r (a − P) A`: the discounted loss, plus the cost of the
/// initial capital held while losses are outstanding (see the
/// [module](self)). With [`PremiumRule::Distortion`], the risk-loaded
/// premium is discounted with the losses, `P = ρ(X) d`. `price` must come
/// from the same rule.
///
/// ```
/// use prospicio_pricing::payout::{PaymentPattern, price_over_payout};
/// use prospicio_pricing::risk_load::{PremiumRule, Price};
///
/// let rule = PremiumRule::cost_of_capital(0.1).unwrap();
/// let one_year = Price { expected_loss: 10.0, premium: (10.0 + 0.1 * 30.0) / 1.1, assets: 30.0 };
/// // Paid within the year, undiscounted: the one-period price.
/// let now = PaymentPattern::new(vec![0.5], vec![1.0]).unwrap();
/// let p = price_over_payout(&one_year, &rule, &now, 0.0).unwrap();
/// assert!((p.premium - one_year.premium).abs() < 1e-12);
/// // Paid over three years: more capital years, a higher premium.
/// let slow = PaymentPattern::new(vec![1.0, 2.0, 3.0], vec![0.2, 0.3, 0.5]).unwrap();
/// assert!(price_over_payout(&one_year, &rule, &slow, 0.0).unwrap().premium > p.premium);
/// ```
pub fn price_over_payout(
    price: &Price,
    rule: &PremiumRule,
    pattern: &PaymentPattern,
    rate: f64,
) -> Result<PayoutPrice> {
    if !(rate.is_finite() && rate > -1.0) {
        return Err(Error::InvalidParameter {
            name: "rate",
            value: rate,
            reason: "must be finite and above -1",
        });
    }
    let d = pattern.discount_factor(rate);
    let discounted_loss = price.expected_loss * d;
    let (premium, capital_years) = match rule {
        PremiumRule::Distortion(_) => (price.premium * d, 1.0),
        PremiumRule::CostOfCapital(r) => {
            let a = pattern.capital_years(rate);
            ((discounted_loss + r * a * price.assets) / (1.0 + r * a), a)
        }
    };
    Ok(PayoutPrice {
        expected_loss: price.expected_loss,
        discounted_loss,
        premium,
        assets: price.assets,
        discount_factor: d,
        capital_years,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_forms() {
        // An annuity of 1/4 at 1, 2, 3, 4 years at 5%.
        let p = PaymentPattern::new(vec![1.0, 2.0, 3.0, 4.0], vec![0.25; 4]).unwrap();
        let v: f64 = 1.0 / 1.05;
        let annuity = (1.0 - v.powi(4)) / 0.05;
        assert!((p.discount_factor(0.05) - annuity / 4.0).abs() < 1e-15);
        assert_eq!(p.duration(), 2.5);
        assert_eq!(p.outstanding(0.0), 1.0);
        assert_eq!(p.outstanding(1.0), 0.75);
        assert_eq!(p.outstanding(4.0), 0.0);
        // Capital years: 1, 0.75, 0.5, 0.25 at the starts of years 0 to 3.
        let want = 1.0 + 0.75 * v + 0.5 * v * v + 0.25 * v.powi(3);
        assert!((p.capital_years(0.05) - want).abs() < 1e-15);
        assert_eq!(p.discount_factor(0.0), 1.0);
    }

    #[test]
    fn the_unpaid_part_reprices_the_tail_of_the_pattern() {
        let p = PaymentPattern::from_development(&[1.0, 2.0, 3.0, 4.0], &[4.0, 2.0, 1.25, 1.0])
            .unwrap();
        // Paid 0.25, 0.5, 0.8, 1: at 1.5 the share outstanding is 0.75, paid
        // 0.25, 0.3 and 0.2 at 0.5, 1.5 and 2.5 years from the valuation.
        let u = p.unpaid_from(1.5).unwrap();
        assert_eq!(u.times(), [0.5, 1.5, 2.5]);
        let want = [0.25 / 0.75, 0.3 / 0.75, 0.2 / 0.75];
        for (a, b) in u.shares().iter().zip(want) {
            assert!((a - b).abs() < 1e-14);
        }
        // Discounting the unpaid part from inception is the original
        // discount on the payments after the age, over the outstanding.
        let rate = 0.05;
        let after: f64 = p
            .times()
            .iter()
            .zip(p.shares())
            .filter(|(t, _)| **t > 1.5)
            .map(|(t, s)| s * (1.0f64 + rate).powf(-t))
            .sum();
        let at_age = u.discount_factor(rate) * (1.0f64 + rate).powf(-1.5) * 0.75;
        assert!((after - at_age).abs() < 1e-15);
        // At age 0 the pattern is unchanged.
        assert_eq!(p.unpaid_from(0.0).unwrap(), p);
        assert!(p.unpaid_from(4.0).is_err());
        assert!(p.unpaid_from(-1.0).is_err());
        assert!(p.unpaid_from(f64::NAN).is_err());
    }

    #[test]
    fn one_payment_within_the_year_undiscounted_is_the_one_period_price() {
        for r in [0.05, 0.1, 0.2] {
            let rule = PremiumRule::cost_of_capital(r).unwrap();
            let e = 7.0;
            let a = 40.0;
            let one = Price {
                expected_loss: e,
                premium: (e + r * a) / (1.0 + r),
                assets: a,
            };
            for t in [0.0, 0.5, 1.0] {
                let now = PaymentPattern::new(vec![t], vec![1.0]).unwrap();
                let p = price_over_payout(&one, &rule, &now, 0.0).unwrap();
                assert!((p.premium - one.premium).abs() < 1e-14, "{r} {t}");
                assert_eq!(p.capital_years, 1.0);
                assert!((p.capital() * r - p.margin()).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn the_premium_closes_over_the_payout() {
        let rule = PremiumRule::cost_of_capital(0.1).unwrap();
        let one = Price {
            expected_loss: 10.0,
            premium: 13.0 / 1.1,
            assets: 30.0,
        };
        let slow = PaymentPattern::new(vec![1.0, 2.0, 3.0], vec![0.2, 0.3, 0.5]).unwrap();
        let p = price_over_payout(&one, &rule, &slow, 0.04).unwrap();
        // P = E d + r (a − P) A, each side computed on its own.
        let d = 0.2 / 1.04 + 0.3 / 1.04f64.powi(2) + 0.5 / 1.04f64.powi(3);
        let a_years = 1.0 + 0.8 / 1.04 + 0.5 / 1.04f64.powi(2);
        assert!((p.discount_factor - d).abs() < 1e-15);
        assert!((p.capital_years - a_years).abs() < 1e-14);
        assert!((p.premium - (10.0 * d + 0.1 * (30.0 - p.premium) * a_years)).abs() < 1e-12);

        let g = PremiumRule::Distortion(prospicio_prob::Distortion::tvar(0.5).unwrap());
        let q = price_over_payout(&one, &g, &slow, 0.04).unwrap();
        assert!((q.premium - one.premium * d).abs() < 1e-14);
        assert!(price_over_payout(&one, &rule, &slow, -1.0).is_err());
    }

    #[test]
    fn patterns_from_development_and_their_errors() {
        let p = PaymentPattern::from_development(&[1.0, 2.0], &[2.0, 1.0]).unwrap();
        assert_eq!(p.shares(), &[0.5, 0.5]);
        assert_eq!(p.shifted(0.5).unwrap().times(), &[1.5, 2.5]);
        assert!(PaymentPattern::from_development(&[1.0, 2.0], &[2.0, 1.1]).is_err());
        assert!(PaymentPattern::from_development(&[1.0, 2.0], &[1.0, 2.0]).is_err());
        assert!(PaymentPattern::from_development(&[1.0], &[0.5]).is_err());
        assert!(PaymentPattern::new(vec![1.0, 1.0], vec![0.5, 0.5]).is_err());
        assert!(PaymentPattern::new(vec![1.0], vec![0.9]).is_err());
        assert!(PaymentPattern::new(vec![-1.0], vec![1.0]).is_err());
    }

    #[test]
    fn a_monthly_pattern_meets_the_annual_one_at_whole_years() {
        let p = PaymentPattern::from_development(&[1.0, 2.0, 3.0, 5.0], &[5.0, 2.0, 1.25, 1.0])
            .unwrap();
        for how in [
            Interpolation::Step,
            Interpolation::Linear,
            Interpolation::LogLinear,
        ] {
            let m = p.to_monthly(72, how).unwrap();
            for (k, f) in [(1, 0.2), (2, 0.5), (3, 0.8), (5, 1.0)] {
                assert!((m[12 * k - 1] - f).abs() < 1e-12, "{how:?} at {k}");
            }
            assert!(m[0] >= 0.0 && m.windows(2).all(|w| w[1] >= w[0] - 1e-15));
            assert_eq!(m[71], 1.0);
        }
        // Log-linear lies above linear where the unpaid share falls.
        let lin = p.to_monthly(12, Interpolation::Linear).unwrap();
        let log = p.to_monthly(12, Interpolation::LogLinear).unwrap();
        assert!(log[5] > lin[5]);
        assert!(p.to_monthly(0, Interpolation::Linear).is_err());
    }

    #[test]
    fn monthly_steps_discount_as_the_pattern() {
        // Times on month ends: the step pattern's increments, discounted at
        // their month ends, are the pattern's discount factor.
        let p = PaymentPattern::new(vec![0.25, 1.0, 2.5], vec![0.3, 0.5, 0.2]).unwrap();
        let m = p.to_monthly(36, Interpolation::Step).unwrap();
        let rate: f64 = 0.06;
        let pv: f64 = (0..m.len())
            .map(|t| {
                let inc = m[t] - if t == 0 { 0.0 } else { m[t - 1] };
                inc * (1.0 + rate).powf(-((t + 1) as f64) / 12.0)
            })
            .sum();
        assert!((pv - p.discount_factor(rate)).abs() < 1e-14);
    }
}
