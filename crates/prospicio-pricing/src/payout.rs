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
}
