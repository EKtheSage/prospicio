//! Short-rate models (Brigo and Mercurio, *Interest Rate Models: Theory
//! and Practice*, 2nd ed., chapter 3): Vasicek, CIR and Hull-White one
//! factor, each with its zero-coupon bond prices in closed form.

use prospicio_core::{Error, Result};

use crate::svensson::Svensson;

fn positive(name: &'static str, value: f64) -> Result<()> {
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

fn finite(name: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name,
            value,
            reason: "must be finite",
        })
    }
}

/// `(1 - e^(-a τ)) / a`, the bond's loading on the short rate.
pub(crate) fn loading(a: f64, tau: f64) -> f64 {
    -(-a * tau).exp_m1() / a
}

/// Vasicek's `dr = a (b - r) dt + σ dW`: a Gaussian short rate reverting
/// to `b` at speed `a` (it can go below zero).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vasicek {
    /// The speed of mean reversion, per year.
    pub a: f64,
    /// The long-run mean of the short rate.
    pub b: f64,
    /// The volatility, per square root of a year.
    pub sigma: f64,
    /// The short rate today.
    pub r0: f64,
}

impl Vasicek {
    /// Fails unless `a` and `sigma` are finite and positive and `b` and
    /// `r0` finite.
    pub fn new(a: f64, b: f64, sigma: f64, r0: f64) -> Result<Self> {
        positive("a", a)?;
        finite("b", b)?;
        positive("sigma", sigma)?;
        finite("r0", r0)?;
        Ok(Self { a, b, sigma, r0 })
    }

    /// The mean of `r(t)`, `b + (r0 - b) e^(-a t)`.
    pub fn mean(&self, t: f64) -> f64 {
        self.b + (self.r0 - self.b) * (-self.a * t).exp()
    }

    /// The variance of `r(t)`, `σ² (1 - e^(-2 a t)) / (2 a)`.
    pub fn variance(&self, t: f64) -> f64 {
        self.sigma * self.sigma * -(-2.0 * self.a * t).exp_m1() / (2.0 * self.a)
    }

    /// The price at short rate `r` of a bond paying 1 in `tau` years,
    /// `A(τ) e^(-B(τ) r)` with `B = (1 - e^(-a τ)) / a` and
    /// `ln A = (b - σ²/(2a²)) (B - τ) - σ² B² / (4a)`.
    pub fn bond_price(&self, r: f64, tau: f64) -> f64 {
        let (a, s2) = (self.a, self.sigma * self.sigma);
        let b = loading(a, tau);
        let ln_a = (self.b - s2 / (2.0 * a * a)) * (b - tau) - s2 * b * b / (4.0 * a);
        (ln_a - b * r).exp()
    }
}

/// Cox, Ingersoll and Ross's `dr = a (b - r) dt + σ √r dW`: a short rate
/// that stays non-negative, simulated by its exact noncentral chi-square
/// transition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cir {
    /// The speed of mean reversion, per year.
    pub a: f64,
    /// The long-run mean of the short rate.
    pub b: f64,
    /// The volatility coefficient.
    pub sigma: f64,
    /// The short rate today.
    pub r0: f64,
}

impl Cir {
    /// Fails unless `a`, `b` and `sigma` are finite and positive and `r0`
    /// finite and non-negative. The Feller condition is not required:
    /// [`feller`](Self::feller) reports it.
    pub fn new(a: f64, b: f64, sigma: f64, r0: f64) -> Result<Self> {
        positive("a", a)?;
        positive("b", b)?;
        positive("sigma", sigma)?;
        if !(r0.is_finite() && r0 >= 0.0) {
            return Err(Error::InvalidParameter {
                name: "r0",
                value: r0,
                reason: "must be finite and non-negative",
            });
        }
        Ok(Self { a, b, sigma, r0 })
    }

    /// Whether `2 a b ≥ σ²`, under which the rate never reaches zero.
    pub fn feller(&self) -> bool {
        2.0 * self.a * self.b >= self.sigma * self.sigma
    }

    /// The mean of `r(t)`, `b + (r0 - b) e^(-a t)`.
    pub fn mean(&self, t: f64) -> f64 {
        self.b + (self.r0 - self.b) * (-self.a * t).exp()
    }

    /// The variance of `r(t)`,
    /// `r0 σ² e^(-a t) (1 - e^(-a t)) / a + b σ² (1 - e^(-a t))² / (2a)`.
    pub fn variance(&self, t: f64) -> f64 {
        let (a, s2) = (self.a, self.sigma * self.sigma);
        let e = (-a * t).exp();
        self.r0 * s2 * e * (1.0 - e) / a + self.b * s2 * (1.0 - e).powi(2) / (2.0 * a)
    }

    /// The price at short rate `r` of a bond paying 1 in `tau` years,
    /// `A(τ) e^(-B(τ) r)` with `h = √(a² + 2σ²)`,
    /// `A = [2h e^((a+h)τ/2) / (2h + (a+h)(e^(hτ) - 1))]^(2ab/σ²)` and
    /// `B = 2(e^(hτ) - 1) / (2h + (a+h)(e^(hτ) - 1))`.
    pub fn bond_price(&self, r: f64, tau: f64) -> f64 {
        let (a, s2) = (self.a, self.sigma * self.sigma);
        let h = (a * a + 2.0 * s2).sqrt();
        let g = (h * tau).exp_m1();
        let denominator = 2.0 * h + (a + h) * g;
        let ln_a =
            2.0 * a * self.b / s2 * ((2.0 * h).ln() + (a + h) * tau / 2.0 - denominator.ln());
        let b = 2.0 * g / denominator;
        (ln_a - b * r).exp()
    }
}

/// Today's curve, which Hull-White reproduces.
#[derive(Debug, Clone, PartialEq)]
pub enum InitialCurve {
    /// Annually compounded spot rates by maturity in years, linear in the
    /// rate between maturities and flat beyond the ends, the convention of
    /// `prospicio_reserving::DiscountCurve::spot`.
    Spot {
        /// Maturities in years, positive and increasing.
        maturities: Vec<f64>,
        /// The spot rate at each, above -1.
        rates: Vec<f64>,
    },
    /// A Nelson-Siegel-Svensson curve (continuously compounded).
    Svensson(Svensson),
}

impl InitialCurve {
    /// A spot curve; fails unless the maturities are finite, positive and
    /// increasing and the rates finite, above -1 and as many.
    pub fn spot(maturities: Vec<f64>, rates: Vec<f64>) -> Result<Self> {
        if maturities.is_empty() || maturities.len() != rates.len() {
            return Err(Error::Data(format!(
                "a spot curve needs as many rates as maturities, at least one: {} and {}",
                maturities.len(),
                rates.len()
            )));
        }
        let mut previous = 0.0;
        for &m in &maturities {
            if !(m.is_finite() && m > previous) {
                return Err(Error::InvalidParameter {
                    name: "maturity",
                    value: m,
                    reason: "must be finite, positive and increasing",
                });
            }
            previous = m;
        }
        if let Some(&r) = rates.iter().find(|r| !(r.is_finite() && **r > -1.0)) {
            return Err(Error::InvalidParameter {
                name: "rate",
                value: r,
                reason: "must be finite and above -1",
            });
        }
        Ok(Self::Spot { maturities, rates })
    }

    /// The spot rate and its slope at `t` of a spot curve.
    fn spot_and_slope(maturities: &[f64], rates: &[f64], t: f64) -> (f64, f64) {
        let last = maturities.len() - 1;
        if t <= maturities[0] {
            return (rates[0], 0.0);
        }
        if t >= maturities[last] {
            return (rates[last], 0.0);
        }
        let k = maturities.partition_point(|&x| x <= t);
        let slope = (rates[k] - rates[k - 1]) / (maturities[k] - maturities[k - 1]);
        (rates[k - 1] + slope * (t - maturities[k - 1]), slope)
    }

    /// The price today of a bond paying 1 at `t` years.
    pub fn discount(&self, t: f64) -> f64 {
        match self {
            Self::Spot { maturities, rates } => {
                if t == 0.0 {
                    return 1.0;
                }
                let (y, _) = Self::spot_and_slope(maturities, rates, t);
                (1.0 + y).powf(-t)
            }
            Self::Svensson(s) => s.discount(t),
        }
    }

    /// The instantaneous forward rate today at `t` years, `-d ln P / dt`
    /// (the right derivative at a node of a spot curve).
    pub fn forward(&self, t: f64) -> f64 {
        match self {
            Self::Spot { maturities, rates } => {
                let (y, slope) = Self::spot_and_slope(maturities, rates, t);
                (1.0 + y).ln() + t * slope / (1.0 + y)
            }
            Self::Svensson(s) => s.forward(t),
        }
    }
}

/// Hull and White's one-factor `dr = (θ(t) - a r) dt + σ dW`, with `θ`
/// fitted so that the model's bond prices today are the initial curve's
/// (Brigo and Mercurio, section 3.3): `r = x + φ`, `x` an Ornstein-Uhlenbeck
/// process from 0 and `φ(t) = f(0, t) + σ² (1 - e^(-a t))² / (2a²)`.
#[derive(Debug, Clone, PartialEq)]
pub struct HullWhite {
    /// The speed of mean reversion, per year.
    pub a: f64,
    /// The volatility, per square root of a year.
    pub sigma: f64,
    /// Today's curve.
    pub initial: InitialCurve,
}

impl HullWhite {
    /// Fails unless `a` and `sigma` are finite and positive.
    pub fn new(a: f64, sigma: f64, initial: InitialCurve) -> Result<Self> {
        positive("a", a)?;
        positive("sigma", sigma)?;
        Ok(Self { a, sigma, initial })
    }

    /// The variance of `∫_t^(t+τ) x`, given `x(t)`:
    /// `σ²/a² [τ + (2/a) e^(-a τ) - (1/(2a)) e^(-2 a τ) - 3/(2a)]`.
    pub(crate) fn integral_variance(&self, tau: f64) -> f64 {
        let (a, s2) = (self.a, self.sigma * self.sigma);
        s2 / (a * a)
            * (tau + 2.0 / a * (-a * tau).exp() - 0.5 / a * (-2.0 * a * tau).exp() - 1.5 / a)
    }

    /// `φ(t)`, the short rate less the factor `x`.
    pub fn shift(&self, t: f64) -> f64 {
        let a = self.a;
        self.initial.forward(t)
            + self.sigma * self.sigma / (2.0 * a * a) * (-a * t).exp_m1().powi(2)
    }

    /// The price at time `t`, factor `x`, of a bond paying 1 at `t + tau`:
    /// `P(0, t+τ) / P(0, t) · exp(-B(τ) x + [V(τ) - V(t+τ) + V(t)] / 2)`.
    pub fn bond_price(&self, t: f64, x: f64, tau: f64) -> f64 {
        let ratio = self.initial.discount(t + tau) / self.initial.discount(t);
        let v = |s: f64| self.integral_variance(s);
        ratio * (-loading(self.a, tau) * x + 0.5 * (v(tau) - v(t + tau) + v(t))).exp()
    }
}

/// The short-rate model of an [`Esg`](crate::Esg).
#[derive(Debug, Clone, PartialEq)]
pub enum ShortRate {
    /// Vasicek.
    Vasicek(Vasicek),
    /// CIR.
    Cir(Cir),
    /// Hull-White one factor.
    HullWhite(HullWhite),
}

impl From<Vasicek> for ShortRate {
    fn from(m: Vasicek) -> Self {
        Self::Vasicek(m)
    }
}

impl From<Cir> for ShortRate {
    fn from(m: Cir) -> Self {
        Self::Cir(m)
    }
}

impl From<HullWhite> for ShortRate {
    fn from(m: HullWhite) -> Self {
        Self::HullWhite(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bond_prices_start_at_one_and_match_their_short_ends() {
        let v = Vasicek::new(0.3, 0.04, 0.01, 0.02).unwrap();
        let c = Cir::new(0.3, 0.04, 0.05, 0.02).unwrap();
        for (p, r) in [
            (v.bond_price(0.02, 0.0), 1.0),
            (c.bond_price(0.02, 0.0), 1.0),
        ] {
            assert!((p - r).abs() < 1e-15);
        }
        // The yield of a very short bond is the short rate.
        let h = 1e-6;
        assert!((-(v.bond_price(0.02, h)).ln() / h - 0.02).abs() < 1e-6);
        assert!((-(c.bond_price(0.02, h)).ln() / h - 0.02).abs() < 1e-6);
        assert!(c.feller());
        assert!(!Cir::new(0.1, 0.01, 0.1, 0.01).unwrap().feller());
    }

    #[test]
    fn hull_white_reproduces_the_initial_curve_today() {
        let initial = InitialCurve::spot(
            vec![1.0, 2.0, 5.0, 10.0, 30.0],
            vec![0.03, 0.032, 0.036, 0.04, 0.042],
        )
        .unwrap();
        let hw = HullWhite::new(0.1, 0.01, initial.clone()).unwrap();
        for tau in [0.5, 1.0, 3.7, 10.0, 25.0] {
            assert!((hw.bond_price(0.0, 0.0, tau) - initial.discount(tau)).abs() < 1e-14);
        }
        // The forward is the derivative of the log price.
        for t in [0.5, 1.5, 3.7, 12.0] {
            let h = 1e-6;
            let numeric =
                -(initial.discount(t + h).ln() - initial.discount(t - h).ln()) / (2.0 * h);
            assert!((numeric - initial.forward(t)).abs() < 1e-7, "{t}");
        }
        assert!((hw.shift(0.0) - 0.03f64.ln_1p()).abs() < 1e-15);
    }

    #[test]
    fn the_parameters_are_checked() {
        assert!(Vasicek::new(0.0, 0.04, 0.01, 0.02).is_err());
        assert!(Vasicek::new(0.3, f64::NAN, 0.01, 0.02).is_err());
        assert!(Cir::new(0.3, 0.04, 0.01, -0.01).is_err());
        assert!(Cir::new(0.3, 0.0, 0.01, 0.01).is_err());
        assert!(InitialCurve::spot(vec![1.0, 1.0], vec![0.01, 0.02]).is_err());
        assert!(InitialCurve::spot(vec![1.0], vec![-1.0]).is_err());
        let flat = InitialCurve::spot(vec![1.0], vec![0.03]).unwrap();
        assert!(HullWhite::new(0.1, 0.0, flat).is_err());
    }
}
