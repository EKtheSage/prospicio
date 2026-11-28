//! Economic inflation and superimposed claims inflation (`esg-dfa.md`,
//! decision 5): an Ornstein-Uhlenbeck process on the continuously
//! compounded inflation rate, and an AR(1) on the claims' annual excess
//! over it.

use prospicio_core::{Error, Result};

use crate::short_rate::loading;

fn check(name: &'static str, value: f64, ok: bool, reason: &'static str) -> Result<()> {
    if value.is_finite() && ok {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name,
            value,
            reason,
        })
    }
}

/// Economic inflation, `di = κ (μ - i) dt + σ dW`, on the continuously
/// compounded rate `i`. The price index is `I_t = exp(∫_0^t i)`, 1 today.
///
/// ```
/// use prospicio_esg::Inflation;
///
/// let cpi = Inflation::new(0.5, 0.025, 0.01, 0.04)?;
/// // The rate reverts from 4% towards 2.5%.
/// assert!((cpi.mean(1.0) - (0.025 + 0.015 * (-0.5f64).exp())).abs() < 1e-15);
/// assert!((cpi.stationary_variance() - 1e-4).abs() < 1e-18);
/// # Ok::<(), prospicio_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Inflation {
    /// The speed of mean reversion, per year.
    pub kappa: f64,
    /// The long-run mean of the rate.
    pub mean: f64,
    /// The volatility, per square root of a year.
    pub sigma: f64,
    /// The rate today.
    pub i0: f64,
}

impl Inflation {
    /// Fails unless `kappa` and `sigma` are finite and positive and `mean`
    /// and `i0` finite.
    pub fn new(kappa: f64, mean: f64, sigma: f64, i0: f64) -> Result<Self> {
        check("kappa", kappa, kappa > 0.0, "must be finite and positive")?;
        check("mean", mean, true, "must be finite")?;
        check("sigma", sigma, sigma > 0.0, "must be finite and positive")?;
        check("i0", i0, true, "must be finite")?;
        Ok(Self {
            kappa,
            mean,
            sigma,
            i0,
        })
    }

    /// The mean of `i(t)`, `μ + (i0 - μ) e^(-κ t)`.
    pub fn mean(&self, t: f64) -> f64 {
        self.mean + (self.i0 - self.mean) * (-self.kappa * t).exp()
    }

    /// The variance of `i(t)`, `σ² (1 - e^(-2κt)) / (2κ)`.
    pub fn variance(&self, t: f64) -> f64 {
        self.sigma * self.sigma * -(-2.0 * self.kappa * t).exp_m1() / (2.0 * self.kappa)
    }

    /// The stationary variance of the rate, `σ² / (2κ)`.
    pub fn stationary_variance(&self) -> f64 {
        self.sigma * self.sigma / (2.0 * self.kappa)
    }

    /// The mean of `ln I_t = ∫_0^t i`, `μ t + (i0 - μ) B(t)` with
    /// `B = (1 - e^(-κt)) / κ`.
    pub fn log_index_mean(&self, t: f64) -> f64 {
        self.mean * t + (self.i0 - self.mean) * loading(self.kappa, t)
    }

    /// The variance of `ln I_t`,
    /// `σ²/κ² [t - 2B(t) + (1 - e^(-2κt)) / (2κ)]`.
    pub fn log_index_variance(&self, t: f64) -> f64 {
        let k = self.kappa;
        self.sigma * self.sigma / (k * k) * (t - 2.0 * loading(k, t) + loading(2.0 * k, t))
    }

    /// The mean of the price index, `E I_t = exp(m + v / 2)` from the
    /// log index's mean and variance, which is Gaussian.
    pub fn index_mean(&self, t: f64) -> f64 {
        (self.log_index_mean(t) + 0.5 * self.log_index_variance(t)).exp()
    }
}

/// Superimposed claims inflation: the claims' annual excess over economic
/// inflation, an AR(1) from year to year,
/// `s_y = m + φ (s_(y-1) - m) + σ ε_y`, from `s_0 = m`. Year `y`'s excess
/// is continuously compounded and spread evenly over the year, so the
/// claims index is the price index times `exp(∫ s)`.
///
/// ```
/// use prospicio_esg::Superimposed;
///
/// let excess = Superimposed::new(0.01, 0.6, 0.008)?;
/// assert!((excess.stationary_variance() - 0.008f64.powi(2) / 0.64).abs() < 1e-18);
/// assert!(Superimposed::new(0.01, 1.0, 0.008).is_err());
/// # Ok::<(), prospicio_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Superimposed {
    /// The long-run mean excess.
    pub mean: f64,
    /// The persistence `φ` from one year to the next.
    pub persistence: f64,
    /// The standard deviation of the annual shock.
    pub sigma: f64,
}

impl Superimposed {
    /// Fails unless `mean` is finite, `persistence` inside `(-1, 1)` and
    /// `sigma` finite and positive.
    pub fn new(mean: f64, persistence: f64, sigma: f64) -> Result<Self> {
        check("mean", mean, true, "must be finite")?;
        check(
            "persistence",
            persistence,
            persistence.abs() < 1.0,
            "must be inside (-1, 1)",
        )?;
        check("sigma", sigma, sigma > 0.0, "must be finite and positive")?;
        Ok(Self {
            mean,
            persistence,
            sigma,
        })
    }

    /// The variance of `s_y`, `σ² (1 - φ^(2y)) / (1 - φ²)`.
    pub fn variance(&self, years: usize) -> f64 {
        let p2 = self.persistence * self.persistence;
        self.sigma * self.sigma * (1.0 - p2.powi(years as i32)) / (1.0 - p2)
    }

    /// The stationary variance, `σ² / (1 - φ²)`.
    pub fn stationary_variance(&self) -> f64 {
        self.sigma * self.sigma / (1.0 - self.persistence * self.persistence)
    }
}
