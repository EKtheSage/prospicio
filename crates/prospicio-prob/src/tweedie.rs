//! The Tweedie distribution with power `1 < p < 2`: compound Poisson with
//! gamma severities.

use prospicio_core::{Error, Result};
use prospicio_math::roots::bisect;
use prospicio_math::special::ln_gamma;

use crate::distribution::{Distribution, check_probability};
use crate::gamma::Gamma;
use crate::severity::Severity;

/// Tweedie distribution with mean `μ`, dispersion `φ` and power
/// `1 < p < 2`: variance `φ μ^p`, a point mass at 0 and a continuous
/// density above it.
///
/// It is the compound Poisson sum `Y = X_1 + … + X_N` with
///
/// ```text
/// N ~ Poisson(λ),        λ = μ^(2-p) / (φ (2 - p))
/// X ~ Gamma(α, θ),       α = (2 - p) / (p - 1),   θ = φ (p - 1) μ^(p-1)
/// ```
///
/// the GLM family for pure premium (losses per exposure), where claim
/// counts and severities are not modelled separately. Every quantity is a
/// Poisson-weighted sum over the number of claims `n`, of the matching
/// quantity of `Gamma(nα, θ)`, summed until the remaining terms are
/// negligible:
///
/// - `P(Y = 0) = e^(-λ)`;
/// - the distribution function, survival function and layer moments from
///   the gamma's, each tail summed directly so both keep their precision;
/// - the density (Dunn & Smyth's series), in log space.
///
/// # Example
///
/// ```
/// use prospicio_prob::{Distribution, Severity, Tweedie};
///
/// let y = Tweedie::new(500.0, 40.0, 1.6).unwrap();
/// assert!((y.mean() - 500.0).abs() < 1e-9);
/// assert!((y.variance() - 40.0 * 500f64.powf(1.6)).abs() < 1e-6);
/// // P(Y = 0) = e^(-λ).
/// assert!((y.cdf(0.0) - (-y.lambda()).exp()).abs() < 1e-15);
/// assert!((y.lev(800.0) + y.stop_loss(800.0) - 500.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tweedie {
    mean: f64,
    dispersion: f64,
    power: f64,
    lambda: f64,
    severity: Gamma,
}

impl Tweedie {
    /// Tweedie with mean `μ > 0`, dispersion `φ > 0` and power `p` in
    /// `(1, 2)`.
    pub fn new(mean: f64, dispersion: f64, power: f64) -> Result<Self> {
        positive("mean", mean)?;
        positive("dispersion", dispersion)?;
        if !(power > 1.0 && power < 2.0) {
            return Err(Error::InvalidParameter {
                name: "power",
                value: power,
                reason: "must be in (1, 2)",
            });
        }
        let lambda = mean.powf(2.0 - power) / (dispersion * (2.0 - power));
        let shape = (2.0 - power) / (power - 1.0);
        let scale = dispersion * (power - 1.0) * mean.powf(power - 1.0);
        Ok(Self {
            mean,
            dispersion,
            power,
            lambda,
            severity: Gamma::new(shape, scale)?,
        })
    }

    /// The Tweedie equal to a Poisson(`lambda`) number of
    /// Gamma(`shape`, `scale`) losses: power `(α + 2) / (α + 1)`, mean
    /// `λαθ`.
    ///
    /// ```
    /// use prospicio_prob::Tweedie;
    ///
    /// let y = Tweedie::from_poisson_gamma(3.0, 2.0, 100.0).unwrap();
    /// assert!((y.power() - 4.0 / 3.0).abs() < 1e-15);
    /// assert!((y.lambda() - 3.0).abs() < 1e-12);
    /// ```
    pub fn from_poisson_gamma(lambda: f64, shape: f64, scale: f64) -> Result<Self> {
        positive("lambda", lambda)?;
        Gamma::new(shape, scale)?;
        let power = (shape + 2.0) / (shape + 1.0);
        let mean = lambda * shape * scale;
        let dispersion = scale / ((power - 1.0) * mean.powf(power - 1.0));
        Self::new(mean, dispersion, power)
    }

    /// Mean `μ`.
    pub fn mean_param(&self) -> f64 {
        self.mean
    }

    /// Dispersion `φ`.
    pub fn dispersion(&self) -> f64 {
        self.dispersion
    }

    /// Power `p`.
    pub fn power(&self) -> f64 {
        self.power
    }

    /// Poisson mean `λ` of the number of losses.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// The gamma distribution of each loss.
    pub fn severity(&self) -> Gamma {
        self.severity
    }

    /// Log density at `y > 0` (the continuous part); at `y = 0`, the log
    /// of the point mass `-λ`. `-inf` below 0.
    ///
    /// ```
    /// use prospicio_prob::Tweedie;
    ///
    /// // λ = 1 with unit exponential losses:
    /// // f(y) = e^(-1-y) Σ_n y^(n-1) / (n! (n-1)!).
    /// let y = Tweedie::from_poisson_gamma(1.0, 1.0, 1.0).unwrap();
    /// let (mut series, mut term) = (0.0, 1.0); // term = 2^(n-1) / (n! (n-1)!)
    /// for n in 1..40 {
    ///     series += term;
    ///     term *= 2.0 / (f64::from(n + 1) * f64::from(n));
    /// }
    /// assert!((y.ln_pdf(2.0) - (-3.0 + f64::ln(series))).abs() < 1e-13);
    /// ```
    pub fn ln_pdf(&self, y: f64) -> f64 {
        if y < 0.0 || y.is_nan() {
            return f64::NEG_INFINITY;
        }
        if y == 0.0 {
            return -self.lambda;
        }
        // ln of term n: ln Pois(n; λ) + ln Gamma(nα, θ).pdf(y). Terms rise to
        // a maximum and fall; sum them relative to the largest.
        let (alpha, theta) = (self.severity.shape(), self.severity.scale());
        let ln_term = |n: f64| {
            n * self.lambda.ln() - self.lambda - ln_gamma(n + 1.0) + (n * alpha - 1.0) * y.ln()
                - y / theta
                - ln_gamma(n * alpha)
                - n * alpha * theta.ln()
        };
        // The terms peak near n where the Poisson and gamma pulls balance.
        let peak = {
            let mut best = (1.0, ln_term(1.0));
            let guess = (y / (alpha * theta)).max(self.lambda).max(1.0);
            for n in [guess.floor(), guess.ceil(), self.lambda.floor().max(1.0)] {
                let v = ln_term(n.max(1.0));
                if v > best.1 {
                    best = (n.max(1.0), v);
                }
            }
            // Walk uphill to the exact peak.
            let (mut n, mut v) = best;
            loop {
                let up = ln_term(n + 1.0);
                if up > v {
                    (n, v) = (n + 1.0, up);
                    continue;
                }
                if n > 1.0 {
                    let down = ln_term(n - 1.0);
                    if down > v {
                        (n, v) = (n - 1.0, down);
                        continue;
                    }
                }
                break (n, v);
            }
        };
        let (n0, v0) = peak;
        let mut sum = 1.0;
        let mut n = n0 + 1.0;
        loop {
            let r = (ln_term(n) - v0).exp();
            sum += r;
            if r <= 1e-17 * sum {
                break;
            }
            n += 1.0;
        }
        let mut n = n0 - 1.0;
        while n >= 1.0 {
            let r = (ln_term(n) - v0).exp();
            sum += r;
            if r <= 1e-17 * sum {
                break;
            }
            n -= 1.0;
        }
        v0 + sum.ln()
    }

    /// `Σ_{n≥1} Pois(n; λ) g(Gamma(nα, θ))` for a `g` that grows with `n`
    /// at most polynomially. Sums up from `n = 1` past both the Poisson
    /// mode and `beyond` (where `g` levels off), then until a term is
    /// negligible.
    fn poisson_sum(&self, beyond: f64, mut g: impl FnMut(&Gamma) -> f64) -> f64 {
        let (alpha, theta) = (self.severity.shape(), self.severity.scale());
        let ln_lambda = self.lambda.ln();
        let floor = self.lambda.max(beyond);
        let mut sum = 0.0;
        for n in 1..10_000_000u32 {
            let nf = f64::from(n);
            let w = (nf * ln_lambda - self.lambda - ln_gamma(nf + 1.0)).exp();
            let term = if w == 0.0 {
                0.0
            } else {
                w * g(&Gamma::new(nf * alpha, theta).expect("valid shape and scale"))
            };
            sum += term;
            if nf > floor && (term <= 1e-17 * sum || w == 0.0) {
                break;
            }
        }
        sum
    }

    /// The `n` beyond which `Gamma(nα, θ)` sits mostly above `y`.
    fn claims_to_reach(&self, y: f64) -> f64 {
        let k = y / (self.severity.shape() * self.severity.scale());
        k + 10.0 * k.sqrt() + 10.0
    }
}

impl Distribution for Tweedie {
    fn mean(&self) -> f64 {
        self.mean
    }

    fn variance(&self) -> f64 {
        self.dispersion * self.mean.powf(self.power)
    }

    fn cdf(&self, y: f64) -> f64 {
        if y < 0.0 {
            return 0.0;
        }
        let zero = (-self.lambda).exp();
        if y == 0.0 {
            return zero;
        }
        (zero + self.poisson_sum(0.0, |g| g.cdf(y))).min(1.0)
    }

    fn survival(&self, y: f64) -> f64 {
        if y < 0.0 {
            return 1.0;
        }
        if y == 0.0 {
            return -(-self.lambda).exp_m1();
        }
        self.poisson_sum(self.claims_to_reach(y), |g| g.survival(y))
            .min(1.0)
    }

    /// 0 when `p` is within the point mass; otherwise by bisection on the
    /// distribution or survival function, whichever is the smaller tail.
    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        if p <= (-self.lambda).exp() {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        let below = |y: f64| {
            if p <= 0.5 {
                self.cdf(y) < p
            } else {
                self.survival(y) > 1.0 - p
            }
        };
        let mut hi = self.mean + self.variance().sqrt();
        while below(hi) {
            hi *= 2.0;
        }
        Ok(bisect(0.0, hi, below))
    }
}

impl Severity for Tweedie {
    fn lev(&self, limit: f64) -> f64 {
        if limit <= 0.0 {
            return limit;
        }
        if limit == f64::INFINITY {
            return self.mean;
        }
        self.poisson_sum(0.0, |g| g.lev(limit))
    }

    fn stop_loss(&self, retention: f64) -> f64 {
        if retention <= 0.0 {
            return self.mean - retention;
        }
        if retention == f64::INFINITY {
            return 0.0;
        }
        self.poisson_sum(self.claims_to_reach(retention), |g| g.stop_loss(retention))
    }

    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        self.poisson_sum(self.claims_to_reach(a + limit.min(1e300)), |g| {
            g.layer(limit, a)
        })
    }

    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        self.poisson_sum(self.claims_to_reach(a + limit.min(1e300)), |g| {
            g.layer_second_moment(limit, a)
        })
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameterizations_agree() {
        let y = Tweedie::new(500.0, 40.0, 1.6).unwrap();
        let z = Tweedie::from_poisson_gamma(y.lambda(), y.severity().shape(), y.severity().scale())
            .unwrap();
        assert!((z.mean() / 500.0 - 1.0).abs() < 1e-14);
        assert!((z.dispersion() / 40.0 - 1.0).abs() < 1e-12);
        assert!((z.power() - 1.6).abs() < 1e-14);
        assert!(Tweedie::new(1.0, 1.0, 2.0).is_err());
        assert!(Tweedie::new(1.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn moments_and_layers_from_the_series() {
        let y = Tweedie::new(500.0, 40.0, 1.6).unwrap();
        // The unlimited layer is the whole distribution.
        assert!((y.layer(f64::INFINITY, 0.0) / y.mean() - 1.0).abs() < 1e-12);
        let m2 = y.layer_second_moment(f64::INFINITY, 0.0);
        let want = y.variance() + y.mean() * y.mean();
        assert!((m2 / want - 1.0).abs() < 1e-12, "{m2} {want}");
        for d in [10.0, 500.0, 5_000.0] {
            assert!((y.lev(d) + y.stop_loss(d) - 500.0).abs() < 1e-9);
            assert!((y.cdf(d) + y.survival(d) - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn density_integrates_to_the_distribution_function() {
        let y = Tweedie::new(10.0, 2.0, 1.4).unwrap();
        // ∫_a^b f by composite Gauss–Legendre equals F(b) - F(a).
        let (a, b) = (0.5, 30.0);
        let panels = 400;
        let h = (b - a) / f64::from(panels);
        let integral: f64 = (0..panels)
            .map(|i| {
                let lo = a + h * f64::from(i);
                prospicio_math::integrate::gauss_legendre(
                    |x| Ok::<_, ()>(y.ln_pdf(x).exp()),
                    lo,
                    lo + h,
                )
                .unwrap()
            })
            .sum();
        assert!(
            (integral - (y.cdf(b) - y.cdf(a))).abs() < 1e-12,
            "{integral}"
        );
    }

    #[test]
    fn quantiles_invert_and_respect_the_atom() {
        let y = Tweedie::new(10.0, 2.0, 1.4).unwrap();
        let p0 = y.cdf(0.0);
        assert_eq!(y.quantile(p0 * 0.5), Ok(0.0));
        for p in [0.5, 0.9, 0.999999] {
            let x = y.quantile(p).unwrap();
            assert!((y.cdf(x) - p).abs() < 1e-12, "{p}");
        }
    }
}
