//! Claim-count (frequency) distributions on `0, 1, 2, …`.
//!
//! A separate trait from [`crate::Distribution`]: counts are integers, and
//! Panjer's recursion needs the `(a, b, 0)` form, `p_k = (a + b / k) p_{k-1}`.
//! See `docs/design/distributions.md`.

use act_core::{Error, Result, StreamRng};
use act_math::special::ln_gamma;

use crate::distribution::check_probability;

/// A distribution of claim counts in the `(a, b, 0)` class.
pub trait Counting {
    /// `P(N = k)`.
    fn pmf(&self, k: u64) -> f64;

    /// `E[N]`.
    fn mean(&self) -> f64;

    /// `Var[N]`.
    fn variance(&self) -> f64;

    /// `(a, b)` with `p_k = (a + b / k) p_{k-1}` for `k ≥ 1`.
    fn panjer_ab(&self) -> (f64, f64);

    /// Probability generating function `E[z^N]` for `0 ≤ z ≤ 1`. Panjer's
    /// recursion starts from `P(S = 0) = pgf(f_0)`.
    fn pgf(&self, z: f64) -> f64;

    /// The pgf at a complex `z = (re, im)` with `|z| ≤ 1`, as `(re, im)`.
    /// FFT aggregation applies it to the transform of the severity grid.
    fn pgf_complex(&self, z: (f64, f64)) -> (f64, f64);

    /// `P(N ≤ k)`.
    fn cdf(&self, k: u64) -> f64 {
        (0..=k).map(|j| self.pmf(j)).sum::<f64>().min(1.0)
    }

    /// Smallest `k` with `P(N ≤ k) ≥ p`. `p = 1` gives `u64::MAX`.
    fn quantile(&self, p: f64) -> Result<u64> {
        check_probability(p)?;
        if p == 1.0 {
            return Ok(u64::MAX);
        }
        let mut total = 0.0;
        let mut k = 0;
        loop {
            let q = self.pmf(k);
            total += q;
            // Past the mean with no mass left, the sum cannot grow further.
            if total >= p || (q == 0.0 && k as f64 > self.mean()) {
                return Ok(k);
            }
            k += 1;
        }
    }

    /// `n` counts from stream `rng`, by inverse transform.
    ///
    /// Each draw walks the pmf from 0, so the cost grows with the mean;
    /// fine for the claim counts of a year, slow for means in the
    /// hundreds of thousands.
    fn sample(&self, rng: &mut StreamRng, n: usize) -> Vec<u64> {
        (0..n)
            .map(|_| {
                self.quantile(rng.next_open01())
                    .expect("next_open01 is always in (0, 1)")
            })
            .collect()
    }
}

/// Poisson claim counts with mean `lambda`.
///
/// ```
/// use act_prob::{Counting, Poisson};
///
/// let n = Poisson::new(3.0).unwrap();
/// assert!((n.pmf(0) - (-3f64).exp()).abs() < 1e-15);
/// assert_eq!(n.panjer_ab(), (0.0, 3.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Poisson {
    lambda: f64,
}

impl Poisson {
    /// Poisson with mean `lambda`; fails unless `lambda` is finite and
    /// non-negative.
    pub fn new(lambda: f64) -> Result<Self> {
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(Error::InvalidParameter {
                name: "lambda",
                value: lambda,
                reason: "must be finite and non-negative",
            });
        }
        Ok(Self { lambda })
    }

    /// The mean.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl Counting for Poisson {
    fn pmf(&self, k: u64) -> f64 {
        if self.lambda == 0.0 {
            return if k == 0 { 1.0 } else { 0.0 };
        }
        let k = k as f64;
        (k * self.lambda.ln() - self.lambda - ln_gamma(k + 1.0)).exp()
    }

    fn mean(&self) -> f64 {
        self.lambda
    }

    fn variance(&self) -> f64 {
        self.lambda
    }

    fn panjer_ab(&self) -> (f64, f64) {
        (0.0, self.lambda)
    }

    /// `exp(lambda (z - 1))`.
    fn pgf(&self, z: f64) -> f64 {
        (self.lambda * (z - 1.0)).exp()
    }

    fn pgf_complex(&self, (re, im): (f64, f64)) -> (f64, f64) {
        let modulus = (self.lambda * (re - 1.0)).exp();
        let angle = self.lambda * im;
        (modulus * angle.cos(), modulus * angle.sin())
    }
}

/// Negative binomial claim counts in the actuarial parameterization of
/// Klugman, Panjer & Willmot: mean `r β`, variance `r β (1 + β)`.
///
/// SciPy's `nbinom(n=r, p=1/(1+β))` and R's `dnbinom(size=r, prob=1/(1+β))`
/// are the same distribution.
///
/// ```
/// use act_prob::{Counting, NegativeBinomial};
///
/// let n = NegativeBinomial::from_mean_variance(10.0, 30.0).unwrap();
/// assert!((n.mean() - 10.0).abs() < 1e-12);
/// assert!((n.variance() - 30.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegativeBinomial {
    r: f64,
    beta: f64,
}

impl NegativeBinomial {
    /// Negative binomial with shape `r` and scale `beta`; both must be finite
    /// and positive.
    pub fn new(r: f64, beta: f64) -> Result<Self> {
        if !r.is_finite() || r <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "r",
                value: r,
                reason: "must be finite and positive",
            });
        }
        if !beta.is_finite() || beta <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "beta",
                value: beta,
                reason: "must be finite and positive",
            });
        }
        Ok(Self { r, beta })
    }

    /// The negative binomial with this mean and variance; the variance
    /// must exceed the mean (over-dispersion).
    pub fn from_mean_variance(mean: f64, variance: f64) -> Result<Self> {
        if !mean.is_finite() || mean <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "mean",
                value: mean,
                reason: "must be finite and positive",
            });
        }
        if !variance.is_finite() || variance <= mean {
            return Err(Error::InvalidParameter {
                name: "variance",
                value: variance,
                reason: "must exceed the mean",
            });
        }
        let beta = variance / mean - 1.0;
        Self::new(mean / beta, beta)
    }

    /// Shape `r`.
    pub fn r(&self) -> f64 {
        self.r
    }

    /// Scale `β`.
    pub fn beta(&self) -> f64 {
        self.beta
    }
}

impl Counting for NegativeBinomial {
    fn pmf(&self, k: u64) -> f64 {
        let (r, beta) = (self.r, self.beta);
        let k = k as f64;
        let ln_1p_beta = beta.ln_1p();
        (ln_gamma(k + r) - ln_gamma(r) - ln_gamma(k + 1.0) - r * ln_1p_beta
            + k * (beta.ln() - ln_1p_beta))
            .exp()
    }

    fn mean(&self) -> f64 {
        self.r * self.beta
    }

    fn variance(&self) -> f64 {
        self.r * self.beta * (1.0 + self.beta)
    }

    fn panjer_ab(&self) -> (f64, f64) {
        let a = self.beta / (1.0 + self.beta);
        (a, (self.r - 1.0) * a)
    }

    /// `(1 - beta (z - 1))^(-r)`.
    fn pgf(&self, z: f64) -> f64 {
        (-self.r * (self.beta * (1.0 - z)).ln_1p()).exp()
    }

    fn pgf_complex(&self, (re, im): (f64, f64)) -> (f64, f64) {
        // w = 1 + beta (1 - z); w^(-r) = exp(-r (ln|w| + i arg w)).
        let (wr, wi) = (1.0 + self.beta * (1.0 - re), -self.beta * im);
        let ln_modulus = 0.5 * (wr * wr + wi * wi).ln();
        let arg = wi.atan2(wr);
        let modulus = (-self.r * ln_modulus).exp();
        let angle = -self.r * arg;
        (modulus * angle.cos(), modulus * angle.sin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pmf satisfies the (a, b, 0) recursion it reports.
    fn check_recursion(n: &impl Counting) {
        let (a, b) = n.panjer_ab();
        for k in 1..60u64 {
            let want = (a + b / k as f64) * n.pmf(k - 1);
            let got = n.pmf(k);
            assert!((got - want).abs() <= 1e-12 * want.max(1e-300), "k {k}");
        }
    }

    #[test]
    fn panjer_recursion_holds() {
        check_recursion(&Poisson::new(4.5).unwrap());
        check_recursion(&NegativeBinomial::new(2.5, 1.5).unwrap());
        check_recursion(&NegativeBinomial::new(0.7, 10.0).unwrap());
    }

    #[test]
    fn pmf_sums_to_one_and_matches_moments() {
        for n in [
            &Poisson::new(7.0).unwrap() as &dyn Counting,
            &NegativeBinomial::new(3.0, 2.0).unwrap(),
        ] {
            let pmf: Vec<f64> = (0..400).map(|k| n.pmf(k)).collect();
            let total: f64 = pmf.iter().sum();
            let mean: f64 = pmf.iter().enumerate().map(|(k, p)| k as f64 * p).sum();
            let var: f64 = pmf
                .iter()
                .enumerate()
                .map(|(k, p)| (k as f64 - mean).powi(2) * p)
                .sum();
            assert!((total - 1.0).abs() < 1e-12);
            assert!((mean - n.mean()).abs() < 1e-10);
            assert!((var - n.variance()).abs() < 1e-9);
        }
    }

    #[test]
    fn quantile_and_cdf_are_inverse() {
        let n = Poisson::new(3.0).unwrap();
        for k in 0..15 {
            assert_eq!(n.quantile(n.cdf(k)), Ok(k));
        }
        assert_eq!(n.quantile(0.0), Ok(0));
        assert_eq!(n.quantile(1.0), Ok(u64::MAX));
        assert!(n.quantile(1.5).is_err());
    }

    #[test]
    fn sampling_is_reproducible_and_centred() {
        let n = NegativeBinomial::new(4.0, 2.5).unwrap();
        let a = n.sample(&mut StreamRng::new(5, 0), 100_000);
        assert_eq!(a, n.sample(&mut StreamRng::new(5, 0), 100_000));
        let mean = a.iter().sum::<u64>() as f64 / a.len() as f64;
        // Standard error is sqrt(35 / 100_000) ≈ 0.019.
        assert!((mean - 10.0).abs() < 0.1, "{mean}");
    }

    #[test]
    fn pgf_matches_the_pmf() {
        for n in [
            &Poisson::new(2.5).unwrap() as &dyn Counting,
            &NegativeBinomial::new(1.5, 3.0).unwrap(),
        ] {
            for z in [0.0f64, 0.3, 0.9, 1.0] {
                let series: f64 = (0..500).map(|k| n.pmf(k) * z.powi(k as i32)).sum();
                assert!((n.pgf(z) - series).abs() < 1e-12, "z {z}");
            }
        }
    }

    #[test]
    fn complex_pgf_matches_the_series() {
        for n in [
            &Poisson::new(2.5).unwrap() as &dyn Counting,
            &NegativeBinomial::new(1.5, 3.0).unwrap(),
        ] {
            for theta in [0.0f64, 0.7, 2.0, 3.1] {
                let (zr, zi) = (0.9 * theta.cos(), 0.9 * theta.sin());
                // Sum p_k z^k, tracking z^k by repeated multiplication.
                let (mut pr, mut pi, mut sr, mut si) = (1.0, 0.0, 0.0, 0.0);
                for k in 0..600 {
                    let p = n.pmf(k);
                    sr += p * pr;
                    si += p * pi;
                    (pr, pi) = (pr * zr - pi * zi, pr * zi + pi * zr);
                }
                let (gr, gi) = n.pgf_complex((zr, zi));
                assert!(
                    (gr - sr).abs() < 1e-12 && (gi - si).abs() < 1e-12,
                    "theta {theta}"
                );
            }
            assert!((n.pgf_complex((0.4, 0.0)).0 - n.pgf(0.4)).abs() < 1e-15);
        }
    }

    #[test]
    fn degenerate_poisson() {
        let n = Poisson::new(0.0).unwrap();
        assert_eq!(n.pmf(0), 1.0);
        assert_eq!(n.pmf(3), 0.0);
        assert_eq!(n.quantile(0.999), Ok(0));
    }

    #[test]
    fn rejects_bad_parameters() {
        assert!(Poisson::new(-1.0).is_err());
        assert!(NegativeBinomial::new(0.0, 1.0).is_err());
        assert!(NegativeBinomial::new(1.0, f64::NAN).is_err());
        assert!(NegativeBinomial::from_mean_variance(10.0, 10.0).is_err());
    }
}
