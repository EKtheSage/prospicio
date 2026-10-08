//! Classical premium principles, the pre-spectral rules that Mildenhall
//! and Major compare against (*Pricing Insurance Risk*, 2022, chapter 9),
//! each with one loading parameter that [`calibrate`] solves for.
//!
//! On a discrete distribution with values `x` and probabilities `p`, mean
//! `μ` and variance `σ²`:
//!
//! | [`Principle`] | Premium |
//! |---|---|
//! | `ExpectedValue(θ)` | `(1 + θ) μ` |
//! | `Variance(θ)` | `μ + θ σ²` |
//! | `StandardDeviation(θ)` | `μ + θ σ` |
//! | `SemiVariance(θ)` | `μ + θ E[(X - μ)₊²]` |
//! | `Exponential(k)` | `log E[e^(kX)] / k`, the zero-utility premium |
//! | `Esscher(h)` | `E[X e^(hX)] / E[e^(hX)]` |
//! | `Dutch(θ)` | `μ + θ E[(X - μ)₊]` |
//! | `Fischer { theta, q }` | `μ + θ E[(X - μ)₊^q]^(1/q)` |
//! | `Var(p)` | the lower `p` quantile |
//!
//! Exponential and Esscher are computed with the largest exponent factored
//! out, so large losses do not overflow. Semi-variance follows `aggregate`
//! (and Artzner) in adding the semi-variance itself, not its root.

use prospicio_core::{Error, Result};
use prospicio_math::roots::bisect_log;

/// A premium principle and its loading. See the [module](self).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Principle {
    ExpectedValue(f64),
    Variance(f64),
    StandardDeviation(f64),
    SemiVariance(f64),
    Exponential(f64),
    Esscher(f64),
    Dutch(f64),
    Fischer { theta: f64, q: f64 },
    Var(f64),
}

/// A principle without its loading, for [`calibrate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    ExpectedValue,
    Variance,
    StandardDeviation,
    SemiVariance,
    Exponential,
    Esscher,
    Dutch,
    /// With this power `q >= 1`.
    Fischer {
        q: f64,
    },
    Var,
}

struct Moments {
    mean: f64,
    var: f64,
}

fn moments(values: &[f64], probs: &[f64]) -> Moments {
    let mean: f64 = values.iter().zip(probs).map(|(x, p)| x * p).sum();
    let var: f64 = values
        .iter()
        .zip(probs)
        .map(|(x, p)| (x - mean).powi(2) * p)
        .sum();
    Moments { mean, var }
}

fn check(values: &[f64], probs: &[f64]) -> Result<()> {
    if values.is_empty() || values.len() != probs.len() {
        return Err(Error::Data("give one probability per value".into()));
    }
    if values.iter().chain(probs).any(|v| !v.is_finite()) || probs.iter().any(|p| *p < 0.0) {
        return Err(Error::Data(
            "values and probabilities must be finite, probabilities non-negative".into(),
        ));
    }
    Ok(())
}

/// `log E[e^(kX)]` with the largest exponent factored out.
fn log_mgf(values: &[f64], probs: &[f64], k: f64) -> f64 {
    let top = values
        .iter()
        .zip(probs)
        .filter(|(_, p)| **p > 0.0)
        .map(|(x, _)| k * x)
        .fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = values
        .iter()
        .zip(probs)
        .map(|(x, p)| p * (k * x - top).exp())
        .sum();
    top + sum.ln()
}

impl Principle {
    /// The premium of the discrete distribution with `values` and their
    /// `probs` (summing to 1).
    ///
    /// ```
    /// use prospicio_pricing::classical::Principle;
    ///
    /// let x = [0.0, 10.0];
    /// let p = [0.5, 0.5];
    /// assert_eq!(Principle::StandardDeviation(0.2).premium(&x, &p).unwrap(), 6.0);
    /// assert_eq!(Principle::Dutch(1.0).premium(&x, &p).unwrap(), 7.5);
    /// ```
    pub fn premium(&self, values: &[f64], probs: &[f64]) -> Result<f64> {
        check(values, probs)?;
        let Moments { mean, var } = moments(values, probs);
        let upper = |q: f64| -> f64 {
            values
                .iter()
                .zip(probs)
                .map(|(x, p)| (x - mean).max(0.0).powf(q) * p)
                .sum()
        };
        Ok(match *self {
            Self::ExpectedValue(t) => (1.0 + t) * mean,
            Self::Variance(t) => mean + t * var,
            Self::StandardDeviation(t) => mean + t * var.sqrt(),
            Self::SemiVariance(t) => mean + t * upper(2.0),
            Self::Exponential(k) => {
                if k == 0.0 {
                    mean
                } else {
                    log_mgf(values, probs, k) / k
                }
            }
            Self::Esscher(h) => {
                let top = values
                    .iter()
                    .zip(probs)
                    .filter(|(_, p)| **p > 0.0)
                    .map(|(x, _)| h * x)
                    .fold(f64::NEG_INFINITY, f64::max);
                let (mut num, mut den) = (0.0, 0.0);
                for (x, p) in values.iter().zip(probs) {
                    let w = p * (h * x - top).exp();
                    num += x * w;
                    den += w;
                }
                num / den
            }
            Self::Dutch(t) => mean + t * upper(1.0),
            Self::Fischer { theta, q } => {
                if q < 1.0 {
                    return Err(Error::InvalidParameter {
                        name: "q",
                        value: q,
                        reason: "must be at least 1",
                    });
                }
                mean + theta * upper(q).powf(1.0 / q)
            }
            Self::Var(level) => {
                if !(0.0..=1.0).contains(&level) {
                    return Err(Error::InvalidParameter {
                        name: "p",
                        value: level,
                        reason: "must be in [0, 1]",
                    });
                }
                let mut order: Vec<usize> = (0..values.len()).collect();
                order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
                let mut cum = 0.0;
                let mut q = values[order[order.len() - 1]];
                for &i in &order {
                    cum += probs[i];
                    if cum >= level && probs[i] > 0.0 {
                        q = values[i];
                        break;
                    }
                }
                q
            }
        })
    }
}

/// The loading of `kind` that gives `premium` on the discrete distribution:
/// closed form for the expected value, variance, standard deviation,
/// semi-variance, Dutch and Fischer principles (linear in their loading);
/// by bisection for the exponential and Esscher (increasing in theirs);
/// `P(X <= premium)` for VaR. The premium must exceed the mean (be at least
/// the smallest value, for VaR).
///
/// ```
/// use prospicio_pricing::classical::{calibrate, Kind, Principle};
///
/// let x = [22.0, 28.0, 36.0, 40.0, 55.0, 65.0, 100.0];
/// let p = [0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1];
/// let esscher = calibrate(Kind::Esscher, &x, &p, 53.565217391304344).unwrap();
/// assert!((esscher.premium(&x, &p).unwrap() - 53.565217391304344).abs() < 1e-9);
/// ```
pub fn calibrate(kind: Kind, values: &[f64], probs: &[f64], premium: f64) -> Result<Principle> {
    check(values, probs)?;
    let Moments { mean, var } = moments(values, probs);
    if kind != Kind::Var && premium.partial_cmp(&mean) != Some(std::cmp::Ordering::Greater) {
        return Err(Error::Data(format!(
            "the premium must exceed the mean {mean}, not {premium}"
        )));
    }
    let upper = |q: f64| -> f64 {
        values
            .iter()
            .zip(probs)
            .map(|(x, p)| (x - mean).max(0.0).powf(q) * p)
            .sum()
    };
    let load = premium - mean;
    let positive = |d: f64| -> Result<f64> {
        if d > 0.0 {
            Ok(load / d)
        } else {
            Err(Error::Data("a constant loss cannot carry a loading".into()))
        }
    };
    Ok(match kind {
        Kind::ExpectedValue => Principle::ExpectedValue(premium / mean - 1.0),
        Kind::Variance => Principle::Variance(positive(var)?),
        Kind::StandardDeviation => Principle::StandardDeviation(positive(var.sqrt())?),
        Kind::SemiVariance => Principle::SemiVariance(positive(upper(2.0))?),
        Kind::Dutch => Principle::Dutch(positive(upper(1.0))?),
        Kind::Fischer { q } => {
            if q < 1.0 {
                return Err(Error::InvalidParameter {
                    name: "q",
                    value: q,
                    reason: "must be at least 1",
                });
            }
            Principle::Fischer {
                theta: positive(upper(q).powf(1.0 / q))?,
                q,
            }
        }
        Kind::Var => {
            let mut order: Vec<usize> = (0..values.len()).collect();
            order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
            let cum: f64 = order
                .iter()
                .filter(|&&i| values[i] <= premium)
                .map(|&i| probs[i])
                .sum();
            Principle::Var(cum.min(1.0))
        }
        Kind::Exponential | Kind::Esscher => {
            let make = |k: f64| match kind {
                Kind::Exponential => Principle::Exponential(k),
                _ => Principle::Esscher(k),
            };
            let max = values
                .iter()
                .zip(probs)
                .filter(|(_, p)| **p > 0.0)
                .map(|(x, _)| *x)
                .fold(f64::NEG_INFINITY, f64::max);
            if premium.partial_cmp(&max) != Some(std::cmp::Ordering::Less) {
                return Err(Error::Data(format!(
                    "the premium must be below the maximum {max}"
                )));
            }
            let below = |k: f64| make(k).premium(values, probs).map_or(true, |v| v < premium);
            let mut hi = 1.0 / (max - mean).abs().max(1e-300);
            while below(hi) {
                hi *= 2.0;
                if !hi.is_finite() {
                    return Err(Error::Data("no loading reaches the premium".into()));
                }
            }
            make(bisect_log(hi * 1e-30, hi, below))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: [f64; 7] = [22.0, 28.0, 36.0, 40.0, 55.0, 65.0, 100.0];
    const P: [f64; 7] = [0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1];

    #[test]
    fn every_principle_calibrates_to_the_premium() {
        let target = 53.565217391304344;
        for kind in [
            Kind::ExpectedValue,
            Kind::Variance,
            Kind::StandardDeviation,
            Kind::SemiVariance,
            Kind::Exponential,
            Kind::Esscher,
            Kind::Dutch,
            Kind::Fischer { q: 2.0 },
        ] {
            let p = calibrate(kind, &X, &P, target).unwrap();
            assert!(
                (p.premium(&X, &P).unwrap() - target).abs() < 1e-9,
                "{kind:?}: {p:?}"
            );
        }
        // VaR at the calibrated level is the quantile at or below the premium.
        let v = calibrate(Kind::Var, &X, &P, 53.0).unwrap();
        assert_eq!(v, Principle::Var(0.7000000000000001));
        assert_eq!(v.premium(&X, &P).unwrap(), 40.0);
        assert!(calibrate(Kind::Variance, &X, &P, 40.0).is_err());
    }

    #[test]
    fn closed_forms() {
        let (x, p) = ([0.0, 10.0], [0.5, 0.5]);
        assert_eq!(Principle::Variance(0.1).premium(&x, &p).unwrap(), 7.5);
        assert_eq!(Principle::SemiVariance(0.1).premium(&x, &p).unwrap(), 6.25);
        // Fischer with q = 1 is Dutch.
        assert_eq!(
            Principle::Fischer { theta: 0.4, q: 1.0 }
                .premium(&x, &p)
                .unwrap(),
            Principle::Dutch(0.4).premium(&x, &p).unwrap()
        );
        // Exponential: log((1 + e^k) / 2) / k; large k does not overflow.
        let k = 0.3f64;
        let want = ((1.0 + (10.0 * k).exp()) / 2.0).ln() / k;
        assert!((Principle::Exponential(k).premium(&x, &p).unwrap() - want).abs() < 1e-12);
        let huge = Principle::Exponential(1000.0).premium(&x, &p).unwrap();
        assert!(huge.is_finite() && huge < 10.0 && huge > 9.99);
        // Esscher: 10 e^(10h) / (1 + e^(10h)).
        let h = 0.2f64;
        let want = 10.0 * (10.0 * h).exp() / (1.0 + (10.0 * h).exp());
        assert!((Principle::Esscher(h).premium(&x, &p).unwrap() - want).abs() < 1e-12);
    }
}
