//! A severity scaled and shifted, `Y = scale · X + shift`: `aggregate`'s
//! `10 * lognorm 1 + 20`, with the continuous form kept (#318).

use std::sync::Arc;

use prospicio_core::{Error, Result, StreamRng};

use crate::dist::{Dist, SeverityDist};
use crate::distribution::{Distribution, check_probability};
use crate::severity::Severity;

/// `Y = scale · X + shift` for a severity `X`, `scale > 0` and
/// `shift >= 0`, so that `Y` is a loss too.
///
/// Its distribution function is `F((y − shift) / scale)`, and its layer
/// moments come from the inner severity's: below `shift` the survival is 1,
/// and above it `S_Y(y) = S_X((y − shift) / scale)`, so a layer of `Y` is a
/// length plus `scale` times a layer of `X`.
///
/// ```
/// use prospicio_prob::{Affine, Dist, Distribution, Lognormal, Severity, SeverityDist};
///
/// let x = Lognormal::new(0.0, 1.0).unwrap();
/// let y = Affine::new(SeverityDist::try_from(Dist::from(x)).unwrap(), 10.0, 20.0).unwrap();
/// assert!((y.mean() - (10.0 * x.mean() + 20.0)).abs() < 1e-9);
/// assert!((y.layer(30.0, 40.0) - 10.0 * x.layer(3.0, 2.0)).abs() < 1e-9);
/// ```
#[derive(Debug, Clone)]
pub struct Affine {
    inner: SeverityDist,
    scale: f64,
    shift: f64,
}

impl Affine {
    /// `scale · X + shift`, `scale` finite and positive, `shift` finite and
    /// non-negative.
    pub fn new(inner: SeverityDist, scale: f64, shift: f64) -> Result<Self> {
        if !(scale.is_finite() && scale > 0.0) {
            return Err(Error::InvalidParameter {
                name: "scale",
                value: scale,
                reason: "must be finite and positive",
            });
        }
        if !(shift.is_finite() && shift >= 0.0) {
            return Err(Error::InvalidParameter {
                name: "shift",
                value: shift,
                reason: "must be finite and non-negative",
            });
        }
        Ok(Self {
            inner,
            scale,
            shift,
        })
    }

    /// The severity before scaling and shifting.
    pub fn inner(&self) -> &SeverityDist {
        &self.inner
    }

    /// The scale.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// The shift.
    pub fn shift(&self) -> f64 {
        self.shift
    }

    /// The part of `[a, b]` below the shift, where `S_Y = 1`, and the part
    /// above it as `(lo, hi)`.
    fn split(&self, a: f64, b: f64) -> (f64, f64, f64) {
        let below = (b.min(self.shift) - a).max(0.0);
        (below, a.max(self.shift), b)
    }
}

impl From<Affine> for Dist {
    fn from(t: Affine) -> Self {
        Self::Affine(Arc::new(t))
    }
}

impl Distribution for Affine {
    fn mean(&self) -> f64 {
        self.scale * self.inner.mean() + self.shift
    }

    fn variance(&self) -> f64 {
        self.scale * self.scale * self.inner.variance()
    }

    fn cdf(&self, x: f64) -> f64 {
        self.inner.cdf((x - self.shift) / self.scale)
    }

    fn survival(&self, x: f64) -> f64 {
        self.inner.survival((x - self.shift) / self.scale)
    }

    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        Ok(self.scale * self.inner.quantile(p)? + self.shift)
    }

    fn sample(&self, rng: &mut StreamRng, n: usize) -> Vec<f64> {
        self.inner
            .sample(rng, n)
            .into_iter()
            .map(|x| self.scale * x + self.shift)
            .collect()
    }

    fn is_parallel_safe(&self) -> bool {
        self.inner.is_parallel_safe()
    }
}

impl Severity for Affine {
    fn lev(&self, limit: f64) -> f64 {
        if limit <= 0.0 {
            return limit;
        }
        self.layer(limit, 0.0)
    }

    fn stop_loss(&self, retention: f64) -> f64 {
        if retention <= 0.0 {
            return self.mean() - retention;
        }
        self.layer(f64::INFINITY, retention)
    }

    /// `(min(b, shift) − a)⁺ + scale · layer_X((hi − lo) / scale, (lo − shift) / scale)`.
    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        let (below, lo, hi) = self.split(a, a + limit);
        if hi <= lo {
            return below;
        }
        let s = self.scale;
        below + s * self.inner.layer((hi - lo) / s, (lo - self.shift) / s)
    }

    /// `2 ∫_a^b (y − a) S_Y(y) dy`: `c²` below the shift, and above it
    /// `s² m₂ + 2 (lo − a) s m₁` with `m₁`, `m₂` the inner layer's moments
    /// over `y = shift + s t`.
    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        let (c, lo, hi) = self.split(a, a + limit);
        if hi <= lo {
            return c * c;
        }
        let s = self.scale;
        let (w, t0) = ((hi - lo) / s, (lo - self.shift) / s);
        let m2 = self.inner.layer_second_moment(w, t0);
        if m2 == f64::INFINITY {
            return f64::INFINITY;
        }
        let mut v = c * c + s * s * m2;
        if lo > a {
            v += 2.0 * (lo - a) * s * self.inner.layer(w, t0);
        }
        v
    }

    /// `E[(sX + b)³] = s³ m₃ + 3 s² b m₂ + 3 s b² m₁ + b³`.
    fn third_moment(&self) -> Option<f64> {
        let m3 = self.inner.third_moment()?;
        let (s, b) = (self.scale, self.shift);
        let m1 = self.inner.mean();
        let m2 = self.inner.variance() + m1 * m1;
        Some(s.powi(3) * m3 + 3.0 * s * s * b * m2 + 3.0 * s * b * b * m1 + b.powi(3))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Gamma, Lognormal, Pareto};

    fn sev(d: impl Into<Dist>) -> SeverityDist {
        SeverityDist::try_from(d.into()).unwrap()
    }

    /// `∫ S_Y` and `2 ∫ (y − a) S_Y` over a layer by the midpoint rule.
    fn numeric(t: &Affine, a: f64, b: f64) -> (f64, f64) {
        let n = 200_000;
        let h = (b - a) / f64::from(n);
        let (mut m1, mut m2) = (0.0, 0.0);
        for i in 0..n {
            let x = a + (f64::from(i) + 0.5) * h;
            let s = t.survival(x);
            m1 += s * h;
            m2 += 2.0 * (x - a) * s * h;
        }
        (m1, m2)
    }

    #[test]
    fn layers_match_integration() {
        let g = sev(Gamma::new(2.0, 500.0).unwrap());
        for (s, b) in [(1.0, 0.0), (10.0, 20.0), (0.5, 800.0)] {
            let t = Affine::new(g.clone(), s, b).unwrap();
            for (a, top) in [
                (0.0, 300.0),
                (200.0, 1200.0),
                (700.0, 2000.0),
                (0.0, 9000.0),
            ] {
                let (m1, m2) = numeric(&t, a, top);
                let got1 = t.layer(top - a, a);
                let got2 = t.layer_second_moment(top - a, a);
                assert!((got1 - m1).abs() < 1e-6 * m1.max(1.0), "{s} {b} {a} {top}");
                assert!((got2 - m2).abs() < 1e-6 * m2.max(1.0), "{s} {b} {a} {top}");
            }
            assert!((t.lev(1500.0) + t.stop_loss(1500.0) - t.mean()).abs() < 1e-9 * t.mean());
            let m2 = t.layer_second_moment(f64::INFINITY, 0.0);
            assert!((m2 - t.variance() - t.mean().powi(2)).abs() < 1e-9 * m2);
            for p in [0.01, 0.5, 0.99] {
                let y = t.quantile(p).unwrap();
                assert!((t.cdf(y) - p).abs() < 1e-12, "{s} {b} {p}");
            }
        }
    }

    #[test]
    fn moments_and_the_shift() {
        let x = Lognormal::new(0.0, 1.0).unwrap();
        let y = Affine::new(sev(x), 10.0, 20.0).unwrap();
        assert!((y.mean() - (10.0 * x.mean() + 20.0)).abs() < 1e-12);
        assert!((y.variance() - 100.0 * x.variance()).abs() < 1e-9);
        // Below the shift the loss is sure.
        assert_eq!(y.cdf(20.0), 0.0);
        assert_eq!(y.lev(15.0), 15.0);
        assert!((y.layer(10.0, 15.0) - (5.0 + 10.0 * x.lev(0.5))).abs() < 1e-12);
        let m3 = x.third_moment().unwrap();
        let want = 1000.0 * m3
            + 3.0 * 100.0 * 20.0 * (x.variance() + x.mean().powi(2))
            + 3.0 * 10.0 * 400.0 * x.mean()
            + 8000.0;
        assert!((y.third_moment().unwrap() / want - 1.0).abs() < 1e-12);
        let mut r1 = StreamRng::new(3, 0);
        let mut r2 = StreamRng::new(3, 0);
        let d = y.sample(&mut r1, 4);
        let e = x.sample(&mut r2, 4);
        for (a, b) in d.iter().zip(&e) {
            assert!((a - (10.0 * b + 20.0)).abs() < 1e-9);
        }
        // An infinite variance stays infinite.
        let p = Affine::new(sev(Pareto::new(100.0, 1.5).unwrap()), 2.0, 5.0).unwrap();
        assert_eq!(p.layer_second_moment(f64::INFINITY, 0.0), f64::INFINITY);
    }

    #[test]
    fn rejects_bad_parameters() {
        let g = sev(Gamma::new(2.0, 500.0).unwrap());
        assert!(Affine::new(g.clone(), 0.0, 1.0).is_err());
        assert!(Affine::new(g.clone(), f64::INFINITY, 1.0).is_err());
        assert!(Affine::new(g.clone(), 1.0, -1.0).is_err());
        assert!(Affine::new(g, 1.0, f64::NAN).is_err());
    }
}
