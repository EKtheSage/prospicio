//! A severity conditioned on a window, and splicing by mixing such pieces.

use std::sync::Arc;

use prospicio_core::{Error, Result};

use crate::dist::{Dist, SeverityDist};
use crate::distribution::{Distribution, bisect_quantile, check_probability};
use crate::mixture::Mixture;
use crate::severity::Severity;

/// A severity conditioned on `lower < X <= upper`: its distribution
/// function is `(F(x) - F(lower)) / (F(upper) - F(lower))` on the window,
/// as `aggregate`'s `sev_lb` and `sev_ub` with `sev_conditional`. An
/// upper bound alone truncates a tail; a lower bound alone gives the
/// losses above a threshold, from zero (not excess of it).
///
/// Splicing a body and a tail is a [`Mixture`] of truncated pieces with
/// disjoint windows: [`Truncated::splice`].
///
/// Layer moments come from the inner severity's: below `lower` the
/// survival is 1, on the window it is `(S(x) - S(upper)) / p` with
/// `p = S(lower) - S(upper)`, so `∫ S_T` over a layer is a length plus an
/// inner layer less a rectangle, all over `p`.
///
/// ```
/// use prospicio_prob::{Distribution, Lognormal, Severity, SeverityDist, Truncated};
///
/// let ln = SeverityDist::try_from(prospicio_prob::Dist::from(
///     Lognormal::from_mean_cv(1000.0, 1.0).unwrap(),
/// ))
/// .unwrap();
/// let t = Truncated::new(ln, 0.0, 5000.0).unwrap();
/// assert_eq!(t.cdf(5000.0), 1.0);
/// assert!(t.mean() < 1000.0);
/// assert!((t.lev(f64::INFINITY) - t.mean()).abs() < 1e-9);
/// ```
#[derive(Debug, Clone)]
pub struct Truncated {
    inner: SeverityDist,
    lower: f64,
    upper: f64,
    /// `F(lower)`, `S(lower)` and `S(upper)` of the inner severity.
    f_lower: f64,
    s_lower: f64,
    s_upper: f64,
    /// `P(lower < X <= upper)`.
    p: f64,
}

impl Truncated {
    /// `inner` conditioned on `lower < X <= upper`, for
    /// `0 <= lower < upper <= ∞`; fails when the window has no probability.
    pub fn new(inner: SeverityDist, lower: f64, upper: f64) -> Result<Self> {
        if !(lower.is_finite() && lower >= 0.0) {
            return Err(Error::InvalidParameter {
                name: "lower",
                value: lower,
                reason: "must be finite and non-negative",
            });
        }
        if upper.is_nan() || upper <= lower {
            return Err(Error::InvalidParameter {
                name: "upper",
                value: upper,
                reason: "must be above lower",
            });
        }
        let f_lower = inner.cdf(lower);
        let s_lower = inner.survival(lower);
        let s_upper = if upper == f64::INFINITY {
            0.0
        } else {
            inner.survival(upper)
        };
        let p = if f_lower < 0.5 && upper < f64::INFINITY {
            inner.cdf(upper) - f_lower
        } else {
            s_lower - s_upper
        };
        if p.is_nan() || p <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "upper",
                value: upper,
                reason: "the window (lower, upper] must have positive probability",
            });
        }
        Ok(Self {
            inner,
            lower,
            upper,
            f_lower,
            s_lower,
            s_upper,
            p,
        })
    }

    /// A splice: piece `i` is component `i` conditioned on
    /// `(breaks[i], breaks[i + 1]]`, with weight `weights[i]`. `breaks` has
    /// one more entry than there are components, starts at 0 and
    /// increases; it may end at infinity.
    ///
    /// ```
    /// use prospicio_prob::{Dist, Distribution, Lognormal, Pareto, SeverityDist, Truncated};
    ///
    /// let body = SeverityDist::try_from(Dist::from(Lognormal::from_mean_cv(50.0, 1.0).unwrap())).unwrap();
    /// let tail = SeverityDist::try_from(Dist::from(Pareto::new(100.0, 1.8).unwrap())).unwrap();
    /// let s = Truncated::splice(vec![(0.9, body), (0.1, tail)], &[0.0, 100.0, f64::INFINITY]).unwrap();
    /// assert!((s.cdf(100.0) - 0.9).abs() < 1e-12);
    /// ```
    pub fn splice(parts: Vec<(f64, SeverityDist)>, breaks: &[f64]) -> Result<Mixture> {
        if breaks.len() != parts.len() + 1 {
            return Err(Error::InvalidParameter {
                name: "breaks",
                value: breaks.len() as f64,
                reason: "must have one more entry than there are components",
            });
        }
        if breaks[0] != 0.0 {
            return Err(Error::InvalidParameter {
                name: "breaks",
                value: breaks[0],
                reason: "must start at 0",
            });
        }
        let pieces = parts
            .into_iter()
            .zip(breaks.windows(2))
            .map(|((w, d), b)| {
                let t = Self::new(d, b[0], b[1])?;
                let s = SeverityDist::try_from(Dist::from(t)).expect("a truncated severity");
                Ok((w, s))
            })
            .collect::<Result<Vec<_>>>()?;
        Mixture::from_dists(pieces)
    }

    /// The severity before conditioning.
    pub fn inner(&self) -> &SeverityDist {
        &self.inner
    }

    /// Lower end of the window (excluded).
    pub fn lower(&self) -> f64 {
        self.lower
    }

    /// Upper end of the window (included), possibly infinite.
    pub fn upper(&self) -> f64 {
        self.upper
    }

    /// `P(lower < X <= upper)` under the inner severity.
    pub fn probability(&self) -> f64 {
        self.p
    }

    /// `[a', b']`: `[a, b]` clamped to the window.
    fn clamp(&self, a: f64, b: f64) -> (f64, f64) {
        (
            a.clamp(self.lower, self.upper),
            b.clamp(self.lower, self.upper),
        )
    }
}

impl From<Truncated> for Dist {
    fn from(t: Truncated) -> Self {
        Self::Truncated(Arc::new(t))
    }
}

impl Distribution for Truncated {
    fn mean(&self) -> f64 {
        self.layer(f64::INFINITY, 0.0)
    }

    fn variance(&self) -> f64 {
        let m = self.mean();
        self.layer_second_moment(f64::INFINITY, 0.0) - m * m
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= self.lower {
            return 0.0;
        }
        if x >= self.upper {
            return 1.0;
        }
        let v = if self.f_lower < 0.5 {
            self.inner.cdf(x) - self.f_lower
        } else {
            self.s_lower - self.inner.survival(x)
        };
        (v / self.p).clamp(0.0, 1.0)
    }

    fn survival(&self, x: f64) -> f64 {
        if x <= self.lower {
            return 1.0;
        }
        if x >= self.upper {
            return 0.0;
        }
        ((self.inner.survival(x) - self.s_upper) / self.p).clamp(0.0, 1.0)
    }

    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        if p == 0.0 {
            return Ok(self.lower);
        }
        if p == 1.0 {
            return Ok(self.upper);
        }
        let start = if self.upper.is_finite() {
            self.upper
        } else {
            (2.0 * self.lower)
                .max(self.inner.quantile(0.5)?)
                .max(1e-300)
        };
        Ok(bisect_quantile(
            p,
            self.lower,
            start,
            self.upper,
            |x| self.cdf(x),
            |x| self.survival(x),
        ))
    }

    fn is_parallel_safe(&self) -> bool {
        self.inner.is_parallel_safe()
    }
}

impl Severity for Truncated {
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

    /// `(min(b, lower) - a)⁺ + (∫_{a'}^{b'} S - S(upper)(b' - a')) / p`.
    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        let b = a + limit;
        let below = (b.min(self.lower) - a).max(0.0);
        let (lo, hi) = self.clamp(a, b);
        if hi <= lo {
            return below;
        }
        let mut mid = self.inner.layer(hi - lo, lo);
        if self.s_upper > 0.0 {
            mid -= self.s_upper * (hi - lo);
        }
        below + mid.max(0.0) / self.p
    }

    /// `2 ∫_a^b (x - a) S_T(x) dx`: `c²` for the part `c` below `lower`,
    /// and on the window the inner layer's second moment and mean, shifted
    /// from `a'` to `a`, less the rectangle under `S(upper)`.
    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        let b = a + limit;
        let c = (b.min(self.lower) - a).max(0.0);
        let (lo, hi) = self.clamp(a, b);
        if hi <= lo {
            return c * c;
        }
        let w = hi - lo;
        let mut mid = self.inner.layer_second_moment(w, lo);
        if mid == f64::INFINITY {
            return f64::INFINITY;
        }
        if lo > a {
            mid += 2.0 * (lo - a) * self.inner.layer(w, lo);
        }
        if self.s_upper > 0.0 {
            mid -= self.s_upper * ((hi - a).powi(2) - (lo - a).powi(2));
        }
        c * c + mid.max(0.0) / self.p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Gamma, Lognormal, Pareto};

    fn sev(d: impl Into<Dist>) -> SeverityDist {
        SeverityDist::try_from(d.into()).unwrap()
    }

    /// `∫ S_T` and `2 ∫ (x - a) S_T` over a layer by the midpoint rule.
    fn numeric(t: &Truncated, a: f64, b: f64) -> (f64, f64) {
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
        for (lo, hi) in [(0.0, 3000.0), (400.0, 2500.0), (800.0, f64::INFINITY)] {
            let t = Truncated::new(g.clone(), lo, hi).unwrap();
            for (a, b) in [
                (0.0, 300.0),
                (200.0, 1200.0),
                (1000.0, 2000.0),
                (0.0, 6000.0),
            ] {
                let (m1, m2) = numeric(&t, a, b);
                let got1 = t.layer(b - a, a);
                let got2 = t.layer_second_moment(b - a, a);
                assert!((got1 - m1).abs() < 1e-6 * m1.max(1.0), "{lo} {hi} {a} {b}");
                assert!((got2 - m2).abs() < 1e-6 * m2.max(1.0), "{lo} {hi} {a} {b}");
            }
            let total = t.lev(1500.0) + t.stop_loss(1500.0);
            assert!((total / t.mean() - 1.0).abs() < 1e-12);
            for p in [0.01, 0.5, 0.99] {
                let x = t.quantile(p).unwrap();
                assert!((t.cdf(x) - p).abs() < 1e-12, "{lo} {hi} {p}");
            }
        }
    }

    #[test]
    fn splice_puts_the_weights_on_the_pieces() {
        let body = sev(Lognormal::from_mean_cv(50.0, 1.0).unwrap());
        let tail = sev(Pareto::new(100.0, 2.5).unwrap());
        let s = Truncated::splice(vec![(0.9, body), (0.1, tail)], &[0.0, 100.0, f64::INFINITY])
            .unwrap();
        assert!((s.cdf(100.0) - 0.9).abs() < 1e-12);
        assert!((s.survival(400.0) - 0.1 * 0.25f64.powf(2.5)).abs() < 1e-14);
        assert!(s.quantile(0.95).unwrap() > 100.0);
        assert!(Truncated::splice(vec![], &[0.0]).is_err());
    }

    #[test]
    fn rejects_bad_windows() {
        let g = sev(Gamma::new(2.0, 500.0).unwrap());
        assert!(Truncated::new(g.clone(), 10.0, 10.0).is_err());
        assert!(Truncated::new(g.clone(), -1.0, 10.0).is_err());
        let p = sev(Pareto::new(100.0, 2.0).unwrap());
        assert!(Truncated::new(p, 0.0, 50.0).is_err());
    }
}
