//! Operations on a loss severity that are exact for parametric (and later
//! discretized) distributions: limited expected values, stop-loss and layers.
//!
//! `Sampled` does not implement [`Severity`]; an estimate from draws is
//! [`crate::Empirical::mean_of`]. See `docs/design/distributions.md`.

use crate::distribution::Distribution;

/// A non-negative loss severity with exact limited moments.
///
/// The first three methods are related by `lev(d) + stop_loss(d) = mean()`
/// and `layer(l, a) = stop_loss(a) - stop_loss(a + l)`. Implementations should
/// compute `stop_loss` directly rather than as `mean() - lev(d)`, which
/// loses all precision for retentions far in the tail.
pub trait Severity: Distribution {
    /// Limited expected value `E[min(X, limit)]`.
    ///
    /// `limit <= 0` gives `limit` (the loss is never below 0) and
    /// `limit = +inf` gives the mean.
    fn lev(&self, limit: f64) -> f64;

    /// Expected excess over a retention, `E[max(X - retention, 0)]`.
    fn stop_loss(&self, retention: f64) -> f64 {
        self.mean() - self.lev(retention)
    }

    /// Expected loss to the layer `limit` xs `attachment`,
    /// `E[min(max(X - attachment, 0), limit)]`.
    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        self.stop_loss(attachment) - self.stop_loss(attachment + limit)
    }

    /// Second moment of the loss to the layer `limit` xs `attachment`,
    /// `E[min(max(X - attachment, 0), limit)^2]`. `limit = +inf` gives the
    /// unlimited layer (infinite if the second moment is).
    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64;

    /// Variance of the loss to the layer `limit` xs `attachment`.
    fn layer_variance(&self, limit: f64, attachment: f64) -> f64 {
        let m = self.layer(limit, attachment);
        self.layer_second_moment(limit, attachment) - m * m
    }
}

impl<T: Severity + ?Sized> Severity for Box<T> {
    fn lev(&self, limit: f64) -> f64 {
        (**self).lev(limit)
    }

    fn stop_loss(&self, retention: f64) -> f64 {
        (**self).stop_loss(retention)
    }

    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        (**self).layer(limit, attachment)
    }

    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        (**self).layer_second_moment(limit, attachment)
    }

    fn layer_variance(&self, limit: f64, attachment: f64) -> f64 {
        (**self).layer_variance(limit, attachment)
    }
}

/// Limited moments of a family with closed forms, from which
/// [`severity_from_moments!`] builds the [`Severity`] methods.
///
/// Layer integrals `∫_a^b x^(j-1) S(x) dx` are differences of limited
/// moments `E[min(X, u)^j]` below the pivot (where those are small) and of
/// tail moments `E[X^j; X > u] - u^j S(u)` above it (where limited moments
/// are close to the full moment and would cancel). Without a finite `j`-th
/// moment there is no tail form and limited moments are used throughout.
pub(crate) trait Moments: Distribution {
    /// `E[min(X, u)^j]` for `u >= 0`, `j` in 1 and 2; the full moment
    /// (possibly infinite) at `u = ∞`.
    fn limited(&self, j: i32, u: f64) -> f64;

    /// `E[X^j; X > u] - u^j S(u)` for `u >= 0`, called only when the `j`-th
    /// moment is finite; zero at `u = ∞`.
    fn tail(&self, j: i32, u: f64) -> f64;

    /// Where the layer integral switches from limited to tail moments, or
    /// `None` when the `j`-th moment is infinite.
    fn pivot(&self, j: i32) -> Option<f64>;

    /// `∫_a^b x^(j-1) S(x) dx` for `0 <= a <= b <= ∞`.
    fn partial(&self, j: i32, a: f64, b: f64) -> f64 {
        if b <= a {
            return 0.0;
        }
        let jf = f64::from(j);
        let lim = |u: f64| self.limited(j, u);
        let tail = |u: f64| self.tail(j, u);
        match self.pivot(j) {
            None => (lim(b) - lim(a)) / jf,
            Some(m) if b <= m => (lim(b) - lim(a)) / jf,
            Some(m) if a >= m => (tail(a) - tail(b)) / jf,
            Some(m) => (lim(m) - lim(a) + tail(m) - tail(b)) / jf,
        }
    }
}

/// Implements [`Severity`] for a type that implements [`Moments`].
macro_rules! severity_from_moments {
    ($ty:ty) => {
        impl $crate::severity::Severity for $ty {
            fn lev(&self, limit: f64) -> f64 {
                use $crate::severity::Moments;
                if limit <= 0.0 {
                    return limit;
                }
                self.limited(1, limit)
            }

            /// From the tail, so it does not cancel against the mean;
            /// infinite with the mean.
            fn stop_loss(&self, retention: f64) -> f64 {
                use $crate::severity::Moments;
                if retention <= 0.0 {
                    return $crate::Distribution::mean(self) - retention;
                }
                if self.pivot(1).is_none() {
                    return f64::INFINITY;
                }
                self.tail(1, retention)
            }

            /// `∫_a^(a+l) S(x) dx`.
            fn layer(&self, limit: f64, attachment: f64) -> f64 {
                use $crate::severity::Moments;
                let a = attachment.max(0.0);
                self.partial(1, a, a + limit)
            }

            /// `2 ∫_a^b (x - a) S(x) dx` with `b = a + limit`.
            fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
                use $crate::severity::Moments;
                let a = attachment.max(0.0);
                let b = a + limit;
                let p1 = self.partial(1, a, b);
                if p1 == 0.0 {
                    return 0.0;
                }
                let p2 = self.partial(2, a, b);
                if p2 == f64::INFINITY {
                    return f64::INFINITY;
                }
                (2.0 * (p2 - a * p1)).max(0.0)
            }
        }
    };
}
pub(crate) use severity_from_moments;
