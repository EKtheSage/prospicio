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
