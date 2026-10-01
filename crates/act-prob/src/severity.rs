//! Operations on a loss severity that are exact for parametric (and later
//! discretized) distributions: limited expected values, stop-loss and layers.
//!
//! `Sampled` does not implement [`Severity`]; an estimate from draws is
//! [`crate::Empirical::mean_of`]. See `docs/design/distributions.md`.

use crate::distribution::Distribution;

/// A non-negative loss severity with exact limited moments.
///
/// The three methods are related by `lev(d) + stop_loss(d) = mean()` and
/// `layer(l, a) = stop_loss(a) - stop_loss(a + l)`. Implementations should
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
}
