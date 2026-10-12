//! Statistics of a per-occurrence layer on a severity grid: the numbers on a
//! reinsurance submission (`docs/design/aggregate.md`), as the Python package
//! `aggregate`'s `reins_stats_df`.
//!
//! For a layer `share × min(max(X − a, 0), l)` on the grid of the severity
//! `X` and a claim count `N`:
//!
//! - the chances that a claim reaches the layer, `P(X > a)`, and exhausts
//!   it, `P(X ≥ a + l)` (attach exclusive, detach inclusive, as
//!   `aggregate`), and that the year has a loss to it,
//!   `1 − G_N(F(a))`, the occurrence exceedance at `a` ([`crate::pml`]);
//! - the excess count, the count thinned to the claims that reach the
//!   layer, by its cumulants (no cancellation at large counts);
//! - the layer severity given a loss to the layer;
//! - the annual ceded loss at the share ([`crate::moments`]), and the loss
//!   on line, its mean over the placed limit `share × l`.

use prospicio_core::{Error, Result};
use prospicio_prob::{Counting, Distribution, Grid, Severity};

use crate::moments::{CompoundMoments, compound_moments};
use crate::pml::occurrence_exceedance;

/// A per-occurrence layer's statistics; see the module documentation.
/// Moments are given as their first three cumulants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerStatistics {
    pub share: f64,
    pub limit: f64,
    pub attachment: f64,
    /// `P(X > a)`, per claim.
    pub pr_attach: f64,
    /// `P(X ≥ a + l)`, per claim.
    pub pr_detach: f64,
    /// The chance of a loss to the layer in the year.
    pub pr_loss: f64,
    /// The annual ceded mean over `share × limit`.
    pub loss_on_line: f64,
    /// The number of claims that reach the layer.
    pub count: CompoundMoments,
    /// The ceded loss of a claim that reaches the layer, at the share.
    pub severity: CompoundMoments,
    /// The annual ceded loss, at the share.
    pub aggregate: CompoundMoments,
}

fn from_raw(m1: f64, m2: f64, m3: f64) -> CompoundMoments {
    CompoundMoments {
        mean: m1,
        variance: m2 - m1 * m1,
        third_cumulant: m3 - 3.0 * m1 * m2 + 2.0 * m1 * m1 * m1,
    }
}

fn scale(m: CompoundMoments, share: f64) -> CompoundMoments {
    CompoundMoments {
        mean: share * m.mean,
        variance: share * share * m.variance,
        third_cumulant: share.powi(3) * m.third_cumulant,
    }
}

/// The statistics of `share × limit xs attachment`, per occurrence, on
/// `severity`'s grid with `frequency` claims a year.
///
/// The layer's boundaries need not fall on grid points: a ceded loss
/// between points is split between them with its mean kept
/// ([`Grid::map`]), so means are exact and the higher moments slightly
/// smeared. The share scales the moments exactly.
///
/// ```
/// use prospicio_aggregate::layer_statistics::occurrence_layer_statistics;
/// use prospicio_prob::{Grid, Poisson};
///
/// let sev = Grid::new(1.0, vec![0.0, 0.4, 0.3, 0.2, 0.1]).unwrap();
/// let n = Poisson::new(3.0).unwrap();
/// let s = occurrence_layer_statistics(&n, &sev, 1.0, 2.0, 2.0).unwrap();
/// assert!((s.pr_attach - 0.3).abs() < 1e-15); // X = 3 or 4
/// assert!((s.pr_detach - 0.1).abs() < 1e-15); // X = 4
/// assert!((s.count.mean - 0.9).abs() < 1e-15); // 3 × 0.3
/// assert!((s.aggregate.mean - 1.2).abs() < 1e-12);
/// ```
pub fn occurrence_layer_statistics<N: Counting + ?Sized>(
    frequency: &N,
    severity: &Grid,
    share: f64,
    limit: f64,
    attachment: f64,
) -> Result<LayerStatistics> {
    if !(share > 0.0 && share <= 1.0) {
        return Err(Error::InvalidParameter {
            name: "share",
            value: share,
            reason: "must be in (0, 1]",
        });
    }
    if limit.is_nan() || limit <= 0.0 {
        return Err(Error::InvalidParameter {
            name: "limit",
            value: limit,
            reason: "must be positive",
        });
    }
    if !(attachment.is_finite() && attachment >= 0.0) {
        return Err(Error::InvalidParameter {
            name: "attachment",
            value: attachment,
            reason: "must be finite and not negative",
        });
    }
    // Grid points are multiples of the step: compare within a sliver of
    // it, so a boundary on a point is not missed to rounding.
    let eps = 1e-9 * severity.step();
    let (mut pr_attach, mut pr_detach) = (0.0, 0.0);
    for (j, &p) in severity.probs().iter().enumerate() {
        let x = severity.x(j);
        if x > attachment + eps {
            pr_attach += p;
        }
        if x >= attachment + limit - eps {
            pr_detach += p;
        }
    }
    // The layer at a full share; the share scales the k-th cumulant by
    // `share^k` exactly, where mapping onto the grid would smear it.
    let (image, _) = severity.map(|x| (x - attachment).clamp(0.0, limit))?;
    let aggregate = scale(compound_moments(frequency, &image)?, share);

    // The excess count, N thinned with probability p:
    // κ₁ = p E[N], κ₂ = p² Var[N] + p(1 − p) E[N],
    // κ₃ = p³ κ₃[N] + 3p²(1 − p) Var[N] + p(1 − p)(1 − 2p) E[N].
    let p = pr_attach;
    let (n1, vn, k3n) = (
        frequency.mean(),
        frequency.variance(),
        frequency.third_cumulant(),
    );
    let count = CompoundMoments {
        mean: p * n1,
        variance: p * p * vn + p * (1.0 - p) * n1,
        third_cumulant: p.powi(3) * k3n
            + 3.0 * p * p * (1.0 - p) * vn
            + p * (1.0 - p) * (1.0 - 2.0 * p) * n1,
    };

    // The layer severity given a loss to it: the image's raw moments over
    // the chance of reaching the layer.
    let severity_given_loss = if p > 0.0 {
        let m1 = image.mean();
        let m2 = image.variance() + m1 * m1;
        let m3 = image.third_moment().expect("a grid has a third moment");
        scale(from_raw(m1 / p, m2 / p, m3 / p), share)
    } else {
        CompoundMoments {
            mean: f64::NAN,
            variance: f64::NAN,
            third_cumulant: f64::NAN,
        }
    };

    Ok(LayerStatistics {
        share,
        limit,
        attachment,
        pr_attach,
        pr_detach,
        pr_loss: occurrence_exceedance(frequency, severity, attachment),
        loss_on_line: if limit.is_finite() {
            aggregate.mean / (share * limit)
        } else {
            0.0
        },
        count,
        severity: severity_given_loss,
        aggregate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_prob::{NegativeBinomial, Poisson};

    fn sev() -> Grid {
        Grid::new(1.0, vec![0.1, 0.3, 0.25, 0.2, 0.1, 0.05]).unwrap()
    }

    #[test]
    fn thinned_poisson_is_poisson() {
        let s = occurrence_layer_statistics(&Poisson::new(4.0).unwrap(), &sev(), 1.0, 2.0, 2.0)
            .unwrap();
        // P(X > 2) = 0.35: a Poisson with mean 1.4, all cumulants 1.4.
        assert!((s.count.mean - 1.4).abs() < 1e-14);
        assert!((s.count.variance - 1.4).abs() < 1e-14);
        assert!((s.count.third_cumulant - 1.4).abs() < 1e-14);
        // The annual ceded loss is the excess count times the conditional
        // severity: its mean is the product.
        assert!((s.aggregate.mean - s.count.mean * s.severity.mean).abs() < 1e-13);
    }

    #[test]
    fn share_scales_and_negative_binomial_thins() {
        let n = NegativeBinomial::new(2.0, 1.5).unwrap();
        let full = occurrence_layer_statistics(&n, &sev(), 1.0, 3.0, 1.0).unwrap();
        let half = occurrence_layer_statistics(&n, &sev(), 0.5, 3.0, 1.0).unwrap();
        assert!((half.aggregate.mean / full.aggregate.mean - 0.5).abs() < 1e-14);
        assert!((half.loss_on_line - full.loss_on_line).abs() < 1e-14);
        assert!((half.aggregate.skew() - full.aggregate.skew()).abs() < 1e-12);
        // The excess count's variance agrees with the compound formula for
        // a Bernoulli severity.
        let p = full.pr_attach;
        let v = n.mean() * p * (1.0 - p) + n.variance() * p * p;
        assert!((full.count.variance - v).abs() < 1e-13);
        assert!(occurrence_layer_statistics(&n, &sev(), 0.0, 3.0, 1.0).is_err());
    }
}
