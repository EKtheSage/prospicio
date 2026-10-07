//! The collective risk model: a claim count and a severity, with the
//! layer moments treaty pricing needs in closed form (see
//! `docs/design/pareto.md`).

use prospicio_core::Result;
use prospicio_prob::{Counting, Severity};

use crate::monte_carlo::{EventSet, simulate_events};

/// `S = X_1 + … + X_N` with claim count `N` and independent severities
/// `X_i`, priced per layer `limit` xs `attachment` applied to each loss
/// (a per-risk or per-event excess of loss with unlimited
/// reinstatements).
///
/// With `Y = min(max(X - attachment, 0), limit)` the loss to the layer
/// from one claim, the layer's aggregate has
///
/// ```text
/// E[S_layer]   = E[N] E[Y]
/// Var[S_layer] = E[N] Var[Y] + Var[N] E[Y]^2
/// ```
///
/// and the excess frequency over `x` is `E[N] P(X > x)`. With a Pareto
/// family severity starting at a threshold `t` and `E[N]` the expected
/// number of losses above `t`, this is the R package Pareto's
/// `PPP_Model` (piecewise Pareto) or `PGP_Model` (generalized Pareto).
///
/// Aggregate distributions come from the existing algorithms:
/// [`CollectiveModel::simulate`] for Monte Carlo, or a severity grid
/// ([`prospicio_prob::Grid::local_moment`]) passed to [`crate::panjer()`] or
/// [`crate::fft()`].
///
/// ```
/// use prospicio_aggregate::CollectiveModel;
/// use prospicio_prob::{PanjerClass, Pareto};
///
/// // Two losses a year above 1m, Pareto alpha 2, dispersion 1.5.
/// let model = CollectiveModel::new(
///     PanjerClass::from_mean_dispersion(2.0, 1.5).unwrap(),
///     Pareto::new(1.0e6, 2.0).unwrap(),
/// );
/// // 4m xs 1m: E[Y] = 1m (1 - 1/5) = 0.8m per loss above 1m.
/// assert!((model.layer_mean(4.0e6, 1.0e6) - 1.6e6).abs() < 1e-3);
/// assert!((model.excess_frequency(2.0e6) - 0.5).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CollectiveModel<N, X> {
    frequency: N,
    severity: X,
}

impl<N: Counting, X: Severity> CollectiveModel<N, X> {
    pub fn new(frequency: N, severity: X) -> Self {
        Self {
            frequency,
            severity,
        }
    }

    /// The claim count `N`.
    pub fn frequency(&self) -> &N {
        &self.frequency
    }

    /// The severity `X`.
    pub fn severity(&self) -> &X {
        &self.severity
    }

    /// Expected number of losses above `x`, `E[N] P(X > x)`.
    pub fn excess_frequency(&self, x: f64) -> f64 {
        self.frequency.mean() * self.severity.survival(x)
    }

    /// Expected aggregate loss to the layer, `E[N] E[Y]`.
    pub fn layer_mean(&self, limit: f64, attachment: f64) -> f64 {
        self.frequency.mean() * self.severity.layer(limit, attachment)
    }

    /// Variance of the aggregate loss to the layer,
    /// `E[N] Var[Y] + Var[N] E[Y]^2`.
    pub fn layer_variance(&self, limit: f64, attachment: f64) -> f64 {
        let (en, vn) = (self.frequency.mean(), self.frequency.variance());
        let m1 = self.severity.layer(limit, attachment);
        let m2 = self.severity.layer_second_moment(limit, attachment);
        // E[N] (E[Y^2] - E[Y]^2) + Var[N] E[Y]^2, with E[Y^2] kept whole
        // so an infinite second moment stays infinite.
        en * m2 + (vn - en) * m1 * m1
    }

    /// Standard deviation of the aggregate loss to the layer.
    pub fn layer_std_dev(&self, limit: f64, attachment: f64) -> f64 {
        self.layer_variance(limit, attachment).sqrt()
    }

    /// Expected aggregate loss, `E[N] E[X]`.
    pub fn mean(&self) -> f64 {
        self.layer_mean(f64::INFINITY, 0.0)
    }

    /// Variance of the aggregate loss.
    pub fn variance(&self) -> f64 {
        self.layer_variance(f64::INFINITY, 0.0)
    }

    /// `n_sims` simulated years of losses ([`simulate_events`]); apply a
    /// [`crate::Tower`] to them for layer results with aggregate terms.
    pub fn simulate(&self, n_sims: usize, seed: u64) -> Result<EventSet>
    where
        N: Sync,
        X: Sync,
    {
        simulate_events(&self.frequency, &self.severity, n_sims, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_prob::{Distribution, Lognormal, NegativeBinomial, Pareto, Poisson};

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-300)
    }

    #[test]
    fn whole_aggregate_moments() {
        let sev = Lognormal::from_mean_cv(1000.0, 2.0).unwrap();
        let freq = NegativeBinomial::new(4.0, 0.5).unwrap();
        let model = CollectiveModel::new(freq, sev);
        assert!(close(model.mean(), 2000.0, 1e-12));
        // E[N] Var[X] + Var[N] E[X]^2 = 2 · 4e6 + 3 · 1e6.
        assert!(close(model.variance(), 1.1e7, 1e-10));
    }

    #[test]
    fn layer_moments_match_simulation() {
        let model = CollectiveModel::new(
            NegativeBinomial::new(2.0, 1.5).unwrap(),
            Pareto::new(100.0, 1.8).unwrap(),
        );
        let (limit, att) = (400.0, 200.0);
        let events = model.simulate(200_000, 7).unwrap();
        let totals: Vec<f64> = (0..events.n_sims())
            .map(|i| {
                events
                    .events(i)
                    .iter()
                    .map(|x| (x - att).clamp(0.0, limit))
                    .sum()
            })
            .collect();
        let n = totals.len() as f64;
        let mean = totals.iter().sum::<f64>() / n;
        let var = totals.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let (m, v) = (
            model.layer_mean(limit, att),
            model.layer_variance(limit, att),
        );
        // Standard error of the mean is sqrt(v / n).
        assert!((mean - m).abs() < 4.0 * (v / n).sqrt(), "{mean} {m}");
        assert!(close(var, v, 0.03), "{var} {v}");
    }

    #[test]
    fn poisson_layer_variance_is_the_second_moment() {
        let model =
            CollectiveModel::new(Poisson::new(3.0).unwrap(), Pareto::new(1.0, 2.5).unwrap());
        let m2 = model.severity().layer_second_moment(5.0, 2.0);
        assert!(close(model.layer_variance(5.0, 2.0), 3.0 * m2, 1e-14));
        assert!(close(
            model.excess_frequency(4.0),
            3.0 * 4f64.powf(-2.5),
            1e-14
        ));
        assert_eq!(model.excess_frequency(0.5), 3.0);
        // α ≤ 2: the unlimited layer's variance is infinite.
        let heavy =
            CollectiveModel::new(Poisson::new(1.0).unwrap(), Pareto::new(1.0, 1.5).unwrap());
        assert_eq!(heavy.variance(), f64::INFINITY);
        assert!(heavy.mean().is_finite() && heavy.severity().mean() > 0.0);
    }
}
