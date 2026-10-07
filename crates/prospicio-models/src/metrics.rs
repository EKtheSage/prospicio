//! Scores for fitted models: deviance-based, ranking (Gini, lift),
//! calibration (actual vs expected), and probabilistic (CRPS, coverage,
//! log score, PIT).
//!
//! Every metric takes plain slices, so it scores any engine's output the
//! same way.

use prospicio_core::{Error, Result, StreamRng};

use crate::family::Family;

/// `Σ w d(y, μ)`, the family's deviance.
///
/// ```
/// use prospicio_models::{Family, metrics};
///
/// let d = metrics::deviance(Family::Gaussian, &[1.0, 3.0], &[2.0, 2.0], None).unwrap();
/// assert_eq!(d, 2.0);
/// ```
pub fn deviance(family: Family, y: &[f64], mu: &[f64], weights: Option<&[f64]>) -> Result<f64> {
    same_length(y.len(), mu.len(), "mu")?;
    if let Some(w) = weights {
        same_length(y.len(), w.len(), "weights")?;
    }
    Ok(y.iter()
        .zip(mu)
        .enumerate()
        .map(|(i, (&y, &m))| weights.map_or(1.0, |w| w[i]) * family.unit_deviance(y, m))
        .sum())
}

/// Weighted mean deviance, `Σ w d / Σ w`: the score to compare models on
/// held-out data.
pub fn mean_deviance(
    family: Family,
    y: &[f64],
    mu: &[f64],
    weights: Option<&[f64]>,
) -> Result<f64> {
    let total = weights.map_or(y.len() as f64, |w| w.iter().sum());
    Ok(deviance(family, y, mu, weights)? / total)
}

/// Root mean squared error.
pub fn rmse(y: &[f64], pred: &[f64]) -> Result<f64> {
    same_length(y.len(), pred.len(), "pred")?;
    let n = y.len() as f64;
    Ok((y
        .iter()
        .zip(pred)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        / n)
        .sqrt())
}

/// Mean absolute error.
pub fn mae(y: &[f64], pred: &[f64]) -> Result<f64> {
    same_length(y.len(), pred.len(), "pred")?;
    let n = y.len() as f64;
    Ok(y.iter().zip(pred).map(|(a, b)| (a - b).abs()).sum::<f64>() / n)
}

/// Weighted mean pinball (quantile) loss of predictions `pred` of the
/// `alpha` quantile, `Σ w ρ(y − q) / Σ w` with `ρ(u) = u (alpha − 1{u < 0})`.
/// Its expectation is smallest at the true `alpha` quantile, so it scores a
/// quantile model as the deviance scores a mean model; at `alpha = 0.5` it
/// is half the mean absolute error.
///
/// ```
/// use prospicio_models::metrics::pinball;
///
/// // Under-predicting the 90% quantile by 1 costs 0.9; over-predicting, 0.1.
/// assert!((pinball(&[1.0, 0.0], &[0.0, 1.0], 0.9, None).unwrap() - 0.5).abs() < 1e-15);
/// ```
pub fn pinball(y: &[f64], pred: &[f64], alpha: f64, weights: Option<&[f64]>) -> Result<f64> {
    same_length(y.len(), pred.len(), "pred")?;
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(Error::InvalidParameter {
            name: "alpha",
            value: alpha,
            reason: "must be in (0, 1)",
        });
    }
    if let Some(w) = weights {
        same_length(y.len(), w.len(), "weights")?;
    }
    let (mut loss, mut total) = (0.0, 0.0);
    for (i, (&y, &q)) in y.iter().zip(pred).enumerate() {
        let w = weights.map_or(1.0, |w| w[i]);
        let u = y - q;
        loss += w * u * if u < 0.0 { alpha - 1.0 } else { alpha };
        total += w;
    }
    Ok(loss / total)
}

/// Gini index of the ordered Lorenz curve: rows sorted by prediction
/// ascending, cumulative exposure share against cumulative loss share.
/// Twice the area between the diagonal and the curve, so 0 for a model
/// that ranks no better than chance and larger for sharper ranking.
/// Exposure defaults to 1 per row; tied predictions are taken together.
///
/// ```
/// use prospicio_models::metrics::gini;
///
/// // Perfect ranking of two risks with losses 0 and 1: area 1/4, Gini 1/2.
/// assert!((gini(&[0.0, 1.0], &[0.1, 0.9], None).unwrap() - 0.5).abs() < 1e-15);
/// // A constant prediction ranks nothing.
/// assert!(gini(&[0.0, 1.0], &[0.5, 0.5], None).unwrap().abs() < 1e-15);
/// ```
pub fn gini(y: &[f64], pred: &[f64], exposure: Option<&[f64]>) -> Result<f64> {
    same_length(y.len(), pred.len(), "pred")?;
    if let Some(e) = exposure {
        same_length(y.len(), e.len(), "exposure")?;
    }
    let e = |i: usize| exposure.map_or(1.0, |e| e[i]);
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| pred[a].total_cmp(&pred[b]));
    let total_e: f64 = (0..y.len()).map(e).sum();
    let total_y: f64 = y.iter().sum();
    if total_e <= 0.0 || total_y == 0.0 {
        return Err(Error::InvalidParameter {
            name: "y",
            value: total_y,
            reason: "needs positive total exposure and non-zero total loss",
        });
    }
    // Trapezoid area under the Lorenz curve, ties grouped into one step.
    let (mut area, mut cum_e, mut cum_y) = (0.0, 0.0, 0.0);
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        let (mut de, mut dy) = (0.0, 0.0);
        while j < order.len() && pred[order[j]] == pred[order[i]] {
            de += e(order[j]);
            dy += y[order[j]];
            j += 1;
        }
        let (x0, y0) = (cum_e / total_e, cum_y / total_y);
        cum_e += de;
        cum_y += dy;
        let (x1, y1) = (cum_e / total_e, cum_y / total_y);
        area += 0.5 * (x1 - x0) * (y0 + y1);
        i = j;
    }
    Ok(1.0 - 2.0 * area)
}

/// One row of a lift table: a band of rows with similar predictions.
#[derive(Debug, Clone, PartialEq)]
pub struct LiftBand {
    /// Total exposure in the band.
    pub exposure: f64,
    /// Total predicted loss.
    pub expected: f64,
    /// Total actual loss.
    pub actual: f64,
}

impl LiftBand {
    /// `actual / expected`.
    pub fn ratio(&self) -> f64 {
        self.actual / self.expected
    }
}

/// Lift table: rows sorted by predicted rate (`pred / exposure`) and cut
/// into `bands` groups of about equal exposure, each with its exposure,
/// expected and actual totals. A calibrated model has `actual / expected`
/// near 1 in every band; a sharp one spreads the expected rates widely.
pub fn lift(
    y: &[f64],
    pred: &[f64],
    exposure: Option<&[f64]>,
    bands: usize,
) -> Result<Vec<LiftBand>> {
    same_length(y.len(), pred.len(), "pred")?;
    if let Some(e) = exposure {
        same_length(y.len(), e.len(), "exposure")?;
    }
    if bands == 0 {
        return Err(Error::InvalidParameter {
            name: "bands",
            value: 0.0,
            reason: "must be positive",
        });
    }
    let e = |i: usize| exposure.map_or(1.0, |e| e[i]);
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| (pred[a] / e(a)).total_cmp(&(pred[b] / e(b))));
    let total: f64 = (0..y.len()).map(e).sum();
    let mut out = vec![
        LiftBand {
            exposure: 0.0,
            expected: 0.0,
            actual: 0.0,
        };
        bands
    ];
    let mut cum = 0.0;
    for &i in &order {
        // Band by the exposure midpoint of the row.
        let mid = (cum + 0.5 * e(i)) / total;
        cum += e(i);
        let b = ((mid * bands as f64) as usize).min(bands - 1);
        out[b].exposure += e(i);
        out[b].expected += pred[i];
        out[b].actual += y[i];
    }
    Ok(out)
}

/// Continuous ranked probability score of `draws` (equally likely) for the
/// outcome `y`: `E|X - y| - E|X - X'| / 2`, lower is better. Exact for the
/// empirical distribution of the draws, in `O(m log m)`.
///
/// ```
/// use prospicio_models::metrics::crps;
///
/// // One draw: the absolute error.
/// assert_eq!(crps(&[3.0], 1.0).unwrap(), 2.0);
/// ```
pub fn crps(draws: &[f64], y: f64) -> Result<f64> {
    if draws.is_empty() {
        return Err(Error::InvalidParameter {
            name: "draws",
            value: 0.0,
            reason: "must not be empty",
        });
    }
    let m = draws.len() as f64;
    let mut sorted = draws.to_vec();
    sorted.sort_by(f64::total_cmp);
    let abs_err: f64 = sorted.iter().map(|x| (x - y).abs()).sum::<f64>() / m;
    // Σ_{i<j} (x_j - x_i) = Σ_i x_(i) (2i - m + 1), 0-based.
    let spread: f64 = sorted
        .iter()
        .enumerate()
        .map(|(i, x)| x * (2.0 * i as f64 - m + 1.0))
        .sum::<f64>();
    Ok(abs_err - spread / (m * m))
}

/// Share of outcomes inside their intervals `[lo, hi]`.
pub fn coverage(y: &[f64], lo: &[f64], hi: &[f64]) -> Result<f64> {
    same_length(y.len(), lo.len(), "lo")?;
    same_length(y.len(), hi.len(), "hi")?;
    let inside = (0..y.len())
        .filter(|&i| lo[i] <= y[i] && y[i] <= hi[i])
        .count();
    Ok(inside as f64 / y.len() as f64)
}

/// Mean log score, `-(1/n) Σ log f(yᵢ)`, of each outcome under its
/// predictive distribution: the family with mean `μᵢ`, dispersion `φ` and
/// weight `wᵢ` ([`Family::log_density`]). Lower is better; it is the
/// proper score that rewards a sharp and calibrated density, where CRPS
/// works from draws alone.
///
/// ```
/// use prospicio_models::{Family, metrics};
///
/// // Poisson(1) at y = 0: -log e^-1 = 1.
/// assert!((metrics::log_score(Family::Poisson, &[0.0], &[1.0], 1.0, None).unwrap() - 1.0).abs() < 1e-15);
/// ```
pub fn log_score(
    family: Family,
    y: &[f64],
    mu: &[f64],
    dispersion: f64,
    weights: Option<&[f64]>,
) -> Result<f64> {
    same_length(y.len(), mu.len(), "mu")?;
    if let Some(w) = weights {
        same_length(y.len(), w.len(), "weights")?;
    }
    let mut total = 0.0;
    for i in 0..y.len() {
        let w = weights.map_or(1.0, |w| w[i]);
        total -= family.log_density(y[i], mu[i], dispersion, w)?;
    }
    Ok(total / y.len() as f64)
}

/// Probability integral transform of each outcome under its predictive
/// distribution: `F(yᵢ)`, uniform on `(0, 1)` when the model is
/// calibrated. Where the distribution has an atom (counts, a Tweedie's
/// zero) the PIT is randomized, `F(y⁻) + u (F(y) - F(y⁻))` with `u` drawn
/// from stream 0 of `seed` in row order (Czado, Gneiting and Held, 2009),
/// so it is still uniform under the model. Check it with
/// [`pit_histogram`] or [`ks_uniform`].
pub fn pit(
    family: Family,
    y: &[f64],
    mu: &[f64],
    dispersion: f64,
    weights: Option<&[f64]>,
    seed: u64,
) -> Result<Vec<f64>> {
    same_length(y.len(), mu.len(), "mu")?;
    if let Some(w) = weights {
        same_length(y.len(), w.len(), "weights")?;
    }
    let mut rng = StreamRng::new(seed, 0);
    (0..y.len())
        .map(|i| {
            let w = weights.map_or(1.0, |w| w[i]);
            let (lo, hi) = family.cdf_bounds(y[i], mu[i], dispersion, w)?;
            Ok(if hi > lo {
                lo + rng.next_open01() * (hi - lo)
            } else {
                hi
            })
        })
        .collect()
}

/// The PIT of `y` under the empirical distribution of `draws` (one
/// component of a predictive distribution), randomized over ties:
/// `(#{x < y} + u #{x = y}) / m`.
///
/// ```
/// use prospicio_models::metrics::pit_from_draws;
///
/// assert_eq!(pit_from_draws(&[1.0, 2.0, 3.0, 4.0], 2.5, 0.5).unwrap(), 0.5);
/// ```
pub fn pit_from_draws(draws: &[f64], y: f64, u: f64) -> Result<f64> {
    if draws.is_empty() {
        return Err(Error::InvalidParameter {
            name: "draws",
            value: 0.0,
            reason: "must not be empty",
        });
    }
    let below = draws.iter().filter(|&&x| x < y).count() as f64;
    let ties = draws.iter().filter(|&&x| x == y).count() as f64;
    Ok((below + u * ties) / draws.len() as f64)
}

/// Counts of PIT values in `bins` equal-width bins of `[0, 1]`: flat for a
/// calibrated model, U-shaped when it is too sharp, humped when too wide.
pub fn pit_histogram(pit: &[f64], bins: usize) -> Result<Vec<usize>> {
    if bins == 0 {
        return Err(Error::InvalidParameter {
            name: "bins",
            value: 0.0,
            reason: "must be positive",
        });
    }
    let mut counts = vec![0; bins];
    for &p in pit {
        let b = ((p * bins as f64) as usize).min(bins - 1);
        counts[b] += 1;
    }
    Ok(counts)
}

/// Kolmogorov–Smirnov distance between the empirical distribution of
/// `values` and the uniform on `[0, 1]`: `max |F̂(x) − x|`. Under
/// uniformity it is about `1.36 / √n` or less 95% of the time.
pub fn ks_uniform(values: &[f64]) -> Result<f64> {
    if values.is_empty() {
        return Err(Error::InvalidParameter {
            name: "values",
            value: 0.0,
            reason: "must not be empty",
        });
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len() as f64;
    Ok(v.iter()
        .enumerate()
        .map(|(i, &x)| (x - i as f64 / n).max((i as f64 + 1.0) / n - x))
        .fold(0.0, f64::max))
}

fn same_length(n: usize, m: usize, name: &'static str) -> Result<()> {
    if n == m {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name,
            value: m as f64,
            reason: "must have one value per observation",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Responses drawn from each family's own predictive distribution give
    /// uniform PITs; a model with the wrong mean does not.
    #[test]
    fn pit_is_uniform_under_the_model() {
        let families = [
            (Family::Gaussian, 2.0),
            (Family::Poisson, 1.0),
            (Family::Poisson, 2.5),
            (Family::Binomial, 1.0),
            (Family::NegativeBinomial { theta: 1.5 }, 1.0),
            (Family::Gamma, 0.4),
            (Family::InverseGaussian, 0.3),
            (Family::Tweedie { power: 1.5 }, 2.0),
        ];
        let n = 4000;
        for (k, &(family, phi)) in families.iter().enumerate() {
            let mut rng = StreamRng::new(9, k as u64);
            let mu: Vec<f64> = (0..n)
                .map(|i| match family {
                    Family::Binomial => 0.2 + 0.6 * (i % 7) as f64 / 7.0,
                    _ => 0.5 + (i % 5) as f64,
                })
                .collect();
            let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64).collect();
            let y: Vec<f64> = (0..n)
                .map(|i| family.draw(mu[i], phi, w[i], rng.next_open01()).unwrap())
                .collect();
            let p = pit(family, &y, &mu, phi, Some(&w), 1).unwrap();
            let ks = ks_uniform(&p).unwrap();
            assert!(ks < 1.63 / (n as f64).sqrt(), "{family:?}: ks {ks}");
            let off: Vec<f64> = mu
                .iter()
                .map(|m| match family {
                    Family::Binomial => m * 0.7,
                    _ => m * 1.5,
                })
                .collect();
            let bad = ks_uniform(&pit(family, &y, &off, phi, Some(&w), 1).unwrap()).unwrap();
            assert!(bad > 3.0 / (n as f64).sqrt(), "{family:?}: misfit ks {bad}");
            // The true means score better than the wrong ones.
            let good = log_score(family, &y, &mu, phi, Some(&w)).unwrap();
            let worse = log_score(family, &y, &off, phi, Some(&w)).unwrap();
            assert!(good < worse, "{family:?}");
        }
    }

    #[test]
    fn log_density_is_the_log_likelihood_where_they_agree() {
        for family in [
            Family::Gaussian,
            Family::Gamma,
            Family::InverseGaussian,
            Family::Tweedie { power: 1.4 },
        ] {
            let a = family.log_density(1.7, 2.0, 0.8, 1.5).unwrap();
            let b = family.log_likelihood(1.7, 2.0, 1.5, 0.8);
            assert!((a - b).abs() < 1e-12, "{family:?}");
        }
        for (family, y) in [
            (Family::Poisson, 3.0),
            (Family::NegativeBinomial { theta: 2.0 }, 3.0),
        ] {
            let a = family.log_density(y, 2.0, 1.0, 1.0).unwrap();
            let b = family.log_likelihood(y, 2.0, 1.0, 1.0);
            assert!((a - b).abs() < 1e-12, "{family:?}");
        }
        assert_eq!(
            Family::Poisson.log_density(1.5, 2.0, 1.0, 1.0).unwrap(),
            f64::NEG_INFINITY
        );
        assert_eq!(
            pit_histogram(&[0.05, 0.15, 0.95, 1.0], 2).unwrap(),
            vec![2, 2]
        );
    }

    #[test]
    fn crps_matches_the_pairwise_definition() {
        let x: [f64; 6] = [0.3, -1.2, 2.5, 0.3, 4.0, 1.1];
        let y = 0.7;
        let m = x.len() as f64;
        let a: f64 = x.iter().map(|v| (v - y).abs()).sum::<f64>() / m;
        let b: f64 = x
            .iter()
            .flat_map(|u| x.iter().map(move |v| (u - v).abs()))
            .sum::<f64>()
            / (m * m);
        assert!((crps(&x, y).unwrap() - (a - 0.5 * b)).abs() < 1e-14);
    }

    #[test]
    fn gini_with_exposure_and_lift() {
        // Rates 0, 1, 2 on exposures 1, 2, 1 and a model that ranks them.
        let y = [0.0, 2.0, 2.0];
        let pred = [0.1, 1.8, 2.1];
        let e = [1.0, 2.0, 1.0];
        let g = gini(&y, &pred, Some(&e)).unwrap();
        // Lorenz points: (0, 0), (1/4, 0), (3/4, 1/2), (1, 1).
        let area = 0.5 * 0.5 * 0.5 + 0.5 * 0.25 * 1.5;
        assert!((g - (1.0 - 2.0 * area)).abs() < 1e-15);
        let bands = lift(&y, &pred, Some(&e), 2).unwrap();
        assert_eq!(bands[0].exposure + bands[1].exposure, 4.0);
        assert!(bands[1].actual / bands[1].exposure >= bands[0].actual / bands[0].exposure);
    }

    #[test]
    fn deviance_scores_and_coverage() {
        let y = [1.0, 0.0, 3.0];
        let mu = [1.0, 0.5, 2.0];
        let d = deviance(Family::Poisson, &y, &mu, None).unwrap();
        let want = 2.0 * (0.5 + (3.0 * 1.5f64.ln() - 1.0));
        assert!((d - want).abs() < 1e-14);
        assert!(
            (mean_deviance(Family::Poisson, &y, &mu, None).unwrap() - want / 3.0).abs() < 1e-15
        );
        assert_eq!(
            coverage(&y, &[0.0, 0.0, 0.0], &[2.0, 2.0, 2.0]).unwrap(),
            2.0 / 3.0
        );
        assert!(rmse(&y, &[1.0]).is_err());
    }
}
