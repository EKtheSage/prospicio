//! Scores for fitted models: deviance-based, ranking (Gini, lift),
//! calibration (actual vs expected), and probabilistic (CRPS, coverage).
//!
//! Every metric takes plain slices, so it scores any engine's output the
//! same way.

use act_core::{Error, Result};

use crate::family::Family;

/// `Σ w d(y, μ)`, the family's deviance.
///
/// ```
/// use act_models::{Family, metrics};
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

/// Gini index of the ordered Lorenz curve: rows sorted by prediction
/// ascending, cumulative exposure share against cumulative loss share.
/// Twice the area between the diagonal and the curve, so 0 for a model
/// that ranks no better than chance and larger for sharper ranking.
/// Exposure defaults to 1 per row; tied predictions are taken together.
///
/// ```
/// use act_models::metrics::gini;
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
/// use act_models::metrics::crps;
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
