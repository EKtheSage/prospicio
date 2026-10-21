//! Scores for fitted models: deviance-based, ranking (Gini, lift),
//! calibration (actual vs expected), and probabilistic (CRPS, coverage,
//! log score, PIT).
//!
//! Every metric takes plain slices, so it scores any engine's output the
//! same way.

use prospicio_core::{Error, Result, StreamRng};
use prospicio_math::special::norm_quantile;

use crate::design::Design;
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

/// Randomized quantile residuals (Dunn and Smyth, 1996): `Φ⁻¹` of the
/// randomized [`pit`] at the same `seed`, standard normal when the model is
/// right. A continuous family's are exact, `Φ⁻¹(F(yᵢ))`; a count's, and the
/// Tweedie's at zero, are drawn between `Φ⁻¹(F(y⁻))` and `Φ⁻¹(F(y))`, as
/// R's `statmod::qresiduals`.
///
/// ```
/// use prospicio_models::{metrics::quantile_residuals, Family};
///
/// // A Gaussian's is the standardized residual (y − μ) / √(φ / w).
/// let r = quantile_residuals(Family::Gaussian, &[3.0], &[1.0], 4.0, None, 0).unwrap();
/// assert!((r[0] - 1.0).abs() < 1e-12);
/// ```
pub fn quantile_residuals(
    family: Family,
    y: &[f64],
    mu: &[f64],
    dispersion: f64,
    weights: Option<&[f64]>,
    seed: u64,
) -> Result<Vec<f64>> {
    Ok(pit(family, y, mu, dispersion, weights, seed)?
        .into_iter()
        .map(norm_quantile)
        .collect())
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

/// The normalized Gini: [`gini`] divided by the Gini of a perfect
/// ranking, the rows sorted by their actual rate `y / exposure`. 1 for a
/// model that ranks every row as its outcome does. For a 0/1 response with
/// unit exposure, `AUROC = (1 + normalized Gini) / 2` (CAS Monograph 5,
/// section 7.3.1, footnote 17).
///
/// ```
/// use prospicio_models::metrics::normalized_gini;
///
/// let y = [0.0, 1.0, 0.0, 3.0];
/// assert!((normalized_gini(&y, &y, None).unwrap() - 1.0).abs() < 1e-15);
/// ```
///
/// Fails as [`gini`], or when the actual rates are all equal.
pub fn normalized_gini(y: &[f64], pred: &[f64], exposure: Option<&[f64]>) -> Result<f64> {
    let g = gini(y, pred, exposure)?;
    let rate: Vec<f64> = match exposure {
        Some(e) => y.iter().zip(e).map(|(y, e)| y / e).collect(),
        None => y.to_vec(),
    };
    let best = gini(y, &rate, exposure)?;
    if best == 0.0 {
        return Err(Error::InvalidParameter {
            name: "y",
            value: best,
            reason: "the actual rates must not all be equal",
        });
    }
    Ok(g / best)
}

/// One band of a [`double_lift`] chart.
#[derive(Debug, Clone, PartialEq)]
pub struct DoubleLiftBand {
    /// Total exposure in the band.
    pub exposure: f64,
    /// Total actual loss.
    pub actual: f64,
    /// Total loss predicted by the first model.
    pub model_a: f64,
    /// Total loss predicted by the second model.
    pub model_b: f64,
    /// The band's actual loss per exposure over the overall average.
    pub actual_relativity: f64,
    /// The first model's average in the band over its overall average.
    pub a_relativity: f64,
    /// The second model's average in the band over its overall average.
    pub b_relativity: f64,
}

/// The double lift chart of two models (CAS Monograph 5, section 7.2.2):
/// rows sorted by the ratio of their predictions `pred_a / pred_b`, cut
/// into `bands` groups of about equal exposure, each with the actual and
/// both models' totals and their averages relative to the overall averages
/// (each model against its own, so a difference in overall level does not
/// show). The model whose relativities follow the actual's better is the
/// better one where the two disagree.
///
/// Fails when the lengths differ, `bands` is 0, a prediction of the second
/// model is not positive, or a total is 0.
pub fn double_lift(
    y: &[f64],
    pred_a: &[f64],
    pred_b: &[f64],
    exposure: Option<&[f64]>,
    bands: usize,
) -> Result<Vec<DoubleLiftBand>> {
    same_length(y.len(), pred_a.len(), "pred_a")?;
    same_length(y.len(), pred_b.len(), "pred_b")?;
    if let Some(&bad) = pred_b.iter().find(|&&p| p.is_nan() || p <= 0.0) {
        return Err(Error::InvalidParameter {
            name: "pred_b",
            value: bad,
            reason: "must be positive",
        });
    }
    let ratio: Vec<f64> = pred_a.iter().zip(pred_b).map(|(a, b)| a / b).collect();
    let groups = banded(&ratio, exposure, y.len(), bands)?;
    let e = |i: usize| exposure.map_or(1.0, |e| e[i]);
    let total = |v: &dyn Fn(usize) -> f64| (0..y.len()).map(v).sum::<f64>();
    let total_e = total(&e);
    let (avg_y, avg_a, avg_b) = (
        total(&|i| y[i]) / total_e,
        total(&|i| pred_a[i]) / total_e,
        total(&|i| pred_b[i]) / total_e,
    );
    if avg_y == 0.0 || avg_a == 0.0 {
        return Err(Error::InvalidParameter {
            name: "y",
            value: avg_y,
            reason: "the actual and predicted totals must not be 0",
        });
    }
    Ok(groups
        .iter()
        .map(|rows| {
            let sum = |v: &dyn Fn(usize) -> f64| rows.iter().map(|&i| v(i)).sum::<f64>();
            let (ex, actual, a, b) = (
                sum(&e),
                sum(&|i| y[i]),
                sum(&|i| pred_a[i]),
                sum(&|i| pred_b[i]),
            );
            DoubleLiftBand {
                exposure: ex,
                actual,
                model_a: a,
                model_b: b,
                actual_relativity: actual / ex / avg_y,
                a_relativity: a / ex / avg_a,
                b_relativity: b / ex / avg_b,
            }
        })
        .collect())
}

/// One band of a [`loss_ratio_chart`].
#[derive(Debug, Clone, PartialEq)]
pub struct LossRatioBand {
    /// Total exposure in the band.
    pub exposure: f64,
    /// Total current premium.
    pub premium: f64,
    /// Total predicted loss.
    pub expected: f64,
    /// Total actual loss.
    pub actual: f64,
}

impl LossRatioBand {
    /// `actual / premium`, the band's actual loss ratio.
    pub fn loss_ratio(&self) -> f64 {
        self.actual / self.premium
    }

    /// `expected / premium`, the band's predicted loss ratio.
    pub fn expected_loss_ratio(&self) -> f64 {
        self.expected / self.premium
    }
}

/// The loss ratio chart (CAS Monograph 5, section 7.2.3): rows sorted by
/// their predicted loss ratio `pred / premium` against the current
/// premium, cut into `bands` groups of about equal exposure, each with its
/// actual loss ratio. A model that finds what the current rates miss shows
/// actual loss ratios rising across the bands.
///
/// Fails when the lengths differ, `bands` is 0 or a premium is not
/// positive.
pub fn loss_ratio_chart(
    y: &[f64],
    pred: &[f64],
    premium: &[f64],
    exposure: Option<&[f64]>,
    bands: usize,
) -> Result<Vec<LossRatioBand>> {
    same_length(y.len(), pred.len(), "pred")?;
    same_length(y.len(), premium.len(), "premium")?;
    if let Some(&bad) = premium.iter().find(|&&p| p.is_nan() || p <= 0.0) {
        return Err(Error::InvalidParameter {
            name: "premium",
            value: bad,
            reason: "must be positive",
        });
    }
    let ratio: Vec<f64> = pred.iter().zip(premium).map(|(p, q)| p / q).collect();
    let groups = banded(&ratio, exposure, y.len(), bands)?;
    let e = |i: usize| exposure.map_or(1.0, |e| e[i]);
    Ok(groups
        .iter()
        .map(|rows| LossRatioBand {
            exposure: rows.iter().map(|&i| e(i)).sum(),
            premium: rows.iter().map(|&i| premium[i]).sum(),
            expected: rows.iter().map(|&i| pred[i]).sum(),
            actual: rows.iter().map(|&i| y[i]).sum(),
        })
        .collect())
}

/// Row indices sorted by `key` and cut into `bands` groups of about equal
/// exposure, each row by its exposure midpoint, as [`lift`].
fn banded(
    key: &[f64],
    exposure: Option<&[f64]>,
    n: usize,
    bands: usize,
) -> Result<Vec<Vec<usize>>> {
    if let Some(e) = exposure {
        same_length(n, e.len(), "exposure")?;
    }
    if bands == 0 {
        return Err(Error::InvalidParameter {
            name: "bands",
            value: 0.0,
            reason: "must be positive",
        });
    }
    let e = |i: usize| exposure.map_or(1.0, |e| e[i]);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| key[a].total_cmp(&key[b]));
    let total: f64 = (0..n).map(e).sum();
    let mut out = vec![Vec::new(); bands];
    let mut cum = 0.0;
    for &i in &order {
        let mid = (cum + 0.5 * e(i)) / total;
        cum += e(i);
        out[((mid * bands as f64) as usize).min(bands - 1)].push(i);
    }
    Ok(out)
}

/// A receiver operating characteristic curve, from [`roc`].
#[derive(Debug, Clone, PartialEq)]
pub struct Roc {
    /// The thresholds, descending: a row is predicted positive when its
    /// score is at least the threshold. The first is `+∞` (nothing
    /// positive).
    pub thresholds: Vec<f64>,
    /// False positive rate at each threshold.
    pub false_positive_rate: Vec<f64>,
    /// True positive rate (sensitivity) at each threshold.
    pub true_positive_rate: Vec<f64>,
    /// The area under the curve, by the trapezoid rule (ties in the score
    /// count half).
    pub auc: f64,
}

/// The ROC curve of a binary model (CAS Monograph 5, section 7.3.1): the
/// true and false positive rates at every distinct score as threshold, and
/// the area under the curve. `y` holds 0 and 1; `weights` default to 1.
///
/// ```
/// use prospicio_models::metrics::roc;
///
/// let r = roc(&[0.0, 0.0, 1.0, 1.0], &[0.1, 0.4, 0.35, 0.8], None).unwrap();
/// assert!((r.auc - 0.75).abs() < 1e-15);
/// ```
///
/// Fails when the lengths differ, a response is not 0 or 1, or either
/// class has no weight.
pub fn roc(y: &[f64], score: &[f64], weights: Option<&[f64]>) -> Result<Roc> {
    same_length(y.len(), score.len(), "score")?;
    if let Some(w) = weights {
        same_length(y.len(), w.len(), "weights")?;
    }
    if let Some(&bad) = y.iter().find(|&&v| v != 0.0 && v != 1.0) {
        return Err(Error::InvalidParameter {
            name: "y",
            value: bad,
            reason: "must be 0 or 1",
        });
    }
    let w = |i: usize| weights.map_or(1.0, |w| w[i]);
    let positives: f64 = (0..y.len()).filter(|&i| y[i] == 1.0).map(w).sum();
    let negatives: f64 = (0..y.len()).filter(|&i| y[i] == 0.0).map(w).sum();
    if positives <= 0.0 || negatives <= 0.0 {
        return Err(Error::InvalidParameter {
            name: "y",
            value: positives,
            reason: "needs weight in both classes",
        });
    }
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| score[b].total_cmp(&score[a]));
    let (mut thresholds, mut fpr, mut tpr) = (vec![f64::INFINITY], vec![0.0], vec![0.0]);
    let (mut tp, mut fp, mut auc) = (0.0, 0.0, 0.0);
    let mut i = 0;
    while i < order.len() {
        let s = score[order[i]];
        let (x0, y0) = (fp / negatives, tp / positives);
        while i < order.len() && score[order[i]] == s {
            let k = order[i];
            if y[k] == 1.0 {
                tp += w(k);
            } else {
                fp += w(k);
            }
            i += 1;
        }
        let (x1, y1) = (fp / negatives, tp / positives);
        auc += 0.5 * (x1 - x0) * (y0 + y1);
        thresholds.push(s);
        fpr.push(x1);
        tpr.push(y1);
    }
    Ok(Roc {
        thresholds,
        false_positive_rate: fpr,
        true_positive_rate: tpr,
        auc,
    })
}

/// Weighted counts of a binary model's predictions at a threshold, from
/// [`confusion_matrix`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Confusion {
    pub true_positive: f64,
    pub false_positive: f64,
    pub true_negative: f64,
    pub false_negative: f64,
}

impl Confusion {
    /// `TP / (TP + FN)`.
    pub fn sensitivity(&self) -> f64 {
        self.true_positive / (self.true_positive + self.false_negative)
    }

    /// `TN / (TN + FP)`.
    pub fn specificity(&self) -> f64 {
        self.true_negative / (self.true_negative + self.false_positive)
    }
}

/// The confusion matrix of a binary model at `threshold`: a row is
/// predicted positive when its score is at least the threshold. `y` holds
/// 0 and 1; `weights` default to 1.
///
/// Fails when the lengths differ or a response is not 0 or 1.
pub fn confusion_matrix(
    y: &[f64],
    score: &[f64],
    threshold: f64,
    weights: Option<&[f64]>,
) -> Result<Confusion> {
    same_length(y.len(), score.len(), "score")?;
    if let Some(w) = weights {
        same_length(y.len(), w.len(), "weights")?;
    }
    let mut c = Confusion {
        true_positive: 0.0,
        false_positive: 0.0,
        true_negative: 0.0,
        false_negative: 0.0,
    };
    for (i, (&y, &s)) in y.iter().zip(score).enumerate() {
        let w = weights.map_or(1.0, |w| w[i]);
        let cell = match (y, s >= threshold) {
            (1.0, true) => &mut c.true_positive,
            (0.0, true) => &mut c.false_positive,
            (0.0, false) => &mut c.true_negative,
            (1.0, false) => &mut c.false_negative,
            _ => {
                return Err(Error::InvalidParameter {
                    name: "y",
                    value: y,
                    reason: "must be 0 or 1",
                });
            }
        };
        *cell += w;
    }
    Ok(c)
}

/// Variance inflation factors of a design's columns: `1 / (1 − R²_j)`, with
/// `R²_j` that of column `j` regressed on the others and an intercept
/// (CAS Monograph 5, section 2.9; R `car::vif` for single-column terms).
/// Computed as the diagonal of the inverse of the columns' correlation
/// matrix. A constant column (the intercept) gets `NaN`.
///
/// ```
/// use prospicio_models::Design;
/// use prospicio_models::metrics::vif;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0];
/// let z = vec![1.0, -1.0, -1.0, 1.0]; // uncorrelated with x
/// let d = Design::new(
///     vec!["(Intercept)".into(), "x".into(), "z".into()],
///     vec![vec![1.0; 4], x, z],
/// )
/// .unwrap();
/// let v = vif(&d).unwrap();
/// assert!(v[0].is_nan());
/// assert!((v[1] - 1.0).abs() < 1e-12 && (v[2] - 1.0).abs() < 1e-12);
/// ```
///
/// Fails when the non-constant columns are collinear.
pub fn vif(design: &Design) -> Result<Vec<f64>> {
    let n = design.n_rows() as f64;
    let centred: Vec<Option<Vec<f64>>> = (0..design.n_cols())
        .map(|j| {
            let x = design.column(j);
            let mean = x.iter().sum::<f64>() / n;
            let c: Vec<f64> = x.iter().map(|v| v - mean).collect();
            let ss: f64 = c.iter().map(|v| v * v).sum();
            (ss > 0.0).then(|| c.iter().map(|v| v / ss.sqrt()).collect())
        })
        .collect();
    let kept: Vec<&Vec<f64>> = centred.iter().flatten().collect();
    let k = kept.len();
    let mut corr = vec![0.0; k * k];
    for a in 0..k {
        for b in 0..=a {
            let r: f64 = kept[a].iter().zip(kept[b]).map(|(x, y)| x * y).sum();
            corr[a * k + b] = r;
            corr[b * k + a] = r;
        }
    }
    let l = prospicio_math::linalg::cholesky(&corr, k).ok_or(Error::InvalidParameter {
        name: "design",
        value: k as f64,
        reason: "has collinear columns",
    })?;
    let inverse = prospicio_math::linalg::cholesky_inverse(&l, k);
    let mut next = 0;
    Ok(centred
        .iter()
        .map(|c| match c {
            Some(_) => {
                let v = inverse[next * k + next];
                next += 1;
                v
            }
            None => f64::NAN,
        })
        .collect())
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
    fn quantile_residuals_are_standard_normal_under_the_model() {
        // Counts and a Tweedie with many zeros, drawn from the model.
        let mut rng = StreamRng::new(3, 0);
        let n = 4000;
        for family in [
            Family::Poisson,
            Family::NegativeBinomial { theta: 2.0 },
            Family::Tweedie { power: 1.5 },
        ] {
            let mu: Vec<f64> = (0..n).map(|i| 0.3 + (i % 7) as f64 * 0.4).collect();
            let phi = if matches!(family, Family::Tweedie { .. }) {
                1.3
            } else {
                1.0
            };
            let y: Vec<f64> = mu
                .iter()
                .map(|&m| family.draw(m, phi, 1.0, rng.next_open01()).unwrap())
                .collect();
            let r = quantile_residuals(family, &y, &mu, phi, None, 11).unwrap();
            let mean = r.iter().sum::<f64>() / n as f64;
            let var = r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            assert!(
                mean.abs() < 0.05 && (var - 1.0).abs() < 0.07,
                "{family:?}: {mean} {var}"
            );
            // `Φ⁻¹` of the PIT at the same seed, whose uniformity
            // `pit_is_uniform_under_the_model` checks.
            let p = pit(family, &y, &mu, phi, None, 11).unwrap();
            for (r, p) in r.iter().zip(&p) {
                assert!((prospicio_math::special::norm_cdf(*r) - p).abs() < 1e-12);
            }
        }
    }

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

    #[test]
    fn auc_is_the_mann_whitney_probability_and_tracks_the_gini() {
        let n = 200;
        let score: Vec<f64> = (0..n).map(|i| ((i * 37) % 101) as f64 / 101.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                if score[i] + ((i * 17) % 13) as f64 / 26.0 > 0.8 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let r = roc(&y, &score, None).unwrap();
        // P(score of a positive > score of a negative), ties counting half.
        let (mut wins, mut pairs) = (0.0, 0.0);
        for i in 0..n {
            for j in 0..n {
                if y[i] == 1.0 && y[j] == 0.0 {
                    pairs += 1.0;
                    wins += if score[i] > score[j] {
                        1.0
                    } else if score[i] == score[j] {
                        0.5
                    } else {
                        0.0
                    };
                }
            }
        }
        assert!((r.auc - wins / pairs).abs() < 1e-12);
        let ng = normalized_gini(&y, &score, None).unwrap();
        assert!((r.auc - 0.5 * (1.0 + ng)).abs() < 1e-12, "{} {}", r.auc, ng);
        assert_eq!(r.thresholds[0], f64::INFINITY);
        assert_eq!(*r.true_positive_rate.last().unwrap(), 1.0);
        let c = confusion_matrix(&y, &score, 0.5, None).unwrap();
        let total = c.true_positive + c.false_positive + c.true_negative + c.false_negative;
        assert_eq!(total, n as f64);
        assert!(roc(&[0.0, 2.0], &[0.1, 0.2], None).is_err());
    }

    #[test]
    fn double_lift_and_loss_ratio_bands_add_up() {
        let n = 100;
        let e: Vec<f64> = (0..n).map(|i| 0.5 + (i % 3) as f64 * 0.25).collect();
        let a: Vec<f64> = (0..n)
            .map(|i| e[i] * (1.0 + (i % 7) as f64 / 10.0))
            .collect();
        let b: Vec<f64> = (0..n)
            .map(|i| e[i] * (1.2 + (i % 5) as f64 / 10.0))
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| a[i] * (0.8 + (i % 4) as f64 / 10.0))
            .collect();
        let bands = double_lift(&y, &a, &b, Some(&e), 5).unwrap();
        assert_eq!(bands.len(), 5);
        let total_e: f64 = e.iter().sum();
        // Exposure-weighted relativities average to 1.
        for rel in [
            bands
                .iter()
                .map(|b| b.exposure * b.actual_relativity)
                .sum::<f64>(),
            bands
                .iter()
                .map(|b| b.exposure * b.a_relativity)
                .sum::<f64>(),
            bands
                .iter()
                .map(|b| b.exposure * b.b_relativity)
                .sum::<f64>(),
        ] {
            assert!((rel / total_e - 1.0).abs() < 1e-12);
        }
        // Sorted by a / b: the first model's relativity to the second's rises.
        let r: Vec<f64> = bands.iter().map(|b| b.model_a / b.model_b).collect();
        assert!(r.windows(2).all(|w| w[0] <= w[1]));
        let lr = loss_ratio_chart(&y, &a, &b, Some(&e), 4).unwrap();
        let actual: f64 = lr.iter().map(|b| b.actual).sum();
        assert!((actual - y.iter().sum::<f64>()).abs() < 1e-9);
        let er: Vec<f64> = lr.iter().map(LossRatioBand::expected_loss_ratio).collect();
        assert!(er.windows(2).all(|w| w[0] <= w[1]));
        assert!(loss_ratio_chart(&y, &a, &vec![0.0; n], None, 4).is_err());
    }
}
