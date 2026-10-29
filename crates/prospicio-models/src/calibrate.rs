//! Auto-calibration: whether a model's predictions are right on average at
//! every level, and two repairs (#545).
//!
//! A model is auto-calibrated when the mean response among the rows with
//! prediction `p` is `p` (Wüthrich and Merz, *Statistical Foundations of
//! Actuarial Learning*, section 7.4.2). A model can rank risks well and fail
//! this, and a network or boosted model also loses the balance property of a
//! canonical-link GLM, so its total prediction does not match the total
//! response. This module gives:
//!
//! - [`isotonic`], weighted isotonic regression by pool-adjacent-violators,
//!   as scikit-learn's `IsotonicRegression`, and [`Recalibrated`], a model
//!   whose predictions are mapped through it (the ranking kept, the levels
//!   fixed);
//! - [`balance_correction`] and [`Balanced`], the multiplier that makes the
//!   total prediction equal the total response;
//! - [`reliability`], binned mean prediction against mean response, and
//!   [`murphy`], the CORP decomposition of a mean score into miscalibration,
//!   discrimination and uncertainty (Dimitriadis, Gneiting and Jordan, 2021),
//!   whose miscalibration term is the auto-calibration diagnostic of Denuit,
//!   Charpentier and Trufin (2021): the score the isotonic recalibration
//!   gains.

use prospicio_core::{Error, Result};
use prospicio_prob::PredictiveDistribution;

use crate::design::Design;
use crate::family::Family;
use crate::model::Fitted;

/// How an [`Isotonic`] fit reads between its points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// Linear between neighbouring points, as scikit-learn's `predict`.
    Linear,
    /// The value of the nearest point at or below (a step function).
    Step,
}

/// A fitted isotonic regression: values at the distinct inputs, monotone.
#[derive(Debug, Clone, PartialEq)]
pub struct Isotonic {
    /// The distinct inputs, ascending.
    x: Vec<f64>,
    /// The fitted value at each.
    values: Vec<f64>,
    increasing: bool,
    interpolation: Interpolation,
}

/// Weighted isotonic regression of `y` on `x`: the monotone function
/// (non-decreasing when `increasing`, else non-increasing) closest to `y` in
/// weighted squared error, by pool-adjacent-violators. Tied `x` are merged
/// first into their weighted mean response and summed weight, as
/// scikit-learn does, so the fit is a function of `x`. It is also the
/// closest monotone fit under every Bregman score (Poisson, gamma or any
/// other deviance), so it serves each family.
///
/// ```
/// use prospicio_models::calibrate::isotonic;
///
/// let f = isotonic(&[1.0, 2.0, 3.0, 4.0], &[1.0, 3.0, 2.0, 4.0], None, true).unwrap();
/// assert_eq!(f.values(), [1.0, 2.5, 2.5, 4.0]);
/// assert_eq!(f.predict(&[2.5, 0.0, 9.0]), [2.5, 1.0, 4.0]);
/// ```
pub fn isotonic(
    x: &[f64],
    y: &[f64],
    weights: Option<&[f64]>,
    increasing: bool,
) -> Result<Isotonic> {
    let n = x.len();
    if n == 0 || y.len() != n || weights.is_some_and(|w| w.len() != n) {
        return Err(Error::Data(format!(
            "give x, y and weights of one positive length; got {}, {} and {}",
            n,
            y.len(),
            weights.map_or(n, <[f64]>::len)
        )));
    }
    if let Some(&v) = x.iter().chain(y).find(|v| !v.is_finite()) {
        return Err(Error::InvalidParameter {
            name: "x, y",
            value: v,
            reason: "must be finite",
        });
    }
    if let Some(w) = weights
        && let Some(&v) = w.iter().find(|v| !(v.is_finite() && **v >= 0.0))
    {
        return Err(Error::InvalidParameter {
            name: "weights",
            value: v,
            reason: "must be finite and non-negative",
        });
    }
    let w = |i: usize| weights.map_or(1.0, |w| w[i]);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    // Merge tied inputs.
    let (mut ux, mut uy, mut uw): (Vec<f64>, Vec<f64>, Vec<f64>) = (vec![], vec![], vec![]);
    for &i in &order {
        if ux.last() == Some(&x[i]) {
            let k = ux.len() - 1;
            let total = uw[k] + w(i);
            if total > 0.0 {
                uy[k] = (uy[k] * uw[k] + y[i] * w(i)) / total;
            }
            uw[k] = total;
        } else {
            ux.push(x[i]);
            uy.push(y[i]);
            uw.push(w(i));
        }
    }
    if uw.iter().sum::<f64>() <= 0.0 {
        return Err(Error::Data("the weights sum to zero".into()));
    }
    let sign = if increasing { 1.0 } else { -1.0 };
    let target: Vec<f64> = uy.iter().map(|v| sign * v).collect();
    let fitted = pava(&target, &uw);
    let values = fitted.iter().map(|v| sign * v).collect();
    Ok(Isotonic {
        x: ux,
        values,
        increasing,
        interpolation: Interpolation::Linear,
    })
}

/// Non-decreasing pool-adjacent-violators of `y` with weights `w`; a block
/// of zero weight takes its neighbour's value.
fn pava(y: &[f64], w: &[f64]) -> Vec<f64> {
    // Blocks of (mean, weight, length).
    let mut blocks: Vec<(f64, f64, usize)> = Vec::with_capacity(y.len());
    for (&v, &wt) in y.iter().zip(w) {
        blocks.push((v, wt, 1));
        while blocks.len() > 1 {
            let (m2, w2, l2) = blocks[blocks.len() - 1];
            let (m1, w1, l1) = blocks[blocks.len() - 2];
            // A zero-weight block pools with whatever precedes it.
            if w2 > 0.0 && m1 <= m2 {
                break;
            }
            let total = w1 + w2;
            let mean = if total > 0.0 {
                (m1 * w1 + m2 * w2) / total
            } else {
                m1
            };
            blocks.truncate(blocks.len() - 2);
            blocks.push((mean, total, l1 + l2));
        }
    }
    blocks
        .iter()
        .flat_map(|&(m, _, l)| std::iter::repeat_n(m, l))
        .collect()
}

impl Isotonic {
    /// The same fit read between its points by `interpolation`.
    pub fn with_interpolation(mut self, interpolation: Interpolation) -> Self {
        self.interpolation = interpolation;
        self
    }

    /// The same fit with its values clipped to `[lo, hi]`, as
    /// scikit-learn's `y_min` and `y_max` clip them after pooling. A
    /// frequency recalibrated on held-out data can pool to 0 at its low end,
    /// where a Poisson deviance of a later claim is infinite; a floor such
    /// as a small share of the mean keeps every level positive.
    pub fn with_bounds(mut self, lo: f64, hi: f64) -> Result<Self> {
        if lo.is_nan() || hi.is_nan() || lo > hi {
            return Err(Error::Data(format!(
                "bounds must satisfy lo <= hi; got {lo} and {hi}"
            )));
        }
        for v in &mut self.values {
            *v = v.clamp(lo, hi);
        }
        Ok(self)
    }

    /// The distinct inputs, ascending.
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// The fitted value at each distinct input.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Whether the fit is non-decreasing.
    pub fn increasing(&self) -> bool {
        self.increasing
    }

    /// The fit at `x`, interpolated between its points and held at the end
    /// values beyond them (scikit-learn's `out_of_bounds="clip"`).
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.at(v)).collect()
    }

    fn at(&self, v: f64) -> f64 {
        let n = self.x.len();
        if v.is_nan() {
            return f64::NAN;
        }
        if v <= self.x[0] {
            return self.values[0];
        }
        if v >= self.x[n - 1] {
            return self.values[n - 1];
        }
        let k = self.x.partition_point(|&t| t <= v);
        let (x0, x1, y0, y1) = (self.x[k - 1], self.x[k], self.values[k - 1], self.values[k]);
        match self.interpolation {
            Interpolation::Step => y0,
            Interpolation::Linear => y0 + (y1 - y0) * (v - x0) / (x1 - x0),
        }
    }
}

/// The multiplier `Σ w y / Σ w ŷ` that makes the weighted total prediction
/// equal the weighted total response: the balance property a canonical-link
/// GLM with an intercept has on its training data. `weights` are exposures
/// when `y` and `pred` are rates.
///
/// ```
/// use prospicio_models::calibrate::balance_correction;
///
/// assert_eq!(balance_correction(&[1.0, 3.0], &[1.0, 1.0], None).unwrap(), 2.0);
/// ```
pub fn balance_correction(y: &[f64], pred: &[f64], weights: Option<&[f64]>) -> Result<f64> {
    if y.is_empty() || y.len() != pred.len() || weights.is_some_and(|w| w.len() != y.len()) {
        return Err(Error::Data(
            "give y, pred and weights of one positive length".into(),
        ));
    }
    let w = |i: usize| weights.map_or(1.0, |w| w[i]);
    let (num, den) = (0..y.len()).fold((0.0, 0.0), |(a, b), i| {
        (a + w(i) * y[i], b + w(i) * pred[i])
    });
    if !(den.is_finite() && den > 0.0) || !num.is_finite() {
        return Err(Error::Data(format!(
            "the weighted total prediction is {den}; it must be positive"
        )));
    }
    Ok(num / den)
}

/// A model whose predictions are mapped through an isotonic fit of the
/// response on its own predictions, made on held-out or out-of-fold data.
/// Its predictive distribution scales each row's draws of the inner model
/// by the ratio of the recalibrated to the inner prediction, so the
/// distribution's mean moves with the prediction and its shape is kept.
#[derive(Debug, Clone)]
pub struct Recalibrated<F> {
    inner: F,
    map: Isotonic,
}

impl<F: Fitted> Recalibrated<F> {
    /// `inner` with predictions mapped through `map`.
    pub fn new(inner: F, map: Isotonic) -> Self {
        Self { inner, map }
    }

    /// Fits the map on `design` and response `y` (weights from `weights`,
    /// or the design's when `None`): an increasing isotonic regression of
    /// `y` on the inner model's predictions there.
    pub fn fit(inner: F, design: &Design, y: &[f64], weights: Option<&[f64]>) -> Result<Self> {
        let pred = inner.predict(design)?;
        let w = weights.or(Some(design.weights()));
        let map = isotonic(&pred, y, w, true)?;
        Ok(Self { inner, map })
    }

    /// The same model with the map's values clipped to `[lo, hi]`
    /// ([`Isotonic::with_bounds`]).
    pub fn with_bounds(mut self, lo: f64, hi: f64) -> Result<Self> {
        self.map = self.map.with_bounds(lo, hi)?;
        Ok(self)
    }

    /// The inner model.
    pub fn inner(&self) -> &F {
        &self.inner
    }

    /// The isotonic map.
    pub fn map(&self) -> &Isotonic {
        &self.map
    }
}

impl<F: Fitted> Fitted for Recalibrated<F> {
    fn predict(&self, design: &Design) -> Result<Vec<f64>> {
        Ok(self.map.predict(&self.inner.predict(design)?))
    }

    fn predict_distribution(
        &self,
        design: &Design,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        let base = self.inner.predict(design)?;
        let ratio: Vec<f64> = self
            .map
            .predict(&base)
            .iter()
            .zip(&base)
            .map(|(r, b)| if *b != 0.0 { r / b } else { 1.0 })
            .collect();
        scale_rows(
            self.inner.predict_distribution(design, n_sims, seed)?,
            &ratio,
        )
    }
}

/// A model whose predictions are multiplied by one factor, as
/// [`balance_correction`] gives it.
#[derive(Debug, Clone)]
pub struct Balanced<F> {
    inner: F,
    factor: f64,
}

impl<F: Fitted> Balanced<F> {
    /// `inner` with predictions multiplied by `factor` (positive).
    pub fn new(inner: F, factor: f64) -> Result<Self> {
        if !(factor.is_finite() && factor > 0.0) {
            return Err(Error::InvalidParameter {
                name: "factor",
                value: factor,
                reason: "must be positive and finite",
            });
        }
        Ok(Self { inner, factor })
    }

    /// The factor that balances `inner` on `design` and `y` (weights from
    /// `weights`, or the design's when `None`).
    pub fn fit(inner: F, design: &Design, y: &[f64], weights: Option<&[f64]>) -> Result<Self> {
        let pred = inner.predict(design)?;
        let w = weights.or(Some(design.weights()));
        let factor = balance_correction(y, &pred, w)?;
        Self::new(inner, factor)
    }

    /// The factor.
    pub fn factor(&self) -> f64 {
        self.factor
    }

    /// The inner model.
    pub fn inner(&self) -> &F {
        &self.inner
    }
}

impl<F: Fitted> Fitted for Balanced<F> {
    fn predict(&self, design: &Design) -> Result<Vec<f64>> {
        Ok(self
            .inner
            .predict(design)?
            .iter()
            .map(|p| p * self.factor)
            .collect())
    }

    fn predict_distribution(
        &self,
        design: &Design,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        let d = self.inner.predict_distribution(design, n_sims, seed)?;
        let ratio = vec![self.factor; d.components().len()];
        scale_rows(d, &ratio)
    }
}

fn scale_rows(d: PredictiveDistribution, ratio: &[f64]) -> Result<PredictiveDistribution> {
    let k = d.components().len();
    if ratio.len() != k {
        return Err(Error::Data(format!(
            "{} predictions for {k} components",
            ratio.len()
        )));
    }
    let draws: Vec<f64> = d
        .draw_matrix()
        .iter()
        .enumerate()
        .map(|(i, x)| x * ratio[i % k])
        .collect();
    PredictiveDistribution::from_draws(
        d.dims().to_vec(),
        d.components().to_vec(),
        draws,
        d.provenance().clone(),
    )
}

/// One band of a [`reliability`] table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReliabilityBand {
    /// The lowest and highest prediction in the band.
    pub lo: f64,
    pub hi: f64,
    /// The weighted mean prediction.
    pub mean_pred: f64,
    /// The weighted mean response.
    pub mean_actual: f64,
    /// The band's total weight.
    pub weight: f64,
    /// The number of rows.
    pub count: usize,
}

/// Mean prediction against mean response in `bins` bands of about equal
/// weight, by ascending prediction (rows with the same prediction stay in
/// one band): an auto-calibrated model's bands lie on the diagonal.
///
/// ```
/// use prospicio_models::calibrate::reliability;
///
/// let r = reliability(&[0.0, 2.0, 1.0, 3.0], &[1.0, 1.0, 2.0, 2.0], None, 2).unwrap();
/// assert_eq!((r[0].mean_pred, r[0].mean_actual), (1.0, 1.0));
/// assert_eq!((r[1].mean_pred, r[1].mean_actual), (2.0, 2.0));
/// ```
pub fn reliability(
    y: &[f64],
    pred: &[f64],
    weights: Option<&[f64]>,
    bins: usize,
) -> Result<Vec<ReliabilityBand>> {
    let n = y.len();
    if n == 0 || pred.len() != n || weights.is_some_and(|w| w.len() != n) {
        return Err(Error::Data(
            "give y, pred and weights of one positive length".into(),
        ));
    }
    if bins == 0 {
        return Err(Error::InvalidParameter {
            name: "bins",
            value: 0.0,
            reason: "must be at least 1",
        });
    }
    let w = |i: usize| weights.map_or(1.0, |w| w[i]);
    let total: f64 = (0..n).map(w).sum();
    if total.is_nan() || total <= 0.0 {
        return Err(Error::Data("the weights sum to zero".into()));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| pred[a].total_cmp(&pred[b]));
    let mut bands = Vec::with_capacity(bins);
    let mut start = 0;
    let mut cum = 0.0;
    while start < n {
        let band_no = bands.len() + 1;
        let target = total * band_no as f64 / bins as f64;
        let mut end = start;
        while end < n && (end == start || cum < target - 1e-12 * total || band_no == bins) {
            cum += w(order[end]);
            end += 1;
        }
        // Keep tied predictions together.
        while end < n && pred[order[end]] == pred[order[end - 1]] {
            cum += w(order[end]);
            end += 1;
        }
        let rows = &order[start..end];
        let wt: f64 = rows.iter().map(|&i| w(i)).sum();
        let (sp, sy) = rows.iter().fold((0.0, 0.0), |(a, b), &i| {
            (a + w(i) * pred[i], b + w(i) * y[i])
        });
        bands.push(ReliabilityBand {
            lo: pred[rows[0]],
            hi: pred[rows[rows.len() - 1]],
            mean_pred: if wt > 0.0 { sp / wt } else { f64::NAN },
            mean_actual: if wt > 0.0 { sy / wt } else { f64::NAN },
            weight: wt,
            count: rows.len(),
        });
        start = end;
    }
    Ok(bands)
}

/// The CORP decomposition of a mean score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Murphy {
    /// The mean score of the predictions, `S`.
    pub score: f64,
    /// `MCB = S − S_rc`, the score the isotonic recalibration gains: 0 for
    /// an auto-calibrated model.
    pub miscalibration: f64,
    /// `DSC = S_mg − S_rc`, what the predictions' ranking is worth.
    pub discrimination: f64,
    /// `UNC = S_mg`, the score of the weighted mean response alone.
    pub uncertainty: f64,
}

/// The mean deviance of `family` (a consistent score for the mean)
/// decomposed as `S = MCB − DSC + UNC` (Dimitriadis, Gneiting and Jordan,
/// 2021), with the recalibrated predictions the increasing isotonic
/// regression of `y` on `pred` and the reference the weighted mean of `y`.
/// The Gaussian family's deviance is the squared error.
///
/// ```
/// use prospicio_models::Family;
/// use prospicio_models::calibrate::murphy;
///
/// let m = murphy(Family::Gaussian, &[0.0, 1.0, 1.0, 2.0], &[0.5, 0.5, 2.0, 2.0], None).unwrap();
/// assert!((m.score - (m.miscalibration - m.discrimination + m.uncertainty)).abs() < 1e-12);
/// ```
pub fn murphy(family: Family, y: &[f64], pred: &[f64], weights: Option<&[f64]>) -> Result<Murphy> {
    let rc = isotonic(pred, y, weights, true)?.predict(pred);
    let n = y.len();
    let w = |i: usize| weights.map_or(1.0, |w| w[i]);
    let total: f64 = (0..n).map(w).sum();
    let mean = (0..n).map(|i| w(i) * y[i]).sum::<f64>() / total;
    let score = |m: &dyn Fn(usize) -> f64| -> f64 {
        (0..n)
            .map(|i| w(i) * family.unit_deviance(y[i], m(i)))
            .sum::<f64>()
            / total
    };
    let s = score(&|i| pred[i]);
    let s_rc = score(&|i| rc[i]);
    let s_mg = score(&|_| mean);
    Ok(Murphy {
        score: s,
        miscalibration: s - s_rc,
        discrimination: s_mg - s_rc,
        uncertainty: s_mg,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pava_pools_violators_with_weights_and_ties() {
        // Ties at x = 2 merge to their weighted mean (3·1 + 1·3)/4 = 1.5.
        let f = isotonic(
            &[1.0, 2.0, 2.0, 3.0, 4.0],
            &[2.0, 1.0, 3.0, 0.0, 5.0],
            Some(&[1.0, 3.0, 1.0, 2.0, 1.0]),
            true,
        )
        .unwrap();
        assert_eq!(f.x(), [1.0, 2.0, 3.0, 4.0]);
        // 2 (w1), 1.5 (w4), 0 (w2): pooled (2 + 6 + 0)/7.
        let m = 8.0 / 7.0;
        for (a, b) in f.values().iter().zip([m, m, m, 5.0]) {
            assert!((a - b).abs() < 1e-15);
        }
        // Decreasing is the increasing fit of −y, negated.
        let d = isotonic(&[1.0, 2.0, 3.0], &[1.0, 3.0, 2.0], None, false).unwrap();
        assert_eq!(d.values(), [2.0, 2.0, 2.0]);
        let s = isotonic(&[1.0, 3.0], &[1.0, 3.0], None, true)
            .unwrap()
            .with_interpolation(Interpolation::Step);
        assert_eq!(s.predict(&[2.0, 3.0]), [1.0, 3.0]);
        assert!(isotonic(&[], &[], None, true).is_err());
        assert!(isotonic(&[1.0], &[f64::NAN], None, true).is_err());
        assert!(isotonic(&[1.0, 2.0], &[1.0, 2.0], Some(&[0.0, 0.0]), true).is_err());
        let b = isotonic(&[1.0, 2.0, 3.0], &[0.0, 1.0, 5.0], None, true)
            .unwrap()
            .with_bounds(0.5, 4.0)
            .unwrap();
        assert_eq!(b.values(), [0.5, 1.0, 4.0]);
        assert!(b.clone().with_bounds(2.0, 1.0).is_err());
    }

    #[test]
    fn the_murphy_terms_sum_to_the_score_and_vanish_when_calibrated() {
        let y: Vec<f64> = (0..200).map(|i| ((i * 37) % 11) as f64).collect();
        let pred: Vec<f64> = (0..200)
            .map(|i| 1.0 + ((i * 37) % 11) as f64 * 0.7 + (i % 3) as f64)
            .collect();
        for family in [Family::Gaussian, Family::Poisson] {
            let m = murphy(family, &y, &pred, None).unwrap();
            assert!(
                (m.score - (m.miscalibration - m.discrimination + m.uncertainty)).abs() < 1e-12
            );
            assert!(m.miscalibration >= 0.0 && m.discrimination >= 0.0);
            // The recalibrated predictions have no miscalibration left.
            let rc = isotonic(&pred, &y, None, true).unwrap().predict(&pred);
            let r = murphy(family, &y, &rc, None).unwrap();
            assert!(r.miscalibration.abs() < 1e-12, "{family:?} {r:?}");
            assert!((r.discrimination - m.discrimination).abs() < 1e-12);
        }
    }

    #[test]
    fn balance_and_reliability() {
        let y = [1.0, 2.0, 3.0, 6.0];
        let pred = [2.0, 2.0, 2.0, 2.0];
        let w = [1.0, 1.0, 2.0, 0.0];
        let f = balance_correction(&y, &pred, Some(&w)).unwrap();
        assert!((f - 9.0 / 8.0).abs() < 1e-15);
        assert!(balance_correction(&y, &[0.0; 4], None).is_err());
        // One band holds every tie.
        let r = reliability(&y, &pred, None, 3).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!((r[0].count, r[0].mean_actual), (4, 3.0));
        let r = reliability(&y, &[1.0, 2.0, 3.0, 4.0], Some(&[1.0, 1.0, 1.0, 1.0]), 4).unwrap();
        assert_eq!(r.iter().map(|b| b.count).collect::<Vec<_>>(), [1, 1, 1, 1]);
        assert!(reliability(&y, &pred, None, 0).is_err());
    }
}
