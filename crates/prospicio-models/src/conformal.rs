//! Conformal prediction intervals around any model (#546).
//!
//! Conformal methods give a prediction interval with coverage of at least
//! `1 − α` in finite samples, under exchangeability alone, from held-out
//! residuals: no family is assumed. Every method here takes the conformal
//! quantile of a score as the `⌈(1 − α)(n + 1)⌉`-th smallest of `n` scores
//! (infinite when that rank exceeds `n`), as MAPIE's split and CV+ methods
//! do:
//!
//! - [`split`]: a model fitted on one set, its scores on a calibration set,
//!   `ŷ ± q` (absolute residuals), `μ ± q √V(μ)` (Pearson residuals of a
//!   family) or `ŷ ± q s` (residuals scaled by a given spread);
//! - [`weighted_split`]: the same with a weight for each calibration row and
//!   test row (Tibshirani, Barber, Candès and Ramdas, 2019);
//! - [`cqr`]: conformalised quantile regression on a lower and an upper
//!   quantile model (Romano, Patterson and Candès, 2019);
//! - [`cv_plus`]: CV+ and, with one fold a row, the jackknife+ (Barber,
//!   Candès, Ramdas and Tibshirani, 2021), from each fold's model's
//!   out-of-fold and test predictions, with coverage of at least `1 − 2α`;
//!   [`cv_plus_fit`] fits the folds of any [`Model`] in parallel.

use prospicio_core::{Error, Result};

use crate::design::Design;
use crate::family::Family;
use crate::model::{Fitted, Model};
use crate::resample::{Split, map_splits};

/// Prediction intervals, one lower and upper bound a row.
#[derive(Debug, Clone, PartialEq)]
pub struct Intervals {
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
}

/// The conformity score of [`split`].
#[derive(Debug, Clone, PartialEq)]
pub enum Score {
    /// `|y − ŷ|`; the interval is `ŷ ± q`.
    Absolute,
    /// `|y − μ| / √V(μ)` for the family's variance function; the interval is
    /// `μ ± q √V(μ)`.
    Pearson(Family),
    /// `|y − ŷ| / s` with a given spread `s > 0` for each calibration row
    /// and each test row (a dispersion model's, say); the interval is
    /// `ŷ ± q s`.
    Scaled {
        calibration: Vec<f64>,
        test: Vec<f64>,
    },
}

fn check_alpha(alpha: f64) -> Result<()> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(Error::InvalidParameter {
            name: "alpha",
            value: alpha,
            reason: "must be in (0, 1)",
        });
    }
    Ok(())
}

/// The rank `⌈(1 − α)(n + 1)⌉` of the conformal quantile among `n` scores.
fn rank(alpha: f64, n: usize) -> usize {
    ((1.0 - alpha) * (n as f64 + 1.0)).ceil() as usize
}

/// The conformal quantile of `scores`: the `⌈(1 − α)(n + 1)⌉`-th smallest,
/// or infinity when that rank exceeds `n` (too few scores for the level).
///
/// ```
/// use prospicio_models::conformal::conformal_quantile;
///
/// // n = 9, α = 0.2: rank ⌈0.8 · 10⌉ = 8.
/// let s = [5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0];
/// assert_eq!(conformal_quantile(&s, 0.2).unwrap(), 8.0);
/// assert_eq!(conformal_quantile(&s[..3], 0.2).unwrap(), f64::INFINITY);
/// ```
pub fn conformal_quantile(scores: &[f64], alpha: f64) -> Result<f64> {
    check_alpha(alpha)?;
    if scores.is_empty() {
        return Err(Error::Data(
            "conformal intervals need calibration scores".into(),
        ));
    }
    if let Some(&s) = scores.iter().find(|s| s.is_nan()) {
        return Err(Error::InvalidParameter {
            name: "scores",
            value: s,
            reason: "must not be NaN",
        });
    }
    let k = rank(alpha, scores.len());
    if k > scores.len() {
        return Ok(f64::INFINITY);
    }
    let mut s = scores.to_vec();
    s.sort_by(f64::total_cmp);
    Ok(s[k - 1])
}

fn check_len(what: &str, got: usize, want: usize) -> Result<()> {
    if got != want {
        return Err(Error::Data(format!("{got} {what} for {want} rows")));
    }
    Ok(())
}

fn spreads(score: &Score, cal_pred: &[f64], test_pred: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    let (c, t): (Vec<f64>, Vec<f64>) = match score {
        Score::Absolute => (vec![1.0; cal_pred.len()], vec![1.0; test_pred.len()]),
        Score::Pearson(f) => (
            cal_pred.iter().map(|&m| f.variance(m).sqrt()).collect(),
            test_pred.iter().map(|&m| f.variance(m).sqrt()).collect(),
        ),
        Score::Scaled { calibration, test } => {
            check_len("calibration spreads", calibration.len(), cal_pred.len())?;
            check_len("test spreads", test.len(), test_pred.len())?;
            (calibration.clone(), test.clone())
        }
    };
    if let Some(&s) = c.iter().chain(&t).find(|s| !(s.is_finite() && **s > 0.0)) {
        return Err(Error::InvalidParameter {
            name: "spread",
            value: s,
            reason: "must be positive and finite (a family's variance at a prediction, or a given scale)",
        });
    }
    Ok((c, t))
}

/// Split conformal intervals: the model's predictions `cal_pred` on a
/// calibration set with responses `y`, scored by `score`, give the
/// quantile `q`, and each test prediction its interval.
///
/// ```
/// use prospicio_models::conformal::{Score, split};
///
/// let y: Vec<f64> = (0..19).map(|i| i as f64).collect();
/// let pred = vec![9.0; 19];
/// // Residuals 0 to 9, each twice but 9: at α = 0.1 the rank is 18, q = 9.
/// let i = split(&y, &pred, &[100.0], 0.1, &Score::Absolute).unwrap();
/// assert_eq!((i.lo[0], i.hi[0]), (91.0, 109.0));
/// ```
pub fn split(
    y: &[f64],
    cal_pred: &[f64],
    test_pred: &[f64],
    alpha: f64,
    score: &Score,
) -> Result<Intervals> {
    check_len("predictions", cal_pred.len(), y.len())?;
    let (cs, ts) = spreads(score, cal_pred, test_pred)?;
    let scores: Vec<f64> = y
        .iter()
        .zip(cal_pred)
        .zip(&cs)
        .map(|((y, p), s)| (y - p).abs() / s)
        .collect();
    let q = conformal_quantile(&scores, alpha)?;
    Ok(Intervals {
        lo: test_pred.iter().zip(&ts).map(|(p, s)| p - q * s).collect(),
        hi: test_pred.iter().zip(&ts).map(|(p, s)| p + q * s).collect(),
    })
}

/// Weighted split conformal intervals: each calibration row's score carries
/// its weight and each test row its own (likelihood ratios under a covariate
/// shift, or exposures), and the quantile is that of the weighted scores
/// with the test row's weight placed at infinity: the smallest score whose
/// share of the total weight, the test row's included, reaches `1 − α`.
/// Equal weights give [`split`].
pub fn weighted_split(
    y: &[f64],
    cal_pred: &[f64],
    test_pred: &[f64],
    alpha: f64,
    score: &Score,
    cal_weights: &[f64],
    test_weights: &[f64],
) -> Result<Intervals> {
    check_alpha(alpha)?;
    check_len("predictions", cal_pred.len(), y.len())?;
    check_len("calibration weights", cal_weights.len(), y.len())?;
    check_len("test weights", test_weights.len(), test_pred.len())?;
    if let Some(&w) = cal_weights
        .iter()
        .chain(test_weights)
        .find(|w| !(w.is_finite() && **w >= 0.0))
    {
        return Err(Error::InvalidParameter {
            name: "weights",
            value: w,
            reason: "must be finite and non-negative",
        });
    }
    let (cs, ts) = spreads(score, cal_pred, test_pred)?;
    let mut scored: Vec<(f64, f64)> = y
        .iter()
        .zip(cal_pred)
        .zip(&cs)
        .zip(cal_weights)
        .map(|(((y, p), s), w)| ((y - p).abs() / s, *w))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    let cal_total: f64 = cal_weights.iter().sum();
    let mut lo = Vec::with_capacity(test_pred.len());
    let mut hi = Vec::with_capacity(test_pred.len());
    for ((p, s), &wt) in test_pred.iter().zip(&ts).zip(test_weights) {
        let total = cal_total + wt;
        if total <= 0.0 {
            return Err(Error::Data("the weights sum to zero".into()));
        }
        let target = (1.0 - alpha) * total;
        let mut cum = 0.0;
        let mut q = f64::INFINITY;
        for &(sc, w) in &scored {
            cum += w;
            // A relative tolerance keeps equal weights at split's rank.
            if cum >= target * (1.0 - 1e-12) {
                q = sc;
                break;
            }
        }
        lo.push(p - q * s);
        hi.push(p + q * s);
    }
    Ok(Intervals { lo, hi })
}

/// Conformalised quantile regression: from a lower and an upper quantile
/// model's predictions on a calibration set (`cal_lo`, `cal_hi`) and the
/// test set, the score `max(lo − y, y − hi)` and the intervals
/// `[lo − q, hi + q]`. The quantile models are fitted at `α/2` and
/// `1 − α/2`, say; `q` may be negative, which narrows them.
pub fn cqr(
    y: &[f64],
    cal_lo: &[f64],
    cal_hi: &[f64],
    test_lo: &[f64],
    test_hi: &[f64],
    alpha: f64,
) -> Result<Intervals> {
    check_len("lower predictions", cal_lo.len(), y.len())?;
    check_len("upper predictions", cal_hi.len(), y.len())?;
    check_len("upper test predictions", test_hi.len(), test_lo.len())?;
    let scores: Vec<f64> = y
        .iter()
        .zip(cal_lo.iter().zip(cal_hi))
        .map(|(y, (lo, hi))| (lo - y).max(y - hi))
        .collect();
    let q = conformal_quantile(&scores, alpha)?;
    Ok(Intervals {
        lo: test_lo.iter().map(|v| v - q).collect(),
        hi: test_hi.iter().map(|v| v + q).collect(),
    })
}

/// CV+ intervals (the jackknife+ with one fold a row): `y` the training
/// responses, `oof_pred[i]` row `i`'s prediction by the model fitted
/// without its fold `fold[i]`, and `test_pred[k]` fold `k`'s model's
/// predictions on the test rows. A test row's bounds are the
/// `⌊α(n + 1)⌋`-th smallest of `μ₋ₖ₍ᵢ₎(x) − Rᵢ` and the
/// `⌈(1 − α)(n + 1)⌉`-th smallest of `μ₋ₖ₍ᵢ₎(x) + Rᵢ` over the training
/// rows, `Rᵢ = |yᵢ − oof_pred[i]|` (infinite when a rank falls outside
/// `1..=n`).
pub fn cv_plus(
    y: &[f64],
    oof_pred: &[f64],
    fold: &[usize],
    test_pred: &[Vec<f64>],
    alpha: f64,
) -> Result<Intervals> {
    check_alpha(alpha)?;
    let n = y.len();
    check_len("out-of-fold predictions", oof_pred.len(), n)?;
    check_len("folds", fold.len(), n)?;
    if n == 0 || test_pred.is_empty() {
        return Err(Error::Data(
            "CV+ needs training rows and fold predictions".into(),
        ));
    }
    let m = test_pred[0].len();
    if test_pred.iter().any(|t| t.len() != m) {
        return Err(Error::Data(
            "every fold's model must predict every test row".into(),
        ));
    }
    if let Some(&k) = fold.iter().find(|&&k| k >= test_pred.len()) {
        return Err(Error::Data(format!(
            "fold {k} has no test predictions ({} folds given)",
            test_pred.len()
        )));
    }
    let r: Vec<f64> = y.iter().zip(oof_pred).map(|(y, p)| (y - p).abs()).collect();
    let k_hi = rank(alpha, n);
    // ⌊α(n + 1)⌋ = n + 1 − ⌈(1 − α)(n + 1)⌉.
    let k_lo = (n + 1).saturating_sub(k_hi);
    let (mut lo, mut hi) = (Vec::with_capacity(m), Vec::with_capacity(m));
    let mut low = vec![0.0; n];
    let mut up = vec![0.0; n];
    // `test_pred` is indexed by each training row's fold, then by `j`.
    #[allow(clippy::needless_range_loop)]
    for j in 0..m {
        for (((l, u), &k), ri) in low.iter_mut().zip(up.iter_mut()).zip(fold).zip(&r) {
            let mu = test_pred[k][j];
            *l = mu - ri;
            *u = mu + ri;
        }
        lo.push(if k_lo == 0 {
            f64::NEG_INFINITY
        } else {
            low.select_nth_unstable_by(k_lo - 1, f64::total_cmp);
            low[k_lo - 1]
        });
        hi.push(if k_hi > n {
            f64::INFINITY
        } else {
            up.select_nth_unstable_by(k_hi - 1, f64::total_cmp);
            up[k_hi - 1]
        });
    }
    Ok(Intervals { lo, hi })
}

/// Split conformal intervals of a fitted model: scored on the calibration
/// design `cal` with responses `y`, applied to `test`.
pub fn split_fitted<F: Fitted>(
    fitted: &F,
    cal: &Design,
    y: &[f64],
    test: &Design,
    alpha: f64,
    score: &Score,
) -> Result<Intervals> {
    split(
        y,
        &fitted.predict(cal)?,
        &fitted.predict(test)?,
        alpha,
        score,
    )
}

/// CV+ intervals of `model` on `design` and `y` over `splits` (each row in
/// one split's test set; `resample::k_fold`, or one row a split for the
/// jackknife+), the folds fitted in parallel, applied to `test`.
pub fn cv_plus_fit<M>(
    model: &M,
    design: &Design,
    y: &[f64],
    splits: &[Split],
    test: &Design,
    alpha: f64,
) -> Result<Intervals>
where
    M: Model + Sync,
{
    let n = y.len();
    check_len("responses", n, design.n_rows())?;
    let mut fold = vec![usize::MAX; n];
    for (k, s) in splits.iter().enumerate() {
        for &i in &s.test {
            if i >= n || fold[i] != usize::MAX {
                return Err(Error::Data(
                    "each row must be in exactly one split's test set".into(),
                ));
            }
            fold[i] = k;
        }
    }
    if fold.contains(&usize::MAX) {
        return Err(Error::Data(
            "each row must be in exactly one split's test set".into(),
        ));
    }
    let per_fold = map_splits(splits, |s| {
        let yt: Vec<f64> = s.train.iter().map(|&i| y[i]).collect();
        let f = model.fit(&design.select(&s.train), &yt)?;
        Ok((f.predict(&design.select(&s.test))?, f.predict(test)?))
    })?;
    let mut oof = vec![0.0; n];
    let mut test_pred = Vec::with_capacity(splits.len());
    for (s, (held, t)) in splits.iter().zip(per_fold) {
        for (&i, p) in s.test.iter().zip(held) {
            oof[i] = p;
        }
        test_pred.push(t);
    }
    cv_plus(y, &oof, &fold, &test_pred, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_equal_weights_is_split_and_a_heavy_test_row_widens() {
        let y: Vec<f64> = (0..50).map(|i| ((i * 13) % 17) as f64).collect();
        let pred = vec![8.0; 50];
        let a = split(&y, &pred, &[1.0, 2.0], 0.2, &Score::Absolute).unwrap();
        let b = weighted_split(
            &y,
            &pred,
            &[1.0, 2.0],
            0.2,
            &Score::Absolute,
            &[1.0; 50],
            &[1.0; 2],
        )
        .unwrap();
        assert_eq!(a, b);
        let c = weighted_split(
            &y,
            &pred,
            &[1.0],
            0.2,
            &Score::Absolute,
            &[1.0; 50],
            &[100.0],
        )
        .unwrap();
        assert_eq!(c.hi[0], f64::INFINITY);
    }

    #[test]
    fn pearson_scales_by_the_variance_and_inputs_are_checked() {
        let y = [1.0, 4.0, 9.0, 16.0];
        let pred = [1.0, 4.0, 9.0, 16.0];
        let i = split(&y, &pred, &[4.0], 0.5, &Score::Pearson(Family::Poisson)).unwrap();
        assert_eq!((i.lo[0], i.hi[0]), (4.0, 4.0));
        assert!(split(&y, &pred[..3], &[1.0], 0.1, &Score::Absolute).is_err());
        assert!(split(&y, &pred, &[1.0], 1.5, &Score::Absolute).is_err());
        assert!(split(&y, &pred, &[0.0], 0.5, &Score::Pearson(Family::Poisson)).is_err());
        let scaled = Score::Scaled {
            calibration: vec![1.0, 2.0, 3.0, 4.0],
            test: vec![2.0],
        };
        assert!(split(&y, &pred, &[1.0], 0.5, &scaled).is_ok());
        assert!(cv_plus(&y, &pred, &[0, 0, 1, 2], &[vec![1.0], vec![2.0]], 0.1).is_err());
    }

    /// Coverage over many seeds: split and CQR at least 1 − α on average,
    /// CV+ at least 1 − 2α, each within its Monte Carlo error.
    #[test]
    fn coverage_holds_over_many_seeds() {
        use prospicio_core::StreamRng;
        let normal = |rng: &mut StreamRng| {
            let (u, v) = (rng.next_open01(), rng.next_open01());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        };
        let (alpha, reps) = (0.1, 400);
        let (mut split_hits, mut cqr_hits, mut plus_hits) = (0usize, 0usize, 0usize);
        for r in 0..reps {
            let mut rng = StreamRng::new(9, r);
            // y = 2x + noise with sd 1 + x; a deliberately wrong model 1.8x.
            let mut draw = || {
                let x = 3.0 * rng.next_open01();
                (x, 2.0 * x + (1.0 + x) * normal(&mut rng))
            };
            let cal: Vec<(f64, f64)> = (0..30).map(|_| draw()).collect();
            let (xt, yt) = draw();
            let y: Vec<f64> = cal.iter().map(|c| c.1).collect();
            let p: Vec<f64> = cal.iter().map(|c| 1.8 * c.0).collect();
            let i = split(&y, &p, &[1.8 * xt], alpha, &Score::Absolute).unwrap();
            split_hits += usize::from(i.lo[0] <= yt && yt <= i.hi[0]);
            let lo: Vec<f64> = cal.iter().map(|c| 1.8 * c.0 - 1.5 * (1.0 + c.0)).collect();
            let hi: Vec<f64> = cal.iter().map(|c| 1.8 * c.0 + 1.5 * (1.0 + c.0)).collect();
            let q = cqr(
                &y,
                &lo,
                &hi,
                &[1.8 * xt - 1.5 * (1.0 + xt)],
                &[1.8 * xt + 1.5 * (1.0 + xt)],
                alpha,
            )
            .unwrap();
            cqr_hits += usize::from(q.lo[0] <= yt && yt <= q.hi[0]);
            // Jackknife+ of the mean's slope through the origin.
            let n = y.len();
            let slope = |skip: usize| {
                let (sxy, sxx) = cal
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k != skip)
                    .fold((0.0, 0.0), |(a, b), (_, c)| (a + c.0 * c.1, b + c.0 * c.0));
                sxy / sxx
            };
            let oof: Vec<f64> = (0..n).map(|k| slope(k) * cal[k].0).collect();
            let test: Vec<Vec<f64>> = (0..n).map(|k| vec![slope(k) * xt]).collect();
            let fold: Vec<usize> = (0..n).collect();
            let j = cv_plus(&y, &oof, &fold, &test, alpha).unwrap();
            plus_hits += usize::from(j.lo[0] <= yt && yt <= j.hi[0]);
        }
        let se = (alpha * (1.0 - alpha) / reps as f64).sqrt();
        let cover = |h: usize| h as f64 / reps as f64;
        assert!(
            cover(split_hits) >= 1.0 - alpha - 3.0 * se,
            "{}",
            cover(split_hits)
        );
        assert!(
            cover(cqr_hits) >= 1.0 - alpha - 3.0 * se,
            "{}",
            cover(cqr_hits)
        );
        assert!(
            cover(plus_hits) >= 1.0 - 2.0 * alpha - 3.0 * se,
            "{}",
            cover(plus_hits)
        );
    }
}
