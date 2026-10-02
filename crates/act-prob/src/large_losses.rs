//! Large-loss data with the two usual defects, and maximum likelihood
//! fits of Pareto-family severities to it (see `docs/design/pareto.md`).
//!
//! - **Reporting thresholds:** loss `i` was only recorded because it
//!   exceeded `r_i`, so its likelihood is conditional on `X > r_i`.
//! - **Censoring:** a loss capped by a policy limit is known only to be at
//!   least its recorded value, and contributes `P(X ≥ y_i | X > r_i)`.
//!
//! Weights count a loss several times (or a fraction of a time).

use act_core::Result;

use crate::pareto::invalid;
use crate::piecewise_pareto::Truncation;
use crate::{Pareto, PiecewisePareto};

/// Large losses for maximum likelihood fits: values, and optionally a
/// reporting threshold, a censoring flag and a weight per loss.
///
/// ```
/// use act_prob::{LargeLosses, Pareto};
///
/// let data = LargeLosses::new(vec![1500.0, 2500.0, 4000.0, 10_000.0])
///     .unwrap()
///     .censored(vec![false, false, false, true])
///     .unwrap();
/// // α = (uncensored count) / Σ ln(y / t) = 3 / ln(1.5 · 2.5 · 4 · 10).
/// let fit = Pareto::fit(1000.0, &data, None).unwrap();
/// assert!((fit.alpha() - 3.0 / 150f64.ln()).abs() < 1e-14);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LargeLosses {
    values: Vec<f64>,
    reporting: Vec<f64>,
    censored: Vec<bool>,
    weights: Vec<f64>,
}

impl LargeLosses {
    /// Losses (finite, positive), uncensored, with weight 1 and no
    /// reporting threshold beyond the fitted distribution's own.
    pub fn new(values: Vec<f64>) -> Result<Self> {
        if values.is_empty() {
            return Err(invalid("losses", 0.0, "need at least one loss"));
        }
        for &x in &values {
            if !x.is_finite() || x <= 0.0 {
                return Err(invalid("losses", x, "must be finite and positive"));
            }
        }
        let n = values.len();
        Ok(Self {
            values,
            reporting: vec![0.0; n],
            censored: vec![false; n],
            weights: vec![1.0; n],
        })
    }

    /// Per-loss reporting thresholds `r_i ≤ y_i` (finite, non-negative).
    pub fn reporting_thresholds(self, reporting: Vec<f64>) -> Result<Self> {
        self.check_len(reporting.len())?;
        for (&r, &x) in reporting.iter().zip(&self.values) {
            if !r.is_finite() || r < 0.0 || r > x {
                return Err(invalid(
                    "reporting_thresholds",
                    r,
                    "must be finite, non-negative and at most the loss",
                ));
            }
        }
        Ok(Self { reporting, ..self })
    }

    /// Per-loss censoring: `true` where the loss was capped by a policy
    /// limit at its recorded value.
    pub fn censored(self, censored: Vec<bool>) -> Result<Self> {
        self.check_len(censored.len())?;
        Ok(Self { censored, ..self })
    }

    /// Per-loss weights (finite, positive).
    pub fn weights(self, weights: Vec<f64>) -> Result<Self> {
        self.check_len(weights.len())?;
        for &w in &weights {
            if !w.is_finite() || w <= 0.0 {
                return Err(invalid("weights", w, "must be finite and positive"));
            }
        }
        Ok(Self { weights, ..self })
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn check_len(&self, n: usize) -> Result<()> {
        if n != self.values.len() {
            return Err(invalid("length", n as f64, "must have one entry per loss"));
        }
        Ok(())
    }

    /// `(y, r, censored, w)` per loss, with `r` raised to at least `t`;
    /// fails if a loss lies below `t`.
    fn above(&self, t: f64) -> Result<Vec<(f64, f64, bool, f64)>> {
        (0..self.len())
            .map(|i| {
                let y = self.values[i];
                if y < t {
                    return Err(invalid(
                        "losses",
                        y,
                        "must be at least the lowest threshold of the fitted distribution",
                    ));
                }
                Ok((
                    y,
                    self.reporting[i].max(t),
                    self.censored[i],
                    self.weights[i],
                ))
            })
            .collect()
    }
}

/// Bounds for alphas found numerically (truncated fits).
const ALPHA_MIN: f64 = 1e-3;
const ALPHA_MAX: f64 = 1e3;

impl Pareto {
    /// Maximum likelihood fit of the alpha of `Pareto(t, α)`, optionally
    /// truncated at `truncation`, to losses at or above `t`.
    ///
    /// Each loss is conditioned on exceeding its reporting threshold
    /// raised to `t`, so `t` itself matters only through that floor.
    /// Untruncated, the estimate is in closed form,
    /// `α = Σ_uncensored w / Σ w ln(y / r)`. Truncated, the score is
    /// solved by bisection and the estimate clamped to `[1e-3, 1e3]`, as
    /// in the R package Pareto (whose upper bound is 10).
    pub fn fit(t: f64, data: &LargeLosses, truncation: Option<f64>) -> Result<Self> {
        let base = Pareto::new(t, 1.0)?;
        let losses = data.above(t)?;
        let alpha = match truncation {
            None => closed_form_alpha(&losses)?,
            Some(tr) => {
                base.truncated(tr)?;
                truncated_alpha(&losses, tr)?
            }
        };
        let p = Pareto::new(t, alpha)?;
        match truncation {
            None => Ok(p),
            Some(tr) => p.truncated(tr),
        }
    }
}

impl PiecewisePareto {
    /// Maximum likelihood fit of the alphas of a piecewise Pareto with
    /// thresholds `t` to losses at or above `t[0]`, optionally with the
    /// last piece truncated at `truncation`.
    ///
    /// The likelihood separates by piece: alpha `k` is the uncensored
    /// weight ending in piece `k` over the weighted log-exposure
    /// `Σ w ln(min(y, t_{k+1}) / max(r, t_k))⁺` inside it, and a truncated
    /// last piece is the truncated Pareto fit of the losses reaching it.
    /// Every piece needs some exposure; a piece where no uncensored loss
    /// ends gets alpha 0 (the last piece must have one). Truncation of the
    /// whole distribution couples the alphas and is not supported.
    ///
    /// ```
    /// use act_prob::{LargeLosses, PiecewisePareto};
    ///
    /// let data = LargeLosses::new(vec![1200.0, 1500.0, 2500.0, 6000.0]).unwrap();
    /// let fit = PiecewisePareto::fit(vec![1000.0, 2000.0], &data, None).unwrap();
    /// // Piece 1: 2 losses end in it; exposure ln 1.2 + ln 1.5 + 2 ln 2.
    /// let want = 2.0 / (1.2f64.ln() + 1.5f64.ln() + 2.0 * 2f64.ln());
    /// assert!((fit.alphas()[0] - want).abs() < 1e-14);
    /// ```
    pub fn fit(
        t: Vec<f64>,
        data: &LargeLosses,
        truncation: Option<(f64, Truncation)>,
    ) -> Result<Self> {
        // Validates the thresholds.
        PiecewisePareto::new(t.clone(), vec![1.0; t.len()])?;
        if let Some((_, Truncation::WholeDistribution)) = truncation {
            return Err(invalid(
                "truncation",
                f64::NAN,
                "fits with whole-distribution truncation are not supported",
            ));
        }
        let losses = data.above(t[0])?;
        let n = t.len();
        let mut alphas = Vec::with_capacity(n);
        for k in 0..n {
            let (lo, hi) = (t[k], t.get(k + 1).copied().unwrap_or(f64::INFINITY));
            let last = k + 1 == n;
            if let (true, Some((tr, _))) = (last, truncation) {
                // Losses reaching the last piece, conditioned on exceeding
                // max(r, t_n): a truncated Pareto fit.
                let reaching: Vec<_> = losses
                    .iter()
                    .filter(|&&(y, ..)| y > lo)
                    .map(|&(y, r, c, w)| (y, r.max(lo), c, w))
                    .collect();
                PiecewisePareto::new(t.clone(), vec![1.0; n])?
                    .truncated(tr, Truncation::LastPiece)?;
                alphas.push(truncated_alpha(&reaching, tr)?);
                continue;
            }
            let (mut count, mut exposure) = (0.0, 0.0);
            for &(y, r, censored, w) in &losses {
                let (a, b) = (r.max(lo), y.min(hi));
                if b > a {
                    exposure += w * (b / a).ln();
                }
                if !censored && y >= lo && y < hi {
                    count += w;
                }
            }
            if exposure <= 0.0 {
                return Err(invalid(
                    "t",
                    lo,
                    "no loss reaches into this piece above its reporting threshold",
                ));
            }
            alphas.push(count / exposure);
        }
        let fit = PiecewisePareto::new(t, alphas)?;
        match truncation {
            None => Ok(fit),
            Some((tr, kind)) => fit.truncated(tr, kind),
        }
    }
}

/// `Σ_uncensored w / Σ w ln(y / r)`.
fn closed_form_alpha(losses: &[(f64, f64, bool, f64)]) -> Result<f64> {
    let count: f64 = losses.iter().filter(|l| !l.2).map(|l| l.3).sum();
    let exposure: f64 = losses.iter().map(|&(y, r, _, w)| w * (y / r).ln()).sum();
    if count == 0.0 {
        return Err(invalid("losses", 0.0, "need at least one uncensored loss"));
    }
    if exposure <= 0.0 {
        return Err(invalid(
            "losses",
            exposure,
            "need a loss above its reporting threshold",
        ));
    }
    Ok(count / exposure)
}

/// The alpha of a Pareto truncated at `tr` that maximizes the conditional
/// likelihood of `losses` (each above its `r`), by bisection on the score,
/// clamped to `[ALPHA_MIN, ALPHA_MAX]`. (A truncated Pareto is a
/// distribution for any alpha, and data that rises towards the truncation
/// point can put the maximum at or below 0.)
///
/// With `g(r) = d/dα ln(r^(−α) − T^(−α)) = −ln r + ln(T/r) q/(1 − q)` and
/// `q = (r/T)^α`, an uncensored loss contributes `1/α − ln y − g(r)` and a
/// censored one `g(y) − g(r)`.
fn truncated_alpha(losses: &[(f64, f64, bool, f64)], tr: f64) -> Result<f64> {
    if losses.iter().all(|l| l.2) {
        return Err(invalid("losses", 0.0, "need at least one uncensored loss"));
    }
    for &(y, ..) in losses {
        if y >= tr {
            return Err(invalid("losses", y, "must lie below the truncation point"));
        }
    }
    let g = |x: f64, alpha: f64| -> f64 {
        let log_q = alpha * (x / tr).ln();
        // q / (1 − q) = 1 / expm1(−ln q).
        -x.ln() + (tr / x).ln() / (-log_q).exp_m1()
    };
    let score = |alpha: f64| -> f64 {
        losses
            .iter()
            .map(|&(y, r, censored, w)| {
                let own = if censored {
                    g(y, alpha)
                } else {
                    1.0 / alpha - y.ln()
                };
                w * (own - g(r, alpha))
            })
            .sum()
    };
    // The score falls through 0 at the maximum: find the first sign change
    // on a log grid, then bisect.
    let steps = 240;
    let at =
        |j: usize| (ALPHA_MIN.ln() + (ALPHA_MAX / ALPHA_MIN).ln() * j as f64 / steps as f64).exp();
    if score(at(0)) <= 0.0 {
        return Ok(ALPHA_MIN);
    }
    let Some(j) = (1..=steps).find(|&j| score(at(j)) <= 0.0) else {
        return Ok(ALPHA_MAX);
    };
    let (mut lo, mut hi) = (at(j - 1), at(j));
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if score(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Distribution, Severity};
    use act_core::StreamRng;

    #[test]
    fn closed_form_pareto_with_thresholds_and_censoring() {
        let data = LargeLosses::new(vec![1500.0, 3000.0, 2500.0, 8000.0])
            .unwrap()
            .reporting_thresholds(vec![0.0, 2000.0, 1000.0, 0.0])
            .unwrap()
            .censored(vec![false, false, false, true])
            .unwrap()
            .weights(vec![1.0, 2.0, 1.0, 1.0])
            .unwrap();
        let fit = Pareto::fit(1000.0, &data, None).unwrap();
        let exposure = 1.5f64.ln() + 2.0 * 1.5f64.ln() + 2.5f64.ln() + 8f64.ln();
        assert!((fit.alpha() - 4.0 / exposure).abs() < 1e-14);
        assert!(Pareto::fit(2000.0, &data, None).is_err());
        let all_censored = LargeLosses::new(vec![2000.0])
            .unwrap()
            .censored(vec![true])
            .unwrap();
        assert!(Pareto::fit(1000.0, &all_censored, None).is_err());
    }

    #[test]
    fn truncated_fit_maximizes_the_likelihood() {
        let data = LargeLosses::new(vec![1100.0, 1300.0, 2000.0, 3500.0, 9000.0, 4000.0])
            .unwrap()
            .reporting_thresholds(vec![0.0, 1200.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap()
            .censored(vec![false, false, false, false, false, true])
            .unwrap();
        let tr = 20_000.0;
        let fit = Pareto::fit(1000.0, &data, Some(tr)).unwrap();
        let ll = |alpha: f64| -> f64 {
            let p = Pareto::new(1000.0, alpha).unwrap().truncated(tr).unwrap();
            let mut ll = 0.0;
            for (i, &y) in data.values().iter().enumerate() {
                let r = data.reporting[i].max(1000.0);
                ll -= p.survival(r).ln();
                ll += if data.censored[i] {
                    p.survival(y).ln()
                } else {
                    // Density by the closed form: α t^α y^(−α−1) / (1 − q).
                    let q = (1000.0 / tr).powf(alpha);
                    (alpha * 1000f64.powf(alpha) * y.powf(-alpha - 1.0) / (1.0 - q)).ln()
                };
            }
            ll
        };
        let a = fit.alpha();
        assert!(ll(a) >= ll(a * 1.001) && ll(a) >= ll(a * 0.999), "{a}");
        // Truncation far away gives the untruncated answer.
        let far = Pareto::fit(1000.0, &data, Some(1e30)).unwrap();
        let open = Pareto::fit(1000.0, &data, None).unwrap();
        assert!((far.alpha() / open.alpha() - 1.0).abs() < 1e-9);
        assert!(Pareto::fit(1000.0, &data, Some(5000.0)).is_err());
    }

    #[test]
    fn fits_recover_simulated_parameters() {
        let truth =
            PiecewisePareto::new(vec![1000.0, 3000.0, 10_000.0], vec![1.2, 2.0, 1.5]).unwrap();
        let mut rng = StreamRng::new(11, 0);
        let draws = truth.sample(&mut rng, 200_000);
        // Policy limits censor at 50,000; a reporting threshold of 2,000 on
        // every other loss drops the ones below it.
        let (mut values, mut reporting, mut censored) = (vec![], vec![], vec![]);
        for (i, &x) in draws.iter().enumerate() {
            let r = if i % 2 == 0 { 2000.0 } else { 0.0 };
            if x <= r {
                continue;
            }
            values.push(x.min(50_000.0));
            reporting.push(r);
            censored.push(x >= 50_000.0);
        }
        let data = LargeLosses::new(values)
            .unwrap()
            .reporting_thresholds(reporting)
            .unwrap()
            .censored(censored)
            .unwrap();
        let fit = PiecewisePareto::fit(vec![1000.0, 3000.0, 10_000.0], &data, None).unwrap();
        for (got, want) in fit.alphas().iter().zip(truth.alphas()) {
            assert!((got / want - 1.0).abs() < 0.03, "{got} {want}");
        }
        // One piece: the Pareto fit.
        let one = PiecewisePareto::fit(vec![1000.0], &data, None).unwrap();
        let p = Pareto::fit(1000.0, &data, None).unwrap();
        assert!((one.alphas()[0] / p.alpha() - 1.0).abs() < 1e-14);
        // A truncated last piece.
        let t = PiecewisePareto::new(vec![1000.0, 3000.0], vec![1.2, 0.8])
            .unwrap()
            .truncated(40_000.0, Truncation::LastPiece)
            .unwrap();
        let draws = t.sample(&mut StreamRng::new(12, 0), 100_000);
        let data = LargeLosses::new(draws).unwrap();
        let fit = PiecewisePareto::fit(
            vec![1000.0, 3000.0],
            &data,
            Some((40_000.0, Truncation::LastPiece)),
        )
        .unwrap();
        assert!(
            (fit.alphas()[1] / 0.8 - 1.0).abs() < 0.03,
            "{:?}",
            fit.alphas()
        );
        assert!(fit.layer(1e4, 1e4) > 0.0);
        assert!(
            PiecewisePareto::fit(
                vec![1000.0, 3000.0],
                &data,
                Some((40_000.0, Truncation::WholeDistribution)),
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_bad_data() {
        assert!(LargeLosses::new(vec![]).is_err());
        assert!(LargeLosses::new(vec![-1.0]).is_err());
        let d = LargeLosses::new(vec![10.0, 20.0]).unwrap();
        assert!(d.clone().reporting_thresholds(vec![15.0, 0.0]).is_err());
        assert!(d.clone().weights(vec![1.0]).is_err());
        assert!(d.clone().weights(vec![1.0, 0.0]).is_err());
        // No loss reaches the top piece.
        assert!(PiecewisePareto::fit(vec![5.0, 50.0], &d, None).is_err());
    }
}
