//! Large-loss data with the two usual defects, and maximum likelihood
//! fits of Pareto-family severities to it (see `docs/design/pareto.md`).
//!
//! - **Reporting thresholds:** loss `i` was only recorded because it
//!   exceeded `r_i`, so its likelihood is conditional on `X > r_i`.
//! - **Censoring:** a loss capped by a policy limit is known only to be at
//!   least its recorded value, and contributes `P(X ≥ y_i | X > r_i)`.
//!
//! Weights count a loss several times (or a fraction of a time).

use prospicio_core::Result;
use prospicio_math::roots::{bisect, bisect_log, illinois};

use crate::evt::Gpd;
use crate::pareto::invalid;
use crate::piecewise_pareto::Truncation;
use crate::{Pareto, PiecewisePareto};

/// Large losses for maximum likelihood fits: values, and optionally a
/// reporting threshold, a censoring flag and a weight per loss.
///
/// ```
/// use prospicio_prob::{LargeLosses, Pareto};
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
    /// ends gets alpha 0 (the last piece must have one).
    ///
    /// Truncation of the whole distribution at `T` couples the alphas,
    /// since each loss's likelihood is conditioned through `S(r) − S(T)`.
    /// The fit then maximizes the likelihood by coordinate ascent from the
    /// untruncated estimates, each alpha by bisection on its analytic
    /// partial derivative and clamped to `[1e-3, 1e3]` (as the R package
    /// clamps to its bounds). `S(a) − S(T)` is computed as
    /// `S(a) (1 − e^D)` with `D` summed piece by piece over `[a, T]`, so
    /// losses just below `T` keep their precision.
    ///
    /// ```
    /// use prospicio_prob::{LargeLosses, PiecewisePareto};
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
        if let Some((tr, Truncation::WholeDistribution)) = truncation {
            PiecewisePareto::new(t.clone(), vec![1.0; t.len()])?
                .truncated(tr, Truncation::WholeDistribution)?;
            let start = PiecewisePareto::fit(t.clone(), data, None)?;
            let losses = data.above(t[0])?;
            let alphas = whole_truncated_alphas(&t, &losses, tr, start.alphas())?;
            return PiecewisePareto::new(t, alphas)?.truncated(tr, Truncation::WholeDistribution);
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

impl Gpd {
    /// Maximum likelihood fit of Riegel's generalized Pareto
    /// ([`Gpd::riegel`]) with threshold `t` to losses at or above `t`,
    /// with the reporting thresholds, censoring and weights of `data`.
    ///
    /// With `z = x/t − 1` and `k = α_ini / α_tail`, `S(x) = (1 + k z)^(−α_tail)`.
    /// For fixed `k` the likelihood is maximized by
    /// `α_tail(k) = Σ_uncensored w / Σ w ln((1 + k z_y) / (1 + k z_r))`, so
    /// the fit is a one-dimensional profile likelihood in `k`: a scan on a
    /// log grid over `[1e-6, 1e6]` brackets the maximum, and bisection on the
    /// analytic score pins it. `k = 1` is the Pareto.
    ///
    /// ```
    /// use prospicio_core::StreamRng;
    /// use prospicio_prob::{Distribution, LargeLosses, evt::Gpd};
    ///
    /// let truth = Gpd::riegel(1000.0, 3.0, 1.5).unwrap();
    /// let draws = truth.sample(&mut StreamRng::new(4, 0), 100_000);
    /// let fit = Gpd::fit_riegel(1000.0, &LargeLosses::new(draws).unwrap()).unwrap();
    /// // ξ = 1/α_tail, β = t/α_ini.
    /// assert!((1.0 / fit.xi() / 1.5 - 1.0).abs() < 0.05);
    /// assert!((1000.0 / fit.beta() / 3.0 - 1.0).abs() < 0.05);
    /// ```
    pub fn fit_riegel(t: f64, data: &LargeLosses) -> Result<Self> {
        Pareto::new(t, 1.0)?;
        let losses = data.above(t)?;
        let count: f64 = losses.iter().filter(|l| !l.2).map(|l| l.3).sum();
        if count == 0.0 {
            return Err(invalid("losses", 0.0, "need at least one uncensored loss"));
        }
        let z = |x: f64| x / t - 1.0;
        // Σ w ln((1 + k z_y) / (1 + k z_r)) and its derivative in k.
        let exposure = |k: f64| -> (f64, f64) {
            losses.iter().fold((0.0, 0.0), |(e, de), &(y, r, _, w)| {
                let (zy, zr) = (z(y), z(r));
                (
                    e + w * ((k * zy).ln_1p() - (k * zr).ln_1p()),
                    de + w * (zy / (1.0 + k * zy) - zr / (1.0 + k * zr)),
                )
            })
        };
        // The profile log-likelihood is n ln(k α_tail(k)) − Σ_uncensored w
        // ln(1 + k z_y) up to a constant; its derivative in k is
        // n/k − n E'(k)/E(k) − Σ_uncensored w z_y/(1 + k z_y).
        let score = |k: f64| -> Option<f64> {
            let (e, de) = exposure(k);
            if e <= 0.0 {
                return None;
            }
            let own: f64 = losses
                .iter()
                .filter(|l| !l.2)
                .map(|&(y, _, _, w)| w * z(y) / (1.0 + k * z(y)))
                .sum();
            Some(count / k - count * de / e - own)
        };
        let profile = |k: f64| -> Option<f64> {
            let (e, _) = exposure(k);
            if e <= 0.0 {
                return None;
            }
            let alpha_tail = count / e;
            let own: f64 = losses
                .iter()
                .filter(|l| !l.2)
                .map(|&(y, _, _, w)| w * (k * z(y)).ln_1p())
                .sum();
            // At the optimal α_tail the α_tail-terms sum to −n, a constant.
            Some(count * (k * alpha_tail).ln() - own)
        };
        let steps = 480;
        let (lo_k, hi_k) = (1e-6f64, 1e6f64);
        let at = |j: usize| (lo_k.ln() + (hi_k / lo_k).ln() * j as f64 / steps as f64).exp();
        let best = (0..=steps)
            .filter_map(|j| profile(at(j)).map(|v| (j, v)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(j, _)| j)
            .ok_or_else(|| {
                invalid(
                    "losses",
                    f64::NAN,
                    "need a loss above its reporting threshold",
                )
            })?;
        if best == 0 || best == steps {
            return Err(invalid(
                "losses",
                at(best),
                "the likelihood has no maximum with alpha_ini / alpha_tail in [1e-6, 1e6]",
            ));
        }
        // The score falls through 0 at the maximum, between the neighbours.
        let k = bisect_log(
            at(best - 1),
            at(best + 1),
            |k| matches!(score(k), Some(s) if s > 0.0),
        );
        let alpha_tail = count / exposure(k).0;
        Gpd::riegel(t, k * alpha_tail, alpha_tail)
    }
}

/// The alphas of a piecewise Pareto with thresholds `t`, truncated as a
/// whole at `tr`, that maximize the conditional likelihood of `losses`:
/// coordinate ascent from `start`, each alpha by bisection on its partial
/// derivative.
///
/// With `E_k(a)` the log-exposure of `[t_0, a]` in piece `k`,
/// `ln S(a) = −Σ α_k E_k(a)`, and with `ΔE_k(a) = E_k(T) − E_k(a)` and
/// `D(a) = −Σ α_k ΔE_k(a)`, `ln(S(a) − S(T)) = ln S(a) + ln(1 − e^D(a))`,
/// whose derivative in `α_k` is `−E_k(a) + ΔE_k(a) / expm1(−D(a))`.
fn whole_truncated_alphas(
    t: &[f64],
    losses: &[(f64, f64, bool, f64)],
    tr: f64,
    start: &[f64],
) -> Result<Vec<f64>> {
    let n = t.len();
    let exposures = |a: f64| -> Vec<f64> {
        (0..n)
            .map(|k| {
                let hi = t.get(k + 1).copied().unwrap_or(f64::INFINITY);
                if a > t[k] {
                    (a.min(hi) / t[k]).ln()
                } else {
                    0.0
                }
            })
            .collect()
    };
    let e_tr = exposures(tr);
    struct Row {
        w: f64,
        censored: bool,
        piece: usize,
        e_y: Vec<f64>,
        d_y: Vec<f64>,
        e_r: Vec<f64>,
        d_r: Vec<f64>,
    }
    let mut rows = Vec::with_capacity(losses.len());
    for &(y, r, censored, w) in losses {
        if y >= tr {
            return Err(invalid("losses", y, "must lie below the truncation point"));
        }
        let (e_y, e_r) = (exposures(y), exposures(r));
        let diff = |e: &[f64]| e_tr.iter().zip(e).map(|(a, b)| a - b).collect::<Vec<_>>();
        rows.push(Row {
            w,
            censored,
            piece: t.partition_point(|&x| x <= y) - 1,
            d_y: diff(&e_y),
            d_r: diff(&e_r),
            e_y,
            e_r,
        });
    }
    if rows.iter().all(|r| r.censored) {
        return Err(invalid("losses", 0.0, "need at least one uncensored loss"));
    }
    let partial = |alphas: &[f64], k: usize| -> f64 {
        // ∂/∂α_k of ln(S(a) − S(T)).
        let term = |e: &[f64], d: &[f64]| -> f64 {
            let minus_d: f64 = alphas.iter().zip(d).map(|(a, x)| a * x).sum();
            -e[k] + d[k] / minus_d.exp_m1()
        };
        rows.iter()
            .map(|r| {
                let own = if r.censored {
                    term(&r.e_y, &r.d_y)
                } else {
                    let count = if r.piece == k { 1.0 / alphas[k] } else { 0.0 };
                    count - r.e_y[k]
                };
                r.w * (own - term(&r.e_r, &r.d_r))
            })
            .sum()
    };
    let mut alphas: Vec<f64> = start
        .iter()
        .map(|&a| a.clamp(ALPHA_MIN, ALPHA_MAX))
        .collect();
    for _ in 0..5000 {
        let mut change = 0.0f64;
        for k in 0..n {
            let old = alphas[k];
            let at = |a: f64, alphas: &mut Vec<f64>| {
                alphas[k] = a;
                partial(alphas, k)
            };
            let new = coordinate_root(old, |a| at(a, &mut alphas));
            alphas[k] = new;
            change = change.max((new / old).ln().abs());
        }
        if change < 1e-13 {
            break;
        }
    }
    Ok(alphas)
}

/// The root in `[ALPHA_MIN, ALPHA_MAX]` of a partial derivative `f` that
/// falls through 0 at the coordinate's maximum, or the bound it is clamped
/// to. Brackets around `start` first, then the Illinois method on
/// `ln α`, which converges in a handful of evaluations.
fn coordinate_root(start: f64, mut f: impl FnMut(f64) -> f64) -> f64 {
    let (mut lo, mut hi) = ((start / 1.5).max(ALPHA_MIN), (start * 1.5).min(ALPHA_MAX));
    let (mut f_lo, mut f_hi) = (f(lo), f(hi));
    while f_lo <= 0.0 && lo > ALPHA_MIN {
        hi = lo;
        f_hi = f_lo;
        lo = (lo / 4.0).max(ALPHA_MIN);
        f_lo = f(lo);
    }
    if f_lo <= 0.0 {
        return ALPHA_MIN;
    }
    while f_hi > 0.0 && hi < ALPHA_MAX {
        lo = hi;
        f_lo = f_hi;
        hi = (hi * 4.0).min(ALPHA_MAX);
        f_hi = f(hi);
    }
    if f_hi > 0.0 {
        return ALPHA_MAX;
    }
    // Illinois on x = ln α: f(lo) > 0 ≥ f(hi).
    illinois(lo.ln(), hi.ln(), f_lo, f_hi, |x| f(x.exp())).exp()
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
    Ok(bisect(at(j - 1), at(j), |a| score(a) > 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Distribution, Severity};
    use prospicio_core::StreamRng;

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
        // Draws reach 40,000 exactly only below it, so a whole-distribution
        // fit at the same point works too.
        assert!(
            PiecewisePareto::fit(
                vec![1000.0, 3000.0],
                &data,
                Some((40_000.0, Truncation::WholeDistribution)),
            )
            .is_ok()
        );
    }

    #[test]
    fn generalized_pareto_fit_maximizes_the_likelihood() {
        let data = LargeLosses::new(vec![
            1100.0, 1300.0, 1750.0, 2000.0, 2600.0, 3500.0, 4100.0, 9000.0, 25_000.0, 40_000.0,
        ])
        .unwrap()
        .reporting_thresholds(vec![
            0.0, 1200.0, 0.0, 1500.0, 0.0, 0.0, 3000.0, 5000.0, 0.0, 0.0,
        ])
        .unwrap()
        .censored(vec![
            false, false, false, false, false, true, false, false, false, true,
        ])
        .unwrap();
        let t = 1000.0;
        let ll = |alpha_ini: f64, alpha_tail: f64| -> f64 {
            let g = Gpd::riegel(t, alpha_ini, alpha_tail).unwrap();
            let k = alpha_ini / alpha_tail;
            (0..data.len())
                .map(|i| {
                    let (y, r) = (data.values[i], data.reporting[i].max(t));
                    let own = if data.censored[i] {
                        g.survival(y).ln()
                    } else {
                        (alpha_ini / t).ln() - (alpha_tail + 1.0) * (k * (y / t - 1.0)).ln_1p()
                    };
                    own - g.survival(r).ln()
                })
                .sum()
        };
        let fit = Gpd::fit_riegel(t, &data).unwrap();
        let (ai, at) = (t / fit.beta(), 1.0 / fit.xi());
        let best = ll(ai, at);
        for (da, dt) in [
            (1.001, 1.0),
            (0.999, 1.0),
            (1.0, 1.001),
            (1.0, 0.999),
            (1.001, 1.001),
        ] {
            assert!(ll(ai * da, at * dt) < best, "{da} {dt}");
        }
        // Equal alphas reduce to the Pareto fit's model family: data drawn
        // from a Pareto give k near 1.
        let p = Pareto::new(t, 2.0).unwrap();
        let draws = p.sample(&mut StreamRng::new(9, 0), 100_000);
        let g = Gpd::fit_riegel(t, &LargeLosses::new(draws).unwrap()).unwrap();
        assert!(((t / g.beta()) * g.xi() - 1.0).abs() < 0.05);
    }

    #[test]
    fn whole_truncated_piecewise_fit_maximizes_the_likelihood() {
        let t = vec![1000.0, 2500.0, 8000.0];
        let tr = 60_000.0;
        let data = LargeLosses::new(vec![
            1100.0, 1300.0, 1750.0, 2000.0, 2600.0, 3500.0, 4100.0, 5200.0, 7000.0, 9000.0,
            12_000.0, 18_000.0, 25_000.0, 40_000.0,
        ])
        .unwrap()
        .reporting_thresholds(vec![
            0.0, 1200.0, 0.0, 1500.0, 0.0, 0.0, 3000.0, 0.0, 0.0, 5000.0, 0.0, 0.0, 0.0, 0.0,
        ])
        .unwrap()
        .censored(vec![
            false, false, false, false, false, true, false, false, false, false, true, false,
            false, true,
        ])
        .unwrap();
        let ll = |alphas: &[f64]| -> f64 {
            let open = PiecewisePareto::new(t.clone(), alphas.to_vec()).unwrap();
            let pp = open
                .clone()
                .truncated(tr, Truncation::WholeDistribution)
                .unwrap();
            let mass = 1.0 - open.survival(tr);
            (0..data.len())
                .map(|i| {
                    let (y, r) = (data.values[i], data.reporting[i].max(t[0]));
                    let own = if data.censored[i] {
                        pp.survival(y).ln()
                    } else {
                        // Density α_k S(y) / (y (1 − S(T))), S untruncated.
                        let k = t.partition_point(|&x| x <= y) - 1;
                        (alphas[k] * open.survival(y) / (y * mass)).ln()
                    };
                    own - pp.survival(r).ln()
                })
                .sum()
        };
        let fit = PiecewisePareto::fit(t.clone(), &data, Some((tr, Truncation::WholeDistribution)))
            .unwrap();
        let a = fit.alphas().to_vec();
        let best = ll(&a);
        for k in 0..3 {
            for f in [1.001, 0.999] {
                let mut b = a.clone();
                b[k] *= f;
                assert!(ll(&b) < best, "{k} {f}");
            }
        }
        // Recovery from simulated data.
        let truth = PiecewisePareto::new(t.clone(), vec![1.2, 0.8, 1.5])
            .unwrap()
            .truncated(tr, Truncation::WholeDistribution)
            .unwrap();
        let draws = truth.sample(&mut StreamRng::new(13, 0), 100_000);
        let fit = PiecewisePareto::fit(
            t,
            &LargeLosses::new(draws).unwrap(),
            Some((tr, Truncation::WholeDistribution)),
        )
        .unwrap();
        for (got, want) in fit.alphas().iter().zip(truth.alphas()) {
            assert!((got / want - 1.0).abs() < 0.05, "{got} {want}");
        }
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
