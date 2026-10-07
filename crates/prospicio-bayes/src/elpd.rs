//! Expected log pointwise predictive density (ELPD): how well a Bayesian
//! model predicts each observation, estimated from posterior draws of the
//! pointwise log-likelihood `log p(yᵢ | θₛ)`.
//!
//! - [`lppd`]: the in-sample log pointwise predictive density,
//!   `Σᵢ log((1/S) Σₛ p(yᵢ | θₛ))`; it overstates out-of-sample fit.
//! - [`waic`]: Watanabe's WAIC, `lppd` less the effective number of
//!   parameters `Σᵢ Varₛ log p(yᵢ | θₛ)`.
//! - [`loo`]: leave-one-out cross-validation by Pareto-smoothed importance
//!   sampling (PSIS-LOO): one fit, with each observation's importance
//!   ratios `1 / p(yᵢ | θₛ)` stabilized by fitting a generalized Pareto to
//!   their largest values. The fitted shape `k̂` per observation is the
//!   reliability diagnostic.
//!
//! References: Vehtari, Gelman and Gabry (2017), *Practical Bayesian model
//! evaluation using leave-one-out cross-validation and WAIC*; Vehtari,
//! Simpson, Gelman, Yao and Gabry (2024), *Pareto smoothed importance
//! sampling*; Zhang and Stephens (2009), *A new and efficient estimation
//! method for the generalized Pareto distribution*. The parity suite checks
//! the results against the R package `loo`.
//!
//! The log-likelihood is given as `S` draws by `N` observations, row-major
//! (`log_lik[s * n + i]`), the layout of a draws-by-observations matrix.

use prospicio_core::{Error, Result};

/// An ELPD estimate: the total, its standard error, and the pointwise
/// values whose spread gives that error.
#[derive(Debug, Clone, PartialEq)]
pub struct Elpd {
    /// `Σᵢ elpdᵢ`.
    pub elpd: f64,
    /// `√(N Var(elpdᵢ))`.
    pub se: f64,
    /// Effective number of parameters, `lppd - elpd`.
    pub p: f64,
    /// The information criterion, `-2 elpd`.
    pub ic: f64,
    /// `elpdᵢ` per observation.
    pub pointwise: Vec<f64>,
}

/// A PSIS-LOO estimate with its per-observation diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct Loo {
    pub estimate: Elpd,
    /// The fitted generalized Pareto shape `k̂` of each observation's
    /// importance ratios: below [`Loo::k_threshold`] the estimate is
    /// reliable; above 0.7 it is not, and that observation needs an exact
    /// refit or moment matching.
    pub pareto_k: Vec<f64>,
    /// `min(1 - 1/log₁₀ S, 0.7)`, the reliability threshold for `S` draws.
    pub k_threshold: f64,
}

impl Loo {
    /// Observations whose `k̂` is above the threshold.
    pub fn bad_observations(&self) -> Vec<usize> {
        (0..self.pareto_k.len())
            .filter(|&i| self.pareto_k[i] > self.k_threshold)
            .collect()
    }
}

/// Checks the matrix shape and returns `(S, N)`.
fn shape(log_lik: &[f64], n: usize) -> Result<usize> {
    if n == 0 || log_lik.is_empty() || log_lik.len() % n != 0 {
        return Err(Error::Data(format!(
            "{} log-likelihood values do not form draws × {n} observations",
            log_lik.len()
        )));
    }
    let s = log_lik.len() / n;
    if s < 2 {
        return Err(Error::Data(
            "at least two posterior draws are needed".into(),
        ));
    }
    if let Some(bad) = log_lik.iter().find(|v| v.is_nan()) {
        return Err(Error::InvalidParameter {
            name: "log_lik",
            value: *bad,
            reason: "must not be NaN",
        });
    }
    Ok(s)
}

/// Observation `i`'s draws.
fn column(log_lik: &[f64], n: usize, i: usize) -> Vec<f64> {
    log_lik.iter().skip(i).step_by(n).copied().collect()
}

/// `log Σ exp(xᵢ)`, without overflow.
fn log_sum_exp(x: &[f64]) -> f64 {
    let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return m;
    }
    m + x.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
}

fn mean_var(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let v = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0);
    (m, v)
}

fn estimate(pointwise: Vec<f64>, lppd_i: &[f64]) -> Elpd {
    let n = pointwise.len() as f64;
    let elpd: f64 = pointwise.iter().sum();
    let se = if pointwise.len() > 1 {
        (n * mean_var(&pointwise).1).sqrt()
    } else {
        f64::NAN
    };
    let p = lppd_i.iter().sum::<f64>() - elpd;
    Elpd {
        elpd,
        se,
        p,
        ic: -2.0 * elpd,
        pointwise,
    }
}

/// `log((1/S) Σₛ p(yᵢ | θₛ))` per observation.
fn lppd_pointwise(log_lik: &[f64], n: usize, s: usize) -> Vec<f64> {
    (0..n)
        .map(|i| log_sum_exp(&column(log_lik, n, i)) - (s as f64).ln())
        .collect()
}

/// In-sample log pointwise predictive density,
/// `Σᵢ log((1/S) Σₛ p(yᵢ | θₛ))`, for `log_lik` with `n` observations.
///
/// ```
/// use prospicio_bayes::elpd::lppd;
///
/// // Two draws giving p(y) = 0.5 and 0.25 for one observation.
/// let v = lppd(&[0.5f64.ln(), 0.25f64.ln()], 1).unwrap();
/// assert!((v - 0.375f64.ln()).abs() < 1e-15);
/// ```
pub fn lppd(log_lik: &[f64], n: usize) -> Result<f64> {
    let s = shape(log_lik, n)?;
    Ok(lppd_pointwise(log_lik, n, s).iter().sum())
}

/// WAIC: `elpdᵢ = lppdᵢ - Varₛ log p(yᵢ | θₛ)` (sample variance), with
/// `p` the effective number of parameters and `ic = -2 elpd`.
pub fn waic(log_lik: &[f64], n: usize) -> Result<Elpd> {
    let s = shape(log_lik, n)?;
    let lppd_i = lppd_pointwise(log_lik, n, s);
    let pointwise = (0..n)
        .map(|i| lppd_i[i] - mean_var(&column(log_lik, n, i)).1)
        .collect();
    Ok(estimate(pointwise, &lppd_i))
}

/// PSIS-LOO: for each observation, the importance ratios
/// `rₛ = 1 / p(yᵢ | θₛ)` have their largest `M = min(⌈0.2 S⌉, ⌈3 √(S / r_eff)⌉)`
/// values replaced by the expected order statistics of a generalized
/// Pareto fitted to them (Zhang and Stephens' estimator, with the
/// weakly informative prior of Vehtari et al. shrinking `k̂` towards 0.5),
/// truncated at the largest raw ratio; then
/// `elpd_looᵢ = log Σₛ wₛ p(yᵢ | θₛ) / Σₛ wₛ`.
///
/// `r_eff` is each observation's relative efficiency of the draws (their
/// effective sample size over `S`); `None` means independent draws (1).
///
/// ```
/// use prospicio_bayes::elpd::loo;
///
/// // Identical draws: every ratio is equal, so LOO is the plain average.
/// let ll = vec![-1.0; 200 * 3];
/// let l = loo(&ll, 3, None).unwrap();
/// assert!((l.estimate.elpd + 3.0).abs() < 1e-12);
/// ```
pub fn loo(log_lik: &[f64], n: usize, r_eff: Option<&[f64]>) -> Result<Loo> {
    let s = shape(log_lik, n)?;
    if let Some(r) = r_eff {
        if r.len() != n || r.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
            return Err(Error::Data(format!(
                "r_eff needs {n} finite positive values"
            )));
        }
    }
    let lppd_i = lppd_pointwise(log_lik, n, s);
    let mut pointwise = Vec::with_capacity(n);
    let mut pareto_k = Vec::with_capacity(n);
    for i in 0..n {
        let ll = column(log_lik, n, i);
        let r = r_eff.map_or(1.0, |r| r[i]);
        let (log_w, k) = psis(&ll.iter().map(|v| -v).collect::<Vec<f64>>(), r);
        let lse_w = log_sum_exp(&log_w);
        let num: Vec<f64> = log_w.iter().zip(&ll).map(|(w, l)| w + l).collect();
        pointwise.push(log_sum_exp(&num) - lse_w);
        pareto_k.push(k);
    }
    Ok(Loo {
        estimate: estimate(pointwise, &lppd_i),
        pareto_k,
        k_threshold: (1.0 - 1.0 / (s as f64).log10()).min(0.7),
    })
}

/// Pareto-smoothed log importance weights for log ratios `log_r`, and the
/// fitted shape `k̂` (infinite when the tail is too short to fit).
fn psis(log_r: &[f64], r_eff: f64) -> (Vec<f64>, f64) {
    let s = log_r.len();
    // Shift so the largest log ratio is 0: ratios are in (0, 1].
    let max = log_r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut lw: Vec<f64> = log_r.iter().map(|v| v - max).collect();
    let m = ((0.2 * s as f64).ceil()).min((3.0 * (s as f64 / r_eff).sqrt()).ceil()) as usize;
    if m < 5 || m >= s {
        return (lw, f64::INFINITY);
    }
    let mut order: Vec<usize> = (0..s).collect();
    order.sort_by(|&a, &b| lw[a].total_cmp(&lw[b]));
    let tail = &order[s - m..];
    let cutoff = lw[order[s - m - 1]];
    let exceed: Vec<f64> = tail.iter().map(|&j| lw[j].exp() - cutoff.exp()).collect();
    if exceed.iter().all(|&x| x <= 0.0) {
        return (lw, f64::INFINITY);
    }
    let (k, sigma) = gpd_fit(&exceed);
    // Replace the tail, in ascending order, by the GPD quantiles at the
    // midpoints (z - 1/2)/M, truncated at the largest raw ratio (0 here).
    for (z, &j) in tail.iter().enumerate() {
        let p = (z as f64 + 0.5) / m as f64;
        let q = gpd_quantile(p, k, sigma);
        lw[j] = (cutoff.exp() + q).ln().min(0.0);
    }
    (lw, k)
}

/// GPD quantile `σ/k ((1 - p)^(-k) - 1)` (the exponential at `k = 0`).
fn gpd_quantile(p: f64, k: f64, sigma: f64) -> f64 {
    if k.abs() < 1e-12 {
        -sigma * (-p).ln_1p()
    } else {
        sigma / k * ((-k * (-p).ln_1p()).exp() - 1.0)
    }
}

/// Zhang and Stephens' (2009) estimate of the generalized Pareto shape `k`
/// and scale `σ` for exceedances `x > 0`, by a posterior mean over a grid
/// of `θ = -k/σ`, then shrunk towards `k = 0.5` as by a weakly informative
/// prior worth 10 observations (Vehtari et al., 2024).
fn gpd_fit(x: &[f64]) -> (f64, f64) {
    let mut x = x.to_vec();
    x.sort_by(f64::total_cmp);
    let n = x.len();
    let nf = n as f64;
    let m = 30 + nf.sqrt().floor() as usize;
    let x_max = x[n - 1];
    // First quartile, from the 1-based index ⌊n/4 + 1/2⌋.
    let x_q = x[((nf / 4.0 + 0.5).floor() as usize).max(1) - 1];
    let thetas: Vec<f64> = (1..=m)
        .map(|j| 1.0 / x_max + (1.0 - (m as f64 / (j as f64 - 0.5)).sqrt()) / (3.0 * x_q))
        .collect();
    let k_of = |theta: f64| x.iter().map(|v| (-theta * v).ln_1p()).sum::<f64>() / nf;
    // Profile log-likelihood of θ: n (log(-θ / k) - k - 1).
    let ll: Vec<f64> = thetas
        .iter()
        .map(|&t| {
            let k = k_of(t);
            nf * ((-t / k).ln() - k - 1.0)
        })
        .collect();
    // Posterior weights over the grid: wⱼ = 1 / Σᵢ exp(lᵢ - lⱼ).
    let weights: Vec<f64> = ll
        .iter()
        .map(|&lj| {
            let total: f64 = ll.iter().map(|&li| (li - lj).exp()).sum();
            if total.is_finite() { 1.0 / total } else { 0.0 }
        })
        .collect();
    let wsum: f64 = weights.iter().sum();
    let theta: f64 = thetas.iter().zip(&weights).map(|(t, w)| t * w).sum::<f64>() / wsum;
    let k = k_of(theta);
    let sigma = -k / theta;
    // Shrink k towards 0.5 as a prior worth 10 observations would.
    let k = (nf * k + 10.0 * 0.5) / (nf + 10.0);
    (k, sigma)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpd_fit_recovers_a_known_tail() {
        // Exact GPD quantiles at plotting positions, k = 0.3, σ = 2.
        let n = 2000;
        let x: Vec<f64> = (1..=n)
            .map(|i| gpd_quantile((f64::from(i) - 0.5) / f64::from(n), 0.3, 2.0))
            .collect();
        let (k, sigma) = gpd_fit(&x);
        // The prior pulls k a little towards 0.5 (10 in 2010).
        assert!((k - (0.3 * 2000.0 + 5.0) / 2010.0).abs() < 0.02, "{k}");
        assert!((sigma / 2.0 - 1.0).abs() < 0.05, "{sigma}");
    }

    #[test]
    fn waic_and_lppd_by_hand() {
        // Two observations, three draws.
        let ll = [-1.0, -2.0, -1.5, -2.5, -0.5, -3.0];
        let w = waic(&ll, 2).unwrap();
        let col0 = [-1.0f64, -1.5, -0.5];
        let lppd0 = (col0.iter().map(|v| v.exp()).sum::<f64>() / 3.0).ln();
        let var0 = mean_var(&col0).1;
        assert!((w.pointwise[0] - (lppd0 - var0)).abs() < 1e-15);
        assert!((w.p + w.elpd - lppd(&ll, 2).unwrap()).abs() < 1e-14);
        assert!(waic(&ll, 4).is_err());
    }
}
