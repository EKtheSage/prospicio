//! Diagnostics for Markov chain Monte Carlo draws, whatever sampler made
//! them (`docs/design/models.md`: samplers are delegated; diagnostics are
//! ours).
//!
//! The estimators are those of Vehtari, Gelman, Simpson, Carpenter and
//! Bürkner (2021), "Rank-normalization, folding, and localization: an
//! improved R̂ for assessing convergence of MCMC", as the R package
//! `posterior` implements them:
//!
//! - [`rhat`]: the larger of the rank-normalized split R̂ of the draws and
//!   of their distance from the median (which catches differences in
//!   scale between chains);
//! - [`ess_bulk`] and [`ess_tail`]: effective sample sizes of the
//!   rank-normalized draws and of the 5% and 95% quantile indicators;
//! - [`ess_mean`] and [`mcse_mean`]: for the mean itself.
//!
//! Chains are given as equal-length slices of draws. Each is split in
//! half (dropping the middle draw of an odd-length chain), so a chain
//! that drifts shows up as two chains that disagree.
//!
//! ```
//! use act_bayes::{ess_bulk, rhat};
//!
//! // Two well-mixed chains of an alternating sequence.
//! let a: Vec<f64> = (0..400).map(|i| ((i * 37) % 101) as f64).collect();
//! let b: Vec<f64> = (0..400).map(|i| ((i * 53 + 7) % 101) as f64).collect();
//! let chains = [a.as_slice(), b.as_slice()];
//! assert!(rhat(&chains).unwrap() < 1.01);
//! assert!(ess_bulk(&chains).unwrap() > 100.0);
//! ```

pub mod elpd;
pub mod glm;

use act_core::{Error, Result};
use act_math::special::norm_quantile;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

/// Rank-normalized split R̂: the larger of the R̂ of the rank-normalized
/// split chains and of their folded draws `|x - median|`. Values above
/// 1.01 suggest the chains have not mixed.
pub fn rhat(chains: &[&[f64]]) -> Result<f64> {
    let split = split_chains(chains)?;
    let bulk = rhat_basic(&z_scale(&split));
    // Folded about the median of all the draws, before splitting.
    let median = quantile7(&all_draws(chains), 0.5);
    let folded: Vec<Vec<f64>> = split
        .iter()
        .map(|c| c.iter().map(|x| (x - median).abs()).collect())
        .collect();
    let tail = rhat_basic(&z_scale(&folded));
    Ok(bulk.max(tail))
}

/// Bulk effective sample size: of the rank-normalized split chains.
pub fn ess_bulk(chains: &[&[f64]]) -> Result<f64> {
    let split = split_chains(chains)?;
    Ok(ess_basic(&z_scale(&split)))
}

/// Tail effective sample size: the smaller of the effective sample sizes
/// of the indicators `x ≤ q₀.₀₅` and `x ≤ q₀.₉₅` on split chains.
pub fn ess_tail(chains: &[&[f64]]) -> Result<f64> {
    Ok(ess_quantile(chains, 0.05)?.min(ess_quantile(chains, 0.95)?))
}

/// Effective sample size for the quantile `prob`: of the indicator
/// `x ≤ q_prob` on split chains, with `q_prob` R's type 7 quantile of all
/// the draws.
pub fn ess_quantile(chains: &[&[f64]], prob: f64) -> Result<f64> {
    let split = split_chains(chains)?;
    let q = quantile7(&all_draws(chains), prob);
    let indicator: Vec<Vec<f64>> = split
        .iter()
        .map(|c| c.iter().map(|&x| f64::from(u8::from(x <= q))).collect())
        .collect();
    Ok(ess_basic(&indicator))
}

/// Effective sample size for the mean: of the split chains as they are.
pub fn ess_mean(chains: &[&[f64]]) -> Result<f64> {
    let split = split_chains(chains)?;
    Ok(ess_basic(&split))
}

/// Monte Carlo standard error of the mean: the standard deviation of all
/// the draws over the square root of [`ess_mean`].
pub fn mcse_mean(chains: &[&[f64]]) -> Result<f64> {
    let split = split_chains(chains)?;
    let all = all_draws(chains);
    let n = all.len() as f64;
    let mean = all.iter().sum::<f64>() / n;
    let var = all.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Ok((var / ess_basic(&split)).sqrt())
}

/// Each chain cut into its first and second halves (the middle draw of an
/// odd-length chain dropped).
fn split_chains(chains: &[&[f64]]) -> Result<Vec<Vec<f64>>> {
    let n = chains.first().map_or(0, |c| c.len());
    if chains.is_empty() || n < 4 {
        return Err(Error::Data(
            "need at least one chain of at least 4 draws".into(),
        ));
    }
    if chains.iter().any(|c| c.len() != n) {
        return Err(Error::Data("chains must all have the same length".into()));
    }
    if chains.iter().flat_map(|c| c.iter()).any(|x| !x.is_finite()) {
        return Err(Error::Data("draws must all be finite".into()));
    }
    let half = n / 2;
    let mut out = Vec::with_capacity(2 * chains.len());
    for c in chains {
        out.push(c[..half].to_vec());
    }
    for c in chains {
        out.push(c[n - half..].to_vec());
    }
    Ok(out)
}

/// Every draw of every chain, before splitting.
fn all_draws(chains: &[&[f64]]) -> Vec<f64> {
    chains.iter().flat_map(|c| c.iter().copied()).collect()
}

fn pooled(chains: &[Vec<f64>]) -> Vec<f64> {
    chains.iter().flatten().copied().collect()
}

/// Ranks over all draws (ties averaged), mapped to normal scores by
/// `Φ⁻¹((r - 3/8) / (S + 1/4))`, kept in chain shape.
fn z_scale(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let all = pooled(chains);
    let s = all.len();
    let mut order: Vec<usize> = (0..s).collect();
    order.sort_by(|&a, &b| all[a].total_cmp(&all[b]));
    let mut rank = vec![0.0; s];
    let mut i = 0;
    while i < s {
        let mut j = i;
        while j + 1 < s && all[order[j + 1]] == all[order[i]] {
            j += 1;
        }
        // 1-based average rank of the tie group.
        let r = (i + j) as f64 / 2.0 + 1.0;
        for &k in &order[i..=j] {
            rank[k] = r;
        }
        i = j + 1;
    }
    let sf = s as f64;
    let z: Vec<f64> = rank
        .iter()
        .map(|r| norm_quantile((r - 0.375) / (sf + 0.25)))
        .collect();
    let len = chains[0].len();
    z.chunks(len).map(<[f64]>::to_vec).collect()
}

/// Split-free R̂ of chains already split.
fn rhat_basic(chains: &[Vec<f64>]) -> f64 {
    let n = chains[0].len() as f64;
    let means: Vec<f64> = chains.iter().map(|c| c.iter().sum::<f64>() / n).collect();
    let vars: Vec<f64> = chains
        .iter()
        .zip(&means)
        .map(|(c, m)| c.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0))
        .collect();
    let between = n * sample_variance(&means);
    let within = vars.iter().sum::<f64>() / vars.len() as f64;
    ((between / within + n - 1.0) / n).sqrt()
}

fn sample_variance(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    if x.len() < 2 {
        return 0.0;
    }
    let m = x.iter().sum::<f64>() / n;
    x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0)
}

/// Biased autocovariances `(1/N) Σ_i (x_i - x̄)(x_{i+k} - x̄)`, `k < N`,
/// by FFT with enough zero padding to make them exact.
fn autocovariance(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let len = (2 * n).next_power_of_two();
    let mut buf: Vec<Complex64> = (0..len)
        .map(|i| Complex64::new(if i < n { x[i] - mean } else { 0.0 }, 0.0))
        .collect();
    let mut planner = FftPlanner::<f64>::new();
    planner.plan_fft_forward(len).process(&mut buf);
    for z in &mut buf {
        *z = Complex64::new(z.norm_sqr(), 0.0);
    }
    planner.plan_fft_inverse(len).process(&mut buf);
    buf[..n].iter().map(|z| z.re / (len * n) as f64).collect()
}

/// Effective sample size of chains already split, by Geyer's initial
/// monotone sequence on the combined autocorrelations (Stan's and
/// `posterior`'s estimator).
fn ess_basic(chains: &[Vec<f64>]) -> f64 {
    let m = chains.len();
    let n = chains[0].len();
    if n < 3 {
        return f64::NAN;
    }
    let nf = n as f64;
    let acov: Vec<Vec<f64>> = chains.iter().map(|c| autocovariance(c)).collect();
    let mean_acov = |t: usize| acov.iter().map(|a| a[t]).sum::<f64>() / m as f64;
    let means: Vec<f64> = chains.iter().map(|c| c.iter().sum::<f64>() / nf).collect();
    let mean_var = mean_acov(0) * nf / (nf - 1.0);
    let mut var_plus = mean_var * (nf - 1.0) / nf;
    if m > 1 {
        var_plus += sample_variance(&means);
    }
    let rho = |t: usize| 1.0 - (mean_var - mean_acov(t)) / var_plus;
    // Geyer's initial positive sequence.
    let mut rho_hat = vec![0.0; n];
    let mut t = 0;
    let mut even = 1.0;
    rho_hat[0] = even;
    let mut odd = rho(1);
    rho_hat[1] = odd;
    while t + 5 < n && (even + odd).is_finite() && even + odd > 0.0 {
        t += 2;
        even = rho(t);
        odd = rho(t + 1);
        if even + odd >= 0.0 {
            rho_hat[t] = even;
            rho_hat[t + 1] = odd;
        }
    }
    let max_t = t;
    if even > 0.0 {
        rho_hat[max_t] = even;
    }
    // Geyer's initial monotone sequence.
    let mut t = 0;
    while t + 4 <= max_t {
        t += 2;
        if rho_hat[t] + rho_hat[t + 1] > rho_hat[t - 2] + rho_hat[t - 1] {
            rho_hat[t] = (rho_hat[t - 2] + rho_hat[t - 1]) / 2.0;
            rho_hat[t + 1] = rho_hat[t];
        }
    }
    let total = (m * n) as f64;
    // R's `sum(rho_hat_t[1:max_t])`; at max_t = 0, `1:0` keeps element 1.
    let head: f64 = if max_t == 0 {
        rho_hat[0]
    } else {
        rho_hat[..max_t].iter().sum()
    };
    let tau = -1.0 + 2.0 * head + rho_hat[max_t];
    let tau = tau.max(1.0 / total.log10());
    total / tau
}

/// R's default (type 7) quantile.
fn quantile7(x: &[f64], p: f64) -> f64 {
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let h = (s.len() - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    s[lo] + (h - lo as f64) * (s[hi] - s[lo])
}

#[cfg(test)]
mod tests {
    use super::*;
    use act_core::StreamRng;

    fn ar1(seed: u64, n: usize, phi: f64, shift: f64) -> Vec<f64> {
        let mut rng = StreamRng::new(seed, 0);
        let mut x = 0.0;
        (0..n)
            .map(|_| {
                x = phi * x + norm_quantile(rng.next_open01());
                x + shift
            })
            .collect()
    }

    #[test]
    fn autocovariance_matches_direct_sums() {
        let x = ar1(1, 37, 0.5, 0.0);
        let ac = autocovariance(&x);
        let m = x.iter().sum::<f64>() / 37.0;
        for k in [0, 1, 5, 36] {
            let direct: f64 = (0..37 - k)
                .map(|i| (x[i] - m) * (x[i + k] - m))
                .sum::<f64>()
                / 37.0;
            assert!((ac[k] - direct).abs() < 1e-12);
        }
    }

    #[test]
    fn independent_draws_have_ess_near_their_count() {
        let chains: Vec<Vec<f64>> = (0..4).map(|s| ar1(s, 1000, 0.0, 0.0)).collect();
        let refs: Vec<&[f64]> = chains.iter().map(Vec::as_slice).collect();
        let ess = ess_bulk(&refs).unwrap();
        assert!((ess / 4000.0 - 1.0).abs() < 0.15, "{ess}");
        assert!(rhat(&refs).unwrap() < 1.01);
    }

    #[test]
    fn autocorrelation_lowers_ess_and_a_stuck_chain_raises_rhat() {
        // AR(1) with φ = 0.9: ESS about n (1 - φ) / (1 + φ) ≈ n / 19.
        let chains: Vec<Vec<f64>> = (0..4).map(|s| ar1(s, 2000, 0.9, 0.0)).collect();
        let refs: Vec<&[f64]> = chains.iter().map(Vec::as_slice).collect();
        let ess = ess_mean(&refs).unwrap();
        assert!(
            ess > 8000.0 / 19.0 * 0.6 && ess < 8000.0 / 19.0 * 1.6,
            "{ess}"
        );
        let mut shifted = chains.clone();
        shifted[3] = ar1(3, 2000, 0.9, 5.0);
        let refs: Vec<&[f64]> = shifted.iter().map(Vec::as_slice).collect();
        assert!(rhat(&refs).unwrap() > 1.1);
    }
}
