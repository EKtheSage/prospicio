//! The draws behind the exact transitions: normals, the noncentral
//! chi-square of CIR, and the Poisson count that mixes it.

use prospicio_core::StreamRng;
use prospicio_math::special::{gamma_inc, ln_gamma, norm_quantile};
use prospicio_prob::{Distribution, Gamma};

/// A standard normal draw by inverse transform.
pub(crate) fn normal(rng: &mut StreamRng) -> f64 {
    norm_quantile(rng.next_open01())
}

/// A chi-square draw with `dof` degrees of freedom, `dof > 0`.
fn chi_square(rng: &mut StreamRng, dof: f64) -> f64 {
    Gamma::new(dof / 2.0, 2.0)
        .expect("a positive shape and scale")
        .sample_one(rng)
}

/// A noncentral chi-square draw, `dof > 0`, `noncentrality >= 0`: for
/// `dof > 1`, `(Z + √λ)² + χ²(dof - 1)`; otherwise a Poisson mixture,
/// `χ²(dof + 2N)` with `N ~ Poisson(λ / 2)` (Johnson, Kotz and
/// Balakrishnan, chapter 29). Both are exact.
pub(crate) fn noncentral_chi_square(rng: &mut StreamRng, dof: f64, noncentrality: f64) -> f64 {
    if dof > 1.0 {
        let z = normal(rng) + noncentrality.sqrt();
        z * z + chi_square(rng, dof - 1.0)
    } else {
        let n = poisson(rng, noncentrality / 2.0);
        let dof = dof + 2.0 * n as f64;
        chi_square(rng, dof)
    }
}

/// A Poisson draw with mean `mean >= 0` by inversion, searching from the
/// mode so that a large mean neither underflows nor walks far.
fn poisson(rng: &mut StreamRng, mean: f64) -> u64 {
    if mean <= 0.0 {
        return 0;
    }
    let u = rng.next_open01();
    let ln_pmf = |k: u64| k as f64 * mean.ln() - mean - ln_gamma(k as f64 + 1.0);
    let mode = mean.floor() as u64;
    // P(N <= mode) is the regularized upper incomplete gamma Q(mode + 1, mean).
    let mut cdf = gamma_inc(mode as f64 + 1.0, mean).1;
    if u <= cdf {
        // Walk down: find the smallest k with P(N <= k) >= u.
        let mut k = mode;
        while k > 0 {
            let below = cdf - ln_pmf(k).exp();
            if below < u {
                return k;
            }
            cdf = below;
            k -= 1;
        }
        0
    } else {
        let mut k = mode;
        loop {
            k += 1;
            let p = ln_pmf(k).exp();
            cdf += p;
            if cdf >= u || p < 1e-300 {
                return k;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moments(draws: &[f64]) -> (f64, f64) {
        let n = draws.len() as f64;
        let mean = draws.iter().sum::<f64>() / n;
        let var = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, var)
    }

    #[test]
    fn poisson_draws_have_their_moments_small_and_large() {
        for mean in [0.3, 4.0, 900.0] {
            let mut rng = StreamRng::new(5, 0);
            let draws: Vec<f64> = (0..40_000)
                .map(|_| poisson(&mut rng, mean) as f64)
                .collect();
            let (m, v) = moments(&draws);
            let se = (mean / 40_000.0).sqrt();
            assert!((m - mean).abs() < 4.0 * se, "{mean}: {m}");
            assert!((v / mean - 1.0).abs() < 0.05, "{mean}: {v}");
        }
    }

    #[test]
    fn noncentral_chi_square_draws_have_their_moments_both_ways() {
        // Mean dof + λ, variance 2 (dof + 2λ).
        for (dof, lambda) in [(0.6, 2.5), (3.0, 7.0)] {
            let mut rng = StreamRng::new(9, 1);
            let draws: Vec<f64> = (0..60_000)
                .map(|_| noncentral_chi_square(&mut rng, dof, lambda))
                .collect();
            let (m, v) = moments(&draws);
            let var = 2.0 * (dof + 2.0 * lambda);
            assert!(
                (m - dof - lambda).abs() < 4.0 * (var / 60_000.0).sqrt(),
                "{m}"
            );
            assert!((v / var - 1.0).abs() < 0.05, "{v} vs {var}");
        }
    }
}
