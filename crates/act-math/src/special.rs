//! Special functions.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// Standard normal density.
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Standard normal distribution function, accurate in both tails.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x * FRAC_1_SQRT_2)
}

/// Standard normal quantile (inverse of [`norm_cdf`]).
///
/// Returns `-inf` at 0, `+inf` at 1 and NaN outside `[0, 1]`. Starts from
/// Abramowitz & Stegun 26.2.23 (error below 4.5e-4) and polishes with Newton
/// steps on `ln Phi`, which keeps full relative accuracy deep in the tail.
///
/// ```
/// use act_math::special::norm_quantile;
///
/// assert!((norm_quantile(0.975) - 1.959963984540054).abs() < 1e-14);
/// ```
pub fn norm_quantile(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p > 0.5 {
        return -lower_quantile(1.0 - p);
    }
    lower_quantile(p)
}

/// Quantile for `0 < p <= 0.5`.
fn lower_quantile(p: f64) -> f64 {
    let t = (-2.0 * p.ln()).sqrt();
    let mut x = -(t
        - (2.515517 + 0.802853 * t + 0.010328 * t * t)
            / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t));
    let target = p.ln();
    for _ in 0..8 {
        let cdf = norm_cdf(x);
        let step = (cdf.ln() - target) * cdf / norm_pdf(x);
        x -= step;
        if step.abs() <= 1e-15 * x.abs().max(1.0) {
            break;
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_inverts_cdf() {
        for &p in &[
            1e-300,
            1e-100,
            1e-20,
            1e-8,
            0.001,
            0.1,
            0.3,
            0.5,
            0.7,
            0.99,
            1.0 - 1e-12,
        ] {
            let x = norm_quantile(p);
            let back = norm_cdf(x);
            assert!(
                ((back - p) / p).abs() < 1e-13,
                "p = {p}, x = {x}, back = {back}"
            );
        }
    }

    #[test]
    fn quantile_known_values() {
        // SciPy 1.x: scipy.stats.norm.ppf.
        let cases = [
            (0.5, 0.0),
            (0.995, 2.5758293035489004),
            (0.001, -3.090232306167813),
            (1e-10, -6.361340902404056),
        ];
        for (p, want) in cases {
            let got = norm_quantile(p);
            assert!(
                (got - want).abs() < 1e-13,
                "p = {p}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn quantile_edges() {
        assert_eq!(norm_quantile(0.0), f64::NEG_INFINITY);
        assert_eq!(norm_quantile(1.0), f64::INFINITY);
        assert!(norm_quantile(1.5).is_nan());
        assert!(norm_quantile(f64::NAN).is_nan());
    }
}
